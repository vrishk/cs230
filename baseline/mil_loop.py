import pytorch_lightning as pl
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.model_helpers import is_overridden

import numpy as np

from typing import *

def _convert_optim_dict(outs, num_optimizers):
        """Converts an optimizer dict to a list in which the key of the dict determines the position of the element.

        Example::
            >>> _convert_optim_dict({0: {"loss": 0.0}, 2: {"loss": 0.2}}, num_optimizers=3)
            [{'loss': 0.0}, None, {'loss': 0.2}]
        """
        return [outs[opt_idx] if opt_idx in outs else None for opt_idx in range(num_optimizers)]
    
def _recursive_unpad(nested: Union[Any, List[Any]], value: Optional[Any] = None) -> Union[Any, List[Any]]:
    """Removes the given pad value from the nested list. Not strictly the reverse operation of
    :func:`_recursive_pad` because it removes the padding element everywhere, not just from the end of a list.

    Example::
        >>> _recursive_unpad([[[0, 1, 0]], [2], [0, 0]], value=0)
        [[[1]], [2], []]
    """
    if not isinstance(nested, list):
        return nested

    return [_recursive_unpad(item, value) for item in nested if item != value]

class MILEpochLoop(pl.loops.TrainingEpochLoop):
    
    def _num_ready_batches_reached(self) -> bool:
        """Checks if we are in the last batch or if there are more batches to follow."""
        epoch_finished_on_ready = self.batch_progress.current.ready == self.trainer.num_training_batches
        return epoch_finished_on_ready or self.batch_progress.is_last_batch
    
    @staticmethod
    def _prepare_outputs_training_batch_end(
        batch_output,
        automatic: bool,
        num_optimizers: int,
    ):
        """Processes the outputs from the batch loop into the format passed to the ``training_batch_end`` hook.

        ``(tbptt_steps, n_opt) -> (n_opt, tbptt_steps)``. The optimizer dimension might have been squeezed.
        """
        if not batch_output:
            return []

        # convert optimizer dicts to list
        if automatic:
            batch_output = apply_to_collection(
                batch_output, dtype=dict, function=_convert_optim_dict, num_optimizers=num_optimizers
            )
        array = np.array(batch_output, dtype=object)
        if array.ndim == 1:
            array = np.expand_dims(array, 1)

        array = array.transpose((1, 0))
        array = array.squeeze()
        array = array.tolist()
        array = _recursive_unpad(array)
        return array
    
    def advance(self, dataloader_iter) -> None:
        """Runs a single training batch.
        Args:
            dataloader_iter: the iterator over the dataloader producing the new batch
        Raises:
            StopIteration: When the epoch is canceled by the user returning -1
        """
        if self.restarting and self._should_check_val_fx(self.batch_idx, self.batch_progress.is_last_batch):
            # skip training and run validation in `on_advance_end`
            return

        batch_idx, (batch, self.batch_progress.is_last_batch) = next(dataloader_iter)


        with self.trainer.profiler.profile("training_batch_to_device"):
            batch = self.trainer.accelerator.batch_to_device(batch)

        self.batch_progress.increment_ready()

        # cache the batch size value to avoid extracting it again after the batch loop runs as the value will be
        # different if tbptt is enabled
        # batch_size = self.trainer.logger_connector.on_batch_start(batch_idx, batch)

        if batch is None:
            self._warning_cache.warn("train_dataloader yielded None. If this was on purpose, ignore this warning...")
            batch_output = []
        else:
            # hook
            response = self.trainer.call_hook("on_batch_start")
            if response == -1:
                self.batch_progress.increment_processed()
                raise StopIteration

            # TODO: Update this in v1.7 (deprecation: #9816)
            model_fx = self.trainer.lightning_module.on_train_batch_start
            extra_kwargs = (
                {"dataloader_idx": 0}
            )

            # hook
            response = self.trainer.call_hook("on_train_batch_start", batch, batch_idx, **extra_kwargs)
            if response == -1:
                self.batch_progress.increment_processed()
                raise StopIteration

            self.batch_progress.increment_started()

            with self.trainer.profiler.profile("run_training_batch"):
                batch_output = self.batch_loop.run(batch, batch_idx, 0)

        # self.trainer._results.batch_size = batch_size

        self.batch_progress.increment_processed()

        # # update non-plateau LR schedulers
        # # update epoch-interval ones only when we are at the end of training epoch
        # self.update_lr_schedulers("step", update_plateau_schedulers=False)
        # if self._num_ready_batches_reached():
        #     self.update_lr_schedulers("epoch", update_plateau_schedulers=False)

        batch_end_outputs = self._prepare_outputs_training_batch_end(
            batch_output,
            automatic=self.trainer.lightning_module.trainer.lightning_module.automatic_optimization,
            num_optimizers=len(self.trainer.optimizers),
        )

        # TODO: Update this in v1.7 (deprecation: #9816)
        model_fx = self.trainer.lightning_module.on_train_batch_end
        extra_kwargs = (
            {"dataloader_idx": 0}
            if callable(model_fx) and is_param_in_hook_signature(model_fx, "dataloader_idx")
            else {}
        )
        self.trainer.call_hook("on_train_batch_end", batch_end_outputs, batch, batch_idx, **extra_kwargs)
        self.trainer.call_hook("on_batch_end")
        self.trainer.logger_connector.on_batch_end()

        self.batch_progress.increment_completed()

        # if is_overridden("training_epoch_end", self.trainer.lightning_module):
        #     self._outputs.append(batch_output)

        # -----------------------------------------
        # SAVE METRICS TO LOGGERS AND PROGRESS_BAR
        # -----------------------------------------
        self.trainer.logger_connector.update_train_step_metrics()
        
    def on_advance_end(self):
        """Runs validation and Checkpointing if necessary.

        Raises:
            StopIteration: if :attr:`done` evaluates to ``True`` to finish this epoch
        """
        # -----------------------------------------
        # VALIDATE IF NEEDED + CHECKPOINT CALLBACK
        # -----------------------------------------
        should_check_val = self._should_check_val_fx(self.batch_idx, self.batch_progress.is_last_batch)
        if should_check_val:
            self.trainer.validating = True
            self._run_validation()
            self.trainer.training = True

        # -----------------------------------------
        # SAVE LOGGERS (ie: Tensorboard, etc...)
        # -----------------------------------------
        self._save_loggers_on_train_batch_end()

        # update plateau LR scheduler after metrics are logged
        self.update_lr_schedulers("step", update_plateau_schedulers=True)

        if not self._should_accumulate():
            # progress global step according to grads progress
            self.global_step += 1

        # if training finished, try to exit in `on_run_end` instead as we should have enough time
        # TODO: @tchaton verify this assumption is True.
        if not self._is_training_done:
            # if fault tolerant is enabled and process has been notified, exit.
            self.trainer._exit_gracefully_on_signal()