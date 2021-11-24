import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.core.mixins import device_dtype_mixin
import torchvision.transforms as transforms

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Local imports

import os
import sys

sys.path.append(os.path.join(os.getcwd(), "./tripletnet"))
sys.path.append(os.path.join(os.getcwd(), "./resnet"))

from dataset import NaiveDataset
from linear import LinearNaive
from tripletnet.tripletnet_naive import TripletNetNaive

# Lightning seed

pl.seed_everything(42)

# Constants

RAW = lambda group: f"/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/data_splits/{group}.hdf5"
MODEL = lambda model, group: f"/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/data_splits/custom_splits/{model}_features/{model}_{group}_features.hdf5"
AUG =  lambda group: f"/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/data_splits/augmented/{group}.hdf5"

CORE_PROPORTIONS = [0.4719, 0.1770, 0.0148, 0.0771, 0.0948, 0.0277, 0.0807, 0.0508, 0.0051]


# Helper Functions

def get_weights():
    weights = 1. / torch.tensor(CORE_PROPORTIONS)
    weights = weights / sum(weights)
    return weights

def make_model(
        model_name: str, use_stored_features: bool,
        lr: float, num_classes: int,
        finetune: bool, layers_tune: int,
        optimizer: str):

    weights = get_weights()
    return {'tripletnet': LinearNaive(
        256 * 3, lr=lr, num_classes=num_classes,
        weights=weights, optimizer=optimizer) if use_stored_features
    else TripletNetNaive(
        finetune=finetune,
        lr=lr, num_classes=num_classes, weights=weights
    ),
        'tripletnet_e2e': TripletNetNaive(
            finetune=finetune, layers_tune=layers_tune,
            lr=lr, num_classes=num_classes,
            weights=weights, optimizer=optimizer
        )
    }[model_name]


def make_dataloaders(num_workers: int, batch_size: int, use_stored_features: bool = False, aug_data: bool = True, model: str = None):

    # If finetuning
    if use_stored_features:
        paths = {'train': MODEL(model, 'train'), 'val': MODEL(model, 'val'), 'test': MODEL(model, 'test')}
        datasets = {i: NaiveDataset(hdf5_path=paths[i], is_features=True) for i in paths}
    elif aug_data:
        paths = {'train': AUG('train'), 'val': AUG('val'), 'test': AUG('test')}
        datasets = {i: NaiveDataset(hdf5_path=paths[i], is_features=False) for i in paths}
    else:
        paths = {'train': RAW('train'), 'val': RAW('val'), 'test': RAW('test')}
        datasets = {i: NaiveDataset(hdf5_path=paths[i], is_features=False) for i in paths}

    dataloaders = {
        i: DataLoader(datasets[i], batch_size=batch_size, pin_memory=True,  num_workers=num_workers, shuffle=True)
        for i in datasets
    }

    return dataloaders


def train(cfg):
    model = make_model(cfg.model_type, cfg.stored_features,
                       cfg.lr, cfg.num_classes, cfg.finetune, cfg.layers_tune,
                       cfg.optimizer)

    dataloaders = make_dataloaders(
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        use_stored_features=cfg.stored_features,
        aug_data=cfg.aug_data,
        model=cfg.model_type
    )

    # training
    trainer = pl.Trainer(
        gpus=[0], num_nodes=1,
        precision=16, limit_train_batches=0.5,
        max_epochs=cfg.epochs, log_every_n_steps=1,
        accelerator="ddp"
    )
    trainer.fit(model, dataloaders['train'], dataloaders['val'])

    trainer.test(model, dataloaders['test'])

    trainer.save_checkpoint(f"../../models/{cfg.ckpt_name}.ckpt")


@hydra.main(config_path="configs", config_name='config')
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
