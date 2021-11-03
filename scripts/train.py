# Imports

from torch.utils.data import DataLoader 

import pytorch_lightning as pl

import argparse as ap

# Local imports

import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "../utils/"))

from dataset import HDF5Dataset
from tripletnet import TripletNet
from resnet.resnet import ResNet
from mil_loop import MILEpochLoop

from typing import Any

# Lightning seed

pl.seed_everything(42)


# Paths

PATH_TO_TRAIN = "/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/data_splits/train.hdf5"
PATH_TO_VAL = "/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/data_splits/val.hdf5"
PATH_TO_TEST = "/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/data_splits/test.hdf5"


def make_model(model_name: str):
    return {
        'triplenet': TripletNet(1e-3, 9, finetune=True),
        'triplenet_e2e': TripletNet(1e-3, 9, finetune=False),
        'resnet18': ResNet(18, 1e-3, 9, finetune=True),
        'resnet18_e2e': ResNet(18, 1e-3, 9, finetune=False),
        'resnet50': ResNet(50, 1e-3, 9, finetune=True),
        'resnet50_e2e': ResNet(50, 1e-3, 9, finetune=False)
    }[model_name]
 

def make_dataloaders(num_workers: int, batch_size: int):
    # Datasets

    paths = {'train': PATH_TO_TRAIN, 'val': PATH_TO_VAL, 'test': PATH_TO_TEST}
    datasets = {i: HDF5Dataset(paths[i]) for i in paths}
    dataloaders = {
        i: DataLoader(datasets[i], batch_size=batch_size, num_workers=num_workers) 
        for i in datasets
    }
    
    return dataloaders


        
# def copy_loop(mil, orig):
#     mil.min_steps = orig.min_steps
#     mil.max_steps = orig.max_steps

#     mil.global_step = orig.global_step
#     mil.batch_progress = orig.batch_progress
#     mil.scheduler_progress = orig.scheduler_progress

#     mil.batch_loop = orig.batch_loop
#     mil.val_loop = orig.val_loop

#     mil._results = orig._results
#     mil._warning_cache = orig._warning_cache

#     try: 
#         mil._outputs = orig._outputs
#         mil._dataloader_iter = orig._dataloader_iter
#         mil._dataloader_state_dict = orig._dataloader_state_dict
#     except:
#         mil._outputs = []
#         mil._dataloader_iter = None
#         mil._dataloader_state_dict = {}
        
#     return mil

def train(args):

    model = make_model(args.model_type)

    dataloaders = make_dataloaders(args.num_workers, args.batch_size)

    # training
    trainer = pl.Trainer(
        gpus=[0], num_nodes=1, num_processes=8,
        precision=16, limit_train_batches=0.5,
        max_epochs=args.epochs, log_every_n_steps=1,
        accelerator="ddp"
    )
    # print(dir(trainer.fit_loop.epoch_loop._dataloader_iter))
    # trainer.fit_loop.connect(copy_loop(MILEpochLoop(0, 0), trainer.fit_loop.epoch_loop))
    trainer.fit(model, dataloaders['train'], dataloaders['val'])
    
    trainer.test(model, dataloaders['test'])

    trainer.save_checkpoint("../../models/test.ckpt")

def main():
    parser = ap.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--model-type", default="triplenet", type=str)

    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()

