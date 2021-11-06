# Imports

from torch.utils.data import DataLoader 

import pytorch_lightning as pl

import argparse as ap

# Local imports

import os
import sys

from dataset import NaiveDataset
from resnet.resnet_naive import ResNetNaive

from typing import Any

# Lightning seed

pl.seed_everything(42)


# Paths

PATH_TO_TRAIN = "/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/data_splits/train.hdf5"
PATH_TO_VAL = "/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/data_splits/val.hdf5"
PATH_TO_TEST = "/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/data_splits/test.hdf5"


def make_model(model_name: str):
    return {
        # 'triplenet': TripletNet(1e-3, 9, finetune=True),
        # 'triplenet_e2e': TripletNet(1e-3, 9, finetune=False),
        'resnet18': ResNetNaive(size=18, lr=1e-3, num_classes=9, finetune=True),
        'resnet18_e2e': ResNetNaive(size=18, lr=1e-3, num_classes=9, finetune=False),
        'resnet50': ResNetNaive(size=50, lr=1e-3, num_classes=9, finetune=True),
        'resnet50_e2e': ResNetNaive(size=18, lr=1e-3, num_classes=9, finetune=False),
    }[model_name]
 

def make_dataloaders(num_workers: int, batch_size: int):
    # Datasets

    paths = {'train': PATH_TO_TRAIN, 'val': PATH_TO_VAL, 'test': PATH_TO_TEST}
    datasets = {i: NaiveDataset(paths[i]) for i in paths}
    dataloaders = {
        i: DataLoader(datasets[i], batch_size=batch_size, num_workers=num_workers, shuffle=True) 
        for i in datasets
    }
    
    return dataloaders

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
    trainer.fit(model, dataloaders['train'], dataloaders['val'])
    
    trainer.test(model, dataloaders['test'])

    trainer.save_checkpoint("../../models/test.ckpt")

def main():
    parser = ap.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--model-type", default="resnet18", type=str)

    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()

