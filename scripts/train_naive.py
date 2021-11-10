import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.core.mixins import device_dtype_mixin

import torch
from torch.utils.data import DataLoader 
import pytorch_lightning as pl

import argparse as ap

# Local imports

import os
import sys

from dataset import NaiveDataset
from resnet.resnet_naive import ResNetNaive

# Lightning seed

pl.seed_everything(42)

# Constants

PATH_TO_TRAIN = "/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/data_splits/train.hdf5"
PATH_TO_VAL = "/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/data_splits/val.hdf5"
PATH_TO_TEST = "/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/data_splits/test.hdf5"

CORE_PROP = [0.4719, 0.1770, 0.0148, 0.0771, 0.0948, 0.0277, 0.0807, 0.0508, 0.0051]

# Helper Functions

def get_weights():
    weights = 1. / torch.tensor(CORE_PROP)
    weights = weights / sum(weights)
    return weights

def make_model(model_name: str):
    
    weights = get_weights()
    print("Weighted Crossentropy:", weights)
    
    return {
        # 'triplenet': TripletNet(1e-3, 9, finetune=True),
        # 'triplenet_e2e': TripletNet(1e-3, 9, finetune=False),
        'resnet18': ResNetNaive(size=18, lr=1e-3, num_classes=9, finetune=True, weights=weights),
        'resnet18_e2e': ResNetNaive(size=18, lr=1e-3, num_classes=9, finetune=False, weights=weights),
        'resnet50': ResNetNaive(size=50, lr=1e-3, num_classes=9, finetune=True, weights=weights),
        'resnet50_e2e': ResNetNaive(size=18, lr=1e-3, num_classes=9, finetune=False, weights=weights),
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

def train(cfg):

    model = make_model(cfg.model_type)

    dataloaders = make_dataloaders(cfg.num_workers, cfg.batch_size)

    # training
    trainer = pl.Trainer(
        gpus=[0], num_nodes=1, num_processes=8,
        precision=16, limit_train_batches=0.5,
        max_epochs=cfg.epochs, log_every_n_steps=1,
        accelerator="ddp"
    )
    trainer.fit(model, dataloaders['train'], dataloaders['val'])
    
    trainer.test(model, dataloaders['test'])

    trainer.save_checkpoint("../../models/test.ckpt")

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)

if __name__ == "__main__":
    main()

