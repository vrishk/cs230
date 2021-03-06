from itertools import cycle
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
import numpy as np

import random
import functools

# Classes


class MILDataset(Dataset):
    def __init__(
        self, hdf5_path: str, bag_size=64, is_features: bool = False,
        transform=transforms.ToTensor()
    ):

        self.hdf5_path = hdf5_path
        self.h5data = h5py.File(self.hdf5_path, "r")
        self.cores = list(self.h5data.keys())

        self.is_features = is_features
        self.bag_size = bag_size
        self.transform = transform

    def __len__(self):
        return len(self.cores)

    def __getitem__(self, idx):

        patient_id = self.cores[idx]
        patches: np.ndarray = self.h5data[patient_id][()]

        if len(patches) < self.bag_size:
           # patches = cycle([im for im in patches])
           # patches = [next(patches) for i in range(self.bag_size)]
           patches = [im for im in patches]
        else:
            patches = random.sample([im for im in patches], self.bag_size)

        if self.is_features:
            label = self.h5data[patient_id].attrs["y"]
        else:
            label = self.h5data[patient_id].attrs["label"]
            if self.transform:
                patches = [self.transform(im) for im in patches]

       # patches = torch.tensor(patches)
       # labels = torch.nn.functional.one_hot(torch.arange(0, 8))[int(label),:]
        labels = torch.tensor(int(label))
        return patches, labels


# Temporary class to use for cutting down number of images with normalization
class NaiveDataset(Dataset):
    def __init__(self, hdf5_path: str, is_features: bool = False, transform=transforms.ToTensor()):

        self.hdf5_path = hdf5_path
        # Workaround for HDF5 not pickleable: https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        self.h5data = None

        h5data = h5py.File(self.hdf5_path, "r")
        self.cores = list(h5data.keys())

        self.lengths = [len(h5data[i]) for i in self.cores]
        self.is_features = is_features

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.00247837, 0.00186819, 0.00263161),
                                 (0.0101850529, 0.0099803880, 0.00758414449))
           ])

        self.transform = transform

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):

        # Workaround for HDF5 not pickleable: https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        if self.h5data is None:
            self.h5data = h5py.File(self.hdf5_path, "r")
        temp_idx = idx
        core_idx = 0
        for l in self.lengths:
            if temp_idx - self.lengths[core_idx] < 0:
                break
            temp_idx -= self.lengths[core_idx]
            core_idx += 1

        core_id = self.cores[core_idx]
        patch: np.ndarray = self.h5data[core_id][()][temp_idx]
        if self.is_features:
            label = self.h5data[core_id].attrs["y"]
            patch = torch.tensor(patch)
        else:
            label = self.h5data[core_id].attrs["label"]
            if self.transform:
                patch = self.transform(patch)
            # patch = torch.tensor(patch)

        return patch, torch.tensor(int(label))
