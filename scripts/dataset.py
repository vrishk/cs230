import torch
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
import numpy as np

import random
import functools

# Classes


class MILDataset(Dataset):
    def __init__(self, hdf5_path: str):

        self.hdf5_path = hdf5_path
        self.h5data = h5py.File(self.hdf5_path, "r")
        self.cores = list(self.h5data.keys())

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.cores)

    def __getitem__(self, idx):

        patient_id = self.cores[idx]

        patches: np.ndarray = self.h5data[patient_id][()]
        label = self.h5data[patient_id].attrs["label"]

        if len(patches) < 16:
            return [self.transform(im) for im in patches], torch.tensor(label)

        return random.sample([self.transform(im) for im in patches], 16), torch.tensor(label)

    
class NaiveDataset(Dataset):
    def __init__(self, hdf5_path: str):

        self.hdf5_path = hdf5_path
        self.h5data = h5py.File(self.hdf5_path, "r")
        self.cores = list(self.h5data.keys())
        
        self.lengths = [len(self.h5data[i]) for i in self.cores]

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):

        core_idx = 0
        for l in self.lengths:
            if idx - self.lengths[core_idx] < 0:
                break
            idx -= self.lengths[core_idx]
            core_idx += 1
        
        core_id = self.cores[core_idx]
        patch: np.ndarray = self.h5data[core_id][()][idx]
        label = self.h5data[core_id].attrs["label"]

        return self.transform(patch), torch.tensor(label)

