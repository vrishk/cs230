import torch
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
import numpy as np

import random

# Class


class HDF5Dataset(Dataset):
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

        if len(patches) < 8:
            return [self.transform(im) for im in patches], torch.tensor(label)

        return random.sample(
            [self.transform(im) for im in patches], 8), torch.tensor(label)

