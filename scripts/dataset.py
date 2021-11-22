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


# class NaiveDataset(Dataset):
#     def __init__(self, hdf5_path: str, is_features: bool = False, transform=transforms.ToTensor()):
#
#         self.hdf5_path = hdf5_path
#         # Workaround for HDF5 not pickleable: https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
#         self.h5data = None
#
#         h5data = h5py.File(self.hdf5_path, "r")
#         self.cores = list(h5data.keys())  # list of single cores on TMA1-8
#
#         self.num_cores = len(self.cores)
#
#         # total patches we would like to iterate
#         # supplement patches that are not long enough
#         self.lengths = [max([len(h5data[i]) for i in self.cores])] * self.num_cores
#
#         self.is_features = is_features
#         self.transform = transform
#
#     def __len__(self):
#         # total number of patches on all TMAs
#         return sum(self.lengths)
#
#     def __getitem__(self, idx):
#         # Workaround for HDF5 not pickleable: https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
#         if self.h5data is None:
#             self.h5data = h5py.File(self.hdf5_path, "r")
#
#         num_iter = idx // self.num_cores
#         num_steps = idx % self.num_cores
#
#         # from which core do we sample this patch
#         core_id = self.cores[num_steps]
#
#         # if the number of patches in the core is not enough
#         if len(self.h5data[core_id][()]) <= num_iter:
#             patch_rand = np.random.choice(range(len(self.h5data[core_id][()])), size=1)
#             patch = self.h5data[core_id][()][patch_rand]
#         else:  # if enough, then pick the one at the num_iter
#             patch = self.h5data[core_id][()][num_iter]
#
#         if self.is_features:
#             print("not implemented for features")
#             label = None
#         else:
#             label = self.h5data[core_id].attrs["label"]
#             if len(patch.shape) == 4:
#                 patch = patch.squeeze(0)
#             if self.transform:
#                 patch = self.transform(patch) / 255
#         return patch, torch.tensor(int(label))
#



# # This is the formal Dataset class!!! (without normalization yet)
# class NaiveDataset(Dataset):
#     def __init__(self, hdf5_path: str, is_features: bool = False, transform=transforms.ToTensor()):
#
#         self.hdf5_path = hdf5_path
#         # Workaround for HDF5 not pickleable: https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
#         self.h5data = None
#
#         h5data = h5py.File(self.hdf5_path, "r")
#         self.cores = list(h5data.keys())
#
#         self.lengths = [len(h5data[i]) for i in self.cores]
#         self.is_features = is_features
#         self.transform = transform
#
#     def __len__(self):
#         return sum(self.lengths)
#
#     def __getitem__(self, idx):
#
#         # Workaround for HDF5 not pickleable: https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
#         if self.h5data is None:
#             self.h5data = h5py.File(self.hdf5_path, "r")
#
#         core_idx = 0
#         for l in self.lengths:
#             if idx - self.lengths[core_idx] < 0:
#                 break
#             idx -= self.lengths[core_idx]
#             core_idx += 1
#
#         core_id = self.cores[core_idx]
#         patch: np.ndarray = self.h5data[core_id][()][idx]
#         if self.is_features:
#             label = self.h5data[core_id].attrs["y"]
#             patch = torch.tensor(patch)
#         else:
#             label = self.h5data[core_id].attrs["label"]
#             if self.transform:
#                 patch = self.transform(patch)
#
#
#         return patch, torch.tensor(int(label))



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
                                 (0.0101850529, 0.0099803880, 0.00758414449))])

        self.transform = transform

    def __len__(self):
        return sum(self.lengths)//2

    def __getitem__(self, idx):

        # Workaround for HDF5 not pickleable: https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        if self.h5data is None:
            self.h5data = h5py.File(self.hdf5_path, "r")

        temp_idx = 2*idx
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

        return patch, torch.tensor(int(label))
