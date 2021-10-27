import h5py
import inspect
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as T
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image

PATH_TO_TRAIN = "/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/data_splits/train.hdf5"
PATH_TO_VAL = "/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/data_splits/val.hdf5"
PATH_TO_TEST = "/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/data_splits/test.hdf5"
PATH_TO_PRETRAINED = '/deep/group/aihc-bootcamp-fall2021/lymphoma/models/Camelyon16_pretrained_model.pt'
use_gpu = torch.cuda.is_available()

def gpu(x, use_gpu=use_gpu):
    if use_gpu:
        return x.cuda()
    else:
        return x

class HDF5Dataset(Dataset):
    
    def __init__(self, hdf5_path: str, transform=None):
        self.hdf5_path = hdf5_path
        self.h5data = h5py.File(self.hdf5_path, "r")
        self.cores = list(self.h5data.keys())
        self.transform = transform
        
    def __len__(self):
        return len(self.cores)
    
    def __getitem__(self, idx):
        patient_id = self.cores[idx]
        patches = self.h5data[patient_id][()]
        label = self.h5data[patient_id].attrs["label"]
        if self.transform:
            patches = self.transform(patches)
        return patches, torch.tensor(label)

class TripletNet_Finetune(nn.Module):

    def __init__(self):
        super(TripletNet_Finetune, self).__init__()

        # set the model
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Sequential()
        self.model = model
        self.fc = nn.Sequential(nn.Linear(512*2, 512),
                                 nn.ReLU(True), nn.Linear(512, 256))

    def forward(self, i):

        E1 = self.model(i)
        E2 = self.model(i)
        E3 = self.model(i)

        # Pairwise concatenation of features
        E12 = torch.cat((E1, E2), dim=1)
        E23 = torch.cat((E2, E3), dim=1)
        E13 = torch.cat((E1, E3), dim=1)

        f12 = self.fc(E12)
        f23 = self.fc(E23)
        f13 = self.fc(E13)

        features = torch.cat((f12, f23, f13), dim=1)
        return features

def build_dataloaders():
	transform = T.Compose([
	    # T.CenterCrop(224),
	    T.Lambda(lambda patches: torch.stack([T.ToTensor()(patch) for patch in patches]))
	    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	train_dataset = HDF5Dataset(PATH_TO_TRAIN, transform=transform)
	val_dataset = HDF5Dataset(PATH_TO_VAL, transform=transform)

	train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=1, shuffle=True)
	return (train_dataloader, val_dataloader)

def build_triple_net_model():
	triplenet_model = TripletNet_Finetune()
	state_dict = torch.load(PATH_TO_PRETRAINED)
	new_state_dict = OrderedDict()
	for k, v in state_dict['model'].items():
	    name = k[7:]  # remove `module.`
	    new_state_dict[name] = v
	triplenet_model.load_state_dict(new_state_dict)

	# Set requires_grad=False for all parameters since we are only
	# performing forward passes for the model.
	for p in triplenet_model.parameters():
		p.requires_grad = False
	return triplenet_model

def process_examples(model, dataloader, model_name, mode):
	output_filename = f"{model_name}_{mode}_features.hdf5"
	print(f"Writing to: {output_filename}")
	with h5py.File(output_filename, 'w') as f:
		batch_num = 0
		mini_batch_size = 8
		for (x,y) in dataloader:
		    print(f"Batches Processed: {batch_num}")
		    patches = x.squeeze(dim=0)
		    temp = 0
		    model_batch_output = []
		    print(f"Input Patches Shape {patches.shape}")
		    for mini_batch_start in range(0, patches.shape[0], mini_batch_size):
		        mini_batch_end = mini_batch_start + mini_batch_size
		        mini_batch_input = patches[mini_batch_start:mini_batch_end]
		        mini_batch_output = model(mini_batch_input)
		        model_batch_output.append(mini_batch_output)
		    model_output = np.concatenate(model_batch_output, axis=0)
		    print(f"Model Output Shape {model_output.shape}")
		    assert(model_output.shape[0] == patches.shape[0])
		    dset = f.create_dataset(str(batch_num), data=model_output, dtype=np.float32)
		    dset.attrs['y'] = y
		    batch_num += 1

if __name__ == "__main__":
	print("Cuda is available:", torch.cuda.is_available())
	print("Cuda device count:", torch.cuda.device_count())
	train_dataloader, val_dataloader = build_dataloaders()
	model = build_triple_net_model()
	process_examples(model, train_dataloader, "triplenet", "train")
	process_examples(model, val_dataloader, "triplenet", "val")