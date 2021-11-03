import torch
from torch import nn
import pytorch_lightning as pl
from torchvision import models
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import ConfusionMatrix

from collections import OrderedDict

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from vanilla import VanillaNet

# Path to pre-trained TripleNet model
PATH_TO_PRETRAINED = '/deep/group/aihc-bootcamp-fall2021/lymphoma/models/Camelyon16_pretrained_model.pt'

# Model adapted from net.py as the template model to load pretrained weights
class TripletNetCore(nn.Module):

    def __init__(self):
        super(TripletNetCore, self).__init__()

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




class TripletNet(VanillaNet):
    def __init__(
        self, 
        lr: float, 
        num_classes: int,
        finetune: bool = False
    ):
        super().__init__(self, lr, num_classes, finetune)
        
        # Load pre-trained network:
        state_dict = torch.load(PATH_TO_PRETRAINED) ## TODO: change this to pytorch lightning

        # create new OrderedDict that does not contain `module`
        new_state_dict = OrderedDict()
        for k, v in state_dict['model'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
            
        # load pretrained weights onto TripletNet model
        model = TripletNetCore()
        model.load_state_dict(new_state_dict)
        
        # if we finetune - only train the classifier, as opposed to e2e - freeze the network
        if self.finetune:
            for name, param in enumerate(model.named_parameters()):
                param = param[1]
                param.requires_grad=False
        
        # set the pretrained weights as the network
        self.feature_extractor = model
        
        # set the linear classifier
        # use the classifier setup in the paper
        self.classifier = nn.Linear(256*3, self.num_classes)
        
        # set the loss criterion -- CE
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        # Forward step
        x = self.feature_extractor(x).flatten(1)   # representations
        x = self.classifier(x)                     # classifications
        return x

if __name__ == "__main__":
    model = TripletNet(1e-5, 9, False)
    print(model)

