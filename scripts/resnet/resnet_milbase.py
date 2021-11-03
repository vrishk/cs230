import torch
from torch import nn
import pytorch_lightning as pl
from torchvision import models
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import ConfusionMatrix

from collections import OrderedDict

import seaborn as sns

from mil_base import MILBase


class ResNetMB(MILBase):
    def __init__(
        self, 
        size: int,
        lr: float, 
        num_classes: int,
        finetune: bool = False
    ):
        super().__init__(lr, num_classes, finetune)

        self.size = size

        # Load pre-trained network:
        if self.size == 18:
            model = models.resnet18(pretrained=True)
        elif self.size == 50:
            model = models.resnet50(pretrained=True)
        else:
            raise NotImplementedError("the size is not supported")

        # if we finetune - only train the classifier, as opposed to e2e - freeze the network
        if self.finetune:
            for param in model.parameters():
                param.requires_grad = False
        
        # set the pretrained weights as the network
        self.feature_extractor = model
        
        # set the linear classifier
        # use the classifier setup in the paper
        self.classifier = nn.Linear(1000, self.num_classes)

    

    def forward(self, x):
        # Forward step
        x = self.feature_extractor(x).flatten(1)   # representations
        x = self.classifier(x)                     # classifications
        return x
    

    def aggregate(self, y_hats):
        return torch.max(y_hats, dim=0)[0].unsqueeze(0)
        

if __name__ == "__main__":
    model = ResNetMB(18, 1e-5, 9, False)
    print(model)

