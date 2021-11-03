import torch
from torch import nn
from torchvision import models

from collections import OrderedDict

from vanilla import VanillaNet

# Path to pre-trained TripleNet model
PATH_TO_PRETRAINED = '/deep/group/aihc-bootcamp-fall2021/lymphoma/models/Camelyon16_pretrained_model.pt'


class ResNetVN(VanillaNet):
    def __init__(
        self, 
        size: int,
        lr: float, 
        num_classes: int,
        finetune: bool = False
    ):
        super().__init__(self, lr, num_classes, finetune)
        
        self.size = size

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
        self.classifier = nn.Linear(100, self.num_classes)
        
        # set the loss criterion -- CE
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        # Forward step
        x = self.feature_extractor(x).flatten(1)   # representations
        x = self.classifier(x)                     # classifications
        return x

if __name__ == "__main__":
    model = ResNetVN(18, 1e-5, 9, False)
    print(model)

