# Imports
import torch
from torch import nn
import numpy as np

from collections import OrderedDict

# Local imports
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "../"))  # noqa

from naive_base import NaiveBase
from tripletnet_core import TripletNetCore

# For `tripletnet_core.py` to be visible for higher dirs

sys.path.append(os.getcwd()) # noqa

# Path to pre-trained TripleNet model
PATH_TO_PRETRAINED = '/deep/group/aihc-bootcamp-fall2021/lymphoma/models/Camelyon16_pretrained_model.pt'


class TripletNetNaive(NaiveBase):
    def __init__(self, finetune: bool = False, layers_tune: int = 9, lr=1e-3, *args, **kwargs):
        super().__init__(lr=lr, **kwargs)

        self.finetune = finetune
        print(f'finetune value:{self.finetune}')
        # Load pre-trained network:
        state_dict = torch.load(PATH_TO_PRETRAINED)

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
            print('we finetune!')
            largest = 0
            for name, param in enumerate(model.named_parameters()):
                param = param[1]
                param.requires_grad = False
                largest = int(name)

            # Set up how many layers to finetune
            if layers_tune is not None:
                print(f"Finetuning {layers_tune} layers!")
                for name, param in enumerate(model.named_parameters()):
                    if int(name) >= largest - layers_tune:
                        param[1].requires_grad = True

        # set the pretrained weights as the network
        self.feature_extractor = model

        # set the linear classifier
        # use the classifier setup in the paper
        self.classifier = nn.Linear(256*3, self.hparams.num_classes)

#         # set the loss criterion -- CE
#         # self.criterion = nn.CrossEntropyLoss()

#         # set the loss criterion -- Focal Loss 
#         # (https://github.com/AdeelH/pytorch-multi-class-focal-loss)
#         self.criterion = torch.hub.load(
#             'adeelh/pytorch-multi-class-focal-loss',
#             model='FocalLoss',
#             alpha=torch.tensor([0.05, 0.05, 0.125, 0.1, 0.1, 0.125, 0.1, 0.1, 0.25]),
#             gamma=2,
#             reduction='mean',
#             force_reload=False
#         )

    def forward(self, x):
        # Forward step
        x = self.feature_extractor(x)              # representations
        x = self.classifier(x)                     # classifications
        return x


if __name__ == "__main__":
    model = TripletNetNaive(lr=1e-5, num_classes=9, finetune=False)
    print(model)












# class TripletNetNaive(NaiveBase):
#     def __init__(self, finetune: bool = False, lr=1e-3, *args, **kwargs):
#         super().__init__(lr=lr, **kwargs)

#         self.finetune = finetune

#         # Load pre-trained network:
#         state_dict = torch.load(PATH_TO_PRETRAINED) ## TODO: change this to pytorch lightning

#         # create new OrderedDict that does not contain `module`
#         new_state_dict = OrderedDict()
#         for k, v in state_dict['model'].items():
#             name = k[7:]  # remove `module.`
#             new_state_dict[name] = v

#         # load pretrained weights onto TripletNet model
#         model = TripletNetCore()
#         model.load_state_dict(new_state_dict)

#         # if we finetune - only train the classifier, as opposed to e2e - freeze the network
#         if self.finetune:
#             for name, param in enumerate(model.named_parameters()):
#                 param = param[1]
#                 param.requires_grad = False

#         # set the pretrained weights as the network
#         self.feature_extractor = model

#         # set the linear classifier
#         # use the classifier setup in the paper
#         self.classifier = nn.Linear(256*3, self.hparams.num_classes)

#         # set the loss criterion -- CE
#         self.criterion = nn.CrossEntropyLoss()


#     def forward(self, x):
#         # Forward step
#         x = self.feature_extractor(x)              # representations
#         print(f"shape of output from extractor: {x.shape}")
#         x = self.classifier(x)                     # classifications
#         print(f'shape of output after classifier: {x.shape}')
#         return x

