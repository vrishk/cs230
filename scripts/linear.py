import torch
from torch import nn

from collections import OrderedDict

from naive_base import NaiveBase
# from mil_base import MILBase


class LinearNaive(NaiveBase):
    def __init__(self, in_features: int, *args, **kwargs):
        super().__init__(**kwargs)
        
        # set the linear classifier
        # use the classifier setup in the paper
        self.classifier = nn.Linear(in_features, self.hparams.num_classes)
        
        # set the loss criterion -- CE
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        
        # Assuming `x` is the representation vector
        
        # Forward step
        x = self.classifier(x)                     # classifications
        return x

    

# class LinearMIL(MILBase):
#     def __init__(self, in_features: int, *args, **kwargs):
#         super().__init__(**kwargs)
        
#         # set the linear classifier
#         # use the classifier setup in the paper
#         self.classifier = nn.Linear(in_features, self.hparams.num_classes)
        
#         # set the loss criterion -- CE
#         self.criterion = nn.CrossEntropyLoss()
    
#     def forward(self, x):
        
#         # Assuming `x` is the representation vector
        
#         # Forward step
#         x = self.classifier(x)                     # classifications
#         return x
    
#     def aggregate(self, y_hats):
#         return torch.max(y_hats, dim=0)[0].unsqueeze(0)

if __name__ == "__main__":
    model = LinearNaive(256 * 3, lr=1e-5, num_classes=9, fine_tune=False)
    print(model)

