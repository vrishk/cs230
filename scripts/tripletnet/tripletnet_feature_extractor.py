# Imports
import torch

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


class TripletNetFeatureExtractor(NaiveBase):
    def __init__(self):
        super().__init__()

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
                param.requires_grad = False

        # set the pretrained weights as the network
        self.feature_extractor = model

    def forward(self, x):
        # Forward step
        x = self.feature_extractor(x)   # representations
        return x


if __name__ == "__main__":
    model = TripletNetFeatureExtractor()
    print(model)
    data_path = ''

    # open the hdf5 file


