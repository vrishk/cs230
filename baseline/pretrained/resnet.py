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

# Path to pre-trained TripleNet model
PATH_TO_PRETRAINED = '/deep/group/aihc-bootcamp-fall2021/lymphoma/models/Camelyon16_pretrained_model.pt'

# Model adapted from net.py as the template model to load pretrained weights
class ResNetCore(nn.Module):

    def __init__(self):
        super(ResNetCore, self).__init__()

        # set the model
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Sequential()
        self.model = model
        self.fc = nn.Sequential(nn.Linear(512*2, 512),
                                 nn.ReLU(True), nn.Linear(512, 256))

    def forward(self, i):

        E1 = self.model(i)

        f12 = self.fc(E1)

        return f12


class ResNet(pl.LightningModule):
    def __init__(
        self, 
        lr: float, 
        num_classes: int,
        finetune: bool = False
    ):
        super().__init__()
        self.lr = lr
        
        # Load pre-trained network:
        model = ResNetCore()
        
        # if we finetune - only train the classifier, as opposed to e2e - freeze the network
        if finetune:
            for name, param in enumerate(model.named_parameters()):
                param = param[1]
                param.requires_grad=False
        
        # set the pretrained weights as the network
        self.feature_extractor = model
        
        # set the linear classifier
        # use the classifier setup in the paper
        self.classifier = nn.Linear(256, num_classes)
        
        # set the loss criterion -- CE
        self.criterion = nn.CrossEntropyLoss()
        
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        acc = Accuracy()
        self.train_accuracy = acc.clone()
        self.val_accuracy = acc.clone()
        self.test_accuracy = acc.clone()
        
        # Add confusion matrix into trianing metrics
        cm = ConfusionMatrix(9)
        self.train_cm = cm.clone()
        self.val_cm = cm.clone()
        self.test_cm = cm.clone()
        
        # ensures params passed to LightningModule will be saved to ckpt
        # allows to access params with 'self.hparams' attribute
        
    def forward(self, x):
        # Forward step
        x = self.feature_extractor(x).flatten(1)   # representations
        x = self.classifier(x)                     # classifications
        return x
    
    def aggregate(self, y_hats):
        return torch.max(y_hats, dim=0)[0].unsqueeze(0)
        
    def infer(self, bag, y):
        y_hats = []
        for x in bag:
            y_hats.append(self(x).squeeze())
            
        y_hat = self.aggregate(torch.stack(y_hats, dim=0))
        
        return y_hat

    def configure_optimizers(self):
        # only train parameters that are not frozen
        parameters = self.parameters()
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        
        optimizer = torch.optim.Adam(trainable_parameters, lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        bag, y = batch
        y_hat = self.infer(bag, y)
        loss = self.criterion(y_hat, y)
        acc = self.train_accuracy(y_hat, y)
    
        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {"loss": loss, "preds": y_hat.detach(), "targets": y.detach()}
    
    def validation_step(self, batch, batch_idx):
        bag, y = batch
        y_hat = self.infer(bag, y)
        loss = self.criterion(y_hat, y)
        acc = self.val_accuracy(y_hat, y)
        
        # log validation metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {"loss": loss, "preds": y_hat.detach(), "targets": y.detach()}
    
    def test_step(self, batch, batch_idx):
        bag, y = batch
        y_hat = self.infer(bag, y)
        loss = self.criterion(y_hat, y)
        acc = self.test_accuracy(y_hat, y)
        
        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "preds": y_hat.detach(), "targets": y.detach()}
    
    def training_epoch_end(self, outputs):
        
        targets = torch.cat([tmp['targets'] for tmp in outputs])
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        confusion_matrix = self.train_cm(preds, targets)

        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index = range(9), columns=range(9))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)
        
        self.logger.experiment.add_figure("train_cm", fig_, self.current_epoch)
    
    def validation_epoch_end(self, outputs):
        
        try:
            if outputs[0].size() == []:
                return
        except:
            return
        
        targets = torch.cat([tmp['targets'] for tmp in outputs])
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        confusion_matrix = self.val_cm(preds, targets)

        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index = range(9), columns=range(9))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)        
        self.logger.experiment.add_figure("val_cm", fig_, self.current_epoch)

if __name__ == "__main__":
    model = ResNet(1e-5, 9, False)
    print(model)