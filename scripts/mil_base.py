import torch
from torch import nn
import pytorch_lightning as pl
from torchvision import models
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import ConfusionMatrix

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



class MILBase(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        
        self.criterion = nn.CrossEntropyLoss()

        self.bare_metrics = {'acc': Accuracy(), 'cm': ConfusionMatrix(9)}

        self.metrics = {
            i: {} 
            for i in self.bare_metrics
        }
        
    def forward(self, x):
        raise NotImplementedError("Forward must be implemented for inheriting MILBase")
    
    def aggregate(self, y_hats):
        raise NotImplementedError("Aggregate must be implemented for inheriting MILBase")
        
    def infer(self, bag):
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
        y_hat = self.infer(bag)
        loss = self.criterion(y_hat, y)
        acc = self.train_accuracy(y_hat, y)
    
        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {"loss": loss, "preds": y_hat.detach(), "targets": y.detach()}
    
    def validation_step(self, batch, batch_idx):
        bag, y = batch
        y_hat = self.infer(bag)
        loss = self.criterion(y_hat, y)
        acc = self.val_accuracy(y_hat, y)
        
        # log validation metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {"loss": loss, "preds": y_hat.detach(), "targets": y.detach()}
    
    def test_step(self, batch, batch_idx):
        bag, y = batch
        y_hat = self.infer(bag)
        loss = self.criterion(y_hat, y)
        acc = self.test_accuracy(y_hat, y)
        
        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "preds": y_hat.detach(), "targets": y.detach()}
    
    def training_epoch_end(self, outputs):
        try:
            if outputs[0].size() == []:
                return
        except:
            return
        
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

