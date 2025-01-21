import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np


class CNN(pl.LightningModule):
    def __init__(self, cls_head=False, num_classes=2, cls_weight=1.0):
        super().__init__()
        self.cls_head = cls_head
        self.cls_weight = cls_weight

        self.latent_dim = 256

        # self.backbone = torch.nn.Sequential(
        #     torch.nn.Conv1d(in_channels=18, out_channels=10, kernel_size=3, padding='same'),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, padding='same'),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv1d(in_channels=10, out_channels=1, kernel_size=3, padding='same'),
        #     torch.nn.ReLU(),
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(50, self.latent_dim),
        #     torch.nn.ReLU(),
        # )

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=18, out_channels=20, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=20, out_channels=20, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=20, out_channels=10, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(500, self.latent_dim),
            torch.nn.ReLU(),
        )

        self.regression = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 1),
            torch.nn.Flatten(start_dim=0)
        )

        if self.cls_head:
            self.classification = torch.nn.Sequential(
                torch.nn.Linear(self.latent_dim, num_classes)
            )
            self.loss_cls = torch.nn.BCEWithLogitsLoss()  # or CE if mutually exclusive concepts!

    def forward(self, x):
        feat = self.backbone(x)
        regres = self.regression(feat)
        if self.cls_head:
            classi = self.classification(feat)
            # return regres, feat, classi
            return regres, classi
        # return regres, feat
        return regres

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y, c = train_batch
        #x = x.view(x.size(0),-1)
        if self.cls_head:
            regres, classi = self.forward(x)
            loss_mse = F.mse_loss(regres, y)
            loss_cls = self.loss_cls(classi, c)
            c_pred = (classi > 0.5).int()
            acc = (c == c_pred).float().mean()
            total_loss = loss_mse + self.cls_weight * loss_cls
            self.log('task_loss', loss_mse, on_epoch=True)
            self.log("rmse_unscaled", np.sqrt(loss_mse.cpu().detach())*100, on_epoch=True)
            self.log('concept_loss', loss_cls, on_epoch=True)
            self.log('c_accuracy', acc, on_epoch=True)
        else:
            regres = self.forward(x)
            loss_mse = F.mse_loss(regres, y)
            total_loss = loss_mse
            self.log('task_loss', loss_mse, on_epoch=True)
            self.log("rmse_unscaled", np.sqrt(loss_mse.cpu().detach())*100, on_epoch=True)
        self.log('loss', total_loss, on_epoch=True)
        return total_loss

    def validation_step(self, val_batch, batch_idx):
        x, y, c = val_batch
        #x = x.view(x.size(0),-1)
        if self.cls_head:
            regres, classi = self.forward(x)
            loss_mse = F.mse_loss(regres, y)
            loss_cls = self.loss_cls(classi, c)
            total_loss = loss_mse + self.cls_weight * loss_cls
            c_pred = (classi > 0.5).int()
            acc = (c == c_pred).float().mean()
            self.log('val_task_loss', loss_mse)
            self.log("val_rmse_unscaled", np.sqrt(loss_mse.cpu().detach())*100, on_epoch=True)
            self.log('val_concept_loss', loss_cls)
            self.log('val_c_accuracy', acc)
        else:
            regres = self.forward(x)
            loss_mse = F.mse_loss(regres, y)
            total_loss = loss_mse
            self.log('val_task_loss', loss_mse)
            self.log("val_rmse_unscaled", np.sqrt(loss_mse.cpu().detach())*100, on_epoch=True)
        self.log('val_loss', total_loss)
        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])
