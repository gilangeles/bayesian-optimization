from typing import TYPE_CHECKING
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer

import torch
from torchmetrics.functional import accuracy

if TYPE_CHECKING:
    from ..model.resnet import ResNet


class LitResNet(LightningModule):
    def __init__(self, model: "ResNet", lr: float = 1e-3):
        """streamlines neural network training, validation and test loops

        Args:
            model (ResNet): ResNet model instance
            lr (_float_, optional): learning rate. Defaults to 1e-3.
        """
        super().__init__()

        self.save_hyperparameters("lr")
        self.model = model
        self.lr = lr
        self.trainer = Trainer

        self.min_val_loss = float("inf")
        self.val_losses: list[torch.Tensor] = []

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)

        if stage:
            self.log(
                f"{stage}_step_loss", loss, on_epoch=True, prog_bar=True, logger=True
            )
            self.log(
                f"{stage}_step_acc", acc, on_epoch=True, prog_bar=True, logger=True
            )
            if self.lr:
                metrics_dict = {f"{stage}_lr": self.hparams.lr}
            self.log_dict(
                metrics_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True
            )
            return {f"{stage}_step_loss": loss, f"{stage}_step_acc": acc}

    def on_validation_epoch_end(self):
        """calculates the minimum loss across all epochs,
        this is the observed f(x) for the bayesian
        optimization function
        """
        loss = self.trainer.callback_metrics["val_step_loss"].item()

        self.val_losses.append(loss)
        self.min_val_loss = min(self.min_val_loss, loss)

        self.log("val_loss_min", self.min_val_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        values = self.evaluate(batch, "val")
        return values["val_step_loss"]

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        print(self.hparams.lr)
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=1e-4,
        )

        return optimizer


# class MetricsLogger(Callback):
#     def __init__(self):
#         super().__init__()
#         self.train_losses = []
#         self.val_losses = []
#         self.train_accs = []
#         self.val_accs = []

#     def on_validation_epoch_end(self, trainer, pl_module):
#         # Extract metrics from the trainer's logger after the validation epoch ends
#         train_loss = trainer.callback_metrics.get('train_loss_epoch', None)
#         val_loss = trainer.callback_metrics.get('val_loss_epoch', None)
#         train_acc = trainer.callback_metrics.get('train_acc_epoch', None)
#         val_acc = trainer.callback_metrics.get('val_acc_epoch', None)
