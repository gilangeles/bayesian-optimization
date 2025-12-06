from pytorch_lightning.callbacks import Callback
from loguru import logger


class StopOnLowValLoss(Callback):
    def __init__(self, target_loss):
        self.tolerance = 0.01
        self.target_loss = target_loss

    def on_validation_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")

        if val_loss is None:
            return

        if abs(val_loss - self.target_loss) <= self.tolerance:
            logger(
                f"Stopping training: val_loss reached {val_loss:.4f} ≤ {self.target_loss}"
            )
            trainer.should_stop = True
