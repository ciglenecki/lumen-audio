import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import Callback


class InvalidArgument(Exception):
    pass


class LogMetricsAsHyperparams(pl.Callback):
    """Registeres metrics in the "hparams" tab in TensorBoard This isn't done automatically by
    tensorboard so it has to be done manually.

    For this callback to work, default_hp_metric has to be set to false when creating
    TensorBoardLogger
    """

    def __init__(self) -> None:
        super().__init__()
        min_value = float(0)
        max_value = float(1e5)
        self.hyperparameter_metrics_init = {
            "train/loss_epoch": max_value,
            "train/acc_epoch": min_value,
            "val/loss_epoch": max_value,
            "val/acc_epoch": min_value,
            "test/loss_epoch": max_value,
            "test/acc_epoch": min_value,
            "epoch": float(0),
            "epoch_true": float(0),
        }

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

        if pl_module.loggers:
            for logger in pl_module.loggers:  # type: ignore
                logger: pl_loggers.TensorBoardLogger
                logger.log_hyperparams(pl_module.hparams, self.hyperparameter_metrics_init)  # type: ignore


class OnTrainEpochStartLogCallback(pl.Callback):
    """Logs metrics."""

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

        total_params = int(sum(p.numel() for p in pl_module.parameters()))
        trainable_params = int(sum(p.numel() for p in pl_module.parameters() if p.requires_grad))
        non_trainable_params = int(sum(p.numel() for p in pl_module.parameters() if not p.requires_grad))

        data_dict = {
            "total_params/epoch": total_params,  # type: ignore
            "trainable_params/epoch": trainable_params,  # type: ignore
            "non_trainable_params/epoch": non_trainable_params,  # type: ignore
            "current_lr/epoch": trainer.optimizers[0].param_groups[0]["lr"],
            "epoch_true": trainer.current_epoch,
            "step": trainer.current_epoch,
        }  # send hparams to all loggers

        for logger in trainer.loggers:  # TODO: check if this works
            logger.log_hyperparams(data_dict)

        # pl_module.log_dict(data_dict)
        # # TODO: maybe, put this in val_start ? and start with --val check = 1 !!!remove this when loading from legacy
        # pl_module.log("val/loss_epoch", 100000)
        # pl_module.log("val/haversine_distance_epoch", 100000)

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, unused: int = 0
    ) -> None:
        data_dict = {
            "current_lr/step": trainer.optimizers[0].param_groups[0]["lr"],
        }
        pl_module.log_dict(data_dict)


class OverrideEpochMetricCallback(Callback):
    """Override the X axis in Tensorboard for all "epoch" events.

    X axis will be epoch index instead of step index
    """

    def __init__(self) -> None:
        super().__init__()

    def on_training_epoch_end(self, trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_training_epoch_start(self, trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_start(self, trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_start(self, trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(self, trainer, pl_module: pl.LightningModule):
        pl_module.log("step", float(trainer.current_epoch))
