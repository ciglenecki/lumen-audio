from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import BaseFinetuning, Callback
from torch.optim.optimizer import Optimizer

from src.utils.utils_train import print_modules


class TensorBoardHparamFixer(pl.Callback):
    """After extensive reserach I've come to the conclusion that tensorboard logger can log
    hyperparameters only ONCE, at the beggining. Therefore we send all dictionaries which we want
    to log, merge them and only then log.

    This callback also registeres metrics in the "hparams" tab in TensorBoard. This isn't done automatically by tensorboard so it has to be done manually.

    For this callback to work, default_hp_metric has to be set to false when creating
    TensorBoardLogger
    """

    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        self.config_dict = config_dict
        self.metric_dict = {
            "train/loss_epoch": 1.0,
            "val/loss_epoch": 1.0,
            "test/loss_epoch": 1.0,
            "epoch": float(0),
            "epoch_true": float(0),
        }

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        logger: pl_loggers.TensorBoardLogger = trainer.logger
        hparam_dict = {
            **pl_module.hparams,
            **self.config_dict,
        }
        logger.log_hyperparams(params=hparam_dict, metrics=self.metric_dict)
        logger.save()


class GeneralMetricsEpochLogger(pl.Callback):
    """Logs metrics."""

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        total_params = float(sum(p.numel() for p in pl_module.parameters()))
        trainable_params = float(
            sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        )
        non_trainable_params = float(
            sum(p.numel() for p in pl_module.parameters() if not p.requires_grad)
        )

        data_dict = {
            "params/total_params_epoch": total_params,
            "params/trainable_params_epoch": trainable_params,
            "params/non_trainable_params_epoch": non_trainable_params,
            "current_lr/epoch": trainer.optimizers[0].param_groups[0]["lr"],
            "epoch_true": trainer.current_epoch,
            "step": trainer.current_epoch,
        }  # send hparams to all loggers

        pl_module.log_dict(data_dict)

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        data_dict = {
            "current_lr/step": trainer.optimizers[0].param_groups[0]["lr"],
        }
        pl_module.log_dict(data_dict)
        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)


class OverrideEpochMetricCallback(Callback):
    """Override all "on_*epoch* functions so that the X axis in Tensorboard is "epoch" for all
    events.

    X axis will be epoch index instead of step index.
    """

    def __init__(self) -> None:
        super().__init__()

    def on_training_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_training_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        # has to be float!
        pl_module.log("step", float(trainer.current_epoch))


class FinetuningCallback(BaseFinetuning):
    """
    The callback which is responsible for managing lifetime of require_grad for all parameters of some model. The callback freezes the whole
    note: this class does NOT handle learning rate!


    https://tableconvert.com/excel-to-ascii
    +---------------------+--------------------------+----------------------+-----------+
    | Epoch               | Backbone (non-trainable) | Backbone (trainable) | Head      |
    +---------------------+--------------------------+----------------------+-----------+
    | n=0                 | frozen                   | frozen               | trainable |
    | n=1                 | frozen                   | frozen               | trainable |
    | n=finetune_head_epochs | frozen                   | trainable            | trainable |
    | n=2                 | frozen                   | trainable            | trainable |
    | n=3                 | frozen                   | trainable            | trainable |
    |                     |                          |                      |           |
    |                     |                          |                      |           |
    |                     |                          |                      |           |
    |                     |                          |                      |           |
    +---------------------+--------------------------+----------------------+-----------+
    """

    def __init__(self, finetune_head_epochs: int, train_bn=False) -> None:
        super().__init__()
        self.finetune_head_epochs = finetune_head_epochs
        self.curr_epoch = 0
        self.train_bn = train_bn

    def setup(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ) -> None:
        """Freeze before training can't be called here because setup happens before state dict
        load.

        Freeze depends on the state of the dict load.
        """

        ASSUMPTIONS_TEXT = "To use this callback, your Lightning Module has to implement the head() and trainable_backbone() which returns appropriate module parameters. You should pass module names in the --backbone-after and --head-after arguments. Check the model summary to find the name of some submodule (e.g. encoder.layer.10.output)"
        assert hasattr(pl_module, "head"), ASSUMPTIONS_TEXT
        assert hasattr(pl_module, "trainable_backbone"), ASSUMPTIONS_TEXT
        assert pl_module.head(), ASSUMPTIONS_TEXT
        assert pl_module.trainable_backbone(), ASSUMPTIONS_TEXT
        return

    def state_dict(self) -> Dict[str, Any]:
        return {
            "internal_optimizer_metadata": self._internal_optimizer_metadata,
            "finetune_head_epochs": self.finetune_head_epochs,
            "curr_epoch": self.curr_epoch,
            "train_bn": self.train_bn,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.finetune_head_epochs = state_dict["finetune_head_epochs"]
        self.curr_epoch = state_dict["curr_epoch"]
        self.train_bn = state_dict["train_bn"]

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_fit_start(trainer, pl_module)
        if self.curr_epoch <= self.finetune_head_epochs:
            self.freeze_before_training(pl_module)

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        """Freezes the whole module but unfreezes module's head.

        Args:
            pl_module: pl.LightningModule
        """
        self.freeze(pl_module)
        head = pl_module.head()
        self.make_trainable(head)
        print_modules(pl_module)

    def finetune_function(
        self,
        pl_module: "pl.LightningModule",
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        """Once the epoch reaches `finetune_head_epochs` the trainable backbone is unfrozen and
        it's parameteres are added to the existing optimizer.

        Args:
            pl_module:
            epoch:
            optimizer:
            opt_idx:
        """
        self.curr_epoch = epoch
        if self.curr_epoch == self.finetune_head_epochs:
            backbone_trainable = pl_module.trainable_backbone()
            BaseFinetuning.make_trainable(backbone_trainable)
            # requires_grad = True because we just made them trainable!
            params = BaseFinetuning.filter_params(
                backbone_trainable, train_bn=self.train_bn, requires_grad=True
            )
            params = BaseFinetuning.filter_on_optimizer(optimizer, params)
            if params:
                optimizer.add_param_group({"params": params})
