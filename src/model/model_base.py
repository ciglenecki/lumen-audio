from abc import ABC
from argparse import Namespace
from typing import Any, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.loggers import TensorBoardLogger

import src.config.config_defaults as config_defaults
from src.model.optimizers import OptimizerType, SchedulerType, our_configure_optimizers
from src.train.metrics import get_metrics
from src.utils.utils_functions import add_prefix_to_keys
from src.utils.utils_train import MetricMode, OptimizeMetric, get_all_modules_after


class ModelBase(pl.LightningModule, ABC):

    loggers: list[TensorBoardLogger]

    def __init__(
        self,
        batch_size: int = config_defaults.DEFAULT_BATCH_SIZE,
        epochs: Optional[int] = config_defaults.DEFAULT_EPOCHS,
        head_after: str | None = None,
        backbone_after: str | None = None,
        lr: float = config_defaults.DEFAULT_LR,
        lr_onecycle_max: int | None = config_defaults.DEFAULT_LR_ONECYCLE_MAX,
        lr_warmup: Optional[float] = config_defaults.DEFAULT_LR_WARMUP,
        metric_mode: MetricMode = config_defaults.DEFAULT_PLATEAU_EPOCH_PATIENCE,
        num_labels: int = config_defaults.DEFAULT_NUM_LABELS,
        optimization_metric: OptimizeMetric = config_defaults.DEFAULT_METRIC_MODE,
        optimizer_type: OptimizerType = config_defaults.DEFAULT_OPTIMIZE_METRIC,
        plateau_epoch_patience: int = config_defaults.DEFAULT_PLATEAU_EPOCH_PATIENCE,
        pretrained: bool = config_defaults.DEFAULT_PRETRAINED,
        scheduler_type: SchedulerType = config_defaults.DEFAULT_LR_SCHEDULER,
        unfreeze_at_epoch: Optional[int] = config_defaults.DEFAULT_UNFREEZE_AT_EPOOCH,
        weight_decay: float = config_defaults.DEFAULT_WEIGHT_DECAY,
        log_per_instrument_metrics=config_defaults.DEFAULT_LOG_PER_INSTRUMENT_METRICS,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            batch_size:

            epochs: number of training epochs. Used only if the LR scheduler depends on epochs size (onecycle, cosine...). Reduce on plateau LR scheduler doesn't get affected by this argument.

            head_after: Name of the submodule after which the all submodules are considered as head, e.g. classifier.dense

            backbone_after: Name of the submodule after which the all submodules are considered as backbone, e.g. layer.11.dense"

            lr: backbone lr, the one which will be used after the finetuning is over

            lr_onecycle_max: used only if onecycle scheduler is used

            lr_warmup: starting lr used for head finetuning, if provided, self.lr gets intialized to this value

            metric_mode: MetricMode.MAX or MetricMode.MIN

            num_labels: number of labels (instruments) we want to classify

            optimization_metric: metric which is responsible for model checkpointing (not done here!) and reducing lr via the reduce on plateau lr scheduler

            optimizer_type: OptimizerType.ADAM, ADAMW

            plateau_epoch_patience: how many epochs should pass, without metric getting better, before reduce on plateau scheduler lowers the lr

            pretrained: use pretrained model

            scheduler_type: SchedulerType.ONECYCLE, SchedulerType.PLATEAU...

            unfreeze_at_epoch: at which epoch should the model be unfrozen? warning: unfreezing is not done here. It's done by the finetuning callback.

            weight_decay
        """

        super().__init__(*args, **kwargs)

        self.backbone_after = backbone_after
        self.batch_size = batch_size
        self.epochs = epochs
        self.has_finetuning = unfreeze_at_epoch is not None
        self.head_after = head_after
        self.log_per_instrument_metrics = log_per_instrument_metrics
        self.lr_backbone = lr  # lr once backbone gets unfrozen
        self.lr_onecycle_max = lr_onecycle_max
        self.lr_warmup = lr_warmup  # starting warmup lr
        self.metric_mode = metric_mode
        self.num_labels = num_labels
        self.optimization_metric = optimization_metric
        self.optimizer_type = optimizer_type
        self.plateau_epoch_patience = plateau_epoch_patience
        self.pretrained = pretrained
        self.scheduler_type = scheduler_type
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.weight_decay = weight_decay

        if self.lr_warmup:
            self.lr = self.lr_warmup
        else:
            self.lr = lr

        assert int(bool(self.unfreeze_at_epoch)) + int(bool(self.lr_warmup)) in [
            0,
            2,
        ], "Both should exist or both shouldn't exist!"

        assert int(bool(self.head_after)) + int(bool(self.backbone_after)) in [
            0,
            2,
        ], "Both should exist or both shouldn't exist!"

        # save in case indices change with config changes
        self.backup_instruments = config_defaults.INSTRUMENT_TO_IDX

    def setup(self, stage: str) -> None:
        """Freezes everything except trainable backbone and head."""
        out = super().setup(stage)
        self.num_of_steps_in_epoch = int(
            self.trainer.estimated_stepping_batches / self.trainer.max_epochs
        )

        if self.head() is not None and self.trainable_backbone() is not None:
            BaseFinetuning.freeze(self, train_bn=False)
            BaseFinetuning.make_trainable(self.trainable_backbone())
            BaseFinetuning.make_trainable(self.head())

        if self.has_finetuning:
            self._set_finetune_until_step()
        return out

    def log_and_return_loss_step(self, loss, y_pred, y_true, type):
        """Has to return dictionary with 'loss'.

        Lightning uses loss variable to perform backwards

        metric_dict = {
            "train/loss": ...,
            "train/f1": ...,
            "val/loss"...,
        }
        """

        metric_dict = add_prefix_to_keys(
            get_metrics(
                y_pred=y_pred,
                y_true=y_true,
                num_labels=self.num_labels,
                return_per_instrument=self.log_per_instrument_metrics,
            ),
            f"{type}/",
        )
        self.log_dict(
            metric_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        metric_dict.update({"loss": loss})
        return metric_dict

    def head(self) -> Union[nn.ModuleList, nn.Module] | None:
        """Returns "head" part of the model. That's usually whatever's after the large feature
        extractor.

        Returns:
            Union[nn.ModuleList, nn.Module]: modules which are considered a head
        """
        if self.head_after:
            return get_all_modules_after(self, self.head_after)
        else:
            return None

    def trainable_backbone(self) -> Union[nn.ModuleList, nn.Module] | None:
        """Returns "backbone" part of the model. That's usually the large feature extractor.

        Returns:
            Union[nn.ModuleList, nn.Module]: modules which are considered a backbone
        """
        if self.backbone_after:
            return get_all_modules_after(self, self.backbone_after)
        else:
            return None

    def log_hparams(
        self,
        params: Union[dict[str, Any], Namespace],
        metrics: Optional[dict[str, Any]] = None,
    ):
        self.loggers[0].log_hyperparams(params)
        self.loggers[0].finalize("success")

    def _set_lr(self, lr: float):
        if self.trainer is not None:
            for optim in self.trainer.optimizers:
                for param_group in optim.param_groups:
                    param_group["lr"] = lr
        self.lr = lr
        self.hparams.update({"lr": lr})

    def count_trainable_params(self):
        """Returns number of total, trainable and non trainable parameters."""
        return {
            "total": int(sum(p.numel() for p in self.parameters())),
            "trainable": int(
                sum(p.numel() for p in self.parameters() if p.requires_grad)
            ),
            "non_trainable": int(
                sum(p.numel() for p in self.parameters() if not p.requires_grad)
            ),
        }

    def is_finetuning_phase(self):
        if not self.has_finetuning:
            return False

        return (self.finetune_until_step is not None) and (
            self.global_step < self.finetune_until_step
        )

    def print_params(self):
        """Print module's parameters."""
        for _, module in self.named_modules():
            for param_name, param in module.named_parameters():
                print(param_name, "requires_grad:", param.requires_grad)

    def print_modules(self):
        """Print module's parameters."""
        for module_name, module in self.named_modules():
            module_req_grad = all(
                [x[1].requires_grad for x in module.named_parameters()]
            )
            print(module_name, "requires_grad:", module_req_grad)

    def _set_finetune_until_step(self):
        """We have to caculate what's the step number after which the fine tuning phase is over. We
        also dynamically set the finetune lr nominator, which will ensure that warmup learning rate
        starts at `lr_warmup` and ends with `lr_backbone`. Once the trainer reaches the step
        `finetune_until_step` and learning rate becomes `lr_backbone`, the finetuning phase is
        over.

        lr = ((lr_warmup * numerator) * numerator) ... * numerator))  =  lr_warmup * (numerator)^unfreeze_backbone_at_epoch
                                                    ^ multiplying unfreeze_backbone_at_epoch times
        """
        assert self.unfreeze_at_epoch is not None

        self.finetune_until_step = self.num_of_steps_in_epoch * self.unfreeze_at_epoch

        _a = self.lr_backbone / self.lr_warmup
        _b = self.finetune_until_step - 1
        self.finetune_lr_nominator = np.exp(np.log(_a) / (_b))

        assert np.isclose(
            np.log(self.lr_backbone),
            np.log(self.lr_warmup)
            + (self.finetune_until_step - 1) * np.log(self.finetune_lr_nominator),
        ), "should be: lr = lr_warmup * (numerator)^unfreeze_backbone_at_epoch"

    def _lr_finetuning_step(self, optimizer_idx):
        """Exponential learning rate update.

        Mupltiplicator is the finetune_lr_nominator
        """
        old_lr = self.trainer.optimizers[optimizer_idx].param_groups[0]["lr"]
        new_lr = old_lr * self.finetune_lr_nominator
        self._set_lr(new_lr)
        return

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        """We ignore the lr scheduler in the fine tuning phase and update lr maunally.

        Once the finetuning phase is over we start using the lr scheduler
        """

        if self.is_finetuning_phase():
            self._lr_finetuning_step(optimizer_idx)
            return

        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.lr_warmup:
            self._set_lr(self.lr_warmup)

    def configure_optimizers(self):

        if self.unfreeze_at_epoch is not None:
            scheduler_epochs = self.epochs - self.unfreeze_at_epoch
            total_lr_sch_steps = self.num_of_steps_in_epoch * scheduler_epochs

        else:
            scheduler_epochs = self.epochs
            total_lr_sch_steps = self.num_of_steps_in_epoch * self.epochs

        out = our_configure_optimizers(
            parameters=self.parameters(),
            scheduler_type=self.scheduler_type,
            metric_mode=self.metric_mode,
            plateau_epoch_patience=(self.plateau_epoch_patience // 2) + 1,
            lr_backbone=self.lr_backbone,
            weight_decay=self.weight_decay,
            optimizer_type=self.optimizer_type,
            optimization_metric=self.optimization_metric,
            total_lr_sch_steps=total_lr_sch_steps,
            num_of_steps_in_epoch=self.num_of_steps_in_epoch,
            scheduler_epochs=scheduler_epochs,
            lr_onecycle_max=self.lr_onecycle_max,
        )
        return out
