from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.io import wavfile
from torch.nn.parameter import Parameter
from torchmetrics.classification import MultilabelF1Score
from torchsummary import summary
from torchvision.models import efficientnet_v2_s
from transformers import ASTConfig, ASTForAudioClassification
from transformers.modeling_outputs import SequenceClassifierOutput

import src.config_defaults as config_defaults
from src.utils_train import (
    MetricMode,
    OptimizeMetric,
    OptimizerType,
    SchedulerType,
    UnsupportedOptimizer,
    UnsupportedScheduler,
)


def our_configure_optimizers(
    parameters: Iterator[Parameter],
    scheduler_type: SchedulerType,
    metric_mode: MetricMode,
    patience: int,
    backbone_lr: float,
    weight_decay: float,
    optimizer_type: OptimizerType,
    optimization_metric: OptimizeMetric,
    trainer_estimated_stepping_batches: int,
    num_of_steps_in_epoch: int,
):
    """Set optimizer's learning rate to backbone.

    We do this because we can't explicitly pass the learning rate to scheduler. The scheduler
    infers the learning rate from the optimizer which is why we set it the lr value which should be
    activie once
    """

    if optimizer_type is OptimizerType.ADAMW:
        optimizer = torch.optim.AdamW(
            parameters,
            lr=backbone_lr,
            weight_decay=weight_decay,
        )
    elif optimizer_type is OptimizerType.ADAM:
        optimizer = torch.optim.Adam(
            parameters,
            lr=backbone_lr,
            weight_decay=weight_decay,
        )
    else:
        raise UnsupportedOptimizer(
            f"Optimizer {optimizer_type} is not implemented",
            optimizer_type,
        )
    if scheduler_type is SchedulerType.AUTO_LR:
        """SchedulerType.AUTO_LR sets it's own scheduler.

        Only the optimizer has to be returned
        """
        return optimizer

    lr_scheduler_config = {
        "monitor": optimization_metric.value,  # "val/loss_epoch",
        # How many epochs/steps should pass between calls to `scheduler.step()`.1 corresponds to updating the learning  rate after every epoch/step.
        # If "monitor" references validation metrics, then "frequency" should be set to a multiple of "trainer.check_val_every_n_epoch".
        "frequency": 1,
        # If using the `LearningRateMonitor` callback to monitor the learning rate progress, this keyword can be used to specify a custom logged name
        "name": scheduler_type.value,
    }

    if scheduler_type == SchedulerType.ONECYCLE:
        min_lr = 2.5e-5
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=backbone_lr,  # TOOD:lr,
            final_div_factor=backbone_lr / min_lr,
            total_steps=int(trainer_estimated_stepping_batches),
            verbose=False,
        )
        interval = "step"

    elif scheduler_type == SchedulerType.PLATEAU:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=metric_mode.value,
            factor=config_defaults.DEFAULT_LR_PLATEAU_FACTOR,
            patience=patience,
            verbose=True,
        )
        interval = "epoch"
    elif scheduler_type == SchedulerType.COSINEANNEALING:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=num_of_steps_in_epoch * 3,
            T_mult=1,
        )
        interval = "step"
    else:
        raise UnsupportedScheduler(
            f"Scheduler {scheduler_type} is not implemented",
            scheduler_type,
        )

    lr_scheduler_config.update(
        {
            "scheduler": scheduler,
            "interval": interval,
        }
    )

    return [optimizer], [lr_scheduler_config]
