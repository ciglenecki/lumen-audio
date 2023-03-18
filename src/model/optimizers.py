from typing import Iterator

import torch
from torch.nn.parameter import Parameter

import src.config.config_defaults as config_defaults
from src.utils.utils_exceptions import UnsupportedOptimizer, UnsupportedScheduler
from src.utils.utils_functions import EnumStr
from src.utils.utils_train import MetricMode, OptimizeMetric


class SchedulerType(EnumStr):
    ONECYCLE = "onecycle"
    PLATEAU = "plateau"
    AUTO_LR = "auto_lr"
    COSINEANNEALING = "cosine_annealing"


class OptimizerType(EnumStr):
    ADAM = "adam"
    ADAMW = "adamw"


def our_configure_optimizers(
    parameters: Iterator[Parameter],
    scheduler_type: SchedulerType,
    metric_mode: MetricMode,
    plateau_patience: int,
    backbone_lr: float,
    weight_decay: float,
    optimizer_type: OptimizerType,
    optimization_metric: OptimizeMetric,
    trainer_estimated_stepping_batches: int,
    num_of_steps_in_epoch: int,
    epochs: int,
    onecycle_max_lr=0.03,
):
    """Set optimizer's learning rate to backbone. Why?

    - lr scheduler starts modifying lr after finetuning, it's starting lr is `backbone_lr`
    - we can't explicitly pass the intial lr to scheduler
    - the scheduler infers the initial lr from the optimizer
    - that's why we set optimizers lr to `backbone_lr`
    - we later change optimizer's lr to `warmup_lr`
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
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=onecycle_max_lr,  # TOOD:lr,
            final_div_factor=onecycle_max_lr / backbone_lr,
            total_steps=int(trainer_estimated_stepping_batches),
            verbose=False,
        )
        interval = "step"

    elif scheduler_type == SchedulerType.PLATEAU:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=metric_mode.value,
            factor=config_defaults.DEFAULT_LR_PLATEAU_FACTOR,
            patience=plateau_patience,
            verbose=True,
        )
        interval = "epoch"
    elif scheduler_type == SchedulerType.COSINEANNEALING:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=num_of_steps_in_epoch * epochs,
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
