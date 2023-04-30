from typing import Iterator

import torch
from torch.nn.parameter import Parameter

import src.config.config_defaults as config_defaults
from src.enums.enums import (
    MetricMode,
    OptimizeMetric,
    SupportedOptimizer,
    SupportedScheduler,
)
from src.utils.utils_exceptions import UnsupportedOptimizer, UnsupportedScheduler


def our_configure_optimizers(
    parameters: Iterator[Parameter],
    scheduler_type: SupportedScheduler,
    metric_mode: MetricMode,
    plateau_epoch_patience: int,
    lr_backbone: float,
    weight_decay: float,
    optimizer_type: SupportedOptimizer,
    optimization_metric: OptimizeMetric,
    total_lr_sch_steps: int,
    num_of_steps_in_epoch: int,
    scheduler_epochs: int,
    lr_onecycle_max=0.03,
) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
    """

    WARNING: returns N optimizers for each parameter group in `list_of_module_params`. You should almost always send a single module.parameters() wrapped with a list []
    Set optimizer's learning rate to backbone. Why?

    - lr scheduler starts modifying lr after finetuning, it's starting lr is `lr_backbone`
    - we can't explicitly pass the intial lr to scheduler
    - the scheduler infers the initial lr from the optimizer
    - that's why we set optimizers lr to `lr_backbone`
    - we later change optimizer's lr to `lr_warmup`
    """

    if optimizer_type is SupportedOptimizer.ADAMW:
        optimizer = torch.optim.AdamW(
            parameters,
            lr=lr_backbone,
            weight_decay=weight_decay,
        )
    elif optimizer_type is SupportedOptimizer.ADAM:
        optimizer = torch.optim.Adam(
            parameters,
            lr=lr_backbone,
            weight_decay=weight_decay,
        )
    else:
        raise UnsupportedOptimizer(
            f"Optimizer {optimizer_type} is not implemented",
            optimizer_type,
        )

    if scheduler_type is SupportedScheduler.AUTO_LR:
        """SupportedScheduler.AUTO_LR sets it's own scheduler.

        Only the optimizer has to be returned
        """
        return optimizer

    lr_scheduler_config = {
        "monitor": optimization_metric.value,  # "val/loss_epoch",
        # How many scheduler_epochs/steps should pass between calls to `scheduler.step()`.1 corresponds to updating the learning  rate after every epoch/step.
        # If "monitor" references validation metrics, then "frequency" should be set to a multiple of "trainer.check_val_every_n_epoch".
        "frequency": 1,
        # If using the `LearningRateMonitor` callback to monitor the learning rate progress, this keyword can be used to specify a custom logged name
        "name": scheduler_type.value,
    }

    if scheduler_type == SupportedScheduler.ONECYCLE:
        # min_lr = initial_lr/final_div_factor
        # initial_lr = max_lr/div_factor

        start_lr = lr_backbone
        end_lr = lr_backbone / 2

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr_onecycle_max,  # TOOD:lr,
            div_factor=lr_onecycle_max / start_lr,  # initial_lr = max_lr/div_factor
            # final_div_factor=lr_onecycle_max / end_lr,
            total_steps=int(total_lr_sch_steps)
            + 1,  # add one more step because the condition is > instead of >=
            three_phase=False,
            verbose=False,
        )
        interval = "step"

    elif scheduler_type == SupportedScheduler.PLATEAU:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=metric_mode.value,
            factor=config_defaults.DEFAULT_LR_PLATEAU_FACTOR,
            patience=plateau_epoch_patience,
            verbose=True,
        )
        interval = "epoch"
    elif scheduler_type == SupportedScheduler.COSINEANNEALING:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=num_of_steps_in_epoch * scheduler_epochs,
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
