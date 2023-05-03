from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import LRSchedulerPLType
from torch_scatter import scatter_max

import src.config.config_defaults as config_defaults
from src.config.config_defaults import ConfigDefault
from src.enums.enums import MetricMode, OptimizeMetric, SupportedModels
from src.model.fluffy import Fluffy, FluffyConfig
from src.model.heads import DeepHead, HeadTypes
from src.model.optimizers import (
    SupportedOptimizer,
    SupportedScheduler,
    our_configure_optimizers,
)
from src.model.loss_functions import InstrumentFamilyLoss

from src.train.metrics import get_metrics
from src.utils.utils_model import (
    count_module_params,
    get_all_modules_after,
    proper_weight_decay,
)


@dataclass
class ForwardInput:
    feature: torch.Tensor
    y_true: torch.Tensor | None


@dataclass
class ForwardOut:
    logits: torch.Tensor
    loss: torch.Tensor | None


class StepResult:
    def __init__(self, step_dict: dict):
        self.loss: torch.Tensor | None = step_dict.get("loss", None)
        self.losses: torch.Tensor | None = step_dict.get("losses", None)
        self.y_pred: torch.Tensor | None = step_dict.get("y_pred", None)
        self.y_pred_prob: torch.Tensor | None = step_dict.get("y_pred_prob", None)
        self.y_true: torch.Tensor | None = step_dict.get("y_true", None)
        self.file_indices: torch.Tensor | None = step_dict.get("file_indices", None)
        self.item_indices: torch.Tensor | None = step_dict.get("item_indices", None)
        self.item_indices_unique: torch.Tensor | None = step_dict.get(
            "item_indices_unique", None
        )
        self.y_true_file: torch.Tensor | None = step_dict.get("y_true_file", None)
        self.y_pred_file: torch.Tensor | None = step_dict.get("y_pred_file", None)
        self.y_pred_prob_file: torch.Tensor | None = step_dict.get(
            "y_pred_prob_file", None
        )
        self.losses_file: torch.Tensor | None = step_dict.get("losses_file", None)


class ModelBase(pl.LightningModule, ABC):
    loggers: list[TensorBoardLogger]
    optimizers_list: list[LightningOptimizer]
    schedulers_list: list[LRSchedulerPLType]
    finetuning_step: int
    instrument_key = "instruments"
    instrument_key_len = len(instrument_key)

    def __init__(
        self,
        batch_size: int,
        epochs: Optional[int],
        finetune_head: bool,
        finetune_head_epochs: Optional[int],
        freeze_train_bn: bool,
        head_after: str | None,
        backbone_after: str | None,
        loss_function: torch.nn.modules.loss._Loss,
        lr: float,
        lr_onecycle_max: int | None,
        lr_warmup: Optional[float],
        metric_mode: MetricMode,
        model_enum: SupportedModels,
        num_labels: int,
        optimization_metric: OptimizeMetric,
        optimizer_type: SupportedOptimizer,
        early_stopping_metric_patience: int,
        pretrained: bool,
        scheduler_type: SupportedScheduler,
        use_fluffy: bool,
        weight_decay: float,
        log_per_instrument_metrics,
        pretrained_tag: str,       
        add_instrument_loss:float | None = None,
        fluffy_config: FluffyConfig | None = None,
        config: None | ConfigDefault = None,
        head_constructor: Callable[[Any], HeadTypes] = DeepHead,
        head_hidden_dim: list[int] = [],
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            batch_size:

            epochs: number of training epochs. Used only if the LR scheduler depends on epochs size (onecycle, cosine...). Reduce on plateau LR scheduler doesn't get affected by this argument.

            freeze_train_bn: Whether or not to train the batch norm during the frozen stage of the training.

            head_after: Name of the submodule after which the all submodules are considered as head, e.g. classifier.dense

            head_constructor: Constructor of the head, usually one from heads.py

            head_hidden_dim: List of integers which specifies the hidden layers of head (classifer). For each integer one extra hidden layer will be created in the middle of the head (classifier). The integer value specifies number of  hidden dimensions.

            backbone_after: Name of the submodule after which the all submodules are considered as backbone, e.g. layer.11.dense"

            lr: backbone lr, the one which will be used after the finetuning is over

            lr_onecycle_max: used only if onecycle scheduler is used

            lr_warmup: starting lr used for head finetuning, if provided, self.lr gets intialized to this value

            metric_mode: MetricMode.MAX or MetricMode.MIN

            num_labels: number of labels (instruments) we want to classify

            optimization_metric: metric which is responsible for model checkpointing (not done here!) and reducing lr via the reduce on plateau lr scheduler

            optimizer_type: SupportedOptimizer.ADAM, ADAMW

            early_stopping_metric_patience: how many epochs should pass, without metric getting better, before reduce on plateau scheduler lowers the lr

            pretrained: use pretrained model

            pretrained_tag: weights which will be loaded

            scheduler_type: SupportedScheduler.ONECYCLE, SupportedScheduler.PLATEAU...

            use_fluffy: bool, use or don't use fluffy

            finetune_head: Performs head only finetuning for finetune_head_epochs epochs with starting lr of lr_warmup which eventually becomes lr.

            finetune_head_epochs: at which epoch should the model be unfrozen? warning: unfreezing is not done here. It's done by the finetuning callback.

            weight_decay
        """

        super().__init__(*args, **kwargs)

        self.backbone_after = backbone_after
        self.batch_size = batch_size
        self.epochs = epochs
        self.fluffy_config = fluffy_config
        self.freeze_train_bn = freeze_train_bn
        self.head_after = head_after
        self.head_constructor = head_constructor
        self.head_hidden_dim = head_hidden_dim
        self.log_per_instrument_metrics = log_per_instrument_metrics
        self.loss_function = loss_function
        self.lr_backbone = lr  # lr once backbone gets unfrozen
        self.lr_onecycle_max = lr_onecycle_max
        self.lr_warmup = lr_warmup  # starting warmup lr
        self.metric_mode = metric_mode
        self.model_enum = model_enum
        self.num_labels = num_labels
        self.optimization_metric = optimization_metric
        self.optimizer_type = optimizer_type
        self.early_stopping_metric_patience = early_stopping_metric_patience
        self.pretrained = pretrained
        self.pretrained_tag = pretrained_tag
        self.scheduler_type = scheduler_type
        self.use_fluffy = use_fluffy
        self.finetune_head = finetune_head
        self.finetune_head_epochs = finetune_head_epochs
        self.weight_decay = weight_decay
        self.config = config

        if self.finetune_head:
            # Initially set to learning rate to warmup. later ot will change to 'normal' lr
            self.lr = self.lr_warmup
        else:
            self.lr = lr
        self.add_instrument_loss = add_instrument_loss
        if add_instrument_loss is not None:
            self.instrument_family_loss = InstrumentFamilyLoss()
        
        # save in case indices change with config changes
        self.backup_instruments = config_defaults.INSTRUMENT_TO_IDX
        self.save_hyperparameters()

    def create_head(self, head_input_size: int) -> torch.nn.Module:
        dimensions = [head_input_size]
        dimensions.extend(self.head_hidden_dim)

        if self.use_fluffy:
            num_of_single_class = 1
            dimensions.append(num_of_single_class)
            classifer_kwargs = dict(dimensions=dimensions)
            classifier = Fluffy(
                head_constructor=self.head_constructor,
                head_kwargs=classifer_kwargs,
            )
        else:
            dimensions.append(self.num_labels)
            classifer_kwargs = dict(dimensions=dimensions)
            classifier = self.head_constructor(**classifer_kwargs)
        return classifier

    def setup(self, stage: str) -> None:
        """Freezes (turn off require_grads) every submodule except trainable backbone and head."""
        out = super().setup(stage)
        self.num_of_steps_in_epoch = int(
            self.trainer.estimated_stepping_batches / self.trainer.max_epochs
        )

        if self.head() is not None and self.trainable_backbone() is not None:
            BaseFinetuning.freeze(self, train_bn=self.freeze_train_bn)
            BaseFinetuning.make_trainable(self.trainable_backbone())
            BaseFinetuning.make_trainable(self.head())

        if self.finetune_head:
            self._set_finetune_until_step()
        self.finetuning_step = 0
        return out

    @abstractmethod
    def forward_wrapper(self, forward_input: ForwardInput) -> ForwardOut:
        pass

    def _step(
        self,
        batch,
        batch_idx,
        type: str,
        log_metric_dict=True,
        only_return_loss=True,
        return_as_object=False,
    ) -> dict[str, float | torch.Tensor | None] | StepResult:
        features, y_true, file_indices, item_indices = batch

        is_pred = type == "pred"

        batch_y = None
        y_true = y_true if not is_pred else None
        loss = None

        sub_batches = torch.split(features, self.batch_size, dim=0)
        y_pred_prob = torch.zeros((len(features), self.num_labels), device=self.device)
        y_pred = torch.zeros((len(features), self.num_labels), device=self.device)
        losses = torch.zeros(len(features), device=self.device) if not is_pred else None

        num_samples = 0
        for batch_feature in sub_batches:
            batch_size = len(batch_feature)
            start = num_samples
            end = num_samples + batch_size

            if not is_pred:
                batch_y = y_true[start:end] * self.instrument_family_loss(batch_y_pred, batch_y))
            forward_out = self.forward_wrapper(
                ForwardInput(feature=batch_feature, y_true=batch_y)
            )

            batch_logits_pred, individual_losses = (
                forward_out.logits,
                forward_out.loss,
            )

            if self.instrument_family_loss:
                individual_losses += (self.add_instrument_loss*self.instrument_family_loss(batch_logits_pred, batch_y))
            if individual_losses is not None:
                if len(individual_losses.shape) != 0:
                    batch_loss = individual_losses.view(batch_size, -1).mean(dim=1)
                else:
                    batch_loss = individual_losses
                losses[start:end] = batch_loss

            batch_y_pred_prob = torch.sigmoid(batch_logits_pred)
            batch_y_pred = (batch_y_pred_prob >= 0.5).float()

            y_pred_prob[start:end] = batch_y_pred_prob
            y_pred[start:end] = batch_y_pred

            num_samples += batch_size

        if losses is not None:
            loss = losses.mean()

        return_dict = dict(loss=loss)
        if log_metric_dict:
            metric_dict = self.get_metric_dict(
                loss=loss, y_pred=y_pred, y_true=y_true, type=type
            )
            self.log_dict(
                metric_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True
            )
            return_dict.update(metric_dict)

        if not only_return_loss:
            y_true_file, _ = scatter_max(y_true, file_indices, dim=0)
            y_pred_file, _ = scatter_max(y_pred, file_indices, dim=0)
            y_pred_prob_file, _ = scatter_max(y_pred_prob, file_indices, dim=0)
            losses_file, _ = scatter_max(losses, file_indices, dim=0)
            item_indices_unique = torch.unique_consecutive(item_indices)
            return_dict.update(
                dict(
                    losses=losses,
                    y_pred=y_pred,
                    y_pred_prob=y_pred_prob,
                    y_true=y_true,
                    file_indices=file_indices,
                    item_indices_unique=item_indices_unique,
                    item_indices=item_indices,
                    y_true_file=y_true_file,
                    y_pred_file=y_pred_file,
                    y_pred_prob_file=y_pred_prob_file,
                    losses_file=losses_file,
                )
            )
        if return_as_object:
            return StepResult(return_dict)
        return return_dict

    def training_step(self, batch, batch_idx):
        return self._step(
            batch, batch_idx, type="train", log_metric_dict=True, only_return_loss=True
        )

    def validation_step(self, batch, batch_idx):
        return self._step(
            batch, batch_idx, type="val", log_metric_dict=True, only_return_loss=True
        )

    def test_step(self, batch, batch_idx):
        return self._step(
            batch, batch_idx, type="test", log_metric_dict=True, only_return_loss=False
        )

    def predict_step(self, batch, batch_idx: int):
        return self._step(
            batch, batch_idx, type="pred", log_metric_dict=False, only_return_loss=False
        )

    def get_metric_dict(self, loss, y_pred, y_true, type):
        """Has to return dictionary with 'loss'.

        Lightning uses loss variable to perform backwards

        metric_dict = {
            "train/loss": ...,
            "train/f1": ...,
            "val/loss"...,
        }
        """

        metric_dict = get_metrics(
            y_pred=y_pred,
            y_true=y_true,
            num_labels=self.num_labels,
            return_per_instrument=self.log_per_instrument_metrics,
        )

        # add "loss" metric which will be converted to "train/loss", "val/loss"...
        metric_dict.update({"loss": loss})

        # add prefix "trian" or "test" but skip adding "train" / "test" to per instrument metrics to avoid clutter in tensorboard
        metric_dict = {
            f"{k[:ModelBase.instrument_key_len]}/{type}{k[ModelBase.instrument_key_len:]}"
            if k.startswith(ModelBase.instrument_key)
            else f"{type}/{k}": v
            for k, v in metric_dict.items()
        }

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
        for logger in self.loggers:
            logger.log_hyperparams(params)
            logger.finalize("success")

    def _set_lr(self, lr: float):
        if self.trainer is not None:
            for optim in self.trainer.optimizers:
                for param_group in optim.param_groups:
                    param_group["lr"] = lr
        self.lr = lr
        self.hparams.update({"lr": lr})

    def count_trainable_params(self):
        """Returns number of total, trainable and non trainable parameters."""
        return count_module_params(self)

    def is_finetuning_phase(self):
        if not self.finetune_head:
            return False

        return (self.finetune_until_step is not None) and (
            self.finetuning_step < self.finetune_until_step
        )

    def _set_finetune_until_step(self):
        """We have to caculate what's the step number after which the fine tuning phase is over. We
        also dynamically set the finetune lr nominator, which will ensure that warmup learning rate
        starts at `lr_warmup` and ends with `lr_backbone`. Once the trainer reaches the step
        `finetune_until_step` and learning rate becomes `lr_backbone`, the finetuning phase is
        over.

        lr = ((lr_warmup * numerator) * numerator) ... * numerator))  =  lr_warmup * (numerator)^finetune_head_epochs
                                                    ^ multiplying finetune_head_epochs times
        """

        assert (
            isinstance(self.finetune_head_epochs, int) and self.finetune_head_epochs > 0
        )

        self.finetune_until_step = (
            self.num_of_steps_in_epoch * self.finetune_head_epochs
        )

        if self.finetune_until_step == 1:
            self.finetune_lr_nominator = 1
            return

        _a = self.lr_backbone / self.lr_warmup
        _b = self.finetune_until_step - 1
        self.finetune_lr_nominator = np.exp(np.log(_a) / (_b))

        assert np.isclose(
            np.log(self.lr_backbone),
            np.log(self.lr_warmup)
            + (self.finetune_until_step - 1) * np.log(self.finetune_lr_nominator),
        ), "should be: lr = lr_warmup * (numerator)^finetune_head_epochs"

    def _lr_finetuning_step(self, optimizer_idx):
        """Exponential learning rate update.

        Mupltiplicator is the finetune_lr_nominator
        """
        self.finetuning_step += 1
        old_lr = self.trainer.optimizers[optimizer_idx].param_groups[0]["lr"]
        new_lr = old_lr * self.finetune_lr_nominator
        self._set_lr(new_lr)
        return

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        """We ignore the lr scheduler in the fine tuning phase and update lr maunally.

        Once the finetuning phase is over we start using the lr scheduler
        """

        if self.is_finetuning_phase():
            self._lr_finetuning_step(optimizer_idx=optimizer_idx)
            return

        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.lr_warmup:
            self._set_lr(self.lr_warmup)
        optimizers = self.optimizers()
        self.optimizers_list = (
            optimizers if isinstance(optimizers, list) else [optimizers]
        )
        schedulers = self.lr_schedulers()
        self.schedulers_list = (
            schedulers if isinstance(schedulers, list) is list else [schedulers]
        )

    def configure_optimizers(self):
        if self.finetune_head:
            scheduler_epochs = self.epochs - self.finetune_head_epochs
            total_lr_sch_steps = self.num_of_steps_in_epoch * scheduler_epochs

        else:
            scheduler_epochs = self.epochs
            total_lr_sch_steps = self.num_of_steps_in_epoch * self.epochs

        module_params = proper_weight_decay(self, self.weight_decay)

        out = our_configure_optimizers(
            parameters=module_params,
            scheduler_type=self.scheduler_type,
            metric_mode=self.metric_mode,
            plateau_epoch_patience=(self.early_stopping_metric_patience // 2) + 1,
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
