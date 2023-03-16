from typing import Optional, Union

import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import MultilabelF1Score
from torchvision.models import (
    efficientnet_v2_l,
    efficientnet_v2_m,
    efficientnet_v2_s,
    resnext50_32x4d,
    resnext101_32x8d,
    resnext101_64x4d,
)

import src.config.config_defaults as config_defaults
from src.model.model import SupportedModels
from src.model.model_base import ModelBase
from src.model.optimizers import OptimizerType, SchedulerType, our_configure_optimizers
from src.utils.utils_train import MetricMode, OptimizeMetric

TORCHVISION_CONSTRUCTOR_DICT = {
    SupportedModels.EFFICIENT_NET_V2_S: efficientnet_v2_s,
    SupportedModels.EFFICIENT_NET_V2_M: efficientnet_v2_m,
    SupportedModels.EFFICIENT_NET_V2_L: efficientnet_v2_l,
    SupportedModels.RESNEXT50_32X4D: resnext50_32x4d,
    SupportedModels.RESNEXT101_32X8D: resnext101_32x8d,
    SupportedModels.RESNEXT101_64X4D: resnext101_64x4d,
}


class TorchvisionModel(ModelBase):
    """Implementation of a torchvision model accessed using a string."""

    loggers: list[TensorBoardLogger]

    def __init__(
        self,
        model_enum: str,
        pretrained: bool = config_defaults.DEFAULT_PRETRAINED,
        batch_size: int = config_defaults.DEFAULT_BATCH_SIZE,
        scheduler_type: SchedulerType = SchedulerType.PLATEAU,
        optimizer_type: OptimizerType = config_defaults.DEFAULT_OPTIMIZER,
        num_labels: int = config_defaults.DEFAULT_NUM_LABELS,
        optimization_metric: OptimizeMetric = config_defaults.DEFAULT_OPTIMIZE_METRIC,
        weight_decay: float = config_defaults.DEFAULT_WEIGHT_DECAY,
        metric_mode: MetricMode = config_defaults.DEFAULT_METRIC_MODE,
        early_stopping_epoch: int = config_defaults.DEFAULT_EARLY_STOPPING_NO_IMPROVEMENT_EPOCHS,
        fc: list[int] = config_defaults.DEFAULT_FC,
        pretrained_weights: Optional[str] = config_defaults.DEFAULT_PRETRAINED_WEIGHTS,
        epoch_patience: int = config_defaults.DEFAULT_EARLY_STOPPING_NO_IMPROVEMENT_EPOCHS,
        epochs: Optional[int] = config_defaults.DEFAULT_EPOCHS,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_enum = model_enum
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.scheduler_type = scheduler_type
        self.epochs = epochs
        self.optimizer_type = optimizer_type
        self.num_labels = num_labels
        self.optimization_metric = optimization_metric
        self.weight_decay = weight_decay
        self.metric_mode = metric_mode
        self.early_stopping_epoch = early_stopping_epoch
        self.pretrained_weights = pretrained_weights
        self.epoch_patience = epoch_patience

        self.backbone = TORCHVISION_CONSTRUCTOR_DICT[model_enum](
            weights=pretrained_weights, progress=True
        )

        print("------------------------------------------")
        print("\n")
        print("Backbone before changing the classifier:")
        print(list(self.backbone.children())[-1])
        print("\n")
        print("------------------------------------------")

        last_module_name = [
            i[0]
            for i in self.backbone.named_modules()
            if "." not in i[0] and i[0] != ""
        ][-1]
        last_module = getattr(self.backbone, last_module_name)
        last_dim = (
            last_module[-1].in_features
            if isinstance(last_module, nn.Sequential)
            else last_module.in_features
        )

        new_fc = []
        if isinstance(last_module, nn.Sequential):
            for k in last_module[
                :-1
            ]:  # in case of existing dropouts in final fc module
                new_fc.append(k)

        fc.insert(0, last_dim)
        fc.append(self.num_labels)
        for i, _ in enumerate(fc[:-1]):
            new_fc.append(
                nn.Linear(in_features=fc[i], out_features=fc[i + 1], bias=True)
            )

        setattr(self.backbone, last_module_name, nn.Sequential(*new_fc))

        print("\n")
        print("Backbone after changing the classifier:")
        print(list(self.backbone.children())[-1])
        print("\n")
        print("------------------------------------------")

        self.hamming_distance = torchmetrics.HammingDistance(
            task="multilabel", num_labels=self.num_labels
        )

        self.f1_score = MultilabelF1Score(num_labels=self.num_labels)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

    def head(self) -> Union[nn.ModuleList, nn.Module]:
        return self.backbone.classifier

    def trainable_backbone(self) -> Union[nn.ModuleList, nn.Module]:
        result = []
        result.extend(list(self.backbone.features)[-3:])
        return result

    def forward(self, audio: torch.Tensor):
        out = self.backbone.forward(audio)
        return out

    def _step(self, batch, batch_idx, type: str):
        audio, y = batch

        logits_pred = self.forward(audio)
        loss = self.loss_function(logits_pred, y)
        y_pred = torch.sigmoid(logits_pred) > 0.5
        hamming_distance = self.hamming_distance(y, y_pred)
        f1_score = self.f1_score(y, y_pred)

        data_dict = {
            "loss": loss,  # the 'loss' key needs to be present
            f"{type}/loss": loss,
            f"{type}/hamming_distance": hamming_distance,
            f"{type}/f1_score": f1_score,
        }

        log_dict = data_dict.copy()
        log_dict.pop("loss", None)
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        return data_dict

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, type="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, type="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, type="test")

    def configure_optimizers(self):
        out = our_configure_optimizers(
            parameters=self.parameters(),
            scheduler_type=self.scheduler_type,
            metric_mode=self.metric_mode,
            plateau_patience=(self.epoch_patience // 2) + 1,
            backbone_lr=self.backbone_lr,
            weight_decay=self.weight_decay,
            optimizer_type=self.optimizer_type,
            optimization_metric=self.optimization_metric,
            trainer_estimated_stepping_batches=int(
                self.trainer.estimated_stepping_batches
            ),
            num_of_steps_in_epoch=self.num_of_steps_in_epoch,
            epochs=self.epochs,
        )
        return out

    def _lr_finetuning_step(self, optimizer_idx):
        """Exponential learning rate update.

        Mupltiplicator is the finetune_lr_nominator
        """
        old_lr = self.trainer.optimizers[optimizer_idx].param_groups[0]["lr"]
        new_lr = old_lr * self.finetune_lr_nominator
        self._set_lr(new_lr)
        return
