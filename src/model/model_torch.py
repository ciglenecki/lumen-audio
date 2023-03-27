from typing import Optional

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
        fc: list[int] = config_defaults.DEFAULT_FC,
        pretrained_weights: Optional[str] = config_defaults.DEFAULT_PRETRAINED_WEIGHTS,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_enum = model_enum
        self.pretrained_weights = pretrained_weights
        self.fc = fc

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

    def forward(self, audio: torch.Tensor):
        out = self.backbone.forward(audio)
        return out

    def _step(self, batch, batch_idx, type: str):
        audio, y = batch

        logits_pred = self.forward(audio)
        loss = self.loss_function(logits_pred, y)
        y_pred = torch.sigmoid(logits_pred) > 0.5
        return self.log_and_return_loss_step(
            loss=loss, y_pred=y_pred, y_true=y, type=type
        )

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, type="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, type="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, type="test")
