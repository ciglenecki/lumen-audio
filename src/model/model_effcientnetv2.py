from typing import Union

import torch
import torch.nn as nn
import torchmetrics
from torchmetrics.classification import MultilabelF1Score
from torchvision.models import efficientnet_v2_s

from src.model.model_base import ModelBase


class EfficientNetV2SmallModel(ModelBase):
    """Implementation of EfficientNet V2 small model (384 x 384)

    S    - (384 x 384)
    M, L - (480 x 480)
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.backbone = efficientnet_v2_s(weights="IMAGENET1K_V1", progress=True)

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(
                in_features=self.backbone.classifier[-1].in_features,
                out_features=self.num_labels,
                bias=True,
            ),
        )
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

    def _lr_finetuning_step(self, optimizer_idx):
        """Exponential learning rate update.

        Mupltiplicator is the finetune_lr_nominator
        """
        old_lr = self.trainer.optimizers[optimizer_idx].param_groups[0]["lr"]
        new_lr = old_lr * self.finetune_lr_nominator
        self._set_lr(new_lr)
        return
