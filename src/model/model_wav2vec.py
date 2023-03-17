import math
from typing import Any

import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import MultilabelF1Score
from transformers import Wav2Vec2Config, Wav2Vec2Model

import src.config.config_defaults as config_defaults
from src.model.deep_head import DeepHead
from src.model.model_base import ModelBase


class Wav2VecWrapper(ModelBase):
    loggers: list[TensorBoardLogger]

    def __init__(
        self,
        model_name: str,
        pooling_mode="mean",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model_name = model_name
        self.pooling_mode = pooling_mode

        self.loss_function = nn.BCEWithLogitsLoss()

        self.hamming_distance = torchmetrics.HammingDistance(
            task="multilabel", num_labels=self.num_labels
        )
        self.f1_score = MultilabelF1Score(num_labels=self.num_labels)

        config = Wav2Vec2Config(
            pretrained_model_name_or_path=model_name,
            id2label=config_defaults.IDX_TO_INSTRUMENT,
            label2id=config_defaults.IDX_TO_INSTRUMENT,
            num_labels=self.num_labels,
        )
        self.backbone: Wav2Vec2Model = Wav2Vec2Model.from_pretrained(
            model_name, config=config, ignore_mismatched_sizes=True
        )
        middle_size = int(
            math.sqrt(config.hidden_size * self.num_labels) + self.num_labels
        )
        self.classifier = DeepHead([config.hidden_size, middle_size, self.num_labels])

        self.save_hyperparameters()

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']"
            )

        return outputs

    def forward(self, audio: torch.Tensor):
        hidden_states = self.backbone.forward(
            input_values=audio,
            output_attentions=False,
            return_dict=True,
        ).last_hidden_state
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)

        logits_pred = self.classifier(hidden_states)
        return logits_pred

    def _step(self, batch, batch_idx, type: str):
        audio, y = batch

        logits_pred = self.forward(audio, labels=y)
        y_pred_prob = torch.sigmoid(logits_pred)
        y_pred = y_pred_prob >= 0.5

        loss = self.loss_function(logits_pred, y)
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

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        # TODO:
        pass
