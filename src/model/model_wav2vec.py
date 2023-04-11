import math
from typing import Any

import torch
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import MultilabelF1Score
from transformers import Wav2Vec2Config, Wav2Vec2Model

import src.config.defaults as defaults
from src.model.heads import DeepHead
from src.model.model_base import ModelBase


class Wav2VecWrapper(ModelBase):
    loggers: list[TensorBoardLogger]

    def __init__(
        self,
        time_dim_pooling_mode="mean",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.time_dim_pooling_mode = time_dim_pooling_mode

        self.hamming_distance = torchmetrics.HammingDistance(
            task="multilabel", num_labels=self.num_labels
        )
        self.f1_score = MultilabelF1Score(num_labels=self.num_labels)

        config_wav2vec = Wav2Vec2Config(
            pretrained_model_name_or_path=self.pretrained_tag,
            id2label=defaults.IDX_TO_INSTRUMENT,
            label2id=defaults.IDX_TO_INSTRUMENT,
            num_labels=self.num_labels,
        )
        self.backbone: Wav2Vec2Model = Wav2Vec2Model.from_pretrained(
            self.pretrained_tag,
            config=config_wav2vec,
            ignore_mismatched_sizes=True,
        )
        middle_size = int(
            math.sqrt(config_wav2vec.hidden_size * self.num_labels) + self.num_labels
        )
        self.classifier = DeepHead(
            [config_wav2vec.hidden_size, middle_size, self.num_labels]
        )

        self.save_hyperparameters()

    def time_dim_pooling(self, hidden_states, mode="mean"):
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
        hidden_states = self.time_dim_pooling(
            hidden_states, mode=self.time_dim_pooling_mode
        )

        logits_pred = self.classifier(hidden_states)
        return logits_pred

    def _step(self, batch, batch_idx, type: str, optimizer_idx=0):
        audio, y, file_indices = batch

        logits_pred = self.forward(audio, labels=y)
        y_pred_prob = torch.sigmoid(logits_pred)
        y_pred = y_pred_prob >= 0.5
        loss = self.loss_function(logits_pred, y)

        return self.log_and_return_loss_step(
            loss=loss, y_pred=y_pred, y_true=y, type=type
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        return self._step(batch, batch_idx, type="train", optimizer_idx=optimizer_idx)

    def validation_step(self, batch, batch_idx, optimizer_idx):
        return self._step(batch, batch_idx, type="val", optimizer_idx=optimizer_idx)

    def test_step(self, batch, batch_idx, optimizer_idx):
        return self._step(batch, batch_idx, type="test", optimizer_idx=optimizer_idx)

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        # TODO:
        pass
