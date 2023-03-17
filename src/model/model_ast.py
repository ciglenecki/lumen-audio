import math
from typing import Any

import torch
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import MultilabelF1Score
from transformers import ASTConfig, ASTForAudioClassification
from transformers.modeling_outputs import SequenceClassifierOutput

import src.config.config_defaults as config_defaults
from src.model.deep_head import DeepHead
from src.model.model_base import ModelBase


class ASTModelWrapper(ModelBase):
    loggers: list[TensorBoardLogger]

    def __init__(
        self,
        model_name: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model_name = model_name

        config = ASTConfig.from_pretrained(
            pretrained_model_name_or_path=model_name,
            id2label=config_defaults.IDX_TO_INSTRUMENT,
            label2id=config_defaults.IDX_TO_INSTRUMENT,
            num_labels=self.num_labels,
            finetuning_task="audio-classification",
            problem_type="multi_label_classification",
        )

        self.hamming_distance = torchmetrics.HammingDistance(
            task="multilabel", num_labels=self.num_labels
        )
        self.f1_score = MultilabelF1Score(num_labels=self.num_labels)

        self.backbone: ASTForAudioClassification = (
            ASTForAudioClassification.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True,
            )
        )

        middle_size = int(
            math.sqrt(config.hidden_size * self.num_labels) + self.num_labels
        )

        self.backbone.classifier = DeepHead(
            [config.hidden_size, middle_size, self.num_labels]
        )

        self.save_hyperparameters()

    def forward(self, audio: torch.Tensor, labels: torch.Tensor):
        out: SequenceClassifierOutput = self.backbone.forward(
            audio,
            output_attentions=True,
            return_dict=True,
            labels=labels,
        )
        return out.loss, out.logits

    def _step(self, batch, batch_idx, type: str):
        audio, y = batch

        loss, logits_pred = self.forward(audio, labels=y)
        y_pred_prob = torch.sigmoid(logits_pred)
        y_pred = y_pred_prob >= 0.5

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
