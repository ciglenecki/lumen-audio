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
from src.train.metrics import get_metrics
from src.utils.utils_functions import add_prefix_to_keys


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
        return self.log_and_return_loss_step(
            loss=loss, y_pred=y_pred, y_true=y, type=type
        )

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
