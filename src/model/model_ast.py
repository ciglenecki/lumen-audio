import math
from typing import Any, Optional

import torch
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import MultilabelF1Score
from transformers import ASTConfig, ASTForAudioClassification
from transformers.modeling_outputs import SequenceClassifierOutput

import src.config.config_defaults as config_defaults
from src.model.deep_head import DeepHead
from src.model.model_base import ModelBase
from src.model.optimizers import our_configure_optimizers
from src.utils.utils_train import (
    MetricMode,
    OptimizeMetric,
    OptimizerType,
    SchedulerType,
)


class ASTModelWrapper(ModelBase):
    loggers: list[TensorBoardLogger]

    def __init__(
        self,
        pretrained: bool,
        batch_size: int,
        scheduler_type: SchedulerType,
        epochs: Optional[int],
        optimizer_type: OptimizerType,
        model_name: str,
        num_labels: int,
        optimization_metric: OptimizeMetric,
        weight_decay: float,
        metric_mode: MetricMode,
        epoch_patience: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.pretrained = pretrained
        self.batch_size = batch_size
        self.scheduler_type = scheduler_type
        self.epochs = epochs
        self.optimizer_type = optimizer_type
        self.num_labels = num_labels
        self.model_name = model_name
        self.optimization_metric = optimization_metric
        self.weight_decay = weight_decay
        self.metric_mode = metric_mode
        self.epoch_patience = epoch_patience

        config = ASTConfig(
            pretrained_model_name_or_path=model_name,
            id2label=config_defaults.IDX_TO_INSTRUMENT,
            label2id=config_defaults.IDX_TO_INSTRUMENT,
            num_labels=num_labels,
            finetuning_task="audio-classification",
            problem_type="multi_label_classification",
        )

        self.hamming_distance = torchmetrics.HammingDistance(
            task="multilabel", num_labels=num_labels
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
