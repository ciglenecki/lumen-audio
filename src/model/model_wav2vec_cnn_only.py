import math
from typing import Any

import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import MultilabelF1Score
from transformers import Wav2Vec2Config, Wav2Vec2Model

import src.config.config_defaults as config_defaults
from src.model.fluffy import Fluffy, FluffyConfig
from src.model.heads import AttentionHead, DeepHead
from src.model.model_base import ModelBase
from src.model.optimizers import our_configure_optimizers


class Wav2VecCNNWrapper(ModelBase):
    loggers: list[TensorBoardLogger]

    def __init__(
        self,
        time_dim_pooling_mode="mean",
        num_layers=2,
        hidden_size=64,
        loss_function=nn.BCEWithLogitsLoss(),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.fluffy_config is not None:
            self.time_dim_pooling_mode = (
                "attention"
                if self.fluffy_config.classifer_constructor == AttentionHead
                else time_dim_pooling_mode
            )
            self.automatic_optimization = not self.fluffy_config.use_multiple_optimizers

        self.loss_function = loss_function

        self.hamming_distance = torchmetrics.HammingDistance(
            task="multilabel", num_labels=self.num_labels
        )
        self.f1_score = MultilabelF1Score(num_labels=self.num_labels)

        self.config = Wav2Vec2Config(
            pretrained_model_name_or_path=self.pretrained_tag,
            id2label=config_defaults.IDX_TO_INSTRUMENT,
            label2id=config_defaults.IDX_TO_INSTRUMENT,
            num_labels=self.num_labels,
            finetuning_task="audio-classification",
            problem_type="multi_label_classification",
        )

        self.backbone = Wav2Vec2Model.from_pretrained(
            self.pretrained_tag,
            config=self.config,
            ignore_mismatched_sizes=True,
        ).feature_extractor

        output_size = self.config.conv_dim[-1]

        if self.use_fluffy:
            dimensions = [output_size]
            dimensions.extend([hidden_size] * num_layers)
            dimensions.append(1)

            classifer_kwargs = dict(dimensions=dimensions)
            self.classifier = Fluffy(
                head_constructor=self.fluffy_config.classifer_constructor,
                head_kwargs=classifer_kwargs,
            )
        else:
            self.classifier = DeepHead([output_size, self.num_labels])

        self.save_hyperparameters()

    def time_dim_pooling(self, hidden_states):
        """Reduce the temporal features of the backbone last hidden state."""

        # Original shape: [Batch size, Feature Dimension, Time]
        # Result shape: [Batch size, Time, Feature Dimension]
        hidden_states = torch.permute(hidden_states, [0, 2, 1])
        if self.time_dim_pooling_mode == "attention":
            return hidden_states
        elif self.time_dim_pooling_mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif self.time_dim_pooling_mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif self.time_dim_pooling_mode == "max":
            outputs, _ = torch.max(hidden_states, dim=1)
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max', 'attention']"
            )

        return outputs

    def forward(self, audio: torch.Tensor):
        cnn_features = self.backbone.forward(audio)
        hidden_states = self.time_dim_pooling(cnn_features)
        logits_pred = self.classifier(hidden_states)
        return logits_pred

    def _step(self, batch, batch_idx, type: str):
        audio, y, file_indices = batch

        logits_pred = self.forward(audio)
        y_pred_prob = torch.sigmoid(logits_pred)
        y_pred = y_pred_prob >= 0.5
        loss = self.loss_function(logits_pred, y)

        return self.log_and_return_loss_step(
            loss=loss, y_pred=y_pred, y_true=y, type=type
        )

    def training_step(self, batch, batch_idx):
        if self.fluffy_config and self.fluffy_config.use_multiple_optimizers:
            optimizers = self.optimizers()
            schedulers = self.lr_schedulers()
            for opt in optimizers:
                opt.zero_grad()
            outputs = self._step(batch, batch_idx, type="train")
            loss = outputs["loss"]
            loss.backward()

            for optimizer, scheduler in zip(optimizers, schedulers):
                optimizer.step()
                scheduler.step(loss)
            return outputs
        else:
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

    def configure_optimizers(self):
        if self.finetune_head:
            scheduler_epochs = self.epochs - self.finetune_head_epochs
            total_lr_sch_steps = self.num_of_steps_in_epoch * scheduler_epochs

        else:
            scheduler_epochs = self.epochs
            total_lr_sch_steps = self.num_of_steps_in_epoch * self.epochs

        if self.fluffy_config.use_multiple_optimizers and not isinstance(
            self.classifier, Fluffy
        ):
            raise Exception("You cant use multiple optimizers without using Fluffy.")

        if self.fluffy_config.use_multiple_optimizers and isinstance(
            self.classifier, Fluffy
        ):
            list_of_module_params = []
            for head in self.classifier.heads.values():
                list_of_module_params.append(head.parameters())
        else:  # any other normal classifier
            list_of_module_params = [self.classifier.parameters()]

        out = our_configure_optimizers(
            list_of_module_params=list_of_module_params,  # [Modul1, Modul2, Modul3], [Modul]
            scheduler_type=self.scheduler_type,
            metric_mode=self.metric_mode,
            plateau_epoch_patience=(self.plateau_epoch_patience // 2) + 1,
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
