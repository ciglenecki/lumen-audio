import math
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import MultilabelF1Score
from torchsummary import summary
from torchvision.models import efficientnet_v2_s
from transformers import ASTConfig, ASTForAudioClassification
from transformers.modeling_outputs import SequenceClassifierOutput

import src.config_defaults as config_defaults
from src.optimizers import our_configure_optimizers
from src.utils_train import (
    MetricMode,
    OptimizeMetric,
    OptimizerType,
    SchedulerType,
    SupportedModels,
    UnsupportedModel,
    UnsupportedOptimizer,
    UnsupportedScheduler,
)


class DeepHead(nn.Module):
    def __init__(self, dimensions: list[int], dropout_p=0.3) -> None:
        """List of input and output features which will create N - 1 fully connected layers.

        Args:
            dimensions: list[int]
        """
        assert len(dimensions) >= 2, "Dimensions have to contain at least two ints"

        super().__init__()
        self.dimensions = dimensions
        layers = []
        if len(self.dimensions) == 2:
            in_features, out_features = self.dimensions[0], self.dimensions[1]
            layers.append(nn.LayerNorm(in_features))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(in_features, out_features))
            return
        else:
            for i in range(len(self.dimensions) - 1):
                in_features, out_features = self.dimensions[i], self.dimensions[i + 1]
                layers.append(nn.LayerNorm(in_features))
                layers.append(nn.ReLU())
                layers.append(nn.Linear(in_features, out_features))
                if i != len(self.dimensions) - 1:
                    # don't add dropout to the classifier itself!
                    layers.append(nn.Dropout(p=dropout_p))

        self.deep_head = nn.Sequential(*layers)

    def forward(self, x):
        return self.deep_head(x)


# for in_features, out_features in zip(dimensions, dimensions[1:]):

# self.deep_head
# modules = [self.dimensions[0]]


class OurLightningModule(pl.LightningModule, ABC):
    """Model hooks:

    def fit(self):
        if global_rank == 0:
            # prepare data is called on GLOBAL_ZERO only
            prepare_data()

        configure_callbacks()

        with parallel(devices):
            # devices can be GPUs, TPUs, ...
            train_on_device(model)


    def train_on_device(model):
        # called PER DEVICE
        setup("fit")
        configure_optimizers()
        on_fit_start()

        # the sanity check runs here

        on_train_start()
        for epoch in epochs:
            fit_loop()
        on_train_end()

        on_fit_end()
        teardown("fit")


    def fit_loop():
        on_train_epoch_start()

        for batch in train_dataloader():
            on_train_batch_start()

            on_before_batch_transfer()
            transfer_batch_to_device()
            on_after_batch_transfer()

            training_step()

            on_before_zero_grad()
            optimizer_zero_grad()

            on_before_backward()
            backward()
            on_after_backward()

            on_before_optimizer_step()
            configure_gradient_clipping()
            optimizer_step()

            on_train_batch_end()

            if should_check_val:
                val_loop()
        # end training epoch
        training_epoch_end()

        on_train_epoch_end()


    def val_loop():
        on_validation_model_eval()  # calls `model.eval()`
        torch.set_grad_enabled(False)

        on_validation_start()
        on_validation_epoch_start()

        val_outs = []
        for batch_idx, batch in enumerate(val_dataloader()):
            on_validation_batch_start(batch, batch_idx)

            batch = on_before_batch_transfer(batch)
            batch = transfer_batch_to_device(batch)
            batch = on_after_batch_transfer(batch)

            out = validation_step(batch, batch_idx)

            on_validation_batch_end(batch, batch_idx)
            val_outs.append(out)

        validation_epoch_end(val_outs)

        on_validation_epoch_end()
        on_validation_end()

        # set up for train
        on_validation_model_train()  # calls `model.train()`
        torch.set_grad_enabled(True)
    """

    def __init__(
        self,
        lr: float,
        warmup_lr: Optional[float],
        unfreeze_at_epoch: Optional[int],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.backbone_lr = lr  # lr once backbone gets unfrozen
        self.warmup_lr = warmup_lr  # starting warmup lr

        if self.warmup_lr:
            self.lr = self.warmup_lr
        else:
            self.lr = lr

        assert int(bool(self.unfreeze_at_epoch)) + int(bool(self.warmup_lr)) in [
            0,
            2,
        ], "Both should exist or both shouldn't exist!"

        # save in case indices change with config changes
        self.backup_instruments = config_defaults.INSTRUMENT_TO_IDX

    @abstractmethod
    def head(self) -> Union[nn.ModuleList, nn.Module]:
        """Returns "head" part of the model. That's usually whatever's after the large feature
        extractor.

        Returns:
            Union[nn.ModuleList, nn.Module]: modules which are considered a head
        """
        return

    @abstractmethod
    def trainable_backbone(self) -> Union[nn.ModuleList, nn.Module]:
        """Returns "backbone" part of the model. That's usually the large feature extractor.

        Returns:
            Union[nn.ModuleList, nn.Module]: modules which are considered a backbone
        """
        return

    def _set_lr(self, lr: float):
        if self.trainer is not None:
            for optim in self.trainer.optimizers:
                for param_group in optim.param_groups:
                    param_group["lr"] = lr
        self.lr = lr
        self.hparams.update({"lr": lr})

    def count_trainable_params(self):
        """Returns number of total, trainable and non trainable parameters."""
        return {
            "total": int(sum(p.numel() for p in self.parameters())),
            "trainable": int(
                sum(p.numel() for p in self.parameters() if p.requires_grad)
            ),
            "non_trainable": int(
                sum(p.numel() for p in self.parameters() if not p.requires_grad)
            ),
        }

    def is_finetuning_phase(self):
        return (self.finetune_until_step is not None) and (
            self.global_step <= self.finetune_until_step
        )

    def print_params(self):
        """Print module's parameters."""
        for _, module in self.named_modules():
            for param, _ in module.named_parameters():
                print(param)

    def _set_finetune_until_step(self):
        """We have to caculate what's the step number after which the fine tuning phase is over. We
        also dynamically set the finetune lr nominator, which will ensure that warmup learning rate
        starts at `warmup_lr` and ends with `backbone_lr`. Once the trainer reaches the step
        `finetune_until_step` and learning rate becomes `backbone_lr`, the finetuning phase is
        over.

        lr = ((warmup_lr * numerator) * numerator) ... * numerator))  =  warmup_lr * (numerator)^unfreeze_backbone_at_epoch
                                                    ^ multiplying unfreeze_backbone_at_epoch times
        """
        assert self.unfreeze_at_epoch is not None
        self.num_of_steps_in_epoch = int(
            self.trainer.estimated_stepping_batches / self.trainer.max_epochs
        )
        self.finetune_until_step = self.num_of_steps_in_epoch * self.unfreeze_at_epoch

        _a = self.backbone_lr / self.warmup_lr
        _b = self.finetune_until_step - 1
        self.finetune_lr_nominator = np.exp(np.log(_a) / (_b))

        assert np.isclose(
            np.log(self.backbone_lr),
            np.log(self.warmup_lr)
            + (self.finetune_until_step - 1) * np.log(self.finetune_lr_nominator),
        ), "should be: lr = warmup_lr * (numerator)^unfreeze_backbone_at_epoch"

    @abstractmethod
    def _lr_finetuning_step(self, optimizer_idx):
        """Manually updates learning rate in the finetuning phase.

        Args:
            optimizer_idx
        """
        return

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        """We ignore the lr scheduler in the fine tuning phase and update lr maunally.

        Once the finetuning phase is over we start using the lr scheduler
        """

        if self.is_finetuning_phase():
            self._lr_finetuning_step(optimizer_idx)
            return

        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def setup(self, stage: str) -> None:
        out = super().setup(stage)
        if self.unfreeze_at_epoch is not None:
            self._set_finetune_until_step()
        return out

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.warmup_lr:
            self._set_lr(self.warmup_lr)


class ASTModelWrapper(OurLightningModule):
    loggers: list[TensorBoardLogger]

    def __init__(
        self,
        pretrained: bool,
        batch_size: int,
        scheduler_type: SchedulerType,
        max_epochs: Optional[int],
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
        self.max_epochs = max_epochs
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

        self.hamming_distance = torchmetrics.HammingDistance(
            task="multilabel", num_labels=num_labels
        )
        self.f1_score = MultilabelF1Score(num_labels=self.num_labels)

        self.save_hyperparameters()

    def trainable_backbone(self):
        result = []
        result.extend(self.backbone.audio_spectrogram_transformer.encoder.layer[-1:])
        result.append(self.backbone.audio_spectrogram_transformer.layernorm)
        return result

    def head(self):
        return self.backbone.classifier

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

    def configure_optimizers(self):

        """Set optimizer's learning rate to backbone.

        We do this because we can't explicitly pass the learning rate to scheduler. The scheduler
        infers the learning rate from the optimizer which is why we set it the lr value which
        should be activie once
        """
        if self.optimizer_type is OptimizerType.ADAMW:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.backbone_lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type is OptimizerType.ADAM:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.backbone_lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise UnsupportedOptimizer(
                f"Optimizer {self.optimizer_type} is not implemented",
                self.optimizer_type,
            )

        if self.scheduler_type is SchedulerType.AUTO_LR:
            """SchedulerType.AUTO_LR sets it's own scheduler.

            Only the optimizer has to be returned
            """
            return optimizer

        lr_scheduler_config = {
            "monitor": self.optimization_metric.value,  # "val/loss_epoch",
            # How many epochs/steps should pass between calls to `scheduler.step()`.1 corresponds to updating the learning  rate after every epoch/step.
            # If "monitor" references validation metrics, then "frequency" should be set to a multiple of "trainer.check_val_every_n_epoch".
            "frequency": 1,
            # If using the `LearningRateMonitor` callback to monitor the learning rate progress, this keyword can be used to specify a custom logged name
            "name": self.scheduler_type.value,
        }

        if self.scheduler_type == SchedulerType.ONECYCLE:
            min_lr = 5e-6
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.lr,
                final_div_factor=self.lr / min_lr,
                epochs=15,
                steps_per_epoch=self.num_of_steps_in_epoch,
                verbose=False,
            )
            interval = "step"

        elif self.scheduler_type == SchedulerType.PLATEAU:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.metric_mode.value,
                factor=config_defaults.DEFAULT_LR_PLATEAU_FACTOR,
                patience=(self.epoch_patience // 2) + 1,
                verbose=True,
            )
            interval = "epoch"
        elif self.scheduler_type == SchedulerType.COSINEANNEALING:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.num_of_steps_in_epoch * 3,
                T_mult=1,
            )
            interval = "step"
        else:
            raise UnsupportedScheduler(
                f"Scheduler {self.scheduler_type} is not implemented",
                self.scheduler_type,
            )

        lr_scheduler_config.update(
            {
                "scheduler": scheduler,
                "interval": interval,
            }
        )

        return [optimizer], [lr_scheduler_config]

    def _lr_finetuning_step(self, optimizer_idx):
        """Exponential learning rate update.

        Mupltiplicator is the finetune_lr_nominator
        """
        old_lr = self.trainer.optimizers[optimizer_idx].param_groups[0]["lr"]
        new_lr = old_lr * self.finetune_lr_nominator
        self._set_lr(new_lr)
        return


class EfficientNetV2SmallModel(OurLightningModule):
    """Implementation of EfficientNet V2 small model (384 x 384)"""

    # S    - (384 x 384)
    # M, L - (480 x 480)

    loggers: list[TensorBoardLogger]

    def __init__(
        self,
        pretrained: bool = config_defaults.DEFAULT_PRETRAINED,
        batch_size: int = config_defaults.DEFAULT_BATCH_SIZE,
        scheduler_type: SchedulerType = SchedulerType.PLATEAU,
        max_epochs: Optional[int] = None,
        optimizer_type: OptimizerType = config_defaults.DEFAULT_OPTIMIZER,
        num_labels: int = config_defaults.DEFAULT_NUM_LABELS,
        optimization_metric: OptimizeMetric = config_defaults.DEFAULT_OPTIMIZE_METRIC,
        weight_decay: float = config_defaults.DEFAULT_WEIGHT_DECAY,
        metric_mode: MetricMode = config_defaults.DEFAULT_METRIC_MODE,
        epoch_patience: int = config_defaults.DEFAULT_EARLY_STOPPING_NO_IMPROVEMENT_EPOCHS,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.scheduler_type = scheduler_type
        self.max_epochs = max_epochs
        self.optimizer_type = optimizer_type
        self.num_labels = num_labels
        self.optimization_metric = optimization_metric
        self.weight_decay = weight_decay
        self.metric_mode = metric_mode
        self.epoch_patience = epoch_patience

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

    def configure_optimizers(self):

        # out = our_configure_optimizers(
        #     parameters=self.parameters(),
        #     scheduler_type=self.scheduler_type,
        #     metric_mode=self.metric_mode,
        #     plateau_patience=(self.epoch_patience // 2) + 1,
        #     backbone_lr=self.backbone_lr,
        #     weight_decay=self.weight_decay,
        #     optimizer_type=self.optimizer_type,
        #     optimization_metric=self.optimization_metric,
        #     trainer_estimated_stepping_batches=int(
        #         self.trainer.estimated_stepping_batches
        #     ),
        #     num_of_steps_in_epoch=self.num_of_steps_in_epoch,
        #     cosine_anneal_epochs_reset=self.epoch_patience - 1,
        # )

        if self.optimizer_type is OptimizerType.ADAMW:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type is OptimizerType.ADAM:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise UnsupportedOptimizer(
                f"Optimizer {self.optimizer_type} is not implemented",
                self.optimizer_type,
            )

        if self.scheduler_type is SchedulerType.AUTO_LR:
            """SchedulerType.AUTO_LR sets it's own scheduler.

            Only the optimizer has to be returned
            """
            return optimizer

        lr_scheduler_config = {
            "monitor": self.optimization_metric.value,  # "val/loss_epoch",
            # How many epochs/steps should pass between calls to `scheduler.step()`.1 corresponds to updating the learning  rate after every epoch/step.
            # If "monitor" references validation metrics, then "frequency" should be set to a multiple of "trainer.check_val_every_n_epoch".
            "frequency": 1,
            # If using the `LearningRateMonitor` callback to monitor the learning rate progress, this keyword can be used to specify a custom logged name
            "name": self.scheduler_type.value,
        }

        if self.scheduler_type == SchedulerType.ONECYCLE:
            min_lr = 2.5e-5
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.lr,  # TOOD:self.lr,
                final_div_factor=self.lr / min_lr,
                total_steps=int(self.trainer.estimated_stepping_batches),
                verbose=False,
            )
            interval = "step"

        elif self.scheduler_type == SchedulerType.PLATEAU:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.metric_mode.value,
                factor=config_defaults.DEFAULT_LR_PLATEAU_FACTOR,
                patience=(self.epoch_patience // 2) + 1,
                verbose=True,
            )
            interval = "epoch"
        elif self.scheduler_type == SchedulerType.COSINEANNEALING:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.num_of_steps_in_epoch * 3,
                T_mult=1,
            )
            interval = "step"
        else:
            raise UnsupportedScheduler(
                f"Scheduler {self.scheduler_type} is not implemented",
                self.scheduler_type,
            )

        lr_scheduler_config.update(
            {
                "scheduler": scheduler,
                "interval": interval,
            }
        )

        return [optimizer], [lr_scheduler_config]

    def _lr_finetuning_step(self, optimizer_idx):
        """Exponential learning rate update.

        Mupltiplicator is the finetune_lr_nominator
        """
        old_lr = self.trainer.optimizers[optimizer_idx].param_groups[0]["lr"]
        new_lr = old_lr * self.finetune_lr_nominator
        self._set_lr(new_lr)
        return


def get_model(args, pl_args):
    model_enum = args.model
    if model_enum == SupportedModels.AST and args.pretrained:
        model = ASTModelWrapper(
            pretrained=args.pretrained,
            lr=args.lr,
            batch_size=args.batch_size,
            scheduler_type=args.scheduler,
            max_epochs=pl_args.max_epochs,
            warmup_lr=args.warmup_lr,
            optimizer_type=args.optimizer,
            model_name=config_defaults.DEFAULT_AST_PRETRAINED_TAG,
            num_labels=args.num_labels,
            optimization_metric=args.metric,
            weight_decay=config_defaults.DEFAULT_WEIGHT_DECAY,
            metric_mode=args.metric_mode,
            epoch_patience=args.patience,
            unfreeze_at_epoch=args.unfreeze_at_epoch,
        )
        return model
    elif model_enum == SupportedModels.EFFICIENT_NET_V2_S and args.pretrained:
        model = EfficientNetV2SmallModel(
            pretrained=args.pretrained,
            lr=args.lr,
            batch_size=args.batch_size,
            scheduler_type=args.scheduler,
            max_epochs=pl_args.max_epochs,
            warmup_lr=args.warmup_lr,
            optimizer_type=args.optimizer,
            num_labels=args.num_labels,
            optimization_metric=args.metric,
            weight_decay=config_defaults.DEFAULT_WEIGHT_DECAY,
            metric_mode=args.metric_mode,
            epoch_patience=args.patience,
            unfreeze_at_epoch=args.unfreeze_at_epoch,
        )
        return model
    raise UnsupportedModel(f"Model {model_enum.value} is not supported")


if __name__ == "__main__":
    # python3 -m src.train --accelerator gpu --devices -1 --dataset-dir data/raw/train --audio-transform mel_spectrogram --model efficient_net_v2_s
    model = ASTModelWrapper()
    summary(
        model,
    )
    ModelSummary(model, max_depth=-1)
