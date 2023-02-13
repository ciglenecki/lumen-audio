from enum import Enum
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.io import wavfile
from torchmetrics import HammingDistance
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification
from transformers.modeling_outputs import SequenceClassifierOutput

import config_defaults
from utils_train import MetricMode, OptimizeMetric, OptimizerType, SchedulerType


class UnsupportedOptimizer(ValueError):
    pass


class UnsupportedScheduler(ValueError):
    pass


class SupportedModels(Enum):
    ast = "ast"

    def __str__(self):
        return self.value


class ASTModelWrapper(pl.LightningModule):
    loggers: list[TensorBoardLogger]
    class_to_crs_centroid_map: torch.Tensor
    class_to_crs_weighted_map: torch.Tensor

    def __init__(
        self,
        pretrained: bool = config_defaults.DEFAULT_PRETRAINED,
        lr: float = config_defaults.DEFAULT_LR,
        batch_size: int = config_defaults.DEFAULT_BATCH_SIZE,
        scheduler_type: SchedulerType = SchedulerType.PLATEAU,
        max_epochs: Optional[int] = None,
        optimizer_type: OptimizerType = OptimizerType.ADAM,
        model_name: str = config_defaults.DEFAULT_AST_PRETRAINED_TAG,
        num_labels: int = config_defaults.DEFAULT_NUM_CLASSES,
        optimization_metric: OptimizeMetric = config_defaults.DEFAULT_OPTIMIZE_METRIC,
        weight_decay: float = config_defaults.DEFAULT_WEIGHT_DECAY,
        metric_mode: MetricMode = config_defaults.DEFAULT_METRIC_MODE,
        early_stopping_epoch: int = config_defaults.DEFAULT_EARLY_STOPPING_NO_IMPROVEMENT_EPOCHS,
    ):
        super().__init__()
        self.pretrained = pretrained
        self.lr = lr
        self.batch_size = batch_size
        self.scheduler_type = scheduler_type
        self.max_epochs = max_epochs
        self.optimizer_type = optimizer_type
        self.num_labels = num_labels
        self.model_name = model_name
        self.optimization_metric = optimization_metric
        self.weight_decay = weight_decay
        self.metric_mode = metric_mode
        self.early_stopping_epoch = early_stopping_epoch

        self.feature_extractor = ASTFeatureExtractor.from_pretrained(
            model_name, sampling_rate=config_defaults.DEFAULT_SAMPLING_RATE
        )

        config = ASTConfig(
            pretrained_model_name_or_path=model_name,
            id2label=config_defaults.IDX_TO_INSTRUMENT,
            label2id=config_defaults.IDX_TO_INSTRUMENT,
            num_labels=num_labels,
            finetuning_task="audio-classification",
        )

        ### Hug version
        self.backbone: ASTForAudioClassification = ASTForAudioClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)  # type: ignore
        self.hamming_distance = HammingDistance(task="multilabel", num_labels=num_labels)

        print(type(self.backbone))
        self.save_hyperparameters()

    def _fake_forward(self):
        sampling_rate, audio = wavfile.read("data/raw/train/cel/[cel][cla]0001__1.wav")
        audio = audio.sum(axis=1) / 2
        features = self.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        x = features["input_values"]
        y = torch.rand((1, 11))
        out: SequenceClassifierOutput = self.backbone.forward(x, output_attentions=True, return_dict=True, labels=y)
        return out.loss, out.logits

    def forward(self, audio: torch.Tensor, labels: torch.Tensor, sampling_rate=config_defaults.DEFAULT_SAMPLING_RATE):
        features = self.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        out: SequenceClassifierOutput = self.backbone.forward(features["input_values"], output_attentions=True, return_dict=True, labels=labels)  # type: ignore
        return out.loss, out.logits

    def training_step(self, batch, batch_idx):
        audio, y = batch
        loss, y_pred = self.forward(audio, y)

        hamming_acc = self.hamming_distance(y, y_pred)

        data_dict = {
            "loss": loss,  # the 'loss' key needs to be present
            "train/loss": loss,
            "train/hamming_acc": hamming_acc,
        }

        log_dict = data_dict.copy()
        log_dict.pop("loss", None)
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        return data_dict

    def validation_step(self, batch, batch_idx):
        audio, y = batch
        loss, y_pred = self.forward(audio, y)

        hamming_acc = self.hamming_distance(y, y_pred)

        data_dict = {
            "loss": loss,  # the 'loss' key needs to be present
            "val/loss": loss,
            "val/hamming_acc": hamming_acc,
        }

        log_dict = data_dict.copy()
        log_dict.pop("loss", None)
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        return data_dict

    def test_step(self, batch, batch_idx):
        audio, y = batch
        loss, y_pred = self.forward(audio, y)  # same as self.forward

        hamming_acc = self.hamming_distance(y, y_pred)

        data_dict = {
            "loss": loss,  # the 'loss' key needs to be present
            "test/loss": loss,
            "test/hamming_acc": hamming_acc,
        }

        log_dict = data_dict.copy()
        log_dict.pop("loss", None)
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        return data_dict

    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0) -> Any:
        pass
        return
        audio, y = batch
        with torch.no_grad():
            loss, y_pred = self.forward(audio, y)  # same as self.forward
        return y_pred

    def configure_optimizers(self):
        print("\n", self.__class__.__name__, "Configure optimizers\n")

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
            raise UnsupportedOptimizer(f"Optimizer {self.optimizer_type} is not implemented", self.optimizer_type)

        if self.scheduler_type is SchedulerType.AUTO_LR:
            """SchedulerType.AUTO_LR sets it's own scheduler.

            Only the optimizer has to be returned
            """
            return optimizer

        config_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "monitor": self.optimization_metric,  # "val/loss_epoch",
                # How many epochs/steps should pass between calls to `scheduler.step()`.1 corresponds to updating the learning  rate after every epoch/step.
                # If "monitor" references validation metrics, then "frequency" should be set to a multiple of "trainer.check_val_every_n_epoch".
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the learning rate progress, this keyword can be used to specify a custom logged name
                "name": self.scheduler_type.value,
            },
        }

        if self.scheduler_type == SchedulerType.ONECYCLE:
            min_lr = 2.5e-5
            initial_lr = 1e-1
            scheduler = torch.optim.lr_scheduler.OneCycleLR(  # type: ignore
                optimizer=optimizer,
                max_lr=initial_lr,  # TOOD:self.lr,
                final_div_factor=initial_lr / min_lr,
                total_steps=int(self.trainer.estimated_stepping_batches),
                verbose=False,
            )
            interval = "step"

        elif self.scheduler_type == SchedulerType.PLATEAU:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.metric_mode.value,
                factor=config_defaults.DEFAULT_LR_PLATEAU_FACTOR,
                patience=self.early_stopping_epoch,
                verbose=True,
            )
            interval = "epoch"
        else:
            raise UnsupportedScheduler(f"Scheduler {self.scheduler_type} is not implemented", self.scheduler_type)

        config_dict["lr_scheduler"].update(
            {
                "scheduler": scheduler,
                "interval": interval,
            }
        )
        return config_dict

    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     if self.current_epoch < self.unfreeze_at_epoch:
    #         return
    #     if metric is None:
    #         scheduler.step()  # type: ignore
    #     else:
    #         scheduler.step(metric)


def get_model(args):
    if args.model == SupportedModels.ast.value and args.pretrained:
        model = ASTModelWrapper(
            pretrained=args.pretrained,
            lr=args.lr,
            batch_size=args.batch_size,
            scheduler_type=SchedulerType[args.scheduler_type],
            max_epochs=args.max_epochs,
            optimizer_type=OptimizerType[args.optimizer_type],
            model_name=args.model_name,
            num_labels=args.num_labels,
        )
        return model


if __name__ == "__main__":
    ASTModelWrapper().forward()
