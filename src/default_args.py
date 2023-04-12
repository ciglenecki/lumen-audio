"""Global default config."""

from __future__ import annotations

from dataclasses import Field
from operator import attrgetter
from pathlib import Path

import configargparse
import pytorch_lightning as pl
import torch
from pydantic import BaseModel, Field, PositiveInt, validator

import src.config.defaults as defaults
import src.utils.utils_functions as utils_functions
from src.enums.enums import (
    AudioTransforms,
    MetricMode,
    OptimizeMetric,
    SupportedAugmentations,
    SupportedDatasets,
    SupportedHeads,
    SupportedLossFunctions,
    SupportedModels,
    SupportedOptimizer,
    SupportedScheduler,
)
from src.utils.utils_dataset import parse_dataset_enum_dirs
from src.utils.utils_exceptions import InvalidArgument


class ConfigDefaults(BaseModel):
    audio_transform: AudioTransforms = Field(
        default=None,
        description="Transformation which will be performed on audio and labels",
    )
    aug_kwargs: str = Field(
        default=None,
        description="Arguments are split by space, mutiple values are sep'ed by comma (,). E.g. stretch_factors=0.8,1.2 freq_mask_param=30 time_mask_param=30 hide_random_pixels_p=0.5",
    )
    augmentations: list[SupportedAugmentations] = Field(
        default=defaults.DEFAULT_AUGMENTATIONS,
        description="Transformation which will be performed on audio and labels",
    )
    backbone_after: str = Field(
        default=None,
        description="Name of the submodule after which the all submodules are considered as backbone, e.g. layer.11.dense",
    )
    bar_update: PositiveInt = Field(
        default=defaults.DEFUALT_TQDM_REFRESH,
        description="Number of TQDM updates in one epoch.",
    )
    batch_size: int = Field(default=defaults.DEFAULT_BATCH_SIZE, description="None")
    check_on_train_epoch_end: bool = Field(
        default=defaults.DEFAULT_CHECK_ON_TRAIN_EPOCH_END,
        description="Whether to run early stopping at the end of the training epoch.",
    )
    ckpt: str = Field(
        default=None,
        description=".ckpt file, automatically restores model, epoch, step, LR schedulers, etc...",
    )

    dataset_fraction: int | float = Field(
        default=defaults.DEFAULT_DATASET_FRACTION,
        description="Reduce each dataset split (train, val, test) by a fraction.",
        ge=0,
    )
    drop_last: bool = Field(
        default=True,
        description="Drop last sample if the size of the sample is smaller than batch size",
    )
    early_stopping_metric_patience: PositiveInt = Field(
        default=defaults.DEFAULT_EARLY_STOPPING_METRIC_PATIENCE,
        description="Number of checks with no improvement after which training will be stopped. Under the default configuration, one check happens after every training epoch",
    )
    epochs: PositiveInt = Field(
        default=defaults.DEFAULT_EPOCHS,
        description="Number epochs. Works only if learning rate scheduler has fixed number of steps (onecycle, cosine...). It won't have an effect on 'reduce on palteau' lr scheduler.",
    )
    finetune_head: bool = Field(
        default=defaults.DEFAULT_FINETUNE_HEAD,
        description="Performs head only finetuning for --finetune-head-epochs epochs with starting lr of --lr-warmup which eventually becomes --lr.",
    )
    finetune_head_epochs: int = Field(
        default=defaults.DEFAULT_FINETUNE_HEAD_EPOCHS,
        description="Epoch at which the backbone will be unfrozen.",
    )
    freeze_train_bn: bool = Field(
        default=defaults.DEFAULT_FREEZE_TRAIN_BN,
        description="If true, the batch norm will be trained even if module is frozen.",
    )
    head: SupportedHeads = Field(
        default=defaults.DEAFULT_HEAD, description="classifier head"
    )
    head_after: str | None = Field(
        default=None,
        description="Name of the submodule after which the all submodules are considered as head, e.g. classifier.dense",
    )
    hop_length: int = Field(default=defaults.DEFAULT_HOP_LENGTH, description="None")
    image_dim: tuple[int, int] = Field(
        default=defaults.DEFAULT_IMAGE_DIM,
        description="The dimension to resize the image to.",
    )
    log_per_instrument_metrics: bool = Field(
        default=defaults.DEFAULT_LOG_PER_INSTRUMENT_METRICS,
        description="Along with aggregated metrics, also log per instrument metrics.",
    )
    loss_function: SupportedLossFunctions = Field(
        default=SupportedLossFunctions.CROSS_ENTROPY, description="Loss function"
    )
    loss_function_kwargs: dict = Field(default={}, description="Loss function kwargs")
    lr: float = Field(default=defaults.DEFAULT_LR, description="Learning rate")
    lr_onecycle_max: float = Field(
        default=defaults.DEFAULT_LR_ONECYCLE_MAX,
        description="Maximum lr OneCycle scheduler reaches",
    )
    lr_warmup: float = Field(
        default=defaults.DEFAULT_LR_WARMUP, description="warmup learning rate"
    )
    max_audio_seconds: float = Field(
        default=defaults.DEFAULT_MAX_AUDIO_SECONDS,
        description="Maximum number of seconds of audio which will be processed at one time.",
    )
    metric: OptimizeMetric = Field(
        default=defaults.DEFAULT_OPTIMIZE_METRIC,
        description="Metric which the model will optimize for.",
    )
    metric_mode: MetricMode = Field(
        default=defaults.DEFAULT_METRIC_MODE,
        description="Maximize or minimize the --metric.",
    )
    model: SupportedModels = Field(description="Models used for training.")
    n_fft: int = Field(default=defaults.DEFAULT_N_FFT, description="None")
    n_mels: int = Field(default=defaults.DEFAULT_N_MELS, description="None")
    n_mfcc: int = Field(default=defaults.DEFAULT_N_MFCC, description="None")

    normalize_audio: bool = Field(
        default=defaults.DEFAULT_NORMALIZE_AUDIO,
        description="Normalize audio to [-1, 1]",
    )
    num_labels: PositiveInt = Field(
        default=defaults.DEFAULT_NUM_LABELS,
        description="Total number of possible lables",
    )
    num_workers: PositiveInt = Field(
        default=defaults.DEFAULT_NUM_WORKERS, description="Number of workers"
    )
    optimizer: str = Field(default=SupportedOptimizer.ADAMW, description="None")
    output_dir: Path = Field(
        default=defaults.PATH_MODELS,
        description="Output directory of the model and report file.",
    )
    pretrained: bool = Field(
        default=defaults.DEFAULT_PRETRAINED,
        description="Use a pretrained model loaded from the web.",
    )
    pretrained_tag: str = Field(
        default=defaults.DEFAULT_PRETRAINED_TAG,
        description="The string that denotes the pretrained weights used.",
    )
    quick: bool = Field(
        default=False,
        description="For testing bugs. Simulates --limit_train_batches 2 --limit_val_batches 2 --limit_test_batches 2",
    )
    sampling_rate: int = Field(
        default=defaults.DEFAULT_SAMPLING_RATE, description="None"
    )
    save_on_train_epoch_end: bool = Field(
        default=defaults.DEFAULT_SAVE_ON_TRAIN_EPOCH_END,
        description="Whether to run checkpointing at the end of the training epoch.",
    )
    scheduler: SupportedScheduler = Field(
        default=SupportedScheduler.ONECYCLE, description="None"
    )
    skip_validation: bool = Field(
        default=defaults.DEFAULT_SKIP_VALIDATION,
        description="Skips validation part during training.",
    )
    train_dirs: list[tuple[SupportedDatasets, Path]] = Field(
        default=[defaults.DEFAULT_TRAIN_DIRS],
        description="Dataset root directories that will be used for training in the following format: --train-dirs irmas:/path/to openmic:/path/to",
    )
    train_only_dataset: bool = Field(
        default=defaults.DEFAULT_ONLY_TRAIN_DATASET,
        description="Use only the train portion of the dataset and split it 0.8 0.2",
    )
    train_override_csvs: Path = Field(
        default=None,
        description="CSV files with columns 'filename, sax, gac, org, ..., cla' where filename is path and each instrument is either 0 or 1",
    )
    use_fluffy: bool = Field(
        default=defaults.DEFAULT_USE_FLUFFY,
        description="Use multiple optimizers for Fluffy.",
    )
    use_multiple_optimizers: bool = Field(
        default=defaults.DEFAULT_USE_MULTIPLE_OPTIMIZERS,
        description="Use multiple optimizers for Fluffy. Each head will have it's own optimizer.",
    )
    use_weighted_train_sampler: bool = Field(
        default=defaults.DEFAULT_USE_WEIGHTED_TRAIN_SAMPLER,
        description="Use weighted train sampler instead of a random one.",
    )
    val_dirs: list[tuple[SupportedDatasets, Path]] = Field(
        default=defaults.DEFAULT_VAL_DIRS,
        description="Dataset root directories that will be used for validation in the following format: --val-dirs irmas:/path/to openmic:/path/to",
    )
    weight_decay: float = Field(
        default=defaults.DEFAULT_WEIGHT_DECAY,
        description="Maximum lr OneCycle scheduler reaches",
    )

    @validator("train_dirs")
    def transform_it(cls, train_dirs):
        return parse_dataset_enum_dirs(train_dirs)

    class Config:
        use_enum_values = True


# config_defaults = ConfigDefaults()
