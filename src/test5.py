"""Global config file.

To see the list of all arguments call `pyhton3 src/train.py -h`
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from dataclasses import Field
from operator import attrgetter
from pathlib import Path
from typing import Optional

import configargparse
import pydantic_argparse
import pytorch_lightning as pl
import torch
from pydantic import BaseModel, Field

import src.config.defaults as defaults
import src.utils.utils_functions as utils_functions

# from src.default_args import ConfigDefaults
from src.enums.enums import (
    AudioTransforms,
    MetricMode,
    OptimizeMetric,
    SupportedAugmentations,
    SupportedHeads,
    SupportedLossFunctions,
    SupportedModels,
    SupportedOptimizer,
    SupportedScheduler,
)
from src.utils.utils_dataset import parse_dataset_enum_dirs
from src.utils.utils_exceptions import InvalidArgument

__all__ = ["config", "pl_args"]

from tap import Tap


class SimpleArgumentParser(Tap):
    audio_transform: AudioTransforms.from_string = None
    aug_kwargs: str = None
    augmentations: SupportedAugmentations.from_string = defaults.DEFAULT_AUGMENTATIONS
    backbone_after: str = None
    bar_update: utils_functions.is_positive_int = defaults.DEFUALT_TQDM_REFRESH
    batch_size: int = defaults.DEFAULT_BATCH_SIZE
    check_on_train_epoch_end: bool = defaults.DEFAULT_CHECK_ON_TRAIN_EPOCH_END
    ckpt: str | None = None
    dataset_fraction: utils_functions.is_between_0_1 = defaults.DEFAULT_DATASET_FRACTION
    drop_last: bool = True
    early_stopping_metric_patience: utils_functions.is_positive_int = (
        defaults.DEFAULT_EARLY_STOPPING_METRIC_PATIENCE
    )
    epochs: utils_functions.is_positive_int = defaults.DEFAULT_EPOCHS
    finetune_head: bool = defaults.DEFAULT_FINETUNE_HEAD
    finetune_head_epochs: int = defaults.DEFAULT_FINETUNE_HEAD_EPOCHS
    freeze_train_bn: bool = defaults.DEFAULT_FREEZE_TRAIN_BN
    head: SupportedHeads = defaults.DEAFULT_HEAD
    head_after: str | None = None
    hop_length: int = defaults.DEFAULT_HOP_LENGTH
    image_dim: tuple[int, int] = defaults.DEFAULT_IMAGE_DIM
    log_per_instrument_metrics: bool = defaults.DEFAULT_LOG_PER_INSTRUMENT_METRICS
    loss_function: SupportedLossFunctions = SupportedLossFunctions.CROSS_ENTROPY
    loss_function_kwargs: dict = {}
    lr: float = defaults.DEFAULT_LR
    lr_onecycle_max: float = defaults.DEFAULT_LR_ONECYCLE_MAX
    lr_warmup: float = defaults.DEFAULT_LR_WARMUP
    max_audio_seconds: float = defaults.DEFAULT_MAX_AUDIO_SECONDS
    metric: OptimizeMetric.from_string = defaults.DEFAULT_OPTIMIZE_METRIC
    metric_mode: MetricMode.from_string = defaults.DEFAULT_METRIC_MODE
    model: SupportedModels.from_string = None
    n_fft: int = defaults.DEFAULT_N_FFT
    n_mels: int = defaults.DEFAULT_N_MELS
    n_mfcc: int = defaults.DEFAULT_N_MFCC
    normalize_audio: bool = defaults.DEFAULT_NORMALIZE_AUDIO
    num_labels: utils_functions.is_positive_int = defaults.DEFAULT_NUM_LABELS
    num_workers: utils_functions.is_positive_int = defaults.DEFAULT_NUM_WORKERS
    optimizer: str = SupportedOptimizer.ADAMW
    output_dir: Path = defaults.PATH_MODELS
    pretrained: bool = defaults.DEFAULT_PRETRAINED
    pretrained_tag: str = defaults.DEFAULT_PRETRAINED_TAG
    quick: bool = False
    sampling_rate: int = defaults.DEFAULT_SAMPLING_RATE
    save_on_train_epoch_end: bool = defaults.DEFAULT_SAVE_ON_TRAIN_EPOCH_END
    scheduler: SupportedScheduler = SupportedScheduler.ONECYCLE
    skip_validation: bool = defaults.DEFAULT_SKIP_VALIDATION
    train_dirs: parse_dataset_enum_dirs = defaults.DEFAULT_TRAIN_DIRS
    train_only_dataset: bool = defaults.DEFAULT_ONLY_TRAIN_DATASET
    train_override_csvs: Path | None = None
    use_fluffy: bool = defaults.DEFAULT_USE_FLUFFY
    use_multiple_optimizers: bool = defaults.DEFAULT_USE_MULTIPLE_OPTIMIZERS
    use_weighted_train_sampler: bool = defaults.DEFAULT_USE_WEIGHTED_TRAIN_SAMPLER
    val_dirs: parse_dataset_enum_dirs = defaults.DEFAULT_VAL_DIRS
    weight_decay: float = defaults.DEFAULT_WEIGHT_DECAY


fun = SimpleArgumentParser().parse_args()
print(fun)
