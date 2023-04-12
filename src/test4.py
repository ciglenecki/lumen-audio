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
    ckpt: Optional[str] = None
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
    head_after: Optional[str] = None
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
    train_override_csvs: Optional[Path] = None
    use_fluffy: bool = defaults.DEFAULT_USE_FLUFFY
    use_multiple_optimizers: bool = defaults.DEFAULT_USE_MULTIPLE_OPTIMIZERS
    use_weighted_train_sampler: bool = defaults.DEFAULT_USE_WEIGHTED_TRAIN_SAMPLER
    val_dirs: parse_dataset_enum_dirs = defaults.DEFAULT_VAL_DIRS
    weight_decay: float = defaults.DEFAULT_WEIGHT_DECAY


class SortingHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Alphabetically sort -h."""

    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter("option_strings"))
        super().add_arguments(actions)


ARGS_GROUP_NAME = "General arguments"


parser = SimpleArgumentParser(formatter_class=SortingHelpFormatter)

lightning_parser = pl.Trainer.add_argparse_args(parser)
lightning_parser.set_defaults(
    log_every_n_steps=defaults.DEFAULT_LOG_EVERY_N_STEPS,
    epochs=defaults.DEFAULT_EPOCHS,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=-1,  # use all devices
)

args, pl_args = parser.parse_known_args()
# args = Namespace(**args)
print(args)
# Separate Namespace into two Namespaces
args_dict: dict[str, argparse.Namespace] = {}
for group in parser._action_groups:
    group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
    if group.title:
        args_dict[group.title] = argparse.Namespace(**group_dict)

args, pl_args = args_dict[ARGS_GROUP_NAME], args_dict["pl.Trainer"]


config_defaults = args
# print(config_defaults)

# User arguments which override PyTorch Lightning arguments
if config_defaults.quick:
    pl_args.limit_train_batches = 2
    pl_args.limit_val_batches = 2
    pl_args.limit_test_batches = 2
    pl_args.log_every_n_steps = 1
    config_defaults.dataset_fraction = 0.01
    config_defaults.batch_size = 2
    config_defaults.output_dir = defaults.PATH_MODELS_QUICK

if config_defaults.epochs:
    pl_args.max_epochs = config_defaults.epochs

# Additional argument checking
if config_defaults.metric and not config_defaults.metric_mode:
    raise InvalidArgument("can't pass --metric without passing --metric-mode")


if config_defaults.aug_kwargs is None:
    config_defaults.aug_kwargs = defaults.DEFAULT_AUGMENTATION_KWARSG
else:
    config_defaults.aug_kwargs = utils_functions.parse_kwargs(
        config_defaults.aug_kwargs
    )

if (
    config_defaults.scheduler == SupportedScheduler.ONECYCLE
    and config_defaults.lr_onecycle_max is None
):
    raise InvalidArgument(
        f"You have to pass the --lr-onecycle-max if you use the {config_defaults.scheduler}",
    )

if (
    config_defaults.model != SupportedModels.WAV2VECCNN
    and config_defaults.use_multiple_optimizers
):
    raise InvalidArgument(
        "You can't use mutliple optimizers if you are not using Fluffy!",
    )

# Dynamically set pretrained tag
if (
    config_defaults.pretrained
    and config_defaults.pretrained_tag == defaults.DEFAULT_PRETRAINED_TAG
):
    print("\n", config_defaults.model)
    if config_defaults.model == SupportedModels.AST:
        config_defaults.pretrained_tag = defaults.DEFAULT_AST_PRETRAINED_TAG
    elif config_defaults.model in [SupportedModels.WAV2VECCNN, SupportedModels.WAV2VEC]:
        config_defaults.pretrained_tag = defaults.DEFAULT_WAV2VEC_PRETRAINED_TAG
    elif config_defaults.model in [
        SupportedModels.EFFICIENT_NET_V2_S,
        SupportedModels.EFFICIENT_NET_V2_M,
        SupportedModels.EFFICIENT_NET_V2_L,
        SupportedModels.RESNEXT50_32X4D,
        SupportedModels.RESNEXT101_32X8D,
        SupportedModels.RESNEXT101_64X4D,
    ]:
        config_defaults.pretrained_tag = defaults.DEFAULT_TORCH_CNN_PRETRAINED_TAG
    else:
        raise Exception("Shouldn't happen")

# Dynamically AST DSP attributes
if config_defaults.model == SupportedModels.AST:
    config_defaults.n_fft = defaults.DEFAULT_AST_N_FFT
    config_defaults.hop_length = defaults.DEFAULT_AST_HOP_LENGTH
    config_defaults.n_mels = defaults.DEFAULT_AST_N_MELS

if config_defaults.skip_validation:
    pl_args.limit_val_batches = 0
config = args
