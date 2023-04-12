"""Global config file.

To see the list of all arguments call `pyhton3 src/train.py -h`
"""

from __future__ import annotations

import argparse
from dataclasses import Field
from operator import attrgetter
from pathlib import Path

import configargparse
import pydantic_argparse
import pytorch_lightning as pl
import torch
from pydantic import BaseModel, Field

import src.config.defaults as defaults
import src.utils.utils_functions as utils_functions
from src.default_args import ConfigDefaults
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


def add_model(parser, model):
    """Add Pydantic model to an ArgumentParser."""
    fields = model.__fields__
    for name, field in fields.items():
        if field.type_ is bool:
            kwargs = dict(
                dest=name,
                help=field.field_info.description,
                action="store_true",
            )
        else:
            kwargs = dict(
                dest=name,
                type=field.type_,
                help=field.field_info.description,
            )
        if field.type_ is list:
            kwargs.update({""})
        if field.default is None:
            kwargs.update({""})
        parser.add_argument(f"--{name}", **kwargs)


class SortingHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Alphabetically sort -h."""

    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter("option_strings"))
        super().add_arguments(actions)


# Intialize parser and it's groups
ARGS_GROUP_NAME = "General arguments"

parser = pydantic_argparse.ArgumentParser(
    model=ConfigDefaults,
    prog="Example Program",
    description="Example Description",
    version="0.0.1",
    epilog="Example Epilog",
    formatter_class=SortingHelpFormatter,
)

lightning_parser = pl.Trainer.add_argparse_args(parser)
lightning_parser.set_defaults(
    log_every_n_steps=defaults.DEFAULT_LOG_EVERY_N_STEPS,
    epochs=defaults.DEFAULT_EPOCHS,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=-1,  # use all devices
)

user_group = parser.add_argument_group(ARGS_GROUP_NAME)

args = parser.parse_typed_args()

# args = parser.parse_args()

# Separate Namespace into two Namespaces
args_dict: dict[str, argparse.Namespace] = {}
for group in parser._action_groups:
    group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
    if group.title:
        args_dict[group.title] = argparse.Namespace(**group_dict)

args, pl_args = args_dict[ARGS_GROUP_NAME], args_dict["pl.Trainer"]


config_defaults = ConfigDefaults(**vars(args))
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
