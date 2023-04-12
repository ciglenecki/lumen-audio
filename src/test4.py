"""Global config file.

To see the list of all arguments call `pyhton3 src/train.py -h`
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from operator import attrgetter

import pytorch_lightning as pl
import simple_parsing
import torch
from simple_parsing import DashVariant

from src.default_args import ConfigDefault
from src.enums.enums import SupportedAugmentations
from src.utils.utils_exceptions import InvalidArgument

a = ConfigDefault()


class SortingHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Alphabetically sort -h."""

    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter("option_strings"))
        super().add_arguments(actions)

    def _get_help_string(self, action) -> str | None:
        string = super()._get_help_string(action)
        # string =


user_dest = "user_args"
# SimpleDataParser creates a group with a specific group format.
user_group_name = f"{ConfigDefault.__name__} ['{user_dest}']"
pl_group_name = "pl.Trainer"


def get_config() -> ConfigDefault:
    parser = simple_parsing.ArgumentParser(
        formatter_class=SortingHelpFormatter,
        add_option_string_dash_variants=DashVariant.DASH,
    )

    parser.add_arguments(ConfigDefault, dest=user_dest)
    lightning_parser = pl.Trainer.add_argparse_args(parser)
    lightning_parser.set_defaults(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,  # use all devices
    )

    args = parser.parse_args()
    args_dict = vars(args)

    """
    args_dict = {
        "accelerator": "gpu",
        "devices": -1,
        "user_dest": ConfigDefault(
            n_fft=400,
            ...
        )
        ...
    }
    """
    config: ConfigDefault = args_dict.pop(user_dest)
    config._validate_train_args()
    pl_args = Namespace(**args_dict)

    if config.quick:
        pl_args.limit_train_batches = 2
        pl_args.limit_val_batches = 2
        pl_args.limit_test_batches = 2
        pl_args.log_every_n_steps = 1
        config.dataset_fraction = 0.01
        config.batch_size = 2
        config.output_dir = config.path_models_quick

    if config.epochs:
        pl_args.epochs = config.epochs
        pl_args.max_epochs = config.epochs

    if config.skip_validation:
        pl_args.limit_val_batches = 0

    return config


config = get_config()
