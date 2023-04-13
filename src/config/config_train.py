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

from src.config.config_defaults import ConfigDefault
from src.enums.enums import all_enums


class SortingHelpFormatter(
    simple_parsing.SimpleHelpFormatter, argparse.RawTextHelpFormatter
):
    """Alphabetically sort -h."""

    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter("option_strings"))
        super().add_arguments(actions)


def get_epipolog():
    """Prints all enums as strings."""
    epilog = "==== Enums ====\n"
    for enum_class in all_enums:
        enums = list(enum_class)
        epilog += enum_class.__name__ + ": "
        epilog += ", ".join([e.name for e in enums]) + "\n\n\n"
    return epilog


def get_config() -> tuple[ConfigDefault, Namespace]:
    user_dest = "user_args"
    user_group_name = f"{ConfigDefault.__name__} ['{user_dest}']"
    pl_group_name = "pl.Trainer"

    # Create a parser
    parser = simple_parsing.ArgumentParser(
        formatter_class=SortingHelpFormatter,
        add_option_string_dash_variants=DashVariant.DASH,
        epilog=get_epipolog(),
    )

    # Add PyTorch Lightning args to CLI
    parser.add_arguments(ConfigDefault, dest=user_dest)
    lightning_parser = pl.Trainer.add_argparse_args(parser)
    lightning_parser.set_defaults(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,  # use all devices
    )

    # Parse and split in two: config and pytorch lightning args
    args = parser.parse_args()
    args_dict = vars(args)
    config: ConfigDefault = args_dict.pop(user_dest)
    config._validate_train_args()
    pl_args = Namespace(**args_dict)

    # Dynamically set some PyTorch lightning arguments
    if config.log_every_n_steps:
        pl_args.log_every_n_steps = config.log_every_n_steps

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

    return config, pl_args


config, pl_args = get_config()
