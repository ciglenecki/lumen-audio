"""Global config file.

To see the list of all arguments call `pyhton3 src/train.py -h`
"""

from __future__ import annotations

import argparse
from argparse import Namespace

import pytorch_lightning as pl
import simple_parsing
import torch
from simple_parsing import DashVariant

from src.config.argparse_with_config import ArgParseWithConfig, SortingHelpFormatter
from src.config.config_defaults import ConfigDefault
from src.enums.enums import all_enums


def get_epipolog():
    """Prints all enums as strings."""
    epilog = "==== Enums ====\n"
    for enum_class in all_enums:
        enums = list(enum_class)
        epilog += enum_class.__name__ + ": "
        epilog += ", ".join([e.name for e in enums]) + "\n\n\n"
    return epilog


def get_config() -> tuple[ConfigDefault, Namespace]:
    parser = ArgParseWithConfig(
        formatter_class=SortingHelpFormatter,
        epilog=get_epipolog(),
    )
    _, config, pl_args = parser.parse_args()
    config._check_train_args()

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
