"""Input arguments for the train.py file.

To see the list of all arguments call `pyhton3 src/train.py -h`
"""
from __future__ import annotations

import argparse
from typing import Dict, Tuple

import pytorch_lightning as pl

import config_defaults
from src.utils_audio import AudioTransforms
import utils_functions
from model import SupportedModels
from utils_train import MetricMode, OptimizeMetric, OptimizerType, SchedulerType

ARGS_GROUP_NAME = "General arguments"


def parse_args_train() -> tuple[argparse.Namespace, argparse.Namespace]:
    parser = argparse.ArgumentParser()

    lightning_parser = pl.Trainer.add_argparse_args(parser)
    lightning_parser.set_defaults(log_every_n_steps=config_defaults.DEFAULT_LOG_EVERY_N_STEPS)

    user_group = parser.add_argument_group(ARGS_GROUP_NAME)

    user_group.add_argument(
        "--dataset-fraction",
        metavar="float",
        default=config_defaults.DEFAULT_DATASET_FRACTION,
        type=utils_functions.is_between_0_1,
        help="Fraction of the dataset that will be used. This fraction reduces the size for all dataset splits.",
    )

    user_group.add_argument(
        "--num-workers",
        metavar="int",
        default=config_defaults.DEFAULT_NUM_WORKERS,
        type=utils_functions.is_positive_int,
        help="Number of workers",
    )
    user_group.add_argument(
        "--num-labels",
        metavar="int",
        default=config_defaults.DEFAULT_NUM_CLASSES,
        type=utils_functions.is_positive_int,
        help="Total number of possible lables",
    )
    user_group.add_argument(
        "--model",
        default=SupportedModels.ast,
        type=SupportedModels,
        choices=list(SupportedModels),
        help="Models used for training. resnext101_32x8d is recommend. We not guarantee that other models will work out the box",
    )
    user_group.add_argument(
        "--audio-transform",
        default=AudioTransforms.ast,
        type=AudioTransforms,
        choices=list(AudioTransforms),
        help="Models used for training. resnext101_32x8d is recommend. We not guarantee that other models will work out the box",
    )

    user_group.add_argument(
        "--lr",
        default=config_defaults.DEFAULT_LR,
        type=float,
        help="Learning rate",
    )
    user_group.add_argument(
        "--dataset-dirs",
        metavar="dir",
        nargs="+",
        type=str,
        help="Dataset root directories that will be used for training, validation and testing",
        required=True,
        default=[config_defaults.PATH_TRAIN],
    )

    user_group.add_argument(
        "--output-report",
        metavar="dir",
        type=str,
        help="Directory where report file will be created.",
        default=config_defaults.PATH_REPORT,
    )

    user_group.add_argument(
        "--pretrained",
        type=bool,
        help="Use the pretrained model.",
        default=config_defaults.DEFAULT_PRETRAINED,
    )

    user_group.add_argument(
        "--drop-last",
        type=bool,
        help="Drop last sample if the size of the sample is smaller than batch size",
        default=True,
    )

    user_group.add_argument(
        "--check-on-train-epoch-end",
        type=bool,
        help="Whether to run early stopping at the end of the training epoch.",
        default=config_defaults.DEFAULT_CHECK_ON_TRAIN_EPOCH_END,
    )

    user_group.add_argument(
        "--save-on-train-epoch-end",
        type=bool,
        help="hether to run checkpointing at the end of the training epoch",
        default=config_defaults.DEFAULT_SAVE_ON_TRAIN_EPOCH_END,
    )
    user_group.add_argument(
        "--metric",
        type=OptimizeMetric,
        help="Which metric to optimize for",
        default=config_defaults.DEFAULT_OPTIMIZE_METRIC,
        choices=list(OptimizeMetric),
    )

    user_group.add_argument(
        "--metric-mode",
        type=MetricMode,
        help="Maximize or minimize the --metric",
        default=config_defaults.DEFAULT_METRIC_MODE,
        choices=list(MetricMode),
    )

    user_group.add_argument(
        "-q",
        "--quick",
        help="Simulates --limit_train_batches 2 --limit_val_batches 2 --limit_test_batches 2",
        action="store_true",
        default=False,
    )

    user_group.add_argument(
        "--ckpt",
        help=".ckpt file, automatically restores model, epoch, step, LR schedulers, etc...",
        metavar="path",
        type=str,
    )
    user_group.add_argument(
        "--patience",
        help="Number of checks with no improvement after which training will be stopped. Under the default configuration, one check happens after every training epoch",
        type=utils_functions.is_positive_int,
    )

    user_group.add_argument(
        "--batch-size",
        type=int,
        default=config_defaults.DEFAULT_BATCH_SIZE,
    )
    user_group.add_argument(
        "--sampling-rate",
        type=int,
        default=config_defaults.DEFAULT_SAMPLING_RATE,
    )

    user_group.add_argument(
        "--scheduler",
        default=SchedulerType.PLATEAU,
        type=SchedulerType,
        choices=list(SchedulerType),
    )

    user_group.add_argument(
        "--optimizer",
        default=OptimizerType.ADAMW,
        type=str,
        choices=list(OptimizerType),
    )

    user_group.add_argument(
        "--epochs",
        default=config_defaults.DEFAULT_EPOCHS,
        type=utils_functions.is_positive_int,
        help="Maximum number epochs. Default number of epochs in other cases is 1000.",
    )

    args = parser.parse_args()

    """Separate Namespace into two Namespaces"""
    args_dict: dict[str, argparse.Namespace] = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        if group.title:
            args_dict[group.title] = argparse.Namespace(**group_dict)

    args, pl_args = args_dict[ARGS_GROUP_NAME], args_dict["pl.Trainer"]

    """User arguments which override PyTorch Lightning arguments"""
    if args.quick:
        pl_args.limit_train_batches = 4
        pl_args.limit_val_batches = 4
        pl_args.limit_test_batches = 4
        pl_args.log_every_n_steps = 1
        args.batch_size = 2
    if args.metric and not args.metric_mode:
        raise argparse.ArgumentError(args.metric, "can't pass --metric without passing --metric-mode")

    return args, pl_args


if __name__ == "__main__":
    pass
