"""Input arguments for the train.py file.

To see the list of all arguments call `pyhton3 src/train.py -h`
"""

from __future__ import annotations

import argparse

import pytorch_lightning as pl
import torch

import src.config.config_defaults as config_defaults
import src.utils.utils_functions as utils_functions
from src.features.audio_transform import AudioTransforms, SupportedSpecAugs
from src.model.model import SupportedModels
from src.model.optimizers import OptimizerType, SchedulerType
from src.utils.utils_train import MetricMode, OptimizeMetric

ARGS_GROUP_NAME = "General arguments"


def parse_args_train() -> tuple[argparse.Namespace, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    lightning_parser = pl.Trainer.add_argparse_args(parser)
    lightning_parser.set_defaults(
        log_every_n_steps=config_defaults.DEFAULT_LOG_EVERY_N_STEPS,
        epochs=config_defaults.DEFAULT_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,  # use all devices
    )

    user_group = parser.add_argument_group(ARGS_GROUP_NAME)

    user_group.add_argument(
        "--dataset-fraction",
        metavar="float",
        default=config_defaults.DEFAULT_DATASET_FRACTION,
        type=utils_functions.is_between_0_1,
        help="Reduce each dataset split (train, val, test) by a fraction.",
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
        default=config_defaults.DEFAULT_NUM_LABELS,
        type=utils_functions.is_positive_int,
        help="Total number of possible lables",
    )

    user_group.add_argument(
        "--lr",
        default=config_defaults.DEFAULT_LR,
        type=float,
        metavar="float",
        help="Learning rate",
    )

    user_group.add_argument(
        "--warmup-lr",
        type=float,
        metavar="float",
        help="warmup learning rate",
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
        "--output-dir",
        metavar="dir",
        type=str,
        help="Output directory of the model and report file.",
        default=config_defaults.PATH_MODELS,
    )

    user_group.add_argument(
        "--pretrained",
        help="Use the pretrained model.",
        action="store_true",
        default=config_defaults.DEFAULT_PRETRAINED,
    )

    user_group.add_argument(
        "--normalize-audio",
        help="Normalize audio to [-1, 1]",
        action="store_true",
        default=config_defaults.DEFAULT_NORMALIZE_AUDIO,
    )

    user_group.add_argument(
        "--drop-last",
        help="Drop last sample if the size of the sample is smaller than batch size",
        action="store_true",
        default=True,
    )

    user_group.add_argument(
        "--check-on-train-epoch-end",
        help="Whether to run early stopping at the end of the training epoch.",
        action="store_true",
        default=config_defaults.DEFAULT_CHECK_ON_TRAIN_EPOCH_END,
    )

    user_group.add_argument(
        "--save-on-train-epoch-end",
        action="store_true",
        default=config_defaults.DEFAULT_SAVE_ON_TRAIN_EPOCH_END,
        help="Whether to run checkpointing at the end of the training epoch.",
    )

    user_group.add_argument(
        "--metric",
        type=OptimizeMetric.from_string,
        help="Metric which the model will optimize for.",
        default=config_defaults.DEFAULT_OPTIMIZE_METRIC,
        choices=list(OptimizeMetric),
    )

    user_group.add_argument(
        "--metric-mode",
        type=MetricMode.from_string,
        help="Maximize or minimize the --metric.",
        default=config_defaults.DEFAULT_METRIC_MODE,
        choices=list(MetricMode),
    )

    user_group.add_argument(
        "--model",
        type=SupportedModels.from_string,
        choices=list(SupportedModels),
        help="Models used for training.",
        required=True,
    )

    user_group.add_argument(
        "--audio-transform",
        type=AudioTransforms.from_string,
        choices=list(AudioTransforms),
        help="Transformation which will be performed on audio and labels",
        required=True,
    )

    user_group.add_argument(
        "--spectrogram-augmentations",
        default=None,
        nargs="+",
        choices=list(SupportedSpecAugs),
        type=SupportedSpecAugs.from_string,
        help="Transformation which will be performed on audio and labels",
    )

    user_group.add_argument(
        "--aug-kwargs",
        default=None,
        nargs="+",
        type=str,
        help="Arguments are split by space, mutiple values are sep'ed by comma (,). E.g. stretch_factors=0.8,1.2 freq_mask_param=30 time_mask_param=30 hide_random_pixels_p=0.5",
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
        metavar="<PATH>",
        type=str,
    )

    user_group.add_argument(
        "--patience",
        help="Number of checks with no improvement after which training will be stopped. Under the default configuration, one check happens after every training epoch",
        metavar="int",
        default=config_defaults.DEFAULT_EARLY_STOPPING_NO_IMPROVEMENT_EPOCHS,
        type=utils_functions.is_positive_int,
    )

    user_group.add_argument(
        "--batch-size",
        metavar="int",
        type=int,
        default=config_defaults.DEFAULT_BATCH_SIZE,
    )

    user_group.add_argument(
        "--unfreeze-at-epoch",
        metavar="int",
        type=int,
    )

    user_group.add_argument(
        "--sampling-rate",
        metavar="int",
        type=int,
        default=config_defaults.DEFAULT_SAMPLING_RATE,
    )

    user_group.add_argument(
        "--scheduler",
        default=SchedulerType.ONECYCLE,
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
        metavar="int",
        default=config_defaults.DEFAULT_EPOCHS,
        type=utils_functions.is_positive_int,
        help="Maximum number epochs. Default number of epochs in other cases is 1000.",
    )

    user_group.add_argument(
        "--bar-update",
        metavar="int",
        default=config_defaults.DEFUALT_TQDM_REFRESH,
        type=utils_functions.is_positive_int,
        help="Number of TQDM updates in one epoch.",
    )

    user_group.add_argument(
        "--backbone-after",
        metavar="str",
        type=str,
        help="Name of the submodule after which the all submodules are considered as backbone, e.g. layer.11.dense",
    )

    user_group.add_argument(
        "--head-after",
        metavar="str",
        type=str,
        help="Name of the submodule after which the all submodules are considered as head, e.g. classifier.dense",
    )

    user_group.add_argument(
        "--dim",
        default=config_defaults.DEFAULT_DIM,
        type=tuple[int, int],
        help="The dimension to resize the image to.",
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
        pl_args.limit_train_batches = 3
        pl_args.limit_val_batches = 2
        pl_args.limit_test_batches = 2
        pl_args.log_every_n_steps = 1
        args.dataset_fraction = 0.01
        args.batch_size = 2

    if args.epochs:
        pl_args.max_epochs = args.epochs

    """Additional argument checking"""
    if args.metric and not args.metric_mode:
        raise Exception("can't pass --metric without passing --metric-mode")

    if bool(args.warmup_lr) != bool(args.unfreeze_at_epoch):
        raise Exception(
            "--warmup-lr and --unfreeze-at-epoch have to be passed together",
        )

    if args.aug_kwargs is None:
        args.aug_kwargs = {}
    else:
        args.aug_kwargs = utils_functions.parse_kwargs(args.aug_kwargs)

    return args, pl_args


if __name__ == "__main__":
    pass