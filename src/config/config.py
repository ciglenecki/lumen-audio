"""Global config file.

To see the list of all arguments call `pyhton3 src/train.py -h`
"""

from __future__ import annotations

import argparse
from operator import attrgetter
from pathlib import Path

import configargparse
import pytorch_lightning as pl
import torch

import src.config.defaults as defaults
import src.utils.utils_functions as utils_functions
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


class SortingHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Alphabetically sort -h."""

    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter("option_strings"))
        super().add_arguments(actions)


# Intialize parser and it's groups
ARGS_GROUP_NAME = "General arguments"
parser = configargparse.get_argument_parser(formatter_class=SortingHelpFormatter)

lightning_parser = pl.Trainer.add_argparse_args(parser)
lightning_parser.set_defaults(
    log_every_n_steps=defaults.DEFAULT_LOG_EVERY_N_STEPS,
    epochs=defaults.DEFAULT_EPOCHS,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=-1,  # use all devices
)
user_group = parser.add_argument_group(ARGS_GROUP_NAME)

user_group.add_argument(
    "--config",
    is_config_file=True,
    help="YAML config file path. Useful for inference or overriding.",
)

user_group.add_argument(
    "--dataset-fraction",
    metavar="float",
    default=defaults.DEFAULT_DATASET_FRACTION,
    type=utils_functions.is_between_0_1,
    help="Reduce each dataset split (train, val, test) by a fraction.",
)

user_group.add_argument(
    "--num-workers",
    metavar="int",
    default=defaults.DEFAULT_NUM_WORKERS,
    type=utils_functions.is_positive_int,
    help="Number of workers",
)

user_group.add_argument(
    "--num-labels",
    metavar="int",
    default=defaults.DEFAULT_NUM_LABELS,
    type=utils_functions.is_positive_int,
    help="Total number of possible lables",
)

user_group.add_argument(
    "--lr",
    default=defaults.DEFAULT_LR,
    type=float,
    metavar="float",
    help="Learning rate",
)

user_group.add_argument(
    "--lr-warmup",
    type=float,
    metavar="float",
    help="warmup learning rate",
    default=defaults.DEFAULT_LR_WARMUP,
)

user_group.add_argument(
    "--lr-onecycle-max",
    type=float,
    metavar="float",
    help="Maximum lr OneCycle scheduler reaches",
    default=defaults.DEFAULT_LR_ONECYCLE_MAX,
)

user_group.add_argument(
    "--weight-decay",
    type=float,
    metavar="float",
    help="Maximum lr OneCycle scheduler reaches",
    default=defaults.DEFAULT_WEIGHT_DECAY,
)


user_group.add_argument(
    "--train-dirs",
    metavar="dir",
    nargs="+",
    type=parse_dataset_enum_dirs,
    help="Dataset root directories that will be used for training in the following format: --train-dirs irmas:/path/to openmic:/path/to",
    default=defaults.DEFAULT_TRAIN_DIRS,
)

user_group.add_argument(
    "--val-dirs",
    metavar="dir",
    nargs="+",
    type=parse_dataset_enum_dirs,
    help="Dataset root directories that will be used for validation in the following format: --val-dirs irmas:/path/to openmic:/path/to",
    default=defaults.DEFAULT_VAL_DIRS,
)

user_group.add_argument(
    "--train-override-csvs",
    metavar="file.csv file2.csv",
    nargs="+",
    type=Path,
    help="CSV files with columns 'filename, sax, gac, org, ..., cla' where filename is path and each instrument is either 0 or 1",
)

user_group.add_argument(
    "--output-dir",
    metavar="dir",
    type=Path,
    help="Output directory of the model and report file.",
    default=defaults.PATH_MODELS,
)

user_group.add_argument(
    "--pretrained",
    help="Use a pretrained model loaded from the web.",
    action="store_true",
    default=defaults.DEFAULT_PRETRAINED,
)

user_group.add_argument(
    "--freeze-train-bn",
    help="If true, the batch norm will be trained even if module is frozen.",
    action="store_true",
    default=defaults.DEFAULT_FREEZE_TRAIN_BN,
)
user_group.add_argument(
    "--normalize-audio",
    help="Normalize audio to [-1, 1]",
    action="store_true",
    default=defaults.DEFAULT_NORMALIZE_AUDIO,
)
user_group.add_argument(
    "--train-only-dataset",
    help="Use only the train portion of the dataset and split it 0.8 0.2",
    action="store_true",
    default=defaults.DEFAULT_ONLY_TRAIN_DATASET,
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
    default=defaults.DEFAULT_CHECK_ON_TRAIN_EPOCH_END,
)

user_group.add_argument(
    "--save-on-train-epoch-end",
    action="store_true",
    default=defaults.DEFAULT_SAVE_ON_TRAIN_EPOCH_END,
    help="Whether to run checkpointing at the end of the training epoch.",
)

user_group.add_argument(
    "--metric",
    type=OptimizeMetric.from_string,
    help="Metric which the model will optimize for.",
    default=defaults.DEFAULT_OPTIMIZE_METRIC,
    choices=list(OptimizeMetric),
)

user_group.add_argument(
    "--metric-mode",
    type=MetricMode.from_string,
    help="Maximize or minimize the --metric.",
    default=defaults.DEFAULT_METRIC_MODE,
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
    "--augmentations",
    default=defaults.DEFAULT_AUGMENTATIONS,
    nargs="*",
    choices=list(SupportedAugmentations),
    type=SupportedAugmentations.from_string,
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
    help="For testing bugs. Simulates --limit_train_batches 2 --limit_val_batches 2 --limit_test_batches 2",
    action="store_true",
    default=False,
)

user_group.add_argument(
    "--use-weighted-train-sampler",
    help="Use weighted train sampler instead of a random one.",
    action="store_true",
    default=defaults.DEFAULT_USE_WEIGHTED_TRAIN_SAMPLER,
)

user_group.add_argument(
    "--ckpt",
    help=".ckpt file, automatically restores model, epoch, step, LR schedulers, etc...",
    metavar="<PATH>",
    type=str,
)

user_group.add_argument(
    "--early-stopping-metric-patience",
    help="Number of checks with no improvement after which training will be stopped. Under the default configuration, one check happens after every training epoch",
    metavar="int",
    default=defaults.DEFAULT_EARLY_STOPPING_METRIC_PATIENCE,
    type=utils_functions.is_positive_int,
)

user_group.add_argument(
    "--batch-size",
    metavar="int",
    type=int,
    default=defaults.DEFAULT_BATCH_SIZE,
)

user_group.add_argument(
    "--finetune-head-epochs",
    metavar="int",
    type=int,
    help="Epoch at which the backbone will be unfrozen.",
    default=defaults.DEFAULT_FINETUNE_HEAD_EPOCHS,
)

user_group.add_argument(
    "--sampling-rate",
    metavar="int",
    type=int,
    default=defaults.DEFAULT_SAMPLING_RATE,
)

user_group.add_argument(
    "--scheduler",
    default=SupportedScheduler.ONECYCLE,
    type=SupportedScheduler,
    choices=list(SupportedScheduler),
)

user_group.add_argument(
    "--optimizer",
    default=SupportedOptimizer.ADAMW,
    type=str,
    choices=list(SupportedOptimizer),
)

user_group.add_argument(
    "--epochs",
    metavar="int",
    default=defaults.DEFAULT_EPOCHS,
    type=utils_functions.is_positive_int,
    help="Number epochs. Works only if learning rate scheduler has fixed number of steps (onecycle, cosine...). It won't have an effect on 'reduce on palteau' lr scheduler.",
)

user_group.add_argument(
    "--bar-update",
    metavar="int",
    default=defaults.DEFUALT_TQDM_REFRESH,
    type=utils_functions.is_positive_int,
    help="Number of TQDM updates in one epoch.",
)


user_group.add_argument(
    "--pretrained-tag",
    default=defaults.DEFAULT_PRETRAINED_TAG,
    type=str,
    help="The string that denotes the pretrained weights used.",
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
    "--image-dim",
    metavar="height width",
    default=defaults.DEFAULT_IMAGE_DIM,
    type=tuple[int, int],
    help="The dimension to resize the image to.",
)

user_group.add_argument(
    "--log-per-instrument-metrics",
    help="Along with aggregated metrics, also log per instrument metrics.",
    action="store_true",
    default=defaults.DEFAULT_LOG_PER_INSTRUMENT_METRICS,
)
user_group.add_argument(
    "--no-log-per-instrument-metrics",
    help="Along with aggregated metrics, also log per instrument metrics.",
    action="store_false",
    dest="log_per_instrument_metrics",
)


user_group.add_argument(
    "--finetune-head",
    help="Performs head only finetuning for --finetune-head-epochs epochs with starting lr of --lr-warmup which eventually becomes --lr.",
    action="store_true",
    default=defaults.DEFAULT_FINETUNE_HEAD,
)

user_group.add_argument(
    "--loss-function",
    type=SupportedLossFunctions,
    choices=list(SupportedLossFunctions),
    help="Loss function",
    default=SupportedLossFunctions.CROSS_ENTROPY,
)

user_group.add_argument(
    "--loss-function-kwargs",
    type=dict,
    help="Loss function kwargs",
    default={},
)

user_group.add_argument(
    "--head",
    type=SupportedHeads,
    help="classifier head",
    choices=list(SupportedHeads),
    default=defaults.DEAFULT_HEAD,
)

user_group.add_argument(
    "--use-fluffy",
    help="Use multiple optimizers for Fluffy.",
    action="store_true",
    default=defaults.DEFAULT_USE_FLUFFY,
)

user_group.add_argument(
    "--use-multiple-optimizers",
    help="Use multiple optimizers for Fluffy. Each head will have it's own optimizer.",
    action="store_true",
    default=defaults.DEFAULT_USE_MULTIPLE_OPTIMIZERS,
)

user_group.add_argument(
    "--n-fft",
    metavar="int",
    type=int,
    default=defaults.DEFAULT_N_FFT,
)

user_group.add_argument(
    "--n-mels",
    metavar="int",
    type=int,
    default=defaults.DEFAULT_N_MELS,
)

user_group.add_argument(
    "--n-mfcc",
    metavar="int",
    type=int,
    default=defaults.DEFAULT_N_MFCC,
)

user_group.add_argument(
    "--hop-length",
    metavar="int",
    type=int,
    default=defaults.DEFAULT_HOP_LENGTH,
)

user_group.add_argument(
    "--max-audio-seconds",
    metavar="float",
    type=float,
    default=defaults.DEFAULT_MAX_AUDIO_SECONDS,
    help="Maximum number of seconds of audio which will be processed at one time.",
)
user_group.add_argument(
    "--skip-validation",
    action="store_true",
    default=defaults.DEFAULT_SKIP_VALIDATION,
    help="Skips validation part during training.",
)


args = parser.parse_args()

# Separate Namespace into two Namespaces
args_dict: dict[str, argparse.Namespace] = {}
for group in parser._action_groups:
    group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
    if group.title:
        args_dict[group.title] = argparse.Namespace(**group_dict)

args, pl_args = args_dict[ARGS_GROUP_NAME], args_dict["pl.Trainer"]

# User arguments which override PyTorch Lightning arguments
if args.quick:
    pl_args.limit_train_batches = 2
    pl_args.limit_val_batches = 2
    pl_args.limit_test_batches = 2
    pl_args.log_every_n_steps = 1
    args.dataset_fraction = 0.01
    args.batch_size = 2
    args.output_dir = defaults.PATH_MODELS_QUICK

if args.epochs:
    pl_args.max_epochs = args.epochs

# Additional argument checking
if args.metric and not args.metric_mode:
    raise InvalidArgument("can't pass --metric without passing --metric-mode")


if args.aug_kwargs is None:
    args.aug_kwargs = defaults.DEFAULT_AUGMENTATION_KWARSG
else:
    args.aug_kwargs = utils_functions.parse_kwargs(args.aug_kwargs)

if args.scheduler == SupportedScheduler.ONECYCLE and args.lr_onecycle_max is None:
    raise InvalidArgument(
        f"You have to pass the --lr-onecycle-max if you use the {args.scheduler}",
    )

if args.model != SupportedModels.WAV2VECCNN and args.use_multiple_optimizers:
    raise InvalidArgument(
        "You can't use mutliple optimizers if you are not using Fluffy!",
    )

# Dynamically set pretrained tag
if args.pretrained and args.pretrained_tag == defaults.DEFAULT_PRETRAINED_TAG:
    if args.model == SupportedModels.AST:
        args.pretrained_tag = defaults.DEFAULT_AST_PRETRAINED_TAG
    elif args.model in [SupportedModels.WAV2VECCNN, SupportedModels.WAV2VEC]:
        args.pretrained_tag = defaults.DEFAULT_WAV2VEC_PRETRAINED_TAG
    elif args.model in [
        SupportedModels.EFFICIENT_NET_V2_S,
        SupportedModels.EFFICIENT_NET_V2_M,
        SupportedModels.EFFICIENT_NET_V2_L,
        SupportedModels.RESNEXT50_32X4D,
        SupportedModels.RESNEXT101_32X8D,
        SupportedModels.RESNEXT101_64X4D,
    ]:
        args.pretrained_tag = defaults.DEFAULT_TORCH_CNN_PRETRAINED_TAG
    else:
        raise Exception("Shouldn't happen")

# Dynamically AST DSP attributes
if args.model == SupportedModels.AST:
    args.n_fft = defaults.DEFAULT_AST_N_FFT
    args.hop_length = defaults.DEFAULT_AST_HOP_LENGTH
    args.n_mels = defaults.DEFAULT_AST_N_MELS

if args.skip_validation:
    pl_args.limit_val_batches = 0
config = args
