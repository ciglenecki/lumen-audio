from collections.abc import Iterator  # Python >=3.9
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.argparse_with_config import ArgParseWithConfig
from src.config.config_defaults import ConfigDefault
from src.data.datamodule import OurDataModule
from src.features.audio_transform import AudioTransformBase, get_audio_transform
from src.features.chunking import collate_fn_feature
from src.model.model import SupportedModels, get_model, model_constructor_map
from src.model.model_base import ModelBase
from src.train.inference_utils import (
    StepResult,
    aggregate_inference_loops,
    get_inference_datamodule,
    get_inference_model_objs,
    validate_inference_args,
)
from src.utils.utils_exceptions import InvalidArgument, UnsupportedModel


def main(args, config: ConfigDefault):
    validate_inference_args(config)
    device = torch.device(args.device)
    model, model_config, audio_transform = get_inference_model_objs(
        config, args, device
    )
    datamodule = get_inference_datamodule(config, audio_transform, model_config)
    data_loader = datamodule.train_dataloader()
    result = aggregate_inference_loops(device, model, datamodule, data_loader)


if __name__ == "__main__":
    parser = ArgParseWithConfig()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to be used eg. cuda:0.",
    )
    args, config, _ = parser.parse_args()
    main(args, config)
