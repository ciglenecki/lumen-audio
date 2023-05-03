from collections.abc import Iterator  # Python >=3.9
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.argparse_with_config import ArgParseWithConfig
from src.config.config_defaults import ConfigDefault
from src.train.inference_utils import (
    aggregate_inference_loops,
    get_inference_datamodule,
    get_inference_model_objs,
    validate_inference_args,
)
from src.train.metrics import get_metrics


def main(args, config: ConfigDefault):
    validate_inference_args(config)
    device = torch.device(args.device)
    model, model_config, audio_transform = get_inference_model_objs(
        config, args, device
    )
    datamodule = get_inference_datamodule(config, audio_transform, model_config)
    data_loader = datamodule.train_dataloader()
    result = aggregate_inference_loops(device, model, datamodule, data_loader)

    y_pred = torch.stack(result.y_pred)
    y_pred_file = torch.stack(result.y_pred_file)
    y_true = torch.stack(result.y_true)
    y_true_file = torch.stack(result.y_true_file)

    metric_dict = get_metrics(
        y_pred=y_pred,
        y_true=y_true,
        num_labels=config.num_labels,
        return_per_instrument=True,
    )
    metric_dict_file = get_metrics(
        y_pred=y_pred_file,
        y_true=y_true_file,
        num_labels=config.num_labels,
        return_per_instrument=True,
    )
    return metric_dict, metric_dict_file, y_pred, y_pred_file


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
