from pathlib import Path

import torch
from tqdm import tqdm

from src.config import config_defaults
from src.config.argparse_with_config import ArgParseWithConfig
from src.enums.enums import SupportedModels
from src.model.model import get_model
from src.utils.utils_model import print_modules


def parse_args():
    parser = ArgParseWithConfig()
    args, config, pl_args = parser.parse_args()
    return args, config, pl_args


if __name__ == "__main__":
    args, config, pl_args = parse_args()

    for m in list(SupportedModels):
        config.model = m
        config.pretrained_tag = config_defaults.DEFAULT_PRETRAINED_TAG_MAP[config.model]
        model = get_model(
            config, torch.nn.BCEWithLogitsLoss(**config.loss_function_kwargs)
        )
        print("Model", m)
        print_modules(model)
        print("\n\n\n\n")
