import functools
from pathlib import Path

import torch

from src.config.argparse_with_config import ArgParseWithConfig
from src.config.config_defaults import get_default_config


@functools.cache
def get_server_args():
    config_pl_args = ["--batch-size"]
    default_config = get_default_config()

    parser = ArgParseWithConfig(add_lightning_args=True, config_pl_args=config_pl_args)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_config.path_models,
        help="Path to model checkpoints.",
    )
    parser.add_argument("--host", type=str, default="localhost", help="Server host.")
    parser.add_argument("--port", type=int, default=8090, help="Server port.")
    parser.add_argument(
        "--hot-reload",
        action="store_true",
        default=False,
        help="Use hot reload, not recommanded.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="The device to be used eg. cuda:0.",
    )
    args, config, pl_args = parser.parse_args()
    return args, config, pl_args
