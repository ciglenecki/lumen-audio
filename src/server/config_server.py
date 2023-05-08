import functools
from pathlib import Path

import torch
from numpy import require

from src.config.argparse_with_config import ArgParseWithConfig


@functools.cache
def get_server_args():
    config_pl_args = ["--batch-size", "--num-workers", "--log-per-instrument-metrics"]

    parser = ArgParseWithConfig(add_lightning_args=True, config_pl_args=config_pl_args)
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Directory which to model checkpoints (.ckpt files)",
        required=True,
    )
    parser.add_argument(
        "--add-instrument-metrics",
        action="store_true",
        help="Add instrument metrics to the response. This will increase number of items in API response size by (instrument count * number of metrics)",
        default=False,
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

    parser_help = parser.format_help()
    args, config, pl_args = parser.parse_args()

    if args.add_instrument_metrics:
        config.log_per_instrument_metrics = True
    else:
        config.log_per_instrument_metrics = False

    return args, config, pl_args, parser_help
