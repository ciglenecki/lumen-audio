import argparse
import json
import os
import re
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Audio
from skmultilearn.model_selection.iterative_stratification import (
    iterative_train_test_split,
)
from tqdm import tqdm

from src.config.argparse_with_config import ArgParseWithConfig
from src.config.config_defaults import (
    INSTRUMENT_TO_FULLNAME,
    InstrumentEnums,
    get_default_config,
)
from src.utils.utils_dataset import encode_instruments

config = get_default_config()


def parse_args():
    parser = ArgParseWithConfig()
    parser.add_argument(
        "--relevance",
        type=float,
        help="fraction of the entire dataset to be stored as test.",
        default=0.8,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Openmic directory path.",
        default=config.path_openmic,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="fraction of the entire dataset to be stored as test.",
        default=config.path_data,
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        help="fraction of the entire dataset to be stored as test.",
        default=config.path_figures,
    )
    args, _, _ = parser.parse_args()
    return args


def create_filename(sample_key):
    subdir = sample_key[:3]
    filepath = Path(config.path_openmic, "audio", subdir, sample_key + ".ogg")
    filepath = str(filepath)
    return filepath


def main():
    args = parse_args()
    csv_path = Path(config.path_openmic, "openmic-2018-aggregated-labels.csv")
    df = pd.read_csv(csv_path)
    mask_guitars = df["instrument"].isin(["guitar"])
    mask_relevance = df["relevance"] > args.relevance
    df = df.loc[mask_guitars & mask_relevance, :]
    df["file"] = df["sample_key"]
    df["file"] = df["file"].apply(create_filename)

    df.to_csv(
        Path(
            config.path_openmic,
            f"guitars_only_n_{len(df)}_relevance_{args.relevance}.csv",
        ),
        index=False,
    )


if __name__ == "__main__":
    main()
