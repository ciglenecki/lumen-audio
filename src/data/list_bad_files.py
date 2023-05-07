import argparse
from itertools import chain
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

import src.config.config_defaults as config_defaults
from src.config.config_defaults import AUDIO_EXTENSIONS
from src.data.dataset_base import DatasetBase, DatasetInternalItem
from src.utils.utils_dataset import encode_instruments, multi_hot_encode


# Create argparse with path to dataset
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Path to dataset to check for bad files",
        required=True,
    )
    args = parser.parse_args()
    glob_expressions = [f"*.{ext}" for ext in AUDIO_EXTENSIONS]
    glob_generators = [
        args.dataset_path.rglob(glob_exp) for glob_exp in glob_expressions
    ]

    for item_idx, p in tqdm(enumerate(chain(*glob_generators))):
        try:
            a, _ = librosa.load(p)
        except Exception:
            print(p)
            continue
        if np.all(np.absolute(a) < 0.01):
            print(p)


if __name__ == "__main__":
    main()
