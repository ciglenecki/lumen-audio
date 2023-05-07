import argparse
from itertools import chain
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from src.config.config_defaults import AUDIO_EXTENSIONS


# Create argparse with path to dataset
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to dataset to check for bad files",
    )
    args = parser.parse_args()
    glob_expressions = [f"*.{ext}" for ext in AUDIO_EXTENSIONS]
    glob_generators = [
        args.dataset_path.rglob(glob_exp) for glob_exp in glob_expressions
    ]
    eps = 0.001
    percentage_of_audio = 0.8

    for item_idx, p in enumerate(chain(*glob_generators)):
        try:
            a, _ = librosa.load(p)
        except Exception:
            print(p)
            continue
        if np.mean(np.absolute(a) < eps) > percentage_of_audio:
            print(p)


if __name__ == "__main__":
    main()
