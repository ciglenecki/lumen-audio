import argparse
from itertools import chain
from pathlib import Path

import librosa
import numpy as np

from src.config.config_defaults import AUDIO_EXTENSIONS


# Create argparse with path to dataset
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to dataset to check for bad files",
    )
    parser.add_argument(
        "eps",
        type=float,
        help="Audio files with mean absolute value less than eps will be considered bad",
        default=0.001,
    )
    parser.add_argument(
        "percentage_of_audio",
        type=float,
        help="Percentage of audio less than eps to be considered 'a bad file'",
        default=0.8,
    )
    args = parser.parse_args()

    eps = args.eps
    percentage_of_audio = args.percentage_of_audio
    glob_expressions = [f"*.{ext}" for ext in AUDIO_EXTENSIONS]
    glob_generators = [
        args.dataset_path.rglob(glob_exp) for glob_exp in glob_expressions
    ]

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
