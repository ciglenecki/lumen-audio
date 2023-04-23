import os
import re
from itertools import chain
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from genericpath import isfile
from torch.utils.data import Dataset
from tqdm import tqdm

import src.config.config_defaults as config_defaults
from src.data.dataset_base import DatasetBase, DatasetGetItem
from src.features.audio_transform_base import AudioTransformBase
from src.utils.utils_exceptions import InvalidDataException

# '*.(wav|mp3|flac)'

config = config_defaults.default_config

glob_expressions = [f"*.{ext}" for ext in config.audio_file_extensions]


class InferenceDataset(DatasetBase):
    def __init__(
        self,
        dataset_path: Path,
        audio_transform: AudioTransformBase | None,
        sampling_rate: int,
        normalize_audio: bool,
    ):
        """_summary_

        Args:
            dataset_path: directory with the following structure:
                ├── waveform1.wav
                ├── waveform2.wav
                ├── avbcfd.wav
                ├── 4q59daxui.ogg
        """
        kwargs = dict(
            num_classes=0,
            sum_two_samples=False,
            concat_n_samples=None,
            train_override_csvs=None,
        )
        super().__init__(
            dataset_path=dataset_path,
            audio_transform=audio_transform,
            sampling_rate=sampling_rate,
            normalize_audio=normalize_audio,
            **kwargs,
        )  # sets self.dataset

    def create_dataset_list(self) -> list[tuple[Path, np.ndarray]]:
        """Reads audio and label files and creates tuples of (audio_path, one hot encoded label)"""
        dataset_list: list[tuple[Path, np.ndarray]] = []
        if self.dataset_path.is_file() and self.dataset_path.suffix == ".csv":
            df = pd.read_csv(self.dataset_path)
            for _, row in df.iterrows():
                filepath = Path(row["file"])
                labels = np.array([0])
                dataset_list.append((filepath, labels))

        elif self.dataset_path.is_dir():
            glob_generators = [
                self.dataset_path.rglob(glob_exp) for glob_exp in glob_expressions
            ]
            for item_idx, audio_path in tqdm(enumerate(chain(*glob_generators))):
                labels = np.array([0])
                dataset_list.append((audio_path, labels))
            if not dataset_list:
                raise InvalidDataException(
                    f"Path {self.dataset_path} is does not contain any audio files"
                )
        else:
            raise InvalidDataException(
                f"Path {self.dataset_path} is not a csv or a directory which contains audio files."
            )
        return dataset_list
