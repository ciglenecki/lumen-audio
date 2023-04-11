from __future__ import annotations

import os
import random
import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import src.config.config_defaults as config_defaults
from src.config.config import config
from src.features.audio_to_ast import AudioTransformAST
from src.features.audio_transform_base import AudioTransformBase
from src.features.augmentations import SupportedAugmentations
from src.utils.utils_audio import load_audio_from_file
from src.utils.utils_dataset import (
    decode_instruments_idx,
    encode_drums,
    encode_genre,
    encode_instruments,
    multi_hot_encode,
)
from src.utils.utils_exceptions import InvalidDataException

# '*.(wav|mp3|flac)'
# glob_expression = f"*\.({'|'.join(config_defaults.DEFAULT_AUDIO_EXTENSIONS)})"
glob_expression = "*.wav"


class IRMASDatasetTrain(Dataset):
    all_instrument_indices = np.array(list(config_defaults.INSTRUMENT_TO_IDX.values()))

    def __init__(
        self,
        dataset_dir: Path = config_defaults.PATH_IRMAS_TRAIN,
        audio_transform: AudioTransformBase | None = None,
        num_classes=config_defaults.DEFAULT_NUM_LABELS,
        sampling_rate=config.sampling_rate,
        normalize_audio=config.normalize_audio,
        concat_two_samples: bool = SupportedAugmentations.CONCAT_TWO
        in config.augmentations,
        train_override_csvs: list[Path] | None = config.train_override_csvs,
    ):
        """_summary_

        Args:
            dataset_dir: directory with the following structure:

                ├── cel
                │   ├── 008__[cel][nod][cla]0058__1.wav
                │   ├── 008__[cel][nod][cla]0058__2.wav
                │   ├── 008__[cel][nod][cla]0058__3.wav
                │   ├── 012__[cel][nod][cla]0043__1.wav
                |   ├── ...
                ├── cla
                ...
                └── voi
        """

        self.dataset: list[tuple[Path, np.ndarray]] = []
        self.dataset_dir = dataset_dir
        self.audio_transform = audio_transform
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.normalize_audio = normalize_audio
        self.concat_two_samples = concat_two_samples
        self.instrument_idx_list: dict[str, list[int]] = {}
        self.train_override_csvs = train_override_csvs
        self._populate_dataset()

        assert (
            len(self.dataset) == config_defaults.DEFAULT_IRMAS_TRAIN_SIZE
        ), f"IRMAS train set should contain {config_defaults.DEFAULT_IRMAS_TRAIN_SIZE} samples"

    def _populate_dataset(self):
        """Reads audio and label files and creates tuples of (audio_path, one hot encoded label)
        self.instrument_idx_list = {
            "guitar": [0, 3, 5, 9, 13, 15]
            "flute": [2,4,6,7,8]
        }
        """

        if self.train_override_csvs:
            dfs = [pd.read_csv(csv_path) for csv_path in self.train_override_csvs]
            df = pd.concat(dfs, ignore_index=True)
            df.set_index("filename", inplace=True)

        self.instrument_idx_list = {
            i.value: [] for i in config_defaults.InstrumentEnums
        }

        for item_idx, path in tqdm(enumerate(self.dataset_dir.rglob(glob_expression))):
            filename = str(path.stem)
            characteristics = re.findall(
                r"\[(.*?)\]", filename
            )  # 110__[org][dru][jaz_blu]1117__2 => ["org", "dru", "jaz_blue"]

            path_str = str(path)
            if self.train_override_csvs and path_str in df.index:  # override label
                inner_instrument_indices = np.where(df.loc[path_str])[0]
                item_instruments = df.columns[inner_instrument_indices]
            else:
                instrument = characteristics[0]
                item_instruments = [instrument]

            labels = encode_instruments(item_instruments)

            # drums, genre = None, None
            # if len(characteristics) == 2:
            #     _, genre = characteristics
            # elif len(characteristics) == 3:
            #     _, drums, genre = characteristics
            # else:
            #     raise InvalidDataException(filename)

            # drums_vector = encode_drums(drums)
            # genre_vector = encode_genre(genre)

            self.dataset.append((str(path), labels))

            for instrument in item_instruments:
                self.instrument_idx_list[instrument].append(item_idx)

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_random_sample_for_instrument(self, instrument_idx: int) -> int:
        """Returns a random sample which contains the instrument with index instrument_idx."""
        instrument = config_defaults.IDX_TO_INSTRUMENT[instrument_idx]
        random_sample_idx = random.choice(self.instrument_idx_list[instrument])
        return random_sample_idx

    def _get_negative_sample(self, original_label: np.ndarray):
        """Returns audio whose labels doesn't contain any label from the original labels.

        Example:
            original_label: [1,0,0,0,0,1]
            returns: other_audio, [0,0,0,0,1,0]
        """

        negative_indices = self.all_instrument_indices[original_label.astype(bool)]
        if len(negative_indices) > 0:
            negative_index = np.random.choice(negative_indices)
        else:
            negative_index = np.random.randint(0, config_defaults.DEFAULT_NUM_LABELS)

        random_sample_idx = self._get_random_sample_for_instrument(negative_index)
        other_audio_path, other_labels = self.dataset[random_sample_idx]
        other_audio, _ = load_audio_from_file(
            other_audio_path,
            method="librosa",
            normalize=self.normalize_audio,
            target_sr=self.sampling_rate,
        )
        return other_audio, other_labels

    def _sum_with_another_sample(
        self,
        audio: np.ndarray,
        labels: np.ndarray,
        other_audio: np.ndarray,
        other_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sum two examples from the dataset in the following way:

        - audio: sum and divide by 2 (mean)
        - labels: logical or, union of ones
        """
        audio = (audio + other_audio) / 2
        labels = np.logical_or(labels, other_labels).astype(labels.dtype)
        return audio, labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        audio_path, labels = self.dataset[index]

        audio, _ = load_audio_from_file(
            audio_path,
            method="librosa",
            normalize=self.normalize_audio,
            target_sr=self.sampling_rate,
        )

        if self.concat_two_samples:
            other_audio, other_labels = self._get_negative_sample(labels)
            audio, labels = self._sum_with_another_sample(
                audio, labels, other_audio, other_labels
            )

        if self.audio_transform is None:
            return audio, labels

        features = self.audio_transform.process(audio)
        labels = torch.tensor(labels).float()
        return features, labels


class IRMASDatasetTest(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        audio_transform: AudioTransformBase | None = None,
        num_classes=config_defaults.DEFAULT_NUM_LABELS,
        sampling_rate=config.sampling_rate,
        normalize_audio=config.normalize_audio,
    ):
        self.num_classes = num_classes
        self.audio_transform = audio_transform
        self.dataset: list[tuple[Path, np.ndarray]] = []
        self.dataset_dir = dataset_dir
        self.sampling_rate = sampling_rate
        self.normalize_audio = normalize_audio
        self._populate_dataset()

        assert (
            len(self.dataset) == config_defaults.DEFAULT_IRMAS_TEST_SIZE
        ), f"IRMAS test set should contain {config_defaults.DEFAULT_IRMAS_TEST_SIZE} samples"

    def _populate_dataset(self):
        """Reads audio and label files and creates tuples of (audio_path, one hot encoded label)"""
        for audio_file in tqdm(self.dataset_dir.rglob(glob_expression)):
            path_without_ext = os.path.splitext(audio_file)[0]
            txt_path = Path(path_without_ext + ".txt")

            if not txt_path.is_file():
                raise FileNotFoundError(
                    f"File {audio_file} doesn't have label file {txt_path}."
                )

            instrument_indices = []
            with open(txt_path) as f:
                for line in f:
                    instrument = line.rstrip("\n").replace("\t", "")
                    instrument_indices.append(
                        config_defaults.INSTRUMENT_TO_IDX[instrument]
                    )

            labels = multi_hot_encode(
                instrument_indices,
                config_defaults.DEFAULT_NUM_LABELS,
            )

            self.dataset.append((str(audio_file), labels, instrument))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        audio_path, labels, _ = self.dataset[index]

        audio, _ = load_audio_from_file(
            audio_path,
            method="librosa",
            normalize=self.normalize_audio,
            target_sr=self.sampling_rate,
        )

        if self.audio_transform is None:
            return audio, labels

        features = self.audio_transform.process(audio)

        labels = torch.tensor(labels).float()
        return features, labels


class InstrumentInference(Dataset):
    pass


if __name__ == "__main__":  # for testing only
    ds = IRMASDatasetTrain(audio_transform=AudioTransformAST())
    for i in range(0, 30):
        print(ds[i][1], ds[i][2])
    # import matplotlib.pyplot as plt

    # item = ds[0]
    # x, sr, y = item

    # filter_banks = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=10)
    # print(filter_banks.shape)

    # plt.figure(figsize=(25, 10))
    # librosa.display.specshow(filter_banks, sr=sr, x_axis="linear")
    # plt.colorbar(format="%+2.f")
    # plt.show()

    # mel_spectrogram = librosa.feature.melspectrogram(
    #     y=x, sr=sr, n_fft=2048, hop_length=512, n_mels=10
    # )
    # print(mel_spectrogram.shape)

    # log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    # print(log_mel_spectrogram.shape)

    # plt.figure(figsize=(25, 10))
    # librosa.display.specshow(log_mel_spectrogram, x_axis="time", y_axis="mel", sr=sr)
    # plt.colorbar(format="%+2.f")
    # plt.show()
