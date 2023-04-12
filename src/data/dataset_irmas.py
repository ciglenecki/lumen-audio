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

import src.config.defaults as defaults
from src.config.config import config
from src.features.audio_to_ast import AudioTransformAST
from src.features.audio_transform_base import AudioTransformBase
from src.features.augmentations import SupportedAugmentations
from src.utils.utils_audio import ast_spec_to_audio, load_audio_from_file, play_audio
from src.utils.utils_dataset import (
    decode_instruments_idx,
    encode_drums,
    encode_genre,
    encode_instruments,
    multi_hot_encode,
)
from src.utils.utils_exceptions import InvalidDataException

# '*.(wav|mp3|flac)'
# glob_expression = f"*\.({'|'.join(defaults.DEFAULT_AUDIO_EXTENSIONS)})"
glob_expression = "*.wav"


class IRMASDatasetTrain(Dataset):
    all_instrument_indices = np.array(list(defaults.INSTRUMENT_TO_IDX.values()))

    def __init__(
        self,
        dataset_dir: Path = defaults.PATH_IRMAS_TRAIN,
        audio_transform: AudioTransformBase | None = None,
        num_classes=defaults.DEFAULT_NUM_LABELS,
        sampling_rate=config.sampling_rate,
        normalize_audio=config.normalize_audio,
        sum_two_samples: bool = False,
        concat_n_samples: int | None = None,
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
        self.sum_two_samples = sum_two_samples
        self.concat_n_samples = concat_n_samples
        self.instrument_idx_list: dict[str, list[int]] = {}
        self.train_override_csvs = train_override_csvs
        self._populate_dataset()

        assert (
            len(self.dataset) == defaults.DEFAULT_IRMAS_TRAIN_SIZE
        ), f"IRMAS train set should contain {defaults.DEFAULT_IRMAS_TRAIN_SIZE} samples"

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

        self.instrument_idx_list = {i.value: [] for i in defaults.InstrumentEnums}

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
        instrument = defaults.IDX_TO_INSTRUMENT[instrument_idx]
        random_sample_idx = random.choice(self.instrument_idx_list[instrument])
        return random_sample_idx

    def _get_negative_sample(self, original_label: np.ndarray):
        """Returns audio whose labels doesn't contain any label from the original labels.

        Example:
            original_label: [1,0,0,0,0,1]
            returns: negative_audio, [0,0,0,0,1,0]
        """

        negative_indices = self.all_instrument_indices[~original_label.astype(bool)]
        if len(negative_indices) > 0:
            negative_index = np.random.choice(negative_indices)
        else:
            negative_index = np.random.randint(0, defaults.DEFAULT_NUM_LABELS)

        random_sample_idx = self._get_random_sample_for_instrument(negative_index)
        negative_audio_path, negative_label = self.dataset[random_sample_idx]
        negative_audio, _ = load_audio_from_file(
            negative_audio_path,
            method="librosa",
            normalize=self.normalize_audio,
            target_sr=self.sampling_rate,
        )
        return negative_audio, negative_label

    def _sum_with_another_sample(
        self,
        audio: np.ndarray,
        labels: np.ndarray,
        other_audio: np.ndarray,
        other_label: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sum two examples from the dataset in the following way:

        - audio: sum and divide by 2 (mean)
        - labels: logical or, union of ones
        """
        audio = (audio + other_audio) / 2
        labels = np.logical_or(labels, other_label).astype(labels.dtype)
        return audio, labels

    def _concat_with_another_sample(
        self,
        audio: np.ndarray,
        labels: np.ndarray,
        other_audio: np.ndarray,
        other_label: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Concat two examples from the dataset in the following way:

        - audio: concat audios
        - labels: logical or, union of ones
        """
        audio = np.concatenate([audio, other_audio])
        labels = np.logical_or(labels, other_label).astype(labels.dtype)
        return audio, labels

    def _concat_and_sum(self, original_audio: np.ndarray, original_labels: np.ndarray):
        """Performs concaternation and summation of original audio and negative audio samples.
        Negative audio sample doesn't share original audio's labels.

        Args:
            original_audio
            labels: original sample's labels

        Example 1:

            concat_n_samples: 3
            sum_two_samples: False

            original waveform: __x__
            negative waveforms: __a__, __b__
            returns: |__x__|__a__|__b__|

        Example 2:

            concat_n_samples: None | 0 | 1
            sum_two_samples: True

            original waveform: __x__
            negative waveform: __a__
            returns: |__xa__|

        Example 1:

            concat_n_samples: 3
            sum_two_samples: True

            original waveform: __x__
            negative waveforms: __a__, __b__, ..., __e__

            |block1 |block2 |block3 |
            |___x___|___b___|___d___| (concated audio 1)
            |___a___|___c___|___e___| (concated audio 2)

            returns: |__xa__|__bc__|__de__| (summed audios)
        """
        if self.concat_n_samples is None and not self.sum_two_samples:
            return original_audio, original_labels

        if self.concat_n_samples > 1:
            final_audio = original_audio
            final_labels = original_labels

            # Handle first block sample case
            if self.sum_two_samples:
                negative_audio, negative_label = self._get_negative_sample(
                    original_labels
                )
                final_audio, final_labels = self._sum_with_another_sample(
                    final_audio, final_labels, negative_audio, negative_label
                )

            # Handle additional samples
            for _ in range(self.concat_n_samples - 1):
                # Get negative sample
                negative_audio, negative_labels = self._get_negative_sample(
                    final_labels
                )

                if self.sum_two_samples:
                    # Get another negative sample
                    labels_up_until_now = np.logical_or(
                        final_labels, negative_labels
                    ).astype(original_labels.dtype)

                    negative_audio_2, negative_labels_2 = self._get_negative_sample(
                        labels_up_until_now
                    )

                    # Stack negative samples in one block
                    negative_audio, negative_labels = self._sum_with_another_sample(
                        negative_audio,
                        negative_labels,
                        negative_audio_2,
                        negative_labels_2,
                    )

                final_audio, final_labels = self._concat_with_another_sample(
                    final_audio, final_labels, negative_audio, negative_labels
                )
            return final_audio, final_labels

        elif (
            self.concat_n_samples is None or self.concat_n_samples in [0, 1]
        ) and self.sum_two_samples:
            negative_audio, negative_label = self._get_negative_sample(original_labels)
            final_audio, final_labels = self._sum_with_another_sample(
                original_audio, original_labels, negative_audio, negative_label
            )
            return final_audio, final_labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        audio_path, labels = self.dataset[index]

        audio, _ = load_audio_from_file(
            audio_path,
            method="librosa",
            normalize=self.normalize_audio,
            target_sr=self.sampling_rate,
        )
        audio, labels = self._concat_and_sum(audio, labels)

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
        num_classes=defaults.DEFAULT_NUM_LABELS,
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
            len(self.dataset) == defaults.DEFAULT_IRMAS_TEST_SIZE
        ), f"IRMAS test set should contain {defaults.DEFAULT_IRMAS_TEST_SIZE} samples"

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
                    instrument_indices.append(defaults.INSTRUMENT_TO_IDX[instrument])

            labels = multi_hot_encode(
                instrument_indices,
                defaults.DEFAULT_NUM_LABELS,
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
