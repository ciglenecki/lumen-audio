from __future__ import annotations

import os
import random
import re
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import src.config.config_defaults as config_defaults
from src.features.audio_transform import AudioTransformAST, AudioTransformBase
from src.features.supported_augmentations import SupportedAugmentations
from src.utils.utils_audio import load_audio_from_file
from src.utils.utils_dataset import encode_drums, encode_genre, multi_hot_indices
from src.utils.utils_exceptions import InvalidDataException

# '*.(wav|mp3|flac)'
# glob_expression = f"*\.({'|'.join(config_defaults.DEFAULT_AUDIO_EXTENSIONS)})"
glob_expression = "*.wav"


class IRMASDatasetTrain(Dataset):
    all_instrument_indices = np.array(list(config_defaults.INSTRUMENT_TO_IDX.values()))

    def __init__(
        self,
        dataset_dirs: list[Path] = [config_defaults.PATH_IRMAS_TRAIN],
        audio_transform: AudioTransformBase | None = None,
        num_classes=config_defaults.DEFAULT_NUM_LABELS,
        sanity_checks=config_defaults.DEFAULT_SANITY_CHECKS,
        sampling_rate=config_defaults.DEFAULT_SAMPLING_RATE,
        normalize_audio=config_defaults.DEFAULT_NORMALIZE_AUDIO,
        concat_two_samples=SupportedAugmentations.CONCAT_TWO
        in config_defaults.DEFAULT_AUGMENTATIONS,
    ):
        """_summary_

        Args:
            dataset_dirs: directories which have the following structure:

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
        self.dataset_dirs = dataset_dirs
        self.audio_transform = audio_transform
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.normalize_audio = normalize_audio
        self.concat_two_samples = concat_two_samples
        self._populate_dataset()

        if sanity_checks:
            assert (
                len(self.dataset) == config_defaults.DEFAULT_IRMAS_TRAIN_SIZE
            ), f"IRMAS train set should contain {config_defaults.DEFAULT_IRMAS_TRAIN_SIZE} samples"

    def _populate_dataset(self):
        """Reads audio and label files and creates tuples of (audio_path, one hot encoded label)"""

        self.instrument_idx_list = {
            k.value: [] for k in config_defaults.InstrumentEnums
        }
        for item_idx, dataset_dir in enumerate(self.dataset_dirs):
            for audio_path in tqdm(dataset_dir.rglob(glob_expression)):
                filename = str(audio_path.stem)
                characteristics = re.findall(
                    r"\[(.*?)\]", filename
                )  # 110__[org][dru][jaz_blu]1117__2 => ["org", "dru", "jaz_blue"]

                drums, genre = None, None
                if len(characteristics) == 2:
                    instrument, genre = characteristics
                elif len(characteristics) == 3:
                    instrument, drums, genre = characteristics
                else:
                    raise InvalidDataException(filename)

                drums_vector = encode_drums(drums)
                genre_vector = encode_genre(genre)

                instrument_idx = config_defaults.INSTRUMENT_TO_IDX[instrument]  # 2
                instrument_indices = []
                instrument_indices.append(instrument_idx)  # [2]

                labels = multi_hot_indices(
                    instrument_indices,
                    config_defaults.DEFAULT_NUM_LABELS,
                )
                labels = torch.tensor(labels).float()  # [0, 0, 1, 0...]

                self.dataset.append((str(audio_path), labels, instrument))
                self.instrument_idx_list[instrument].append(item_idx)

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_another_random_sample_idx(self, not_instrument_idx: int) -> int:
        """Returns a random sample whose label is NOT not_instrument_idx.

        Example:
            not_instrument_idx = 1
            random_instrument_idx = 3 # note: this index cannot be 1
            random_sample_idx ...
            return random_sample_idx
        Args:
            not_instrument_idx: Label instrument whose sampels won't be considered.

        Returns:
            random dataset index
        """
        random_instrument_idx = np.random.choice(
            self.all_instrument_indices[
                self.all_instrument_indices != not_instrument_idx
            ]
        )
        instrument = config_defaults.IDX_TO_INSTRUMENT[random_instrument_idx]
        random_sample_idx = random.choice(self.instrument_idx_list[instrument])
        return random_sample_idx

    def _sum_with_another_sample(
        self, audio: np.ndarray, labels: np.ndarray, instrument_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sum two examples from the dataset in the following way:

            - audio: sum and divide by 2
            - labels: logical or, union of ones
        Args:
            audio
            labels
            instrument_idx

        Returns:
            tuple[np.ndarray, np.ndarray]: audio, label
        """

        # Load random sample
        random_sample_idx = self._get_another_random_sample_idx(instrument_idx)
        other_audio_path, other_labels, _ = self.dataset[random_sample_idx]
        other_audio, _ = load_audio_from_file(
            other_audio_path,
            method="librosa",
            normalize=self.normalize_audio,
            target_sr=self.sampling_rate,
        )

        # Sum audio and labels
        audio = (audio + other_audio) / 2
        labels = np.logical_or(labels, other_labels).astype(labels.dtype)
        return audio, labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        audio_path, labels, instrument = self.dataset[index]
        instrument_idx = config_defaults.INSTRUMENT_TO_IDX[instrument]

        audio, _ = load_audio_from_file(
            audio_path,
            method="librosa",
            normalize=self.normalize_audio,
            target_sr=self.sampling_rate,
        )

        if self.concat_two_samples:
            audio, labels = self._sum_with_another_sample(audio, labels, instrument_idx)

        if self.audio_transform is None:
            return audio, labels, audio_path

        features = self.audio_transform.process(audio)
        return features, labels, audio_path


class IRMASDatasetTest(Dataset):
    def __init__(
        self,
        dataset_dirs: list[Path] = [config_defaults.PATH_IRMAS_TEST],
        num_classes=config_defaults.DEFAULT_NUM_LABELS,
        sanity_checks=config_defaults.DEFAULT_SANITY_CHECKS,
        audio_transform: AudioTransformBase | None = None,
        sampling_rate=config_defaults.DEFAULT_SAMPLING_RATE,
        normalize_audio=config_defaults.DEFAULT_NORMALIZE_AUDIO,
    ):
        self.num_classes = num_classes
        self.audio_transform = audio_transform
        self.dataset: list[tuple[Path, np.ndarray]] = []
        self.dataset_dirs = dataset_dirs
        self.sampling_rate = sampling_rate
        self.normalize_audio = normalize_audio
        self._populate_dataset()

        if sanity_checks:
            assert (
                len(self.dataset) == config_defaults.DEFAULT_IRMAS_TEST_SIZE
            ), f"IRMAS test set should contain {config_defaults.DEFAULT_IRMAS_TEST_SIZE} samples"

    def _populate_dataset(self):
        """Reads audio and label files and creates tuples of (audio_path, one hot encoded label)"""
        for dataset_dir in self.dataset_dirs:
            for audio_file in tqdm(dataset_dir.rglob(glob_expression)):
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

                labels = multi_hot_indices(
                    instrument_indices,
                    config_defaults.DEFAULT_NUM_LABELS,
                )
                labels = torch.tensor(labels).float()

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
            return audio, labels, audio_path

        features = self.audio_transform.process(audio)
        return features, labels, audio_path


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
