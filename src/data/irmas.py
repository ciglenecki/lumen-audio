from __future__ import annotations

import os
import re
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import src.config.config_defaults as config_defaults
from src.features.audio_transform import AudioTransformAST, AudioTransformBase
from src.utils.utils_audio import load_audio_from_file
from src.utils.utils_dataset import encode_drums, encode_genre, multi_hot_indices
from src.utils.utils_exceptions import InvalidDataException

# '*.(wav|mp3|flac)'
# glob_expression = f"*\.({'|'.join(config_defaults.DEFAULT_AUDIO_EXTENSIONS)})"
glob_expression = "*.wav"


class IRMASDatasetTrain(Dataset):
    def __init__(
        self,
        dataset_dirs: list[Path] = [config_defaults.PATH_TRAIN],
        audio_transform: AudioTransformBase | None = None,
        num_classes=config_defaults.DEFAULT_NUM_LABELS,
        sanity_checks=config_defaults.DEFAULT_SANITY_CHECKS,
        sampling_rate=config_defaults.DEFAULT_SAMPLING_RATE,
        normalize_audio=config_defaults.DEFAULT_NORMALIZE_AUDIO,
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
        self.sampling_rate = sampling_rate
        self.audio_transform = audio_transform
        self.num_classes = num_classes
        self.normalize_audio = normalize_audio
        self._populate_dataset()

        if sanity_checks:
            assert (
                len(self.dataset) == config_defaults.DEFAULT_IRMAS_TRAIN_SIZE
            ), f"IRMAS train set should contain {config_defaults.DEFAULT_IRMAS_TRAIN_SIZE} samples"

    def _populate_dataset(self):
        """Reads audio and label files and creates tuples of (audio_path, one hot encoded label)"""

        for dataset_dir in self.dataset_dirs:
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

                instrument_indices = []
                instrument_indices.append(config_defaults.INSTRUMENT_TO_IDX[instrument])

                labels = multi_hot_indices(
                    instrument_indices,
                    config_defaults.DEFAULT_NUM_LABELS,
                )
                labels = torch.tensor(labels).float()

                self.dataset.append((audio_path, labels, drums_vector, genre_vector))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        audio_path, labels, drums_vector, genre_vector = self.dataset[index]

        audio, original_sr = load_audio_from_file(
            audio_path, method="librosa", normalize=self.normalize_audio
        )

        if self.audio_transform is None:
            return audio, labels

        spectrogram = self.audio_transform.process(
            audio=audio,
            original_sr=original_sr,
        )

        return spectrogram, labels


class IRMASDatasetTest(Dataset):
    def __init__(
        self,
        dataset_dirs: list[Path] = [config_defaults.PATH_TEST],
        num_classes=config_defaults.DEFAULT_NUM_LABELS,
        sanity_checks=config_defaults.DEFAULT_SANITY_CHECKS,
        audio_transform: AudioTransformBase | None = None,
        sampling_rate=config_defaults.DEFAULT_SAMPLING_RATE,
    ):
        self.num_classes = num_classes
        self.audio_transform = audio_transform
        self.dataset: list[tuple[Path, np.ndarray]] = []
        self.dataset_dirs = dataset_dirs
        self.sampling_rate = sampling_rate
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

                self.dataset.append((audio_file, labels))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        audio_path, labels = self.dataset[index]
        audio, original_sr = load_audio_from_file(
            audio_path, method="librosa", normalize=self.normalize_audio
        )
        if self.audio_transform is None:
            return audio, labels

        spectrogram = self.audio_transform.process(
            audio=audio,
            original_sr=original_sr,
        )
        return spectrogram, labels


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
