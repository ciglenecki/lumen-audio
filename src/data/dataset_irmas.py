from __future__ import annotations

import os
import re
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

glob_expressions = [f"*.{ext}" for ext in AUDIO_EXTENSIONS]


class IRMASDatasetTrain(DatasetBase):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Args:
            dataset_path: directory with the following structure:

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

        super().__init__(*args, **kwargs)

        if len(self.dataset_list) == config_defaults.DEFAULT_IRMAS_TRAIN_SIZE:
            print(
                f"Warning: IRMAS train set usually contains {config_defaults.DEFAULT_IRMAS_TRAIN_SIZE} samples"
            )

    def create_dataset_list(self) -> list[DatasetInternalItem]:
        """Reads audio and label files and creates tuples of (audio_path, one hot encoded label)
        self.instrument_idx_list = {
            "guitar": [0, 3, 5, 9, 13, 15]
            "flute": [2,4,6,7,8]
        }
        """
        dataset_list = []
        glob_generators = [
            self.dataset_path.rglob(glob_exp) for glob_exp in glob_expressions
        ]
        for item_idx, path in tqdm(enumerate(chain(*glob_generators))):
            instrument = path.parent.name

            # filename = str(path.stem)
            # characteristics = re.findall(
            #     r"\[(.*?)\]", filename
            # )  # 110__[org][dru][jaz_blu]1117__2 => ["org", "dru", "jaz_blue"]
            # instrument = characteristics[0]

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

            dataset_list.append((path, labels))
        return dataset_list


class IRMASDatasetTest(DatasetBase):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """_summary_

        Args:
            dataset_path: directory with the following structure:
                ├── 008__[cel][nod][cla]0058__1.wav
                ├── 008__[cel][nod][cla]0058__1.txt
                ├── 008__[cel][nod][cla]0058__2.wav
                ├── 008__[cel][nod][cla]0058__2.txt
                ├── 008__[cel][nod][cla]0058__3.wav
                ├── 008__[cel][nod][cla]0058__3.txt
                ├── 012__[cel][nod][cla]0043__1.wav
                ├── 012__[cel][nod][cla]0043__1.txt
                ├── ...
        """
        super().__init__(*args, **kwargs)

        if len(self.dataset_list) == config_defaults.DEFAULT_IRMAS_TEST_SIZE:
            print(
                "WARNING:",
                f"IRMAS test set should contain {config_defaults.DEFAULT_IRMAS_TEST_SIZE} samples",
            )

    def create_dataset_list(self) -> list[tuple[Path, np.ndarray]]:
        """Reads audio and label files and creates tuples of (audio_path, one hot encoded label)"""
        dataset_list: list[tuple[Path, np.ndarray]] = []
        glob_generators = [
            self.dataset_path.rglob(glob_exp) for glob_exp in glob_expressions
        ]

        for audio_file in tqdm(chain(*glob_generators)):
            path_without_ext = os.path.splitext(audio_file)[0]
            txt_path = Path(path_without_ext + ".txt")

            if not txt_path.is_file():
                raise FileNotFoundError(
                    f"File {audio_file} doesn't have label file {txt_path}. Please fix your dataset {self.dataset_path}"
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

            dataset_list.append((audio_file, labels))
        return dataset_list


class IRMASDatasetPreTrain(IRMASDatasetTrain):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index: int):
        features, _, _ = super().__getitem__(index)

        valid_indices = np.arange(len(self.dataset))
        valid_indices = valid_indices[valid_indices != index]
        random_index = np.random.choice(valid_indices, 1)[0]
        features_random, _, _ = super().__getitem__(random_index)

        return features, features_random


def test_sum_and_concat():
    config = config_defaults.get_default_config()
    dataset = IRMASDatasetTrain(
        dataset_path=config.path_irmas_train,
        audio_transform=None,
        normalize_audio=False,
        concat_n_samples=None,
        sum_n_samples=None,
        sampling_rate=config.sampling_rate,
        train_override_csvs=None,
        num_classes=config.num_labels,
    )

    dataset.concat_n_samples = 4
    dataset.sum_n_samples = 2

    sr = 16_000
    audio_1, _ = librosa.load("data/irmas/train/cel/[cel][cla]0001__1.wav", sr=sr)
    audio_2, _ = librosa.load("data/irmas/train/cla/[cla][cla]0150__1.wav", sr=sr)
    audio_3, _ = librosa.load("data/irmas/train/flu/[flu][cla]0346__1.wav", sr=sr)
    audio_4, _ = librosa.load(
        "data/irmas/train/flu/008__[flu][nod][cla]0393__1.wav", sr=sr
    )
    # Simulate different legnth
    audio_1, audio_2, audio_3 = audio_1[5:], audio_2[3:], audio_3[2:]
    label_1 = encode_instruments(["cel"])
    label_2 = encode_instruments(["cla"])
    label_3 = encode_instruments(["flu"])
    label_4 = encode_instruments(["flu"])

    audios = [audio_1, audio_2, audio_3, audio_4]
    labels = [label_1, label_2, label_3, label_4]
    final_audio, final_labels = dataset.concat_and_sum(
        audios, labels, use_concat=True, use_sum=True
    )

    top = np.concatenate([audio_1, audio_2, audio_3, audio_4])
    top, bottom = np.array_split(top, 2)
    max_len = len(top)
    if len(bottom) != max_len:
        pad_width = (0, max_len - len(bottom))
        bottom = np.pad(bottom, pad_width, mode="constant", constant_values=0)
    test_audio = np.mean([top, bottom], axis=0)
    test_labels = np.logical_or.reduce(labels).astype(labels[0].dtype)
    print(test_audio.shape, final_audio.shape)
    assert len(test_audio.shape) == 1
    assert len(final_audio.shape) == 1
    assert np.all(np.isclose(test_audio, final_audio))
    assert np.all(np.isclose(test_labels, final_labels))


def test_sum_and_concat_3():
    config = config_defaults.get_default_config()
    dataset = IRMASDatasetTrain(
        dataset_path=config.path_irmas_train,
        audio_transform=None,
        normalize_audio=False,
        concat_n_samples=None,
        sum_n_samples=None,
        sampling_rate=config.sampling_rate,
        train_override_csvs=None,
        num_classes=config.num_labels,
    )

    dataset.concat_n_samples = 4
    dataset.sum_n_samples = 2

    sr = 16_000
    audio_1, _ = librosa.load("data/irmas/train/cel/[cel][cla]0001__1.wav", sr=sr)
    audio_2, _ = librosa.load("data/irmas/train/cla/[cla][cla]0150__1.wav", sr=sr)
    audio_3, _ = librosa.load("data/irmas/train/flu/[flu][cla]0346__1.wav", sr=sr)
    audio_4, _ = librosa.load(
        "data/irmas/train/flu/008__[flu][nod][cla]0393__1.wav", sr=sr
    )
    # Simulate different legnth
    audio_1, audio_2, audio_3 = audio_1[5:], audio_2[3:], audio_3[2:]
    label_1 = encode_instruments(["cel"])
    label_2 = encode_instruments(["cla"])
    label_3 = encode_instruments(["flu"])
    label_4 = encode_instruments(["flu"])

    audios = [audio_1, audio_2, audio_3, audio_4]
    labels = [label_1, label_2, label_3, label_4]
    final_audio, final_labels = dataset.concat_and_sum(
        audios, labels, use_concat=True, use_sum=True
    )

    top = np.concatenate([audio_1, audio_2, audio_3, audio_4])
    top, bottom = np.array_split(top, 2)
    max_len = len(top)

    if len(bottom) != max_len:
        pad_width = (0, max_len - len(bottom))
        bottom = np.pad(bottom, pad_width, mode="constant", constant_values=0)
    test_audio = np.mean([top, bottom], axis=0)
    test_labels = np.logical_or.reduce(labels).astype(labels[0].dtype)
    print(test_audio.shape, final_audio.shape)
    assert len(test_audio.shape) == 1
    assert len(final_audio.shape) == 1
    assert np.all(np.isclose(test_audio, final_audio))
    assert np.all(np.isclose(test_labels, final_labels))


def test_simple_sum():
    config = config_defaults.get_default_config()
    dataset = IRMASDatasetTrain(
        dataset_path=config.path_irmas_train,
        audio_transform=None,
        normalize_audio=False,
        concat_n_samples=None,
        sum_n_samples=None,
        sampling_rate=config.sampling_rate,
        train_override_csvs=None,
        num_classes=config.num_labels,
    )
    dataset.sum_n_samples = 2

    sr = 16_000
    audio_1, _ = librosa.load("data/irmas/train/cel/[cel][cla]0001__1.wav", sr=sr)
    audio_2, _ = librosa.load("data/irmas/train/cla/[cla][cla]0150__1.wav", sr=sr)
    # Simulate different legnth
    audio_1, audio_2 = audio_1[5:], audio_2[3:]
    label_1 = encode_instruments(["cel"])
    label_2 = encode_instruments(["cla"])

    audios = [audio_1, audio_2]
    labels = [label_1, label_2]
    final_audio, final_labels = dataset.concat_and_sum(
        audios, labels, use_concat=False, use_sum=True
    )

    audios = np.concatenate(audios)
    top, bottom = np.array_split(audios, 2)
    max_len = len(top)

    if len(bottom) != max_len:
        pad_width = (0, max_len - len(bottom))
        bottom = np.pad(bottom, pad_width, mode="constant", constant_values=0)

    test_audio = np.mean([top, bottom], axis=0)
    test_labels = np.logical_or.reduce(labels).astype(labels[0].dtype)
    assert len(test_audio.shape) == 1
    assert len(final_audio.shape) == 1
    assert np.all(np.isclose(test_audio, final_audio))
    assert np.all(np.isclose(test_labels, final_labels))


def test_simple_concat():
    config = config_defaults.get_default_config()
    dataset = IRMASDatasetTrain(
        dataset_path=config.path_irmas_train,
        audio_transform=None,
        normalize_audio=False,
        concat_n_samples=None,
        sum_n_samples=None,
        sampling_rate=config.sampling_rate,
        train_override_csvs=None,
        num_classes=config.num_labels,
    )
    dataset.concat_n_samples = 2
    dataset.sum_n_samples = None

    sr = 16_000
    audio_1, _ = librosa.load("data/irmas/train/cel/[cel][cla]0001__1.wav", sr=sr)
    audio_2, _ = librosa.load("data/irmas/train/cla/[cla][cla]0150__1.wav", sr=sr)
    # Simulate different legnth
    audio_1, audio_2 = audio_1[5:], audio_2[3:]
    label_1 = encode_instruments(["cel"])
    label_2 = encode_instruments(["cla"])

    audios = [audio_1, audio_2]
    labels = [label_1, label_2]
    final_audio, final_labels = dataset.concat_and_sum(
        audios, labels, use_concat=True, use_sum=False
    )

    test_audio = np.concatenate([audio_1, audio_2])
    test_labels = np.logical_or.reduce(labels).astype(labels[0].dtype)
    assert len(test_audio.shape) == 1
    assert len(final_audio.shape) == 1
    assert np.all(np.isclose(test_audio, final_audio))
    assert np.all(np.isclose(test_labels, final_labels))


if __name__ == "__main__":
    test_sum_and_concat()
