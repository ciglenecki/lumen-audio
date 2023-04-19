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
from src.features.audio_transform_base import AudioTransformBase
from src.utils.utils_audio import load_audio_from_file, play_audio
from src.utils.utils_dataset import (
    encode_instruments,
    instrument_multihot_to_idx,
    multi_hot_encode,
)

# '*.(wav|mp3|flac)'
# glob_expression = f"*\.({'|'.join(defaults.DEFAULT_AUDIO_EXTENSIONS)})"
glob_expression = "*.wav"

config = config_defaults.get_default_config()


class IRMASDatasetTrain(Dataset):
    all_instrument_indices = np.array(list(config_defaults.INSTRUMENT_TO_IDX.values()))

    def __init__(
        self,
        dataset_dir: Path = config.path_irmas_train,
        audio_transform: AudioTransformBase | None = None,
        num_classes=config_defaults.DEFAULT_NUM_LABELS,
        sampling_rate=config.sampling_rate,
        normalize_audio=config.normalize_audio,
        normalize_image=config.normalize_image,
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
        self.normalize_image = normalize_image
        self.sum_two_samples = sum_two_samples
        self.concat_n_samples = concat_n_samples
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

            self.dataset.append((path, labels))

            for instrument in item_instruments:
                self.instrument_idx_list[instrument].append(item_idx)

    def __len__(self) -> int:
        return len(self.dataset)

    def load_sample(self, item_idx: int) -> tuple[np.ndarray, np.ndarray, Path]:
        audio_path, labels = self.dataset[item_idx]
        audio, _ = load_audio_from_file(
            audio_path,
            target_sr=self.sampling_rate,
            method="librosa",
            normalize=self.normalize_audio,
        )
        return audio, labels, audio_path

    def get_random_sample_for_instrument(self, instrument_idx: int) -> int:
        """Returns a random sample which contains the instrument with index instrument_idx."""
        instrument = config_defaults.IDX_TO_INSTRUMENT[instrument_idx]
        random_sample_idx = random.choice(self.instrument_idx_list[instrument])
        return random_sample_idx

    def sample_n_negative_samples(
        self, n: int, original_labels: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[Path]]:
        # Take all instruments and exclude ones from the original label
        negative_indices_pool = self.all_instrument_indices[
            ~original_labels.astype(bool)
        ]

        allow_repeating_labels = False

        # The pool of negative indices isn't large enough so we have to sample same negative indices mulitple times
        if len(negative_indices_pool) < n:
            allow_repeating_labels = True

        negative_indices = np.random.choice(
            negative_indices_pool,
            size=n,
            replace=allow_repeating_labels,
        )

        negative_audios = []
        negative_labels = []
        negative_paths = []
        for instrument_idx in negative_indices:
            sample_idx = self.get_random_sample_for_instrument(instrument_idx)
            negative_audio, negative_label, negative_path = self.load_sample(sample_idx)
            negative_audios.append(negative_audio)
            negative_labels.append(negative_label)
            negative_paths.append(negative_path)
        return negative_audios, negative_labels, negative_paths

    def _pad_with_zeros(self, audio: np.ndarray, desired_size: int):
        pad_width = (0, desired_size - len(audio))
        return np.pad(audio, pad_width, mode="constant", constant_values=0)

    def concat_and_sum(
        self,
        audios: list[np.ndarray],
        labels: list[np.ndarray],
        use_concat: bool,
        sum_two_samples: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Performs concaternation and summation of multiple audios and multiple labels.

        Args:
            audios: list[np.ndarray]
            labels: list[np.ndarray]

        Example 1:

            use_concat: True
            sum_two_samples: False

            audios: __x__, __a__, __b__
            returns: |__x__|__a__|__b__|

        Example 2:

            use_concat: False
            sum_two_samples: True

            audios: __x__, __a__
            returns: |__xa__|

        Example 1:

            use_concat: 3
            sum_two_samples: True

            audios: __x__, __a__, __b__, ..., __e__

            |block1 |block2 |block3 |
            |___x___|___a___|___b___| (concated audio 1)
            |___c___|___d___|___e___| (concated audio 2)

            returns: |__xc__|__ad__|__be__| (summed audios)
        """

        n = len(audios)

        if use_concat and not sum_two_samples:
            audios = np.concatenate(audios)

        # If we want to sum two samples, create two concatenated audios
        # Otherwise use just one long big audio
        if use_concat and sum_two_samples:
            n_half = n // 2
            top_audio = np.concatenate(audios[:n_half], axis=0)
            bottom_audio = np.concatenate(audios[n_half:], axis=0)
            audios = [top_audio, bottom_audio]

        # If at this point we have more than one audio we want to sum them
        # For summing, audios have to have equal size
        # Pad every audio with 0 to the maximum length
        if sum_two_samples:
            max_len = max(len(a) for a in audios)
            for i, audio in enumerate(audios):
                if len(audio) != max_len:
                    audios[i] = self._pad_with_zeros(audio, max_len)
            # Now that they are equal in size we can create numpy array
            audios = np.mean(audios, axis=0)

        labels = np.logical_or.reduce(labels).astype(labels[0].dtype)
        return audios, labels

    def concat_and_sum_random_negative_samples(self, original_audio, original_labels):
        """Finds (num_negative_sampels * 2) - 1 negative samples which are concated and summed to original audio. Negative audio sample doesn't share original audio's labels"""
        num_negative_sampels = self.concat_n_samples
        if self.sum_two_samples:
            num_negative_sampels = num_negative_sampels * 2
        num_negative_sampels = num_negative_sampels - 1  # exclude original audio

        # Load negative samples
        multiple_audios, multiple_labels, _ = self.sample_n_negative_samples(
            num_negative_sampels, original_labels=original_labels
        )

        # Add original audio at the beggining
        multiple_audios.insert(0, original_audio)
        multiple_labels.insert(0, original_labels)

        audio, labels = self.concat_and_sum(
            multiple_audios,
            multiple_labels,
            use_concat=self.concat_n_samples is not None and self.concat_n_samples > 1,
            sum_two_samples=self.sum_two_samples,
        )

        return audio, labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        audio, labels, _ = self.load_sample(index)

        if (
            self.concat_n_samples is not None and self.concat_n_samples > 1
        ) or self.sum_two_samples:
            audio, labels = self.concat_and_sum_random_negative_samples(audio, labels)

        if self.audio_transform is None:
            return audio, labels, index

        features = self.audio_transform(audio)
        labels = torch.tensor(labels).float()

        # Uncomment for playing audio
        # print(
        #     [
        #         config_defaults.INSTRUMENT_TO_FULLNAME[
        #             config_defaults.IDX_TO_INSTRUMENT[i]
        #         ]
        #         for i in instrument_multihot_to_idx(labels)
        #     ]
        # )
        # print("first time")
        # play_audio(audio, sampling_rate=self.sampling_rate)
        # print("second time")
        # play_audio(audio, sampling_rate=self.sampling_rate)
        return features, labels, index


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


class IRMASDatasetTest(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        audio_transform: AudioTransformBase | None = None,
        num_classes=config_defaults.DEFAULT_NUM_LABELS,
        sampling_rate=config.sampling_rate,
        normalize_audio=config.normalize_audio,
        normalize_image=config.normalize_image,
    ):
        self.num_classes = num_classes
        self.audio_transform = audio_transform
        self.dataset: list[tuple[Path, np.ndarray]] = []
        self.dataset_dir = dataset_dir
        self.sampling_rate = sampling_rate
        self.normalize_audio = normalize_audio
        self.normalize_image = normalize_image

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

            self.dataset.append((str(audio_file), labels))

    def load_sample(self, item_idx: int) -> tuple[np.ndarray, np.ndarray, Path]:
        audio_path, labels = self.dataset[item_idx]
        audio, _ = load_audio_from_file(
            audio_path,
            target_sr=self.sampling_rate,
            method="librosa",
            normalize=self.normalize_audio,
        )
        return audio, labels, audio_path

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        audio, labels, audio_path = self.load_sample(index)

        if self.audio_transform is None:
            return audio, labels, index

        features = self.audio_transform(audio)

        labels = torch.tensor(labels).float()

        return features, labels, index


class InstrumentInference(Dataset):
    pass


def test_sum_and_concat():
    dataset = IRMASDatasetTrain()
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
        audios, labels, use_concat=True, sum_two_samples=True
    )

    top = np.concatenate([audio_1, audio_2])
    bottom = np.concatenate([audio_3, audio_4])
    max_len = np.max([len(top), len(bottom)])

    if len(top) != max_len:
        pad_width = (0, max_len - len(top))
        top = np.pad(top, pad_width, mode="constant", constant_values=0)
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
    dataset = IRMASDatasetTrain()
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
        audios, labels, use_concat=False, sum_two_samples=True
    )

    max_len = np.max([len(audio_1), len(audio_2)])

    if len(audio_1) != max_len:
        pad_width = (0, max_len - len(audio_1))
        audio_1 = np.pad(audio_1, pad_width, mode="constant", constant_values=0)
    if len(audio_2) != max_len:
        pad_width = (0, max_len - len(audio_2))
        bottom = np.pad(audio_2, pad_width, mode="constant", constant_values=0)
    test_audio = np.mean([audio_1, audio_2], axis=0)
    test_labels = np.logical_or.reduce(labels).astype(labels[0].dtype)
    assert len(test_audio.shape) == 1
    assert len(final_audio.shape) == 1
    assert np.all(np.isclose(test_audio, final_audio))
    assert np.all(np.isclose(test_labels, final_labels))


def test_simple_concat():
    dataset = IRMASDatasetTrain()
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
        audios, labels, use_concat=True, sum_two_samples=False
    )

    test_audio = np.concatenate([audio_1, audio_2])
    test_labels = np.logical_or.reduce(labels).astype(labels[0].dtype)
    assert len(test_audio.shape) == 1
    assert len(final_audio.shape) == 1
    assert np.all(np.isclose(test_audio, final_audio))
    assert np.all(np.isclose(test_labels, final_labels))


if __name__ == "__main__":
    test_sum_and_concat()
