from __future__ import annotations

import os
import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

import src.config.config_defaults as config_defaults
from src.data.dataset_base import DatasetBase, DatasetInternalItem
from src.utils.utils_dataset import encode_instruments, multi_hot_encode

# '*.(wav|mp3|flac)'
# glob_expression = f"*\.({'|'.join(defaults.DEFAULT_AUDIO_EXTENSIONS)})"
glob_expression = "*.wav"

config = config_defaults.get_default_config()


class IRMASDatasetTrain(DatasetBase):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """_summary_

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

        super().__init__(*args, **kwargs)  # sets self.dataset

        assert (
            len(self.dataset_list) == config_defaults.DEFAULT_IRMAS_TRAIN_SIZE
        ), f"IRMAS train set should contain {config_defaults.DEFAULT_IRMAS_TRAIN_SIZE} samples"

    def create_dataset_list(self) -> list[DatasetInternalItem]:
        """Reads audio and label files and creates tuples of (audio_path, one hot encoded label)
        self.instrument_idx_list = {
            "guitar": [0, 3, 5, 9, 13, 15]
            "flute": [2,4,6,7,8]
        }
        """
        dataset_list = []
        if self.train_override_csvs:
            dfs = [pd.read_csv(csv_path) for csv_path in self.train_override_csvs]
            df = pd.concat(dfs, ignore_index=True)
            df.set_index("filename", inplace=True)

        # self.instrument_idx_list = {
        #     i.value: [] for i in config_defaults.InstrumentEnums
        # }

        for item_idx, path in tqdm(enumerate(self.dataset_path.rglob(glob_expression))):
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

            dataset_list.append((path, labels))

            # for instrument in item_instruments:
            #     self.instrument_idx_list[instrument].append(item_idx)
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
        super().__init__(*args, **kwargs)  # sets self.dataset

        assert (
            len(self.dataset_list) == config_defaults.DEFAULT_IRMAS_TEST_SIZE
        ), f"IRMAS test set should contain {config_defaults.DEFAULT_IRMAS_TEST_SIZE} samples"

    def create_dataset_list(self) -> list[tuple[Path, np.ndarray]]:
        """Reads audio and label files and creates tuples of (audio_path, one hot encoded label)"""
        dataset_list: list[tuple[Path, np.ndarray]] = []
        for audio_file in tqdm(self.dataset_path.rglob(glob_expression)):
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
