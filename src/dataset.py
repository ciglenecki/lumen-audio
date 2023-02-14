from __future__ import annotations

import os
from pathlib import Path

import librosa
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

import src.config_defaults as config_defaults
from src.audio_transform import AudioTransformAST, AudioTransformBase
from src.utils_dataset import multi_hot_indices

# '*.(wav|mp3|flac)'
# glob_expression = f"*\.({'|'.join(config_defaults.DEFAULT_AUDIO_EXTENSIONS)})"
glob_expression = "*.wav"


class IRMASDatasetTrain(Dataset):
    def __init__(
        self,
        dataset_dirs: list[Path] = [config_defaults.PATH_TRAIN],
        audio_transform: AudioTransformAST = AudioTransformAST(),
        num_classes=config_defaults.DEFAULT_NUM_LABELS,
        sanity_checks=config_defaults.DEFAULT_SANITY_CHECKS,
        sampling_rate=config_defaults.DEFAULT_SAMPLING_RATE,
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
                instrument_indices = []
                for instrument in config_defaults.INSTRUMENT_TO_IDX.keys():
                    if f"[{instrument}]" in filename:
                        instrument_indices.append(
                            config_defaults.INSTRUMENT_TO_IDX[instrument]
                        )
                labels = multi_hot_indices(
                    instrument_indices,
                    config_defaults.DEFAULT_NUM_LABELS,
                )
                self.dataset.append((audio_path, labels))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):

        audio_path, labels = self.dataset[index]
        audio, orig_sampling_rate = librosa.load(audio_path, sr=None)
        spectrogram, labels = self.audio_transform.process(
            audio=audio,
            labels=labels,
            orig_sampling_rate=orig_sampling_rate,
            sampling_rate=self.sampling_rate,
        )

        labels = labels.float()  # avoid errors in loss function
        return spectrogram, labels


class IRMASDatasetTest(Dataset):
    def __init__(
        self,
        dataset_dirs: list[Path] = [config_defaults.PATH_TEST],
        num_classes=config_defaults.DEFAULT_NUM_LABELS,
        sanity_checks=config_defaults.DEFAULT_SANITY_CHECKS,
        audio_transform: AudioTransformAST = AudioTransformAST(),
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

                self.dataset.append((audio_file, labels))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        audio_path, labels = self.dataset[index]
        audio, orig_sampling_rate = librosa.load(audio_path, sr=None)
        spectrogram, labels = self.audio_transform.process(
            audio=audio,
            labels=labels,
            orig_sampling_rate=orig_sampling_rate,
            sampling_rate=self.sampling_rate,
        )

        labels = labels.float()  # avoid errors in loss function
        return spectrogram, labels


class InstrumentInference(Dataset):
    pass


if __name__ == "__main__":
    IRMASDatasetTest().__getitem__(0)
