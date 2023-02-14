from __future__ import annotations

import os
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms
from torch.utils.data import Dataset
from tqdm import tqdm

import config_defaults
from utils_audio import AudioTransformAST, AudioTransformBase, stereo_to_mono
from utils_dataset import multi_hot_indices


def instrument_indices_to_torch(indices: list[int]):
    return torch.tensor(multi_hot_indices(indices, len(config_defaults.INSTRUMENT_TO_IDX)))


class IRMASDatasetTrain(Dataset):
    def __init__(
        self,
        dataset_dirs: list[Path] = [config_defaults.PATH_TRAIN],
        audio_transform: AudioTransformBase = AudioTransformAST(),
        num_classes=len(config_defaults.INSTRUMENT_TO_FULLNAME),
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

            audio_transform: _description_..
            num_classes: _description_..
            sanity_checks: _description_..
        """
        self.sampling_rate = sampling_rate
        self.audio_transform = audio_transform
        self.num_classes = num_classes

        self.dataset: list[tuple[Path, torch.Tensor]] = []

        for dataset_dir in dataset_dirs:
            for audio_file in tqdm(dataset_dir.rglob("*.wav")):
                name = str(audio_file.stem)
                instrument_indices = []
                for instrument in config_defaults.INSTRUMENT_TO_IDX.keys():
                    if f"[{instrument}]" in name:
                        instrument_indices.append(config_defaults.INSTRUMENT_TO_IDX[instrument])
                label = instrument_indices_to_torch(instrument_indices)
                self.dataset.append((audio_file, label))

        if sanity_checks:
            assert (
                len(self.dataset) == config_defaults.DEFAULT_IRMAS_TRAIN_SIZE
            ), f"IRMAS train set should contain {config_defaults.DEFAULT_IRMAS_TRAIN_SIZE} samples"

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        # TODO: fix
        audio_path, labels = self.dataset[index]
        labels = labels.float()

        audio, sampling_rate = librosa.load(audio_path, sr=config_defaults.DEFAULT_SAMPLING_RATE, mono=True)
        audio = torch.tensor(audio)
        return self.audio_transform.process(audio=audio, labels=labels, sampling_rate=sampling_rate)


class IRMASDatasetTest(Dataset):
    def __init__(
        self,
        dataset_dirs: list[Path] = [config_defaults.PATH_TEST],
        num_classes=len(config_defaults.INSTRUMENT_TO_FULLNAME),
        sanity_checks=config_defaults.DEFAULT_SANITY_CHECKS,
        audio_transform: AudioTransformBase = AudioTransformAST(),
    ):
        self.num_classes = num_classes
        self.audio_transform = audio_transform
        self.dataset: list[tuple[Path, torch.Tensor]] = []

        for dataset_dir in dataset_dirs:
            for audio_file in tqdm(dataset_dir.rglob("*.wav")):
                path_without_ext = os.path.splitext(audio_file)[0]
                txt_filename = Path(path_without_ext + ".txt")

                if not txt_filename.is_file():
                    raise FileNotFoundError(f"File {txt_filename} doesn't exist.")

                instrument_indices = []
                with open(txt_filename) as f:
                    for line in f:
                        instrument = line.rstrip("\n").replace("\t", "")
                        instrument_indices.append(config_defaults.INSTRUMENT_TO_IDX[instrument])

                label = instrument_indices_to_torch(instrument_indices)
                self.dataset.append((audio_file, label))

        if sanity_checks:
            assert (
                len(self.dataset) == config_defaults.DEFAULT_IRMAS_TEST_SIZE
            ), f"IRMAS test set should contain {config_defaults.DEFAULT_IRMAS_TEST_SIZE} samples"

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        # TODO:
        audio_path, labels = self.dataset[index]
        labels = labels.float()

        audio, sampling_rate = librosa.load(audio_path, sr=config_defaults.DEFAULT_SAMPLING_RATE, mono=True)
        audio = torch.tensor(audio)
        # audio = audio.unsqueeze(dim=-1)
        return self.audio_transform.process(audio=audio, labels=labels, sampling_rate=sampling_rate)


class InstrumentInference(Dataset):
    pass


if __name__ == "__main__":
    IRMASDatasetTest().__getitem__(0)
