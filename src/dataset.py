from __future__ import annotations

from pathlib import Path

import numpy as np
from utils_audio import AudioTransform, AudioTransformAST, stereo_to_mono
import torchaudio
import torchaudio.transforms
from torch.utils.data import Dataset
from tqdm import tqdm

import config_defaults
from utils_dataset import multi_hot_indices



class IRMASDatasetTrain(Dataset):
    def __init__(
        self,
        dataset_dirs: list[Path] = [config_defaults.PATH_TRAIN],
        audio_transform: AudioTransform = AudioTransformAST(),
        num_classes=len(config_defaults.INSTRUMENT_TO_FULLNAME),
        sanity_checks=True,
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

        self.dataset: list[tuple[Path, np.ndarray]] = []

        for dataset_dir in dataset_dirs:
            for audio_file in tqdm(dataset_dir.rglob("*.wav")):
                name = str(audio_file.stem)
                instrument_indices = []
                for instrument in config_defaults.INSTRUMENT_TO_IDX.keys():
                    if f"[{instrument}]" in name:
                        instrument_indices.append(
                            config_defaults.INSTRUMENT_TO_IDX[instrument]
                        )
                label = multi_hot_indices(
                    instrument_indices, len(config_defaults.INSTRUMENT_TO_IDX)
                )
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
        audio, sampling_rate = torchaudio.load(audio_path)
        return self.audio_transform.process(
            audio, labels=labels, sampling_rate=sampling_rate
        )


class IRMASDatasetTest(Dataset):
    def __init__(
        self,
        dataset_dirs: list[Path] = [config_defaults.PATH_TRAIN],
        num_classes=len(config_defaults.INSTRUMENT_TO_FULLNAME),
        sanity_checks=True,
    ):
        self.num_classes = num_classes

        self.dataset: list[tuple[Path, np.ndarray]] = []

        for dataset_dir in dataset_dirs:
            for audio_file in tqdm(dataset_dir.rglob("*.wav")):
                name = str(audio_file.stem)
                instrument_indices = []
                for instrument in config_defaults.INSTRUMENT_TO_IDX.keys():
                    if f"[{instrument}]" in name:
                        instrument_indices.append(
                            config_defaults.INSTRUMENT_TO_IDX[instrument]
                        )
                label = multi_hot_indices(
                    instrument_indices, len(config_defaults.INSTRUMENT_TO_IDX)
                )
                self.dataset.append((audio_file, label))

        if sanity_checks:
            assert (
                len(self.dataset) == config_defaults.DEFAULT_IRMAS_TRAIN_SIZE
            ), f"IRMAS train set should contain {config_defaults.DEFAULT_IRMAS_TRAIN_SIZE} samples"

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        # TODO:
        audio_path, label = self.dataset[index]

        return "TODO", label


class InstrumentInference(Dataset):
    pass


if __name__ == "__main__":
    IRMASDatasetTrain()
