from pathlib import Path

import librosa
import numpy as np
import torch

from src.data.dataset_base import DatasetBase, DatasetGetItem
from src.features.audio_transform_base import AudioTransformBase


class PureAudioDataset(DatasetBase):
    def __init__(
        self,
        dataset_path: dict[str, tuple[np.ndarray, int]],
        audio_transform: AudioTransformBase | None,
        sampling_rate: int,
        num_lables: int,
        normalize_audio: bool = True,
    ):
        """Pure audio dataset. No labels, just audio.

        Args:
            audio_and_sr: [[0.1,0.2,-0.1], 16_000, "random_name"]
        """
        self.dataset_path = dataset_path
        self.audio_transform = audio_transform
        self.sampling_rate = sampling_rate
        self.normalize_audio = normalize_audio
        self.num_lables = num_lables

        kwargs = dict(
            num_classes=0,
            sum_n_samples=False,
            concat_n_samples=None,
            train_override_csvs=None,
        )

        super().__init__(
            dataset_path=dataset_path,
            audio_transform=audio_transform,
            sampling_rate=sampling_rate,
            normalize_audio=normalize_audio,
            **kwargs,
        )

    def set_need_to_sample(self):
        self.need_to_resample = True

    def create_dataset_list(self) -> list[tuple[Path, np.ndarray]]:
        dataset_list: list[tuple[Path, np.ndarray]] = []
        for filename in self.dataset_path.keys():
            dataset_list.append((filename, np.array([0])))
        return dataset_list

    def load_sample(self, item_idx: int) -> tuple[np.ndarray, np.ndarray, Path]:
        """Gets item from dataset_list and loads the audio."""
        audio_path, labels = self.dataset_list[item_idx]
        audio, sampling_rate = self.dataset_path[audio_path]
        audio = librosa.to_mono(audio)
        if sampling_rate != self.sampling_rate:
            audio = librosa.resample(
                y=audio, orig_sr=sampling_rate, target_sr=self.sampling_rate
            )

        return audio, labels, audio_path

    def __getitem__(self, index: int) -> DatasetGetItem:
        audio, labels, _ = self.load_sample(index)
        labels = torch.tensor(labels).float()
        features = self.audio_transform(audio)
        return features, labels, index
