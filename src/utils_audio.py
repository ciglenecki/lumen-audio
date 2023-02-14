import platform
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from torchaudio.transforms import (
    FrequencyMasking,
    MelScale,
    Spectrogram,
    TimeMasking,
    TimeStretch,
)
from transformers import ASTFeatureExtractor

import config_defaults as config_defaults
from utils_functions import MultiEnum


class SpectogramAugmentation(torch.nn.Module):
    pass


def stereo_to_mono(audio: torch.Tensor | np.ndarray):
    if isinstance(audio, torch.Tensor):
        return audio.sum(dim=1) / 2
    elif isinstance(audio, np.ndarray):
        return audio.sum(axis=-1) / 2


class AudioTransformBase(ABC):
    @abstractmethod
    def process(self, audio: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray, sampling_rate: int):
        pass


class AudioTransformAST(AudioTransformBase):
    def __init__(
        self,
        ast_pretrained_tag=config_defaults.DEFAULT_AST_PRETRAINED_TAG,
    ):
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(ast_pretrained_tag)

    def process(
        self,
        audio: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        sampling_rate: int,
    ):
        features = self.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        spectogram = features["input_values"].squeeze(dim=0)  # mel filter banks
        return spectogram, labels


class AudioAugmentation(torch.nn.Module):
    """Define custom feature extraction pipeline.

    1. Convert to power spectrogram
    2. Apply augmentations
    3. Convert to mel-scale
    """

    def __init__(self, n_fft=1024, n_mel=256, stretch_factor=0.8, sample_rate=config_defaults.DEFAULT_SAMPLING_RATE):
        """_summary_

        Args:
            n_fft: Size of the fast-fourier transform (FFT), creates n_fft // 2 + 1 frequency bins
            n_mel: Number of mel filterbanks
            stretch_factor: rate to speed up or slow down by
            sample_rate: sampling rate
        """

        super().__init__()

        self.spec = Spectrogram(n_fft=n_fft, power=2)

        self.spec_aug = torch.nn.Sequential(
            TimeStretch(fixed_rate=stretch_factor),
            FrequencyMasking(freq_mask_param=80),
            TimeMasking(time_mask_param=80),
        )

        self.mel_scale = MelScale(n_mels=n_mel, sample_rate=sample_rate, n_stft=n_fft // 2 + 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:

        # Convert to power spectrogram
        spec = self.spec(waveform)

        # Apply SpecAugment
        spec = self.spec_aug(spec)

        # Convert to mel-scale
        mel = self.mel_scale(spec)

        return mel


class AudioTransforms(MultiEnum):
    """enumname = AudioTransformBase class, 'key'"""

    AST = AudioTransformAST(ast_pretrained_tag=config_defaults.DEFAULT_AST_PRETRAINED_TAG), "ast"
