from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import pyloudnorm
import torch
import torch_audiomentations as TA
import torchaudio.transforms as TT
from torchvision.transforms import RandomErasing

from src.config.config_defaults import ConfigDefault
from src.enums.enums import SupportedAugmentations
from src.utils.utils_audio import time_mask_audio


class WaveformAugmentation:
    def __init__(
        self,
        augmentations: list[SupportedAugmentations],
        sampling_rate: int,
        stretch_factors: tuple[float, float],
        time_inversion_p,
        path_background_noise: None | Path,
        time_mask_max_percentage: float,
        **kwargs,
    ):
        self.augmentations = augmentations
        self.sampling_rate = sampling_rate
        self.time_mask_max_percentage = time_mask_max_percentage

        self.color_noise = TA.AddColoredNoise(
            min_snr_in_db=3,  # smaller min_snr_in_db => louder nouse
            max_snr_in_db=25,
            p=1,
            sample_rate=self.sampling_rate,
        )

        try:
            self.background_noise = TA.AddBackgroundNoise(
                background_paths=path_background_noise,
                min_snr_in_db=3.0,
                max_snr_in_db=30,
                p=1,
                sample_rate=self.sampling_rate,
            )

        except Exception:
            print(
                f"Warning: skipping background noise because directory {path_background_noise} is invalid or has no sounds."
            )
            self.background_noise = None

        self.timeinv = TA.PolarityInversion(
            p=time_inversion_p,
            sample_rate=self.sampling_rate,
        )

        self.pitch = TA.PitchShift(
            min_transpose_semitones=-2,
            max_transpose_semitones=2,
            p=1,
            sample_rate=self.sampling_rate,
        )
        self.stretch_factors = stretch_factors

        self.time_shift = TA.Shift(
            min_shift=-0.5,
            max_shift=0.5,
            p=1,
            sample_rate=self.sampling_rate,
        )

    def to_type(self, audio: torch.Tensor | np.ndarray, t: Callable):
        """Converts audio to torch or numpy (only if needed)"""
        if isinstance(audio, torch.Tensor) and t == np.array:
            return self.to_numpy(audio)
        elif isinstance(audio, np.ndarray) and t == torch.tensor:
            return self.to_torch(audio)
        return audio

    def to_numpy(self, audio: torch.Tensor) -> np.ndarray:
        return audio.squeeze(0).squeeze(0).numpy()

    def to_torch(self, audio: np.ndarray) -> torch.Tensor:
        return torch.tensor(audio).unsqueeze(0).unsqueeze(0)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if len(self.augmentations) == 0:
            return audio

        if (
            SupportedAugmentations.BACKGROUND_NOISE in self.augmentations
            and self.background_noise is not None
        ):
            audio = self.to_type(audio, torch.tensor)
            audio = self.background_noise(audio, sample_rate=self.sampling_rate)

        if SupportedAugmentations.TIMEINV in self.augmentations:
            audio = self.to_type(audio, torch.tensor)
            audio = self.timeinv(audio, sample_rate=self.sampling_rate)

        if SupportedAugmentations.PITCH in self.augmentations:
            audio = self.to_type(audio, torch.tensor)
            audio = self.pitch(audio, sample_rate=self.sampling_rate)

        if SupportedAugmentations.TIME_STRETCH in self.augmentations:
            audio = self.to_type(audio, np.array)
            random_rate = np.random.uniform(*self.stretch_factors)
            audio = librosa.effects.time_stretch(audio, rate=random_rate)

        if SupportedAugmentations.TIME_SHIFT in self.augmentations:
            audio = self.to_type(audio, torch.tensor)
            audio = self.time_shift(audio, sample_rate=self.sampling_rate)

        if SupportedAugmentations.NORM_AFTER_TIME_AUGS in self.augmentations:
            audio = self.to_type(audio, np.array)
            meter = pyloudnorm.Meter(self.sampling_rate)
            loudness = meter.integrated_loudness(audio)
            audio = pyloudnorm.normalize.loudness(audio, loudness, -12)

        if SupportedAugmentations.COLOR_NOISE in self.augmentations:
            audio = self.to_type(audio, torch.tensor)
            audio = self.color_noise(audio)

        if SupportedAugmentations.TIME_MASK in self.augmentations:
            audio = self.to_type(audio, np.array)
            percentage = np.random.uniform(0, self.time_mask_max_percentage)
            audio = time_mask_audio(audio, percentage)

        audio = self.to_type(audio, np.array)
        return audio


class SpectrogramAugmentation:
    def __init__(
        self,
        augmentations: list[SupportedAugmentations],
        sampling_rate: int,
        freq_mask_param,
        hide_random_pixels_p,
        std_noise,
        **kwargs,
    ):
        self.augmentations = augmentations
        self.sampling_rate = sampling_rate
        self.hide_random_pixels_p = hide_random_pixels_p
        self.std_noise = std_noise
        self.freq_mask = TT.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.random_erase = RandomErasing(scale=(0.02, 0.2), ratio=(1, 2), p=1)

    # @timeit
    def __call__(self, spectrogram: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(spectrogram, np.ndarray):
            spectrogram = torch.tensor(spectrogram)

        if len(self.augmentations) == 0:
            return spectrogram

        return_batch_dim = True

        if len(spectrogram.shape) == 2:
            return_batch_dim = False
            spectrogram = spectrogram.unsqueeze(0)

        if SupportedAugmentations.RANDOM_PIXELS in self.augmentations:
            hide_random_pixels_p = np.random.uniform(0, 0.3)
            mask = (
                torch.FloatTensor(*spectrogram.shape).uniform_() < hide_random_pixels_p
            )
            spectrogram[mask] = 0

        if SupportedAugmentations.FREQ_MASK in self.augmentations:
            spectrogram = self.freq_mask(spectrogram)

        if SupportedAugmentations.RANDOM_ERASE in self.augmentations:
            spectrogram = self.random_erase(spectrogram)

        if not return_batch_dim:
            return spectrogram.squeeze(0)

        return spectrogram


def get_augmentations(config: ConfigDefault):
    train_kwargs = dict(
        augmentations=config.augmentations,
        sampling_rate=config.sampling_rate,
        **config.aug_kwargs,
    )
    val_kwargs = {**train_kwargs, "augmentations": []}

    train_spectrogram_augmentation = SpectrogramAugmentation(**train_kwargs)
    train_waveform_augmentation = WaveformAugmentation(**train_kwargs)
    val_spectrogram_augmentation = SpectrogramAugmentation(**val_kwargs)
    val_waveform_augmentation = WaveformAugmentation(**val_kwargs)
    return (
        train_spectrogram_augmentation,
        train_waveform_augmentation,
        val_spectrogram_augmentation,
        val_waveform_augmentation,
    )
