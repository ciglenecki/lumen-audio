from pathlib import Path

import audiomentations as AA
import numpy as np
import torch
import torch_audiomentations
from torchvision.transforms import RandomErasing

from src.config.config_defaults import ConfigDefault
from src.enums.enums import SupportedAugmentations


class WaveformAugmentation:
    def __init__(
        self,
        augmentations: list[SupportedAugmentations],
        sampling_rate: int,
        stretch_factors,
        time_inversion_p,
        path_background_noise: None | Path,
        **kwargs,
    ):
        self.augmentations = augmentations
        self.sampling_rate = sampling_rate
        # if min_snr_in_db and max_snr_in_db are lower than the noise is louder
        self.color_noise = torch_audiomentations.AddColoredNoise(
            min_snr_in_db=3,
            max_snr_in_db=25,
            p=1,
            sample_rate=self.sampling_rate,
        )
        self.seven_band_eq = AA.SevenBandParametricEQ(
            min_gain_db=-5, max_gain_db=5, p=1
        )

        try:
            self.background_noise = AA.AddBackgroundNoise(
                sounds_path=path_background_noise,
                min_snr_in_db=3.0,
                max_snr_in_db=30,
                noise_transform=AA.PolarityInversion(p=0.5),
                p=1,
            )
            print(
                f"Warning: skipping background noise because directory {path_background_noise} is invalid or has no sounds."
            )
        except Exception:
            self.background_noise = None

        self.timeinv = AA.PolarityInversion(p=time_inversion_p)
        self.pitch = AA.PitchShift(min_semitones=-2, max_semitones=2, p=1)
        self.time_stretch = AA.TimeStretch(
            min_rate=stretch_factors[0],
            max_rate=stretch_factors[1],
            p=1,
            leave_length_unchanged=False,
        )
        self.time_shift = AA.Shift(min_fraction=-0.5, max_fraction=0.5, p=1)
        self.clipping = AA.ClippingDistortion(
            min_percentile_threshold=0, max_percentile_threshold=5, p=0.5
        )
        self.norm_after_time_augs = AA.Normalize(p=1)
        # AA.LoudnessNormalization(max_lufs_in_db=-5, min_lufs_in_db=-15, p=0.5)
        self.time_mask = AA.TimeMask(min_band_part=0, max_band_part=0.2, p=1)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if len(self.augmentations) == 0:
            return audio

        if SupportedAugmentations.SEVEN_BAND_EQ in self.augmentations:
            self.seven_band_eq
        if (
            SupportedAugmentations.BACKGROUND_NOISE in self.augmentations
            and self.background_noise is not None
        ):
            audio = self.background_noise(audio, sample_rate=self.sampling_rate)
        if SupportedAugmentations.TIMEINV in self.augmentations:
            audio = self.timeinv(audio, sample_rate=self.sampling_rate)
        if SupportedAugmentations.PITCH in self.augmentations:
            audio = self.pitch(audio, sample_rate=self.sampling_rate)
        if SupportedAugmentations.TIME_STRETCH in self.augmentations:
            audio = self.time_stretch(audio, sample_rate=self.sampling_rate)
        if SupportedAugmentations.TIME_SHIFT in self.augmentations:
            audio = self.time_shift(audio, sample_rate=self.sampling_rate)
        if SupportedAugmentations.CLIPPING in self.augmentations:
            audio = self.clipping(audio, sample_rate=self.sampling_rate)
        if SupportedAugmentations.NORM_AFTER_TIME_AUGS in self.augmentations:
            audio = self.norm_after_time_augs(audio, sample_rate=self.sampling_rate)
            # AA.LoudnessNormalization(max_lufs_in_db=-5, min_lufs_in_db=-15, p=0.5)

        if SupportedAugmentations.COLOR_NOISE in self.augmentations:
            audio = torch.tensor(audio).unsqueeze(0).unsqueeze(0)
            audio = self.color_noise(audio)
            audio = audio.squeeze(0).squeeze(0).numpy()

        if SupportedAugmentations.TIME_MASK in self.augmentations:
            audio = self.time_mask(audio, sample_rate=self.sampling_rate)

        return audio  # same as x2 squeeze(0)


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

        self.freq_mask = AA.SpecFrequencyMask(p=1)
        self.random_erase = RandomErasing(scale=(0.02, 0.2), ratio=(1, 2), p=1)

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
            spectrogram = [
                torch.tensor(self.freq_mask(s.numpy())) for s in spectrogram
            ]  # [b, h, w] -> [h, w]
            spectrogram = torch.stack(spectrogram)
            # [b, h, w]

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
