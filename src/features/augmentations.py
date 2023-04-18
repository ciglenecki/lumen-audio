import audiomentations as AA
import librosa
import numpy as np
import torch
import torch_audiomentations
from audiomentations import AddGaussianNoise, Compose, PitchShift, Shift, TimeStretch
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchvision.transforms import RandomErasing

from src.config.config_defaults import ConfigDefault
from src.enums.enums import SupportedAugmentations


class WaveformAugmentation:
    def __init__(
        self,
        augmentation_enums: list[SupportedAugmentations],
        sampling_rate: int,
        stretch_factors,
        time_inversion_p,
        **kwargs,
    ):
        self.augmentation_enums = augmentation_enums
        self.sampling_rate = sampling_rate
        self.stretch_factors = stretch_factors
        self.color_noise = torch_audiomentations.AddColoredNoise(
            min_snr_in_db=3,
            max_snr_in_db=25,
            p=1,
            sample_rate=self.sampling_rate,
        )
        self.bandpass_filter = torch_audiomentations.BandPassFilter(
            p=1, sample_rate=self.sampling_rate, target_rate=self.sampling_rate
        )
        self.pitch_shift = torch_audiomentations.PitchShift(
            p=1, sample_rate=self.sampling_rate
        )
        self.timeinv = torch_audiomentations.TimeInversion(
            p=time_inversion_p, sample_rate=self.sampling_rate
        )

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if len(self.augmentation_enums) == 0:
            return audio

        if SupportedAugmentations.TIME_STRETCH in self.augmentation_enums:
            l, r = self.stretch_factors
            stretch_rate = np.random.uniform(l, r)
            # size_before = len(audio)
            audio = librosa.effects.time_stretch(y=audio, rate=stretch_rate)
            # audio = audio[:size_before]

        # train_transforms = AA.Compose(
        #     [
        #         AA.SevenBandParametricEQ(p=1, min_gain_db=-12, max_gain_db=12),
        #         TimeStretch(
        #             min_rate=0.8, max_rate=1.2, p=1, leave_length_unchanged=False
        #         ),
        #         AA.PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        #     ]
        # )
        # audio = train_transforms(audio, sample_rate=self.sampling_rate)

        audio = torch.tensor(audio[np.newaxis, np.newaxis, :])

        if SupportedAugmentations.PITCH in self.augmentation_enums:
            audio = self.pitch_shift(audio)

        if SupportedAugmentations.BANDPASS_FILTER in self.augmentation_enums:
            audio = self.bandpass_filter(audio)

        if SupportedAugmentations.COLOR_NOISE in self.augmentation_enums:
            # if min_snr_in_db and max_snr_in_db are lower than the noise is louder
            audio = self.color_noise(audio)

        if SupportedAugmentations.TIMEINV in self.augmentation_enums:
            audio = self.timeinv(audio)

        return audio[0, 0, :].numpy()  # same as x2 squeeze(0)


class SpectrogramAugmentation:
    def __init__(
        self,
        augmentation_enums: list[SupportedAugmentations],
        sampling_rate: int,
        freq_mask_param,
        time_mask_param,
        hide_random_pixels_p,
        std_noise,
        **kwargs,
    ):
        self.augmentation_enums = augmentation_enums
        self.sampling_rate = sampling_rate
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.hide_random_pixels_p = hide_random_pixels_p
        self.std_noise = std_noise

    def __call__(self, spectrogram: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(spectrogram, np.ndarray):
            spectrogram = torch.tensor(spectrogram)

        if len(self.augmentation_enums) == 0:
            return spectrogram

        return_batch_dim = True

        if len(spectrogram.shape) == 2:
            return_batch_dim = False
            spectrogram = spectrogram.unsqueeze(0)

        spec_mean = spectrogram.mean()
        if SupportedAugmentations.FREQ_MASK in self.augmentation_enums:
            spectrogram = FrequencyMasking(freq_mask_param=self.freq_mask_param)(
                spectrogram,
                mask_value=spec_mean,
            )
        if SupportedAugmentations.TIME_MASK in self.augmentation_enums:
            spectrogram = TimeMasking(time_mask_param=self.time_mask_param)(spectrogram)

        if SupportedAugmentations.RANDOM_ERASE in self.augmentation_enums:
            spectrogram = RandomErasing(p=1)(spectrogram)

        if SupportedAugmentations.RANDOM_PIXELS in self.augmentation_enums:
            mask = (
                torch.FloatTensor(*spectrogram.shape).uniform_()
                < self.hide_random_pixels_p
            )
            num_of_masked_elements = mask.sum()
            noise = torch.normal(
                mean=spec_mean,
                std=self.std_noise,
                size=(num_of_masked_elements,),
            )

            spectrogram[mask] = noise

        if not return_batch_dim:
            return spectrogram.squeeze(0)

        return spectrogram


def get_augmentations(config: ConfigDefault):
    train_kwargs = dict(
        augmentation_enums=config.augmentations,
        sampling_rate=config.sampling_rate,
        **config.aug_kwargs,
    )
    val_kwargs = {**train_kwargs, "augmentation_enums": []}

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
