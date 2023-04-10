import librosa
import numpy as np
import torch
import torch_audiomentations
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchvision.transforms import RandomErasing

from src.utils.utils_functions import EnumStr


class SupportedAugmentations(EnumStr):
    """List of supported spectrogram augmentations we use."""

    TIME_STRETCH = "time_stretch"
    FREQ_MASK = "freq_mask"
    TIME_MASK = "time_mask"
    RANDOM_ERASE = "random_erase"
    RANDOM_PIXELS = "radnom_pixels"
    COLOR_NOISE = "color_noise"
    BANDPASS_FILTER = "bandpass"
    PITCH = "pitch"
    TIMEINV = "timeinv"
    CONCAT_TWO = "concat_two"


class WaveformAugmentation:
    def __init__(
        self,
        augmentation_enums: list[SupportedAugmentations],
        sampling_rate: int,
        stretch_factors=[0.8, 1.2],
        time_inversion_p=0.5,
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
            p=1, sample_rate=self.sampling_rate
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
        freq_mask_param=30,
        time_mask_param=30,
        hide_random_pixels_p=0.25,
        std_noise=0.01,
        **kwargs,
    ):
        self.augmentation_enums = augmentation_enums
        self.sampling_rate = sampling_rate
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.hide_random_pixels_p = hide_random_pixels_p
        self.std_noise = std_noise

    def __call__(self, spectrogram: torch.Tensor | np.ndarray) -> torch.Tensor:
        if len(self.augmentation_enums) == 0:
            return spectrogram

        return_batch_dim = True
        if isinstance(spectrogram, np.ndarray):
            spectrogram = torch.tensor(spectrogram)
        if len(spectrogram.shape) == 2:
            return_batch_dim = False
            spectrogram = spectrogram.unsqueeze(0)

        if SupportedAugmentations.FREQ_MASK in self.augmentation_enums:
            spectrogram = FrequencyMasking(freq_mask_param=self.freq_mask_param)(
                spectrogram
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
                mean=spectrogram.mean(),
                std=self.std_noise,
                size=(num_of_masked_elements,),
            )

            spectrogram[mask] = noise

        if not return_batch_dim:
            return spectrogram.squeeze(0)

        return spectrogram


def get_augmentations(args):
    train_kwargs = dict(
        augmentation_enums=args.augmentations,
        sampling_rate=args.sampling_rate,
        **args.aug_kwargs,
    )
    val_kwargs = dict(
        augmentation_enums=[],
        sampling_rate=args.sampling_rate,
    )
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
