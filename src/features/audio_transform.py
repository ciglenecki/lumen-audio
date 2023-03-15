from abc import ABC, abstractmethod

import librosa
import numpy as np
import torch
from torchaudio.transforms import FrequencyMasking, MelSpectrogram, TimeMasking
from torchvision.transforms import RandomErasing
from transformers import ASTFeatureExtractor

import src.config.config_defaults as config_defaults
from src.utils.utils_audio import stereo_to_mono
from src.utils.utils_functions import EnumStr


class UnsupportedAudioTransforms(ValueError):
    pass


class AudioTransforms(EnumStr):
    """List of supported AudioTransforms we use."""

    AST = "ast"
    MEL_SPECTROGRAM = "mel_spectrogram"
    MEL_SPECTROGRAM_REPEAT = "mel_spectrogram_repeat"


class SupportedSpecAugs(EnumStr):
    """List of supported spectrogram augmentations we use."""

    TIME_STRETCH = "time_stretch"
    FREQ_MASK = "freq_mask"
    TIME_MASK = "time_mask"
    RANDOM_ERASE = "random_erase"
    RANDOM_PIXELS = "radnom_pixels"


class AudioTransformBase(ABC):
    """Base class for all audio transforms. Ideally, each audio transform class should be self
    contained and shouldn't depened on the outside context.

    Audio transfrom can be model dependent. We can create audio transforms which work only for one
    model and that's fine.
    """

    def __init__(
        self,
        spec_aug_enums: list[SupportedSpecAugs] = [
            SupportedSpecAugs.TIME_STRETCH,
            SupportedSpecAugs.FREQ_MASK,
            SupportedSpecAugs.TIME_MASK,
            SupportedSpecAugs.RANDOM_ERASE,
            SupportedSpecAugs.RANDOM_PIXELS,
        ],
        stretch_factors=[0.8, 1.2],
        freq_mask_param=30,
        time_mask_param=30,
        hide_random_pixels_p=0.25,
        std_noise=0.01,
    ) -> None:
        super().__init__()
        self.spec_aug_enums = spec_aug_enums
        self.stretch_factors = stretch_factors
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.has_augmentations = len(spec_aug_enums) > 0
        self.hide_random_pixels_p = hide_random_pixels_p
        self.std_noise = std_noise

    @abstractmethod
    def process(
        self,
        audio: torch.Tensor | np.ndarray,
        sampling_rate: int,
        original_sr: int,
    ) -> tuple[torch.Tensor]:
        """Function which prepares everything for model's .forward() function. It creates the
        spectrogram from audio.

        Args:
            audio: audio data
            original_sr: _description_
            sampling_rate: _description_

        Returns:
            tuple[torch.Tensor, torch.Tensor]: _description_
        """


class AudioTransformAST(AudioTransformBase):

    """Resamples audio, converts it to mono, does AST feature extraction which extracts spectrogram
    (mel filter banks) from audio.

    Warning: resampling should be done here. AST does the job.
    """

    def __init__(
        self,
        ast_pretrained_tag=config_defaults.DEFAULT_AST_PRETRAINED_TAG,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(ast_pretrained_tag)

    def apply_spectrogram_augmentations(self, spectrogram: torch.Tensor):
        if SupportedSpecAugs.FREQ_MASK in self.spec_aug_enums:
            spectrogram = FrequencyMasking(freq_mask_param=self.freq_mask_param)(
                spectrogram
            )
        if SupportedSpecAugs.TIME_MASK in self.spec_aug_enums:
            spectrogram = TimeMasking(time_mask_param=self.time_mask_param)(spectrogram)

        if SupportedSpecAugs.RANDOM_ERASE in self.spec_aug_enums:
            spectrogram = RandomErasing(p=1)(spectrogram)

        if SupportedSpecAugs.RANDOM_PIXELS in self.spec_aug_enums:
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
        return spectrogram

    def process(
        self,
        audio: torch.Tensor | np.ndarray,
        sampling_rate: int,
        original_sr: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio = stereo_to_mono(audio)

        if SupportedSpecAugs.TIME_STRETCH in self.spec_aug_enums:
            l, r = self.stretch_factors
            stretch_rate = np.random.uniform(l, r)
            # size_before = len(audio)
            audio = librosa.effects.time_stretch(y=audio, rate=stretch_rate)
            # audio = audio[:size_before]

        features = self.feature_extractor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        # mel filter banks
        spectrogram = features["input_values"]
        assert (
            len(spectrogram.shape) == 3
        ), "Spectrogram has to have 3 dimensions before torch augmentations!"
        spectrogram = self.apply_spectrogram_augmentations(spectrogram)
        spectrogram = spectrogram.squeeze(dim=0)
        return spectrogram


class AudioTransformMelSpectrogram(AudioTransformBase):

    """Resamples audio, extracts melspectrogram from audio, resizes it to the given dimensions."""

    def __init__(
        self,
        sampling_rate: int = config_defaults.DEFAULT_SAMPLING_RATE,
        n_fft=config_defaults.DEFAULT_N_FFT,
        n_mels=config_defaults.DEFAULT_N_MELS,
        dim=config_defaults.DEFAULT_DIM,
    ):
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.melspec_transform = MelSpectrogram(
            sampling_rate=sampling_rate, n_fft=self.n_fft, n_mels=self.n_mels
        )
        self.dim = dim

    def process(
        self,
        audio: torch.Tensor | np.ndarray,
        original_sr: int,
        sampling_rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        audio = stereo_to_mono(audio)
        spectrogram = self.melspec_transform(audio)[0]
        return spectrogram


class AudioTransformMelSpectrogramRepeat(AudioTransformMelSpectrogram):
    """Calls AudioTransformMelSpectrogram and repeats the output 3 times.

    This is useful for mocking RGB channels.
    """

    def __init__(self, repeat=3, **kwargs):
        super().__init__(**kwargs)
        self.repeat = repeat

    def process(
        self,
        audio: torch.Tensor | np.ndarray,
        original_sr: int,
        sampling_rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        spectrogram = super().process(audio, original_sr, sampling_rate)
        spectrogram = spectrogram.repeat(1, self.repeat, 1, 1)[0]

        return spectrogram


def get_audio_transform(
    audio_transform_enum: AudioTransforms,
    spec_aug_enums: list[SupportedSpecAugs],
    **aug_kwargs,
) -> AudioTransformBase:

    if audio_transform_enum is AudioTransforms.AST:
        return AudioTransformAST(
            ast_pretrained_tag=config_defaults.DEFAULT_AST_PRETRAINED_TAG,
            spec_aug_enums=spec_aug_enums,
            **aug_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.MEL_SPECTROGRAM:
        return AudioTransformMelSpectrogram(spec_aug_enums=spec_aug_enums, **aug_kwargs)
    elif audio_transform_enum is AudioTransforms.MEL_SPECTROGRAM_REPEAT:
        return AudioTransformMelSpectrogramRepeat(
            spec_aug_enums=spec_aug_enums, **aug_kwargs
        )
    raise UnsupportedAudioTransforms(f"Unsupported transform {audio_transform_enum}")
