from abc import ABC, abstractmethod

import librosa
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchaudio.transforms import (
    FrequencyMasking,
    MelScale,
    Spectrogram,
    TimeMasking,
    TimeStretch,
)
from transformers import ASTFeatureExtractor

import src.config_defaults as config_defaults
from src.utils_functions import EnumStr, MultiEnum


def stereo_to_mono(audio: torch.Tensor | np.ndarray):
    if isinstance(audio, torch.Tensor):
        return audio.sum(dim=1) / 2
    elif isinstance(audio, np.ndarray):
        return audio.sum(axis=-1) / 2


class AudioTransformBase(ABC):
    """Base class for all audio transforms. Ideally, each audio transform class should be self
    contained and shouldn't depened on the outside context.

    Audio transfrom can be model dependent. We can create audio transforms which work only for one
    model and that's fine.
    """

    @abstractmethod
    def process(
        self,
        audio: np.ndarray,
        labels: torch.Tensor | np.ndarray,
        orig_sampling_rate: int,
        sampling_rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Function which prepares everything for model's .forward() function. It creates the
        spectrogram from audio and prepares the labels.

        Args:
            audio: audio data
            labels: _description_
            orig_sampling_rate: _description_
            sampling_rate: _description_

        Returns:
            tuple[torch.Tensor, torch.Tensor]: _description_
        """


class AudioTransformAST(AudioTransformBase):

    """Resamples audio, converts it to mono, does AST feature extraction which extracts spectrogram
    (mel filter banks) from audio."""

    def __init__(
        self,
        ast_pretrained_tag=config_defaults.DEFAULT_AST_PRETRAINED_TAG,
    ):
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(ast_pretrained_tag)

    def process(
        self,
        audio: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        orig_sampling_rate: int,
        sampling_rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels)

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().numpy()

        audio_mono = librosa.to_mono(audio)
        audio_resampled = librosa.resample(
            audio_mono,
            orig_sr=orig_sampling_rate,
            target_sr=sampling_rate,
        )
        features = self.feature_extractor(
            audio_resampled,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        # mel filter banks
        spectrogram = features["input_values"].squeeze(dim=0)
        return spectrogram, labels


class AudioTransformMelSpectrogram(AudioTransformBase):

    """Resamples audio, extracts melspectrogram from audio, resizes it to the given dimensions."""

    def __init__(self, n_fft=2048, hop_length=512, n_mels=10, dim=(384, 384)):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.dim = dim

    def process(
        self,
        audio: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        orig_sampling_rate: int,
        sampling_rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels)

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().numpy()

        audio_resampled = librosa.resample(
            audio,
            orig_sr=orig_sampling_rate,
            target_sr=sampling_rate,
        )

        spectrogram = librosa.feature.melspectrogram(
            y=audio_resampled,
            sr=sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        spectrogram = spectrogram.reshape(1, 1, *spectrogram.shape)
        spectrogram = F.resize(torch.tensor(spectrogram), size=self.dim)
        spectrogram = spectrogram.reshape(1, *self.dim)

        return spectrogram, labels


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
        labels: torch.Tensor | np.ndarray,
        orig_sampling_rate: int,
        sampling_rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        spectrogram, labels = super().process(
            audio, labels, orig_sampling_rate, sampling_rate
        )
        spectrogram = spectrogram.repeat(1, self.repeat, 1, 1).reshape(
            self.repeat, *self.dim
        )

        return spectrogram, labels


########################################################################################################################


class AudioAugmentation(torch.nn.Module):
    """Taken from: https://pytorch.org/audio/stable/transforms.html.

    Define custom feature extraction pipeline.

    1. Convert to power spectrogram
    2. Apply augmentations
    3. Convert to mel-scale
    """

    def __init__(
        self,
        n_fft=1024,
        n_mel=256,
        stretch_factor=0.8,
        sample_rate=config_defaults.DEFAULT_SAMPLING_RATE,
    ):
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

        self.mel_scale = MelScale(
            n_mels=n_mel, sample_rate=sample_rate, n_stft=n_fft // 2 + 1
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Convert to power spectrogram
        spec = self.spec(waveform)

        # Apply SpecAugment
        spec = self.spec_aug(spec)

        # Convert to mel-scale
        mel = self.mel_scale(spec)

        return mel


class UnsupportedAudioTransforms(ValueError):
    pass


class AudioTransforms(EnumStr):
    """List of supported AudioTransforms we use."""

    AST = "ast"
    MEL_SPECTROGRAM = "mel_spectrogram"
    MEL_SPECTROGRAM_REPEAT = "mel_spectrogram_repeat"


def get_audio_transform(audio_transform_enum: AudioTransforms) -> AudioTransformBase:
    if audio_transform_enum is AudioTransforms.AST:
        return AudioTransformAST(
            ast_pretrained_tag=config_defaults.DEFAULT_AST_PRETRAINED_TAG
        )
    elif audio_transform_enum is AudioTransforms.MEL_SPECTROGRAM:
        return AudioTransformMelSpectrogram()
    elif audio_transform_enum is AudioTransforms.MEL_SPECTROGRAM_REPEAT:
        return AudioTransformMelSpectrogramRepeat(repeat=3)
    raise UnsupportedAudioTransforms(f"Unsupported transform {audio_transform_enum}")
