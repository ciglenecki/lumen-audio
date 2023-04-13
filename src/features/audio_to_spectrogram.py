import librosa
import numpy as np
import torch
import torchvision.transforms.functional as F

from src.config.config_train import config
from src.features.audio_transform_base import AudioTransformBase


class MelSpectrogramOurs(AudioTransformBase):
    """Resamples audio and extracts melspectrogram from audio."""

    def __init__(
        self,
        n_fft: int = config.n_fft,
        hop_length: int = config.hop_length,
        n_mels: int = config.n_mels,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def process(
        self,
        audio: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        audio = self.waveform_augmentation(audio)
        spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        spectrogram = self.spectrogram_augmentation(spectrogram)
        return spectrogram


# TODO: unify and change 4 classes so it's clear what the do, we might need a single class
class MelSpectrogramResize(MelSpectrogramOurs):
    """Resamples audio, extracts melspectrogram from audio, resizes it to the given dimensions."""

    def __init__(self, image_dim: tuple[int, int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_dim = image_dim

    def process(
        self,
        audio: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spectrogram = super().process(audio)

        spectrogram = spectrogram.reshape(1, 1, *spectrogram.shape)
        spectrogram = F.resize(
            torch.tensor(spectrogram), size=self.image_dim, antialias=True
        )
        spectrogram = spectrogram.reshape(1, *self.image_dim)
        return spectrogram


class MelSpectrogramFixed(MelSpectrogramOurs):
    """Resamples audio, extracts melspectrogram from audio and pads the original spectrogram to
    dimension of spectrogram for max_audio_seconds sequence."""

    def __init__(
        self, max_audio_seconds: int, image_dim: tuple[int, int], *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.max_audio_seconds = max_audio_seconds
        self.image_dim = image_dim

        FAKE_SAMPLE_RATE = 44_100
        dummy_audio = np.random.random(size=(max_audio_seconds * FAKE_SAMPLE_RATE,))
        audio_resampled = librosa.resample(
            dummy_audio,
            orig_sr=FAKE_SAMPLE_RATE,
            target_sr=self.sampling_rate,
        )

        spectrogram = librosa.feature.melspectrogram(
            y=audio_resampled,
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        self.spectrogram_dim = spectrogram.shape

    def process(
        self,
        audio: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor]:
        spectrogram = super().process(audio)
        time_dimension = self.spectrogram_dim[1]
        chunks = list(torch.tensor(spectrogram).split(time_dimension, dim=1))

        # Last chunk might be cut off which mean the time dimension (image width) will be smaller
        # Padding is added for last chunk if it isn't time_dimension in width
        if not spectrogram.shape[1] % time_dimension == 0:
            w, h = chunks[-1].shape
            chunk_padded = torch.zeros(size=self.spectrogram_dim)  # Create empty image
            chunk_padded[:w, :h] = chunks[-1]  # Paste the last chunk on the canvas
            chunks[-1] = chunk_padded

        for i, _ in enumerate(chunks):
            chunks[i] = chunks[i].reshape(1, 1, *self.spectrogram_dim)
            chunks[i] = F.resize(chunks[i], size=self.image_dim, antialias=True)[0][0]
            chunks[i] = chunks[i].float()

        return chunks


class MelSpectrogramFixedRepeat(MelSpectrogramFixed):
    """Calls MelSpectrogramFixed and repeats the output 3 times.

    This is useful for mocking RGB channels.
    """

    def __init__(self, repeat=3, **kwargs):
        super().__init__(**kwargs)
        self.repeat = repeat

    def process(
        self,
        audio: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spectrogram_chunks = super().process(audio)
        for i, _ in enumerate(spectrogram_chunks):
            spectrogram_chunks[i] = spectrogram_chunks[i].repeat(1, self.repeat, 1, 1)[
                0
            ]

        return spectrogram_chunks


class MelSpectrogramResizedRepeat(MelSpectrogramResize):
    """Calls MelSpectrogramResize and repeats the output 3 times.

    This is useful for mocking RGB channels.
    """

    def __init__(self, repeat=3, **kwargs):
        super().__init__(**kwargs)
        self.repeat = repeat

    def process(
        self,
        audio: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spectrogram = super().process(audio)
        spectrogram = spectrogram.repeat(1, self.repeat, 1, 1)[0]

        return spectrogram
