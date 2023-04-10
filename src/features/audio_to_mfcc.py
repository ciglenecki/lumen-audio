import librosa
import numpy as np
import torch
import torchvision.transforms.functional as F

import src.config.config_defaults as config_defaults
from src.config.config import config
from src.features.audio_transform_base import AudioTransformBase


class MFCC(AudioTransformBase):
    """Calculates MFCC (mel-frequency cepstral coefficients) from audio."""

    def __init__(
        self,
        n_mfcc: int,
        n_fft: int = config.n_fft,
        hop_length: int = config.hop_length,
        n_mels: int = config.n_mels,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def process(
        self,
        audio: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio = self.waveform_augmentation(audio)
        spectrogram = librosa.feature.mfcc(
            y=audio,
            sr=self.sampling_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        spectrogram = self.spectrogram_augmentation(spectrogram)
        return spectrogram


class MFCCFixed(MFCC):
    """Resamples audio, extracts MFCC from audio and pads the original spectrogram to dimension of
    spectrogram for max_audio_seconds sequence."""

    def __init__(
        self,
        max_audio_seconds: int,
        image_dim: tuple[int, int],
        *args,
        **kwargs,
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

        spectrogram = librosa.feature.mfcc(
            y=audio_resampled,
            sr=self.sampling_rate,
            n_mfcc=self.n_mfcc,
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

        chunks = list(torch.tensor(spectrogram).split(self.spectrogram_dim[1], dim=1))
        # if theyre not exact padding is added for last chunk
        if not spectrogram.shape[1] % self.spectrogram_dim[1] == 0:
            w, h = chunks[-1].shape
            chunk_padded = torch.zeros(size=self.spectrogram_dim)
            chunk_padded[:w, :h] = chunks[-1]
            chunks[-1] = chunk_padded

        for i, _ in enumerate(chunks):
            chunks[i] = chunks[i].reshape(1, 1, *self.spectrogram_dim)
            chunks[i] = F.resize(chunks[i], size=self.image_dim, antialias=True)[0][0]
            chunks[i] = chunks[i].type(torch.float32)

        return chunks


class MFCCFixedRepeat(MFCCFixed):
    """Calls MFCC and repeats the output 3 times.

    This is useful for mocking RGB channels.
    """

    def __init__(self, repeat=config_defaults.DEFAULT_RGB_CHANNELS, **kwargs):
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
