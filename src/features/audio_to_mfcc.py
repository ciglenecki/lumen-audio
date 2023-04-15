import librosa
import numpy as np
import torch
import torchvision.transforms.functional as F

import src.config.config_defaults as config_defaults
from src.config.config_defaults import (
    DEFAULT_MFCC_FIXED_REPEAT_MEAN,
    DEFAULT_MFCC_FIXED_REPEAT_STD,
)
from src.features.audio_transform_base import AudioTransformBase
from src.utils.utils_audio import caculate_spectrogram_duration_in_seconds


class MFCC(AudioTransformBase):
    """Calculates MFCC (mel-frequency cepstral coefficients) from audio."""

    def __init__(
        self,
        n_mfcc: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
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
        if self.waveform_augmentation is not None:
            audio = self.waveform_augmentation(audio)

        spectrogram = librosa.feature.mfcc(
            y=audio,
            sr=self.sampling_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        if self.spectrogram_augmentation is not None:
            spectrogram = self.spectrogram_augmentation(spectrogram)

        return spectrogram


class MFCCFixed(MFCC):
    """Resamples audio, extracts MFCC from audio and pads the original spectrogram to dimension of
    spectrogram for max_num_width_samples sequence."""

    def __init__(
        self,
        image_size: tuple[int, int],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.image_size = image_size

        self.max_num_width_samples = caculate_spectrogram_duration_in_seconds(
            sampling_rate=self.sampling_rate,
            hop_size=self.hop_length,
            image_width=self.image_size[0],
        )

        FAKE_SAMPLE_RATE = 44_100
        dummy_audio = np.random.random(
            size=(int(self.max_num_width_samples * FAKE_SAMPLE_RATE),)
        )
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

        import matplotlib.pyplot as plt

        plt.imshow(spectrogram)
        plt.show()

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
            chunks[i] = F.resize(chunks[i], size=self.image_size, antialias=False)[0][0]
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

        reshaped_mean = DEFAULT_MFCC_FIXED_REPEAT_MEAN.view(-1, 1, 1)
        reshaped_std = DEFAULT_MFCC_FIXED_REPEAT_STD.view(-1, 1, 1)
        for i, _ in enumerate(spectrogram_chunks):
            spectrogram_chunks[i] = spectrogram_chunks[i].repeat(1, self.repeat, 1, 1)[
                0
            ]
            spectrogram_chunks[i] = (
                spectrogram_chunks[i] - reshaped_mean
            ) / reshaped_std

        return spectrogram_chunks
