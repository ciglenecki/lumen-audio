import librosa
import numpy as np
import torch

from src.config.config_defaults import DEFAULT_MFCC_MEAN, DEFAULT_MFCC_STD
from src.features.audio_transform_base import AudioTransformBase
from src.features.chunking import chunk_image_by_width
from src.utils.utils_dataset import add_rgb_channel


class MFCC(AudioTransformBase):
    """Calculates MFCC (mel-frequency cepstral coefficients) from audio."""

    def __init__(
        self,
        n_mfcc: int,
        n_fft: int,
        hop_length: int,
        image_size: tuple[int, int],
        n_mels: int,
        use_rgb: bool = True,
        normalize_audio=True,
        normalize_image=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.image_size = image_size
        self.n_mels = n_mels
        self.use_rgb = use_rgb
        self.normalize_audio = normalize_audio
        self.normalize_image = normalize_image

    def __call__(
        self,
        audio: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor]:
        if self.waveform_augmentation is not None:
            audio = self.waveform_augmentation(audio)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sampling_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        if self.normalize_image:
            mfcc = self.normalize_mfcc(mfcc)

        if self.spectrogram_augmentation is not None:
            mfcc = self.spectrogram_augmentation(mfcc)
        else:
            mfcc = torch.tensor(mfcc)

        mfcc_chunks = chunk_image_by_width(self.image_size, mfcc, DEFAULT_MFCC_MEAN)

        if self.use_rgb:
            mfcc_chunks = add_rgb_channel(mfcc_chunks)

        return mfcc_chunks

    def normalize_mfcc(self, mfcc: torch.tensor):
        # https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html
        return (mfcc - DEFAULT_MFCC_MEAN) / DEFAULT_MFCC_STD

    def undo_normalize_mfcc(self, mfcc: torch.tensor):
        return (mfcc * DEFAULT_MFCC_STD) + DEFAULT_MFCC_MEAN
