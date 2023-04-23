import librosa
import numpy as np
import torch
import torch.nn.functional
from src.features.audio_transform_base import AudioTransformBase
from src.features.chunking import chunk_image_by_width
from src.config.config_defaults import (
    DEFAULT_MULTI_SPECTROGRAM_MEAN,
    DEFAULT_MULTI_SPECTROGRAM_STD
)

class MultiSpectrogram(AudioTransformBase):
    """Resamples audio, extracts melspectrogram from audio and pads the original spectrogram to
    dimension of spectrogram for max_num_width_samples sequence."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        image_size: tuple[int, int],
        use_rgb: bool = True,
        normalize_audio=True,
        normalize_image=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.image_size = image_size
        self.use_rgb = use_rgb
        self.normalize_audio = normalize_audio
        self.normalize_image = normalize_image

    def __call__(
        self,
        audio: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor]:
        if self.waveform_augmentation is not None:
            audio = self.waveform_augmentation(audio)

        melspec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, 
            sr=self.sampling_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )

        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # if self.spectrogram_augmentation is not None:
        #     spectrogram = self.spectrogram_augmentation(spectrogram)
        # else:
        #     spectrogram = torch.tensor(spectrogram)

        melspec_chunks = chunk_image_by_width(self.image_size, torch.tensor(melspec), "repeat")
        spectral_centroid_chunks = chunk_image_by_width(self.image_size, torch.tensor(spectral_centroid), "repeat")
        chroma_chunks = chunk_image_by_width(self.image_size, torch.tensor(chroma), "repeat")
        multi_spectrogram = torch.stack([melspec_chunks, spectral_centroid_chunks, chroma_chunks]).permute(1, 0, 2, 3)

        if self.normalize_image:
            multi_spectrogram = self.normalize_spectrogram(multi_spectrogram)

        return multi_spectrogram

    @staticmethod
    def normalize_spectrogram(spectrogram: torch.Tensor):
        # https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html
        return (
            spectrogram - DEFAULT_MULTI_SPECTROGRAM_MEAN.view(1, 3, 1, 1)
        ) / DEFAULT_MULTI_SPECTROGRAM_STD.view(1, 3, 1, 1)

    @staticmethod
    def undo_normalize_spectrogram(spectrogram: torch.Tensor):
        return (
            spectrogram * DEFAULT_MULTI_SPECTROGRAM_MEAN.view(1, 3, 1, 1)
        ) + DEFAULT_MULTI_SPECTROGRAM_STD.view(1, 3, 1, 1)


if __name__ == "__main__":
    pass
