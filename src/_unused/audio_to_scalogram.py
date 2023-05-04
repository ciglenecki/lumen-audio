import numpy as np
import torch

from _unused.audio_to_wavelet import WaveletConv
from src.features.audio_transform_base import AudioTransformBase


class WaveletTransform(AudioTransformBase):
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
        self.wavelet = WaveletConv()

    def __call__(self, audio: torch.Tensor | np.ndarray) -> tuple[torch.Tensor]:
        if isinstance(audio, np.ndarray):
            audio = torch.tesor(audio)
        audio = audio.unsqueeze(1)
        return self.wavelet(audio).squeeze(1)
