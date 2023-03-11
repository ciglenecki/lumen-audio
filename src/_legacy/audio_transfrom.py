import torch
from torchaudio.transforms import (
    FrequencyMasking,
    MelScale,
    Spectrogram,
    TimeMasking,
    TimeStretch,
)

import src.config_defaults as config_defaults


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
