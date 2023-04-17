import librosa
import numpy as np
import torch
import torch.nn.functional
import torchvision.transforms.functional

from src.config.argparse_with_config import ArgParseWithConfig
from src.config.config_defaults import (
    DEFAULT_MEL_SPECTROGRAM_MEAN,
    DEFAULT_MEL_SPECTROGRAM_STD,
    get_default_config,
)
from src.features.audio_transform_base import AudioTransformBase
from src.features.chunking import (
    chunk_image_by_width,
    collate_fn_spectrogram,
    undo_image_chunking,
)
from src.utils.utils_audio import plot_spectrograms
from src.utils.utils_dataset import (
    add_rgb_channel,
    get_example_val_sample,
    remove_rgb_channel,
)


class MelSpectrogram(AudioTransformBase):
    """Resamples audio, extracts melspectrogram from audio and pads the original spectrogram to
    dimension of spectrogram for max_num_width_samples sequence."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        image_size: tuple[int, int],
        use_rgb: bool = True,
        normalize=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.image_size = image_size
        self.use_rgb = use_rgb
        self.normalize = normalize

    def __call__(
        self,
        audio: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor]:
        if self.waveform_augmentation is not None:
            audio = self.waveform_augmentation(audio)

        spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        if self.spectrogram_augmentation is not None:
            spectrogram = self.spectrogram_augmentation(spectrogram)
        else:
            spectrogram = torch.tensor(spectrogram)

        spectrogram_chunks = chunk_image_by_width(
            self.image_size, spectrogram, "repeat"
        )

        if self.normalize:
            spectrogram_chunks = self.normalize_spectrogram(spectrogram_chunks)
        if self.use_rgb:
            spectrogram_chunks = add_rgb_channel(spectrogram_chunks)

        return spectrogram_chunks

    def undo(self, spectrogram: torch.Tensor):
        spectrogram = remove_rgb_channel(spectrogram)
        spectrogram = torchvision.transforms.functional.resize(
            spectrogram, size=(self.n_mels, spectrogram.shape[-1]), antialias=False
        )
        spectrogram = (
            spectrogram * DEFAULT_MEL_SPECTROGRAM_STD
        ) + DEFAULT_MEL_SPECTROGRAM_MEAN

        return spectrogram

    @staticmethod
    def normalize_spectrogram(spectrogram: torch.Tensor):
        # https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html
        return (
            spectrogram - DEFAULT_MEL_SPECTROGRAM_MEAN
        ) / DEFAULT_MEL_SPECTROGRAM_STD

    @staticmethod
    def undo_normalize_spectrogram(spectrogram: torch.Tensor):
        return (
            spectrogram * DEFAULT_MEL_SPECTROGRAM_STD
        ) + DEFAULT_MEL_SPECTROGRAM_MEAN


def test_chunking():
    config = get_default_config()
    audio = get_example_val_sample(config.sampling_rate)
    transform = MelSpectrogram(
        sampling_rate=config.sampling_rate,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
        n_mels=config.n_mels,
        image_size=config.image_size,
        spectrogram_augmentation=None,
        waveform_augmentation=None,
        max_num_width_samples=config.max_num_width_samples,
        normalize=True,
    )
    spectrogram_original = librosa.feature.melspectrogram(
        y=audio,
        sr=config.sampling_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
    )

    spectrogram_original_rgb = add_rgb_channel(
        torch.tensor(spectrogram_original).unsqueeze(0)
    )
    assert torch.all(
        torch.isclose(
            spectrogram_original_rgb[0, 0, ...], torch.tensor(spectrogram_original)
        )
    ), "RGB spectrogram isn't good."

    spectrogram = transform(audio)
    spectrogram_unchunked = undo_image_chunking(spectrogram, config.n_mels)
    spectrogram_unchunked = spectrogram_unchunked[
        ..., : spectrogram_original_rgb.shape[-1]
    ]
    spectrogram_unchunked = transform.undo_normalize_spectrogram(spectrogram_unchunked)
    assert torch.all(
        torch.isclose(spectrogram_original_rgb, spectrogram_unchunked, atol=1e-5)
    ), "Reconstructred spectrogram isn't good"


if __name__ == "__main__":
    parser = ArgParseWithConfig()
    args, config, pl_args = parser.parse_args()
    audio = get_example_val_sample(config.sampling_rate)
    transform = MelSpectrogram(
        sampling_rate=config.sampling_rate,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
        n_mels=config.n_mels,
        image_size=config.image_size,
        spectrogram_augmentation=None,
        waveform_augmentation=None,
        max_num_width_samples=config.max_num_width_samples,
        normalize=False,
    )
    spectrogram = transform(audio)
    out = collate_fn_spectrogram(
        [
            (spectrogram, torch.ones(11), torch.tensor([1])),
            (spectrogram, torch.ones(11), torch.tensor([3])),
        ]
    )
    images, _, file_indices, _ = out
    spectrogram_chunls = images[file_indices == 0]
    spectrogram_reconstructed = undo_image_chunking(spectrogram_chunls, config.n_mels)
    spectrogram_reconstructed = remove_rgb_channel(spectrogram_reconstructed)
    plot_spectrograms(
        spectrogram_reconstructed,
        n_fft=config.n_fft,
        sampling_rate=config.sampling_rate,
        n_mels=config.n_mels,
        hop_length=config.hop_length,
    )
