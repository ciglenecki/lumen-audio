import librosa
import numpy as np
import torch
import torch.nn.functional
import torchaudio
import torchvision.transforms.functional
from torchaudio.transforms import MelSpectrogram

from src.config.config_defaults import (
    DEFAULT_MEL_SPECTROGRAM_MEAN,
    DEFAULT_MEL_SPECTROGRAM_STD,
    get_default_config,
)
from src.features.audio_transform_base import AudioTransformBase
from src.features.augmentations import get_augmentations
from src.features.chunking import (
    chunk_image_by_width,
    set_image_height,
    undo_image_chunking,
)
from src.features.torch_melspec import MelSpectrogramTorchFix
from src.utils.utils_audio import repeat_self_to_length
from src.utils.utils_dataset import (
    add_rgb_channel,
    get_example_val_sample,
    remove_rgb_channel,
)


class MelspecGPU(AudioTransformBase):
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
        self.melspec = MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            norm="slaney",
            mel_scale="slaney",
        )

    def __call__(
        self,
        audio: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor]:
        # This is already on GPU at this point
        # after collate function
        if audio.ndim == 2:
            audio = audio.unsqueeze(0)

        if self.waveform_augmentation is not None:
            audio = self.waveform_augmentation(audio)

        spectrogram = self.melspec(audio)

        if self.spectrogram_augmentation is not None:
            spectrogram = self.spectrogram_augmentation(spectrogram)

        spectrogram = spectrogram.abs()
        spectrogram = repeat_self_to_length(spectrogram, self.image_size[-1])
        spectrogram = set_image_height(spectrogram, self.image_size[1])

        if self.normalize_image:
            spectrogram = self.normalize_spectrogram(spectrogram)

        if self.use_rgb:
            spectrogram = add_rgb_channel(spectrogram)

        return spectrogram

    def undo(self, spectrogram: torch.Tensor):
        if self.use_rgb:
            spectrogram = remove_rgb_channel(spectrogram)

        interpolation = (
            torchvision.transforms.functional.InterpolationMode.NEAREST_EXACT
        )
        spectrogram = torchvision.transforms.functional.resize(
            spectrogram,
            size=(self.n_mels, spectrogram.shape[-1]),
            interpolation=interpolation,
            antialias=False,
        )

        if self.normalize_image:
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
    config.image_size = (384, 384)
    audio = get_example_val_sample(config.sampling_rate)
    audio = torch.tensor(audio).unsqueeze(0).unsqueeze(0)

    (
        train_spectrogram_augmentation,
        train_waveform_augmentation,
        _,
        _,
    ) = get_augmentations(config)

    transform = MelspecGPU(
        spectrogram_augmentation=train_spectrogram_augmentation,
        waveform_augmentation=train_waveform_augmentation,
        sampling_rate=config.sampling_rate,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
        n_mels=config.n_mels,
        image_size=config.image_size,
        max_num_width_samples=config.max_num_width_samples,
        normalize_audio=True,
    )
    s = transform(audio)
    # spectrogram_original = librosa.feature.melspectrogram(
    #     y=audio,
    #     sr=config.sampling_rate,
    #     n_fft=config.n_fft,
    #     hop_length=config.hop_length,
    #     n_mels=config.n_mels,
    # )

    # spectrogram_original_rgb = add_rgb_channel(
    #     torch.tensor(spectrogram_original).unsqueeze(0)
    # )
    # assert torch.all(
    #     torch.isclose(
    #         spectrogram_original_rgb[0, 0, ...], torch.tensor(spectrogram_original)
    #     )
    # ), "RGB spectrogram isn't good."

    # spectrogram = transform(audio)
    # spectrogram_unchunked = undo_image_chunking(spectrogram, config.n_mels)
    # spectrogram_unchunked = spectrogram_unchunked[
    #     ..., : spectrogram_original_rgb.shape[-1]
    # ]
    # spectrogram_unchunked = transform.undo_normalize_spectrogram(spectrogram_unchunked)
    # assert torch.all(
    #     torch.isclose(spectrogram_original_rgb, spectrogram_unchunked, atol=1e-5)
    # ), "Reconstructred spectrogram isn't good"


if __name__ == "__main__":
    test_chunking()
