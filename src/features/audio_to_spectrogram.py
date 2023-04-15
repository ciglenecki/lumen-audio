import librosa
import numpy as np
import torch
import torch.nn.functional
import torchvision.transforms.functional

from src.config.argparse_with_config import ArgParseWithConfig
from src.config.config_defaults import (
    DEFAULT_MEL_SPECTROGRAM_FIXED_REPEAT_MEAN,
    DEFAULT_MEL_SPECTROGRAM_FIXED_REPEAT_STD,
    get_default_config,
)
from src.features.audio_transform_base import AudioTransformBase
from src.features.chunking import chunk_image_by_width, collate_fn_spectrogram
from src.utils.utils_audio import (
    caculate_spectrogram_duration_in_seconds,
    play_audio,
    plot_spectrograms,
)
from src.utils.utils_dataset import get_example_val_sample


class MelSpectrogramFixed(AudioTransformBase):
    """Resamples audio, extracts melspectrogram from audio and pads the original spectrogram to
    dimension of spectrogram for max_num_width_samples sequence."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        image_size: tuple[int, int],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.image_size = image_size

    def process(
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

        spectrogram_chunks = chunk_image_by_width(self.image_size, spectrogram)
        return spectrogram_chunks


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

        reshaped_mean = DEFAULT_MEL_SPECTROGRAM_FIXED_REPEAT_MEAN.view(-1, 1, 1)
        reshaped_std = DEFAULT_MEL_SPECTROGRAM_FIXED_REPEAT_STD.view(-1, 1, 1)
        for i, _ in enumerate(spectrogram_chunks):
            spectrogram_chunks[i] = spectrogram_chunks[i].repeat(1, self.repeat, 1, 1)[
                0
            ]
            spectrogram_chunks[i] = (
                spectrogram_chunks[i] - reshaped_mean
            ) / reshaped_std

        return spectrogram_chunks


if __name__ == "__main__":
    parser = ArgParseWithConfig()
    args, config, pl_args = parser.parse_args()
    audio = get_example_val_sample(config.sampling_rate)
    transform = MelSpectrogramFixed(
        sampling_rate=config.sampling_rate,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
        n_mels=config.n_mels,
        image_size=config.image_size,
        spectrogram_augmentation=None,
        waveform_augmentation=None,
    )
    spectrogram = transform.process(audio)
    # spec_plot = torchvision.transforms.functional.resize(
    #     spectrogram, size=(config.n_mels, spectrogram.shape[-1]), antialias=False
    # )

    # plot_spectrograms(
    #     spec_plot,
    #     sampling_rate=config.sampling_rate,
    #     hop_length=config.hop_length,
    #     n_fft=config.n_fft,
    #     n_mels=config.n_mels,
    # )

    # audio_reconstructed = librosa.feature.inverse.mel_to_audio(
    #     undo_image_chunking(spectrogram, config.n_mels).detach().numpy(),
    #     sr=config.sampling_rate,
    #     n_fft=config.n_fft,
    #     hop_length=config.hop_length,
    # )

    out = collate_fn_spectrogram(
        [
            (
                spectrogram,
                torch.rand(11),
                torch.tensor([1]),
            ),
            (
                spectrogram,
                torch.rand(11),
                torch.tensor([3]),
            ),
        ]
    )
    images, _, file_indices, _ = out
    id_1 = file_indices == 0
    spectrogram_get = images[id_1]
    reconstructed = undo_image_chunking(spectrogram_get, config.n_mels)
    plot_spectrograms(
        reconstructed,
        n_fft=config.n_fft,
        sampling_rate=config.sampling_rate,
        n_mels=config.n_mels,
        hop_length=config.hop_length,
    )

####################################################################################
####################################################################################
# TEST
####################################################################################
####################################################################################


def test_chunking():
    config = get_default_config()
    audio = get_example_val_sample(config.sampling_rate)
    transform = MelSpectrogramFixed(
        sampling_rate=config.sampling_rate,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
        n_mels=config.n_mels,
        image_size=config.image_size,
        spectrogram_augmentation=None,
        waveform_augmentation=None,
    )
    spectrogram_original = librosa.feature.melspectrogram(
        y=audio,
        sr=config.sampling_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
    )
    spectrogram_original = torch.tensor(spectrogram_original)
    spectrogram = transform.process(audio)
    reconstructed = undo_image_chunking(spectrogram, config.n_mels)
    reconstructed = reconstructed[:, :, : spectrogram_original.shape[-1]]
    assert torch.all(torch.isclose(spectrogram_original, reconstructed))
