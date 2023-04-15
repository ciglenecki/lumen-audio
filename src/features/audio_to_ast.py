import numpy as np
import torch
from transformers import ASTFeatureExtractor

from src.config.argparse_with_config import ArgParseWithConfig
from src.config.config_defaults import TAG_AST_AUDIOSET
from src.features.audio_to_spectrogram import MelSpectrogramFixedRepeat
from src.features.audio_transform_base import AudioTransformBase
from src.features.chunking import chunk_image_by_width
from src.utils.utils_dataset import get_example_val_sample


class AudioTransformAST(AudioTransformBase):

    """Resamples audio, converts it to mono, does AST feature extraction which extracts spectrogram
    (mel filter banks) from audio.

    Warning: resampling should be done here. AST does the job.
    """

    def __init__(
        self,
        pretrained_tag,
        hop_length: int,
        n_mels: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_mels = n_mels
        self.hop_length = hop_length
        if pretrained_tag is None:
            self.feature_extractor = ASTFeatureExtractor(
                sampling_rate=self.sampling_rate,
                num_mel_bins=self.n_mels,
                max_length=self.max_num_width_samples,
            )
        else:
            self.feature_extractor: ASTFeatureExtractor = (
                ASTFeatureExtractor.from_pretrained(pretrained_tag)
            )
        self.image_size = (
            self.feature_extractor.num_mel_bins,
            self.feature_extractor.max_length,
        )

    def process(
        self, audio: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.waveform_augmentation is not None:
            audio = self.waveform_augmentation(audio)

        spectrogram = self.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )["input_values"]

        if self.spectrogram_augmentation is not None:
            spectrogram = self.spectrogram_augmentation(
                spectrogram
            )  # [Batch, 1024, 128]

        spectrogram_chunks = chunk_image_by_width(self.image_size, spectrogram)
        assert len(spectrogram_chunks.shape) == 3, "Spectrogram chunks are 2D images"
        return spectrogram_chunks


if __name__ == "__main__":
    parser = ArgParseWithConfig()
    args, config, pl_args = parser.parse_args()
    audio = get_example_val_sample(config.sampling_rate)
    transform = AudioTransformAST(
        pretrained_tag=TAG_AST_AUDIOSET,
        sampling_rate=config.sampling_rate,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
        n_mels=config.n_mels,
        image_size=config.image_size,
        spectrogram_augmentation=None,
        waveform_augmentation=None,
    )
    spectrogram = transform.process(audio)
