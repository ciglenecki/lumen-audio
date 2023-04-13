import numpy as np
import torch
from transformers import ASTFeatureExtractor

from src.config.config_train import config
from src.features.audio_transform_base import AudioTransformBase


class AudioTransformAST(AudioTransformBase):

    """Resamples audio, converts it to mono, does AST feature extraction which extracts spectrogram
    (mel filter banks) from audio.

    Warning: resampling should be done here. AST does the job.
    """

    def __init__(
        self,
        pretrained_tag=config.pretrained_tag,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_tag)

    def process(
        self, audio: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio = self.waveform_augmentation(audio)

        spectrogram = self.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )["input_values"]

        spectrogram = self.spectrogram_augmentation(spectrogram)  # [Batch, 1024, 128]

        # TODO: chunking
        spectrogram_chunks = spectrogram
        assert len(spectrogram_chunks.shape) == 3, "Spectrogram chunks are 2D images"
        return spectrogram_chunks
