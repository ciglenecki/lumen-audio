import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

import src.config.config_defaults as config_defaults
from src.features.audio_transform_base import AudioTransformBase


class AudioToWav2Vec2(AudioTransformBase):
    def __init__(
        self,
        pretrained_tag=config_defaults.DEFAULT_WAV2VEC_PRETRAINED_TAG,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            pretrained_tag
        )

    def process(
        self, audio: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio = self.waveform_augmentation(audio)
        features_dict = self.feature_extractor(
            audio, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True
        )
        features = features_dict.input_values.squeeze(0)
        return features


class AudioToWav2Vec2CNN(AudioTransformBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, audio: torch.Tensor | np.ndarray) -> torch.Tensor:
        audio = self.waveform_augmentation(audio)
        audio = torch.tensor(audio)
        return audio
