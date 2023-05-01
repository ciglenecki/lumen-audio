import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

from src.features.audio_transform_base import AudioTransformBase
from src.utils.utils_audio import iron_audios


class AudioToWav2Vec2(AudioTransformBase):
    def __init__(
        self,
        pretrained_tag: str | None,
        max_num_width_samples: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_num_width_samples = max_num_width_samples
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_tag)

    def __call__(
        self, audio: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.waveform_augmentation is not None:
            audio = self.waveform_augmentation(audio)

        if len(audio) > self.max_num_width_samples:
            audio_tensors: tuple[torch.Tensor] = torch.tensor(audio).split(
                self.max_num_width_samples, dim=-1
            )
            audio = [a.numpy() for a in audio_tensors]
        else:
            audio = [audio]

        audio = iron_audios(audio, target_width=self.max_num_width_samples)

        processor_out = self.processor(
            audio, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True
        )

        processed_audio = processor_out.input_values

        # note: confirmed that listening to unnormalized audio (do_normalize=False) sounds good.
        assert len(processed_audio.shape) == 2

        return processed_audio
