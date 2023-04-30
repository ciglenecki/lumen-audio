import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

from src.features.audio_transform_base import AudioTransformBase


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

        last_chunk = audio[-1]
        last_chunk_length = len(last_chunk)
        diff = self.max_num_width_samples - last_chunk_length

        # zero padding:
        # if last_chunk_length < self.max_num_width_samples:
        #     pad_width = (0, self.max_num_width_samples - last_chunk_length)
        #     audio[-1] = np.pad(
        #         last_chunk, pad_width, mode="constant", constant_values=0
        #     )

        first_chunk: torch.Tensor = audio[0]  # if first chunk == first chunk
        first_chunk_width = first_chunk.shape[-1]  # 16_000
        num_first_chunk_repeats = max(1, int(diff / first_chunk_width))  # 8
        repeated_first_chunk = np.concatenate(
            [first_chunk] * num_first_chunk_repeats, axis=-1
        )

        # Remove remove excess width caused by repeating
        repeated_first_chunk = repeated_first_chunk[..., :diff]
        audio[-1] = np.concatenate((audio[-1], repeated_first_chunk), axis=-1)

        processor_out = self.processor(
            audio, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True
        )

        processed_audio = processor_out.input_values
        assert len(processed_audio.shape) == 2
        return processed_audio
