import numpy as np
import torch
from transformers import ASTFeatureExtractor

from src.config.argparse_with_config import ArgParseWithConfig
from src.config.config_defaults import (
    DEFAULT_AST_MEAN,
    DEFAULT_AST_STD,
    TAG_AST_AUDIOSET,
)
from src.features.audio_transform_base import AudioTransformBase
from src.utils.utils_audio import (
    iron_audios,
    plot_spectrograms,
    repeat_self_to_length,
    spec_width_to_num_samples,
)
from src.utils.utils_dataset import get_example_val_sample


class AudioTransformAST(AudioTransformBase):
    def __init__(
        self,
        pretrained_tag,
        hop_length: int,
        n_mels: int,
        n_fft: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        if pretrained_tag is None:
            self.feature_extractor = ASTFeatureExtractor(
                sampling_rate=self.sampling_rate,
                num_mel_bins=self.n_mels,
                max_length=self.max_num_width_samples,
                do_normalize=True,
            )
        else:
            print(
                "Warning: inferring max_num_width_samples from AST pretrained config."
            )
            self.feature_extractor: ASTFeatureExtractor = (
                ASTFeatureExtractor.from_pretrained(pretrained_tag)
            )
        self.image_size = (
            self.feature_extractor.num_mel_bins,
            self.feature_extractor.max_length,
        )
        self.max_audio_length = spec_width_to_num_samples(
            self.image_size[-1], self.hop_length
        )

    @staticmethod
    def ast_feature_to_melspec(spectrogram: torch.Tensor):
        denormalized = (spectrogram * DEFAULT_AST_STD * 2) + DEFAULT_AST_MEAN
        return AudioTransformAST.denorm_ast_feature_to_melspec(denormalized)

    @staticmethod
    def denorm_ast_feature_to_melspec(spectrogram: torch.Tensor):
        spectrogram = spectrogram.exp()
        spectrogram = spectrogram.transpose(-2, -1)
        return spectrogram

    # @timeit
    def __call__(
        self, audio: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.waveform_augmentation is not None:
            audio = self.waveform_augmentation(torch.tensor(audio).view(1, 1, -1))
            audio = audio.view(-1)

        # Split waveform because feature extraction because transformer has a limit (trunc/padding). It creates fixed sized spectrogram. If audio is too long the spectrogram won't contain all of the audio.
        if len(audio) > self.max_audio_length:
            audio_tensors: tuple[torch.Tensor] = torch.tensor(audio).split(
                self.max_audio_length, dim=-1
            )
            audio = [a.numpy() for a in audio_tensors]
        else:
            audio = [audio]

        # Kaldi requires audio's length to be at least n_fft (400)
        # If there's only one chunk pad it with zero up to n_fft (400)
        # If there are multiple chunks discard the last (problematic) one
        min_waveform_length = self.n_fft
        num_chunks = len(audio)
        last_chunk = audio[-1]
        last_chunk_length = len(last_chunk)
        if last_chunk_length < min_waveform_length and num_chunks > 1:
            audio.pop()
        else:
            audio = iron_audios(audio, target_width=self.max_audio_length)

        spectrogram = self.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )["input_values"]
        if self.spectrogram_augmentation is not None:
            spectrogram = self.spectrogram_augmentation(
                spectrogram.permute(0, 2, 1)
            ).unsqueeze(0)
            spectrogram = repeat_self_to_length(spectrogram, self.image_size[-1])
            spectrogram.permute(0, 2, 1)  # [Batch, 1024, 128]

        assert len(spectrogram.shape) == 3, "Spectrogram chunks are 2D images"
        return spectrogram


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
        spectrogram_augmentation=None,
        waveform_augmentation=None,
    )
    spectrogram = transform(audio)
    spectrogram = spectrogram
    plot_spectrograms(
        spectrogram,
        sampling_rate=config.sampling_rate,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
        n_mels=config.n_mels,
    )
