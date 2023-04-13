from src.config.config_train import config
from src.enums.enums import AudioTransforms
from src.features.audio_to_ast import AudioTransformAST
from src.features.audio_to_mfcc import MFCCFixedRepeat
from src.features.audio_to_spectrogram import (
    MelSpectrogramFixedRepeat,
    MelSpectrogramResizedRepeat,
)
from src.features.audio_to_wav2vec import AudioToWav2Vec2, AudioToWav2Vec2CNN
from src.features.audio_transform_base import AudioTransformBase
from src.features.augmentations import SpectrogramAugmentation, WaveformAugmentation
from src.utils.utils_exceptions import UnsupportedAudioTransforms


def get_audio_transform(
    audio_transform_enum: AudioTransforms,
    sampling_rate: int,
    spectrogram_augmentation: SpectrogramAugmentation | None,
    waveform_augmentation: WaveformAugmentation | None,
    image_dim: tuple[int, int],
) -> AudioTransformBase:
    base_kwargs = dict(
        sampling_rate=sampling_rate,
        waveform_augmentation=waveform_augmentation,
        spectrogram_augmentation=spectrogram_augmentation,
    )

    if audio_transform_enum is AudioTransforms.AST:
        return AudioTransformAST(
            pretrained_tag=config.pretrained_tag,
            **base_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.MEL_SPECTROGRAM_FIXED_REPEAT:
        return MelSpectrogramFixedRepeat(
            max_audio_seconds=config.max_audio_seconds,
            image_dim=image_dim,
            **base_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.MEL_SPECTROGRAM_RESIZE_REPEAT:
        return MelSpectrogramResizedRepeat(
            image_dim=image_dim,
            **base_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.WAV2VEC:
        return AudioToWav2Vec2(
            pretrained_tag=config.pretrained_tag,
            **base_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.MFCC_FIXED_REPEAT:
        return MFCCFixedRepeat(
            image_dim=image_dim,
            n_mfcc=config.n_mfcc,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            max_audio_seconds=config.max_audio_seconds,
            **base_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.WAV2VECCNN:
        return AudioToWav2Vec2CNN(
            **base_kwargs,
        )
    raise UnsupportedAudioTransforms(f"Unsupported transform {audio_transform_enum}")
