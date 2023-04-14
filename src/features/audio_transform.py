from src.config.config_defaults import ConfigDefault
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
    config: ConfigDefault,
    audio_transform_enum: AudioTransforms,
    spectrogram_augmentation: SpectrogramAugmentation | None = None,
    waveform_augmentation: WaveformAugmentation | None = None,
) -> AudioTransformBase:
    base_kwargs = dict(
        sampling_rate=config.sampling_rate,
        waveform_augmentation=waveform_augmentation,
        spectrogram_augmentation=spectrogram_augmentation,
    )
    image_kwargs = dict(
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        image_dim=config.image_dim,
    )

    if audio_transform_enum is AudioTransforms.AST:
        return AudioTransformAST(
            pretrained_tag=config.pretrained_tag,
            **base_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.MEL_SPECTROGRAM_FIXED_REPEAT:
        return MelSpectrogramFixedRepeat(
            max_audio_seconds=config.max_audio_seconds,
            **image_kwargs,
            **base_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.MEL_SPECTROGRAM_RESIZE_REPEAT:
        return MelSpectrogramResizedRepeat(
            **image_kwargs,
            **base_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.WAV2VEC:
        return AudioToWav2Vec2(
            pretrained_tag=config.pretrained_tag,
            **image_kwargs,
            **base_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.MFCC_FIXED_REPEAT:
        return MFCCFixedRepeat(
            n_mfcc=config.n_mfcc,
            max_audio_seconds=config.max_audio_seconds,
            **image_kwargs,
            **base_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.WAV2VEC_CNN:
        return AudioToWav2Vec2CNN(
            **base_kwargs,
        )
    raise UnsupportedAudioTransforms(f"Unsupported transform {audio_transform_enum}")
