import src.config.config_defaults as config_defaults
from src.features.audio_to_ast import AudioTransformAST
from src.features.audio_to_mfcc import MFCCFixedRepeat
from src.features.audio_to_spectrogram import (
    MelSpectrogramFixedRepeat,
    MelSpectrogramResizedRepeat,
)
from src.features.audio_to_wav2vec import AudioToWav2Vec2, AudioToWav2Vec2CNN
from src.features.audio_transform_base import (
    AudioTransformBase,
    AudioTransforms,
    UnsupportedAudioTransforms,
)
from src.features.augmentations import (
    SpectrogramAugmentation,
    SupportedAugmentations,
    WaveformAugmentation,
)


def get_audio_transform(
    audio_transform_enum: AudioTransforms,
    sampling_rate: int,
    spectrogram_augmentation: SpectrogramAugmentation | None,
    waveform_augmentation: WaveformAugmentation | None,
    dim: tuple[int, int],
) -> AudioTransformBase:
    base_kwargs = dict(
        sampling_rate=sampling_rate,
        waveform_augmentation=waveform_augmentation,
        spectrogram_augmentation=spectrogram_augmentation,
    )

    if audio_transform_enum is AudioTransforms.AST:
        return AudioTransformAST(
            pretrained_tag=config_defaults.DEFAULT_AST_PRETRAINED_TAG,
            **base_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.MEL_SPECTROGRAM_FIXED_REPEAT:
        return MelSpectrogramFixedRepeat(
            dim=dim,
            repeat=config_defaults.DEFAULT_REPEAT,
            max_len=config_defaults.DEFAULT_MAX_LEN,
            **base_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.MEL_SPECTROGRAM_RESIZE_REPEAT:
        return MelSpectrogramResizedRepeat(
            dim=dim,
            repeat=config_defaults.DEFAULT_REPEAT,
            **base_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.WAV2VEC:
        return AudioToWav2Vec2(
            pretrained_tag=config_defaults.DEFAULT_WAV2VEC_PRETRAINED_TAG,
            **base_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.MFCC_FIXED_REPEAT:
        return MFCCFixedRepeat(
            dim=dim,
            n_mfcc=config_defaults.DEFAULT_N_MFCC,
            dct_type=config_defaults.DEFAULT_DCT_TYPE,
            repeat=config_defaults.DEFAULT_REPEAT,
            n_fft=config_defaults.DEFAULT_N_FFT,
            hop_length=config_defaults.DEFAULT_HOP_LENGTH,
            n_mels=config_defaults.DEFAULT_N_MELS,
            max_len=config_defaults.DEFAULT_MAX_LEN,
            **base_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.WAV2VECCNN:
        return AudioToWav2Vec2CNN(
            dim=dim,
            n_mfcc=config_defaults.DEFAULT_N_MFCC,
            dct_type=config_defaults.DEFAULT_DCT_TYPE,
            max_len=config_defaults.DEFAULT_MAX_LEN,
            **base_kwargs,
        )
    raise UnsupportedAudioTransforms(f"Unsupported transform {audio_transform_enum}")
