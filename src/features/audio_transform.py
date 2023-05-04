from src.config.config_defaults import ConfigDefault
from src.enums.enums import AudioTransforms
from src.features.audio_to_ast import AudioTransformAST
from src.features.audio_to_mfcc import MFCC
from src.features.audio_to_multispec import MultiSpectrogram
from src.features.audio_to_spectrogram import MelSpectrogram
from src.features.audio_to_wav2vec import AudioToWav2Vec2
from src.features.audio_transform_base import AudioTransformBase
from src.features.augmentations import SpectrogramAugmentation, WaveformAugmentation
from src.utils.utils_exceptions import UnsupportedAudioTransforms


def get_audio_transform(
    config: ConfigDefault,
    spectrogram_augmentation: SpectrogramAugmentation | None = None,
    waveform_augmentation: WaveformAugmentation | None = None,
) -> AudioTransformBase:
    audio_transform_enum = config.audio_transform
    base_kwargs = dict(
        sampling_rate=config.sampling_rate,
        max_num_width_samples=config.max_num_width_samples,
        waveform_augmentation=waveform_augmentation,
        spectrogram_augmentation=spectrogram_augmentation,
    )
    image_kwargs = dict(
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        image_size=config.image_size,
        use_rgb=config.use_rgb,
    )

    if audio_transform_enum is AudioTransforms.AST:
        return AudioTransformAST(
            pretrained_tag=config.pretrained_tag,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            **base_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.MEL_SPECTROGRAM:
        return MelSpectrogram(
            **image_kwargs,
            **base_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.MULTI_SPECTROGRAM:
        return MultiSpectrogram(
            **image_kwargs,
            **base_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.WAV2VEC:
        return AudioToWav2Vec2(
            pretrained_tag=config.pretrained_tag,
            **base_kwargs,
        )
    elif audio_transform_enum is AudioTransforms.MFCC:
        return MFCC(
            n_mfcc=config.n_mfcc,
            **image_kwargs,
            **base_kwargs,
        )
    raise UnsupportedAudioTransforms(f"Unsupported transform {audio_transform_enum}")
