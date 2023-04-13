import sys
from enum import Enum, EnumMeta


class SupportedModels(Enum):
    AST = "ast"
    EFFICIENT_NET_V2_S = "efficient_net_v2_s"
    EFFICIENT_NET_V2_M = "efficient_net_v2_m"
    EFFICIENT_NET_V2_L = "efficient_net_v2_l"
    RESNEXT50_32X4D = "resnext50_32x4d"
    RESNEXT101_32X8D = "resnext101_32x8d"
    RESNEXT101_64X4D = "resnext101_64x4d"
    WAV2VEC = "wav2vec"
    WAV2VECCNN = "wav2vec_cnn"


class SupportedHeads(Enum):
    DEEP_HEAD = "deep_head"
    ATTENTION_HEAD = "attention_head"


class SupportedAugmentations(Enum):
    """List of supported spectrogram augmentations we use."""

    TIME_STRETCH = "time_stretch"
    FREQ_MASK = "freq_mask"
    TIME_MASK = "time_mask"
    RANDOM_ERASE = "random_erase"
    RANDOM_PIXELS = "radnom_pixels"
    COLOR_NOISE = "color_noise"
    BANDPASS_FILTER = "bandpass"
    PITCH = "pitch"
    TIMEINV = "timeinv"
    CONCAT_TWO = "concat_two"


class SupportedScheduler(Enum):
    ONECYCLE = "onecycle"
    PLATEAU = "plateau"
    AUTO_LR = "auto_lr"
    COSINEANNEALING = "cosine_annealing"


class SupportedOptimizer(Enum):
    ADAM = "adam"
    ADAMW = "adamw"


class SupportedLossFunctions(Enum):
    CROSS_ENTROPY = "cross_entropy"
    CROSS_ENTROPY_POS_WEIGHT = "cross_entropy_pos_weight"


class AudioTransforms(Enum):
    """List of supported AudioTransforms we use."""

    AST = "ast"
    MEL_SPECTROGRAM_RESIZE_REPEAT = "mel_spectrogram_resize_repeat"
    MEL_SPECTROGRAM_FIXED_REPEAT = "mel_spectrogram_fixed_repeat"
    WAV2VEC = "wav2vec"
    MFCC_FIXED_REPEAT = "mfcc_fixed_repeat"
    WAV2VECCNN = "wav2veccnn"


class SupportedDatasets(Enum):
    """List of SupportedDatasets we use."""

    IRMAS = "irmas"
    OPENMIC = "openmic"


class MetricMode(Enum):
    MIN = "min"
    MAX = "max"


class OptimizeMetric(Enum):
    VAL_HAMMING = "val/hamming_distance"
    VAL_F1 = "val/f1"


class ModelInputDataType(Enum):
    WAVEFORM = "waveform"
    IMAGE = "image"


# Get all enums from this file in one list.
current_module = sys.modules[__name__]
all_enums = [
    getattr(current_module, attr)
    for attr in dir(current_module)
    if issubclass(getattr(current_module, attr).__class__, EnumMeta) and attr != "Enum"
]
