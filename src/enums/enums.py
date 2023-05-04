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
    WAV2VEC_CNN = "wav2vec_cnn"
    CONVNEXT_TINY = "convnext_tiny"
    CONVNEXT_SMALL = "convnext_small"
    CONVNEXT_LARGE = "convnext_large"
    CONVNEXT_BASE = "convnext_base"
    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    CONVLSTM = "conv_lstm"


class SupportedHeads(Enum):
    DEEP_HEAD = "deep_head"
    ATTENTION_HEAD = "attention_head"


class SupportedAugmentations(Enum):
    """List of supported spectrogram augmentations we use."""

    # Dataset sampling augmentations
    CONCAT_N_SAMPLES = "concat_n"
    SUM_N_SAMPLES = "sum_n_samples"

    # Waveform augmentations
    BACKGROUND_NOISE = "background_noise"
    TIME_STRETCH = "time_stretch"
    TIME_SHIFT = "time_shift"
    PITCH = "pitch"
    COLOR_NOISE = "color_noise"
    TIMEINV = "timeinv"
    TIME_MASK = "time_mask"
    NORM_AFTER_TIME_AUGS = "norm_after_time_augs"

    # Spectrogram augmentations
    FREQ_MASK = "freq_mask"
    RANDOM_ERASE = "random_erase"
    RANDOM_PIXELS = "radnom_pixels"

    # Legacy
    SEVEN_BAND_EQ = "seven_band_eq"
    CLIPPING = "clipping"
    SUM_TWO_SAMPLES = "sum_two_samples"


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
    FOCAL_LOSS = "focal_loss"
    FOCAL_LOSS_POS_WEIGHT = "focal_loss_pos_weight"
    INSTRUMENT_FAMILY_LOSS = "instrument_family_loss"


class AudioTransforms(Enum):
    """List of supported AudioTransforms we use."""

    AST = "ast"
    MEL_SPECTROGRAM = "mel_spectrogram"
    MULTI_SPECTROGRAM = "multi_spectrogram"
    WAV2VEC = "wav2vec"
    MFCC = "mfcc"

    # Legacy
    WAVELET = "wavelet"
    MELCHROWAV = "melchrowav"


class SupportedDatasetDirType(Enum):
    """List of SupportedDatasetDirType we use."""

    IRMAS_TRAIN = "irmastrain"
    IRMAS_TEST = "irmastest"
    OPENMIC = "openmic"
    CSV = "csv"
    INFERENCE = "inference"


class MetricMode(Enum):
    MIN = "min"
    MAX = "max"


class OptimizeMetric(Enum):
    VAL_HAMMING = "val/hamming_distance"
    VAL_F1 = "val/f1"


class ModelInputDataType(Enum):
    """Does the model accept waveform or spectrogram as an input?"""

    WAVEFORM = "waveform"
    IMAGE = "image"


# Get all enums from this file in one list.
current_module = sys.modules[__name__]
all_enums = [
    getattr(current_module, attr)
    for attr in dir(current_module)
    if issubclass(getattr(current_module, attr).__class__, EnumMeta) and attr != "Enum"
]
