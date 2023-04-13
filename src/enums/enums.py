from src.utils.utils_functions import EnumStr


class SupportedModels(EnumStr):
    AST = "ast"
    EFFICIENT_NET_V2_S = "efficient_net_v2_s"
    EFFICIENT_NET_V2_M = "efficient_net_v2_m"
    EFFICIENT_NET_V2_L = "efficient_net_v2_l"
    RESNEXT50_32X4D = "resnext50_32x4d"
    RESNEXT101_32X8D = "resnext101_32x8d"
    RESNEXT101_64X4D = "resnext101_64x4d"
    WAV2VEC = "wav2vec"
    WAV2VECCNN = "wav2vec_cnn"


class SupportedHeads(EnumStr):
    DEEP_HEAD = "deep_head"
    ATTENTION_HEAD = "attention_head"


class SupportedAugmentations(EnumStr):
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
    CONCAT_N_SAMPLES = "concat_n"
    SUM_TWO_SAMPLES = "sum_two_samples"


class SupportedScheduler(EnumStr):
    ONECYCLE = "onecycle"
    PLATEAU = "plateau"
    AUTO_LR = "auto_lr"
    COSINEANNEALING = "cosine_annealing"


class SupportedOptimizer(EnumStr):
    ADAM = "adam"
    ADAMW = "adamw"


class SupportedLossFunctions(EnumStr):
    CROSS_ENTROPY = "cross_entropy"
    CROSS_ENTROPY_POS_WEIGHT = "cross_entropy_pos_weight"


class AudioTransforms(EnumStr):
    """List of supported AudioTransforms we use."""

    AST = "ast"
    MEL_SPECTROGRAM_RESIZE_REPEAT = "mel_spectrogram_resize_repeat"
    MEL_SPECTROGRAM_FIXED_REPEAT = "mel_spectrogram_fixed_repeat"
    WAV2VEC = "wav2vec"
    MFCC_FIXED_REPEAT = "mfcc_fixed_repeat"
    WAV2VECCNN = "wav2veccnn"


class SupportedDatasets(EnumStr):
    """List of SupportedDatasets we use."""

    IRMAS = "irmas"
    OPENMIC = "openmic"


class MetricMode(EnumStr):
    MIN = "min"
    MAX = "max"


class OptimizeMetric(EnumStr):
    VAL_HAMMING = "val/hamming_distance"
    VAL_F1 = "val/f1"


class ModelInputDataType(EnumStr):
    WAVEFORM = "waveform"
    IMAGE = "image"
