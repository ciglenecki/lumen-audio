"""To override any configs, set variables in "config_local.py" which is your own config file."""

from enum import Enum
from pathlib import Path

import pyrootutils

from src.enums.enums import (
    MetricMode,
    OptimizeMetric,
    SupportedAugmentations,
    SupportedHeads,
    SupportedOptimizer,
    SupportedScheduler,
)

# ===============
# PATHS START
# ===============

PATH_WORK_DIR = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
PATH_DATA = Path(PATH_WORK_DIR, "data")
PATH_IRMAS = Path(PATH_DATA, "irmas")
PATH_IRMAS_TRAIN = Path(PATH_IRMAS, "train")
PATH_IRMAS_VAL = Path(PATH_IRMAS, "val")
PATH_IRMAS_TEST = Path(PATH_IRMAS, "test")
PATH_IRMAS_TRAIN_FEATURES = Path(PATH_IRMAS, "train_features")
PATH_IRMAS_SAMPLE = Path(PATH_DATA, "irmas_sample")

PATH_OPENMIC = Path(PATH_DATA, "openmic")
PATH_MODELS = Path(PATH_WORK_DIR, "models")
PATH_MODELS_QUICK = Path(PATH_WORK_DIR, "models_quick")

# ===============
# PATHS END
# ===============


# Digital signal processing
DEFAULT_SAMPLING_RATE = 16_000
DEFAULT_N_FFT = 400
DEFAULT_N_MELS = 128
DEFAULT_N_MFCC = 20
DEFAULT_IMAGE_DIM = (384, 384)
DEFAULT_NORMALIZE_AUDIO = True
DEFAULT_AUDIO_CHUNK_SIZE = 16_000 * 3
DEFAULT_SPECTROGRAM_CHUNK_SIZE = 130
DEFAULT_MAX_AUDIO_SECONDS = 3
DEFAULT_HOP_LENGTH = DEFAULT_N_FFT // 2
DEFAULT_AST_N_FFT = 400
DEFAULT_AST_HOP_LENGTH = 160
DEFAULT_AST_N_MELS = 128

_augs = list(SupportedAugmentations)
_augs.remove(SupportedAugmentations.RANDOM_ERASE)
_augs.remove(SupportedAugmentations.CONCAT_TWO)
DEFAULT_AUGMENTATIONS = _augs  # all excepted removed ones

DEFAULT_AUGMENTATION_KWARSG = dict(
    stretch_factors=[0.6, 1.4],
    time_inversion_p=0.5,
    freq_mask_param=30,
    time_mask_param=30,
    hide_random_pixels_p=0.25,
    std_noise=0.01,
)

# DATASET
DEFAULT_TRAIN_DIRS = [f"irmas:{str(PATH_IRMAS_TRAIN)}"]
DEFAULT_VAL_DIRS = [f"irmas:{str(PATH_IRMAS_TEST)}"]
DEFAULT_AUDIO_EXTENSIONS = ["wav"]
DEFAULT_ONLY_TRAIN_DATASET = False

# TRAIN
DEFAULT_EPOCHS = 40
DEFAULT_FINETUNE_HEAD = True
DEFAULT_FINETUNE_HEAD_EPOCHS = 5
DEFAULT_EARLY_STOPPING_METRIC_PATIENCE = 10
DEFAULT_BATCH_SIZE = 3
DEFAULT_NUM_WORKERS = 4
DEFAULT_LOG_EVERY_N_STEPS = 20
DEFAULT_DATASET_FRACTION = 1.0
DEFAULT_USE_WEIGHTED_TRAIN_SAMPLER = False
DEFAULT_USE_FLUFFY = False

# OPTIM
DEFAULT_METRIC_MODE = MetricMode.MAX
DEFAULT_OPTIMIZER = SupportedOptimizer.ADAM
DEFAULT_OPTIMIZE_METRIC = OptimizeMetric.VAL_F1
DEFAULT_LR = 1e-5
DEFAULT_LR_WARMUP = 1e-4
DEFAULT_LR_ONECYCLE_MAX = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_LR_SCHEDULER = SupportedScheduler.ONECYCLE
DEFAULT_FREEZE_TRAIN_BN = True
DEFAULT_USE_MULTIPLE_OPTIMIZERS = False

# MODEL
DEFAULT_PRETRAINED = True
DEFAULT_AST_PRETRAINED_TAG = "MIT/ast-finetuned-audioset-10-10-0.4593"
DEFAULT_WAV2VEC_PRETRAINED_TAG = "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres"
DEFAULT_TORCH_CNN_PRETRAINED_TAG = "IMAGENET1K_V2"
DEFAULT_PRETRAINED_TAG = "DEFAULT"
DEAFULT_HEAD = SupportedHeads.DEEP_HEAD

# LOGS
DEFUALT_TQDM_REFRESH = 30
DEFAULT_CHECK_ON_TRAIN_EPOCH_END = False
DEFAULT_SAVE_ON_TRAIN_EPOCH_END = False
DEFAULT_LOG_PER_INSTRUMENT_METRICS = True


# ===============
# VALUES THAT DON'T CHANGE START
# ===============

DEFAULT_IRMAS_TRAIN_SIZE = 6705
DEFAULT_IRMAS_TEST_SIZE = 2874
DEFAULT_RGB_CHANNELS = 3
DEFAULT_LR_PLATEAU_FACTOR = 0.5

IRMAS_TRAIN_CLASS_COUNT = {
    "voi": 778,
    "gel": 760,
    "pia": 721,
    "org": 682,
    "gac": 637,
    "sax": 626,
    "vio": 580,
    "tru": 577,
    "cla": 505,
    "flu": 451,
    "cel": 388,
}

# ===============
# VALUES THAT DON'T CHANGE END
# ===============

# ===============
# KEYS START
# ===============


class InstrumentEnums(Enum):
    CELLO = "cel"
    CLARINET = "cla"
    FLUTE = "flu"
    ACOUSTIC_GUITAR = "gac"
    ELECTRIC_GUITAR = "gel"
    ORGAN = "org"
    PIANO = "pia"
    SAXOPHONE = "sax"
    TRUMPET = "tru"
    VIOLIN = "vio"
    VOICE = "voi"


INSTRUMENT_TO_IDX = {
    InstrumentEnums.CELLO.value: 0,
    InstrumentEnums.CLARINET.value: 1,
    InstrumentEnums.FLUTE.value: 2,
    InstrumentEnums.ACOUSTIC_GUITAR.value: 3,
    InstrumentEnums.ELECTRIC_GUITAR.value: 4,
    InstrumentEnums.ORGAN.value: 5,
    InstrumentEnums.PIANO.value: 6,
    InstrumentEnums.SAXOPHONE.value: 7,
    InstrumentEnums.TRUMPET.value: 8,
    InstrumentEnums.VIOLIN.value: 9,
    InstrumentEnums.VOICE.value: 10,
}

INSTRUMENT_TO_FULLNAME = {
    InstrumentEnums.CELLO.value: "cello",
    InstrumentEnums.CLARINET.value: "clarinet",
    InstrumentEnums.FLUTE.value: "flute",
    InstrumentEnums.ACOUSTIC_GUITAR.value: "acoustic_guitar",
    InstrumentEnums.ELECTRIC_GUITAR.value: "electric_guitar",
    InstrumentEnums.ORGAN.value: "organ",
    InstrumentEnums.PIANO.value: "piano",
    InstrumentEnums.SAXOPHONE.value: "saxophone",
    InstrumentEnums.TRUMPET.value: "trumpet",
    InstrumentEnums.VIOLIN.value: "violin",
    InstrumentEnums.VOICE.value: "human_voice",
}

IDX_TO_INSTRUMENT = {v: k for k, v in INSTRUMENT_TO_IDX.items()}


class DrumKeys(Enum):
    UNKNOWN = "unknown-dru"
    IS_PRESENT = "dru"
    NOT_PRESENT = "nod"


DRUMS_TO_IDX = {  # no drums is 0 at DrumKeys.IS_PRESENT
    DrumKeys.UNKNOWN.value: 0,
    DrumKeys.IS_PRESENT.value: 1,
}
IDX_TO_DRUMS = {v: k for k, v in DRUMS_TO_IDX.items()}


class GenreKeys(Enum):
    COUNTRY_FOLK = "cou_fol"
    CLASSICAL = "cla"
    POP_ROCK = "pop_roc"
    LATINO_SOUL = "lat_sou"
    JAZZ_BLUES = "jaz_blu"
    UNKNOWN = "unknown"


GENRE_TO_IDX = {
    GenreKeys.COUNTRY_FOLK.value: 0,
    GenreKeys.CLASSICAL.value: 1,
    GenreKeys.POP_ROCK.value: 2,
    GenreKeys.LATINO_SOUL.value: 3,
    GenreKeys.JAZZ_BLUES.value: 4,
}


IDX_TO_GENRE = {v: k for k, v in GENRE_TO_IDX.items()}

# ===============
# KEYS END
# ===============

DEFAULT_NUM_LABELS = len(INSTRUMENT_TO_IDX)
