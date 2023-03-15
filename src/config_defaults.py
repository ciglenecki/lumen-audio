"""To override any configs, set variables in "config_local.py" which is your own config file."""

from enum import Enum
from pathlib import Path

import pyrootutils

from src.utils_train import MetricMode, OptimizeMetric, OptimizerType

# ===============
# PATHS START
# ===============

PATH_WORK_DIR = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
PATH_DATA = Path(PATH_WORK_DIR, "data")
PATH_TRAIN = Path(PATH_DATA, "irmas", "train")
PATH_VAL = Path(PATH_DATA, "irmas", "val")
PATH_TEST = Path(PATH_DATA, "irmas", "test")
PATH_MODELS = Path(PATH_WORK_DIR, "models")

# ===============
# PATHS END
# ===============

DEFAULT_IRMAS_TRAIN_SIZE = 6705
DEFAULT_IRMAS_TEST_SIZE = 2874
DEFAULT_BATCH_SIZE = 3
DEFAULT_NUM_WORKERS = 5
DEFAULT_LOG_EVERY_N_STEPS = 100
DEFAULT_DATASET_FRACTION = 1.0
DEFAULT_LR = 1e-5
DEFAULT_PRETRAINED = True
DEFAULT_EPOCHS = 2000
DEFAULT_EARLY_STOPPING_NO_IMPROVEMENT_EPOCHS = 5
DEFAULT_CHECK_ON_TRAIN_EPOCH_END = False
DEFAULT_SAVE_ON_TRAIN_EPOCH_END = False
DEFAULT_SAMPLING_RATE = 22_050 # if using AST make sure you change to 16_000!
DEFAULT_AST_PRETRAINED_TAG = "MIT/ast-finetuned-audioset-10-10-0.4593"
DEFAULT_OPTIMIZER = OptimizerType.ADAMW
DEFAULT_OPTIMIZE_METRIC = OptimizeMetric.VAL_HAMMING
DEFAULT_METRIC_MODE = MetricMode.MIN
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_LR_PLATEAU_FACTOR = 0.5
DEFAULT_SANITY_CHECKS = False
DEFUALT_TQDM_REFRESH = 20
DEFAULT_AUDIO_EXTENSIONS = ["wav"]
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_MELS = 10
DEFAULT_DIM = (384, 384)
DEFAULT_FC = []
DEFAULT_PRETRAINED_WEIGHTS = "DEFAULT"

# ===============
# KEYS START
# ===============


class InstrumentKeys(Enum):
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
    InstrumentKeys.CELLO.value: 0,
    InstrumentKeys.CLARINET.value: 1,
    InstrumentKeys.FLUTE.value: 2,
    InstrumentKeys.ACOUSTIC_GUITAR.value: 3,
    InstrumentKeys.ELECTRIC_GUITAR.value: 4,
    InstrumentKeys.ORGAN.value: 5,
    InstrumentKeys.PIANO.value: 6,
    InstrumentKeys.SAXOPHONE.value: 7,
    InstrumentKeys.TRUMPET.value: 8,
    InstrumentKeys.VIOLIN.value: 9,
    InstrumentKeys.VOICE.value: 10,
}

INSTRUMENT_TO_FULLNAME = {
    InstrumentKeys.CELLO.value: "cello",
    InstrumentKeys.CLARINET.value: "clarinet",
    InstrumentKeys.FLUTE.value: "flute",
    InstrumentKeys.ACOUSTIC_GUITAR.value: "acoustic guitar",
    InstrumentKeys.ELECTRIC_GUITAR.value: "electric guitar",
    InstrumentKeys.ORGAN.value: "organ",
    InstrumentKeys.PIANO.value: "piano",
    InstrumentKeys.SAXOPHONE.value: "saxophone",
    InstrumentKeys.TRUMPET.value: "trumpet",
    InstrumentKeys.VIOLIN.value: "violin",
    InstrumentKeys.VOICE.value: "human voice",
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

from src.config_local import *
