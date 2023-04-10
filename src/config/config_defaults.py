"""To override any configs, set variables in "config_local.py" which is your own config file."""

from enum import Enum
from pathlib import Path

import pyrootutils

from src.features.augmentations import SupportedAugmentations
from src.model.optimizers import OptimizerType, SchedulerType
from src.utils.utils_train import MetricMode, OptimizeMetric

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

PATH_OPENMIC = Path(PATH_DATA, "openmic")
PATH_MODELS = Path(PATH_WORK_DIR, "models")

# ===============
# PATHS END
# ===============

DEFAULT_IRMAS_TRAIN_SIZE = 6705
DEFAULT_IRMAS_TEST_SIZE = 2874
DEFAULT_BATCH_SIZE = 2
DEFAULT_NUM_WORKERS = 4
DEFAULT_LOG_EVERY_N_STEPS = 20
DEFAULT_DATASET_FRACTION = 1.0
DEFAULT_LR = 1e-5

DEFAULT_LR_WARMUP = 1e-4
DEFAULT_LR_ONECYCLE_MAX = 3e-4
DEFAULT_PRETRAINED = True
DEFAULT_EPOCHS = 60

DEFAULT_UNFREEZE_AT_EPOOCH = 5
DEFAULT_PLATEAU_EPOCH_PATIENCE = 6
DEFAULT_CHECK_ON_TRAIN_EPOCH_END = False
DEFAULT_SAVE_ON_TRAIN_EPOCH_END = False
DEFAULT_SAMPLING_RATE = 16_000
DEFAULT_AST_PRETRAINED_TAG = "MIT/ast-finetuned-audioset-10-10-0.4593"
DEFAULT_WAV2VEC_PRETRAINED_TAG = "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres"
DEFAULT_LR_SCHEDULER = SchedulerType.ONECYCLE
DEFAULT_OPTIMIZER = OptimizerType.ADAM
DEFAULT_OPTIMIZE_METRIC = OptimizeMetric.VAL_F1
DEFAULT_METRIC_MODE = MetricMode.MAX
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_LR_PLATEAU_FACTOR = 0.5
DEFAULT_SANITY_CHECKS = False
DEFUALT_TQDM_REFRESH = 20
DEFAULT_AUDIO_EXTENSIONS = ["wav"]
DEFAULT_N_FFT = 400
DEFAULT_N_MELS = 128
DEFAULT_DIM = (384, 384)
DEFAULT_NORMALIZE_AUDIO = True
DEFAULT_HOP_LENGTH = DEFAULT_N_FFT // 2
DEFAULT_FC = []
DEFAULT_PRETRAINED_WEIGHTS = "DEFAULT"
DEFAULT_LOG_PER_INSTRUMENT_METRICS = False
DEFAULT_N_MFCC = 20
DEFAULT_DCT_TYPE = 2
DEFAULT_MAX_LEN = 3
DEFAULT_REPEAT = 3
DEFAULT_MAX_SEQ_LENGTH = 3
DEFAULT_FREEZE_TRAIN_BN = True
DEFAULT_USE_WEIGHTED_TRAIN_SAMPLER = False
DEFAULT_ONLY_TRAIN_DATASET = False
DEFAULT_AUDIO_CHUNK_SIZE = 16_000 * 3
DEFAULT_SPECTROGRAM_CHUNK_SIZE = 130

DEFAULT_AUGMENTATIONS = [
    SupportedAugmentations.TIME_STRETCH,
    SupportedAugmentations.PITCH,
    SupportedAugmentations.BANDPASS_FILTER,
    SupportedAugmentations.COLOR_NOISE,
    SupportedAugmentations.TIMEINV,
    SupportedAugmentations.FREQ_MASK,
    SupportedAugmentations.TIME_MASK,
    # SupportedAugmentations.RANDOM_ERASE,
    SupportedAugmentations.RANDOM_PIXELS,
]


DEFAULT_TRAIN_DIRS = [f"irmas:{str(PATH_IRMAS_TRAIN)}"]
DEFAULT_VAL_DIRS = [f"irmas:{str(PATH_IRMAS_TEST)}"]

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
