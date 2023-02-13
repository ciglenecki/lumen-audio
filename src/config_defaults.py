from pathlib import Path

import pyrootutils

from utils_train import MetricMode, OptimizeMetric

"""To override any configs, set variables in "config_local.py" which is your own config file."""

# ===============
# PATHS START
# ===============

PATH_WORK_DIR = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
PATH_DATA = Path(PATH_WORK_DIR, "data")
PATH_TRAIN = Path(PATH_DATA, "raw", "train")
PATH_TEST = Path(PATH_DATA, "raw", "test")
PATH_REPORT = Path(PATH_WORK_DIR, "report")
# ===============
# PATHS END
# ===============

DEFAULT_IRMAS_TRAIN_SIZE = 6705
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 8
DEFAULT_LOG_EVERY_N_STEPS = 100
DEFAULT_DATASET_FRACTION = 1.0
DEFAULT_LR = 1e-5
DEFAULT_PRETRAINED = True
DEFAULT_EPOCHS = 1000
DEFAULT_EARLY_STOPPING_NO_IMPROVEMENT_EPOCHS = 5
DEFAULT_CHECK_ON_TRAIN_EPOCH_END = False
DEFAULT_SAVE_ON_TRAIN_EPOCH_END = False
DEFAULT_SAMPLING_RATE = 44_100
DEFAULT_AST_PRETRAINED_TAG = "MIT/ast-finetuned-audioset-10-10-0.4593"
DEFAULT_OPTIMIZE_METRIC = OptimizeMetric.VAL_HAMMING
DEFAULT_METRIC_MODE = MetricMode.max
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_LR_PLATEAU_FACTOR = 0.5
INSTRUMENT_TO_IDX = {
    "cel": 0,
    "cla": 1,
    "flu": 2,
    "gac": 3,
    "gel": 4,
    "org": 5,
    "pia": 6,
    "sax": 7,
    "tru": 8,
    "vio": 9,
    "voi": 10,
}

INSTRUMENT_TO_FULLNAME = {
    "cel": "cello",
    "cla": "clarinet",
    "flu": "flute",
    "gac": "acoustic guitar",
    "gel": "electric guitar",
    "org": "organ",
    "pia": "piano",
    "sax": "saxophone",
    "tru": "trumpet",
    "vio": "violin",
    "voi": "human voice",
}

IDX_TO_INSTRUMENT = {v: k for k, v in INSTRUMENT_TO_IDX.items()}

ADDITIONAL_FEATURES = {
    "dru": 0,
    "nod": 1,
    "cou-fol": 2,
    "cla": 3,
    "pop-roc": 4,
    "lat-sou": 5,
}

DEFAULT_NUM_CLASSES = len(INSTRUMENT_TO_IDX)

from config_local import *
