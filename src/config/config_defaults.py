"""Default global config.

Important: 0 dependencies except to enums!
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pyrootutils
from simple_parsing.helpers import Serializable

from src.enums.enums import (
    AudioTransforms,
    MetricMode,
    OptimizeMetric,
    SupportedAugmentations,
    SupportedDatasets,
    SupportedHeads,
    SupportedLossFunctions,
    SupportedModels,
    SupportedOptimizer,
    SupportedScheduler,
)


class InvalidArgument(Exception):
    """Argument is invalid."""


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

DEFAULT_NUM_LABELS = len(INSTRUMENT_TO_IDX)
DEFAULT_IRMAS_TRAIN_SIZE = 6705
DEFAULT_IRMAS_TEST_SIZE = 2874
DEFAULT_RGB_CHANNELS = 3
DEFAULT_LR_PLATEAU_FACTOR = 0.5

DEFAULT_AST_PRETRAINED_TAG = "MIT/ast-finetuned-audioset-10-10-0.4593"
DEFAULT_WAV2VEC_PRETRAINED_TAG = "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres"
DEFAULT_TORCH_CNN_PRETRAINED_TAG = "IMAGENET1K_V2"
DEFAULT_PRETRAINED_TAG = "DEFAULT"

DEAFULT_HEAD = SupportedHeads.DEEP_HEAD
DEFAULT_AUDIO_EXTENSIONS = ["wav"]

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


_default_augmentations_set = set(SupportedAugmentations)
_default_augmentations_set.discard(SupportedAugmentations.RANDOM_ERASE)
_default_augmentations_set.discard(SupportedAugmentations.CONCAT_N_SAMPLES)
_default_augmentations_set.discard(SupportedAugmentations.SUM_TWO_SAMPLES)
_default_augmentations_list = list(_default_augmentations_set)


def create(arg, **kwargs):
    """We need this useless helper function since you can't type augmentations: SupportedAugmentations = {} in ConfigDefault."""
    return field(default_factory=lambda: arg, **kwargs)


@dataclass
class ConfigDefault(Serializable):
    path_workdir: Path = create(
        Path(pyrootutils.find_root(search_from=__file__, indicator=".project-root"))
    )
    """Path to the root of the project."""

    path_data: Path | None = create(None)
    """Path to the data directory."""

    path_data: Path | None = create(None)
    path_irmas: Path | None = create(None)
    path_irmas_train: Path | None = create(None)
    path_irmas_test: Path | None = create(None)
    path_irmas_train_features: Path | None = create(None)
    path_irmas_sample: Path | None = create(None)
    path_openmic: Path | None = create(None)
    path_models: Path | None = create(None)
    path_models_quick: Path | None = create(None)

    train_dirs: list[str] | None = create(None)
    """Dataset root directories that will be used for training in the following format: --train-dirs irmas:/path/to openmic:/path/to"""

    val_dirs: list[str] | None = create(None)
    """Dataset root directories that will be used for validation in the following format: --val-dirs irmas:/path/to openmic:/path/to"""

    train_only_dataset: bool = create(False)
    """Use only the train portion of the dataset and split it 0.8 0.2"""

    dataset_fraction: float = create(1.0)
    """Reduce each dataset split (train, val, test) by a fraction."""

    num_labels: int = create(DEFAULT_NUM_LABELS)
    """Total number of possible lables"""

    train_override_csvs: Path | None = create(None)
    """CSV files with columns 'filename, sax, gac, org, ..., cla' where filename is path and each instrument is either 0 or 1"""

    # ======================== DPS ===========================

    sampling_rate: int = create(16_000)
    """Audio sampling rate"""

    n_fft: int = create(400)
    """Length of the signal you want to calculate the Fourier transform of"""

    hop_length: int = create(200)
    """Hop length which will be used during STFT cacualtion"""

    n_mels: int = create(128)
    """Number of mel bins you want to caculate"""

    n_mfcc: int = create(20)
    """Number of Mel-frequency cepstrum (MFCC) coefficients"""

    image_dim: tuple[int, int] = create((384, 384))
    """The dimension to resize the image to."""

    normalize_audio: bool = create(True)
    """Do normalize audio"""

    max_audio_seconds: float = create(3)
    """Maximum number of seconds of audio which will be processed at one time."""

    augmentations: list[SupportedAugmentations] = create(_default_augmentations_list)
    """Transformation which will be performed on audio and labels"""

    aug_kwargs: dict | str = create(
        dict(
            stretch_factors=[0.6, 1.4],
            time_inversion_p=0.5,
            freq_mask_param=30,
            time_mask_param=30,
            hide_random_pixels_p=0.25,
            std_noise=0.01,
            concat_n_samples=3,
        )
    )
    """Arguments are split by space, mutiple values are sep'ed by comma (,). E.g. stretch_factors=0.8,1.2 freq_mask_param=30 time_mask_param=30 hide_random_pixels_p=0.5"""

    # ======================== TRAIN ===========================

    audio_transform: AudioTransforms | None = create(None)
    """Transformation which will be performed on audio and labels"""

    batch_size: int = create(4)

    epochs: int = create(40)
    """Number epochs. Works only if learning rate scheduler has fixed number of steps (onecycle, cosine...). It won't have an effect on 'reduce on palteau' lr scheduler."""

    finetune_head_epochs: int = create(5)
    """Epoch at which the backbone will be unfrozen."""

    metric: OptimizeMetric = create(OptimizeMetric.VAL_F1)
    """Metric which the model will optimize for."""

    metric_mode: MetricMode = create(MetricMode.MAX)
    """Maximize or minimize the --metric."""

    early_stopping_metric_patience: int = create(10)
    """Number of checks with no improvement after which training will be stopped. Under the default configuration, one check happens after every training epoch"""

    num_workers: int = create(4)
    """Number of workers"""

    use_weighted_train_sampler: bool = create(False)
    """Use weighted train sampler instead of a random one."""

    weight_decay: float | None = create(None)
    """Maximum lr OneCycle scheduler reaches"""

    freeze_train_bn: bool = create(False)
    """If true, the batch norm will be trained even if module is frozen."""

    quick: bool = create(False)
    """For testing bugs. Simulates --limit_train_batches 2 --limit_val_batches 2 --limit_test_batches 2"""

    save_on_train_epoch_end: bool = create(False)
    """Whether to run checkpointing at the end of the training epoch."""

    skip_validation: bool = create(False)
    """Skips validation part during training."""

    drop_last: bool = create(True)
    """Drop last sample if the size of the sample is smaller than batch size"""

    check_on_train_epoch_end: bool = create(False)
    """Whether to run early stopping at the end of the training epoch."""

    # ======================== MODEL ===========================

    model: SupportedModels | None = create(None)
    """Models used for training."""

    finetune_head: bool = create(True)
    """Performs head only finetuning for --finetune-head-epochs epochs with starting lr of --lr-warmup which eventually becomes --lr."""

    backbone_after: str | None = create(None)
    """Name of the submodule after which the all submodules are considered as backbone, e.g. layer.11.dense"""

    head_after: str | None = create(None)
    """Name of the submodule after which the all submodules are considered as head, e.g. classifier.dense"""

    pretrained: bool = create(True)
    """Use a pretrained model loaded from the web."""

    pretrained_tag: str = create("DEFAULT")
    """The string that denotes the pretrained weights used."""

    head: SupportedHeads = create(SupportedHeads.DEEP_HEAD)
    """Type of classification head which will be used for classification. This is almost always the last layer."""

    ckpt: str | None = create(None)
    """.ckpt file, automatically restores model, epoch, step, LR schedulers, etc..."""

    use_fluffy: bool = create(False)
    """Use multiple optimizers for Fluffy."""

    # ======================== OPTIM ===========================

    optimizer: str = create(SupportedOptimizer.ADAMW)
    """Optimizer"""

    scheduler: SupportedScheduler = create(SupportedScheduler.ONECYCLE)

    loss_function: SupportedLossFunctions = create(SupportedLossFunctions.CROSS_ENTROPY)
    """Loss function"""

    loss_function_kwargs: dict | dict = create({})
    """Loss function kwargs"""

    lr: float = create(1e-4)
    """Learning rate"""

    lr_onecycle_max: float = create(5e-4)
    """Maximum lr OneCycle scheduler reaches"""

    lr_warmup: float = create(1e-4)
    """warmup learning rate"""

    use_multiple_optimizers: bool = create(False)
    """Use multiple optimizers for Fluffy. Each head will have it's own optimizer."""

    # ======================== LOGS ===========================
    log_per_instrument_metrics: bool = create(True)
    """Along with aggregated metrics, also log per instrument metrics."""

    bar_update: int = create(30)
    """Number of TQDM updates in one epoch."""

    log_every_n_steps: int = create(30)
    """How often (steps) to log metrics."""

    def __post_init__(self):
        """This function dynamically changes some of the arguments based on other arguments."""

        self.path_data: Path = Path(self.path_workdir, "data").relative_to(
            self.path_workdir
        )
        self.path_irmas: Path = Path(self.path_data, "irmas")
        self.path_irmas_train: Path = Path(self.path_irmas, "train")
        self.path_irmas_test: Path = Path(self.path_irmas, "test")
        self.path_irmas_train_features: Path = Path(self.path_irmas, "train_features")
        self.path_irmas_sample: Path = Path(self.path_data, "irmas_sample")
        self.path_openmic: Path = Path(self.path_data, "openmic")
        self.path_models: Path = Path(self.path_workdir, "models").relative_to(
            self.path_workdir
        )
        self.path_models_quick: Path = Path(
            self.path_workdir, "models_quick"
        ).relative_to(self.path_workdir)
        self.output_dir: Path = self.path_models
        """Output directory of the model and report file."""

        # We can't put where other default values live because we can't reference `self.path_irmas_test` until the user sets irmas directory.
        if self.train_dirs is None:
            self.train_dirs = [f"irmas:{str(self.path_irmas_train)}"]
        if self.val_dirs is None:
            self.val_dirs = [f"irmas:{str(self.path_irmas_test)}"]

        # Parse strings to dataset type and path
        self.train_dirs = [self.dir_to_enum_and_path(d) for d in self.train_dirs]
        self.val_dirs = [self.dir_to_enum_and_path(d) for d in self.val_dirs]

        # Dynamically set pretrained tag
        default_pretrain_tag = get_default_value_for_field("pretrained_tag", self)
        if self.pretrained and self.pretrained_tag == default_pretrain_tag:
            if self.model == SupportedModels.AST:
                self.pretrained_tag = DEFAULT_AST_PRETRAINED_TAG
            elif self.model in [SupportedModels.WAV2VEC_CNN, SupportedModels.WAV2VEC]:
                self.pretrained_tag = DEFAULT_WAV2VEC_PRETRAINED_TAG
            elif self.model in [
                SupportedModels.EFFICIENT_NET_V2_S,
                SupportedModels.EFFICIENT_NET_V2_M,
                SupportedModels.EFFICIENT_NET_V2_L,
                SupportedModels.RESNEXT50_32X4D,
                SupportedModels.RESNEXT101_32X8D,
                SupportedModels.RESNEXT101_64X4D,
            ]:
                self.pretrained_tag = DEFAULT_TORCH_CNN_PRETRAINED_TAG

        # Dynamically AST DSP attributes
        if (
            self.model == SupportedModels.AST
            and self.pretrained
            and self.pretrained_tag == DEFAULT_AST_PRETRAINED_TAG
        ):
            self.n_fft = 400
            self.hop_length = 160
            self.n_mels = 128
            if self.augmentations == get_default_value_for_field("augmentations", self):
                _augmentations_set = set(self.augmentations)
                _augmentations_set.discard(SupportedAugmentations.TIME_STRETCH)
                _augmentations_set.add(SupportedAugmentations.CONCAT_N_SAMPLES)
                _augmentations_set.add(SupportedAugmentations.SUM_TWO_SAMPLES)
                self.augmentations = list(_augmentations_set)

        if self.weight_decay is None and self.optimizer == SupportedOptimizer.ADAM:
            self.weight_decay = 0
        if self.weight_decay is None and self.optimizer == SupportedOptimizer.ADAMW:
            self.weight_decay = 1e-2

    def _validate_train_args(self):
        """This function validates arguments before training."""

        if self.model is None:
            raise InvalidArgument(
                f"--model is required for training {list(SupportedModels)}"
            )
        if self.audio_transform is None:
            raise InvalidArgument(
                f"--audio-transform is required for training {list(AudioTransforms)}"
            )
        if self.metric and not self.metric_mode:
            raise InvalidArgument("Can't pass --metric without passing --metric-mode")

        # aug_kwargs can be either a dictionary or a string which will be parsed as kwargs dict
        if isinstance(self.aug_kwargs, str):
            override_kwargs = self.parse_kwargs(self.aug_kwargs)
            self.aug_kwargs = get_default_value_for_field("aug_kwargs", self)
            self.aug_kwargs.update(override_kwargs)

        # loss_function_kwargs can be either a dictionary or a string which will be parsed as kwargs dict
        if isinstance(self.loss_function_kwargs, str):
            override_kwargs = self.parse_kwargs(self.loss_function_kwargs)
            self.loss_function_kwargs = get_default_value_for_field(
                "loss_function_kwargs", self
            )
            self.loss_function_kwargs.update(override_kwargs)

        if (
            self.scheduler == SupportedScheduler.ONECYCLE
            and self.lr_onecycle_max is None
        ):
            raise InvalidArgument(
                f"You have to pass the --lr-onecycle-max if you use the {self.scheduler}",
            )

        if self.model != SupportedModels.WAV2VEC_CNN and self.use_multiple_optimizers:
            raise InvalidArgument(
                "You can't use mutliple optimizers if you are not using Fluffy!",
            )

    def dir_to_enum_and_path(
        self,
        string: str,
    ) -> tuple[SupportedDatasets, Path]:
        """
        Example:
            string: irmas:/path/to/irmas
            return (SupportedDatasets.IRMAS, Path("/path/to/irmas"))
        """

        pair = string.split(":")
        if len(pair) != 2:
            raise InvalidArgument(
                f"Pair {pair} needs to have two elements. First arg is {list(SupportedDatasets)} and the second is the path "
            )
        dataset_name, dataset_path = pair
        dataset = SupportedDatasets(dataset_name)
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise InvalidArgument(f"Dataset path {dataset_path} doesn't exist.")
        return dataset, dataset_path

    def isfloat(self, x: str):
        try:
            float(x)
        except (TypeError, ValueError):
            return False
        else:
            return True

    def isint(self, x: str):
        try:
            a = float(x)
            b = int(x)
        except (TypeError, ValueError):
            return False
        else:
            return a == b

    def parse_kwargs(self, kwargs_strs: list[str], list_sep=",", key_value_sep="="):
        """
        Example:
            kwargs_str = stretch_factors=0.8,1.2 freq_mask_param=30
            returns {"stretch_factors": [0.8, 1.2], "freq_mask_param": 30}

        Args:
            kwargs_str: _description_
            list_sep: _description_..
            arg_sep: _description_..
        """
        if isinstance(kwargs_strs, str):
            kwargs_strs = [kwargs_strs]

        def parse_value(value: str):
            if self.isint(value):
                return int(value)
            if self.isfloat(value):
                return float(value)
            return value

        kwargs = {}
        for key_value in kwargs_strs:
            _kv = key_value.split(key_value_sep)
            assert (
                len(_kv) == 2
            ), f"Exactly one `{key_value_sep}` should appear in {key_value}"
            key, value = _kv
            value = [parse_value(v) for v in value.split(list_sep)]
            value = value if len(value) > 1 else value[0]
            kwargs[key] = value
        return kwargs

    def __str__(self):
        return self.dumps_yaml(allow_unicode=True, default_flow_style=False)


def get_default_value_for_field(field_str: str, cls=ConfigDefault):
    return cls.__dataclass_fields__[field_str].default_factory()


def get_default_config():
    return ConfigDefault()


default_config = get_default_config()
