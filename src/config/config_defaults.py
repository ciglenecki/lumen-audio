"""Default global config.

Important: 0 dependencies except to enums and log!
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from pathlib import Path

import pyrootutils
import torch
from simple_parsing.helpers import Serializable

from src.config.logger import log
from src.enums.enums import (
    NON_INFERENCE_DIR_TYPES,
    AudioTransforms,
    MetricMode,
    OptimizeMetric,
    SupportedAugmentations,
    SupportedDatasetDirType,
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


class InstrumentFamily(Enum):
    BRASS = "brass"
    GUITAR = "gutars"
    WOODWIND = "woodwind"
    STRINGS = "strings"
    VOICE = "voice"
    PERCUSSION = "percussion"


FAMILIY_TO_IDX = {
    InstrumentFamily.BRASS.value: 0,
    InstrumentFamily.GUITAR.value: 1,
    InstrumentFamily.WOODWIND.value: 2,
    InstrumentFamily.STRINGS.value: 3,
    InstrumentFamily.VOICE.value: 4,
    InstrumentFamily.PERCUSSION.value: 5,
}

INSTRUMENT_TO_FAMILY = {
    InstrumentEnums.CELLO.value: InstrumentFamily.STRINGS.value,
    InstrumentEnums.CLARINET.value: InstrumentFamily.WOODWIND.value,
    InstrumentEnums.FLUTE.value: InstrumentFamily.WOODWIND.value,
    InstrumentEnums.ACOUSTIC_GUITAR.value: InstrumentFamily.GUITAR.value,
    InstrumentEnums.ELECTRIC_GUITAR.value: InstrumentFamily.GUITAR.value,
    InstrumentEnums.ORGAN.value: InstrumentFamily.BRASS.value,
    InstrumentEnums.PIANO.value: InstrumentFamily.PERCUSSION.value,
    InstrumentEnums.SAXOPHONE.value: InstrumentFamily.BRASS.value,
    InstrumentEnums.TRUMPET.value: InstrumentFamily.BRASS.value,
    InstrumentEnums.VIOLIN.value: InstrumentFamily.STRINGS.value,
    InstrumentEnums.VOICE.value: InstrumentFamily.VOICE.value,
}


IDX_TO_FAMILY = {v: k for k, v in FAMILIY_TO_IDX.items()}


DEFAULT_NUM_LABELS = len(INSTRUMENT_TO_IDX)
DEFAULT_IRMAS_TRAIN_SIZE = 6705
DEFAULT_IRMAS_TEST_SIZE = 2874
NUM_RGB_CHANNELS = 3

DEFAULT_MFCC_MEAN = -7.3612
DEFAULT_MFCC_STD = 56.4464

# DEFAULT_MEL_SPECTROGRAM_MEAN = 0.4125 # with some augmentations?
# DEFAULT_MEL_SPECTROGRAM_STD = 2.3365 # with some augmentations?

DEFAULT_MEL_SPECTROGRAM_MEAN = 0.413
DEFAULT_MEL_SPECTROGRAM_STD = 2.582

DEFAULT_MULTI_SPECTROGRAM_MEAN = torch.tensor([[4.1400e-01, 1.3334e03, 4.6917e-01]])
DEFAULT_MULTI_SPECTROGRAM_STD = torch.tensor([[2.5947e00, 5.5186e02, 3.1212e-01]])

DEFAULT_AST_MEAN = -4.2677393
DEFAULT_AST_STD = 4.5689974

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
# _default_augmentations_set.discard(SupportedAugmentations.SUM_N_SAMPLES)
_default_augmentations_set.discard(SupportedAugmentations.NORM_AFTER_TIME_AUGS)
_default_augmentations_list = list(_default_augmentations_set)

TAG_AST_AUDIOSET = "MIT/ast-finetuned-audioset-10-10-0.4593"
TAG_WAV2VEC2_MUSIC = "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres"
TAG_IMAGENET1K_V2 = "IMAGENET1K_V2"
TAG_IMAGENET1K_V1 = "IMAGENET1K_V1"
DEFAULT_PRETRAINED_TAG = "DEFAULT"

DEFAULT_PRETRAINED_TAG_MAP = {
    SupportedModels.AST: TAG_AST_AUDIOSET,
    SupportedModels.WAV2VEC_CNN: TAG_WAV2VEC2_MUSIC,
    SupportedModels.WAV2VEC: TAG_WAV2VEC2_MUSIC,
    SupportedModels.EFFICIENT_NET_V2_S: TAG_IMAGENET1K_V1,
    SupportedModels.EFFICIENT_NET_V2_M: TAG_IMAGENET1K_V1,
    SupportedModels.EFFICIENT_NET_V2_L: TAG_IMAGENET1K_V1,
    SupportedModels.RESNEXT50_32X4D: TAG_IMAGENET1K_V2,
    SupportedModels.RESNEXT101_32X8D: TAG_IMAGENET1K_V2,
    SupportedModels.RESNEXT101_64X4D: TAG_IMAGENET1K_V1,
    SupportedModels.CONVNEXT_TINY: TAG_IMAGENET1K_V1,
    SupportedModels.CONVNEXT_SMALL: TAG_IMAGENET1K_V1,
    SupportedModels.CONVNEXT_LARGE: TAG_IMAGENET1K_V1,
    SupportedModels.CONVNEXT_BASE: TAG_IMAGENET1K_V1,
    SupportedModels.MOBILENET_V3_LARGE: TAG_IMAGENET1K_V1,
    SupportedModels.MOBNET: TAG_IMAGENET1K_V1,
    SupportedModels.CONVLSTM: None,
}
ALL_INSTRUMENTS = [e.value for e in InstrumentEnums]
ALL_INSTRUMENTS_NAMES = [INSTRUMENT_TO_FULLNAME[ins] for ins in ALL_INSTRUMENTS]

AUDIO_EXTENSIONS = ["wav", "mp3", "ogg"]

# Order is important here, since we use the index as the label
AST_INSTRUMENTS_IRMAS = {
    193: "Cello",
    198: "Clarinet",
    196: "Flute",
    143: "Acoustic guitar",
    141: "Electric guitar",
    155: "Organ",
    153: "Piano",
    197: "Saxophone",
    187: "Trumpet",
    191: "Violin, fiddle",
    27: "Singing",
}

USAGE_TEXT_PATHS = f"Usage:\t--train-paths <TYPE>:/path/to/dataset (for training)\n\t--val-paths <TYPE>:/path/to/dataset (for training)\n\t--test-paths <TYPE>:/path/to/dataset (for inference)\nSupported <TYPE>: {[ d.value for d in SupportedDatasetDirType]}"


def create(arg, **kwargs):
    """We need this useless helper function since you can't type augmentations: SupportedAugmentations = {} in ConfigDefault."""
    return field(default_factory=lambda: arg, **kwargs)


def default_path(path: Path | None, default_value: Path, create_if_none=False):
    """Return default value if object is none."""

    if path is not None:  # return explicit path
        return path

    if create_if_none:  # create and return default value
        default_value.mkdir(parents=True, exist_ok=True)
        return default_value

    if default_value.exists():  # return default path
        return default_value

    return None


def dir_to_enum_and_path(
    string: str,
    allow_raw_path=False,
) -> tuple[SupportedDatasetDirType, Path]:
    """
    Example:
        string: irmas:/path/to/irmas
        return (SupportedDatasetDirType.IRMAS, Path("/path/to/irmas"))
    """
    delimiter = ":"
    pair = string.split(delimiter)

    if len(pair) == 1 and allow_raw_path:
        return SupportedDatasetDirType.INFERENCE, Path(string)

    if len(pair) != 2:
        raise InvalidArgument(
            f"Pair {pair} needs to have two elements split with '{delimiter}'."
        )
    dataset_name, dataset_path = pair

    try:
        dataset = SupportedDatasetDirType(dataset_name)
    except ValueError as e:
        raise ValueError(
            f"{str(e)}. Choose one of the following  {[ d.value for d in SupportedDatasetDirType]} (or if you are developing a new dataset, add a new entry into the SupportedDatasetDirType enum)"
        )
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise InvalidArgument(f"Dataset path {dataset_path} doesn't exist.")
    return dataset, dataset_path


def parse_dataset_paths(
    data_dir: str | list[str],
    allow_raw_path=False,
) -> list[tuple[SupportedDatasetDirType, Path]]:
    # Parse strings to dataset type and path
    try:
        if isinstance(data_dir, str):
            return [dir_to_enum_and_path(data_dir, allow_raw_path)]
        elif isinstance(data_dir, list):
            return [dir_to_enum_and_path(d, allow_raw_path) for d in data_dir]
    except InvalidArgument as e:
        msg = USAGE_TEXT_PATHS
        raise InvalidArgument(f"\n\n{str(e)}\n\n{msg}")


@dataclass
class ConfigDefault(Serializable):
    path_workdir: Path = create(
        Path(pyrootutils.find_root(search_from=__file__, indicator=".project-root"))
    )
    """Path to the root of the project."""

    path_data: Path | None = create(None)
    path_irmas: Path | None = create(None)
    path_irmas_train: Path | None = create(None)
    path_irmas_test: Path | None = create(None)
    path_irmas_train_features: Path | None = create(None)
    path_irmas_sample: Path | None = create(None)
    path_openmic: Path | None = create(None)
    path_models: Path | None = create(None)
    path_models_quick: Path | None = create(None)
    path_background_noise: Path | None = create(None)
    path_figures: Path | None = create(None)
    path_embeddings: Path | None = create(None)

    train_paths: list[str] | None = create(None)
    """Dataset root directories that will be used for training in the following format: --train-paths irmastrain:/path/to/dataset or openmic:/path/to/dataset"""

    val_paths: list[str] | None = create(None)
    """Dataset root directories that will be used for validation in the following format: --val-paths irmastest:/path/to/dataset openmic:/path/to/dataset. If --val-paths is not provided val dir will be split to val and test."""

    test_paths: list[str] | None = create(None)
    """Dataset root directories that will be used for testing in the following format: --val-paths irmastest:/path/to/dataset openmic:/path/to/dataset"""

    dataset_paths: list[str] | None = create(None)
    """Dataset path with the following format format: --dataset-paths inference:/path/to/dataset openmic:/path/to/dataset"""

    train_only_dataset: bool = create(False)
    """Use only the train portion of the dataset and split it 0.8 0.2"""

    dataset_fraction: float = create(1.0)
    """Reduce each dataset split (train, val, test) by a fraction."""

    num_labels: int = create(DEFAULT_NUM_LABELS)
    """Total number of possible lables"""

    train_override_csvs: list[Path] | None = create(None)
    """CSV files with columns 'filename, sax, gac, org, ..., cla' where filename is path and each instrument is either 0 or 1"""

    # ======================== DPS ===========================

    sampling_rate: int = create(16_000)
    """Audio sampling rate"""

    n_fft: int = create(400)
    """Length of the signal you want to calculate the Fourier transform of"""

    hop_length: int = create(160)
    """Hop length which will be used during STFT cacualtion"""

    n_mels: int = create(128)
    """Number of mel bins you want to caculate"""

    n_mfcc: int = create(20)
    """Number of Mel-frequency cepstrum (MFCC) coefficients"""

    image_size: tuple[int, int] | None = create(None)
    """The dimension to resize the image to."""

    normalize_audio: bool = create(True)
    """Do normalize audio"""

    normalize_image: bool = create(True)
    """Do image audio"""

    max_num_width_samples: float | None = create(None)
    """Maximum number samples along the time dimension. For spectrogram: width truncation, for audio: waveform truncation. Useful for limiting transformer input size."""

    augmentations: list[SupportedAugmentations] = create(_default_augmentations_list)
    """Transformation which will be performed on audio and labels"""

    aug_kwargs: list[str] = create(
        dict(
            stretch_factors=[0.8, 1.25],
            time_inversion_p=0.5,
            freq_mask_param=30,
            hide_random_pixels_p=0.25,
            std_noise=0.01,
            concat_n_samples=3,
            sum_n_samples=3,
            path_background_noise=None,
            time_mask_max_percentage=0.3,
        )
    )
    """Arguments are split by space, mutiple values are sep'ed by comma (,). E.g. stretch_factors=0.8,1.2 freq_mask_param=30 hide_random_pixels_p=0.5"""

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

    finetune_train_bn: bool = create(True)
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

    experiment_suffix: str = create("")

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

    pretrained_tag: str | None = create(None)
    """The string that denotes the pretrained weights used."""

    head: SupportedHeads = create(SupportedHeads.DEEP_HEAD)
    """Type of classification head which will be used for classification. This is almost always the last layer."""

    head_hidden_dim: list[int] = create([])
    """List of integers which specifies the hidden layers of head (classifer). For each integer one extra hidden layer will be created in the middle of the head (classifier). The integer value specifies number of  hidden dimensions."""

    ckpt: Path | None = create(None)
    """.ckpt file, automatically restores model, epoch, step, LR schedulers, etc..."""

    use_fluffy: bool = create(False)
    """Use multiple optimizers for Fluffy."""

    use_rgb: bool | None = create(None)

    prediction_threshold: float | None = create(None)

    # ======================== OPTIM ===========================

    optimizer: str = create(SupportedOptimizer.ADAMW)
    """Optimizer"""

    scheduler: SupportedScheduler = create(SupportedScheduler.ONECYCLE)

    loss_function: SupportedLossFunctions = create(SupportedLossFunctions.CROSS_ENTROPY)
    """Loss function"""

    loss_function_kwargs: list[str] = create({"reduction": "none"})
    """Loss function kwargs"""

    add_instrument_loss: float | None = create(None)
    """Instrument Family Loss factor"""

    lr: float = create(5e-5)
    """Learning rate"""

    lr_onecycle_max: float = create(5e-4)
    """Maximum lr OneCycle scheduler reaches"""

    lr_warmup: float = create(3e-4)
    """warmup learning rate"""

    # ======================== LOGS ===========================
    log_per_instrument_metrics: bool = create(True)
    """Along with aggregated metrics, also log per instrument metrics."""

    bar_update: int = create(30)
    """Number of TQDM updates in one epoch."""

    log_every_n_steps: int = create(30)
    """How often (steps) to log metrics."""

    verify_config: bool = create(False)
    """Asks users to press enter to confirm the config"""

    def after_init(self):
        """This function sets defaults, dynamically changes some of the arguments based on other
        arguments."""

        self.path_data = default_path(self.path_data, Path("data"), create_if_none=True)
        self.path_irmas = default_path(
            self.path_irmas, Path("data", "irmas"), create_if_none=True
        )
        self.path_irmas_train = default_path(
            self.path_irmas_train, Path("data", "irmas", "train")
        )
        self.path_irmas_test = default_path(
            self.path_irmas_test, Path("data", "irmas", "test")
        )
        self.path_irmas_sample = default_path(
            self.path_irmas_sample, Path("data", "irmas_sample")
        )
        self.path_irmas_train_features = default_path(
            self.path_irmas_train_features,
            Path(
                "embeddings",
                "data-irmas-train_ast_MIT-ast-finetuned-audioset-10-10-0.4593",
            ),
        )
        self.path_openmic = default_path(self.path_openmic, Path("data", "openmic"))

        self.path_models = default_path(
            self.path_models, Path("models"), create_if_none=True
        )
        self.path_models_quick = default_path(
            self.path_models_quick, Path("models_quick"), create_if_none=True
        )
        self.path_figures = default_path(
            self.path_figures, Path("figures"), create_if_none=True
        )

        self.path_embeddings = default_path(
            self.path_embeddings, Path("embeddings"), create_if_none=False
        )

        self.path_background_noise = default_path(
            self.path_background_noise, Path("data", "ecs50")
        )

        self.output_dir = self.path_models

        # aug_kwargs can be either a dictionary or a string which will be parsed as kwargs dict
        if self.aug_kwargs is not None and not isinstance(self.aug_kwargs, dict):
            try:
                if isinstance(self.aug_kwargs, str):
                    self.aug_kwargs = [self.aug_kwargs]
                override_kwargs = self.parse_kwargs(self.aug_kwargs)
            except Exception as e:
                raise InvalidArgument(
                    f"{str(e)}\n. --aug-kwargs should have the following structure: 'key=value1,value2 key2=value3' e.g. 'stretch_factors=0.8,1.2 freq_mask_param=30'"
                )

            self.aug_kwargs: dict = get_default_value_for_field("aug_kwargs", self)
            self.aug_kwargs.update(override_kwargs)

        if (
            self.path_background_noise is None
            and SupportedAugmentations.BACKGROUND_NOISE in self.augmentations
        ):
            log.warning(
                "Removing BACKGROUND_NOISE augmentation because path_background_noise directory is not found. You can set it with --path-background-noise."
            )
            self.augmentations.remove(SupportedAugmentations.BACKGROUND_NOISE)
        elif (
            self.path_background_noise is not None
            and self.path_background_noise.exists()
        ):
            self.aug_kwargs.update(
                dict(path_background_noise=self.path_background_noise)
            )

        # Dynamically set the RGB option based on model's architecture
        if self.model is not None and self.use_rgb is None:
            USE_RGB = {
                SupportedModels.AST: False,
                SupportedModels.WAV2VEC_CNN: None,
                SupportedModels.WAV2VEC: None,
                SupportedModels.EFFICIENT_NET_V2_S: True,
                SupportedModels.EFFICIENT_NET_V2_M: True,
                SupportedModels.EFFICIENT_NET_V2_L: True,
                SupportedModels.RESNEXT50_32X4D: True,
                SupportedModels.RESNEXT101_32X8D: True,
                SupportedModels.RESNEXT101_64X4D: True,
                SupportedModels.CONVNEXT_TINY: True,
                SupportedModels.CONVNEXT_SMALL: True,
                SupportedModels.CONVNEXT_LARGE: True,
                SupportedModels.CONVNEXT_BASE: True,
                SupportedModels.MOBILENET_V3_LARGE: True,
                SupportedModels.MOBNET: True,
                SupportedModels.CONVLSTM: False,
            }
            self.use_rgb = USE_RGB[self.model]

        # Dynamically set the image size
        if self.model is not None and self.image_size is None:
            IMAGE_SIZE_MAP = {
                SupportedModels.AST: None,
                SupportedModels.WAV2VEC_CNN: None,
                SupportedModels.WAV2VEC: None,
                SupportedModels.EFFICIENT_NET_V2_S: (384, 384),
                SupportedModels.EFFICIENT_NET_V2_M: (480, 480),
                SupportedModels.EFFICIENT_NET_V2_L: (480, 480),
                SupportedModels.RESNEXT50_32X4D: (224, 224),
                SupportedModels.RESNEXT101_32X8D: (224, 224),
                SupportedModels.RESNEXT101_64X4D: (224, 224),
                SupportedModels.CONVNEXT_TINY: (224, 224),
                SupportedModels.CONVNEXT_SMALL: (224, 224),
                SupportedModels.CONVNEXT_LARGE: (224, 224),
                SupportedModels.CONVNEXT_BASE: (224, 224),
                SupportedModels.MOBILENET_V3_LARGE: (224, 224),
                SupportedModels.MOBNET: (224, 224),
                SupportedModels.CONVLSTM: None,
            }
            self.image_size = IMAGE_SIZE_MAP[self.model]
            # Dynamically set pretrained tag
        if self.model is not None and self.pretrained and self.pretrained_tag is None:
            if self.model not in DEFAULT_PRETRAINED_TAG_MAP:
                raise InvalidArgument(
                    f"Couldn't find pretrained tag for pretrained model {self.model}. Add a new tag to the DEFAULT_PRETRAINED_TAG_MAP map or pass the --pretrained-tag <tag> argument."
                )
            self.pretrained_tag = DEFAULT_PRETRAINED_TAG_MAP[self.model]

        # Dynamically AST DSP attributes and augmentations
        if (
            self.model == SupportedModels.AST
            and self.pretrained
            and self.pretrained_tag == TAG_AST_AUDIOSET
        ):
            self.n_fft = 400
            self.hop_length = 160
            self.n_mels = 128

            if self.augmentations == get_default_value_for_field("augmentations", self):
                _augmentations_set = set(self.augmentations)
                _augmentations_set.add(SupportedAugmentations.CONCAT_N_SAMPLES)
                _augmentations_set.add(SupportedAugmentations.SUM_N_SAMPLES)
                self.augmentations = list(_augmentations_set)

        # Dynamically wav2vec attributes and augmentations
        if self.model == SupportedModels.WAV2VEC and self.pretrained:
            if self.augmentations == get_default_value_for_field("augmentations", self):
                _augmentations_set = set(self.augmentations)
                _augmentations_set.add(SupportedAugmentations.CONCAT_N_SAMPLES)
                self.aug_kwargs["concat_n_samples"] = 3
                self.augmentations = list(_augmentations_set)

        # Dynamically wav2vec attributes and augmentations
        if self.model == SupportedModels.WAV2VEC_CNN and self.pretrained:
            if self.augmentations == get_default_value_for_field("augmentations", self):
                _augmentations_set = set(self.augmentations)
                _augmentations_set.add(SupportedAugmentations.CONCAT_N_SAMPLES)
                self.aug_kwargs["concat_n_samples"] = 2
                self.augmentations = list(_augmentations_set)

        # Set typical weight decay for optimizers.
        if self.weight_decay is None and self.optimizer == SupportedOptimizer.ADAM:
            self.weight_decay = 0
        if self.weight_decay is None and self.optimizer == SupportedOptimizer.ADAMW:
            self.weight_decay = 1e-2

        if self.model is not None and len(self.head_hidden_dim) > 0:
            if (
                self.head_hidden_dim != list(sorted(self.head_hidden_dim, reverse=True))
                or len(self.head_hidden_dim) > 2
            ):
                raise InvalidArgument(
                    "--head-hidden-dim values should have descending values and shouldn't contain more than 2 values"
                )

        self.set_train_paths()
        self.set_val_paths()
        self.set_test_paths()
        self.set_dataset_paths()

    def check_all_paths_non_inference(
        self, dataset_paths_with_type: list[tuple[SupportedDatasetDirType, Path]]
    ):
        for ds_type, ds_path in dataset_paths_with_type:
            if ds_type not in NON_INFERENCE_DIR_TYPES:
                raise InvalidArgument(
                    f"Invalid dataset type: {ds_type}. Since labels have to be loaded use one of the following dataset types: {NON_INFERENCE_DIR_TYPES}"
                )

    def set_train_paths(self):
        if self.train_paths is not None:
            self.train_paths = parse_dataset_paths(self.train_paths)
            self.check_all_paths_non_inference(self.train_paths)

    def set_val_paths(self):
        if self.val_paths is not None:
            self.val_paths = parse_dataset_paths(self.val_paths)
            self.check_all_paths_non_inference(self.train_paths)

    def set_test_paths(self):
        if self.test_paths is not None:
            self.test_paths = parse_dataset_paths(self.test_paths)
            self.check_all_paths_non_inference(self.train_paths)

    def set_dataset_paths(self):
        if self.dataset_paths is not None:
            self.dataset_paths = parse_dataset_paths(
                self.dataset_paths, allow_raw_path=True
            )

    def required_train_paths(self):
        if self.train_paths is None:
            raise InvalidArgument(f"--train-paths is required\n{USAGE_TEXT_PATHS}")

    def required_val_paths(self):
        if self.train_paths is None:
            raise InvalidArgument(f"--val-paths is required\n{USAGE_TEXT_PATHS}")

    def required_test_paths(self):
        if self.train_paths is None:
            raise InvalidArgument(f"--test-paths is required\n{USAGE_TEXT_PATHS}")

    def required_dataset_paths(self):
        if self.dataset_paths is None:
            raise InvalidArgument(f"--dataset-paths is required\n{USAGE_TEXT_PATHS}")

    def required_model(self):
        if self.model is None:
            raise InvalidArgument(f"--model is required {list(SupportedModels)}")

    def required_ckpt(self):
        if self.ckpt is None:
            raise InvalidArgument(
                "--ckpt path to a saved model (checkpoint) is required which usually ends with .ckpt"
            )

    def required_audio_transform(self):
        if self.audio_transform is None:
            raise InvalidArgument(
                f"--audio-transform is required {list(AudioTransforms)}"
            )

    def set_model_enum_from_ckpt(self) -> SupportedModels:
        for e in list(SupportedModels):
            if e.value in str(self.ckpt.stem):
                self.model = e
                break
        return self.model

    def validate_train_args(self):
        """This function validates arguments before training."""
        self.required_train_paths()
        self.required_val_paths()
        self.required_model()
        self.required_audio_transform()

        if self.metric and not self.metric_mode:
            raise InvalidArgument("Can't pass --metric without passing --metric-mode")

        # loss_function_kwargs can be either a dictionary or a string which will be parsed as kwargs dict
        if self.loss_function_kwargs is not None and not isinstance(
            self.loss_function_kwargs, dict
        ):
            try:
                override_kwargs = self.parse_kwargs(self.loss_function_kwargs)
            except Exception as e:
                raise InvalidArgument(
                    f"{str(e)}\n. --loss-function-kwargs should have the following structure: 'key=value1 key2=value2'"
                )

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

        if self.max_num_width_samples is None:
            # There's no max num width for image based models because maximum is defined by their architecture.
            MAX_NUM_WIDTH_SAMPLE = {
                SupportedModels.AST: 1024,
                SupportedModels.WAV2VEC_CNN: self.sampling_rate * 3 * 2,
                SupportedModels.WAV2VEC: self.sampling_rate * 10,
                SupportedModels.EFFICIENT_NET_V2_S: None,
                SupportedModels.EFFICIENT_NET_V2_M: None,
                SupportedModels.EFFICIENT_NET_V2_L: None,
                SupportedModels.RESNEXT50_32X4D: None,
                SupportedModels.RESNEXT101_32X8D: None,
                SupportedModels.RESNEXT101_64X4D: None,
                SupportedModels.CONVNEXT_TINY: None,
                SupportedModels.CONVNEXT_SMALL: None,
                SupportedModels.CONVNEXT_LARGE: None,
                SupportedModels.CONVNEXT_BASE: None,
                SupportedModels.MOBILENET_V3_LARGE: None,
                SupportedModels.MOBNET: None,
                SupportedModels.CONVLSTM: None,
            }
            self.max_num_width_samples = MAX_NUM_WIDTH_SAMPLE[self.model]

        if (
            self.finetune_head
            and (self.finetune_head_epochs > 0)
            and (self.finetune_head_epochs >= self.epochs)
        ):
            raise InvalidArgument(
                "Please set --finetune-heads-epochs int so it's bigger than 0 and less than --epochs int."
            )

        if self.head == SupportedHeads.ATTENTION_HEAD:
            SUPPORTS_ATTENTION_HEAD = {
                SupportedModels.AST: True,
                SupportedModels.WAV2VEC_CNN: True,
                SupportedModels.WAV2VEC: True,
                SupportedModels.EFFICIENT_NET_V2_S: False,
                SupportedModels.EFFICIENT_NET_V2_M: False,
                SupportedModels.EFFICIENT_NET_V2_L: False,
                SupportedModels.RESNEXT50_32X4D: False,
                SupportedModels.RESNEXT101_32X8D: False,
                SupportedModels.RESNEXT101_64X4D: False,
                SupportedModels.CONVNEXT_TINY: False,
                SupportedModels.CONVNEXT_SMALL: False,
                SupportedModels.CONVNEXT_LARGE: False,
                SupportedModels.CONVNEXT_BASE: False,
                SupportedModels.MOBILENET_V3_LARGE: False,
                SupportedModels.MOBNET: False,
            }
            if not SUPPORTS_ATTENTION_HEAD[self.model]:
                raise InvalidArgument(
                    f"You can't use ATTENTION_HEAD with {self.model}. Only models {[k.name for k, v in SUPPORTS_ATTENTION_HEAD.items() if v]} support it."
                )
        if self.model and self.audio_transform:
            pair = (self.model, self.audio_transform)
            supported_combinations = list(
                product(
                    {
                        SupportedModels.EFFICIENT_NET_V2_S,
                        SupportedModels.EFFICIENT_NET_V2_M,
                        SupportedModels.EFFICIENT_NET_V2_L,
                        SupportedModels.RESNEXT50_32X4D,
                        SupportedModels.RESNEXT101_32X8D,
                        SupportedModels.RESNEXT101_64X4D,
                        SupportedModels.CONVNEXT_TINY,
                        SupportedModels.CONVNEXT_SMALL,
                        SupportedModels.CONVNEXT_LARGE,
                        SupportedModels.CONVNEXT_BASE,
                        SupportedModels.MOBILENET_V3_LARGE,
                        SupportedModels.MOBNET,
                    },
                    {
                        AudioTransforms.MEL_SPECTROGRAM,
                        AudioTransforms.MULTI_SPECTROGRAM,
                        AudioTransforms.MFCC,
                    },
                )
            ) + [
                (SupportedModels.AST, AudioTransforms.AST),
                (SupportedModels.WAV2VEC, AudioTransforms.WAV2VEC),
            ]

            if pair not in supported_combinations:
                raise InvalidArgument(
                    f"You can't use {self.audio_transform.name} with {self.model.name}. Supported audio audiotransforms are {[v.name for k, v in supported_combinations if k == self.model ]} support it."
                )

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

    # def __repr__(self):
    #     pass

    def __str__(self):
        return self.dumps_yaml(allow_unicode=True, default_flow_style=False)


def get_default_value_for_field(field_str: str, cls=ConfigDefault):
    return cls.__dataclass_fields__[field_str].default_factory()


def get_default_config():
    config = ConfigDefault()
    config.after_init()
    return config


def get_default_config_no_init():
    config = ConfigDefault()
    return config
