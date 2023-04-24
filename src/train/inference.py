import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)

from src.config.argparse_with_config import ArgParseWithConfig
from src.data.datamodule import OurDataModule
from src.enums.enums import (
    SupportedAugmentations,
    SupportedLossFunctions,
    SupportedScheduler,
)
from src.features.audio_transform import AudioTransformBase, get_audio_transform
from src.features.augmentations import get_augmentations
from src.features.chunking import get_collate_fn
from src.model.model import SupportedModels, get_model, model_constructor_map
from src.train.callbacks import (
    FinetuningCallback,
    GeneralMetricsEpochLogger,
    OverrideEpochMetricCallback,
    TensorBoardHparamFixer,
)
from src.utils.utils_dataset import calc_instrument_weight
from src.utils.utils_exceptions import InvalidArgument, UnsupportedModel
from src.utils.utils_functions import add_prefix_to_keys

# def experiment_setup(config: ConfigDefault, pl_args: Namespace):
#     """Create experiment directory."""
#     timestamp = get_timestamp()
#     experiment_codeword = random_codeword()
#     experiment_name = f"{timestamp}_{experiment_codeword}_{config.model.value}"

#     output_dir = Path(config.output_dir)
#     output_dir.mkdir(exist_ok=True)
#     experiment_dir = Path(output_dir, experiment_name)
#     experiment_dir.mkdir(exist_ok=True)

#     filename_config = Path(experiment_dir, "config.yaml")
#     with open(filename_config, "w") as outfile:
#         yaml.dump(config, outfile)
#     filename_report = Path(output_dir, experiment_name, "log.txt")

#     stdout_to_file(filename_report)
#     print()
#     print("Created experiment directory:", str(experiment_dir))
#     print("Created log file:", str(filename_report))
#     print()
#     print("================== Config ==================\n\n", config)
#     print()
#     print(
#         "================== PyTorch Lightning ==================\n\n",
#         to_yaml(vars(pl_args)),
#     )
#     input("Review the config above. Press enter if you wish to continue: ")
#     return experiment_name, experiment_dir, output_dir


if __name__ == "__main__":
    parser = ArgParseWithConfig()
    args, config, pl_args = parser.parse_args()

    if config.model is None:
        raise InvalidArgument(f"--model is required {list(SupportedModels)}")
    if config.model not in model_constructor_map:
        raise UnsupportedModel(
            f"Model {config.model} is not in the model_constructor_map. Add the model enum to the model_constructor_map."
        )
    config.parse_dataset_paths()

    model_constructor: pl.LightningModule = model_constructor_map[config.model]
    model = model_constructor.load_from_checkpoint(config.ckpt)
    model_config = model.config

    audio_transform: AudioTransformBase = get_audio_transform(
        model_config,
        spectrogram_augmentation=None,
        waveform_augmentation=None,
    )

    collate_fn = get_collate_fn(model_config)

    datamodule = OurDataModule(
        train_paths=None,
        val_paths=None,
        test_paths=config.dataset_paths,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dataset_fraction=config.dataset_fraction,
        drop_last_sample=False,
        train_audio_transform=None,
        val_audio_transform=audio_transform,
        collate_fn=collate_fn,
        normalize_audio=model_config.normalize_audio,
        normalize_image=model_config.normalize_image,
        train_only_dataset=False,
        concat_n_samples=None,
        sum_two_samples=None,
        use_weighted_train_sampler=False,
        sampling_rate=model_config.sampling_rate,
    )

    # ================= SETUP CALLBACKS (auto checkpoint, tensorboard, early stopping...)========================

    train_dataloader_size = len(datamodule.train_dataloader())
    bar_refresh_rate = min(int(train_dataloader_size / config.bar_update), 200)

    log_dictionary = {
        **add_prefix_to_keys(vars(config), "user_args/"),
        **add_prefix_to_keys(vars(pl_args), "lightning_args/"),
        "test_size": len(datamodule.test_dataloader().dataset),
    }

    callbacks = [
        TQDMProgressBar(refresh_rate=bar_refresh_rate),
        TensorBoardHparamFixer(config_dict=log_dictionary),
        OverrideEpochMetricCallback(),
        GeneralMetricsEpochLogger(),
    ]

    callbacks.append(ModelSummary(max_depth=4))

    # ================= TRAINER ========================
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        pl_args,
        default_root_dir="inference",
        callbacks=callbacks,
    )
    trainer.test(model, datamodule=datamodule, ckpt_path=config.ckpt)
