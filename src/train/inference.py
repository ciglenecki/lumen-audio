from argparse import Namespace
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)

from src.config.argparse_with_config import ArgParseWithConfig
from src.config.config_defaults import ConfigDefault
from src.config.config_train import get_config
from src.data.datamodule import IRMASDataModule
from src.enums.enums import (
    MetricMode,
    OptimizeMetric,
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
from src.utils.utils_functions import (
    add_prefix_to_keys,
    get_timestamp,
    random_codeword,
    stdout_to_file,
    to_yaml,
)
from src.utils.utils_model import print_modules

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

    # TODO continue here
    (
        train_spectrogram_augmentation,
        train_waveform_augmentation,
        val_spectrogram_augmentation,
        val_waveform_augmentation,
    ) = get_augmentations(model_config)

    inference_audio_transform: AudioTransformBase = get_audio_transform(
        config,
        spectrogram_augmentation=val_spectrogram_augmentation,
        waveform_augmentation=val_waveform_augmentation,
    )

    concat_n_samples = (
        config.aug_kwargs["concat_n_samples"]
        if SupportedAugmentations.CONCAT_N_SAMPLES in config.augmentations
        else None
    )
    collate_fn = get_collate_fn(config)

    datamodule = IRMASDataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dataset_fraction=config.dataset_fraction,
        drop_last_sample=config.drop_last,
        train_audio_transform=None,
        val_audio_transform=None,
        collate_fn=collate_fn,
        normalize_audio=config.normalize_audio,
        train_only_dataset=config.train_only_dataset,
        concat_n_samples=concat_n_samples,
        sum_two_samples=SupportedAugmentations.SUM_TWO_SAMPLES in config.augmentations,
        use_weighted_train_sampler=config.use_weighted_train_sampler,
        normalize_image=config.normalize_image,
    )

    if config.loss_function == SupportedLossFunctions.CROSS_ENTROPY:
        loss_function = torch.nn.BCEWithLogitsLoss(**config.loss_function_kwargs)
    if config.loss_function == SupportedLossFunctions.CROSS_ENTROPY_POS_WEIGHT:
        kwargs = {
            **config.loss_function_kwargs,
            "pos_weight": calc_instrument_weight(datamodule.count_classes()),
        }
        loss_function = torch.nn.BCEWithLogitsLoss(**kwargs)

    model = get_model(config, loss_function=loss_function)

    # ================= SETUP CALLBACKS (auto checkpoint, tensorboard, early stopping...)========================

    train_dataloader_size = len(datamodule.train_dataloader())
    bar_refresh_rate = int(train_dataloader_size / config.bar_update)

    callback_early_stopping = EarlyStopping(
        monitor=optimizer_metric_str,
        mode=metric_mode_str,
        patience=config.early_stopping_metric_patience,
        check_on_train_epoch_end=config.check_on_train_epoch_end,
        verbose=True,
    )

    callback_checkpoint = ModelCheckpoint(
        monitor=optimizer_metric_str,
        mode=metric_mode_str,
        filename="_".join(
            [
                experiment_name,
                "val_acc_{val/f1_score_epoch:.4f}",
                "val_loss_{val/loss_epoch:.4f}",
            ]
        ),
        auto_insert_metric_name=False,
        save_on_train_epoch_end=config.save_on_train_epoch_end,
        verbose=True,
    )

    log_dictionary = {
        **add_prefix_to_keys(vars(config), "user_args/"),
        **add_prefix_to_keys(vars(pl_args), "lightning_args/"),
        "train_size": len(datamodule.train_dataloader().dataset),
        "val_size": len(datamodule.val_dataloader().dataset),
        "test_size": len(datamodule.test_dataloader().dataset),
    }

    callbacks = [
        callback_checkpoint,
        callback_early_stopping,
        TQDMProgressBar(refresh_rate=bar_refresh_rate),
        TensorBoardHparamFixer(config_dict=log_dictionary),
        OverrideEpochMetricCallback(),
        GeneralMetricsEpochLogger(),
    ]

    if config.finetune_head:
        callbacks.append(
            FinetuningCallback(finetune_head_epochs=config.finetune_head_epochs)
        )

    callbacks.append(ModelSummary(max_depth=4))

    auto_lr_find = config.scheduler == SupportedScheduler.AUTO_LR

    # ================= TRAINER ========================
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        pl_args,
        logger=[tensorboard_logger],
        default_root_dir=output_dir,
        callbacks=callbacks,
    )

    if config.scheduler == SupportedScheduler.AUTO_LR.value:
        lr_finder = trainer.tuner.lr_find(
            model, datamodule=datamodule, num_training=100
        )
        if lr_finder is None:
            print("Cant find best learning rate")
            exit(1)
        # Results can be found in
        lr_finder.results

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.savefig("best_auti_lr.png")
        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        print(new_lr)
        exit(1)

    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt)
    trainer.test(model, datamodule)
