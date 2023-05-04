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

from src.config import config_defaults
from src.config.config_defaults import ConfigDefault
from src.config.config_train import get_config
from src.data.datamodule import OurDataModule
from src.enums.enums import (
    MetricMode,
    OptimizeMetric,
    SupportedAugmentations,
    SupportedLossFunctions,
    SupportedScheduler,
)
from src.features.audio_transform import AudioTransformBase, get_audio_transform
from src.features.augmentations import get_augmentations
from src.features.chunking import collate_fn_feature
from src.model.loss_functions import FocalLoss
from src.model.model import get_model
from src.train.callbacks import (
    FinetuningCallback,
    GeneralMetricsEpochLogger,
    OverrideEpochMetricCallback,
    TensorBoardHparamFixer,
)
from src.utils.utils_dataset import calc_instrument_weight
from src.utils.utils_functions import (
    add_prefix_to_keys,
    dict_with_keys,
    get_timestamp,
    random_codeword,
    stdout_to_file,
    to_yaml,
)
from src.utils.utils_model import print_params

# from pytorch_lightning.utilities.seed import seed_everything


def experiment_setup(config: ConfigDefault, pl_args: Namespace):
    """Create experiment directory."""
    timestamp = get_timestamp()
    experiment_codeword = random_codeword()
    experiment_name_list = [timestamp, experiment_codeword, config.model.value]
    if config.experiment_suffix:
        experiment_name_list.append(config.experiment_suffix)
    experiment_name = "_".join(experiment_name_list)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    experiment_dir = Path(output_dir, experiment_name)
    experiment_dir.mkdir(exist_ok=True)

    filename_config = Path(experiment_dir, "config.yaml")
    with open(filename_config, "w") as outfile:
        yaml.dump(config, outfile)
    filename_report = Path(output_dir, experiment_name, "log.txt")

    stdout_to_file(filename_report)
    print()
    print("Created experiment directory:", str(experiment_dir))
    print("Created log file:", str(filename_report))
    print()
    print("================== Config ==================\n\n", config)
    print()
    print(
        "================== PyTorch Lightning ==================\n\n",
        to_yaml(vars(pl_args)),
    )
    if config.verify_config:
        input("Review the config above. Press enter if you wish to continue: ")
    return experiment_name, experiment_dir, output_dir


if __name__ == "__main__":
    # seed_everything(42)

    config, pl_args = get_config()

    experiment_name, experiment_dir, output_dir = experiment_setup(config, pl_args)

    (
        train_spectrogram_augmentation,
        train_waveform_augmentation,
        val_spectrogram_augmentation,
        val_waveform_augmentation,
    ) = get_augmentations(config)

    train_audio_transform: AudioTransformBase = get_audio_transform(
        config,
        spectrogram_augmentation=train_spectrogram_augmentation,
        waveform_augmentation=train_waveform_augmentation,
    )
    val_audio_transform: AudioTransformBase = get_audio_transform(
        config,
        spectrogram_augmentation=val_spectrogram_augmentation,
        waveform_augmentation=val_waveform_augmentation,
    )

    concat_n_samples = (
        config.aug_kwargs["concat_n_samples"]
        if SupportedAugmentations.CONCAT_N_SAMPLES in config.augmentations
        else None
    )
    sum_n_samples = (
        config.aug_kwargs["sum_n_samples"]
        if SupportedAugmentations.SUM_N_SAMPLES in config.augmentations
        else None
    )
    collate_fn = collate_fn_feature

    datamodule = OurDataModule(
        train_paths=config.train_paths,
        val_paths=config.val_paths,
        test_paths=config.test_paths,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dataset_fraction=config.dataset_fraction,
        drop_last_sample=config.drop_last,
        train_audio_transform=train_audio_transform,
        val_audio_transform=val_audio_transform,
        collate_fn=collate_fn,
        normalize_audio=config.normalize_audio,
        normalize_image=config.normalize_image,
        train_only_dataset=config.train_only_dataset,
        concat_n_samples=concat_n_samples,
        sum_n_samples=sum_n_samples,
        use_weighted_train_sampler=config.use_weighted_train_sampler,
        sampling_rate=config.sampling_rate,
        train_override_csvs=config.train_override_csvs,
    )
    datamodule.setup_for_train()
    datamodule.setup_for_inference()

    if config.loss_function == SupportedLossFunctions.CROSS_ENTROPY:
        loss_function = torch.nn.BCEWithLogitsLoss(**config.loss_function_kwargs)
    elif config.loss_function == SupportedLossFunctions.CROSS_ENTROPY_POS_WEIGHT:
        instrument_count = dict_with_keys(
            datamodule.get_train_dataset_stats(), config_defaults.ALL_INSTRUMENTS
        )
        kwargs = {
            **config.loss_function_kwargs,
            "pos_weight": calc_instrument_weight(instrument_count),
        }
        loss_function = torch.nn.BCEWithLogitsLoss(**kwargs, reduction="none")
    elif config.loss_function == SupportedLossFunctions.FOCAL_LOSS:
        loss_function = FocalLoss(**config.loss_function_kwargs)
    elif config.loss_function == SupportedLossFunctions.FOCAL_LOSS_POS_WEIGHT:
        instrument_count = dict_with_keys(
            datamodule.get_train_dataset_stats(), config_defaults.ALL_INSTRUMENTS
        )
        kwargs = {
            **config.loss_function_kwargs,
            "pos_weight": calc_instrument_weight(instrument_count),
        }
        loss_function = FocalLoss(**kwargs)

    model = get_model(config, loss_function=loss_function)
    print_params(model)

    # ================= SETUP CALLBACKS (auto checkpoint, tensorboard, early stopping...)========================
    metric_mode_str = MetricMode(config.metric_mode).value
    optimizer_metric_str = OptimizeMetric(config.metric).value

    csv_logger = pl_loggers.CSVLogger(
        save_dir=str(output_dir),
        name=experiment_name,
        version=".",
    )
    tensorboard_logger = pl_loggers.TensorBoardLogger(
        save_dir=str(output_dir),
        name=experiment_name,
        default_hp_metric=False,  # Enables a placeholder metric with key `hp_metric` when `log_hyperparams` is called without a metric (otherwise calls to log_hyperparams without a metric are ignored).
        log_graph=True,
        version=".",
    )

    train_dataloader_size = len(datamodule.train_dataloader())
    bar_refresh_rate = min(int(train_dataloader_size / config.bar_update), 200)

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
                "val_acc_{val/f1_epoch:.4f}",
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
            FinetuningCallback(
                finetune_head_epochs=config.finetune_head_epochs,
                train_bn=config.finetune_train_bn,
            )
        )

    callbacks.append(ModelSummary(max_depth=1))

    auto_lr_find = config.scheduler == SupportedScheduler.AUTO_LR

    # ================= TRAINER ========================
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        pl_args,
        logger=[tensorboard_logger, csv_logger],
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
