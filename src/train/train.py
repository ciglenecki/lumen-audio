"""Some configurations."""
import os
from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)

from src.config.config_train import config, pl_args
from src.data.datamodule import IRMASDataModule
from src.enums.enums import MetricMode, ModelInputDataType, OptimizeMetric
from src.features.audio_transform import AudioTransformBase, get_audio_transform
from src.features.augmentations import SupportedAugmentations, get_augmentations
from src.model.loss_function import get_loss_fn
from src.model.model import get_data_input_type, get_model
from src.model.optimizers import SupportedScheduler
from src.train.callbacks import (
    FinetuningCallback,
    GeneralMetricsEpochLogger,
    OverrideEpochMetricCallback,
    TensorBoardHparamFixer,
)
from src.utils.utils_dataset import chunk_collate_audio, collate_fn_spectrogram
from src.utils.utils_functions import (
    add_prefix_to_keys,
    get_timestamp,
    random_codeword,
    stdout_to_file,
    to_yaml,
)
from src.utils.utils_train import print_modules

if __name__ == "__main__":
    timestamp = get_timestamp()
    experiment_codeword = random_codeword()
    experiment_name = f"{timestamp}_{experiment_codeword}_{config.model.value}"

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    experiment_dir = Path(output_dir, experiment_name)
    experiment_dir.mkdir(exist_ok=True)

    num_labels = config.num_labels
    batch_size = config.batch_size
    sampling_rate = config.sampling_rate
    metric_mode_str = MetricMode(config.metric_mode).value
    optimizer_metric_str = OptimizeMetric(config.metric).value
    normalize_audio = config.normalize_audio
    aug_kwargs = config.aug_kwargs
    image_dim = config.image_dim
    use_weighted_train_sampler = config.use_weighted_train_sampler

    filename_config = Path(experiment_dir, "config.yaml")
    with open(filename_config, "w") as outfile:
        yaml.dump(config, outfile)

    filename_report = Path(output_dir, experiment_name, "log.txt")
    stdout_to_file(filename_report)
    print(str(filename_report))
    print("Config:", to_yaml(vars(config)), sep="\n")
    print("Config PyTorch Lightning:", to_yaml(vars(pl_args)), sep="\n")

    data_input_type = get_data_input_type(model_enum=config.model)
    if data_input_type == ModelInputDataType.IMAGE:
        collate_fn = collate_fn_spectrogram
    elif data_input_type == ModelInputDataType.WAVEFORM:
        collate_fn = partial(
            chunk_collate_audio,
            max_audio_width=config.max_audio_seconds * config.sampling_rate,
        )
    else:
        raise Exception(f"Unsupported data input type {data_input_type}")

    (
        train_spectrogram_augmentation,
        train_waveform_augmentation,
        val_spectrogram_augmentation,
        val_waveform_augmentation,
    ) = get_augmentations(config)

    train_audio_transform: AudioTransformBase = get_audio_transform(
        audio_transform_enum=config.audio_transform,
        sampling_rate=sampling_rate,
        spectrogram_augmentation=train_spectrogram_augmentation,
        waveform_augmentation=train_waveform_augmentation,
        image_dim=config.image_dim,
    )
    val_audio_transform: AudioTransformBase = get_audio_transform(
        audio_transform_enum=config.audio_transform,
        sampling_rate=sampling_rate,
        spectrogram_augmentation=val_spectrogram_augmentation,
        waveform_augmentation=val_waveform_augmentation,
        image_dim=config.image_dim,
    )

    datamodule = IRMASDataModule(
        batch_size=batch_size,
        num_workers=config.num_workers,
        dataset_fraction=config.dataset_fraction,
        drop_last_sample=config.drop_last,
        train_audio_transform=train_audio_transform,
        val_audio_transform=val_audio_transform,
        collate_fn=collate_fn,
        normalize_audio=normalize_audio,
        train_only_dataset=config.train_only_dataset,
        concat_two_samples=SupportedAugmentations.CONCAT_TWO in config.augmentations,
        use_weighted_train_sampler=use_weighted_train_sampler,
    )

    loss_function = get_loss_fn(
        config.loss_function,
        datamodule=datamodule,
        **config.loss_function_kwargs,
    )
    model = get_model(config, pl_args, loss_function=loss_function)
    print_modules(model)

    train_dataloader_size = len(datamodule.train_dataloader())

    log_dictionary = {
        **add_prefix_to_keys(vars(config), "user_args/"),
        **add_prefix_to_keys(vars(pl_args), "lightning_args/"),
        "train_size": len(datamodule.train_dataloader().dataset),
        "val_size": len(datamodule.val_dataloader().dataset),
        "test_size": len(datamodule.test_dataloader().dataset),
    }

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

    bar_refresh_rate = int(train_dataloader_size / config.bar_update)

    tensorboard_logger = pl_loggers.TensorBoardLogger(
        save_dir=str(output_dir),
        name=experiment_name,
        default_hp_metric=False,  # Enables a placeholder metric with key `hp_metric` when `log_hyperparams` is called without a metric (otherwise calls to log_hyperparams without a metric are ignored).
        log_graph=True,
        version=".",
    )

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
