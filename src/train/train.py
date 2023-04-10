import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)

from src.config.config import config, pl_args
from src.data.datamodule import IRMASDataModule
from src.features.audio_transform import AudioTransformBase, get_audio_transform
from src.features.augmentations import SupportedAugmentations, get_augmentations
from src.model.model import ModelInputDataType, get_data_input_type, get_model
from src.model.optimizers import SchedulerType
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
from src.utils.utils_train import MetricMode, OptimizeMetric, print_modules

if __name__ == "__main__":
    output_dir = config.output_dir
    num_labels = config.num_labels
    batch_size = config.batch_size
    sampling_rate = config.sampling_rate
    unfreeze_at_epoch: int = config.unfreeze_at_epoch
    metric_mode_str = MetricMode(config.metric_mode).value
    optimizer_metric_str = OptimizeMetric(config.metric).value
    normalize_audio = config.normalize_audio
    aug_kwargs = config.aug_kwargs
    dim = config.dim
    use_weighted_train_sampler = config.use_weighted_train_sampler

    timestamp = get_timestamp()
    experiment_codeword = random_codeword()
    experiment_name = f"{timestamp}_{experiment_codeword}_{config.model.value}"

    os.makedirs(output_dir, exist_ok=True)
    filename_report = Path(output_dir, experiment_name + ".txt")

    stdout_to_file(filename_report)
    print(str(filename_report))
    print("Config:", to_yaml(vars(args)), sep="\n")
    print("Config PyTorch Lightning:", to_yaml(vars(pl_args)), sep="\n")

    data_input_type = get_data_input_type(model_enum=config.model)
    if data_input_type == ModelInputDataType.IMAGE:
        collate_fn = collate_fn_spectrogram
    elif data_input_type == ModelInputDataType.WAVEFORM:
        collate_fn = chunk_collate_audio
    else:
        raise Exception(f"Unsupported data input type {data_input_type}")

    (
        train_spectrogram_augmentation,
        train_waveform_augmentation,
        val_spectrogram_augmentation,
        val_waveform_augmentation,
    ) = get_augmentations(args)

    train_audio_transform: AudioTransformBase = get_audio_transform(
        audio_transform_enum=config.audio_transform,
        sampling_rate=sampling_rate,
        spectrogram_augmentation=train_spectrogram_augmentation,
        waveform_augmentation=train_waveform_augmentation,
        dim=dim,
    )
    val_audio_transform: AudioTransformBase = get_audio_transform(
        audio_transform_enum=config.audio_transform,
        sampling_rate=sampling_rate,
        spectrogram_augmentation=val_spectrogram_augmentation,
        waveform_augmentation=val_waveform_augmentation,
        dim=dim,
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

    model = get_model(args, pl_args)
    print_modules(model)

    train_dataloader_size = len(datamodule.train_dataloader())

    log_dictionary = {
        **add_prefix_to_keys(vars(args), "user_args/"),
        **add_prefix_to_keys(vars(pl_args), "lightning_args/"),
        "train_size": len(datamodule.train_dataloader().dataset),
        "val_size": len(datamodule.val_dataloader().dataset),
        "test_size": len(datamodule.test_dataloader().dataset),
    }

    callback_early_stopping = EarlyStopping(
        monitor=optimizer_metric_str,
        mode=metric_mode_str,
        patience=config.patience,
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

    if unfreeze_at_epoch is not None:
        callbacks.append(
            FinetuningCallback(unfreeze_backbone_at_epoch=config.unfreeze_at_epoch)
        )

    callbacks.append(ModelSummary(max_depth=4))

    auto_lr_find = config.scheduler == SchedulerType.AUTO_LR
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        pl_args,
        logger=[tensorboard_logger],
        default_root_dir=output_dir,
        callbacks=callbacks,
    )

    if config.scheduler == SchedulerType.AUTO_LR.value:
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
