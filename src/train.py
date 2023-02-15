import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)

from src.audio_transform import AudioTransformBase, get_audio_transform
from src.callbacks import (
    LogMetricsAsHyperparams,
    OnTrainEpochStartLogCallback,
    OverrideEpochMetricCallback,
)
from src.datamodule import IRMASDataModule
from src.model import get_model
from src.train_args import parse_args_train
from src.utils_functions import (
    add_prefix_to_keys,
    get_timestamp,
    random_codeword,
    stdout_to_file,
    to_yaml,
)
from src.utils_train import MetricMode, OptimizeMetric, SchedulerType

if __name__ == "__main__":
    args, pl_args = parse_args_train()
    output_dir = args.output_dir
    num_labels = args.num_labels
    batch_size = args.batch_size
    sampling_rate = args.sampling_rate
    metric_mode_str = MetricMode(args.metric_mode).value
    optimizer_metric_str = OptimizeMetric(args.metric).value

    timestamp = get_timestamp()
    experiment_codeword = random_codeword()
    experiment_name = f"{timestamp}_{experiment_codeword}_{args.model}"

    os.makedirs(output_dir, exist_ok=True)
    filename_report = Path(output_dir, experiment_name + ".txt")

    stdout_to_file(filename_report)
    print(str(filename_report))
    print("Config:", to_yaml(vars(args)), sep="\n")
    print("Config PyTorch Lightning:", to_yaml(vars(pl_args)), sep="\n")

    audio_transform: AudioTransformBase = get_audio_transform(args.audio_transform)

    datamodule = IRMASDataModule(
        batch_size=batch_size,
        num_workers=args.num_workers,
        dataset_fraction=args.dataset_fraction,
        drop_last_sample=args.drop_last,
        train_audio_transform=audio_transform,
        val_audio_transform=audio_transform,
    )

    train_dataloader_size = len(datamodule.train_dataloader())

    log_dictionary = {
        **add_prefix_to_keys(vars(args), "user_args/"),
        **add_prefix_to_keys(vars(pl_args), "lightning_args/"),
        "train_size": len(datamodule.train_dataloader().dataset),  # type: ignore
        "val_size": len(datamodule.val_dataloader().dataset),  # type: ignore
        "test_size": len(datamodule.test_dataloader().dataset),  # type: ignore
    }

    callback_early_stopping = EarlyStopping(
        monitor=optimizer_metric_str,
        mode=metric_mode_str,
        patience=args.patience,
        check_on_train_epoch_end=args.check_on_train_epoch_end,
        verbose=True,
    )

    callback_checkpoint = ModelCheckpoint(
        monitor=optimizer_metric_str,
        mode=metric_mode_str,
        filename="_".join(
            [
                experiment_name,
                "val_acc_{val/acc_epoch:.4f}",
                "val_loss_{val/loss_epoch:.4f}",
            ]
        ),
        auto_insert_metric_name=False,
        save_on_train_epoch_end=args.save_on_train_epoch_end,
        verbose=True,
    )

    bar_refresh_rate = int(train_dataloader_size / args.bar_update)

    callbacks = [
        callback_checkpoint,
        callback_early_stopping,
        TQDMProgressBar(refresh_rate=bar_refresh_rate),
        ModelSummary(max_depth=4),
        LogMetricsAsHyperparams(),
        OverrideEpochMetricCallback(),
        OnTrainEpochStartLogCallback(),
        LearningRateMonitor(log_momentum=True),
    ]

    model = get_model(args, pl_args)

    tensorboard_logger = pl_loggers.TensorBoardLogger(
        save_dir=str(output_dir),
        name=experiment_name,
        default_hp_metric=False,  # default_hp_metric should be turned off unless you log hyperparameters (logger.log_hyperparams(dict)) before the module starts with training
        log_graph=True,
        version=".",
    )

    tensorboard_logger.log_hyperparams(log_dictionary)

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        pl_args,
        logger=[tensorboard_logger],
        default_root_dir=output_dir,
        callbacks=callbacks,
        auto_lr_find=args.scheduler == SchedulerType.AUTO_LR,
    )

    if args.scheduler == SchedulerType.AUTO_LR.value:
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

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt)
    trainer.test(model, datamodule)
