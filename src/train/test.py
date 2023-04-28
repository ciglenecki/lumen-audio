import argparse
from typing import Callable

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config_defaults import ConfigDefault
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
from src.model.model_base import ModelBase, SupportedModels
from src.train.callbacks import (
    FinetuningCallback,
    GeneralMetricsEpochLogger,
    OverrideEpochMetricCallback,
    TensorBoardHparamFixer,
)
from src.train.metrics import get_metrics
from src.utils.utils_dataset import calc_instrument_weight
from src.utils.utils_exceptions import InvalidArgument, UnsupportedModel


def get_model(
    config: ConfigDefault, args
) -> tuple[SupportedModels, ConfigDefault, AudioTransformBase]:
    assert args.device
    config.required_ckpt()

    # Automatically extract model type if it was not explicitly provided.
    if config.model is None:
        for e in list(SupportedModels):
            e.value in str(config.ckpt.stem)
            config.model = e
            break

    config.required_test_paths()
    config.required_model()
    if config.model is None:
        raise InvalidArgument(
            f"--model is required {[e.name for e in SupportedModels]}"
        )
    if config.model not in model_constructor_map:
        raise UnsupportedModel(
            f"Model {config.model} is not in the model_constructor_map. Add the model enum to the model_constructor_map."
        )

    device: str = args.device

    model_constructor: pl.LightningModule = model_constructor_map[config.model]
    model = model_constructor.load_from_checkpoint(config.ckpt, strict=True)
    model.eval()
    model = model.to(device)
    model_config = model.config

    audio_transform: AudioTransformBase = get_audio_transform(
        model_config,
        spectrogram_augmentation=None,
        waveform_augmentation=None,
    )

    collate_fn = get_collate_fn(model_config)
    return model, model_config, audio_transform, collate_fn


def get_datamodule(
    config: ConfigDefault,
    audio_transform: AudioTransformBase,
    collate_fn: Callable,
    model_config: ConfigDefault,
):
    datamodule = OurDataModule(
        train_paths=None,
        val_paths=None,
        test_paths=config.test_paths,
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
    datamodule.setup_for_inference()
    data_loader = datamodule.test_dataloader()
    return datamodule, data_loader


def test_loop(
    device: str, model: ModelBase, datamodule: OurDataModule, data_loader: DataLoader
):
    losses, y_preds, y_outs = [], [], []
    for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch = batch.to(device)
        images, y_true, file_indices, item_index = batch
        loss, y_pred, y_out = model.predict_step(batch, batch_idx)
        losses.append(loss.detach())
        y_preds.append(y_pred.detach())
        y_outs.append(y_out.detach())

    metric_dict = get_metrics(
        y_pred=y_pred,
        y_true=y_true,
        num_labels=config.num_labels,
        return_per_instrument=True,
    )
    datamodule.get_item_from_internal_structure()
    return metric_dict


def main(args, config: ConfigDefault):
    model, model_config, audio_transform, collate_fn = get_model(config, args)
    datamodule, data_loader = get_datamodule(
        config, audio_transform, collate_fn, model_config
    )
    test_loop(args.device, model, datamodule, data_loader)


if __name__ == "__main__":
    parser = ArgParseWithConfig()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="The device to be used eg. cuda:0.",
    )
    args, config, _ = parser.parse_args()
    main(args, config)
