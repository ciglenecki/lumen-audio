import argparse
from collections.abc import Iterator  # Python >=3.9
from typing import Callable, Generator

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

from src.config.argparse_with_config import ArgParseWithConfig
from src.config.config_defaults import ConfigDefault
from src.data.datamodule import OurDataModule
from src.features.audio_transform import AudioTransformBase, get_audio_transform
from src.features.chunking import collate_fn_feature, get_collate_fn
from src.model.model import SupportedModels, get_model, model_constructor_map
from src.model.model_base import ModelBase, StepResult, SupportedModels
from src.train.metrics import get_metrics
from src.utils.utils_exceptions import InvalidArgument, UnsupportedModel


def get_model(
    config: ConfigDefault, args, device: torch.DeviceObjType
) -> tuple[SupportedModels, ConfigDefault, AudioTransformBase]:
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

    return model, model_config, audio_transform


def get_datamodule(
    config: ConfigDefault,
    audio_transform: AudioTransformBase,
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
        collate_fn=collate_fn_feature,
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


def testing_generator(
    device: torch.device,
    model: ModelBase,
    data_loader: DataLoader,
) -> Iterator[StepResult]:
    for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch = [e.to(device) for e in batch]
        with torch.no_grad():
            out: StepResult = model._step(
                batch,
                batch_idx,
                type="test",
                log_metric_dict=False,
                only_return_loss=False,
                return_as_object=True,
            )
        yield out


def testing_loop(
    device: torch.device,
    model: ModelBase,
    datamodule: OurDataModule,
    data_loader: DataLoader,
):
    losses, y_preds, y_preds_file, y_trues, y_trues_file, filenames = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for out in testing_generator(device=device, model=model, data_loader=data_loader):
        losses.extend(out.losses.detach().cpu().numpy())
        y_preds.extend(out.y_pred.detach().cpu().numpy())
        y_preds_file.extend(out.y_pred_file.detach().cpu().numpy())
        y_trues.extend(out.y_true.detach().cpu().numpy())
        y_trues_file.extend(out.y_true_file.detach().cpu().numpy())
        for file_index in out.item_indices_unique.detach().cpu().numpy():
            audio_path, _ = datamodule.get_item_from_internal_structure(
                file_index, split="test"
            )
            filenames.append(audio_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    y_preds = torch.stack(y_preds)
    y_preds_file = torch.stack(y_preds_file)
    y_trues = torch.stack(y_trues)
    y_trues_file = torch.stack(y_trues_file)
    metric_dict = get_metrics(
        y_pred=y_preds,
        y_true=y_trues,
        num_labels=config.num_labels,
        return_per_instrument=True,
    )
    datamodule.get_item_from_internal_structure()
    return metric_dict


def validate_test_args(config: ConfigDefault):
    config.required_ckpt()

    # Automatically extract model type if it was not explicitly provided.
    if config.model is None:
        for e in list(SupportedModels):
            if e.value in str(config.ckpt.stem):
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


def main(args, config: ConfigDefault):
    validate_test_args(config)
    device = torch.device(args.device)
    model, model_config, audio_transform = get_model(config, args, device)
    datamodule, data_loader = get_datamodule(config, audio_transform, model_config)
    testing_loop(device, model, datamodule, data_loader)


if __name__ == "__main__":
    parser = ArgParseWithConfig()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to be used eg. cuda:0.",
    )
    args, config, _ = parser.parse_args()
    main(args, config)
