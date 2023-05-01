from collections.abc import Iterator  # Python >=3.9
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.argparse_with_config import ArgParseWithConfig
from src.config.config_defaults import ConfigDefault
from src.data.datamodule import OurDataModule
from src.features.audio_transform import AudioTransformBase, get_audio_transform
from src.features.chunking import collate_fn_feature
from src.model.model import SupportedModels, get_model, model_constructor_map
from src.model.model_base import ModelBase
from src.train.metrics import get_metrics
from src.utils.utils_exceptions import InvalidArgument, UnsupportedModel


def dict_torch_to_npy(d: dict):
    return {
        k: v.detach().cpu().numpy()
        for k, v in d.items()
        if isinstance(v, torch.torch.Tensor)
    }


class StepResult:
    def __init__(self, step_dict: dict):
        step_dict = {
            k: v.detach().cpu().numpy()
            for k, v in step_dict.items()
            if isinstance(v, torch.torch.Tensor)
        }
        self.loss: np.ndarray | None = step_dict.get("loss", None)
        self.losses: np.ndarray | None = step_dict.get("losses", None)
        self.y_pred: np.ndarray | None = step_dict.get("y_pred", None)
        self.y_pred_prob: np.ndarray | None = step_dict.get("y_pred_prob", None)
        self.y_true: np.ndarray | None = step_dict.get("y_true", None)
        self.file_indices: np.ndarray | None = step_dict.get("file_indices", None)
        self.item_indices: np.ndarray | None = step_dict.get("item_indices", None)
        self.item_indices_unique: np.ndarray | None = step_dict.get(
            "item_indices_unique", None
        )
        self.y_true_file: np.ndarray | None = step_dict.get("y_true_file", None)
        self.y_pred_file: np.ndarray | None = step_dict.get("y_pred_file", None)
        self.y_pred_prob_file: np.ndarray | None = step_dict.get(
            "y_pred_prob_file", None
        )
        self.losses_file: np.ndarray | None = step_dict.get("losses_file", None)
        self.filenames: list[Path] | None = step_dict.get("filenames", None)


def get_model_config_transform(
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
        sum_n_samples=None,
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
    datamodule: OurDataModule,
) -> Iterator[dict]:
    for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch = [e.to(device) for e in batch]
        with torch.no_grad():
            step_dict = model._step(
                batch,
                batch_idx,
                type="test",
                log_metric_dict=False,
                only_return_loss=False,
            )
            step_dict = dict_torch_to_npy(step_dict)

            step_dict["filenames"] = []
            for file_index in step_dict["item_indices_unique"]:
                audio_path, _ = datamodule.get_item_from_internal_structure(
                    file_index, split="test"
                )
                step_dict["filenames"].append(audio_path)
        yield step_dict


def testing_loop(
    device: torch.device,
    model: ModelBase,
    datamodule: OurDataModule,
    data_loader: DataLoader,
):
    result_dict = {}
    for step_dict in testing_generator(
        device=device, model=model, data_loader=data_loader, datamodule=datamodule
    ):
        for k, v in step_dict.items():
            if k not in result_dict:
                result_dict[k] = []
            result_dict[k].append(v)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = StepResult(result_dict)

    y_pred = torch.stack(result.y_pred)
    y_pred_file = torch.stack(result.y_pred_file)
    y_true = torch.stack(result.y_true)
    y_true_file = torch.stack(result.y_true_file)

    metric_dict = get_metrics(
        y_pred=y_pred,
        y_true=y_true,
        num_labels=config.num_labels,
        return_per_instrument=True,
    )
    metric_dict_file = get_metrics(
        y_pred=y_pred_file,
        y_true=y_true_file,
        num_labels=config.num_labels,
        return_per_instrument=True,
    )
    return metric_dict, metric_dict_file


def validate_test_args(config: ConfigDefault):
    config.required_ckpt()

    # Automatically extract model type if it was not explicitly provided.
    if config.model is None:
        config.set_model_enum_from_ckpt()

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
    model, model_config, audio_transform = get_model_config_transform(
        config, args, device
    )
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
