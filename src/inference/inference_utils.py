from collections.abc import Iterator  # Python >=3.9
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.config_defaults import ConfigDefault
from src.data.datamodule import OurDataModule
from src.features.audio_transform import AudioTransformBase, get_audio_transform
from src.features.chunking import collate_fn_feature
from src.model.model import SupportedModels, model_constructor_map
from src.model.model_base import ModelBase
from src.utils.utils_dataset import multihot_to_dict
from src.utils.utils_exceptions import InvalidArgument, UnsupportedModel
from src.utils.utils_functions import dict_torch_to_npy, flatten


def validate_inference_args(config: ConfigDefault):
    config.required_ckpt()

    # Automatically extract model type if it was not explicitly provided.
    if config.model is None:
        config.set_model_enum_from_ckpt()

    config.required_dataset_paths()
    config.required_model()
    if config.model is None:
        raise InvalidArgument(
            f"--model is required {[e.name for e in SupportedModels]}"
        )
    if config.model not in model_constructor_map:
        raise UnsupportedModel(
            f"Model {config.model} is not in the model_constructor_map. Add the model enum to the model_constructor_map."
        )


def get_inference_model_objs(
    config: ConfigDefault, args, device: torch.DeviceObjType
) -> tuple[SupportedModels, ConfigDefault, AudioTransformBase]:
    model_constructor: pl.LightningModule = model_constructor_map[config.model]
    model = model_constructor.load_from_checkpoint(
        config.ckpt, strict=False, finetune_train_bn=True
    )
    model.eval()
    model = model.to(device)
    model_config = model.config

    audio_transform: AudioTransformBase = get_audio_transform(
        model_config,
        spectrogram_augmentation=None,
        waveform_augmentation=None,
    )

    return model, model_config, audio_transform


class StepResult:
    def __init__(self, step_dict: dict):
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


def aggregate_step_dicts(step_dicts: list[dict]) -> StepResult:
    example_dict = step_dicts[0]
    result_dict = {}
    for key in example_dict.keys():
        value = []
        for step_dict in step_dicts:
            value.append(step_dict[key])
        if key in ["loss"]:  # 0D
            value = np.array(value)
        elif key in [  # 1D
            "losses_file",
            "losses",
            "file_indices",
            "item_indices",
            "item_indices_unique",
            "y_true_file",
            "y_pred_file",
            "y_pred_prob_file",
            "y_pred",
            "y_pred_prob",
            "y_true",
        ]:
            value = np.concatenate(value)
        elif key in ["filenames"]:
            value = flatten(value)

        result_dict[key] = value
    out = StepResult(result_dict)
    return out


def inference_loop(
    device: torch.device,
    model: ModelBase,
    data_loader: DataLoader,
    datamodule: OurDataModule,
    step_type: str,
) -> Iterator[dict]:
    for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch = [e.to(device) for e in batch]
        with torch.no_grad():
            step_dict = model._step(
                batch,
                batch_idx,
                type=step_type,
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


def aggregate_inference_loops(
    device: torch.device,
    model: ModelBase,
    datamodule: OurDataModule,
    data_loader: DataLoader,
    step_type: str,
) -> StepResult:
    step_dicts = []
    for step_dict in inference_loop(
        device=device,
        model=model,
        data_loader=data_loader,
        datamodule=datamodule,
        step_type=step_type,
    ):
        step_dicts.append(step_dict)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = aggregate_step_dicts(step_dicts)
    return result


def json_from_step_result(result: StepResult):
    json_dict = {}
    y_pred_file = result.y_pred_file
    filenames = result.filenames

    for filename, y_pred_file in zip(filenames, y_pred_file):
        dict_pred = multihot_to_dict(y_pred_file)
        json_dict[str(filename.stem)] = dict_pred

    return json_dict


def get_inference_datamodule(
    config: ConfigDefault,
    audio_transform: AudioTransformBase,
    model_config: ConfigDefault,
) -> OurDataModule:
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
    return datamodule
