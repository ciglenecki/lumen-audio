import io
import json
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch
from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset_audio_pure import PureAudioDataset
from src.enums.enums import SupportedDatasetDirType
from src.features.chunking import collate_fn_feature
from src.inference.inference_utils import (
    StepResult,
    aggregate_inference_loops,
    aggregate_step_dicts,
    inference_loop,
    json_from_step_result,
)
from src.model.model_base import ModelBase
from src.server.server_store import server_store
from src.train.metrics import get_metrics_npy
from src.utils.utils_functions import dict_npy_to_list, dict_torch_to_npy


def get_test_json_dict(step_result: StepResult) -> dict:
    metrics = get_metrics_npy(step_result.y_pred, step_result.y_true)
    metrics["predictions"] = json_from_step_result(step_result)
    json_dict = dict_npy_to_list(metrics)
    return json_dict


def test_directory_stream() -> str:
    for result_dict in inference_loop(
        device=server_store.device,
        model=server_store.model,
        datamodule=server_store.datamodule,
        data_loader=server_store.data_loader,
        step_type="test",
    ):
        step_result = StepResult(result_dict)
        json_dict = get_test_json_dict(step_result)
        json_encoded = json.dumps(jsonable_encoder(json_dict)).encode("utf-8") + b"\n"
        yield json_encoded


def test_directory() -> dict:
    step_result = aggregate_inference_loops(
        device=server_store.device,
        model=server_store.model,
        datamodule=server_store.datamodule,
        data_loader=server_store.data_loader,
        step_type="test",
    )
    json_dict = get_test_json_dict(step_result)
    return json_dict


def predict_directory_stream() -> str:
    for result_dict in inference_loop(
        device=server_store.device,
        model=server_store.model,
        datamodule=server_store.datamodule,
        data_loader=server_store.data_loader,
        step_type="pred",
    ):
        step_result = StepResult(result_dict)
        json_dict = json_from_step_result(step_result)
        json_encoded = json.dumps(jsonable_encoder(json_dict)).encode("utf-8") + b"\n"
        yield json_encoded


def predict_directory() -> dict:
    step_result = aggregate_inference_loops(
        device=server_store.device,
        model=server_store.model,
        datamodule=server_store.datamodule,
        data_loader=server_store.data_loader,
        step_type="pred",
    )
    json_dict = json_from_step_result(step_result)
    return json_dict


def predict_files() -> dict:
    device = server_store.device
    model: ModelBase = server_store.model
    data_loader = server_store.data_loader
    step_dicts = []

    for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch = [e.to(device) for e in batch]
        with torch.no_grad():
            step_dict = model._step(
                batch,
                batch_idx,
                type="pred",
                log_metric_dict=False,
                only_return_loss=False,
            )
            step_dict = dict_torch_to_npy(step_dict)

            step_dict["filenames"] = []
            for file_index in step_dict["item_indices_unique"]:
                audio_path, _ = data_loader.dataset.dataset_list[file_index]
                step_dict["filenames"].append(audio_path)

            step_dicts.append(step_dict)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    step_result = aggregate_step_dicts(step_dicts)
    json_dict = json_from_step_result(step_result)
    return json_dict


def set_server_store_model(model_path: str):
    if server_store.config.ckpt != model_path:
        server_store.config.ckpt = model_path
        server_store.config.set_model_enum_from_ckpt()
        server_store.set_model()


def set_inference_datamodule(
    dataset_pairs: list[tuple[SupportedDatasetDirType, Path]],
    type=Literal["test", "pred"],
):
    if dataset_pairs != server_store.config.test_paths:
        server_store.config.test_paths = dataset_pairs
        server_store.set_dataset(type=type)


def set_io_dataloader(audio_files: list[UploadFile]):
    audio_sr_names: dict[str, tuple[np.ndarray, int]] = {}
    for audio_file in audio_files:
        audio, sr = sf.read(io.BytesIO(audio_file.file.read()))
        audio = audio.transpose()
        filename = Path(audio_file.filename)
        audio_sr_names[filename] = audio, sr

    dataset = PureAudioDataset(
        audio_sr_names,
        audio_transform=server_store.audio_transform,
        sampling_rate=server_store.model_config.sampling_rate,
        normalize_audio=server_store.model_config.normalize_audio,
        num_lables=server_store.model_config.num_labels,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=server_store.config.batch_size,
        num_workers=server_store.config.num_workers,
        drop_last=False,
        collate_fn=collate_fn_feature,
        pin_memory=True,
    )

    server_store.data_loader = data_loader
