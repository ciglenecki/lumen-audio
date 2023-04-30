import json
from pathlib import Path

from fastapi.encoders import jsonable_encoder

from src.server.interface import DatasetDirDict
from src.server.server_store import server_store
from src.train.run_test import testing_generator
from src.utils.utils_dataset import multihot_to_dict


def test_directory() -> str:
    for out in testing_generator(
        server_store.device,
        server_store.model,
        server_store.data_loader,
    ):
        data = []
        for y_pred in out.y_pred.detach().cpu().numpy():
            data.append(multihot_to_dict(y_pred))
        json_encoded = json.dumps(jsonable_encoder(data)).encode("utf-8") + b"\n"
        yield json_encoded


def set_server_store_model(model_path: str):
    server_store.config.ckpt = model_path
    server_store.config.set_model_enum_from_ckpt()
    server_store.set_model()


def set_server_store_directory(
    dataset_dirs: list[DatasetDirDict],
):
    dataset_pairs = [(d.dataset_dir_type, d.dataset_dir) for d in dataset_dirs]
    server_store.config.test_paths = dataset_pairs
    server_store.set_dataset()
