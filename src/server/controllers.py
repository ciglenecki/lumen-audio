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
        entries = {}

        y_pred_file = out.y_pred_file.detach().cpu().numpy()
        item_indices_unique = out.item_indices_unique.detach().cpu().numpy()
        for file_index, y_pred_file in zip(item_indices_unique, y_pred_file):
            audio_path, _ = server_store.datamodule.get_item_from_internal_structure(
                file_index, split="test"
            )
            dict_pred = multihot_to_dict(y_pred_file)
            entries[str(audio_path.stem)] = dict_pred

        json_encoded = json.dumps(jsonable_encoder(entries)).encode("utf-8") + b"\n"
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
