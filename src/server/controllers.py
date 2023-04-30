from fastapi.encoders import jsonable_encoder

from src.server.server_store import server_store
from src.train.run_test import testing_generator
from src.utils.utils_dataset import multihot_to_dict


def predict_directory() -> str:
    for out in testing_generator(
        server_store.device,
        server_store.model,
        server_store.data_loader,
    ):
        data = []
        for y_pred in out.y_pred.detach().cpu().numpy():
            data.append(multihot_to_dict(y_pred))
        json_encoded = jsonable_encoder(data)
        yield json_encoded


def set_server_store_model(model_path: str):
    server_store.config.ckpt = model_path


def set_server_store_directory(model_path: str):
    pass
