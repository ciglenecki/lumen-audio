from typing import List

from description import get_models_desc, predict_images_desc
from fastapi import APIRouter, UploadFile
from fastapi.responses import StreamingResponse

import src.server.controllers as controllers
from src.server.interface import MultilabelPrediction
from src.server.server_store import server_store

router = APIRouter(prefix="/model")


@router.get(
    "s",
    tags=["available models"],
    response_model=List[str],
    description=get_models_desc,
)
async def get_models():
    return server_store.get_available_models()


@router.post(
    "/predict-directory",
    tags=["predict"],
    response_model=List[MultilabelPrediction],
    description=predict_images_desc,
)
async def predict_sound(model_path: str, directory: List[UploadFile]):
    controllers.set_server_store_model(model_path)
    controllers.set_server_store_directory(directory)

    return StreamingResponse(
        controllers.predict_directory(), media_type="application/json"
    )


# @router.post(
#     "/{model_name}/predict-directory",
#     tags=["predict"],
#     response_model=List[models.PredictDirectoryReponse],
#     description=predict_directory_desc,
# )
# def predict_dataset(model_name: str, body: models.PostPredictDatasetRequest):
#     return controller.predict_dataset(model_name, body)
