from typing import List

from description import get_models_desc, predict_images_desc
from fastapi import APIRouter, UploadFile

from src.config.config_defaults import InstrumentEnums
from src.enums.enums import SupportedModels
from src.server.interface import MultilabelPrediction

router = APIRouter(prefix="/model")


@router.get(
    "s",
    tags=["available models"],
    response_model=List[str],
    description=get_models_desc,
)
async def get_models():
    return [e.name for e in SupportedModels]


@router.post(
    "/predict-sound-file",
    tags=["predict"],
    response_model=List[MultilabelPrediction],
    description=predict_images_desc,
)
async def predict_sound(model_name: str, images: List[UploadFile]):
    return [{e: 0 for e in InstrumentEnums}]


@router.post(
    "/predict-sound-file",
    tags=["predict"],
    response_model=List[MultilabelPrediction],
    description=predict_images_desc,
)
async def predict_sound(model_name: str, images: List[UploadFile]):
    return [{e: 0 for e in InstrumentEnums}]


# @router.post(
#     "/{model_name}/predict-directory",
#     tags=["predict"],
#     response_model=List[models.PredictDirectoryReponse],
#     description=predict_directory_desc,
# )
# def predict_dataset(model_name: str, body: models.PostPredictDatasetRequest):
#     return controller.predict_dataset(model_name, body)
