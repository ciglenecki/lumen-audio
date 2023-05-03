from pathlib import Path
from typing import List

from description import get_models_desc, predict_images_desc
from fastapi import APIRouter, Depends, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

import src.server.controllers as controllers
from src.enums.enums import SupportedDatasetDirType
from src.server import interface
from src.server.interface import JsonPrediction, JsonPredictions, TypeDatasetDict
from src.server.middleware import dep_audio_file, dep_model_checkpoint
from src.server.server_store import server_store

router = APIRouter(prefix="/model")


@router.get(
    "s",
    tags=["available models"],
    response_model=List[Path],
    description=get_models_desc,
)
async def get_models():
    return server_store.get_available_models()


@router.get(
    "/dataset-types",
    tags=["available models"],
    response_model=List[str],
    description=get_models_desc,
)
async def get_supported_datasets():
    return [e.value for e in SupportedDatasetDirType]


@router.post(
    "/test-directory-stream",
    tags=["predict"],
    response_model=interface.PredictionsWithMetrics,
    description=predict_images_desc,
)
async def test_directory_stream(
    model_checkpoint: Path,
    dataset_dirs: list[TypeDatasetDict],
):
    controllers.set_server_store_model(model_checkpoint)
    controllers.set_inference_datamodule(dataset_dirs, type="test")

    return StreamingResponse(
        controllers.test_directory_stream(), media_type="application/json"
    )


@router.post(
    "/test-directory",
    tags=["predict"],
    response_model=interface.PredictionsWithMetrics,
    description=predict_images_desc,
)
async def test_directory(
    model_checkpoint: Path,
    dataset_dirs: list[TypeDatasetDict],
):
    controllers.set_server_store_model(model_checkpoint)
    controllers.set_inference_datamodule(dataset_dirs, type="test")

    return JSONResponse(controllers.test_directory(), media_type="application/json")


@router.post(
    "/predict-directory-stream",
    tags=["predict"],
    response_model=JsonPredictions,
    description=predict_images_desc,
)
async def predict_directory_stream(
    model_checkpoint: Path,
    dataset_dirs: list[TypeDatasetDict],
):
    controllers.set_server_store_model(model_checkpoint)
    controllers.set_inference_datamodule(dataset_dirs, type="predict")

    return StreamingResponse(
        controllers.predict_directory_stream(), media_type="application/json"
    )


@router.post(
    "/predict-directory",
    tags=["predict"],
    response_model=JsonPredictions,
    description=predict_images_desc,
)
async def predict_directory(
    model_checkpoint: Path,
    dataset_dirs: list[TypeDatasetDict],
):
    controllers.set_server_store_model(model_checkpoint)
    controllers.set_inference_datamodule(dataset_dirs, type="predict")

    return JSONResponse(controllers.predict_directory(), media_type="application/json")


@router.post(
    "/predict-files",
    tags=["predict"],
    response_model=List[JsonPrediction],
    description=predict_images_desc,
)
async def predict_files(
    audio_files: list[UploadFile] = Depends(dep_audio_file),
    model_checkpoint: Path = Depends(dep_model_checkpoint),
):
    controllers.set_server_store_model(model_checkpoint)
    controllers.set_io_dataloader(audio_files)

    return JSONResponse(controllers.predict_files(), media_type="application/json")
