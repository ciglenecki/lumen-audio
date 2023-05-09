from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

import src.server.controllers as controllers
from src.enums.enums import SupportedDatasetDirType
from src.server import interface
from src.server.description import (
    GET_DATASET_DESC,
    GET_MODELS_DESC,
    MODELS_INFERENCE_TAG,
    PREDICT_DIR_DESC,
    PREDICT_DIR_STREAM_DESC,
    PREDICT_FILES_DESC,
    RESOURCES_TAG,
    TEST_DIR_DESC,
    TEST_DIR_STREAM_DESC,
)
from src.server.interface import (
    DatasetTypedPath,
    InstrumentPrediction,
    InstrumentPredictions,
    SupportedDatasetDirTypeTrain,
)
from src.server.middleware import (
    dep_audio_file,
    dep_dataset_paths,
    dep_dataset_paths_with_type,
    dep_model_ckpt_path,
)
from src.server.server_store import server_store

dataset_router = APIRouter(prefix="/dataset-type")

default_model_path = server_store.config.path_models / "model.ckpt"


@dataset_router.get(
    "s",
    tags=[RESOURCES_TAG],
    response_model=List[str],
    description=GET_DATASET_DESC,
)
async def get_supported_datasets():
    return [e.value for e in SupportedDatasetDirTypeTrain]


model_router = APIRouter(prefix="/model")


@model_router.get(
    "s",
    tags=[RESOURCES_TAG],
    response_model=List[Path],
    description=GET_MODELS_DESC,
)
async def get_models():
    return server_store.get_available_models()


@model_router.post(
    "/test-directory-stream",
    tags=[MODELS_INFERENCE_TAG],
    response_model=interface.PredictionsWithMetrics,
    description=TEST_DIR_STREAM_DESC,
)
async def test_directory_stream(
    dataset_paths_with_type: list[DatasetTypedPath] = Depends(
        dep_dataset_paths_with_type
    ),
    model_ckpt_path: Path = Depends(dep_model_ckpt_path),
):
    dataset_pairs = [(d.dataset_type, d.dataset_path) for d in dataset_paths_with_type]
    controllers.set_server_store_model(model_ckpt_path)
    controllers.set_inference_datamodule(dataset_pairs, type="test")

    return StreamingResponse(
        controllers.test_directory_stream(), media_type="application/json"
    )


@model_router.post(
    "/test-directory",
    tags=[MODELS_INFERENCE_TAG],
    response_model=interface.PredictionsWithMetrics,
    description=TEST_DIR_DESC,
)
async def test_directory(
    dataset_paths_with_type: list[DatasetTypedPath] = Depends(
        dep_dataset_paths_with_type
    ),
    model_ckpt_path: Path = Depends(dep_model_ckpt_path),
):
    dataset_pairs = [(d.dataset_type, d.dataset_path) for d in dataset_paths_with_type]
    controllers.set_server_store_model(model_ckpt_path)
    controllers.set_inference_datamodule(dataset_pairs, type="test")

    return JSONResponse(controllers.test_directory(), media_type="application/json")


@model_router.post(
    "/predict-directory-stream",
    tags=[MODELS_INFERENCE_TAG],
    response_model=InstrumentPredictions,
    description=PREDICT_DIR_STREAM_DESC,
)
async def predict_directory_stream(
    dataset_paths: list[Path] = Depends(dep_dataset_paths),
    model_ckpt_path: Path = Depends(dep_model_ckpt_path),
):
    dataset_paths_with_type = [
        (SupportedDatasetDirType.INFERENCE, path) for path in dataset_paths
    ]
    controllers.set_server_store_model(model_ckpt_path)
    controllers.set_inference_datamodule(dataset_paths_with_type, type="predict")

    return StreamingResponse(
        controllers.predict_directory_stream(), media_type="application/json"
    )


@model_router.post(
    "/predict-directory",
    tags=[MODELS_INFERENCE_TAG],
    response_model=InstrumentPredictions,
    description=PREDICT_DIR_DESC,
)
async def predict_directory(
    dataset_paths: list[Path] = Depends(dep_dataset_paths),
    model_ckpt_path: Path = Depends(dep_model_ckpt_path),
):
    dataset_paths_with_type = [
        (SupportedDatasetDirType.INFERENCE, path) for path in dataset_paths
    ]
    controllers.set_server_store_model(model_ckpt_path)
    controllers.set_inference_datamodule(dataset_paths_with_type, type="predict")

    return JSONResponse(controllers.predict_directory(), media_type="application/json")


@model_router.post(
    "/predict-files",
    tags=[MODELS_INFERENCE_TAG],
    response_model=List[InstrumentPrediction],
    description=PREDICT_FILES_DESC,
)
async def predict_files(
    model_ckpt_path: Path = Depends(dep_model_ckpt_path),
    audio_files: list[UploadFile] = Depends(dep_audio_file),
):
    controllers.set_server_store_model(model_ckpt_path)
    controllers.set_io_dataloader(audio_files)

    return JSONResponse(controllers.predict_files(), media_type="application/json")
