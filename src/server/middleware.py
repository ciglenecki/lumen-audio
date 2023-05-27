from itertools import chain
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

import src.config.config_defaults as config_defaults
from src.enums.enums import SupportedDatasetDirType
from src.server.interface import DatasetTypedPath, SupportedDatasetDirTypeTrain
from src.server.server_store import server_store


def dep_dataset_path(audio_path: Path = "data/my_dataset"):
    glob_expressions = [f"*.{ext}" for ext in config_defaults.AUDIO_EXTENSIONS]

    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{audio_path} not found.",
        )
    if audio_path.is_file():
        if audio_path.suffix != ".csv":
            raise HTTPException(
                status_code=406,
                detail="Path that's not a directory has to be a .csv file",
            )
        return audio_path

    glob_generators = chain(
        *[audio_path.rglob(glob_exp) for glob_exp in glob_expressions]
    )
    if next(glob_generators, None) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Directory {audio_path} has no audio files. Supported audio extensions are {config_defaults.AUDIO_EXTENSIONS}.",
        )

    return audio_path


def dep_dataset_paths(dataset_paths: list[Path] = ["data/my_dataset"]):
    return [dep_dataset_path(dataset_path) for dataset_path in dataset_paths]


def dep_typed_paths(typed_paths: DatasetTypedPath):
    if typed_paths.dataset_path.is_file():
        if typed_paths.dataset_type != SupportedDatasetDirTypeTrain.CSV:
            raise HTTPException(
                status_code=406,
                detail="Path that's not a directory has to be a .csv file",
            )
    dep_dataset_path(typed_paths.dataset_path)

    typed_paths.dataset_type = SupportedDatasetDirType(typed_paths.dataset_type.value)
    return typed_paths


def dep_dataset_paths_with_type(dataset_paths_with_type: list[DatasetTypedPath]):
    return [(dep_typed_paths(typed_paths)) for typed_paths in dataset_paths_with_type]


async def dep_model_ckpt_path(
    model_ckpt_path: Path,
):
    if (
        not model_ckpt_path.is_relative_to(server_store.args.model_dir)
        or not model_ckpt_path.exists()
    ):
        raise HTTPException(
            status_code=404,
            detail="Model not found. Please send GET /models to get supported models.",
        )
    return model_ckpt_path


def dep_audio_file(audio_files: list[UploadFile]):
    """Require request MIME-type to be application/vnd.api+json."""

    if any(
        [not audio_file.content_type.startswith("audio") for audio_file in audio_files]
    ):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "All files must be audio files.",
        )
    return audio_files


def dep_train_dataset_type(dataset_paths_with_type: list[DatasetTypedPath]):
    """Require request MIME-type to be application/vnd.api+json."""
    return [SupportedDatasetDirType(e) for e in dataset_paths_with_type]
