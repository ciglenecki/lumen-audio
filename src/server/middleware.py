from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from src.server.server_store import server_store


async def dep_model_checkpoint(model_checkpoint: Path):
    if (
        not model_checkpoint.is_relative_to(server_store.args.model_dir)
        or not model_checkpoint.exists()
    ):
        raise HTTPException(
            status_code=404,
            detail=f"Model not found. Please send GET /models to get supported models.",
        )
    return model_checkpoint


def dep_audio_file(audio_files: list[UploadFile]):
    """Require request MIME-type to be application/vnd.api+json."""

    if any(
        [not audio_file.content_type.startswith("audio") for audio_file in audio_files]
    ):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"All files have to must be audio/ files.",
        )
    return audio_files
