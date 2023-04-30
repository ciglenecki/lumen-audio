from pathlib import Path
from typing import Optional

from pydantic import BaseModel, create_model

from src.config.config_defaults import InstrumentEnums


class PostPredictDirectory(BaseModel):
    model_path: Path
    directory: Path


class PostPredictDatasetRequest(BaseModel):
    dataset_directory_path: str
    csv_filename: Optional[str] = None


instrument_fields = {e.value: int for e in InstrumentEnums}
MultilabelPrediction = create_model("MultilabelPrediction", **instrument_fields)
