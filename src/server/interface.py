from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, create_model

from src.config.config_defaults import InstrumentEnums
from src.enums.enums import SupportedDatasetDirType


class DatasetDirDict(BaseModel):
    dataset_dir_type: SupportedDatasetDirType
    dataset_dir: Path

    class Config:
        schema_extra = {
            "example": {
                "dataset_dir_type": "irmastest",
                "dataset_dir": "data/irmas/test",
            }
        }


class DatasetDirsI(BaseModel):
    dataset_dirs: list[DatasetDirDict]


class PostPredictDatasetRequest(BaseModel):
    dataset_directory_path: str
    csv_filename: Optional[str] = None


instrument_fields = {e.value: int for e in InstrumentEnums}
MultilabelPrediction = create_model("MultilabelPrediction", **instrument_fields)
