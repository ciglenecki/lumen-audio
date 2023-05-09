from enum import Enum
from itertools import chain
from pathlib import Path

import torch
from fastapi import HTTPException
from pydantic import BaseModel, create_model, validator

import src.config.config_defaults as config_defaults
from src.config.config_defaults import InstrumentEnums
from src.enums.enums import SupportedDatasetDirType
from src.server.server_store import server_store
from src.train.metrics import get_metrics


class SupportedDatasetDirTypeTrain(Enum):
    IRMAS_TEST = SupportedDatasetDirType.IRMAS_TEST.value
    IRMAS_TRAIN = SupportedDatasetDirType.IRMAS_TRAIN.value
    CSV = SupportedDatasetDirType.CSV.value


class DatasetTypedPath(BaseModel):
    dataset_type: SupportedDatasetDirTypeTrain
    dataset_path: Path

    class Config:
        schema_extra = {
            "example": {
                "dataset_type": "irmastest",
                "dataset_path": "data/irmas/test",
            }
        }


example_metrics = get_metrics(
    torch.zeros(2, server_store.config.num_labels),
    torch.zeros(2, server_store.config.num_labels),
    num_labels=server_store.config.num_labels,
    return_per_instrument=server_store.config.log_per_instrument_metrics,
)
metrics = example_metrics.keys()

instrument_fields = {e.value: (float, ...) for e in InstrumentEnums}
metric_fields = {m: (float, ...) for m in metrics}

InstrumentPrediction = create_model("InstrumentPrediction", **instrument_fields)
InstrumentPredictions = dict[str, InstrumentPrediction]


PredictionsWithMetrics = create_model(
    "PredictionsWithMetrics",
    **{**metric_fields, "predictions": (InstrumentPredictions, ...)},
)
