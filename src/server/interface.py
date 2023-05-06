from pathlib import Path
from typing import Literal, Optional

import torch
from pydantic import BaseModel, create_model

from src.config.config_defaults import InstrumentEnums
from src.enums.enums import SupportedDatasetDirType
from src.server.server_store import server_store
from src.train.metrics import get_metrics


class TypeDatasetDict(BaseModel):
    dataset_type: SupportedDatasetDirType
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

JsonPrediction = create_model("JsonPrediction", **instrument_fields)
JsonPredictions = dict[str, JsonPrediction]


PredictionsWithMetrics = create_model(
    "PredictionsWithMetrics", **{**metric_fields, "predictions": (JsonPredictions, ...)}
)
