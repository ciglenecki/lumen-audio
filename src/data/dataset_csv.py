from pathlib import Path

import numpy as np
import pandas as pd
import torch

import src.config.config_defaults as config_defaults
from src.config.config_defaults import AUDIO_EXTENSIONS, get_default_config
from src.data.dataset_base import DatasetBase, DatasetGetItem
from src.utils.utils_exceptions import InvalidArgument

config = get_default_config()
glob_expressions = [f"*.{ext}" for ext in AUDIO_EXTENSIONS]


class CSVDataset(DatasetBase):
    def __init__(self, *args, **kwargs):
        """
        Args:
            dataset_path: CSV with the following structure:
            file,cel,cla,flu,gac,gel,org,pia,sax,tru,vio,voi
            data/openmic/audio/000/000135_483840.ogg,0,0,0,0,0,0,0,0,0,0,1
            data/openmic/audio/000/000178_3840.ogg,0,0,0,0,0,0,0,0,0,0,1
            ...

        """
        if not kwargs["dataset_path"].is_file():
            raise InvalidArgument(f"{str(self.dataset_path)} is not a file.")

        super().__init__(*args, **kwargs)

    def create_dataset_list(self) -> list[tuple[Path, np.ndarray]]:
        """Reads audio and label files and creates tuples of (audio_path, one hot encoded label)"""
        dataset_list: list[tuple[Path, np.ndarray]] = []

        df = pd.read_csv(self.dataset_path)
        for _, row in df.iterrows():
            filepath = Path(row["file"])
            sorted_instruments = [
                config_defaults.IDX_TO_INSTRUMENT[i]
                for i in range(len(config_defaults.IDX_TO_INSTRUMENT))
            ]
            labels = np.array(
                [row[instrument] for instrument in sorted_instruments], dtype=int
            )
            dataset_list.append((filepath, labels))

        return dataset_list
