from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config.config_defaults import AUDIO_EXTENSIONS
from src.data.dataset_base import DatasetBase
from src.features.audio_transform_base import AudioTransformBase
from src.utils.utils_exceptions import InvalidDataException

glob_expressions = [f"*.{ext}" for ext in AUDIO_EXTENSIONS]


class InferenceDataset(DatasetBase):
    def __init__(
        self,
        dataset_path: Path,
        audio_transform: AudioTransformBase | None,
        sampling_rate: int,
        normalize_audio: bool,
    ):
        """_summary_

        Args:
            dataset_path directory with the following structure:
                ├── waveform1.wav
                ├── waveform2.wav
                ├── avbcfd.wav
                ├── 4q59daxui.ogg

            OR

            dataset_path CSV with the following structure:
            file,cel,cla,flu,gac,gel,org,pia,sax,tru,vio,voi
            data/openmic/audio/000/000135_483840.ogg,0,0,0,0,0,0,0,0,0,0,1
            data/openmic/audio/000/000178_3840.ogg,0,0,0,0,0,0,0,0,0,0,1
        """
        kwargs = dict(
            num_classes=0,
            sum_n_samples=False,
            concat_n_samples=None,
            train_override_csvs=None,
        )
        super().__init__(
            dataset_path=dataset_path,
            audio_transform=audio_transform,
            sampling_rate=sampling_rate,
            normalize_audio=normalize_audio,
            **kwargs,
        )  # sets self.dataset

    def create_dataset_list(self) -> list[tuple[Path, np.ndarray]]:
        """Reads audio and label files and creates tuples of (audio_path, one hot encoded label)"""
        dataset_list: list[tuple[Path, np.ndarray]] = []
        if self.dataset_path.is_file() and self.dataset_path.suffix == ".csv":
            df = pd.read_csv(self.dataset_path)
            for _, row in df.iterrows():
                filepath = Path(row["file"])
                labels = np.array([0])
                dataset_list.append((filepath, labels))

        elif self.dataset_path.is_dir():
            glob_generators = [
                self.dataset_path.rglob(glob_exp) for glob_exp in glob_expressions
            ]
            for item_idx, audio_path in tqdm(enumerate(chain(*glob_generators))):
                labels = np.array([0])
                dataset_list.append((audio_path, labels))
            if not dataset_list:
                raise InvalidDataException(
                    f"Path {self.dataset_path} is does not contain any audio files"
                )
        else:
            raise InvalidDataException(
                f"Path {self.dataset_path} is not a csv or a directory which contains audio files."
            )
        return dataset_list
