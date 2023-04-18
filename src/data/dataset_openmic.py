from torch.utils.data import Dataset
import pandas as pd

import src.config.config_defaults as config_defaults
from src.utils.utils_audio import load_audio_from_file

class OpenMICDatasetTrain(Dataset):
    """
    OpenMIC directory structure:
    |- data_dir
        |- clsmap.json
        |- openmic-2018.npz
        |- audio
            |-000
                |-000046_3840.ogg
                |- ...
            |-001
                |-001020_395520.ogg
                |- ...
            ... 
    """

    def __init__(
            self,data_csv,
            audio_transform=None,
            sampling_rate = config_defaults.sampling_rate):
        """
        dataframe cols:
        "name", "file", "pia", "voi", "vio". . ., "gac", "gel"
        "file": relative path to file
        """
        self.dataframe = pd.read_csv(data_csv,index_col=0)
        self.sampling_rate = sampling_rate
        self.audio_transform = audio_transform

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        instance = self.dataframe.iloc[index]
        data, = load_audio_from_file(
            audio_path = instance["file"],
            target_sr=self.sampling_rate
            )
        if self.audio_transform is not None:
            data = self.audio_transform.process(data)
        labels = ...
        return 


