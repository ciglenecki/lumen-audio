from pathlib import Path

from enums.enums import SupportedModels
from features.audio_transform_base import AudioTransformBase
from src.config.config_defaults import ConfigDefault
from src.train.test import get_datamodule, get_model, prepare_objects


class ServerStore:
    def __init__(self):
        self.config = None

    def set_config(self, config: ConfigDefault, args):
        self.config = config
        self.args = args
        # self.model: None | SupportedModels = None
        # self.model_config: None | ConfigDefault = None
        # self.audio_transform: None | AudioTransformBase = None
        # self.collate_fn: None = None
        (
            self.model,
            self.model_config,
            self.audio_transform,
            self.collate_fn,
        ) = get_model(self.config, self.args)
        self.set_model()
        self.set_dataset(self.audio_transform, self.collate_fn, self.model_config)

    def set_model(self):
        model, model_config, audio_transform, collate_fn = get_model(
            self.config, self.args
        )
        self.model = model
        self.model_config = model_config
        self.audio_transform = audio_transform
        self.collate_fn = collate_fn

    def set_dataset(
        self,
        audio_transform=None,
        collate_fn=None,
        model_config=None,
    ):
        if audio_transform:
            self.audio_transform = audio_transform
        if collate_fn is not None:
            self.collate_fn = collate_fn
        if model_config is not None:
            self.model_config = model_config

        self.datamodule, self.data_loader = get_datamodule(
            self.config, self.audio_transform, self.collate_fn, self.model_config
        )


server_store = ServerStore()
