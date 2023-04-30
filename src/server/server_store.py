import os
from pathlib import Path

import torch

from features.chunking import collate_fn_feature
from src.config.config_defaults import ConfigDefault
from src.enums.enums import SupportedModels
from src.features.audio_transform_base import AudioTransformBase
from src.train.run_test import get_datamodule, get_model


class ServerStore:
    def __init__(self):
        self.config = None

    def set_config(self, config: ConfigDefault, args):
        self.config = config
        self.args = args
        self.device = torch.device(args.device)

        # self.set_model()
        # self.set_dataset(self.audio_transform, self.collate_fn, self.model_config)

    def set_model(self):
        model, model_config, audio_transform = get_model(self.config, self.args)
        self.model = model
        self.model_config = model_config
        self.audio_transform = audio_transform
        self.collate_fn = collate_fn_feature

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

    def get_available_models(self) -> list[Path]:
        return [path for path in self.args.model_dir.rglob("*.ckpt")]


server_store = ServerStore()
