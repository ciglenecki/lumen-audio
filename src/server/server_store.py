from pathlib import Path

import torch

from src.config.config_defaults import ConfigDefault
from src.server.config_server import get_server_args
from src.train.run_test import get_datamodule, get_model_config_transform


class ServerStore:
    def __init__(self):
        args, config, _ = get_server_args()
        self.set_config(config, args)

    def set_config(self, config: ConfigDefault, args):
        self.config = config
        self.config.bar_update = 1
        self.args = args
        self.device = torch.device(args.device)

        # self.set_model()
        # self.set_dataset(self.audio_transform, self.collate_fn, self.model_config)

    def set_model(self):
        model, model_config, audio_transform = get_model_config_transform(
            self.config, self.args, self.device
        )
        self.model = model
        self.model_config = model_config
        self.audio_transform = audio_transform

    def set_dataset(
        self,
        audio_transform=None,
        model_config=None,
    ):
        if audio_transform:
            self.audio_transform = audio_transform

        if model_config is not None:
            self.model_config = model_config

        self.datamodule, self.data_loader = get_datamodule(
            self.config, self.audio_transform, self.model_config
        )

    def get_available_models(self) -> list[Path]:
        return [
            str(path)
            for path in sorted(self.args.model_dir.rglob("*.ckpt"), reverse=True)
        ]


server_store = ServerStore()
