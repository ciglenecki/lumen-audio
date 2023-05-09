from pathlib import Path
from typing import Literal

import torch

from src.config.config_defaults import ConfigDefault
from src.inference.run_test import get_inference_datamodule, get_inference_model_objs
from src.server.config_server import get_server_args


class ServerStore:
    def __init__(self):
        args, config, _, parser_help = get_server_args()
        self.parser_help = parser_help
        self.set_config(config, args)

    def set_config(self, config: ConfigDefault, args):
        self.config = config
        self.config.bar_update = 1
        self.args = args
        self.device = torch.device(args.device)

        # self.set_model()
        # self.set_dataset(self.audio_transform, self.collate_fn, self.model_config)

    def set_model(self):
        model, model_config, audio_transform = get_inference_model_objs(
            self.config, self.args, self.device
        )
        self.model = model
        self.model_config = model_config
        self.audio_transform = audio_transform

    def set_dataset(self, type=Literal["test", "pred"]):
        self.datamodule = get_inference_datamodule(
            self.config, self.audio_transform, self.model_config
        )

        self.data_loader = (
            self.datamodule.test_dataloader()
            if type == "test"
            else self.datamodule.predict_dataloader()
        )

    def get_available_models(self) -> list[Path]:
        return [
            str(path)
            for path in sorted(self.args.model_dir.rglob("*.ckpt"), reverse=True)
        ]


server_store = ServerStore()
