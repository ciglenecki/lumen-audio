import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
from torch_scatter import scatter_max
from torchmetrics.classification import MultilabelF1Score
from torchvision.models import (
    convnext_base,
    convnext_large,
    convnext_small,
    convnext_tiny,
    efficientnet_v2_l,
    efficientnet_v2_m,
    efficientnet_v2_s,
    mobilenet_v3_large,
    resnext50_32x4d,
    resnext101_32x8d,
    resnext101_64x4d,
)

from src.model.heads import DeepHead
from src.model.model import SupportedModels
from src.model.model_base import ForwardInput, ForwardOut, ModelBase
from src.utils.utils_audio import plot_spectrograms
from src.utils.utils_dataset import decode_instruments
from src.utils.utils_exceptions import UnsupportedModel

TORCHVISION_CONSTRUCTOR_DICT = {
    SupportedModels.EFFICIENT_NET_V2_S: efficientnet_v2_s,
    SupportedModels.EFFICIENT_NET_V2_M: efficientnet_v2_m,
    SupportedModels.EFFICIENT_NET_V2_L: efficientnet_v2_l,
    SupportedModels.RESNEXT50_32X4D: resnext50_32x4d,
    SupportedModels.RESNEXT101_32X8D: resnext101_32x8d,
    SupportedModels.RESNEXT101_64X4D: resnext101_64x4d,
    SupportedModels.CONVNEXT_TINY: convnext_tiny,
    SupportedModels.CONVNEXT_SMALL: convnext_small,
    SupportedModels.CONVNEXT_LARGE: convnext_large,
    SupportedModels.CONVNEXT_BASE: convnext_base,
    SupportedModels.MOBILENET_V3_LARGE: mobilenet_v3_large,
}
import time

import src.config.config_defaults as config_defaults
from src.utils.utils_functions import timeit


class TorchvisionModel(ModelBase):
    """Implementation of a torchvision model accessed using a string."""

    loggers: list[TensorBoardLogger]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        torch.set_float32_matmul_precision("medium")

        if self.model_enum not in TORCHVISION_CONSTRUCTOR_DICT:
            raise UnsupportedModel(
                f"If you want to use {self.model_enum} in TorchvisionModel you need to add the enum to TORCHVISION_CONSTRUCTOR_DICT map."
            )

        backbone_constructor = TORCHVISION_CONSTRUCTOR_DICT[self.model_enum]
        backbone_kwargs = {}

        if backbone_constructor in {
            resnext50_32x4d,
            resnext101_32x8d,
            resnext101_64x4d,
        }:
            backbone_kwargs.update({"zero_init_residual": True})

        self.backbone: torch.nn.Module = backbone_constructor(
            weights=self.pretrained_tag, progress=True, **backbone_kwargs
        )

        print("------------------------------------------")
        print("\n")
        print("Backbone before changing the classifier:")
        print(list(self.backbone.children())[-1])
        print("\n")
        print("------------------------------------------")

        last_module_name = [
            i[0]
            for i in self.backbone.named_modules()
            if "." not in i[0] and i[0] != ""
        ][-1]
        last_module = getattr(self.backbone, last_module_name)
        last_dim = (
            last_module[-1].in_features
            if isinstance(last_module, nn.Sequential)
            else last_module.in_features
        )

        head = self.create_head(head_input_size=last_dim)
        setattr(self.backbone, last_module_name, head)

        print("\n")
        print("Backbone after changing the classifier:")
        print(list(self.backbone.children())[-1])
        print("\n")
        print("------------------------------------------")

        self.save_hyperparameters()

    def forward(self, image: torch.Tensor):
        out = self.backbone.forward(image)
        return out

    def forward_wrapper(self, forward_input: ForwardInput) -> ForwardOut:
        images, y_true = forward_input.feature, forward_input.y_true
        logits_pred = self.forward(images)
        if y_true is not None:
            loss = self.loss_function(logits_pred, y_true)
        else:
            loss = None
        return ForwardOut(logits=logits_pred, loss=loss)
