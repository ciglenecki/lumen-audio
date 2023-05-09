import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_metric_learning import losses
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

from src.model.model import SupportedModels
from src.model.model_arcface import ArcFaceModel
from src.model.model_base import ModelBase
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


class TorchvisionModel(ModelBase):
    """Vision model which can load Torch Vision pretrained models."""

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

        print("\n================== Backbone before changing the classifier:\n")
        print(list(self.backbone.children())[-1])
        if backbone_constructor in {
            resnext50_32x4d,
            resnext101_32x8d,
            resnext101_64x4d,
        }:
            old_linear = self.backbone.fc
            last_dim = old_linear.in_features
            head = self.create_head(head_input_size=last_dim)

            self.backbone.fc = torch.nn.ModuleList([ArcFaceModel(last_dim), head])
        elif backbone_constructor in {
            mobilenet_v3_large,
            efficientnet_v2_s,
            efficientnet_v2_m,
            efficientnet_v2_l,
            convnext_tiny,
            convnext_small,
            convnext_base,
            convnext_large,
        }:
            old_linear = self.backbone.classifier[-1]
            last_dim = old_linear.in_features
            head = self.create_head(head_input_size=last_dim)
            self.backbone.classifier[-1] = torch.nn.ModuleList(
                [ArcFaceModel(last_dim), head]
            )
        else:
            raise UnsupportedModel(
                f"Please implement classifier logic for model {self.model_enum}"
            )

        print("\n\nBackbone after changing the classifier:")
        print(list(self.backbone.children())[-1])

        self.save_hyperparameters()

    def forward(self, image: torch.Tensor):
        out = self.backbone.forward(image)
        return out
