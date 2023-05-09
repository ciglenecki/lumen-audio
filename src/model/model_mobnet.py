import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.models import mobilenet_v3_large

from src.config.config_defaults import DEFAULT_NUM_LABELS
from src.model.model import SupportedModels
from src.model.model_base import ModelBase

TORCHVISION_CONSTRUCTOR_DICT = {
    SupportedModels.MOBILENET_V3_LARGE: mobilenet_v3_large,
}


class MobNet(ModelBase):
    """Vision model which can load Torch Vision pretrained models."""

    loggers: list[TensorBoardLogger]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        torch.set_float32_matmul_precision("medium")

        self.mobilenets = nn.ModuleList([])

        self.num_labels = 1
        for i in range(DEFAULT_NUM_LABELS):
            model = mobilenet_v3_large(weights=self.pretrained_tag, progress=True)

            if i == 0:
                print("\n================== Backbone before changing the classifier:\n")
                print(list(model.children())[-1])

            old_linear = model.classifier[-1]
            last_dim = old_linear.in_features
            head = self.create_head(head_input_size=last_dim)
            model.classifier[-1] = head

            if i == 0:
                print("\n\nBackbone after changing the classifier:")
                print(list(model.children())[-1])

            self.mobilenets.append(model)
        self.num_labels = DEFAULT_NUM_LABELS

        self.save_hyperparameters()

    def forward(self, image: torch.Tensor):
        outs = [model.forward(image) for model in self.mobilenets]
        stacked_outs = torch.cat(outs, dim=-1)
        return stacked_outs
