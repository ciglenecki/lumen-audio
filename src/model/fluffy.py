from dataclasses import dataclass
from typing import Callable, Type, Union

import torch
import torch.nn as nn

import src.config.config_defaults as config_defaults
from src.model.heads import AttentionHead, DeepHead


@dataclass
class FluffyConfig:
    """Class for keeping track of an item in inventory."""

    use_multiple_optimizers: bool = True
    classifer_constructor: Callable = DeepHead

    def __post_init__(self):
        assert self.classifer_constructor in [DeepHead, AttentionHead]


class Fluffy(nn.Module):
    """
    Fluffy model - named after Hagrid's three headed dog.
    This model creates len(instrument_list) decoupled head models
    where each head is used to find its own instrument in audio.
    NOTE: This is meant to be used along with LightningFluffy which
    uses a pretrained feature extraction backbone.
    """

    def __init__(
        self,
        head_constructor,
        head_kwargs,
    ):
        super().__init__()

        heads_dict = {}
        for idx in range(config_defaults.DEFAULT_NUM_LABELS):
            instrument_name = config_defaults.IDX_TO_INSTRUMENT[idx]
            key = instrument_name
            value = head_constructor(**head_kwargs)
            heads_dict[key] = value

        self.heads = nn.ModuleDict(heads_dict)

    def forward(self, features):
        out = [self.heads[instrument](features) for instrument in self.heads.keys()]
        out = torch.hstack(out)
        return out

    def freeze_instrument_head(self, instrument):
        """Used for instrument finetuning.

        If an instrument head performs badly and we want to finetune or repeat the training
        altogether, use this to freeze working heads.
        """
        for param in self.heads[instrument].parameters():
            param.requires_grad = False
