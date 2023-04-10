import torch
import torch.nn as nn

import src.config.config_defaults as config_defaults


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
        instrument_list: list[config_defaults.InstrumentEnums] = list(
            config_defaults.InstrumentEnums
        ),
    ):
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                instrument.value: head_constructor(**head_kwargs)
                for instrument in instrument_list
            }
        )

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
        # TODO: before using this function check implications of .eval()
        # self.heads[instrument].eval()
