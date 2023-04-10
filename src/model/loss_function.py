import torch

from src.enums.enums import SupportedLossFunctions
from src.utils.utils_dataset import calc_instrument_weight


def get_loss_fn(loss_function_enum: SupportedLossFunctions, datamodule, **kwargs):
    if loss_function_enum == SupportedLossFunctions.CROSS_ENTROPY:
        return torch.nn.BCEWithLogitsLoss(**kwargs)
    if loss_function_enum == SupportedLossFunctions.CROSS_ENTROPY_POS_WEIGHT:
        kwargs = {
            **kwargs,
            "pos_weight": calc_instrument_weight(datamodule.count_classes()),
        }
        return torch.nn.BCEWithLogitsLoss(**kwargs)
