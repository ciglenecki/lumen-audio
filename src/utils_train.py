from enum import Enum
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from src.utils_functions import EnumStr


class SchedulerType(EnumStr):
    ONECYCLE = "onecycle"
    PLATEAU = "plateau"
    AUTO_LR = "auto_lr"
    COSINEANNEALING = "cosine_annealing"


class OptimizerType(EnumStr):
    ADAM = "adam"
    ADAMW = "adamw"


class MetricMode(EnumStr):
    MIN = "min"
    MAX = "max"


class OptimizeMetric(Enum):
    VAL_HAMMING = "val/hamming_distance"
    VAL_F1 = "val/f1_score"


class SupportedModels(EnumStr):
    AST = "ast"
    EFFICIENT_NET_V2_S = "efficient_net_v2_s"


class UnsupportedOptimizer(ValueError):
    pass


class UnsupportedScheduler(ValueError):
    pass


class UnsupportedModel(ValueError):
    pass


def multi_acc(y_pred_log, y_test):
    """
    Args:
        y_pred_log: softmaxed prediciton from the model
        y_test: true value
    Returns:
        mean accuracy (e.g. 0.78)
    """
    _, y_pred_k = torch.max(y_pred_log, dim=1)
    _, y_test_tags = torch.max(y_test, dim=1)
    correct_pred = (y_pred_k == y_test_tags).float()

    acc = correct_pred.sum() / len(correct_pred)
    return float(acc)


def topk_accuracy(y_pred_log, y_test, k=3):
    """
    Args:
        y_pred_log: softmaxed prediciton from the model
        y_test: true value
    Returns:
        top k accuracy (e.g. 0.78)
    """

    _, y_pred_k = torch.topk(y_pred_log, k, dim=1)

    """
    y_test_class = [[1],
        [0],
        [1],
        [0]]
    """
    y_test_class = torch.argmax(y_test, dim=1).unsqueeze(axis=1)
    y_test_class_expanded = y_test_class.expand(-1, k)

    """
    y_test_class = [[1,1,1],
        [0,0,0],
        [1,1,1],
        [0,0,0]]
    """

    """
    equal = [[True, False, False],
        [False, False, False],
        [False, False, False],
        [False, True, False]]
    """
    equal = torch.eq(y_pred_k, y_test_class_expanded)
    """
    correct = [True, False, False, True]
    """
    correct = equal.any(dim=1)
    return correct.double().mean().item()


def get_last_fc_in_channels(
    backbone: Union[nn.ModuleList, nn.Module],
    image_size: int,
    num_channels: int = 3,
) -> Any:

    """Caculate output of the fully connected layer by forward passing a dummy image throught the
    backbone.

    Returns:
        number of input channels for the last fc layer (number of variables of the second dimension of the flatten layer). Fake image is created, passed through the backbone and flattened (while perseving batches).
    """
    # TODO: check if it should be 2 instead of 1
    batch_size = 1
    with torch.no_grad():
        image_batch = torch.rand(batch_size, num_channels, image_size, image_size)
        out_backbone = backbone(image_batch)
        out_backbone_cat = torch.cat(out_backbone, dim=1)
        flattened_output = torch.flatten(out_backbone_cat, 1)
    return flattened_output.shape[1]
