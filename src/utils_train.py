import torch

from src.utils_functions import EnumStr


class SchedulerType(EnumStr):
    ONECYCLE = "onecycle"
    PLATEAU = "plateau"
    AUTO_LR = "auto_lr"


class OptimizerType(EnumStr):
    ADAM = "adam"
    ADAMW = "adamw"


class MetricMode(EnumStr):
    MIN = "min"
    MAX = "max"


class OptimizeMetric(EnumStr):
    VAL_HAMMING = "val/hamming_acc"
    VAL_F1 = "val/f1_score"


class SupportedModels(EnumStr):
    AST = "ast"
    EFFICIENT_NET_V2_S = "efficient_net_v2_s"
    EFFICIENT_NET_V2_S_MULTI_TASK = "efficient_net_v2_s_multi_task"


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
