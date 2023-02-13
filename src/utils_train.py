from enum import Enum

import torch


class SchedulerType(Enum):
    ONECYCLE = "onecycle"
    PLATEAU = "plateau"
    AUTO_LR = "auto_lr"

    def __str__(self):
        return self.value


class OptimizerType(Enum):
    ADAM = "adam"
    ADAMW = "adamw"

    def __str__(self):
        return self.value


class MetricMode(Enum):
    min = "min"
    max = "max"

    def __str__(self):
        return self.value


class OptimizeMetric(Enum):
    VAL_HAMMING = "val/hamming_acc"

    def __str__(self):
        return self.value


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
