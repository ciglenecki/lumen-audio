from itertools import combinations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torchmetrics.functional.classification import (
    f1_score,
    multilabel_accuracy,
    multilabel_f1_score,
    multilabel_hamming_distance,
    multilabel_precision,
    multilabel_recall,
)

import src.config.config_defaults as config_defaults
from src.config.config_defaults import (
    ALL_INSTRUMENTS,
    INSTRUMENT_TO_FULLNAME,
    INSTRUMENT_TO_IDX,
)
from src.utils.utils_functions import dict_torch_to_npy


def mlb_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[tuple[str, str], np.ndarray]:
    """Compute the confusion matrix for each pair of instruments."""
    labels = ALL_INSTRUMENTS
    lables_indices = [INSTRUMENT_TO_IDX[label] for label in labels]
    labels_names = [INSTRUMENT_TO_FULLNAME[label] for label in labels]

    conf_mat_dict = {}
    for label1, label2 in combinations(lables_indices, 2):
        y_true_conf = y_true[:, label1]
        y_pred_conf = y_pred[:, label2]
        label_name1, label_name2 = labels_names[label1], labels_names[label2]
        label1, label2 = labels[label1], labels[label2]
        conf_mat_dict[(label_name1, label_name2)] = confusion_matrix(
            y_pred=y_pred_conf, y_true=y_true_conf, labels=[0, 1]
        )

    return conf_mat_dict


def find_best_threshold_per_class(
    y_pred_prob: torch.Tensor,
    y_true: torch.Tensor,
    num_labels: int,
    num_iter=1000,
    metric_fn=f1_score,
    min_or_max="max",
) -> list[float]:
    """Find the best threshold for each class.

    # WARNING: this function shouldn't be used as it's heavly biased towards the distribution of the test dataset.
    Args:
        y_pred_prob
        y_true
        num_labels
        num_iter: number of iterations to find the best threshold
        metric_fn: metric function to use to find the best threshold
        min_or_max: whether to find the minimum or maximum of the metric function
    """

    arg_max_or_min = torch.argmax if min_or_max == "max" else torch.argmin

    y_pred_prob = torch.tensor(y_pred_prob)
    y_true = torch.tensor(y_true)
    thresh_coarse = torch.tensor(np.linspace(0, 1, num=num_iter))

    # Find the best threshold for each class
    coarse_indices = []
    for label_idx in range(num_labels):
        y_pred_prob_i = y_pred_prob[:, label_idx]
        y_true_i = y_true[:, label_idx]
        metrics = []
        for t in thresh_coarse:
            metrics.append(metric_fn(y_pred_prob_i > t, y_true_i, task="binary"))
        t_coarse = arg_max_or_min(torch.tensor(metrics))
        coarse_indices.append(t_coarse)

    t_coarses = [thresh_coarse[idx] for idx in coarse_indices]

    # Find the best threshold for each class
    thresh_fine: list[list[float]] = []
    for label_idx, t_coarse in zip(range(num_labels), t_coarses):
        y_pred_prob_i = y_pred_prob[:, label_idx]
        y_true_i = y_true[:, label_idx]
        coarse_start = t_coarse - 0.1
        coarse_end = t_coarse + 0.1
        thresh_fine = torch.tensor(np.linspace(coarse_start, coarse_end, num=num_iter))
        metrics = []
        for t in thresh_fine:
            metrics.append(metric_fn(y_pred_prob_i > t, y_true_i, task="binary"))
        t_fine = arg_max_or_min(torch.tensor(metrics))
        thresh_fine.append(float(t_fine))

    return thresh_fine


def find_best_threshold(
    y_pred_prob: torch.Tensor,
    y_true: torch.Tensor,
    num_labels: int,
    num_iter=1000,
    metric_fn=multilabel_f1_score,
    min_or_max="max",
):
    """Find the best threshold for a given metric function.

    The function first finds the best threshold in a coarse range and then finds the best threshold
    in a fine range around the coarse threshold.
    """
    arg_max_or_min = torch.argmax if min_or_max == "max" else torch.argmin

    y_pred_prob = torch.tensor(y_pred_prob)
    y_true = torch.tensor(y_true)
    thresh_coarse = torch.tensor(np.linspace(0, 1, num=num_iter))
    coarse_idx = arg_max_or_min(
        torch.stack(
            [
                metric_fn(y_pred_prob > t, y_true, num_labels=num_labels)
                for t in thresh_coarse
            ]
        )
    )

    t_coarse = thresh_coarse[coarse_idx]
    coarse_start = t_coarse - 0.1
    coarse_end = t_coarse + 0.1

    thresh_fine = torch.tensor(np.linspace(coarse_start, coarse_end, num=num_iter))
    fine_idx = arg_max_or_min(
        torch.stack(
            [
                metric_fn(y_pred_prob > t, y_true, num_labels=num_labels)
                for t in thresh_fine
            ]
        )
    )
    return float(thresh_fine[fine_idx])


def get_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_labels=config_defaults.DEFAULT_NUM_LABELS,
    return_per_instrument=False,
    threshold=0.5,
    return_deep_dict=False,
    kwargs={},
) -> dict[str, float | torch.Tensor]:
    """Compute the metrics for a given set of predictions and targets.

    Args:
        y_pred
        y_true
        num_labels..
        return_per_instrument: return the metrics per instrument. Dictionary will stay flat.
        threshold: the threshold to use for the multilabel classification.
        return_deep_dict: return a deep dictionary with metrics for each instrument.
        kwargs: _description_..

    Returns dictionary with metrics
        {"f1": f1, "precision": precision, ... "instruments/gel_f1": gel_f1, ...}
    """
    kwargs = dict(
        preds=y_pred,
        target=y_true,
        num_labels=num_labels,
        threshold=threshold,
        **kwargs,
    )

    accuracy = multilabel_accuracy(**kwargs)
    hamming_distance = multilabel_hamming_distance(**kwargs)
    f1 = multilabel_f1_score(**kwargs)
    precision = multilabel_precision(**kwargs)
    recall = multilabel_recall(**kwargs)

    metrics = dict(
        accuracy=accuracy,
        hamming_distance=hamming_distance,
        f1=f1,
        precision=precision,
        recall=recall,
    )

    if not return_per_instrument:
        return metrics

    kwargs = dict(preds=y_pred, target=y_true, num_labels=num_labels, average=None)

    metrics_per_instrument = dict(
        accuracy=multilabel_accuracy(**kwargs),
        hamming_distance=multilabel_hamming_distance(**kwargs),
        f1=multilabel_f1_score(**kwargs),
        precision=multilabel_precision(**kwargs),
        recall=multilabel_recall(**kwargs),
    )

    if return_deep_dict:
        for metric_name, metric_value in metrics_per_instrument.items():
            for instrument_enum, idx in INSTRUMENT_TO_IDX.items():
                fullname = INSTRUMENT_TO_FULLNAME[instrument_enum]
                if fullname not in metrics:
                    metrics[fullname] = {}
                metrics[fullname][metric_name] = metric_value[idx]
    else:
        for metric_name, metric_value in metrics_per_instrument.items():
            for instrument_enum, idx in INSTRUMENT_TO_IDX.items():
                fullname = INSTRUMENT_TO_FULLNAME[instrument_enum]
                metrics[f"instruments/{fullname}_{metric_name}"] = metric_value[idx]

    return metrics


def get_metrics_npy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    num_labels=config_defaults.DEFAULT_NUM_LABELS,
    return_per_instrument=False,
    **kwargs,
):
    """Same as get_metrics but for numpy."""
    return dict_torch_to_npy(
        get_metrics(
            torch.tensor(y_pred),
            torch.tensor(y_true),
            num_labels,
            return_per_instrument,
            **kwargs,
        )
    )
