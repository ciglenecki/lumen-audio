from itertools import combinations

import numpy as np
import torch
from sklearn.calibration import label_binarize
from sklearn.metrics import confusion_matrix
from torchmetrics.functional.classification import (
    multilabel_accuracy,
    multilabel_f1_score,
    multilabel_hamming_distance,
    multilabel_precision,
    multilabel_recall,
)

import src.config.config_defaults as config_defaults
from src.config.config_defaults import (
    ALL_INSTRUMENTS,
    ALL_INSTRUMENTS_NAMES,
    INSTRUMENT_TO_FULLNAME,
    INSTRUMENT_TO_IDX,
)
from src.utils.utils_functions import dict_torch_to_npy


def mlb_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[tuple[str, str], np.ndarray]:
    # Convert the true labels and predicted labels to corresponding custom labels
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


def find_best_threshold(
    y_pred_prob: torch.Tensor, y_true: torch.Tensor, num_labels: int, num_iter=1000
):
    y_pred_prob = torch.tensor(y_pred_prob)
    y_true = torch.tensor(y_true)
    threshold_values_coarse = torch.tensor(np.linspace(0, 1, num=num_iter))
    coarse_idx = torch.argmax(
        torch.stack(
            [
                multilabel_f1_score(y_pred_prob > t, y_true, num_labels=num_labels)
                for t in threshold_values_coarse
            ]
        )
    )

    t_coarse = threshold_values_coarse[coarse_idx]
    coarse_start = t_coarse - 0.1
    coarse_end = t_coarse + 0.1

    threshold_values_fine = torch.tensor(
        np.random.uniform(np.linspace(coarse_start, coarse_end, num=num_iter))
    )
    fine_idx = torch.argmax(
        torch.stack(
            [
                multilabel_f1_score(y_pred_prob > t, y_true, num_labels=num_labels)
                for t in threshold_values_fine
            ]
        )
    )
    return float(threshold_values_fine[fine_idx])


def get_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_labels=config_defaults.DEFAULT_NUM_LABELS,
    return_per_instrument=False,
    threshold=0.5,
):
    kwargs = dict(
        preds=y_pred, target=y_true, num_labels=num_labels, threshold=threshold
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
):
    return dict_torch_to_npy(
        get_metrics(
            torch.tensor(y_pred),
            torch.tensor(y_true),
            num_labels,
            return_per_instrument,
        )
    )
