import torch
from torchmetrics.functional.classification import (
    multilabel_accuracy,
    multilabel_f1_score,
    multilabel_hamming_distance,
    multilabel_precision,
    multilabel_recall,
)

import src.config.config_defaults as config_defaults
from src.config.config_defaults import INSTRUMENT_TO_FULLNAME, INSTRUMENT_TO_IDX


def get_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_labels=config_defaults.DEFAULT_NUM_LABELS,
    return_per_instrument=False,
):
    kwargs = dict(preds=y_pred, target=y_true, num_labels=num_labels)

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
