from collections.abc import Iterator  # Python >=3.9
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import sklearn.metrics as skmetrics
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from config import config_defaults
from src.config.argparse_with_config import ArgParseWithConfig
from src.config.config_defaults import (
    ALL_INSTRUMENTS,
    ALL_INSTRUMENTS_NAMES,
    INSTRUMENT_TO_FULLNAME,
    ConfigDefault,
)
from src.train.inference_utils import (
    aggregate_inference_loops,
    get_inference_datamodule,
    get_inference_model_objs,
    validate_inference_args,
)
from src.train.metrics import find_best_threshold, get_metrics, mlb_confusion_matrix
from src.utils.utils_dataset import decode_instruments
from src.utils.utils_functions import dataset_path_to_str


def get_model_description(config):
    model_name = config.model.value
    train_datasets = ", ".join([dataset_path_to_str(p[1]) for p in config.train_paths])
    val_datasets = ", ".join([dataset_path_to_str(p[1]) for p in config.val_paths])
    return model_name, train_datasets, val_datasets


def main(args, config: ConfigDefault):
    validate_inference_args(config)
    config.test_paths = config.dataset_paths
    device = torch.device(args.device)
    model, model_config, audio_transform = get_inference_model_objs(
        config, args, device
    )

    model_text, text_train, _ = get_model_description(model_config)

    train_ds_text = "_".join(
        [dataset_path_to_str(p[1]) for p in model_config.train_paths]
    )
    val_ds_text = "_".join([dataset_path_to_str(p[1]) for p in model_config.val_paths])
    test_ds_text = "_".join([dataset_path_to_str(p[1]) for p in config.dataset_paths])

    experiment_name = f"{str(model_config.model.value)}_train_{train_ds_text}_val_{val_ds_text}_test_{test_ds_text}"
    experiment_desc = (
        f"{model_text}\nTrained on: {text_train}\nResults for: {test_ds_text}"
    )

    output_dir = Path(config.path_figures)
    output_dir_conf = Path(output_dir, "conf_matrix")
    output_dir_conf.mkdir(parents=True, exist_ok=True)
    output_dir_roc = Path(output_dir, "roc")
    output_dir_roc.mkdir(parents=True, exist_ok=True)

    datamodule = get_inference_datamodule(config, audio_transform, model_config)
    data_loader = datamodule.test_dataloader()
    result = aggregate_inference_loops(
        device, model, datamodule, data_loader, step_type="test"
    )

    threshold = find_best_threshold(
        y_pred_prob=result.y_pred_prob_file,
        y_true=result.y_true_file,
        num_labels=config.num_labels,
    )
    print("Best threshold: ", threshold)

    y_pred = result.y_pred.astype(int)
    y_pred_file = result.y_pred_file.astype(int)
    y_true = result.y_true.astype(int)
    y_true_file = result.y_true_file.astype(int)

    metric_dict = get_metrics(
        y_pred=torch.tensor(y_pred),
        y_true=torch.tensor(y_true),
        num_labels=config.num_labels,
        return_per_instrument=True,
        threshold=threshold,
    )
    metric_dict_file = get_metrics(
        y_pred=torch.tensor(y_pred_file),
        y_true=torch.tensor(y_true_file),
        num_labels=config.num_labels,
        return_per_instrument=True,
        threshold=threshold,
    )

    conf_matr_dict = mlb_confusion_matrix(y_pred_file, y_true_file)
    print("Saving to:", output_dir_conf)
    # for (name1, name2), conf_matrix in tqdm(conf_matr_dict.items()):
    #     cm_display = skmetrics.ConfusionMatrixDisplay(
    #         conf_matrix,
    #         display_labels=[name1, name2],
    #     )
    #     cm_display.plot()

    #     plt.title(
    #         f"Confusion matrix for \n{experiment_desc}",
    #     )
    #     plt.tight_layout()
    #     plt.savefig(
    #         Path(
    #             output_dir_conf,
    #             f"conf_matrix_{experiment_name}_{name1.replace(' ', '')}{name2.replace(' ', '')}.png",
    #         )
    #     )
    # plt.close("all")

    print("Saving to:", output_dir_roc)

    fig, ax = plt.subplots()
    num_colors = y_true.shape[-1]
    cm = plt.get_cmap("gist_rainbow")
    # ax.set_prop_cycle(color=[cm(1.0 * i / num_colors) for i in range(num_colors)])

    for i, instrument in tqdm(enumerate(ALL_INSTRUMENTS_NAMES)):
        y_true_instrument = y_true[:, i]
        y_pred_instrument = result.y_pred_prob[:, i]
        fpr, tpr, threshold = skmetrics.roc_curve(y_true_instrument, y_pred_instrument)
        roc_auc = skmetrics.auc(fpr, tpr)

        plt.title(
            f"ROC for {instrument}\n{experiment_desc}",
        )
        plt.plot(
            fpr,
            tpr,
            "b",
            label=f"AUC {instrument}={roc_auc:.2f}",
            color=cm(i / num_colors),
        )
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.tight_layout()
    plt.savefig(
        Path(
            output_dir_roc,
            f"roc_{experiment_name}.png",
        )
    )

    return metric_dict, metric_dict_file, y_pred, y_pred_file


if __name__ == "__main__":
    parser = ArgParseWithConfig()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to be used eg. cuda:0.",
    )
    args, config, _ = parser.parse_args()
    main(args, config)
