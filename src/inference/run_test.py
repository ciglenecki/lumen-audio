"""python3 src/inference/run_test.py  --dataset-paths irmastest:data/irmas/test --ckpt
models/05-08-11-38-04_SlickDelta_ast_astfiliteredhead-irmas-
audioset/checkpoints/05-08-11-38-04_SlickDelta_ast_astfiliteredhead-irmas-
audioset_val_acc_0.3742_val_loss_0.3504.ckpt --batch-size 2 --num-workers 1."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics
import torch
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

from src.config.argparse_with_config import ArgParseWithConfig
from src.config.config_defaults import ALL_INSTRUMENTS_NAMES, ConfigDefault
from src.enums.enums import NON_INFERENCE_DIR_TYPES, SupportedDatasetDirType
from src.inference.inference_utils import (
    aggregate_inference_loops,
    get_inference_datamodule,
    get_inference_model_objs,
    json_from_step_result,
    validate_inference_args,
)
from src.train.metrics import find_best_threshold, get_metrics, mlb_confusion_matrix
from src.utils.utils_exceptions import InvalidArgument
from src.utils.utils_functions import (
    dataset_path_to_str,
    save_json,
    save_yaml,
    to_yaml,
    torch_to_list,
)


def get_model_description(config):
    model_name = config.model.value
    train_datasets = ", ".join([dataset_path_to_str(p[1]) for p in config.train_paths])
    val_datasets = ", ".join([dataset_path_to_str(p[1]) for p in config.val_paths])
    return model_name, train_datasets, val_datasets


def parse_args():
    config_pl_args = ["--ckpt", "--dataset-paths", "--batch-size", "--num-workers"]
    parser = ArgParseWithConfig(config_pl_args=config_pl_args)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to be used eg. cuda:0.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Sets the output directory",
    )
    parser.add_argument(
        "--save-confusion",
        action="store_true",
        default=False,
        help="Caculate and save confusion matrices plot",
    )
    parser.add_argument(
        "--save-roc",
        action="store_true",
        default=False,
        help="Caculate and save ROC plot for each instrument",
    )
    parser.add_argument(
        "--save-metric-hist",
        action="store_true",
        default=False,
        help="Caculate and save histogram plot for distribution of each metric",
    )
    parser.add_argument(
        "--save-instrument-metrics",
        action="store_true",
        default=False,
        help="Caculate and save the plot metrics for each instrument",
    )
    args, config, _ = parser.parse_args()
    config.required_dataset_paths()

    return args, config


def check_inf_non_inf_dataset_paths(config: ConfigDefault):
    is_inf = all([d[0] not in NON_INFERENCE_DIR_TYPES for d in config.dataset_paths])
    is_non_inf = all([d[0] in NON_INFERENCE_DIR_TYPES for d in config.dataset_paths])
    if is_inf and is_non_inf:
        raise InvalidArgument(
            f"All dataset paths must be either inference or non-inference. Use inference to get only the predictions. To specify inference paths use /path/to/data or inference:/path/to/data format. If you want to get predictions and metrics, specify non-inference dataset e.g. irmastrain:/path/to/data. Possible non-inference dataset directory types are {NON_INFERENCE_DIR_TYPES}"
        )
    return is_inf


def get_dataset_text(
    dataset_paths_with_type: list[tuple[SupportedDatasetDirType, Path]]
):
    return "_".join([dataset_path_to_str(p[1]) for p in dataset_paths_with_type])


def get_out_dir(args, config: ConfigDefault):
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = Path(config.ckpt).parent
        if out_dir.name == "checkpoints":
            out_dir = out_dir.parent
    return out_dir


def main():
    args, config = parse_args()
    validate_inference_args(config)
    is_inf = check_inf_non_inf_dataset_paths(config)

    config.test_paths = config.dataset_paths
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    model, model_config, audio_transform = get_inference_model_objs(
        config, args, device
    )

    model_text, text_train, _ = get_model_description(model_config)
    train_ds_text = get_dataset_text(model_config.train_paths)
    val_ds_text = get_dataset_text(model_config.val_paths)
    test_ds_text = get_dataset_text(config.dataset_paths)
    out_dir = get_out_dir(args, config)
    experiment_name = f"{model_config.experiment_suffix}__train__{train_ds_text}__val__{val_ds_text}__pred__{test_ds_text}"
    experiment_desc = (
        f"{model_text}\nTrained on: {text_train}\nResults for: {test_ds_text}"
    )

    datamodule = get_inference_datamodule(config, audio_transform, model_config)
    data_loader = (
        datamodule.predict_dataloader() if not is_inf else datamodule.test_dataloader()
    )
    result = aggregate_inference_loops(
        device, model, datamodule, data_loader, step_type="pred" if is_inf else "test"
    )

    json_dict = json_from_step_result(result)
    json_path = out_dir / f"preds_{experiment_name}.json"
    save_json(json_dict, json_path)

    if is_inf:
        return

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

    metric_dict_patch = torch_to_list(
        get_metrics(
            y_pred=torch.tensor(y_pred),
            y_true=torch.tensor(y_true),
            num_labels=config.num_labels,
            return_per_instrument=True,
            threshold=threshold,
        )
    )
    print(to_yaml(metric_dict_patch))
    save_yaml(metric_dict_patch, Path(out_dir, f"metrics_patch_{experiment_name}.yaml"))

    metric_dict_file = torch_to_list(
        get_metrics(
            y_pred=torch.tensor(y_pred_file),
            y_true=torch.tensor(y_true_file),
            num_labels=config.num_labels,
            return_per_instrument=True,
            threshold=threshold,
        )
    )
    print(to_yaml(metric_dict_file))
    save_yaml(metric_dict_file, Path(out_dir, f"metrics_files_{experiment_name}.yaml"))

    if args.save_instrument_metrics:
        # Plot metrics by instrument
        metric_deep_file = get_metrics(
            y_pred=torch.tensor(y_pred_file),
            y_true=torch.tensor(y_true_file),
            num_labels=config.num_labels,
            return_per_instrument=True,
            threshold=threshold,
            return_deep_dict=True,
        )
        instrument_dict = {
            k: torch_to_list(v)
            for k, v in metric_deep_file.items()
            if k in ALL_INSTRUMENTS_NAMES
        }
        flat = []
        for instrument in instrument_dict:
            for metric in instrument_dict[instrument]:
                value = instrument_dict[instrument][metric]
                flat.append([instrument, metric, value])
        df = pd.DataFrame(flat, columns=["instrument", "metric", "value"])
        df.pivot(columns="metric", index="instrument", values="value").plot(
            kind="bar", figsize=(14, 4), width=0.6, edgecolor="black"
        )
        png_path = Path(
            out_dir,
            f"metrics_hist_{experiment_name}.png",
        )
        print("Saving to: ", png_path)

        plt.xticks(rotation=0, fontsize=14)
        plt.title(f"Metrics by instrument for \n{experiment_desc}")
        plt.ylabel("Values", fontsize=14)
        plt.xlabel("")
        plt.legend(
            bbox_to_anchor=(1, 1),
            loc="upper right",
            bbox_transform=plt.gcf().transFigure,
            ncol=3,
        )
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()

    if args.save_metric_hist:
        # Plot frequency of value for each metric
        metrics_no_reduction_file = get_metrics(
            y_pred=torch.tensor(y_pred_file).T,
            y_true=torch.tensor(y_true_file).T,
            num_labels=len(y_true_file),
            return_per_instrument=False,
            threshold=threshold,
            kwargs={"average": "none"},
        )
        out_dir_hist = Path(out_dir, "metric_hist")
        out_dir_hist.mkdir(parents=True, exist_ok=True)
        print("Saving metric histograms to:", out_dir_hist)
        for metric_name, metric_values in metrics_no_reduction_file.items():
            png_path = Path(
                out_dir_hist,
                f"{metric_name}_hist_{experiment_name}.png",
            )

            plt.figure(figsize=(12, 5))
            plt.hist(metric_values, bins=np.linspace(0, 1, 11))
            plt.xticks(np.linspace(0, 1, 11), fontsize=14)
            plt.gca().xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            plt.xlabel("Values", fontsize=14)
            plt.ylabel("Frequency", fontsize=14)
            plt.title(f"Histogram of {metric_name} for \n{experiment_desc}")
            plt.tight_layout()
            plt.savefig(png_path)
            plt.close()

    if args.save_confusion:
        # Plot confusion matrices
        out_dir_conf = Path(out_dir, "conf_matrix")
        out_dir_conf.mkdir(parents=True, exist_ok=True)

        conf_matr_dict = mlb_confusion_matrix(y_pred_file, y_true_file)
        print("Saving confusion matrices to:", out_dir_conf)
        for (name1, name2), conf_matrix in tqdm(conf_matr_dict.items()):
            cm_display = skmetrics.ConfusionMatrixDisplay(
                conf_matrix,
                display_labels=[name1, name2],
            )
            cm_display.plot()
            png_path = Path(
                out_dir_conf,
                f"conf_matrix_{experiment_name}_{name1.replace(' ', '')}{name2.replace(' ', '')}.png",
            )
            plt.title(
                f"Confusion matrix for \n{experiment_desc}",
            )
            plt.tight_layout()
            plt.savefig(png_path)
            plt.close()

    if args.save_roc:
        # Plot ROC
        out_dir_roc = Path(out_dir, "roc")
        out_dir_roc.mkdir(parents=True, exist_ok=True)
        print("Saving ROC to:", out_dir_roc)
        num_colors = y_true.shape[-1]
        cm = plt.get_cmap("gist_rainbow")

        for i, instrument in tqdm(enumerate(ALL_INSTRUMENTS_NAMES)):
            y_true_instrument = y_true[:, i]
            y_pred_instrument = result.y_pred_prob[:, i]
            fpr, tpr, threshold = skmetrics.roc_curve(
                y_true_instrument, y_pred_instrument
            )
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
        png_path = Path(
            out_dir_roc,
            f"roc_{experiment_name}.png",
        )
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()


if __name__ == "__main__":
    main()
