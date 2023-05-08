"""python3 src/visual/plot_dataset_distrib.py -dataset-paths irmastrain:data/irmas/train
irmastest:data/irmas/test --dataset-names 'IRMAS Train' 'Irmas Test' --title-suffix 'Train Dataset
Distribution'."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config.argparse_with_config import ArgParseWithConfig
from src.data.datamodule import OurDataModule
from src.data.dataset_base import DatasetBase
from src.utils.utils_functions import clean_str


def parse_args():
    parser = ArgParseWithConfig()
    parser.add_argument("--title-suffix", type=str, default="")
    parser.add_argument(
        "--dataset-names",
        type=str,
        nargs="+",
        help="Pretty datset names",
        required=True,
    )
    args, config, _ = parser.parse_args()
    config.required_dataset_paths()
    assert len(args.dataset_names) == len(
        config.dataset_paths
    ), "Dataset names and paths must be the same length"
    return args, config


def main():
    args, config = parse_args()
    dataset_names = args.dataset_names
    datamodule = OurDataModule(
        train_paths=None,
        val_paths=None,
        test_paths=config.dataset_paths,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dataset_fraction=config.dataset_fraction,
        drop_last_sample=config.drop_last,
        train_audio_transform=None,
        val_audio_transform=None,
        collate_fn=None,
        normalize_audio=config.normalize_audio,
        normalize_image=config.normalize_image,
        train_only_dataset=False,
        concat_n_samples=None,
        sum_n_samples=False,
        use_weighted_train_sampler=False,
        sampling_rate=config.sampling_rate,
        train_override_csvs=None,
    )
    datamodule.setup_for_inference()
    datasets: list[DatasetBase] = datamodule.test_dataset.datasets
    plot_title = "{}\n{}".format(args.title_suffix, "\n".join(dataset_names))

    instrument_items = []
    for dataset, dataset_name in zip(datasets, dataset_names):
        for key, count in dataset.stats.items():
            instrument = key.split("instrument ")
            if len(instrument) == 2:
                instrument = instrument[1]
                instrument_items.append((dataset_name, instrument, count))
    instrument_items.sort(key=lambda x: x[1])
    df = pd.DataFrame(instrument_items, columns=["Dataset name", "instrument", "value"])
    df.pivot(columns="Dataset name", index="instrument", values="value").plot.bar(
        stacked=True, figsize=(16, 5), width=0.4, edgecolor="black"
    )
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=18)

    plt.title(plot_title, fontsize=26)
    plt.ylabel("Count", fontsize=20)
    plt.xlabel("")

    plt.legend(
        bbox_to_anchor=(1, 1),
        loc="upper right",
        bbox_transform=plt.gcf().transFigure,
        ncol=3,
        prop={"size": 18},
    )
    plt.tight_layout()
    png_path = Path(
        config.path_figures,
        f"instr_distrib_{clean_str('_'.join(dataset_names))}.png",
    )
    print("Saving to:", png_path)
    plt.savefig(png_path)
    plt.close()

    df.pivot(columns="instrument", index="Dataset name", values="value").plot.bar(
        stacked=True, figsize=(13, 8), width=0.6, edgecolor="black"
    )
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18)

    plt.title(plot_title, fontsize=26)
    plt.ylabel("Count", fontsize=20)
    plt.xlabel("")

    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", prop={"size": 18})
    plt.tight_layout()
    png_path = Path(
        config.path_figures,
        f"dataset_distrib_{clean_str('_'.join(dataset_names))}.png",
    )
    print("Saving to:", png_path)
    plt.savefig(png_path)
    plt.close()


if __name__ == "__main__":
    main()
