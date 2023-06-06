"""python3 src/visual/plot_dataset_distrib.py --dataset-paths irmastrain:data/irmas/train
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
    parser.add_argument(
        "--use-seconds",
        action="store_true",
    )
    parser.add_argument(
        "--color",
        default="blue",
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
    print(datamodule.get_test_dataset_stats())
    datasets: list[DatasetBase] = datamodule.test_dataset.datasets
    plot_title = args.title_suffix
    # plot_title = "{}\n{}".format(args.title_suffix, "\n".join(dataset_names))

    instrument_items = []
    split_str = "instrument sec " if args.use_seconds else "instrument"
    for dataset, dataset_name in zip(datasets, dataset_names):
        for key, count in dataset.stats.items():
            instrument = key.split(split_str)
            if len(instrument) == 2:
                instrument = instrument[1]
                instrument_items.append((dataset_name, instrument, count))
    # instrument_items.sort(key=lambda x: x[1])
    df = pd.DataFrame(instrument_items, columns=["Dataset name", "instrument", "value"])
    df_pivot = df.pivot(columns="Dataset name", index="instrument", values="value")
    # Reorder columns (train first, test second)
    df_pivot = df_pivot.reindex(sorted(df_pivot.columns, reverse=True), axis=1)
    color = args.color if len(dataset_names) == 1 else None
    df_pivot.plot.bar(
        stacked=True, figsize=(16, 3.5), width=0.4, edgecolor="black", color=color
    )

    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=18)

    plt.title(plot_title, fontsize=26)

    if args.use_seconds:
        ax = plt.gca()
        plt.ticklabel_format(style="plain", axis="y")
        ax.yaxis.set_major_formatter("{x} s")
        plt.ylabel("Seconds [s]", fontsize=20)
    else:
        plt.ylabel("Count", fontsize=20)

    plt.xlabel("")

    if len(dataset_names) != 1:
        plt.legend(
            bbox_to_anchor=(1, 1),
            loc="upper right",
            bbox_transform=plt.gcf().transFigure,
            ncol=1,
            prop={"size": 18},
        )
    else:
        plt.gca().get_legend().remove()

    plt.tight_layout()
    png_path = Path(
        config.path_figures,
        f"instr_distrib_{clean_str('_'.join(dataset_names))}.png",
    )
    print("Saving to:", png_path)
    plt.savefig(png_path)
    plt.close()

    df_pivot = df.pivot(columns="instrument", index="Dataset name", values="value")
    df_pivot = df_pivot.sort_values(by=["Dataset name"], ascending=True)
    # Reorder columns (train first, test second)
    df_pivot = df_pivot.sort_values(by="Dataset name", ascending=False)
    df_pivot.plot.bar(
        stacked=True, figsize=(13, 8), width=0.6, edgecolor="black", color=color
    )

    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18)

    plt.title(plot_title, fontsize=26)
    if args.use_seconds:
        ax = plt.gca()
        plt.ticklabel_format(style="plain", axis="y")
        ax.yaxis.set_major_formatter("{x} s")
        plt.ylabel("Seconds [s]", fontsize=20)
    else:
        plt.ylabel("Count", fontsize=20)
    plt.xlabel("")

    if len(dataset_names) != 1:
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", prop={"size": 18})
    else:
        plt.gca().get_legend().remove()
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
