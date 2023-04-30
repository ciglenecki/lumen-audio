"""Takes IRMAS test directory structure and splits it into 2 csvs with --frac ratio.

It Interanally uses iterative_train_test_split to split the dataset without changing the underlying
distribution.
"""

import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skmultilearn.model_selection.iterative_stratification import (
    iterative_train_test_split,
)
from tqdm import tqdm

from src.config.config_defaults import (
    ALL_INSTRUMENTS,
    INSTRUMENT_TO_FULLNAME,
    InstrumentEnums,
    get_default_config,
)
from src.utils.utils_dataset import encode_instruments

config = get_default_config()


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Irmas_Val_split",
        description="Splits IRMAS test dataset in 2 datasets based on split",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Path to IRMAS-like test directory.",
        default="data/irmas/test",
    )
    parser.add_argument(
        "--frac",
        type=float,
        help="fraction of the entire dataset to be stored as test.",
        default=0.5,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="fraction of the entire dataset to be stored as test.",
        default=config.path_data,
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        help="fraction of the entire dataset to be stored as test.",
        default=config.path_figures,
    )
    args = parser.parse_args()
    return args


AUDIO_NAME_KEY = "audio_name"
WAVEFORM_PATH_KEY = "waveform_path"
LABEL_PATH_KEY = "label_path"
LABELS_KEY = "label"
SONG_KEY = "song"
OUTPUT_PATH_KEY = "file"

all_instruments_name = [INSTRUMENT_TO_FULLNAME[i] for i in ALL_INSTRUMENTS]


def make_song_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Dataframe which contains songs and their labels caculated as union of labels which are a
    part of one song."""
    song_dict = {SONG_KEY: [], LABELS_KEY: []}
    for group, data in df.groupby(SONG_KEY):
        # [0, 1, 1] + [0, 0, 1]
        # [0, 1, 2]
        # [0, 1, 1]
        label = np.sum(data[ALL_INSTRUMENTS].values, axis=0).astype(bool).astype(int)
        song_dict[SONG_KEY].append(group)
        song_dict[LABELS_KEY].append(label)
    song_df = pd.DataFrame().from_dict(song_dict)
    return song_df


def make_dataframe(data_dir: Path) -> pd.DataFrame:
    dataframe_columns = [
        AUDIO_NAME_KEY,
        WAVEFORM_PATH_KEY,
        LABEL_PATH_KEY,
        LABELS_KEY,
        SONG_KEY,
    ] + ALL_INSTRUMENTS

    data_dict = {col: [] for col in dataframe_columns}
    waveform_paths = data_dir.rglob("*.wav")

    for waveform_path in tqdm(waveform_paths):
        audio_name = waveform_path.stem
        label_path = Path(os.path.splitext(waveform_path)[0] + ".txt")
        if not label_path.is_file():
            raise Exception(
                f"Path {str(label_path)} does not exist for audio {str(waveform_path)}"
            )

        waveform_instruments = []
        with open(label_path) as f:
            for line in f:
                instrument = line.rstrip("\n").replace("\t", "")
                waveform_instruments.append(instrument)
        song_name = re.sub(r"-\d+$", "", audio_name)  # Track 01-1 => Track 01

        data_dict[SONG_KEY].append(song_name)
        data_dict[AUDIO_NAME_KEY].append(audio_name)
        data_dict[WAVEFORM_PATH_KEY].append(str(waveform_path))
        data_dict[LABEL_PATH_KEY].append(label_path)
        data_dict[LABELS_KEY].append(encode_instruments(waveform_instruments))
        for instrument in ALL_INSTRUMENTS:
            is_instrument_present = 1 if instrument in waveform_instruments else 0
            data_dict[instrument].append(is_instrument_present)

    dataframe = pd.DataFrame.from_dict(data_dict)
    return dataframe


def main():
    args = parse_args()
    data_dir, frac = args.data_dir, args.frac

    waveform_df = make_dataframe(data_dir)
    song_df = make_song_dataframe(waveform_df)

    song_labels = np.array(song_df[LABELS_KEY].values.tolist())
    fake_X = np.array(range(len(song_labels)))[..., np.newaxis]

    train_indices, _, test_indices, _ = iterative_train_test_split(
        fake_X, song_labels, test_size=frac
    )

    train_indices, test_indices = train_indices.ravel(), test_indices.ravel()

    train_songs = song_df.loc[train_indices, SONG_KEY].values
    test_songs = song_df.loc[test_indices, SONG_KEY].values

    train_mask = waveform_df[SONG_KEY].isin(train_songs)
    test_mask = waveform_df[SONG_KEY].isin(test_songs)

    train_df = waveform_df[train_mask]
    test_df = waveform_df[test_mask]

    train_df = train_df.loc[:, [WAVEFORM_PATH_KEY] + ALL_INSTRUMENTS]
    test_df = test_df.loc[:, [WAVEFORM_PATH_KEY] + ALL_INSTRUMENTS]
    train_df = train_df.rename(columns={WAVEFORM_PATH_KEY: OUTPUT_PATH_KEY})
    test_df = test_df.rename(columns={WAVEFORM_PATH_KEY: OUTPUT_PATH_KEY})

    assert not set(train_df[OUTPUT_PATH_KEY]).intersection(
        set(test_df[OUTPUT_PATH_KEY])
    )
    assert len(train_df) + len(test_df) == len(waveform_df)

    train_df_path = Path(args.out_dir, f"irmas_train_from_test_n_{len(train_df)}.csv")

    test_df_path = Path(args.out_dir, f"irmas_test_from_test_n_{len(test_df)}.csv")

    print("Saving file:", str(train_df_path))
    train_df.to_csv(train_df_path, index=False)
    print("Saving file:", str(test_df_path))
    test_df.to_csv(test_df_path, index=False)

    # ========== PLOT ==========

    # Calculate the class frequencies for each dataset
    train_freq = train_df[ALL_INSTRUMENTS].sum()
    test_freq = test_df[ALL_INSTRUMENTS].sum()

    bar_width = 0.4

    # Set the positions of the bars on the x-axis
    train_pos = np.arange(len(ALL_INSTRUMENTS))
    test_pos = train_pos + bar_width

    # Plot the histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    bar_train = ax.bar(
        train_pos,
        train_freq,
        width=bar_width,
        color="blue",
        label=f"Train n={len(train_df)}",
    )
    bar_test = ax.bar(
        test_pos,
        test_freq,
        width=bar_width,
        color="orange",
        label=f"Test n={len(test_df)}",
    )

    for rect in bar_train + bar_test:
        height = rect.get_height()
        plt.text(
            rect.get_x() + rect.get_width() / 2.0,
            height,
            f"{height:.0f}",
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Class")
    ax.set_ylabel("Frequency")
    ax.set_title("Instrument frequency for IRMAS test split")
    ax.set_xticks(train_pos + bar_width / 2)
    ax.set_xticklabels(all_instruments_name, fontsize=10, rotation=45, ha="right")
    ax.legend()

    # Save plot to file
    plot_path = str(Path(args.fig_dir, "irmas_test_internal_split"))
    print("Saving plot:", str(plot_path) + ".png")
    plt.tight_layout()
    plt.savefig(plot_path + ".png", dpi=150)
    plt.savefig(plot_path + ".svg")


if __name__ == "__main__":
    main()
