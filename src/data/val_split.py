import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm

from src.enums.enums import InstrumentEnums

parser = argparse.ArgumentParser(
    prog="Irmas_Val_split",
    description="Splits IRMAS test dataset in 2 datasets based on split",
    epilog="",
)

parser.add_argument("--data-dir", type=Path, help="Path to IRMAS-like test directory.")
parser.add_argument(
    "--frac",
    type=float,
    help="fraction of the entire dataset to be stored as test.",
    default=0.5,
)

instruments = [e.value for e in InstrumentEnums]

dataframe_columns = [
    "track_name",  # column used only for song grouping
    "file",
    "txt_file",
] + instruments


def get_track_labels(dataframe):
    for i in range(dataframe.shape[0]):
        with open(dataframe.iloc[i]["txt_file"]) as f:
            for line in f:
                instrument = line.rstrip("\n").replace("\t", "")
                dataframe[instrument].iat[i] = 1
    return dataframe


def get_track_name(track):
    track = str(track).strip("irmas/test/").strip(".wav")
    track_name = "".join(c for c in track if not c.isnumeric())
    return track_name


def get_wav_list(data_dir):
    song_list = list(data_dir.rglob("*.wav"))
    return sorted(song_list)


def get_txt_list(data_dir):
    txt_list = list(data_dir.rglob("*.txt"))
    return sorted(txt_list)


def make_dataframe(data_dir):
    data_dict = {col: [] for col in dataframe_columns}
    songs = get_wav_list(data_dir)
    txt_list = get_txt_list(data_dir)
    for song, text in tqdm(zip(songs, txt_list)):
        data_dict["track_name"].append(get_track_name(song))
        data_dict["file"].append(song)
        data_dict["txt_file"].append(text)
        for instrument in instruments:
            data_dict[instrument].append(0)

    dataframe = pd.DataFrame.from_dict(data_dict)
    dataframe = get_track_labels(dataframe)
    return dataframe


def get_song_dataframe(df):
    song_dict = {"song": [], "labels": []}
    for group, data in df.groupby("track_name"):
        song_dict["song"].append(group)
        song_dict["labels"].append(
            np.sum(data[instruments].values, axis=0).astype(bool).astype(int)
        )
    song_df = pd.DataFrame().from_dict(song_dict)
    return song_df


def get_idx_and_y(song_df):
    y = np.array(song_df["labels"].values.tolist())
    idx = np.array(range(len(y)))
    idx = np.expand_dims(idx, 1)
    return idx, y


def train_test_split(idx, y, test_frac=0.5):
    idx_train, y_train, idx_test, y_test = iterative_train_test_split(
        idx, y, test_size=test_frac
    )
    return idx_train.ravel(), idx_test.ravel()


def get_train_test_dataframes(full_dataframe, song_dataframe, idx_train, idx_test):
    """
    Args:
    full_dataframe: dataframe conaining all .wav and .txt files, returned by make_dataframe()
    song_dataframe: dataframe containing song names and their instruments, returned by get_song_dataframe()
    idx_train,idx_test: returned by train_test_split(), which songs should go into train/test set
    Returns:
    train_df,test_df
    """
    train_indices = []
    test_indices = []

    train_songs = song_dataframe.loc[idx_train, "song"].values
    test_songs = song_dataframe.loc[idx_test, "song"].values
    for i in range(full_dataframe.shape[0]):
        if full_dataframe["track_name"].iloc[i] in train_songs:
            train_indices.append(i)
        if full_dataframe["track_name"].iloc[i] in test_songs:
            test_indices.append(i)

    train_df = full_dataframe.iloc[train_indices].drop(columns="track_name")
    test_df = full_dataframe.iloc[test_indices].drop(columns="track_name")
    return train_df, test_df


def main(data_dir: Path, frac: float):
    args = parser.parse_args()
    data_dir, frac = args.data_dir, args.frac

    full_df = make_dataframe(data_dir)
    get_track_labels(full_df)
    song_df = get_song_dataframe(full_df)
    idx, y = get_idx_and_y(song_df)
    train_idx, test_idx = train_test_split(idx, y, test_frac=frac)
    train_df, test_df = get_train_test_dataframes(
        full_dataframe=full_df,
        song_dataframe=song_df,
        idx_train=train_idx,
        idx_test=test_idx,
    )
    train_df.to_csv("train_tracks.csv", index=False)
    test_df.to_csv("test_tracks.csv", index=False)
    return train_df, test_df


if __name__ == "__main__":
    main()
