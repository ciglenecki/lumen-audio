import json
import os
import re
from itertools import chain
from pathlib import Path

import pandas as pd

from src.config.config_defaults import (
    PATH_IRMAS_SAMPLE,
    PATH_IRMAS_TRAIN,
    DrumKeys,
    GenreKeys,
    InstrumentEnums,
)

file_path = Path(PATH_IRMAS_SAMPLE, "random_samples_relabeling.csv")
out_path = Path(PATH_IRMAS_SAMPLE, "normalized_random_samples_relabeling.csv")


def add_square_brackets_to_important_keys(string):
    for s in important_keys:
        string = string.replace(s, f"[{s}]")
    return string


def add_class_prefix(string):
    start = string.find("[")
    end = string.find("]")
    instrument = string[start + 1 : end]
    return str(Path(PATH_IRMAS_TRAIN, instrument, string).relative_to(os.getcwd()))


important_keys = {e.value for e in chain(InstrumentEnums, GenreKeys, DrumKeys, ["---"])}

instrument_map = {
    "Piano": "pia",
    "Saxophone": "sax",
    "Clarinet": "cla",
    "Flute": "flu",
    "Voice": "voi",
    "Electric guitar": "gel",
    "Acoustic guitar": "gac",
    "Trumpet": "tru",
    "Organ": "org",
    "Violin": "vio",
    "Cello": "cel",
}


df = pd.read_csv(file_path, delimiter=",,")


# Select and rename columns
df = df.loc[:, ["url", "label"]]
df.rename(columns={"url": "filename"}, inplace=True)

# Strip URL to get filename, remove weird prefix, retrieve brackets
df["filename"] = df["filename"].apply(lambda x: x.split("/")[-1])
df["filename"] = df["filename"].apply(lambda x: re.sub(r"^\w+-", "", x))
df["filename"] = df["filename"].apply(
    lambda x: add_square_brackets_to_important_keys(x)
)
df["filename"] = df["filename"].apply(lambda x: add_class_prefix(x))

# Convert "choices" column into separate columns for each instrument
instruments = set()
for index, row in df.iterrows():
    label = row["label"]
    if label.startswith("{"):
        label = json.loads(label)
        choices = label["choices"]
    else:
        choices = [label]
    for instrument in choices:
        instrument_short = instrument_map[instrument]
        instruments.add(instrument_short)
        df.loc[index, instrument_short] = 1

# Fill NaN values with 0
df.fillna(0, inplace=True)

# Keep only the filename and instrument columns
columns_to_keep = ["filename"] + list(instruments)
df = df[columns_to_keep]

df.to_csv(out_path, index=False)
print("Saved to file:", out_path)
