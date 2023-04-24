import json
import re
from itertools import chain
from pathlib import Path

import pandas as pd
import pyrootutils

from src.config.config_defaults import (
    DrumKeys,
    GenreKeys,
    InstrumentEnums,
    get_default_config,
)


def add_square_brackets_to_important_keys(string, important_keys):
    for s in important_keys:
        string = string.replace(s, f"[{s}]")
    return string


def add_class_prefix(string, path_irmas_train):
    start = string.find("[")
    end = string.find("]")
    instrument = string[start + 1 : end]
    return str(Path(path_irmas_train, instrument, string))


def main():
    config = get_default_config()
    file_path = Path(config.path_irmas_sample, "random_samples_relabeling.csv")
    out_path = Path(
        config.path_irmas_sample, "normalized_random_samples_relabeling.csv"
    )
    root_project = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root"
    )

    important_keys = {e.value for e in chain(InstrumentEnums, GenreKeys, DrumKeys)}
    important_keys.add("---")

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
        lambda x, important_keys=important_keys: add_square_brackets_to_important_keys(
            x, important_keys
        )
    )
    df["filename"] = df["filename"].apply(
        lambda x, path=config.path_irmas_train: add_class_prefix(x, path)
    )

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
    columns_to_keep = ["filename"] + list(sorted(instruments))
    df = df[columns_to_keep]

    df.to_csv(out_path, index=False)
    print("Saved to file:", out_path)


if __name__ == "__main__":
    main()
