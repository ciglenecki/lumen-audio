"""python3 src/scripts/tsv_features.py --input-dir embeddings/astMIT-ast-finetuned-
audioset-10-10-0.4593_irmas_train/"""

import bisect
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config.argparse_with_config import ArgParseWithConfig
from src.config.config_defaults import INSTRUMENT_TO_FULLNAME, get_default_config


def parse_args():
    parser = ArgParseWithConfig()
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory which has a 'embeddings' subdirectory which contains jsons.",
    )
    args, _, _ = parser.parse_args()

    assert os.path.isdir(args.input_dir), "Input directory should exist."
    return args, _, _


def main():
    args, _, _ = parse_args()
    input_dir = Path(args.input_dir)
    embeddings_dir = Path(input_dir, "embeddings")
    assert os.path.isdir(
        args.input_dir
    ), "Please provide --input-dir directory which has an 'embeddings' directory inside of it. 'embeddings' directory contains .json files."

    out_dir = input_dir

    embeddings = None
    sample_paths = []
    all_instrument_names = []

    def get_fields(item):
        embedding = item["embedding"]
        sample_path = item["sample_path"]
        indices = item["indices"]
        instruments = item["instruments"]
        instrument_names = item["instrument_names"]
        return embedding, sample_path, indices, instruments, instrument_names

    embeddings = {}
    for _, json_path in tqdm(enumerate(embeddings_dir.glob("*.json"))):
        item = json.load(open(json_path))
        (
            embedding,
            sample_path,
            indices,
            instruments,
            instrument_names,
        ) = get_fields(item)

        # Create embeddings dict if it isn't created yet
        if not embeddings:
            embeddings = {f"e{i}": [] for i in range(len(embedding))}

        # Save each embedding
        for e_idx, e in enumerate(embedding):
            embeddings[f"e{e_idx}"].append(e)

        # WARNING: ADDING ONLY ONE/FIRST INSTRUMENT!
        all_instrument_names.append(instrument_names[0])
        sample_paths.append(sample_path)

    dict = {
        **embeddings,
    }

    dict_metadata = {
        "instrument": all_instrument_names,
        "path": sample_paths,
    }

    df = pd.DataFrame.from_dict(dict)
    df_metadata = pd.DataFrame.from_dict(dict_metadata)

    df_path = Path(out_dir, "embeddings.tsv")
    df_meta_path = Path(out_dir, "embeddings_meta.tsv")

    df.to_csv(
        df_path,
        sep="\t",
        header=False,
        index=False,
    )
    print("Saved file:", str(df_path))

    df_metadata.to_csv(
        df_meta_path,
        sep="\t",
        index=False,
    )
    print("Saved file:", str(df_meta_path))


if __name__ == "__main__":
    main()
