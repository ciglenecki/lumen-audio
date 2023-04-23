import argparse
import json
import os
import pickle
import re
from math import e
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Audio
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skmultilearn.model_selection.iterative_stratification import (
    iterative_train_test_split,
)
from torch import embedding
from tqdm import tqdm

from src.config.config_defaults import (
    INSTRUMENT_TO_FULLNAME,
    InstrumentEnums,
    get_default_config,
)
from src.utils.utils_dataset import decode_instruments, encode_instruments
from src.utils.utils_exceptions import InvalidDataException

config = get_default_config()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--irmas-embeddings",
        type=float,
        help="fraction of the entire dataset to be stored as test.",
        default=0.5,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Openmic directory path.",
        default=config.path_openmic,
    )
    parser.add_argument(
        "--load-model",
        type=Path,
        help="Load to trained SVM.",
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
    parser.add_argument(
        "--irmas-train-embeddings",
        type=Path,
        help="fraction of the entire dataset to be stored as test.",
        default=Path(config.path_irmas_train_features, "embeddings"),
    )
    parser.add_argument(
        "--openmic-guitars",
        type=Path,
        help="Path to openmic guitars only csv.",
        default=Path(config.path_openmic, "guitars_only_n_1144_relevance_0.8.csv"),
    )
    parser.add_argument(
        "--openmic-guitar-embeddings",
        type=Path,
        help="Path to openmic guitar embeddings.",
        default=Path(
            config.path_embeddings,
            "data-openmic-guitars_only_n_1144_relevance_0.8.csv_ast_MIT-ast-finetuned-audioset-10-10-0.4593",
        ),
    )
    args = parser.parse_args()
    if not args.openmic_guitars.isfile():
        raise InvalidDataException(
            "Please run python3 data/convert_openmic_guitras_to_csv.py to generate a OpenMIC guitars only csv."
        )
    return args


def main():
    args, unknown_args = parse_args()
    irmas_train_embeddings: Path = args.irmas_train_embeddings
    openmic_guitar_embeddings: Path = args.openmic_guitar_embeddings
    OPENMIC_EMBEDDINGS_KEY = "openmic_embeddings"
    OPENMIC_AUDIO_NAMES_KEY = "openmic_audio_names"

    store = {
        InstrumentEnums.ACOUSTIC_GUITAR: [],
        InstrumentEnums.ELECTRIC_GUITAR: [],
        OPENMIC_EMBEDDINGS_KEY: [],
        OPENMIC_AUDIO_NAMES_KEY: [],
    }

    for json_path in openmic_guitar_embeddings.rglob(".json"):
        with open(json_path) as file:
            json_item = json.load(file)
            embedding = json_item["embedding"]
            store[OPENMIC_EMBEDDINGS_KEY].append(embedding)
            store[OPENMIC_AUDIO_NAMES_KEY].append(Path(json_path).stem)

    if args.load_model:
        model = pickle.load(open(args.load_model, "rb"))
    else:
        for json_path in irmas_train_embeddings.rglob(".json"):
            with open(json_path) as file:
                json_item = json.load(file)
                instruments = json_item["instrument"]
                embedding = json_item["embedding"]
                for instrument in instruments:
                    if instrument in store:
                        store[instrument].append(embedding)

        num_acoustic = len(store[InstrumentEnums.ACOUSTIC_GUITAR])
        # num_electric = len(store[InstrumentEnums.ELECTRIC_GUITAR])
        X = np.array(
            [
                store[InstrumentEnums.ACOUSTIC_GUITAR]
                + store[InstrumentEnums.ELECTRIC_GUITAR],
            ]
        )

        y = np.zeros(len(X))  # 0 <- ACOUSTIC_GUITAR
        y[num_acoustic:] = 1  # 1 <- ELECTRIC_GUITAR
        model = make_pipeline(StandardScaler(), SVC(gamma="auto"))
        model.fit(X, y)
        filename = "svm_guitars.pkl"
        pickle.dump(model, open(Path(config.path_models, filename), "wb"))

    x_openmic = np.array(store[OPENMIC_EMBEDDINGS_KEY])
    model.predict(x_openmic)


if __name__ == "__main__":
    main()
