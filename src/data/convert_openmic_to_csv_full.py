"""
Requirements:
- IRMAS train embeddings (guitar)
- OpenMIC guitar embeddings
- OpenMIC original CSV

Does:
- Trains SVM on IRMAS' guitar embeddings
- predicts the class for OpenMIC embeddings and sets the class "guitar" to either electric guitar or acoustic guitar
- reassignes the guitar labels
- creates appropriate csv that can be used for training
- plots frequency of each instrument

"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config.config_defaults import (
    ALL_INSTRUMENTS,
    ALL_INSTRUMENTS_NAMES,
    InstrumentEnums,
    get_default_config,
)
from src.utils.utils_exceptions import InvalidDataException

config = get_default_config()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--svm-confidence",
        type=float,
        default=0.7,
        help="Minimal SVM confidence for classification",
    )
    parser.add_argument(
        "--relevance",
        type=float,
        default=0.8,
        help="Minimal OpenMIC relevance filter",
    )

    parser.add_argument(
        "--irmas-emb-dir",
        type=Path,
        help="IRMAS embeddings which will be used to train SVM.",
        default=Path(config.path_irmas_train_features, "embeddings"),
    )

    parser.add_argument(
        "--openmic-csv",
        type=Path,
        help="Path to openmic csv with (sample_key, instrument...)",
        default=Path(
            config.path_openmic,
            "openmic-2018-aggregated-labels.csv",
        ),
    )
    parser.add_argument(
        "--openmic-emb-dir",
        type=Path,
        help="Path to openmic embeddings.",
        default=Path(
            config.path_embeddings,
            "data-openmic-guitars_only_n_1144_relevance_0.8.csv_ast_MIT-ast-finetuned-audioset-10-10-0.4593",
            "embeddings",
        ),
    )

    args = parser.parse_args()

    if not args.openmic_csv.is_file():
        raise InvalidDataException(
            f"{args.openmic_csv} is not a file. Please run python3 data/convert_openmic_guitras_to_csv.py to generate a OpenMIC guitars only csv."
        )
    if len(os.listdir(args.openmic)) == 0:
        raise InvalidDataException(
            f"{args.openmic} is emptyPlease generate IRMAS train embeddings which will be used to train a SVM for guitars. Run python3 --model AST --audio-transform AST --pretrained-tag MIT/ast-finetuned-audioset-10-10-0.4593 --dataset-paths inference:data/openmic/guitars_only_n_1144_relevance_0.8.csv --batch-size 1 --num-workers 1"
        )
    if len(os.listdir(args.irmas_emb_dir)) == 0:
        raise InvalidDataException(
            f"{args.irmas_emb_dir} Please generate IRMAS train embeddings which will be used to train a SVM for guitars. Run python3 --model AST --audio-transform AST --pretrained-tag MIT/ast-finetuned-audioset-10-10-0.4593 --dataset-paths irmastrain:data/irmas/train --batch-size 1 --num-workers 1"
        )
    return args


irmas_to_opemic = {
    InstrumentEnums.CELLO.value: "cello",
    InstrumentEnums.CLARINET.value: "clarinet",
    InstrumentEnums.FLUTE.value: "flute",
    InstrumentEnums.ACOUSTIC_GUITAR.value: InstrumentEnums.ACOUSTIC_GUITAR.value,
    InstrumentEnums.ELECTRIC_GUITAR.value: InstrumentEnums.ELECTRIC_GUITAR.value,
    InstrumentEnums.ORGAN.value: "organ",
    InstrumentEnums.PIANO.value: "piano",
    InstrumentEnums.SAXOPHONE.value: "saxophone",
    InstrumentEnums.TRUMPET.value: "trumpet",
    InstrumentEnums.VIOLIN.value: "violin",
    InstrumentEnums.VOICE.value: "voice",
}
openmic_to_irmas = {v: k for k, v in irmas_to_opemic.items()}
openmic_to_irmas.update({"guitar": None})


def create_filename_from_sample_key(sample_key):
    subdir = sample_key[:3]
    filepath = Path(config.path_openmic, "audio", subdir, sample_key + ".ogg")
    filepath = str(filepath)
    return filepath


def openmic_to_irmas_labels(sample_key: str):
    if sample_key not in openmic_to_irmas:
        return None
    return openmic_to_irmas[sample_key]


def main():
    args = parse_args()

    openmic_embeddings = []
    openmic_audio_names = []

    for json_path in args.openmic_emb_dir.rglob("*.json"):
        with open(json_path) as file:
            json_item = json.load(file)
            embedding = json_item["embedding"]
            audio_name = Path(json_path).stem
            openmic_embeddings.append(embedding)
            openmic_audio_names.append(audio_name)

    y_list = []
    irmas_embs = []
    for json_path in args.irmas_emb_dir.rglob("*.json"):
        json_item = json.load(open(json_path))
        file_instruments = json_item["instruments"]
        embedding = json_item["embedding"]

        is_acoustic = InstrumentEnums.ACOUSTIC_GUITAR.value in file_instruments
        is_electric = InstrumentEnums.ELECTRIC_GUITAR.value in file_instruments
        if is_acoustic and is_electric or (not is_acoustic and not is_electric):
            continue
        idx = 0 if is_acoustic else 1

        irmas_embs.append(embedding)
        y_list.append(idx)

    X = np.array(irmas_embs)
    y = np.array(y_list)

    model = make_pipeline(
        StandardScaler(),
        SVC(gamma="auto", C=1, verbose=True, probability=True, kernel="rbf"),
    )
    model.fit(X, y)
    filename = "irmas_two_class_guitars.pkl"

    x_openmic = np.array(openmic_embeddings)
    y_pred = model.predict(x_openmic).astype(int)
    y_pred_prob = model.predict_proba(x_openmic)
    confidence = y_pred_prob[np.arange(len(y_pred)), y_pred]
    conf_mask = confidence > args.svm_confidence
    y_pred_high_confidence = y_pred[conf_mask]

    print("Before:")
    print("Number of audio (guitars)", len(y_pred))
    print("Fraction of electric guitras", y_pred.sum() / len(y_pred))

    print("After:")
    print("Number of audio (guitars)", len(y_pred_high_confidence))
    print(
        "Fraction of electric guitras",
        y_pred_high_confidence.sum() / len(y_pred_high_confidence),
    )

    print(
        f"Keeping {len(y_pred_high_confidence) / len(y_pred) * 100}% ({len(y_pred_high_confidence)}/{len(y_pred)}), diff: {len(y_pred_high_confidence) - len(y_pred)} of audio samples based on SVM confidence"
    )
    filtered_files = {}
    for i, file in enumerate(openmic_audio_names):
        if conf_mask[i] == 1:
            filtered_files[file] = y_pred[i]

    df = pd.read_csv(args.openmic_csv)

    for file_name, class_idx in filtered_files.items():
        guitar_name = (
            InstrumentEnums.ACOUSTIC_GUITAR.value
            if class_idx == 0
            else InstrumentEnums.ELECTRIC_GUITAR.value
        )
        df.loc[df["sample_key"] == file_name, "instrument"] = guitar_name

    # Create file names, and drop low relevance rows
    df["file"] = df["sample_key"].apply(create_filename_from_sample_key)

    # Change instruments to IRMAS naming and drop the rest of instruments
    df["instrument"] = df["instrument"].apply(openmic_to_irmas_labels)
    df = df[df["instrument"].notna()]

    # Drop low OpenMIC relevance data
    df = df.loc[df["relevance"] > args.relevance, :]

    # One hot encoding
    df = df.loc[:, df.columns.isin(["file", "instrument"])]
    one_hots = pd.get_dummies(df["instrument"])
    df = pd.concat([df, one_hots], axis=1)

    # Onehot to int and sort columns
    df.drop(["instrument"], axis=1, inplace=True)
    instrument_cols = ~df.columns.isin(["file"])
    df.loc[:, instrument_cols] = df.loc[:, instrument_cols].astype(int)

    # Sort instrument columns
    df.loc[:, instrument_cols] = df.loc[:, instrument_cols].reindex(
        sorted(df.columns), axis=1
    )

    df_out_path = Path(config.path_data, f"openmic_r{args.relevance}_n_{len(df)}.csv")

    df.to_csv(df_out_path, index=False)
    print("Saved file to:", df_out_path)

    # ========== PLOT ==========

    # Calculate the class frequencies for each dataset
    freq = df[ALL_INSTRUMENTS].sum()

    bar_width = 0.4

    # Set the positions of the bars on the x-axis
    pos = np.arange(len(ALL_INSTRUMENTS))
    pos = pos + bar_width

    # Plot the histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    bar = ax.bar(
        pos,
        freq,
        width=bar_width,
        color="blue",
        label=f"n={len(df)}",
    )

    for rect in bar:
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
    ax.set_title(
        f"Instrument frequency for OpenMIC dataset filtered with relevance > {args.relevance}"
    )
    ax.set_xticks(pos + bar_width / 2)
    ax.set_xticklabels(ALL_INSTRUMENTS_NAMES, fontsize=10, rotation=45, ha="right")
    ax.legend()

    # Save plot to file
    plot_path = str(Path(config.path_figures, "openmic_filtered_dataset"))
    print("Saving plot:", str(plot_path) + ".png")
    plt.tight_layout()
    plt.savefig(plot_path + ".png", dpi=150)
    plt.savefig(plot_path + ".svg")


if __name__ == "__main__":
    main()
