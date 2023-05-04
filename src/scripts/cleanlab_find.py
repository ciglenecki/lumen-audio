import json
from pathlib import Path

import cleanlab
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm

import src.config.config_defaults as config_defaults
from src.config.config_defaults import get_default_config
from utils.utils_dataset import decode_instruments


def main():
    config = get_default_config()

    EMBEDDINGS_DIR = (
        "embeddings/data-irmas-train_ast_MIT-ast-finetuned-audioset-10-10-0.4593"
    )
    TRAIN_DATASET_PATH = config.path_irmas_train
    embeddings = []
    labels = []
    file_paths = []
    one_hot_label = []
    for i, json_path in tqdm(
        enumerate(Path(EMBEDDINGS_DIR, "embeddings").glob("*.json"))
    ):
        item = json.load(open(json_path))
        file_path = item["sample_path"]
        idx = item["label"]
        embedding = item["embedding"]
        one_hot_label.append(decode_instruments(item["instruments"]))
        embeddings.append(embedding)
        labels.append(idx)
        file_paths.append(file_path)

    embeddings_array = np.array(embeddings)
    labels_array = np.array(labels)

    model = LogisticRegression(C=0.01, max_iter=3000, tol=1e-1, random_state=42)

    # can decrease this value to reduce runtime, or increase it to get better results
    num_crossval_folds = 10
    pred_probs = cross_val_predict(
        estimator=model,
        X=embeddings_array,
        y=labels_array,
        cv=num_crossval_folds,
        method="predict_proba",
        verbose=True,
    )

    predicted_labels = pred_probs.argmax(axis=1)
    cv_accuracy = accuracy_score(labels_array, predicted_labels)
    print(f"Cross-validated estimate of accuracy on held-out data: {cv_accuracy}")

    label_issues_indices = cleanlab.filter.find_label_issues(
        labels=labels_array,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    # Print bad files
    # TODO: finish writing bad files to a csv
    bad_files = {}
    for i in label_issues_indices:
        instrument = config_defaults.IDX_TO_INSTRUMENT[labels[i]]
        full_path = str(Path(TRAIN_DATASET_PATH, instrument, file_paths[i]))
        bad_files.add(full_path)

    labels = []
    file_paths = []


if __name__ == "__main__":
    main()
