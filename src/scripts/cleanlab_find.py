import json
import os
from pathlib import Path

import cleanlab
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm

import src.config.config_defaults as config_defaults


def main():
    config = config_defaults.get_default_config()

    EMBEDDINGS_DIR = config.path_irmas_train_features
    TRAIN_DATASET_PATH = config.path_irmas_train

    embeddings = []
    labels = []
    file_paths = []
    for i, json_path in tqdm(
        enumerate(Path(EMBEDDINGS_DIR, "embeddings").glob("*.json"))
    ):
        item = json.load(open(json_path))
        file_path = item["sample_path"]
        idx = item["label"]
        embedding = item["embedding"]
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
    for i in label_issues_indices:
        instrument = config_defaults.IDX_TO_INSTRUMENT[labels[i]]
        full_path = Path(TRAIN_DATASET_PATH, instrument, file_paths[i])
        print()


if __name__ == "__main__":
    main()
