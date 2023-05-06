"""
1. Uses embeddings from a model trained on AST features to predict labels for the IRMAS dataset with a logistic regression model.
2. Uses cleanlab to find the worst examples.
3. Prints the worst examples and their scores.


cutoff 0.13
{'cel': 35, 'cla': 36, 'flu': 45, 'gac': 50, 'gel': 68, 'org': 37, 'pia': 65, 'sax': 40, 'tru': 58, 'vio': 44, 'voi': 19}
Number of examples: 6705
Number of removed examples: 497
Percentage removed: 0.07412378821774795
"""

import json
from pathlib import Path

import cleanlab
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm

import src.config.config_defaults as config_defaults
from src.config.config_defaults import get_default_config
from src.utils.utils_dataset import decode_instruments


def main():
    config = get_default_config()

    EMBEDDINGS_DIR = Path(
        config.path_embeddings,
        "data-irmas-train_ast_MIT-ast-finetuned-audioset-10-10-0.4593",
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
        idx = item["indices"][0]
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

    scores = cleanlab.rank.get_label_quality_scores(
        labels=labels_array,
        pred_probs=pred_probs,
        # confident_joint=0.99,
        adjust_pred_probs=True,
    )
    sorted_indices = np.argsort(scores)
    # sorted_scores = scores[sorted_indices]
    # num_examples = len(sorted_indices)

    # plt.show()

    data = scores

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    fig.set_dpi(100)

    # Create a histogram with 20 bins
    bins = 20

    # Compute the cumulative histogram
    n_cumulative, bins_cumulative, bars_cum = ax.hist(
        data, bins=bins, alpha=1, cumulative=True, label="Cumulative Histogram"
    )
    ax.bar_label(
        bars_cum,
        labels=[f"{int(p / n_cumulative.max() * 100)}%" for p in n_cumulative],
    )

    n, bins, patches = ax.hist(data, bins=bins, alpha=1, label="Histogram")

    # Add labels and a title
    ax.set_xlabel("Label quality score")
    ax.set_ylabel("Number of examples")
    ax.set_title(
        "CleanLab label quality score (Logistic regression trained on AST features)"
    )

    ax.legend()

    fig.savefig(
        Path(
            config.path_figures,
            f"cleanlab_{str(EMBEDDINGS_DIR)[:20].replace('/', '_')}.png",
        )
    )

    # Print bad files
    result: list[tuple[float, str]] = []

    for i in sorted_indices[::-1]:
        score = scores[i]
        instrument = config_defaults.IDX_TO_INSTRUMENT[labels[i]]
        full_path = Path(file_paths[i])
        result.append((score, str(full_path), instrument))
        print(f"{score:.4f}", str(full_path))

    cutoff = input("Enter a cutoff score:")

    class_dict = {e.value: 0 for e in config_defaults.InstrumentEnums}
    num_examples = 0
    for score, path, instrument in result:
        if score < float(cutoff):
            class_dict[instrument] += 1
            print(path)
            num_examples += 1
    print("Cut off:", cutoff)
    print(class_dict)
    print("Number of examples:", len(result))
    print("Number of removed examples:", num_examples)
    print("Number of remaining examples:", len(result) - num_examples)
    print("Percentage removed:", num_examples / len(result))


if __name__ == "__main__":
    main()
