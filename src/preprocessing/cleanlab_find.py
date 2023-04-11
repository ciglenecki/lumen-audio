import os
from pathlib import Path

import cleanlab
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm

from src.config.defaults import (
    IDX_TO_INSTRUMENT,
    PATH_IRMAS_TRAIN,
    PATH_IRMAS_TRAIN_FEATURES,
)

EMBEDDINGS_DIR = PATH_IRMAS_TRAIN_FEATURES
TRAIN_DATASET_PATH = PATH_IRMAS_TRAIN

embeddings = []
labels = []
filepaths = []

for i, npy_path in tqdm(enumerate(EMBEDDINGS_DIR.glob("*.npy"))):
    path_no_ext, _ = os.path.splitext(npy_path)
    label_path = Path(path_no_ext + ".label")
    npy_stem = Path(npy_path).stem
    embeddings.append(np.load(npy_path, allow_pickle=True)[0])
    labels.append(int(label_path.read_text()))
    filepaths.append(npy_path)

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
    instrument = IDX_TO_INSTRUMENT[labels[i]]
    full_path = Path(TRAIN_DATASET_PATH, instrument, filepaths[i])
    print()
