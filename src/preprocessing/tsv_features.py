import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import src.config.defaults as defaults

OUT_DIR = embeddings_dir = defaults.PATH_IRMAS_TRAIN_FEATURES

embeddings = None
labels = []
sample_paths = []
name = []
instrument_names = []


def get_fields(item):
    embedding = item["embedding"]
    sample_path = item["sample_path"]
    label = item["label"]
    instrument = item["instrument"]
    instrument_name = item["instrument_name"]
    return embedding, sample_path, label, instrument, instrument_name


for i, json_path in tqdm(enumerate(embeddings_dir.glob("*.json"))):
    item = json.load(open(json_path))
    (
        embedding,
        sample_path,
        label,
        instrument,
        instrument_name,
    ) = get_fields(item)

    # Create embeddings dict if it isn't created yet
    if embeddings is None:
        embeddings = {f"e{i}": [] for i in range(len(embedding))}

    # Save each embedding
    for e_idx, e in enumerate(embedding):
        embeddings[f"e{e_idx}"].append(e)

    instrument_names.append(instrument_name)
    sample_paths.append(sample_path)

dict = {
    **embeddings,
}

dict_metadata = {
    "instrument": instrument_names,
    "path": sample_paths,
}

df = pd.DataFrame.from_dict(dict)
df_metadata = pd.DataFrame.from_dict(dict_metadata)

df_path = Path(OUT_DIR, "ast_embeddings.tsv")
df_meta_path = Path(OUT_DIR, "ast_embeddings_meta.tsv")

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
