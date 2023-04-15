"""Export AST features for the whole dataset.

Save features to .npy file along with the instrument label in .label file
"""
import json
import os
from pathlib import Path

import torch
import torch.utils.data
from tqdm import tqdm
from transformers import ASTConfig, ASTModel

import src.config.config_defaults as config_defaults
from src.data.dataset_irmas import IRMASDatasetTrain
from src.features.audio_to_ast import AudioTransformAST

device = "cuda" if torch.cuda.is_available() else "cpu"
current_working_dir = os.getcwd()


def main():
    config = config_defaults.get_default_config()
    BATCH_SIZE = 1
    OUTPUT_DIR = config.path_irmas_train_features
    OUTPUT_DIR.mkdir(exist_ok=True)
    # Create model and transform
    model_name = config.TAG_AST_AUDIOSET
    ast_conf = ASTConfig.from_pretrained(pretrained_model_name_or_path=model_name)
    model = ASTModel.from_pretrained(
        model_name, config=ast_conf, ignore_mismatched_sizes=True
    )

    model.eval()
    model = model.to(device)

    audio_transform = AudioTransformAST(
        sampling_rate=config.DEFAULT_SAMPLING_RATE,
        pretrained_tag=config.TAG_AST_AUDIOSET,
        augmentation_enums=[],
    )

    # Create dataset
    train_dataset = IRMASDatasetTrain(audio_transform=audio_transform)
    training_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
    )

    print("Saving embeddings to: ", str(OUTPUT_DIR))
    for data in tqdm(training_loader, total=len(training_loader)):
        spectrogram, onehot_labels, audio_paths = data
        spectrogram, onehot_labels = spectrogram.to(device), onehot_labels.to(device)
        labels = torch.argmax(onehot_labels, dim=-1)  # extract label number

        ast_embeddings = model.forward(
            spectrogram,
            output_attentions=False,
            return_dict=True,
        )
        ast_embeddings_list = ast_embeddings.pooler_output.detach().cpu().tolist()

        # Iterate over each file from the batch
        for audio_path, ast_embedding, label in zip(
            audio_paths, ast_embeddings_list, labels
        ):
            stem = Path(audio_path).stem  # e.g. [cel][cla]0001__1
            audio_path = str(Path(audio_path).relative_to(current_working_dir))
            instrument_idx = int(label)
            instrument = config.IDX_TO_INSTRUMENT[instrument_idx]
            instrument_name = config.INSTRUMENT_TO_FULLNAME[instrument]

            json_item = dict(
                sample_path=audio_path,
                label=instrument_idx,
                instrument=instrument,
                instrument_name=instrument_name,
                embedding=ast_embedding,
            )

            json_name = Path(OUTPUT_DIR, f"{stem}.json")

            with open(json_name, "w") as fp:
                json.dump(json_item, fp)


if __name__ == "__main__":
    main()
