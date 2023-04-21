"""python3 src/scripts/save_embeddings.py --model AST --audio-transform AST --pretrained-tag
MIT/ast-finetuned-audioset-10-10-0.4593 --train-paths irmas:data/irmas/train --batch-size 1.

python3 src/scripts/save_embeddings.py  --model EFFICIENT_NET_V2_S --audio-transform MEL_SPECTROGRAM

python3 src/scripts/save_embeddings.py --checkpoint models/04-13-14-20-12_GoodSinisa_ast/checkpoints/04-13-14-20-12_GoodSinisa_ast_val_acc_0.0000_val_loss_0.6611.ckpt --model AST --audio-transform AST

python3 src/scripts/save_embeddings.py --model EFFICIENT_NET_V2_S --audio-transform MEL_SPECTROGRAM --pretrained-tag IMAGENET1K_V1
"""
import bisect
import json
from pathlib import Path

import torch
import torch_scatter
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from tqdm import tqdm

import src.config.config_defaults as config_defaults
from src.config.argparse_with_config import ArgParseWithConfig
from src.data.datamodule import IRMASDataModule
from src.data.dataset_irmas import IRMASDatasetTest, IRMASDatasetTrain
from src.enums.enums import AudioTransforms, SupportedModels
from src.features.audio_transform import AudioTransformBase, get_audio_transform
from src.features.chunking import get_collate_fn
from src.model.model import get_model, model_constructor_map
from src.utils.utils_dataset import instrument_multihot_to_idx
from src.utils.utils_exceptions import InvalidArgument, UnsupportedModel


def parse_args():
    parser = ArgParseWithConfig()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("embeddings"))
    parser.add_argument("--target-model-layer", type=str)

    args, config, pl_args = parser.parse_args()

    if config.model is None:
        raise InvalidArgument(
            f"Please provide --model {[e.name for e in SupportedModels]}"
        )
    if config.audio_transform is None:
        raise InvalidArgument(
            f"Please provide --audio-transform {[e.name for e in AudioTransforms]}"
        )

    if int(config.pretrained_tag is not None) + int(args.checkpoint is not None) != 1:
        raise InvalidArgument(
            "Please provide either --pretrained-tag or --checkpoint <PATH>"
        )

    config.parse_dataset_paths()
    return args, config, pl_args


def get_feature_extractor(
    model: torch.nn.Module, model_enum: SupportedModels, target_model_layer: str = None
) -> tuple[torch.nn.Module, dict]:
    """Get only the subset of the original model.

    This is usually something that ends with a flattened linaer layer.
    """
    # Different models might have different forward passes
    forward_kwargs = {}

    if model_enum == SupportedModels.AST:
        model = model.backbone.audio_spectrogram_transformer
        forward_kwargs = dict(head_mask=None, output_attentions=False, return_dict=True)
        return model, forward_kwargs

    if model_enum == SupportedModels.WAV2VEC:
        return model.backbone

    if target_model_layer is not None:
        pass
    elif model_enum == SupportedModels.WAV2VEC_CNN:
        target_model_layer = "backbone.conv_layers.6.activation"
    elif model_enum == SupportedModels.EFFICIENT_NET_V2_S:
        target_model_layer = "flatten"
    elif model_enum == SupportedModels.EFFICIENT_NET_V2_M:
        target_model_layer = "flatten"
    elif model_enum == SupportedModels.EFFICIENT_NET_V2_L:
        target_model_layer = "flatten"
    elif model_enum == SupportedModels.RESNEXT50_32X4D:
        target_model_layer = "flatten"
    elif model_enum == SupportedModels.RESNEXT101_32X8D:
        target_model_layer = "flatten"
    elif model_enum == SupportedModels.RESNEXT101_64X4D:
        target_model_layer = "flatten"
    else:
        raise UnsupportedModel(
            f"Please add appropriate target_model_layer for model {model_enum}. You can pass the --target-model-layer instead."
        )

    print(get_graph_node_names(model))

    model = create_feature_extractor(
        model, return_nodes={target_model_layer: "target_layer"}
    )
    return model, forward_kwargs


def extract_embeddings(embeddings: torch.Tensor, model: SupportedModels):
    """Some models may output a dictionary instead of a tensor.

    This function unpacks the output of each model
    """
    if model == SupportedModels.AST:
        return embeddings["pooler_output"]
    return embeddings["target_layer"]


if __name__ == "__main__":
    # Todo cacualte for each dataset train/val
    args, config, pl_args = parse_args()

    base_experiment_name = (
        Path(args.checkpoint).stem
        if args.checkpoint
        else f'{config.model.value}_{config.pretrained_tag.replace("/", "-")}'
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_audio_transform: AudioTransformBase = get_audio_transform(
        config,
        spectrogram_augmentation=None,
        waveform_augmentation=None,
    )
    val_audio_transform: AudioTransformBase = get_audio_transform(
        config,
        spectrogram_augmentation=None,
        waveform_augmentation=None,
    )

    collate_fn = get_collate_fn(config)
    datamodule = IRMASDataModule(
        train_paths=config.train_paths,
        val_paths=config.val_paths,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dataset_fraction=config.dataset_fraction,
        drop_last_sample=config.drop_last,
        train_audio_transform=train_audio_transform,
        val_audio_transform=val_audio_transform,
        collate_fn=collate_fn,
        normalize_audio=config.normalize_audio,
        normalize_image=config.normalize_image,
        train_only_dataset=False,
        concat_n_samples=None,
        sum_two_samples=False,
        use_weighted_train_sampler=False,
    )

    if args.checkpoint:
        model_constructor = model_constructor_map[config.model]
        model = model_constructor.load_from_checkpoint(args.checkpoint)
    elif config.pretrained_tag:
        model = get_model(config, loss_function=torch.nn.BCEWithLogitsLoss())

    model, forward_kwargs = get_feature_extractor(
        model, model_enum=config.model, target_model_layer=args.target_model_layer
    )
    model = model.to(device)
    model.eval()

    print("Saving embeddings to directory:", args.output_dir)

    train_data_loader = datamodule.train_dataloader()
    val_data_loader = datamodule.val_dataloader()

    for data_loader in [train_data_loader, val_data_loader]:
        for data in tqdm(data_loader, total=len(data_loader)):
            spectrogram, multihot_labels, file_indices, item_indices = data

            # Transfer to device
            spectrogram, multihot_labels, file_indices = (
                spectrogram.to(device),
                multihot_labels.to(device),
                file_indices.to(device),
            )

            # Get exact label n label number
            indices_list: list[list[int]] = []
            for multihot in multihot_labels:
                multihot = multihot.cpu().numpy()
                indices = instrument_multihot_to_idx(multihot).tolist()
                indices_list.append(indices)

            # Create and merge embeddings for each  file
            # spectrogram = prepare_model_input(spectrogram)
            with torch.no_grad():
                embeddings = model.forward(spectrogram, **forward_kwargs)
            embeddings = extract_embeddings(embeddings, config.model)
            embeddings = torch_scatter.scatter_mean(embeddings, file_indices, dim=0)

            # Data convesions
            item_indices = item_indices.detach().cpu().tolist()
            embeddings_list = embeddings.detach().cpu().tolist()

            # Iterate over each sample from the batch
            for item_index, embedding, indices in zip(
                item_indices, embeddings_list, indices_list
            ):
                # Find the exact dataset which the file originate from
                dataset_idx = bisect.bisect_right(
                    data_loader.dataset.cumulative_sizes, item_index
                )
                exact_dataset = data_loader.dataset.datasets[dataset_idx]
                if isinstance(exact_dataset, IRMASDatasetTrain) or isinstance(
                    exact_dataset, IRMASDatasetTest
                ):
                    audio_path, _ = exact_dataset.dataset_list[item_index]
                else:
                    raise Exception(
                        "Add 'isinstance(exact, YourDataset) and use item index to unpack the path to the file"
                    )

                dataset_enum, _ = datamodule.train_paths[dataset_idx]
                dataset_str = dataset_enum.value

                stem = Path(audio_path).stem  # e.g. [cel][cla]0001__1
                audio_path = str(Path(audio_path))
                instruments = [config_defaults.IDX_TO_INSTRUMENT[i] for i in indices]
                instrument_names = [
                    config_defaults.INSTRUMENT_TO_FULLNAME[n] for n in instruments
                ]

                json_item = dict(
                    sample_path=audio_path,
                    indices=indices,
                    instruments=instruments,
                    instrument_names=instrument_names,
                    embedding=embedding,
                )

                subdir = f"{base_experiment_name}_{dataset_str}"

                if data_loader == train_data_loader:
                    subdir += "_train"
                else:
                    subdir += "_test"

                embedding_dir = Path(args.output_dir, subdir, "embeddings")
                embedding_dir.mkdir(parents=True, exist_ok=True)
                json_file_name = Path(embedding_dir, f"{stem}.json")

                with open(json_file_name, "w") as file:
                    json.dump(json_item, file)
