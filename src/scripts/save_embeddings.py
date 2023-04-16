"""python3 src/scripts/save_embeddings.py  --model EFFICIENT_NET_V2_S --audio-transform
MEL_SPECTROGRAM.

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
from src.enums.enums import (
    AudioTransforms,
    SupportedAugmentations,
    SupportedLossFunctions,
    SupportedModels,
)
from src.features.audio_transform import AudioTransformBase, get_audio_transform
from src.features.chunking import get_collate_fn
from src.model.model import get_model, model_constructor_map
from src.utils.utils_dataset import calc_instrument_weight
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

    return args, config, pl_args


def get_feature_extractor(
    model: torch.nn.Module, model_enum: SupportedModels, target_model_layer: str = None
) -> tuple[torch.nn.Module, dict]:
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


def clean_embeddings_after_foward(embeddings: torch.Tensor, model: SupportedModels):
    if model == SupportedModels.AST:
        return embeddings["pooler_output"]
    return embeddings["target_layer"]


if __name__ == "__main__":
    # Todo cacualte for each dataset train/val
    args, config, pl_args = parse_args()

    base_experiment_name = (
        Path(args.checkpoint).stem
        if args.checkpoint
        else config.model.value + config.pretrained_tag
    )
    config.batch_size = 1

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

    concat_n_samples = (
        config.aug_kwargs["concat_n_samples"]
        if SupportedAugmentations.CONCAT_N_SAMPLES in config.augmentations
        else None
    )
    collate_fn = get_collate_fn(config)
    datamodule = IRMASDataModule(
        train_dirs=config.train_dirs,
        val_dirs=config.val_dirs,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dataset_fraction=config.dataset_fraction,
        drop_last_sample=config.drop_last,
        train_audio_transform=train_audio_transform,
        val_audio_transform=val_audio_transform,
        collate_fn=collate_fn,
        normalize_audio=config.normalize_audio,
        train_only_dataset=config.train_only_dataset,
        concat_n_samples=concat_n_samples,
        sum_two_samples=SupportedAugmentations.SUM_TWO_SAMPLES in config.augmentations,
        use_weighted_train_sampler=config.use_weighted_train_sampler,
    )
    datamodule.setup()

    if config.loss_function == SupportedLossFunctions.CROSS_ENTROPY:
        loss_function = torch.nn.BCEWithLogitsLoss(**config.loss_function_kwargs)
    if config.loss_function == SupportedLossFunctions.CROSS_ENTROPY_POS_WEIGHT:
        kwargs = {
            **config.loss_function_kwargs,
            "pos_weight": calc_instrument_weight(datamodule.count_classes()),
        }
        loss_function = torch.nn.BCEWithLogitsLoss(**kwargs)

    if args.checkpoint:
        model_constructor = model_constructor_map[config.model]
        model = model_constructor.load_from_checkpoint(args.checkpoint)
    elif config.pretrained_tag:
        model = get_model(config, loss_function=loss_function)

    model, forward_kwargs = get_feature_extractor(
        model, model_enum=config.model, target_model_layer=args.target_model_layer
    )
    model.eval()
    model = model.to(device)
    print("Saving embeddings to:", args.output_dir)

    train_data_loader = datamodule.train_dataloader()
    val_data_loader = datamodule.val_dataloader()

    for data_loader in [train_data_loader, val_data_loader]:
        for data in tqdm(data_loader, total=len(data_loader)):
            spectrogram, onehot_labels, file_indices, item_indices = data

            # Transfer to device
            spectrogram, onehot_labels, file_indices = (
                spectrogram.to(device),
                onehot_labels.to(device),
                file_indices.to(device),
            )

            # Get exact label n label number
            labels = torch.argmax(onehot_labels, dim=-1)

            # Create and merge embeddings for each  file
            # spectrogram = prepare_model_input(spectrogram)
            embeddings = model.forward(spectrogram, **forward_kwargs)
            embeddings = clean_embeddings_after_foward(embeddings, config.model)
            embeddings = torch_scatter.scatter_mean(embeddings, file_indices, dim=0)
            labels = torch_scatter.scatter_max(labels, file_indices, dim=0)

            # Data convesions
            item_indices = item_indices.detach().cpu().tolist()
            embeddings_list = embeddings.detach().cpu().tolist()
            labels_list = [int(label.detach().cpu()) for label in labels]

            # Iterate over each sample from the batch
            for item_index, embedding, label in zip(
                item_indices, embeddings_list, labels_list
            ):
                # Find the exact dataset which the file originate from
                dataset_idx = bisect.bisect_right(
                    data_loader.dataset.cumulative_sizes, item_index
                )
                exact_dataset = data_loader.dataset.datasets[dataset_idx]
                if isinstance(exact_dataset, IRMASDatasetTrain) or isinstance(
                    exact_dataset, IRMASDatasetTest
                ):
                    audio_path, _ = exact_dataset.dataset[item_index]
                else:
                    raise Exception(
                        "Add 'isinstance(exact, YourDataset) and use item index to unpack the path to the file"
                    )

                dataset_enum, _ = datamodule.train_dirs[dataset_idx]
                dataset_str = dataset_enum.value

                stem = Path(audio_path).stem  # e.g. [cel][cla]0001__1
                audio_path = str(Path(audio_path))
                instrument_idx = label
                instrument = config_defaults.IDX_TO_INSTRUMENT[instrument_idx]
                instrument_name = config_defaults.INSTRUMENT_TO_FULLNAME[instrument]

                json_item = dict(
                    sample_path=audio_path,
                    label=instrument_idx,
                    instrument=instrument,
                    instrument_name=instrument_name,
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
