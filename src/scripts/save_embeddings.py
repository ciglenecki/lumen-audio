import json
import os
from itertools import chain
from pathlib import Path

import pytorch_lightning.callbacks
import torch
import transformers.utils.fx
from torch.fx import symbolic_trace
from torchvision.models.feature_extraction import (
    _get_leaf_modules_for_ops,
    create_feature_extractor,
    get_graph_node_names,
)
from tqdm import tqdm

from src.config.argparse_with_config import ArgParseWithConfig
from src.data.datamodule import IRMASDataModule
from src.enums.enums import (
    AudioTransforms,
    SupportedAugmentations,
    SupportedLossFunctions,
    SupportedModels,
)
from src.features.audio_transform import AudioTransformBase, get_audio_transform
from src.features.augmentations import get_augmentations
from src.features.chunking import get_collate_fn
from src.model.model import get_model, model_constructor_map
from src.model.model_ast import ASTModelWrapper
from src.model.model_torch import TORCHVISION_CONSTRUCTOR_DICT, TorchvisionModel
from src.model.model_wav2vec import Wav2VecWrapper
from src.model.model_wav2vec_cnn import Wav2VecCnnWrapper
from src.utils.utils_dataset import calc_instrument_weight
from src.utils.utils_exceptions import InvalidArgument, UnsupportedModel
from src.utils.utils_model import find_model_parameter, print_modules


def parse_args():
    parser = ArgParseWithConfig()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
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

    if int(bool(config.pretrained_tag)) + int(bool(args.checkpoint)) == 1:
        raise InvalidArgument(
            "Please provide either --pretrained-tag or --checkpoint <PATH>"
        )
    if args.output_dir is None:
        subdir = (
            Path(args.checkpoint).stem if args.checkpoint else config.pretrained_tag
        )  # ast-finetuned-audioset-10-10-0.4593 or 04-06-23-56-05_DeftYankee_ast_val_acc_0.0000_val_loss_0.3650
        args.output_dir = Path(config.path_irmas_train_features, subdir)

    return args, config, pl_args


def get_feature_extractor(
    model: torch.nn.Module, model_enum: SupportedModels, target_model_layer: str = None
) -> tuple[torch.nn.Module, dict]:
    forward_kwargs = {}
    if args.target_model_layer is None:
        if model_enum == SupportedModels.AST:
            model = model.backbone.audio_spectrogram_transformer
            forward_kwargs = dict(
                head_mask=None, output_attentions=False, return_dict=True
            )
            return model, forward_kwargs

        elif model_enum == SupportedModels.WAV2VEC:
            args.target_model_layer = "layers.11.final_layer_norm"
        elif model_enum == SupportedModels.WAV2VEC_CNN:
            args.target_model_layer = "conv_layers.6.activation"
        elif model_enum == SupportedModels.EFFICIENT_NET_V2_S:
            args.target_model_layer = "features.7.2"
        elif model_enum == SupportedModels.EFFICIENT_NET_V2_M:
            args.target_model_layer = "features.8.2"
        elif model_enum == SupportedModels.EFFICIENT_NET_V2_L:
            args.target_model_layer = "features.8.2"
        elif model_enum == SupportedModels.RESNEXT50_32X4D:
            args.target_model_layer = "layer4.2.relu"
        elif model_enum == SupportedModels.RESNEXT101_32X8D:
            args.target_model_layer = "layer4.2.relu"
        elif model_enum == SupportedModels.RESNEXT101_64X4D:
            args.target_model_layer = "layer4.2.relu"
        else:
            raise UnsupportedModel(
                f"Please add appropriate target_model_layer for model {model_enum}. You can pass the --target-model-layer instead."
            )
        target_layer_str, _ = find_model_parameter(model, args.target_model_layer)
        model = create_feature_extractor(
            model, return_nodes={target_layer_str: "target_layer"}
        )
        return model, forward_kwargs

    pass


def clean_embeddings_after_foward(embeddings: torch.Tensor, model: SupportedModels):
    if model == SupportedModels.AST:
        return embeddings["pooler_output"]

    assert (
        embeddings.size() == 2
    ), f"Embeddings {embeddings.shape} should be [Batch size, flattened embeddings]"
    return embeddings


if __name__ == "__main__":
    # Todo cacualte for each dataset train/val
    args, config, pl_args = parse_args()
    config.batch_size = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    (
        train_spectrogram_augmentation,
        train_waveform_augmentation,
        val_spectrogram_augmentation,
        val_waveform_augmentation,
    ) = get_augmentations(config)

    train_audio_transform: AudioTransformBase = get_audio_transform(
        config,
        spectrogram_augmentation=train_spectrogram_augmentation,
        waveform_augmentation=train_waveform_augmentation,
    )
    val_audio_transform: AudioTransformBase = get_audio_transform(
        config,
        spectrogram_augmentation=val_spectrogram_augmentation,
        waveform_augmentation=val_waveform_augmentation,
    )

    concat_n_samples = (
        config.aug_kwargs["concat_n_samples"]
        if SupportedAugmentations.CONCAT_N_SAMPLES in config.augmentations
        else None
    )
    # collate_fn = get_collate_fn(config)
    # TODO: fix this
    collate_fn = None
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
    print("Saving embeddings to: ", args.output_dir)

    data_loader = datamodule.train_dataloader()

    for data in tqdm(data_loader, total=len(data_loader)):
        spectrogram, onehot_labels, file_indices, dataset_indices = data
        spectrogram, onehot_labels = spectrogram.to(device), onehot_labels.to(device)

        labels = torch.argmax(onehot_labels, dim=-1)  # extract label number

        embeddings = model.forward(spectrogram, **forward_kwargs)
        embeddings = clean_embeddings_after_foward(embeddings, config.model)
        # TODO SCATETR MEAN ACCROSS FILE INDICES
        audio_paths_list = dataset_indices.detach().cpu().tolist()
        embeddings_list = embeddings.detach().cpu().tolist()
        labels_list = labels.detach().cpu().tolist()

        # Iterate over each sample from the batch
        for audio_path, embedding, label in zip(
            audio_paths_list, embeddings_list, labels_list
        ):
            stem = Path(audio_path).stem  # e.g. [cel][cla]0001__1
            audio_path = str(Path(audio_path).relative_to(os.getcwd()))
            instrument_idx = int(label)
            instrument = config.IDX_TO_INSTRUMENT[instrument_idx]
            instrument_name = config.INSTRUMENT_TO_FULLNAME[instrument]

            json_item = dict(
                sample_path=audio_path,
                label=instrument_idx,
                instrument=instrument,
                instrument_name=instrument_name,
                embedding=embedding,
            )

            json_file_name = Path(args.output_dir, f"{stem}.json")

            with open(json_file_name, "w") as file:
                json.dump(json_item, file)
