from pathlib import Path

import torch
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
from src.model.model import get_model
from src.model.model_ast import ASTModelWrapper
from src.model.model_torch import TORCHVISION_CONSTRUCTOR_DICT, TorchvisionModel
from src.model.model_wav2vec import Wav2VecWrapper
from src.model.model_wav2vec_cnn import Wav2VecCnnWrapper
from src.utils.utils_dataset import calc_instrument_weight
from src.utils.utils_exceptions import InvalidArgument
from src.utils.utils_model import print_modules


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
    if args.output_dir is not None:
        subdir = (
            Path(args.checkpoint).stem if args.checkpoint else config.pretrained_tag
        )  # ast-finetuned-audioset-10-10-0.4593 or 04-06-23-56-05_DeftYankee_ast_val_acc_0.0000_val_loss_0.3650
        args.output_dir = Path(config.path_irmas_train_features, subdir)
        raise InvalidArgument("Please provide --model argument")

    return args, config, pl_args


if __name__ == "__main__":
    args, config, pl_args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(config)
    print(args)
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

    if config.loss_function == SupportedLossFunctions.CROSS_ENTROPY:
        loss_function = torch.nn.BCEWithLogitsLoss(**config.loss_function_kwargs)
    if config.loss_function == SupportedLossFunctions.CROSS_ENTROPY_POS_WEIGHT:
        kwargs = {
            **config.loss_function_kwargs,
            "pos_weight": calc_instrument_weight(datamodule.count_classes()),
        }
        loss_function = torch.nn.BCEWithLogitsLoss(**kwargs)

    if args.checkpoint:
        if config.model == SupportedModels.AST:
            model = ASTModelWrapper
        elif config.model in TORCHVISION_CONSTRUCTOR_DICT:
            model = TorchvisionModel
        elif config.model == SupportedModels.WAV2VEC:
            model = Wav2VecWrapper
        elif config.model == SupportedModels.WAV2VEC_CNN:
            model = Wav2VecCnnWrapper
        model = model.load_from_checkpoint(args.checkpoint)
    elif config.pretrained_tag:
        model = get_model(config, loss_function=loss_function)

    if args.target_model_layer is None:
        if isinstance(model, Wav2VecCnnWrapper):
            pass
        if isinstance(model, ASTModelWrapper):
            pass
        if isinstance(model, TorchvisionModel):
            pass
    print_modules(model)
    model.eval()

    print("Saving embeddings to: ", args.output_dir)

    datamodule.train_dataset.return_filename = True
    datamodule.test_dataset.return_filename = True
    data_loader = datamodule.train_dataloader()

    for data in tqdm(data_loader, total=len(data_loader)):
        spectrogram, onehot_labels, audio_paths = data
        # exit(1)
        spectrogram, onehot_labels = spectrogram.to(device), onehot_labels.to(device)

        labels = torch.argmax(onehot_labels, dim=-1)  # extract label number

        embeddings = model.forward(
            spectrogram,
            output_attentions=False,
            return_dict=True,
        )
        exit(1)
        # ast_embeddings_list = ast_embeddings.pooler_output.detach().cpu().tolist()

        # # Iterate over each file from the batch
        # for audio_path, ast_embedding, label in zip(
        #     audio_paths, ast_embeddings_list, labels
        # ):
        #     stem = Path(audio_path).stem  # e.g. [cel][cla]0001__1
        #     audio_path = str(Path(audio_path).relative_to(current_working_dir))
        #     instrument_idx = int(label)
        #     instrument = config.IDX_TO_INSTRUMENT[instrument_idx]
        #     instrument_name = config.INSTRUMENT_TO_FULLNAME[instrument]

        #     json_item = dict(
        #         sample_path=audio_path,
        #         label=instrument_idx,
        #         instrument=instrument,
        #         instrument_name=instrument_name,
        #         embedding=ast_embedding,
        #     )

        #     json_name = Path(OUTPUT_DIR, f"{stem}.json")

        #     with open(json_name, "w") as fp:
        #         json.dump(json_item, fp)
