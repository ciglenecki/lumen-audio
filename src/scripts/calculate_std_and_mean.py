import argparse

import simple_parsing
import torch
from tqdm import tqdm

from src.config.config_defaults import ConfigDefault
from src.data.datamodule import IRMASDataModule
from src.enums.enums import SupportedAugmentations
from src.features.audio_transform import get_audio_transform
from src.features.chunking import collate_fn_spectrogram


def parse():
    destination_str = "user_args"
    parser = simple_parsing.ArgumentParser(
        add_option_string_dash_variants=simple_parsing.DashVariant.DASH
    )
    parser.add_arguments(ConfigDefault, dest=destination_str)
    args = parser.parse_args()

    args_dict = vars(args)
    config: ConfigDefault = args_dict.pop(destination_str)
    args = argparse.Namespace(**args_dict)

    return args, config


if __name__ == "__main__":
    args, config = parse()

    config.audio_transform = get_audio_transform(config, config.audio_transform)

    datamodule = IRMASDataModule(
        train_dirs=config.train_dirs,
        val_dirs=config.val_dirs,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dataset_fraction=config.dataset_fraction,
        drop_last_sample=config.drop_last,
        train_audio_transform=config.audio_transform,
        val_audio_transform=config.audio_transform,
        collate_fn=collate_fn_spectrogram,
        normalize_audio=config.normalize_audio,
        train_only_dataset=config.train_only_dataset,
        concat_n_samples=(
            config.aug_kwargs["concat_n_samples"]
            if SupportedAugmentations.CONCAT_N_SAMPLES in config.augmentations
            else None
        ),
        sum_two_samples=SupportedAugmentations.SUM_TWO_SAMPLES in config.augmentations,
        use_weighted_train_sampler=config.use_weighted_train_sampler,
    )

    train_dataloader = datamodule.train_dataloader()

    mean = 0.0
    for images, _, _ in tqdm(train_dataloader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(train_dataloader.dataset)

    print(f"Mean for each channel is: {mean}")

    var = 0.0
    for images, _, _ in tqdm(train_dataloader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])

    std = torch.sqrt(var / (len(train_dataloader.dataset) * images.size(2)))

    print(images.size(2))
    print(f"Mean for each channel is: {std}")
