import argparse

import librosa
import simple_parsing
import torch
from tqdm import tqdm

from src.config.argparse_with_config import ArgParseWithConfig
from src.config.config_defaults import NUM_RGB_CHANNELS, ConfigDefault
from src.data.datamodule import IRMASDataModule
from src.utils.utils_dataset import add_rgb_channel, create_and_repeat_channel
from src.utils.utils_exceptions import InvalidArgument


def parse():
    parser = ArgParseWithConfig()
    args, config, pl_args = parser.parse_args()
    config.required_train_paths()
    return args, config


def audios_to_flat_spectrograms(
    audio: torch.Tensor, num_channels: int, config: ConfigDefault
):
    audio = audio.numpy()
    spectrograms = [
        torch.tensor(
            librosa.feature.melspectrogram(
                y=a,
                sr=config.sampling_rate,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                n_mels=config.n_mels,
            )
        )
        for a in audio
    ]
    images = torch.stack(spectrograms)  # list to tensor
    batch_size = images.size(0)
    images = create_and_repeat_channel(
        images, num_channels
    )  # [Batch, Channel, Height, Width]

    flat_images = images.view(
        batch_size, images.size(1), -1
    )  # [Batch, Channel, Height x Width]
    return flat_images


if __name__ == "__main__":
    args, config = parse()

    datamodule = IRMASDataModule(
        train_paths=config.train_paths,
        val_paths=config.val_paths,
        test_paths=config.test_paths,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dataset_fraction=config.dataset_fraction,
        drop_last_sample=False,
        train_audio_transform=None,
        val_audio_transform=None,
        collate_fn=None,
        normalize_audio=config.normalize_audio,
        normalize_image=config.normalize_image,
        train_only_dataset=config.train_only_dataset,
        concat_n_samples=None,
        sum_two_samples=None,
        use_weighted_train_sampler=False,
    )

    train_dataloader = datamodule.train_dataloader()

    num_channels = 1
    mean = torch.zeros(num_channels)

    # We can perform means of means because each image has the same number of pixels.

    for audio, _, _ in tqdm(train_dataloader):
        flat_images = audios_to_flat_spectrograms(
            audio, num_channels, config
        )  # [Batch, Channel, Height x Width]

        mean_per_channel_per_batch = flat_images.mean(2)  # [Batch, Channel]
        mean_per_channel = mean_per_channel_per_batch.sum(0)  # [Channel]
        mean += mean_per_channel

    mean = mean / len(train_dataloader.dataset)
    print(f"Mean for each channel is: {mean}")

    var = torch.zeros(num_channels)
    for audio, _, _ in tqdm(train_dataloader):
        flat_images = audios_to_flat_spectrograms(
            audio, num_channels, config
        )  # [Batch, Channel, Height x Width]

        # unsqueeze 1 because we have to bring the `mean` to the channel dimension instead of zero-th dimension.
        diff = (flat_images - mean.unsqueeze(1)) ** 2
        var += diff.sum([0, 2])

    num_of_pixels = flat_images.size(2)
    std = torch.sqrt(var / (len(train_dataloader.dataset) * num_of_pixels))

    print(f"Std for each channel is: {std}")
