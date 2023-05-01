"""python3 src/scripts/calculate_std_and_mean.py --audio-transform MFCC --test-paths
irmastrain:data/irmas/train.

python3 src/scripts/calculate_std_and_mean.py --audio-transform MEL_SPECTROGRAM --test-paths
irmastrain:data/irmas/train
"""
import torch
from tqdm import tqdm

from src.config.argparse_with_config import ArgParseWithConfig
from src.config.config_defaults import ConfigDefault
from src.data.datamodule import OurDataModule
from src.features.audio_transform import get_audio_transform
from src.features.chunking import collate_fn_feature
from src.utils.utils_dataset import create_and_repeat_channel


def parse():
    parser = ArgParseWithConfig()
    args, config, pl_args = parser.parse_args()
    config.required_test_paths()
    config.required_audio_transform()
    return args, config


def audios_to_flat_spectrograms(
    images: torch.Tensor, num_channels: int, config: ConfigDefault
):
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
    transform = get_audio_transform(
        config, spectrogram_augmentation=None, waveform_augmentation=None
    )
    collate_fn = collate_fn_feature
    datamodule = OurDataModule(
        train_paths=None,
        val_paths=None,
        test_paths=config.test_paths,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dataset_fraction=config.dataset_fraction,
        drop_last_sample=False,
        train_audio_transform=None,
        val_audio_transform=transform,
        collate_fn=collate_fn,
        normalize_audio=config.normalize_audio,
        normalize_image=config.normalize_image,
        train_only_dataset=config.train_only_dataset,
        concat_n_samples=None,
        sum_n_samples=None,
        use_weighted_train_sampler=False,
        sampling_rate=config.sampling_rate,
        train_override_csvs=config.train_override_csvs,
    )
    datamodule.setup_for_inference()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader = datamodule.test_dataloader()

    num_channels = 3
    mean = torch.zeros(num_channels).to(device)

    # We can perform means of means because each image has the same number of pixels.

    num_patches = 0
    for audio, _, _, _ in tqdm(dataloader):
        num_patches += audio.shape[0]
        flat_images = audios_to_flat_spectrograms(audio, num_channels, config).to(
            device
        )  # [Batch, Channel, Height x Width]

        mean_per_channel_per_batch = flat_images.mean(2)  # [Batch, Channel]
        mean_per_channel = mean_per_channel_per_batch.sum(0)  # [Channel]
        mean += mean_per_channel

    mean = mean / num_patches
    print(f"Mean for each channel is: {mean}")

    var = torch.zeros(num_channels).to(device)
    for audio, _, _, _ in tqdm(dataloader):
        flat_images = audios_to_flat_spectrograms(audio, num_channels, config).to(
            device
        )  # [Batch, Channel, Height x Width]

        # unsqueeze 1 because we have to bring the `mean` to the channel dimension instead of zero-th dimension.
        diff = (flat_images - mean.unsqueeze(1)) ** 2
        var += diff.sum([0, 2])

    num_of_pixels = flat_images.size(2)
    std = torch.sqrt(var / (num_patches * num_of_pixels))

    print(f"Std for each channel is: {std}")
