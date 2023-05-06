import torch
from tqdm import tqdm

from src.config.argparse_with_config import ArgParseWithConfig
from src.data.datamodule import OurDataModule
from src.features.audio_transform import get_audio_transform
from src.features.chunking import collate_fn_feature
from src.model.model import get_model
from time import time
import json
import numpy as np

def parse():
    parser = ArgParseWithConfig()
    args, config, pl_args = parser.parse_args()
    config.required_test_paths()
    config.required_audio_transform()
    return args, config

def main():
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

    num_measurements = 10
    dataloader = datamodule.test_dataloader()

    model = get_model(config, loss_function=torch.nn.BCEWithLogitsLoss())

    infer_times = []
    step = 0
    for b in tqdm(dataloader):
        time_for_ex = []
        for _ in range(num_measurements):
            start = time()
            _ = model.test_step(b, 0)
            end = time()
            time_for_ex.append(end - start)
        
        infer_times.append({
            "step": step,
            "mean": np.mean(time_for_ex),
            "std": np.std(time_for_ex),
        })

        step += 1

    with open(f"data/{config.experiment_suffix}_infer_res.json", 'w') as f:
        json.dump(infer_times, f)

main()