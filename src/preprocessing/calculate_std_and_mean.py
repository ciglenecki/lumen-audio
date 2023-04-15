from src.features.audio_transform import get_audio_transform
from src.enums.enums import SupportedAugmentations, AudioTransforms
from src.features.chunking import collate_fn_spectrogram
from src.data.datamodule import IRMASDataModule
from src.config.config_defaults import ConfigDefault
from tqdm import tqdm
import torch
import simple_parsing
import argparse

def parse():
    destination_str = "user_args"
    parser = simple_parsing.ArgumentParser(
        add_option_string_dash_variants=simple_parsing.DashVariant.DASH
    )
    #parser.add_argument(
    #    "--audio-transform", 
    #    type=AudioTransforms, 
    #    default=None, 
    #    choices=list(AudioTransforms), 
    #    help="The audio transform for which the mean and std will be calculated."
    #)
    parser.add_arguments(ConfigDefault, dest=destination_str)
    args = parser.parse_args()

    args_dict = vars(args)
    config: ConfigDefault = args_dict.pop(destination_str)
    args = argparse.Namespace(**args_dict)

    return args, config

if __name__ == "__main__":
    args, config = parse()
    
    config.audio_transform= get_audio_transform(config, config.audio_transform)

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

    mean, std = torch.tensor([0., 0., 0.]), torch.tensor([0., 0., 0.])

    max_, min_ = None, None
    train_dataloader = datamodule.train_dataloader()
    for idx, (inputs, _, _) in tqdm(enumerate(train_dataloader)):
        max_curr, min_curr = inputs.max(), inputs.min()

        if max_ is None or max_curr > max_:
            max_ = max_curr
 
        if min_ is None or min_curr < min_:
            min_ = min_curr

    print(min_, max_)

    for idx, (inputs, _, _) in tqdm(enumerate(train_dataloader)):
        if inputs.shape[0] != config.batch_size:
            print(f"Something is wrong because chunking is being used... {inputs.shape}")
        
        inputs_scaled = (inputs - min_) / (max_ - min_)
        
        if inputs_scaled.max() > 1 or inputs_scaled.min() < 0:
            print(f"Something is wrong with the scaling... {inputs_scaled.max(), inputs_scaled.min()}")
        
        for i in range(len(mean)):
            mean[i] += torch.sum(inputs_scaled[:, i])

    mean = mean / (len(train_dataloader) * inputs.shape[2] * inputs.shape[3] * config.batch_size)
    
    print(f"Mean for each channel is {mean}")



    
    for idx, (inputs, _, _) in tqdm(enumerate(train_dataloader)):
        inputs_scaled = (inputs - min_) / (max_ - min_)
        
        for i in range(len(std)):
            std[i] += (inputs_scaled[:, i] - mean[i]).sum()
    
    std = torch.sqrt( std**2 / ((len(train_dataloader) * inputs.shape[2] * inputs.shape[3] * config.batch_size) - 1) ) # unbiased estimate
    
    print(f"Stdev for each channel is {std}")



## MFCC FIXED REPEAT ##
# min_ = -723.2180
# max_ = 282.6545
# mean = [0.7117, 0.7117, 0.7117]
# std = [0.0219, 0.0219, 0.0219]

## MELSPEC FIXED REPEAT ##
# min_ = 0. 
# max_ = 314.9380
# mean = [0.0013, 0.0013, 0.0013]
# std = [0.0367, 0.0367, 0.0367]
