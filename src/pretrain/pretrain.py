import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config.argparse_with_config import ArgParseWithConfig
from src.config.config_defaults import *
from src.data.dataset_irmas import *
from src.data.dataset_irmas import IRMASDatasetPreTrain, IRMASDatasetTest
from src.features.audio_transform import get_audio_transform
from src.features.chunking import collate_fn_feature
from src.model.model import get_model


def parse():
    parser = ArgParseWithConfig()
    parser.add_argument(
        "--shuffle_prob",
        type=float,
        default=0.3,
        help="The percentage of patches that will be shuffled within the same image.",
    )
    parser.add_argument(
        "--random_prob",
        type=float,
        default=0.3,
        help="The percentage of patches that will be replaced with random patches from different spectrograms.",
    )
    parser.add_argument(
        "--black_prob",
        type=float,
        default=0.1,
        help="The percentage of patches that will be blacked out. This isn't used to calculate the loss, it just makes the problem harder.",
    )
    parser.add_argument(
        "--width_div",
        type=int,
        default=4,
        help="The number of chunks in width dimension.",
    )
    parser.add_argument(
        "--height_div",
        type=int,
        default=4,
        help="The number of chunks in height dimension.",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--save_heads", action="store_true")
    parser.add_argument("--save_path", type=str, default="pretrained_models")
    args, config, pl_args = parser.parse_args()
    return args, config, pl_args


def split_img_into_tiles(img, width_divisor, height_divisor):
    _, _, H, W = img.shape

    chunk_width = W // width_divisor
    chunk_height = H // height_divisor

    # plt.imshow(img[0].permute(1,2,0))
    # plt.show()

    chunks = []
    for j in range(height_divisor):
        for i in range(width_divisor):
            chunk_ij = img[
                :,
                :,
                j * chunk_height : (j + 1) * chunk_height,
                i * chunk_width : (i + 1) * chunk_width,
            ]
            chunks.append(chunk_ij)

    return chunks


def get_model_for_pretraining(model, width_divisor, height_divisor):
    model = get_model(
        config=config,
        loss_function=nn.CrossEntropyLoss(),
    )

    last_module_name, last_module = list(model.named_modules())[-1]
    last_dim = last_module.in_features

    module_name_split = last_module_name.split(".")
    module = model
    for i in module_name_split[:-1]:
        module = getattr(module, i)
    module[-1] = nn.Identity()

    print(model)

    # create a head for each chunk for 3-way classification
    model.heads = nn.ModuleList(
        [
            nn.Linear(in_features=last_dim, out_features=3)
            for _ in range(width_divisor * height_divisor)
        ]
    )

    return model


def create_corrupted_patches_and_labels(
    features_tiles, features_random_tiles, shuffle_prob, random_prob, black_prob
):
    features_tiles = torch.stack(features_tiles)  # P, B, C, H, W
    features_random_tiles = torch.stack(features_random_tiles)

    # iterate over batch dimension so different examples in batch get shuffled differently
    all_labels = []
    for b in range(features_tiles.shape[1]):
        # generate label tensor
        dist_tensor = torch.zeros(features_tiles.shape[0], 4)
        dist_tensor[
            :, 0
        ] = shuffle_prob  # label 0 -> randomly shuffled within same exapmle
        dist_tensor[
            :, 1
        ] = random_prob  # label 1 -> randomly taken from different example
        dist_tensor[:, 2] = 1 - (
            shuffle_prob + random_prob + black_prob
        )  # label 2 -> original
        dist_tensor[:, 3] = black_prob
        sampled = torch.multinomial(dist_tensor, num_samples=1).view(-1)
        sampled[sampled == 3] = -100
        label = sampled

        # shuffling
        sample_shuffle_indices = torch.nonzero(label == 0).view(-1)
        sample_shuffle_rand_perm_indices = sample_shuffle_indices[
            torch.randperm(len(sample_shuffle_indices))
        ]

        # print(sample_shuffle_indices)
        # print(sample_shuffle_rand_perm_indices)

        features_tiles[sample_shuffle_indices] = features_tiles[
            sample_shuffle_rand_perm_indices
        ]
        # in case there are indices where shuffling didnt occur, these stay original
        same_index_mask = sample_shuffle_indices[
            torch.nonzero(sample_shuffle_indices == sample_shuffle_rand_perm_indices)
        ]
        label[same_index_mask] = 2

        # replacement with same parts of different example
        sample_random_indices = torch.nonzero(label == 1).view(-1)
        features_tiles[sample_random_indices] = features_random_tiles[
            sample_random_indices
        ]

        # replace with black chunks to make the task harder
        black_indices = torch.nonzero(label == -100)
        features_tiles[black_indices] = torch.zeros_like(features_tiles[0])

        # print(label)
        # for k in features_tiles:
        #    plt.imshow(k[b].permute(1,2,0))
        #    plt.show()
        # print(len(features_tiles))
        all_labels.append(label)
    return features_tiles, torch.stack(all_labels, dim=1)


def glue_final_images(features_tiles, width_divisor, height_divisor):
    B, C, H, W = features_tiles.shape[1:]
    img = torch.zeros((B, C, H * height_divisor, W * width_divisor))

    idx = 0
    for j in range(height_divisor):
        for i in range(width_divisor):
            img[:, :, j * H : (j + 1) * H, i * W : (i + 1) * W] = features_tiles[idx]
            idx += 1

    # plt.imshow(img[0].permute(1,2,0)/255.)
    # plt.show()
    return img


def save_models(model, epoch, path, save_heads):
    save_path_model = os.path.join(path, f"epoch_{epoch}", f"epoch_{epoch}_backbone.pt")
    torch.save(model, save_path_model)

    if save_heads:
        for i, h in enumerate(model.heads):
            save_path_heads = os.path.join(
                path, f"epoch_{epoch}", f"epoch_{epoch}_head_{i}.pt"
            )
            torch.save(h, save_path_heads)


if __name__ == "__main__":
    args, config, pl_args = parse()

    train_dataset = IRMASDatasetPreTrain(
        audio_transform=get_audio_transform(
            config=config,
            spectrogram_augmentation=None,
            waveform_augmentation=None,
        )
    )
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=config.batch_size, shuffle=True
    )

    validation_dataset = IRMASDatasetTest(
        dataset_dir=config.path_irmas_test,
        audio_transform=get_audio_transform(
            config=config,
            spectrogram_augmentation=None,
            waveform_augmentation=None,
        ),
    )
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn_feature,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model_for_pretraining(
        config, width_divisor=args.width_div, height_divisor=args.height_div
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), config.lr)

    num_training_steps = (
        len(train_dataloader) * config.epochs // pl_args.accumulate_grad_batches
    )
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )

    full_save_path = os.path.join(
        f"{args.save_path}", f"{config.model.value}_{config.audio_transform.value}"
    )
    os.makedirs(full_save_path, exist_ok=True)
    print(f"Models will be saved to : {full_save_path}")

    train_loss, val_loss = [], []
    step = 0
    for e in range(config.epochs):
        for batch in tqdm(train_dataloader):
            features, features_random = batch

            if len(features) == 1 and len(features_random) == 1:
                features = features[0]
                features_random = features_random[0]
            else:
                print(f"Warning . . . {features.shape, features_random.shape}")

            features_tiles = split_img_into_tiles(
                features, width_divisor=args.width_div, height_divisor=args.height_div
            )
            features_random_tiles = split_img_into_tiles(
                features_random,
                width_divisor=args.width_div,
                height_divisor=args.height_div,
            )

            inputs, labels = create_corrupted_patches_and_labels(
                features_tiles,
                features_random_tiles,
                shuffle_prob=args.shuffle_prob,
                random_prob=args.random_prob,
                black_prob=args.black_prob,
            )

            # print(type(inputs), type(labels))

            inputs = glue_final_images(
                inputs, width_divisor=args.width_div, height_divisor=args.height_div
            ).to(device)

            representation = model(inputs)
            heads_out = torch.stack([h.forward(representation) for h in model.heads])

            # heads_out.shape = width_div * height_div, B, 3 -> has to be N, C, d1
            # labels.shape    = width_div * height_d, B      -> has to be N, C

            heads_out = heads_out.view(-1, heads_out.shape[2], heads_out.shape[0])
            labels = labels.view(-1, labels.shape[0]).to(device)

            loss = cross_entropy(heads_out, labels, ignore_index=-100)
            loss.backward()

            step += 1
            if (
                pl_args.accumulate_grad_batches is None
                or step % pl_args.accumulate_grad_batches == 0
            ):
                optimizer.zero_grad()
                optimizer.step()
                scheduler.step()

        save_models(model, e, full_save_path, args.save_heads)
