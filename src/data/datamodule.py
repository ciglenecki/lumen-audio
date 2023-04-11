from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils
import torch.utils.data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from tqdm import tqdm

import src.config.defaults as defaults
from src.config.config import config
from src.data.dataset_irmas import IRMASDatasetTest, IRMASDatasetTrain
from src.enums.enums import SupportedDatasets
from src.features.audio_transform_base import AudioTransformBase
from src.features.augmentations import SupportedAugmentations
from src.utils.utils_functions import split_by_ratio


class IRMASDataModule(pl.LightningDataModule):
    train_size: int
    val_size: int
    test_size: int
    train_dataset: torch.utils.data.ConcatDataset
    val_dataset: torch.utils.data.ConcatDataset
    test_dataset: torch.utils.data.ConcatDataset
    train_sampler: SubsetRandomSampler
    val_sampler: SubsetRandomSampler
    test_sampler: SubsetRandomSampler

    """
    IRMASDataModule is responsible for efficiently creating datasets creating a
    indexing strategy (SubsetRandomSampler) for each dataset.
    Any preprocessing which requires aggregation of data,
    such as caculating the mean and standard deviation of the dataset
    should be performed here.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        dataset_fraction: int,
        drop_last_sample: bool,
        train_audio_transform: AudioTransformBase,
        val_audio_transform: AudioTransformBase,
        collate_fn: Callable | None = None,
        train_dirs: list[tuple[SupportedDatasets, Path]] = config.train_dirs,
        val_dirs: list[tuple[SupportedDatasets, Path]] = config.val_dirs,
        test_dirs: list[tuple[SupportedDatasets, Path]] = config.val_dirs,
        train_only_dataset: bool = config.train_only_dataset,
        normalize_audio: bool = config.normalize_audio,
        concat_two_samples: bool = SupportedAugmentations.CONCAT_TWO
        in config.augmentations,
        use_weighted_train_sampler=config.use_weighted_train_sampler,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_fraction = dataset_fraction
        self.drop_last_sample = drop_last_sample
        self.train_audio_transform: AudioTransformBase = train_audio_transform
        self.val_audio_transform: AudioTransformBase = val_audio_transform
        self.prepare_data_per_node = False
        self.train_dirs = train_dirs
        self.val_dirs = val_dirs
        self.test_dirs = test_dirs
        self.train_only_dataset = train_only_dataset
        self.normalize_audio = normalize_audio
        self.concat_two_samples = concat_two_samples
        self.use_weighted_train_sampler = use_weighted_train_sampler
        self.collate_fn = collate_fn

        # Read: if there is only one dataset and that dataset is IRMAS:
        if (
            len(self.train_dirs) == 1
            and self.train_dirs[0][0] == SupportedDatasets.IRMAS
        ):
            self.class_count_dict = defaults.IRMAS_TRAIN_CLASS_COUNT
        else:
            self.class_count_dict = {}
        self.setup()

    def prepare_data(self) -> None:
        """Has to be implemented to avoid object has no attribute 'prepare_data_per_node' error."""

    def get_sample_class_weights(self, dataset: torch.utils.data.Dataset):
        """The function returns n weights for n samples in the dataset.

        Weights are caculated as (1 / class_count_dict) of a particular example.

        [0,  1] <- 1.2
        [0,  1] <- 1.2
        [0,  1]
        [0,  1] <- 1.2
        [0,  1]
        [1,  0] <- 3.5
        [1,  1] <- 3.5
        3.5 1.2
        """

        print("Caculating sample classes...")
        one_hots = []
        for _, one_hot, _ in tqdm(dataset):
            one_hots.append(one_hot)
        one_hots = torch.stack(one_hots)
        class_count_dicts = torch.sum(one_hots, dim=0)
        weight_per_class = 1 / class_count_dicts
        examples_weights = one_hots * weight_per_class
        _, example_max_weight = examples_weights.max(dim=-1, keepdim=False)
        return example_max_weight

    def _get_train_dataset_concated(self):
        datasets = []
        for dataset_enum, dataset_path in self.train_dirs:
            if dataset_enum == SupportedDatasets.IRMAS:
                dataset = IRMASDatasetTrain(
                    dataset_dir=dataset_path,
                    audio_transform=self.train_audio_transform,
                    normalize_audio=self.normalize_audio,
                    concat_two_samples=self.concat_two_samples,
                )
            datasets.append(dataset)
        return torch.utils.data.ConcatDataset(datasets)

    def _get_val_dataset_concated(self):
        datasets = []
        for dataset_enum, dataset_path in self.val_dirs:
            if dataset_enum == SupportedDatasets.IRMAS:
                dataset = IRMASDatasetTest(
                    dataset_dir=dataset_path,
                    audio_transform=self.val_audio_transform,
                    normalize_audio=self.normalize_audio,
                )
            datasets.append(dataset)
        return torch.utils.data.ConcatDataset(datasets)

    def _log_indices(self):
        train_indices = self.train_sampler.indices
        val_indices = self.val_sampler.indices
        test_indices = self.test_sampler.indices

        print()
        print(
            "Train size",
            self.train_size,
            "indices:",
            train_indices[0:5],
            train_indices[-5:],
        )
        print(
            "Val size",
            self.val_size,
            "indices:",
            val_indices[0:5],
            val_indices[-5:],
        )
        print(
            "Test size",
            self.test_size,
            "indices:",
            test_indices[0:5],
            test_indices[-5:],
        )
        print()

    def setup(self, stage=None):
        super().setup(stage)

        self.train_dataset = self._get_train_dataset_concated()

        if self.train_only_dataset:
            self.test_dataset = self.train_dataset
            indices = np.arange(len(self.train_dataset))
            train_indices, val_indices = train_test_split(indices, test_size=0.2)
            test_indices = np.array([])
            # test_indices = val_indices
            self._sanity_check_difference(train_indices, val_indices)
        else:
            self.test_dataset = self._get_val_dataset_concated()
            train_indices = np.arange(len(self.train_dataset))
            val_test_indices = np.arange(len(self.test_dataset))
            val_indices, test_indices = split_by_ratio(val_test_indices, 0.8, 0.2)
            self._sanity_check_difference(val_indices, test_indices)

        if self.dataset_fraction != 1:
            train_indices = np.random.choice(
                train_indices,
                int(self.dataset_fraction * len(train_indices)),
                replace=False,
            )
            val_indices = np.random.choice(
                val_indices,
                int(self.dataset_fraction * len(val_indices)),
                replace=False,
            )
            test_indices = np.random.choice(
                test_indices,
                int(self.dataset_fraction * len(test_indices)),
                replace=False,
            )

        self._sanity_check_difference(val_indices, test_indices)

        self.train_size = len(train_indices)
        self.val_size = len(val_indices)
        self.test_size = len(test_indices)

        if self.use_weighted_train_sampler and stage == "train":
            samples_weight = self.get_sample_class_weights(self.train_dataset)
            self.train_sampler = WeightedRandomSampler(
                samples_weight, len(samples_weight)
            )
        else:
            self.train_sampler = SubsetRandomSampler(train_indices.tolist())
        self.val_sampler = SubsetRandomSampler(val_indices.tolist())
        self.test_sampler = SubsetRandomSampler(test_indices.tolist())
        self._log_indices()

    def _sanity_check_difference(
        self,
        indices_a: np.ndarray,
        indices_b: np.ndarray,
    ):
        """Checks if there are overlaping val and test indicies to avoid data leakage."""

        for ind_a, ind_b in combinations([indices_a, indices_b], 2):
            assert (
                len(np.intersect1d(ind_a, ind_b)) == 0
            ), f"Some indices share an index {np.intersect1d(ind_a, ind_b)}"
        set_ind = set(indices_a)
        set_ind.update(indices_b)
        assert len(set_ind) == (
            len(indices_a) + len(indices_b)
        ), "Some indices might contain non-unqiue values"

    def count_classes(self) -> dict[str, int]:
        if self.class_count_dict:
            return self.class_count_dict

        output = {}
        for dataset in self.train_dataset.datasets:
            for _, label in tqdm(dataset, desc="Counting classes"):
                idx = int(np.where(label == 1)[0])
                instrument = defaults.IDX_TO_INSTRUMENT[idx]
                if instrument not in output:
                    output[instrument] = 0
                output[instrument] += 1
        self.class_count_dict = output
        return output

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.train_sampler,
            drop_last=self.drop_last_sample,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Uses test dataset files but sampler takes care that validation and test get different
        files."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.val_sampler,
            drop_last=self.drop_last_sample,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.test_sampler,
            drop_last=self.drop_last_sample,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
