from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils
import torch.utils.data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from tqdm import tqdm

import src.config.config_defaults as config_defaults
from src.data.dataset_irmas import IRMASDatasetTest, IRMASDatasetTrain
from src.features.audio_transform import AudioTransformBase
from src.features.supported_augmentations import SupportedAugmentations
from src.utils.utils_functions import split_by_ratio


class IRMASDataModule(pl.LightningDataModule):
    train_size: int
    val_size: int
    test_size: int
    train_dataset: IRMASDatasetTrain
    test_dataset: IRMASDatasetTest
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
        train_dirs: list[Path] = [config_defaults.PATH_IRMAS_TRAIN],
        val_dirs: list[Path] = [config_defaults.PATH_IRMAS_VAL],
        test_dirs: list[Path] = [config_defaults.PATH_IRMAS_TEST],
        train_only_dataset: bool = False,
        normalize_audio: bool = config_defaults.DEFAULT_NORMALIZE_AUDIO,
        concat_two_samples: bool = SupportedAugmentations.CONCAT_TWO
        in config_defaults.DEFAULT_AUGMENTATIONS,
        use_weighted_train_sampler=config_defaults.DEFAULT_USE_WEIGHTED_TRAIN_SAMPLER,
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
        self.train_only = train_only_dataset
        self.normalize_audio = normalize_audio
        self.concat_two_samples = concat_two_samples
        self.use_weighted_train_sampler = use_weighted_train_sampler
        self.setup()

    def prepare_data(self) -> None:
        """Has to be implemented to avoid object has no attribute 'prepare_data_per_node' error."""

    def get_sample_class_weights(self, dataset: torch.utils.data.Dataset):
        """For n samples in the dataset the function returns n weights which are caculated as 1 /
        class_count of a particular example."""

        print("Caculating sample classes")
        one_hots = []
        for _, one_hot, _ in tqdm(dataset):
            one_hots.append(one_hot)
        one_hots = torch.stack(one_hots)
        class_counts = torch.sum(one_hots, dim=0)
        weight_per_class = 1 / class_counts
        examples_weights = one_hots * weight_per_class
        _, example_max_weight = examples_weights.max(dim=-1, keepdim=False)
        return example_max_weight

    def setup(self, stage=None):
        super().setup(stage)

        self.train_dataset = IRMASDatasetTrain(
            dataset_dirs=self.train_dirs,
            audio_transform=self.train_audio_transform,
            normalize_audio=self.normalize_audio,
            concat_two_samples=self.concat_two_samples,
        )

        if self.train_only:
            self.test_dataset = self.train_dataset
        else:
            self.test_dataset = IRMASDatasetTest(
                dataset_dirs=self.test_dirs,
                audio_transform=self.val_audio_transform,
                normalize_audio=self.normalize_audio,
            )

        if self.train_only:
            indices = np.arange(len(self.train_dataset))
            train_indices, val_indices = train_test_split(indices, test_size=0.2)
            test_indices = np.array([])
            # test_indices = val_indices
            self._sanity_check_difference(train_indices, val_indices)
        else:
            train_indices = np.arange(len(self.train_dataset))
            val_test_indices = np.arange(len(self.test_dataset))
            val_indices, test_indices = split_by_ratio(val_test_indices, 0.8, 0.2)

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

        if self.use_weighted_train_sampler and stage not in ["fit", "test", "predict"]:
            samples_weight = self.get_sample_class_weights(self.train_dataset)
            self.train_sampler = WeightedRandomSampler(
                samples_weight, len(samples_weight)
            )
        else:
            self.train_sampler = SubsetRandomSampler(train_indices.tolist())
        self.val_sampler = SubsetRandomSampler(val_indices.tolist())
        self.test_sampler = SubsetRandomSampler(test_indices.tolist())

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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.train_sampler,
            drop_last=self.drop_last_sample,
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
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.test_sampler,
            drop_last=self.drop_last_sample,
        )
