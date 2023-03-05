from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, SubsetRandomSampler

import src.config_defaults as config_defaults
from src.audio_transform import AudioTransformAST, AudioTransformBase
from src.dataset import IRMASDatasetTest, IRMASDatasetTrain, IRMASDatasetTrainMultiTask
from src.utils_functions import split_by_ratio


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
        train_audio_transform: AudioTransformAST,
        val_audio_transform: AudioTransformAST,
        train_dirs: list[Path] = [config_defaults.PATH_TRAIN],
        val_dirs: list[Path] = [config_defaults.PATH_VAL],
        test_dirs: list[Path] = [config_defaults.PATH_TEST],
        multi_task: bool = config_defaults.DEFAULT_MULTI_TASK,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_fraction = dataset_fraction
        self.drop_last_sample = drop_last_sample
        self.train_audio_transform: AudioTransformAST = train_audio_transform
        self.val_audio_transform: AudioTransformAST = val_audio_transform
        self.prepare_data_per_node = False
        self.train_dirs = train_dirs
        self.val_dirs = val_dirs
        self.test_dirs = test_dirs
        self.multi_task = multi_task
        self.setup()

    def prepare_data(self) -> None:
        """Has to be implemented to avoid object has no attribute 'prepare_data_per_node' error."""

    def setup(self, stage=None):
        super().setup(stage)

        train_constructor = (
            IRMASDatasetTrainMultiTask if self.multi_task else IRMASDatasetTrain
        )
        self.train_dataset = train_constructor(
            dataset_dirs=self.train_dirs,
            audio_transform=self.train_audio_transform,
        )

        self.test_dataset = IRMASDatasetTest(
            dataset_dirs=self.test_dirs,
            audio_transform=self.val_audio_transform,
        )

        train_indices = np.arange(len(self.train_dataset))
        val_test_indices = np.arange(len(self.test_dataset))
        val_indices, test_indices = split_by_ratio(val_test_indices, 0.5, 0.5)

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

        self._sanity_check_indices(val_indices, test_indices)

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

        self.train_sampler = SubsetRandomSampler(train_indices.tolist())
        self.val_sampler = SubsetRandomSampler(val_indices.tolist())
        self.test_sampler = SubsetRandomSampler(test_indices.tolist())

    def _sanity_check_indices(
        self,
        val_indices: np.ndarray,
        test_indices: np.ndarray,
    ):
        """Checks if there are overlaping val and test indicies to avoid data leakage."""

        for ind_a, ind_b in combinations([val_indices, test_indices], 2):
            assert (
                len(np.intersect1d(ind_a, ind_b)) == 0
            ), f"Some indices share an index {np.intersect1d(ind_a, ind_b)}"
        set_ind = set(val_indices)
        set_ind.update(test_indices)
        assert len(set_ind) == (
            len(val_indices) + len(test_indices)
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
