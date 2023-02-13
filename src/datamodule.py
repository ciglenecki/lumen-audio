from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm

from dataset import IRMASDatasetTest, IRMASDatasetTrain
from utils_audio import AudioTransform, AudioTransformAST


class IRMASDataModule(pl.LightningDataModule):
    train_size: int
    val_size: int
    test_size: int

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        dataset_fraction: int,
        drop_last_sample: bool,
        audio_transform: AudioTransform
    ):
        self.train_dataset = IRMASDatasetTrain(audio_transform=audio_transform)
        self.test_dataset = IRMASDatasetTest()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_fraction = dataset_fraction
        self.drop_last_sample = drop_last_sample
        self.audio_transform: AudioTransform = AudioTransformAST()

    def setup(self, stage: str | None = None):
        train_indices = np.arange(len(self.train_dataset))
        val_indices = np.array([])  # TODO: handle
        test_indices = np.arange(len(self.test_dataset))

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

        self._sanity_check_indices(train_indices, val_indices)

        self.train_size = len(train_indices)
        self.val_size = len(val_indices)
        self.test_size = len(test_indices)

        print("Train size", self.train_size, train_indices[0:5], train_indices[-5:])
        print("Val size", self.val_size, val_indices[0:5], val_indices[-5:])
        print("Test size", self.test_size)

        self.train_sampler = SubsetRandomSampler(train_indices)
        self.val_sampler = SubsetRandomSampler(val_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)

    def _sanity_check_indices(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
    ):
        for ind_a, ind_b in combinations([train_indices, val_indices], 2):
            assert (
                len(np.intersect1d(ind_a, ind_b)) == 0
            ), "Some indices share an index {}".format(np.intersect1d(ind_a, ind_b))
        set_ind = set(train_indices)
        set_ind.update(val_indices)
        assert len(set_ind) == (
            len(train_indices) + len(val_indices)
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
        """Uses train dataset but validation sampler."""
        return DataLoader(
            self.train_dataset,
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
