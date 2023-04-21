from __future__ import annotations

import bisect
from itertools import combinations
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils
import torch.utils.data
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    SequentialSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
)
from tqdm import tqdm

from src.data.dataset_base import DatasetBase, DatasetGetItem, DatasetInternalItem
from src.data.dataset_irmas import IRMASDatasetTest, IRMASDatasetTrain
from src.enums.enums import SupportedDatasetDirType
from src.features.audio_transform_base import AudioTransformBase


class IRMASDataModule(pl.LightningDataModule):
    train_size: int
    val_size: int
    test_size: int

    class_count_dict: dict[str, int]

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
        collate_fn: Callable | None,
        train_paths: list[tuple[SupportedDatasetDirType, Path]] | None,
        val_paths: list[tuple[SupportedDatasetDirType, Path]] | None,
        test_paths: list[tuple[SupportedDatasetDirType, Path]] | None,
        train_only_dataset: bool,
        normalize_audio: bool,
        normalize_image: bool,
        concat_n_samples: int | None,
        sum_two_samples: bool,
        use_weighted_train_sampler,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_fraction = dataset_fraction
        self.drop_last_sample = drop_last_sample
        self.train_audio_transform: AudioTransformBase = train_audio_transform
        self.val_audio_transform: AudioTransformBase = val_audio_transform
        self.prepare_data_per_node = False
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths
        self.train_only_dataset = train_only_dataset
        self.normalize_audio = normalize_audio
        self.normalize_image = normalize_image
        self.concat_n_samples = concat_n_samples
        self.sum_two_samples = sum_two_samples
        self.use_weighted_train_sampler = use_weighted_train_sampler
        self.collate_fn = collate_fn

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self._train_stats: dict | None = None
        self._val_stats: dict | None = None
        self._test_stats: dict | None = None

        self.train_sampler: SubsetRandomSampler | None = None
        self.val_sampler: SequentialSampler | None = None
        self.test_sampler: SequentialSampler | None = None

    def prepare_data(self) -> None:
        """Has to be implemented to avoid object has no attribute 'prepare_data_per_node' error."""

    def setup(self, stage=None):
        super().setup(stage)
        if stage in ["fit"]:  # train + validate
            self.train_dataset = self.concat_datasets_from_tuples(self.train_paths)
            self.val_dataset = self.concat_datasets_from_tuples(self.val_dataset)

            assert (
                self.train_dataset is not None
            ), "Please provide --train-paths if you want to train a model."
            assert (
                self.val_dataset is not None
            ), "Please provide --val-paths if you want to train a model."

            if self.train_only_dataset:
                self.val_dataset = self.train_dataset
                indices = np.arange(len(self.train_dataset))
                train_indices, val_indices = train_test_split(indices, test_size=0.2)
                self._sanity_check_difference(train_indices, val_indices)
            else:
                train_indices = np.arange(len(self.train_dataset))
                val_indices = np.arange(len(self.val_dataset))

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

            self.train_size = len(train_indices)
            self.val_size = len(val_indices)

            if self.use_weighted_train_sampler:
                samples_weight = self.get_sample_class_weights(self.train_dataset)
                self.train_sampler = WeightedRandomSampler(
                    samples_weight, len(samples_weight)
                )
            else:
                self.train_sampler = SubsetRandomSampler(train_indices.tolist())
            self.val_sampler = SequentialSampler(val_indices.tolist())

            print("Train dataset stats\n", yaml.dump(self.get_train_dataset_stats()))
            print("Val dataset stats\n", yaml.dump(self.get_val_dataset_stats()))

        elif stage in ["predict", "test"]:
            self.test_dataset = self.get_test_dataset_concated()
            if self.test_dataset is not None:
                test_indices = np.arange(len(self.test_dataset))
            else:
                test_indices = np.array([])

            if self.dataset_fraction != 1:
                test_indices = np.random.choice(
                    test_indices,
                    int(self.dataset_fraction * len(test_indices)),
                    replace=False,
                )

            self.test_size = len(test_indices)
            self.test_sampler = SequentialSampler(test_indices.tolist())
            print("Test dataset classes", yaml.dump(self.get_test_dataset_stats()))

        self._log_indices()

    def concat_datasets_from_tuples(
        self,
        dataset_paths: list[tuple[SupportedDatasetDirType, Path]] | None,
    ) -> None | ConcatDataset:
        if dataset_paths is None:
            return None

        datasets: list[Dataset] = []
        for dataset_enum, dataset_path in dataset_paths:
            print(f"Creating {type} dataset {dataset_enum.value.upper()}")
            if dataset_enum == SupportedDatasetDirType.IRMAS_TRAIN:
                dataset = IRMASDatasetTrain(
                    dataset_path=dataset_path,
                    audio_transform=self.train_audio_transform,
                    normalize_audio=self.normalize_audio,
                    concat_n_samples=self.concat_n_samples,
                    sum_two_samples=self.sum_two_samples,
                )
                datasets.append(dataset)
            elif dataset_enum == SupportedDatasetDirType.IRMAS_TEST:
                dataset = IRMASDatasetTest(
                    dataset_path=dataset_path,
                    audio_transform=self.val_audio_transform,
                    normalize_audio=self.normalize_audio,
                    concat_n_samples=False,
                    sum_two_samples=False,
                )
                datasets.append(dataset)
            elif dataset_enum == SupportedDatasetDirType.OPENMIC:
                pass
            elif dataset_enum == SupportedDatasetDirType.OPENMIC:
                pass
            elif dataset_enum == SupportedDatasetDirType.OPENMIC:
                pass

        print()
        if len(datasets) == 0:
            return None

        return ConcatDataset(datasets)

    def get_train_dataset_concated(self) -> None | ConcatDataset:
        return self.concat_datasets_from_tuples(self.train_paths)

    def get_val_dataset_concated(self) -> None | ConcatDataset:
        return self.concat_datasets_from_tuples(self.val_paths)

    def get_test_dataset_concated(self) -> None | ConcatDataset:
        return self.concat_datasets_from_tuples(self.test_paths)

    def get_dataset_stats(self, concat_dataset: ConcatDataset):
        stats = {}
        datasets: list[DatasetBase] = concat_dataset.datasets  # type: ignore
        for dataset in datasets:
            stats_dataset = dataset.stats
            for k, v in stats_dataset.items():
                if k not in stats:
                    stats[k] = 0
                stats[k] += v
        return stats

    def train_dataloader(self) -> DataLoader[ConcatDataset[DatasetGetItem]]:
        assert (
            self.train_dataset is not None
        ), 'To access the train dataloader please call datamodule.setup("fit") after creating datamodule.'
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.train_sampler,
            drop_last=self.drop_last_sample,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader[ConcatDataset[DatasetGetItem]]:
        assert (
            self.val_dataset is not None
        ), 'To access the val dataloader please call datamodule.setup("fit") after creating datamodule.'
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.val_sampler,
            drop_last=self.drop_last_sample,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader[ConcatDataset[DatasetGetItem]]:
        assert (
            self.test_dataset is not None
        ), 'To access the test dataloader please call datamodule.setup("test") after creating datamodule.'
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.test_sampler,
            drop_last=self.drop_last_sample,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader[ConcatDataset[DatasetGetItem]]:
        assert (
            self.test_dataset is not None
        ), "Can't use predict_dataloader without test dataset"
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.test_sampler,
            drop_last=False,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

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

    def get_sample_class_weights(self, dataset: Dataset):
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

    def get_item_from_internal_structure(
        self, item_index: int, split: Literal["train", "val", "test"]
    ) -> DatasetInternalItem:
        # Find which dataset (they are concated) the exact dataset which the file originate from
        split_map = {
            "train": self.train_dataloader,
            "val": self.val_dataloader,
            "test": self.test_dataloader,
        }
        data_loader = split_map[split]()
        concated_dataset: ConcatDataset = data_loader.dataset  # type: ignore
        dataset_idx = bisect.bisect_right(concated_dataset.cumulative_sizes, item_index)
        exact_dataset: DatasetBase = concated_dataset.datasets[dataset_idx]  # type: ignore
        audio_path, label = exact_dataset.dataset_list[item_index]
        return audio_path, label

    def _log_indices(self):
        if self.train_sampler is not None:
            train_indices = self.train_sampler.indices
            print()
            print(
                "Train size",
                self.train_size,
                "indices:",
                train_indices[0:5],
                train_indices[-5:],
            )

        if self.val_sampler is not None:
            val_indices = self.val_sampler.data_source
            print(
                "Val size",
                self.val_size,
                "indices:",
                val_indices[0:5],
                val_indices[-5:],
            )
        if self.test_sampler is not None:
            test_indices = self.test_sampler.data_source
            print(
                "Test size",
                self.test_size,
                "indices:",
                test_indices[0:5],
                test_indices[-5:],
            )
            print()

    def get_train_dataset_stats(self):
        if self._train_stats is not None:
            return self._train_stats
        self._train_stats = self.get_dataset_stats(self.train_dataset)
        return self._train_stats

    def get_val_dataset_stats(self):
        if self._val_stats is not None:
            return self._val_stats
        self._val_stats = self.get_dataset_stats(self.val_dataset)
        return self._val_stats

    def get_test_dataset_stats(self):
        if self._test_stats is not None:
            return self._test_stats
        self._test_stats = self.get_dataset_stats(self.test_dataset)
        return self._test_stats
