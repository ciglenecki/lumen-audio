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
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    SubsetRandomSampler,
    WeightedRandomSampler,
)
from tqdm import tqdm

import src.config.config_defaults as config_defaults
from src.data.dataset_base import DatasetBase, DatasetGetItem, DatasetInternalItem
from src.data.dataset_irmas import IRMASDatasetTest, IRMASDatasetTrain
from src.enums.enums import SupportedDatasets
from src.features.audio_transform_base import AudioTransformBase
from src.utils.utils_functions import split_by_ratio


class IRMASDataModule(pl.LightningDataModule):
    train_size: int
    val_size: int
    test_size: int
    train_dataset: ConcatDataset
    val_dataset: ConcatDataset
    test_dataset: ConcatDataset
    train_sampler: SubsetRandomSampler
    val_sampler: SubsetRandomSampler
    test_sampler: SubsetRandomSampler
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
        train_paths: list[tuple[SupportedDatasets, Path]] | None,
        val_paths: list[tuple[SupportedDatasets, Path]] | None,
        test_paths: list[tuple[SupportedDatasets, Path]] | None,
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
        self._set_class_count_dict()
        self.setup()

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

    def _set_class_count_dict(self):
        # f there is only one dataset and that dataset is IRMAS:
        if (
            self.train_paths is not None
            and len(self.train_paths) == 1
            and self.train_paths[0][0] == SupportedDatasets.IRMAS
        ):
            self.class_count_dict = config_defaults.IRMAS_TRAIN_CLASS_COUNT
        else:
            self.class_count_dict = {}

    def prepare_data(self) -> None:
        """Has to be implemented to avoid object has no attribute 'prepare_data_per_node' error."""

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

    def _concat_datasets_from_tuples(
        self, dataset_paths: list[tuple[SupportedDatasets, Path]], type: str
    ) -> None | ConcatDataset:
        datasets: list[Dataset] = []
        for dataset_enum, dataset_path in dataset_paths:
            if dataset_enum == SupportedDatasets.IRMAS and type == "train":
                dataset = IRMASDatasetTrain(
                    dataset_path=dataset_path,
                    audio_transform=self.train_audio_transform,
                    normalize_audio=self.normalize_audio,
                    normalize_image=self.normalize_image,
                    concat_n_samples=self.concat_n_samples,
                    sum_two_samples=self.sum_two_samples,
                )
                datasets.append(dataset)
            if dataset_enum == SupportedDatasets.IRMAS and type in "val":
                dataset = IRMASDatasetTest(
                    dataset_path=dataset_path,
                    audio_transform=self.val_audio_transform,
                    normalize_audio=self.normalize_audio,
                    normalize_image=self.normalize_image,
                )
                datasets.append(dataset)
            if dataset_enum == SupportedDatasets.IRMAS and type in "test":
                pass
            if dataset_enum == SupportedDatasets.OPENMIC and type == "train":
                pass
            if dataset_enum == SupportedDatasets.OPENMIC and type == "val":
                pass
            if dataset_enum == SupportedDatasets.OPENMIC and type == "test":
                pass
        if len(datasets) == 0:
            return None
        return ConcatDataset(datasets)

    def _get_train_dataset_concated(self):
        self.train_paths = self._concat_datasets_from_tuples(self.train_paths)

    def _get_val_dataset_concated(self):
        datasets = []
        for dataset_enum, dataset_path in self.val_paths:
            if dataset_enum == SupportedDatasets.IRMAS:
                dataset = IRMASDatasetTest(
                    dataset_path=dataset_path,
                    audio_transform=self.val_audio_transform,
                    normalize_audio=self.normalize_audio,
                    normalize_image=self.normalize_image,
                )
            datasets.append(dataset)
        return ConcatDataset(datasets)

    def _get_test_dataset_concated(self):
        datasets = []
        for dataset_enum, dataset_path in self.train_paths:
            if dataset_enum == SupportedDatasets.IRMAS:
                dataset = IRMASDatasetTest(
                    dataset_path=dataset_path,
                    audio_transform=self.val_audio_transform,
                    normalize_audio=self.normalize_audio,
                    normalize_image=self.normalize_image,
                )
            datasets.append(dataset)
        return ConcatDataset(datasets)

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
        if stage == "fit":  # train + validate
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

        elif stage == "predict":
            self.test_paths
            self.train_size = len(train_indices)

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
                instrument = config_defaults.IDX_TO_INSTRUMENT[idx]
                if instrument not in output:
                    output[instrument] = 0
                output[instrument] += 1
        self.class_count_dict = output
        return output

    def train_dataloader(self) -> DataLoader[ConcatDataset[DatasetGetItem]]:
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

    def test_dataloader(self) -> DataLoader[ConcatDataset[DatasetGetItem]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.test_sampler,
            drop_last=self.drop_last_sample,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return super().predict_dataloader()
