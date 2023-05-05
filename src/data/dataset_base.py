from __future__ import annotations

import random
from abc import abstractmethod
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

import src.config.config_defaults as config_defaults
from src.features.audio_transform_base import AudioTransformBase
from src.utils.utils_audio import load_audio_from_file
from src.utils.utils_dataset import decode_instruments, encode_instruments
from src.utils.utils_functions import dict_without_keys

DatasetInternalItem = tuple[Path, np.ndarray]
DatasetGetItem = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class DatasetBase(Dataset[DatasetGetItem]):
    all_instrument_indices = np.array(list(config_defaults.INSTRUMENT_TO_IDX.values()))
    dataset_path: Path

    def __init__(
        self,
        dataset_path: Path,
        audio_transform: AudioTransformBase | None,
        num_classes: int,
        sampling_rate: int,
        normalize_audio: bool,
        sum_n_samples: int | None,
        concat_n_samples: int | None,
        train_override_csvs: list[Path] | None,
    ):
        self.dataset_path = dataset_path
        self.audio_transform = audio_transform
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.normalize_audio = normalize_audio
        self.sum_n_samples = sum_n_samples
        self.concat_n_samples = concat_n_samples
        self.train_override_csvs = train_override_csvs
        self.use_concat = concat_n_samples is not None and concat_n_samples > 1
        self.use_sum = sum_n_samples is not None and sum_n_samples > 1

        self.dataset_list: list[tuple[Path, np.ndarray]] = self.create_dataset_list()
        if self.train_override_csvs:
            self.override_with_csv()
        self.instrument_idx_list: dict[
            str, list[int]
        ] = self.create_instrument_idx_list()
        self.stats = self.caculate_stats()
        print(
            yaml.dump(
                dict_without_keys(self.stats, config_defaults.ALL_INSTRUMENTS),
                default_flow_style=False,
            )
        )

        self.set_need_to_sample()

        assert (
            self.dataset_list
        ), "Property `dataset_list` (type: list[tuple[Path, np.ndarray]]) should be set in create_dataset_list() function."

    def set_need_to_sample(self):
        example_sr = librosa.get_samplerate(self.dataset_list[0][0])
        self.need_to_resample: bool = example_sr != self.sampling_rate

    @abstractmethod
    def create_dataset_list(self) -> list[tuple[Path, np.ndarray]]:
        """Please implement this class so that `self.dataset_path` becomes a list of tuples which
        contains paths and multihot encoded labels. Use `self.dataset_path` (directory or a .csv
        file) to load and save files to a list.

        e.g. [("file1.wav", [1,0,0,0,0]), ("file2.wav", [0,1,1,0,1])]
        """

    def override_with_csv(self):
        dfs = [pd.read_csv(csv_path) for csv_path in self.train_override_csvs]
        df = pd.concat(dfs, ignore_index=True)
        df.set_index("filename", inplace=True)

        override_dict = {}
        for path_str, row in df.iterrows():
            inner_instrument_indices = np.where(row)[0]
            item_instruments = df.columns[inner_instrument_indices]
            labels = encode_instruments(item_instruments)
            override_dict[path_str] = labels

        for path_str, labels in override_dict.items():
            for i, (original_path, _) in enumerate(self.dataset_list):
                if str(original_path) == path_str:
                    self.dataset_list[i] = (original_path, labels)

    def create_instrument_idx_list(self) -> dict[str, list[int]]:
        """
        Creates a dictionary of instruments, values are indices of dataset items (audios)
        {
            "cel": [83, 13, 34, ...]
            "gel": [23, 10, 19, ...]
        }
        """
        instrument_idx_list = {e.value: [] for e in config_defaults.InstrumentEnums}

        for item_idx, (_, labels) in enumerate(self.dataset_list):
            item_instruments = decode_instruments(labels)
            for instrument in item_instruments:
                instrument_idx_list[instrument].append(item_idx)
        return instrument_idx_list

    def __len__(self) -> int:
        return len(self.dataset_list)

    def caculate_stats(self) -> dict:
        """Caculates dataset statistics.

        Number of audios per instrument, total size and number of instruments in audios
        """
        stats = {}
        for k, v in self.instrument_idx_list.items():
            # Set short and full name
            stats.update(
                {f"instrument {config_defaults.INSTRUMENT_TO_FULLNAME[k]}": len(v)}
            )
            stats.update({k: len(v)})

        num_of_instruments_per_sample = {}
        for _, label in self.dataset_list:
            n = np.sum(label).astype(int)
            key = f"{n} instruments"
            if key not in num_of_instruments_per_sample:
                num_of_instruments_per_sample[key] = 0
            num_of_instruments_per_sample[key] += 1
        stats.update(num_of_instruments_per_sample)
        stats.update({"total size": len(self.dataset_list)})
        return stats

    # @timeit
    def load_sample(self, item_idx: int) -> tuple[np.ndarray, np.ndarray, Path]:
        """Gets item from dataset_list and loads the audio."""
        audio_path, labels = self.dataset_list[item_idx]
        audio, _ = load_audio_from_file(
            audio_path,
            target_sr=self.sampling_rate if self.need_to_resample else None,
            method="librosa",
            normalize=self.normalize_audio,
        )
        return audio, labels, audio_path

    def get_random_sample_for_instrument(self, instrument_idx: int) -> int:
        """Returns a random sample which contains the instrument with index instrument_idx."""
        instrument = config_defaults.IDX_TO_INSTRUMENT[instrument_idx]
        random_sample_idx = random.choice(self.instrument_idx_list[instrument])
        return random_sample_idx

    def sample_n_negative_samples(
        self, n: int, original_labels: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[Path]]:
        """Returns n negative samples.

        Negative sample is sample that has different label compared to original_labels.
        """

        # Take all instruments and exclude ones from the original label
        negative_indices_pool = self.all_instrument_indices[
            ~original_labels.astype(bool)
        ]

        allow_repeating_labels = False

        # The pool of negative indices isn't large enough so we have to sample same negative indices mulitple times
        if len(negative_indices_pool) < n:
            allow_repeating_labels = True

        negative_indices = np.random.choice(
            negative_indices_pool,
            size=n,
            replace=allow_repeating_labels,
        )

        negative_audios = []
        negative_labels = []
        negative_paths = []
        for instrument_idx in negative_indices:
            sample_idx = self.get_random_sample_for_instrument(instrument_idx)
            negative_audio, negative_label, negative_path = self.load_sample(sample_idx)
            negative_audios.append(negative_audio)
            negative_labels.append(negative_label)
            negative_paths.append(negative_path)
        return negative_audios, negative_labels, negative_paths

    def _pad_with_zeros(self, audio: np.ndarray, desired_size: int):
        pad_width = (0, desired_size - len(audio))
        return np.pad(audio, pad_width, mode="constant", constant_values=0)

    def concat_and_sum(
        self,
        audios: list[np.ndarray],
        labels: list[np.ndarray],
        use_concat: bool,
        use_sum: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Performs concaternation and summation of multiple audios and multiple labels.

        Args:
            audios: list[np.ndarray]
            labels: list[np.ndarray]

        Example 1:

            concat_n_samples: 3
            sum_n_samples: 1

            audios: __x__, __a__, __b__
            returns: |__x__|__a__|__b__|

        Example 2:

            concat_n_samples: 1
            sum_n_samples: 2

            audios: __x__, __a___
            returns: |__xa__|

        Example 3:

            concat_n_samples: 3
            sum_n_samples: 2

            audios: __x__, __a__, __b__, ..., __e__

            |block1 |block2 |block3 |
            |___x___|___a___|___b___| (concated audio 1)
            |___c___|___d___|___e___| (concated audio 2)

            returns: |__xc__|__ad__|__be__| (summed audios)
        """

        if use_concat and not use_sum:
            audios = np.concatenate(audios)

        # If we want to sum n samples, split audio to n equal parts
        # Otherwise use just one long big audio
        if use_concat and use_sum:
            audios = np.concatenate(audios)
            audios = np.array_split(audios, self.sum_n_samples)

        # If at this point we have more than one audio we want to sum them
        # For summing, audios have to have equal size
        # Pad every audio with 0 to the maximum length
        if use_sum:
            max_len = max(len(a) for a in audios)
            for i, audio in enumerate(audios):
                if len(audio) != max_len:
                    audios[i] = self._pad_with_zeros(audio, max_len)
            # Now that they are equal in size we can create numpy array
            audios = np.mean(audios, axis=0)

        labels = np.logical_or.reduce(labels).astype(labels[0].dtype)
        return audios, labels

    def concat_and_sum_random_negative_samples(self, original_audio, original_labels):
        """Mines appropriate number of negative samples which are concated and summed to original
        audio.

        Negative audio sample has different label compared to original audio's label
        """

        if self.use_concat and self.use_sum:
            num_negative_sampels = self.concat_n_samples * self.sum_n_samples
            num_negative_sampels = num_negative_sampels - 1  # exclude original audio
        elif self.use_concat and not self.use_sum:
            num_negative_sampels = self.concat_n_samples
        elif not self.use_concat and self.use_sum:
            num_negative_sampels = self.sum_n_samples - 1
        else:
            return original_audio, original_labels

        # Load negative samples
        multiple_audios, multiple_labels, _ = self.sample_n_negative_samples(
            num_negative_sampels, original_labels=original_labels
        )

        # Add original audio at the beggining
        multiple_audios.insert(0, original_audio)
        multiple_labels.insert(0, original_labels)

        audio, labels = self.concat_and_sum(
            multiple_audios,
            multiple_labels,
            use_concat=self.use_concat,
            use_sum=self.use_sum,
        )

        return audio, labels

    # @timeit
    def __getitem__(self, index: int) -> DatasetGetItem:
        audio, labels, _ = self.load_sample(index)

        if self.use_concat or self.use_sum:
            audio, labels = self.concat_and_sum_random_negative_samples(audio, labels)

        # if self.audio_transform is None:
        #     return audio, labels, index

        labels = torch.tensor(labels).float()
        features = torch.tensor(audio).float()

        # features = self.audio_transform(audio)

        # Uncomment for playing audio
        # print(
        #     [
        #         config_defaults.INSTRUMENT_TO_FULLNAME[
        #             config_defaults.IDX_TO_INSTRUMENT[i]
        #         ]
        #         for i in instrument_multihot_to_idx(labels)
        #     ]
        # )
        # while True:
        #     play_audio(audio, sampling_rate=self.sampling_rate)

        return features, labels, index


if __name__ == "__main__":
    pass
