from pathlib import Path

import numpy as np
import torch

from src.config import defaults
from src.enums.enums import SupportedDatasets
from src.utils.utils_exceptions import InvalidArgument, InvalidDataException


def encode_instruments(instruments: list[str]) -> np.ndarray:
    """Returns multi hot encoded array.

    Example
        instruments = ["gel", "flu"]
        returns [0,0,0,1,0,0,0,1,0]
    """
    size = defaults.DEFAULT_NUM_LABELS
    indices = [defaults.INSTRUMENT_TO_IDX[i] for i in instruments]
    return multi_hot_encode(indices=indices, size=size)


def decode_instruments(multi_hot_array: np.ndarray) -> list[str]:
    """Return [unknown_drum, drum/no_drum]
    Example
        instruments = [0,0,0,1,0,0,0,1,0]
        returns ["gel", "flu"]
    """
    indices = np.where(multi_hot_array)[0]
    instruments = [defaults.IDX_TO_INSTRUMENT[i] for i in indices]
    return instruments


def decode_instruments_idx(multi_hot_array: np.ndarray) -> np.ndarray:
    """Return [unknown_drum, drum/no_drum]
    Example
        instruments = [0,0,0,1,0,0,0,1,0]
        returns [3, 7]
    """
    return np.where(multi_hot_array)[0]


def encode_drums(drum: str | None) -> np.ndarray:
    """Return [unknown_drum, drum/no_drum]
    Example
        drum = None
        returns [1, 0]

    Example
        drum = DrumKeys.IS_PRESENT
        returns [0, 1]

    Example
        drum = DrumKeys.NOT_PRESENT
        genre = None
        returns [0, 0]
    """
    array = np.zeros(len(defaults.DRUMS_TO_IDX), dtype=np.int8)

    if drum is None or drum == "---":
        drum = defaults.DrumKeys.UNKNOWN.value

    if not drum == defaults.DrumKeys.NOT_PRESENT.value:
        drum_index = defaults.DRUMS_TO_IDX[drum]  # "unknown-dru" or DrumKeys.IS_PRESENT
        array[drum_index] = 1

    return array


def decode_drums(one_hot: np.ndarray) -> str:
    """Return key from one hot encoded drum vector."""
    indices = np.where(one_hot == 1)[0]

    if len(indices) == 0:
        return defaults.DrumKeys.NOT_PRESENT.value

    if len(indices) != 1:
        raise InvalidDataException(f"Has to be one hot encoded vector {indices}")

    i = indices[0]
    return defaults.IDX_TO_DRUMS[i]


def encode_genre(genre: str | None) -> np.ndarray:
    """Return [unknown_drum, drum/no_drum, cou-fol, cla, pop-roc, lat-sou]. If unknown_drum is 1
    drum/no_drum should be ignored?

    Example
        genre = "cou-fol"
        returns [1, 0, 1, 0, 0, 0]

    Example
        genre = "cou-fol"
        returns [0, 1, 1, 0, 0, 0]

    Example
        genre = None
        returns [0, 0, 0, 0, 0, 0]

    Args:
        genre: <genre> | None

    Returns:
        np.ndarray: multi-hot label of size `size`
    """

    array = np.zeros(len(defaults.GENRE_TO_IDX), dtype=np.int8)
    if genre is None:
        return array
    genre_index = defaults.GENRE_TO_IDX[genre]
    array[genre_index] = 1
    return array


def decode_genre(one_hot: np.ndarray) -> str:
    """Return key from one hot encoded genre vector."""
    indices = np.where(one_hot == 1)[0]
    if len(indices) == 0:
        return "unknown-genre"

    elif len(indices) != 1:
        raise InvalidDataException(f"Has to be one hot encoded vector {indices}")

    i = indices[0]
    return defaults.IDX_TO_GENRE[i]


def multi_hot_encode(indices: np.ndarray | list, size: int) -> np.ndarray:
    """Returns mutli-hot encoded label vector for given indices.

    Example
        indices = [1, 2]
        size = 5
        returns [0, 1, 1, 0, 0]

    Args:
        indices:
        size: _description_

    Returns:
        np.ndarray: multi-hot label of size `size`
    """
    if type(indices) is list:
        indices = np.array(indices)
    array = np.zeros(size, dtype=np.int8)
    array[indices] = 1
    return array


def collate_fn_spectrogram(
    examples: list[tuple[torch.Tensor], torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Example:
    batch size = 3
    max_size = 2

    Batch is tuple (spectrogram image, label)
    batch = [
        [[3,4], [5,6]], [1,0,0]
        [[2,3], [1,7]], [0,0,1]
        [[9,3], [2,1]], [0,1,0]
    ]
    """

    all_audio_chunks, all_labels, file_indices = [], [], []
    for i, (audio_chunks, label) in enumerate(examples):
        for audio_chunk in audio_chunks:
            all_audio_chunks.append(audio_chunk.unsqueeze(0))
            all_labels.append(label.unsqueeze(0))
            file_indices.append(i)

    return (
        torch.cat(tuple(all_audio_chunks), dim=0),
        torch.cat(tuple(all_labels), dim=0),
        torch.tensor(file_indices),
    )


def chunk_collate_audio(
    batch: list[tuple[torch.Tensor, torch.Tensor]], max_audio_width
):
    """Batch is tuple (waveform, label)

    Example:

        batch size = 3
        max_size = 2

        batch = [
            [5, 6, 1, 2, 3], [1,0,0]
            [1, 5, 3, 8, 9], [0,0,1]
            [2, 5, 3, 1, 7], [0,1,0]
        ]

        output_audio_batch = [
            [5, 6],
            [1, 2],
            [3, 0],
              ....,
            [3, 1],
            [7, 0],
        ]

        output_lables_batch = [
            [1,0,0],
            [1,0,0],
            [1,0,0],
            .......,
            [0,1,0],
            [0,1,0]
        ]

        output_labels = [
            0,
            0,
            0,
            ...,
            2,
            2
        ]
        output = output_image_batch, output_lables_batch, output_labels
    """
    audio_chunks, labels, file_indices = [], [], []
    for file_idx, (audio, label) in enumerate(batch):
        chunks = list(torch.split(audio, max_audio_width))
        audio_chunks.extend(chunks)  # [[5,6], [1,2], [3]]
        repeat_times = len(chunks)

        repeated_labels = label.repeat([repeat_times, 1])
        labels.extend(repeated_labels)  # [[1,0,0], [1,0,0], [1,0,0]]

        file_id_repeated = torch.full([repeat_times], file_idx)
        file_indices.extend(file_id_repeated)  # [0, 0, 0]

    audio_chunks = torch.nn.utils.rnn.pad_sequence(
        audio_chunks,
        batch_first=True,
        padding_value=0.0,
    )  # Shape [chunks, width, height]

    labels = torch.vstack(labels)  # Shape [audio_chunks, labels]
    file_indices = torch.vstack(file_indices)  # Shape [audio_chunks, 1]

    return audio_chunks, labels, file_indices


def parse_dataset_enum_dirs(
    string: str,
) -> list[tuple[SupportedDatasets, Path]]:
    pair = string.split(":")
    if len(pair) != 2:
        raise InvalidArgument(
            f"Pair {pair} needs to have two elements. First arg is {list(SupportedDatasets)} and the second is the path "
        )
    dataset_name, dataset_path = pair
    dataset = SupportedDatasets(dataset_name)
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise InvalidArgument(f"Dataset path {dataset_path} doesn't exist.")
    return dataset, dataset_path


def calc_instrument_weight(per_instrument_count: dict[str, int], as_tensor=True):
    """Caculates weight for each class in the following way: count all negative samples and divide
    them with positive samples. Positive is the same label, negative is a different one.

    Example:
        guitar: 50       70/50
        flute: 30        90/30
        piano: 40        80/40
    """

    instruments = [k.value for k in defaults.InstrumentEnums]
    weight_dict = {}
    total = 0
    for count in per_instrument_count.values():
        total += count

    for instrument in instruments:
        positive = per_instrument_count[instrument]
        negative = total - positive
        weight_dict[instrument] = negative / positive

    if as_tensor:
        weights = torch.zeros(defaults.DEFAULT_NUM_LABELS)
        for instrument in weight_dict.keys():
            instrument_idx = defaults.INSTRUMENT_TO_IDX[instrument]
            weights[instrument_idx] = weight_dict[instrument]
        return weights
    else:
        return weight_dict


if __name__ == "__main__":
    assert decode_genre(encode_genre(None)) == "unknown-genre"
    for genre in defaults.GENRE_TO_IDX.keys():
        assert decode_genre(encode_genre(genre)) == genre
    assert decode_drums(encode_drums(None)) == "unknown-dru"
