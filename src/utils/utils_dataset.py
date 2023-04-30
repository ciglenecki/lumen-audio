import re
from pathlib import Path

import numpy as np
import torch

from src.config import config_defaults
from src.utils.utils_audio import load_audio_from_file
from src.utils.utils_exceptions import InvalidDataException


def create_and_repeat_channel(images: torch.Tensor, num_repeat: int):
    # Create new dimension then repeat along it.
    return images.unsqueeze(dim=1).repeat(1, num_repeat, 1, 1)


def add_rgb_channel(images: torch.Tensor):
    return create_and_repeat_channel(images, 3)


def remove_rgb_channel(images: torch.Tensor):
    # Pick only one channel out of 3..
    return images[:, 0, :, :]


def get_example_val_sample(target_sr: int = None) -> np.ndarray:
    config = config_defaults.get_default_config()
    audio_path = Path(config.path_irmas_test, "1 - Hank's Other Bag-14.wav")
    audio, _ = load_audio_from_file(audio_path, target_sr=target_sr)
    return audio


def multihot_to_dict(multi_hot_array: np.ndarray) -> dict[str, int]:
    """Returns JSON-like structure given multi hot array
    Example:
        input: [0,0,0,1,0,0,0,1,0]
        returns {
            "cel": 0,
            ...
            "gac": 1,
            ...
        }
    """
    return {
        instrument: int(flag)
        for flag, instrument in zip(multi_hot_array, config_defaults.ALL_INSTRUMENTS)
    }


def encode_instruments(instruments: list[str]) -> np.ndarray:
    """Returns multi hot encoded array.

    Example
        instruments = ["gel", "flu"]
        returns [0,0,0,1,0,0,0,1,0]
    """
    size = config_defaults.DEFAULT_NUM_LABELS
    indices = [config_defaults.INSTRUMENT_TO_IDX[i] for i in instruments]
    return multi_hot_encode(indices=indices, size=size)


def decode_instruments(multi_hot_array: np.ndarray) -> list[str]:
    """Return [unknown_drum, drum/no_drum]
    Example
        instruments = [0,0,0,1,0,0,0,1,0]
        returns ["gel", "flu"]
    """
    indices = np.where(multi_hot_array)[0]
    instruments = [config_defaults.IDX_TO_INSTRUMENT[i] for i in indices]
    return instruments


def instrument_multihot_to_idx(multi_hot_array: np.ndarray) -> np.ndarray:
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
    array = np.zeros(len(config_defaults.DRUMS_TO_IDX), dtype=np.int8)

    if drum is None or drum == "---":
        drum = config_defaults.DrumKeys.UNKNOWN.value

    if not drum == config_defaults.DrumKeys.NOT_PRESENT.value:
        drum_index = config_defaults.DRUMS_TO_IDX[
            drum
        ]  # "unknown-dru" or DrumKeys.IS_PRESENT
        array[drum_index] = 1

    return array


def decode_drums(one_hot: np.ndarray) -> str:
    """Return key from one hot encoded drum vector."""
    indices = np.where(one_hot == 1)[0]

    if len(indices) == 0:
        return config_defaults.DrumKeys.NOT_PRESENT.value

    if len(indices) != 1:
        raise InvalidDataException(f"Has to be one hot encoded vector {indices}")

    i = indices[0]
    return config_defaults.IDX_TO_DRUMS[i]


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

    array = np.zeros(len(config_defaults.GENRE_TO_IDX), dtype=np.int8)
    if genre is None:
        return array
    genre_index = config_defaults.GENRE_TO_IDX[genre]
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
    return config_defaults.IDX_TO_GENRE[i]


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


def calc_instrument_weight(per_instrument_count: dict[str, int], as_tensor=True):
    """Caculates weight for each class in the following way: count all negative samples and divide
    them with positive samples. Positive is the same label, negative is a different one.

    Example:
        guitar: 50       70/50
        flute: 30        90/30
        piano: 40        80/40
    """

    instruments = [k.value for k in config_defaults.InstrumentEnums]
    weight_dict = {}
    total = 0
    for count in per_instrument_count.values():
        total += count

    for instrument in instruments:
        positive = per_instrument_count[instrument]
        negative = total - positive
        weight_dict[instrument] = negative / positive

    if as_tensor:
        weights = torch.zeros(config_defaults.DEFAULT_NUM_LABELS)
        for instrument in weight_dict.keys():
            instrument_idx = config_defaults.INSTRUMENT_TO_IDX[instrument]
            weights[instrument_idx] = weight_dict[instrument]
        return weights
    else:
        return weight_dict


def test_genre_encode_decode():
    assert decode_genre(encode_genre(None)) == "unknown-genre"
    for genre in config_defaults.GENRE_TO_IDX.keys():
        assert decode_genre(encode_genre(genre)) == genre
    assert decode_drums(encode_drums(None)) == "unknown-dru"
