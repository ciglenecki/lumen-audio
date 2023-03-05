import numpy as np
import torch

from src import config_defaults


class InvalidDataException(Exception):
    pass


def encode_drums(drum: str | None, return_tensors: bool = False) -> np.ndarray:
    """Return [drum/no_drum]
    Example
        drum = None
        returns -1

    Example
        drum = DrumKeys.IS_PRESENT
        returns 1

    Example
        drum = DrumKeys.NOT_PRESENT
        genre = None
        returns 0
    """
    if drum is None or drum == "---":
        drum = config_defaults.DrumKeys.UNKNOWN.value

    drum = [config_defaults.DRUMS_TO_IDX[drum]]

    return torch.tensor(drum) if return_tensors else drum


def decode_drums(value: np.ndarray) -> str | None:
    """Return key from drum value."""
    return config_defaults.IDX_TO_DRUMS[value[0]]


def encode_genre(genre: str | None, return_tensors: bool = False) -> np.ndarray:
    """Return [cou-fol, cla, pop-roc, lat-sou, jaz-blu].

    Example
        genre = "cou-fol"
        returns [1, 0, 0, 0, 0]

    Example
        genre = "cla"
        returns [0, 1, 0, 0, 0]

    Example
        genre = None
        returns [0, 0, 0, 0, 0]

    Args:
        genre: <genre> | None

    Returns:
        np.ndarray: one-hot label of size `size`
    """

    array = np.zeros(len(config_defaults.GENRE_TO_IDX), dtype=np.int8)
    if genre is None:
        return array
    genre_index = config_defaults.GENRE_TO_IDX[genre]
    array[genre_index] = 1
    return torch.tensor(array) if return_tensors else array


def decode_genre(one_hot: np.ndarray) -> tuple[str | None]:
    """Return key from one hot encoded genre vector."""
    indices = np.where(one_hot == 1)[0]
    if len(indices) == 0:
        return "unknown-genre"

    elif len(indices) != 1:
        raise InvalidDataException(f"Has to be one hot encoded vector {indices}")

    i = indices[0]
    return config_defaults.IDX_TO_GENRE[i]


def multi_hot_indices(indices: np.ndarray | list, size: int) -> np.ndarray:
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


if __name__ == "__main__":
    assert decode_drums(encode_drums(None))
    assert decode_drums(encode_drums("---"))
    assert decode_drums(encode_drums("unknown-dru"))
    assert decode_drums(encode_drums("dru"))
    assert decode_drums(encode_drums("nod"))
    assert decode_genre(encode_genre(None))
    assert decode_genre(encode_genre("cou_fol"))
    assert decode_genre(encode_genre("cla"))
    assert decode_genre(encode_genre("pop_roc"))
    assert decode_genre(encode_genre("lat_sou"))
    assert decode_genre(encode_genre("jaz_blu"))
