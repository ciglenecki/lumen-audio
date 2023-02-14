from typing import Literal

import numpy as np
import torch


def multi_hot_instruments(instruments: list[str], instrument_to_idx: dict[str, int]):
    indices = np.array([instrument_to_idx[instrument] for instrument in instruments])
    return multi_hot_indices(indices, len(instrument_to_idx))


def multi_hot_indices(indices: np.ndarray | list, size: int):
    if type(indices) is list:
        indices = np.array(indices)
    array = np.zeros(size, dtype=np.int8)
    array[indices] = 1
    return array
