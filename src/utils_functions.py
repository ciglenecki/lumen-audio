import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from enum import Enum
from math import floor
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset


class InvalidRatios(Exception):
    pass


T = TypeVar("T")


def get_dirs_only(path: Path):
    """Return only top level directories in the path."""
    return [
        d
        for d in (os.path.join(path, d1) for d1 in os.listdir(path))
        if os.path.isdir(d)
    ]


def tensor_sum_of_elements_to_one(tensor: torch.Tensor, dim):
    """Scales elements of the tensor so that the sum is 1."""
    return tensor / torch.sum(tensor, dim=dim, keepdim=True)


def split_by_ratio(
    array: np.ndarray, *ratios, use_whole_array=False
) -> list[np.ndarray]:
    """Splits the ndarray for given ratios.

    Arguments:
        array: array that will be splited
        use_whole_array: if set to True elements won't be discarted. Sum of ratios will be scaled to 1
        ratios: ratios used for splitting

    Example 1 use_whole_array = False:
        ratios = (0.2, 0.3)
        array = [1,2,3,4,5,6,7,8,9,10]
        returns [[1, 2], [3, 4, 5]]

    Example 2 use_whole_array = True:
        ratios = (0.2, 0.3)
        array = [1,2,3,4,5,6,7,8,9,10]
        returns [[1, 2, 3, 4], [5, 6, 7, 8, 9, 10]]

    Example 3 use_whole_array = False:
        ratios = (0.2)
        array = [1,2,3,4,5,6,7,8,9,10]
        returns [[1, 2]]
    """

    ratios = np.array(ratios)
    if use_whole_array:
        ratios = ratios / ratios.sum()
        ratios = np.around(ratios, 3)
    ind = np.add.accumulate(np.array(ratios) * len(array)).astype(int)
    return [x for x in np.split(array, ind)][: len(ratios)]


def get_timestamp(format="%m-%d-%H-%M-%S"):
    return datetime.today().strftime(format)


def one_hot_encode(index: int, size: int):
    zeros = np.zeros(shape=size)
    zeros[index] = 1
    return zeros


def np_set_default_printoptions():
    np.set_printoptions(
        edgeitems=3,
        infstr="inf",
        linewidth=75,
        nanstr="nan",
        precision=8,
        suppress=False,
        threshold=1000,
        formatter=None,
    )


def npy_to_image(npy):
    return npy.transpose((1, 2, 0))


def is_valid_dataset_split(array):
    if len(array) != 3 or sum(array) != 1:
        raise argparse.ArgumentError(
            array, "There has to be 3 fractions (train, val, test) that sum to 1"
        )
    return array


def is_between_0_1(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x!r} not a floating-point literal")
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x!r} not in range [0.0, 1.0]")
    return x


def is_positive_int(value):
    int_value = int(value)
    if int_value < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return int_value


def is_valid_dir(arg):
    if not os.path.isdir(arg):
        raise argparse.ArgumentError(arg, "Argument should be a path to directory")
    return arg


class SocketConcatenator:
    """Merges multiple sockets (files) into one enabling writing to multiple sockets/files as
    once."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
        self.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def stdout_to_file(file: Path):
    """Pipes standard input to standard input and to a new file."""
    print("Standard output piped to file:")
    f = open(Path(file), "w")
    sys.stdout = SocketConcatenator(sys.stdout, f)
    sys.stderr = SocketConcatenator(sys.stderr, f)


def reset_sockets():
    """Reset stdout and stderr sockets."""
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


def add_prefix_to_keys(dict: dict, prefix) -> dict:
    """
    Example:
        dict = {"a": 1, "b": 2}
        prefix = "text_"
        returns {"text_a": 1, "text_b": 2}
    """
    return {prefix + k: v for k, v in dict.items()}


def flatten(list):
    """
    Example:
        list = [[1, 2], [3, 4]]
        returns [1, 2, 3, 4]
    """
    return [item for sublist in list for item in sublist]


def print_df_sample(df: pd.DataFrame):
    pd.set_option("display.max_columns", None)
    print(
        "\nSample of the dataframe:",
        "First 3 rows:",
        df.head(n=3),
        "Random 3 rows:",
        df.sample(n=3),
        "Last 3 rows:",
        df.tail(n=3),
        "Dataframe stats:",
        df.describe(),
        sep="\n\n\n",
    )
    pd.reset_option("display.max_columns")


def timeit(func):
    def timed(*args, **kwargs):
        print("START", func.__qualname__)
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        print("END", func.__qualname__, "time:", round((te - ts) * 1000, 1), "ms")
        return result

    return timed


class EnumStr(Enum):
    @classmethod
    def keys(cls):
        return [elem.value for elem in list(cls)]

    @classmethod
    def from_string(cls, s):
        try:
            return cls(s)
        except KeyError:
            raise ValueError()


class MultiEnum(Enum):
    """
    Enum which accepts multiple values instead of a single value. This is useful alternative to dict key-value pairs.
    Example:

        class MyCubes(MultiEnum):
            GREEN = GreenCube(), 'green'

        MyCubes('green').value  # returns instance of GreenCube
        MyCubes.GREEN.value     # returns instance of GreenCube
    """

    def __new__(cls, *values):
        obj = object.__new__(cls)
        # first value is canonical value
        obj._value_ = values[0]
        for other_value in values[1:]:
            cls._value2member_map_[other_value] = obj
        obj._all_values = values  # type: ignore
        if len(values) > 1:
            obj.key = values[1]
        return obj

    @classmethod
    def keys(cls):
        return [elem.key for elem in list(cls)]

    @classmethod
    def from_string(cls, s):
        try:
            return cls(s)
        except KeyError:
            raise ValueError()

    def __repr__(self):
        return "<{}.{}: {}>".format(
            self.__class__.__name__,
            self._name_,
            ", ".join([repr(v) for v in self._all_values]),  # type: ignore
        )


def to_yaml(data):
    return yaml.dump(data, allow_unicode=True, default_flow_style=False)


nato_alphabet = [
    "Alpha",
    "Bravo",
    "Charlie",
    "Delta",
    "Echo",
    "Foxtrot",
    "Golf",
    "Hotel",
    "India",
    "Juliett",
    "Kilo",
    "Lima",
    "Mike",
    "November",
    "Oscar",
    "Papa",
    "Quebec",
    "Romeo",
    "Sierra",
    "Tango",
    "Uniform",
    "Victor",
    "Whiskey",
    "X-ray",
    "Yankee",
    "Zulu",
    "Sinisa",
    "Jan",
    "Alan",
]

adjectives = [
    "agile",
    "ample",
    "avid",
    "awed",
    "best",
    "bonny",
    "brave",
    "brisk",
    "calm",
    "clean",
    "clear",
    "comfy",
    "cool",
    "cozy",
    "crisp",
    "cute",
    "deft",
    "eager",
    "eased",
    "easy",
    "elite",
    "fair",
    "famed",
    "fancy",
    "fast",
    "fiery",
    "fine",
    "finer",
    "fond",
    "free",
    "freed",
    "fresh",
    "fun",
    "funny",
    "glad",
    "gold",
    "good",
    "grand",
    "great",
    "hale",
    "handy",
    "happy",
    "hardy",
    "holy",
    "hot",
    "ideal",
    "jolly",
    "keen",
    "lean",
    "like",
    "liked",
    "loved",
    "loyal",
    "lucid",
    "lucky",
    "lush",
    "magic",
    "merry",
    "neat",
    "nice",
    "nicer",
    "noble",
    "plush",
    "prize",
    "proud",
    "pure",
    "quiet",
    "rapid",
    "rapt",
    "ready",
    "regal",
    "rich",
    "right",
    "roomy",
    "rosy",
    "safe",
    "sane",
    "sexy",
    "sharp",
    "shiny",
    "sleek",
    "slick",
    "smart",
    "soft",
    "solid",
    "suave",
    "super",
    "swank",
    "sweet",
    "swift",
    "tidy",
    "top",
    "tough",
    "vivid",
    "warm",
    "well",
    "wise",
    "witty",
    "worth",
    "young",
]


def random_codeword():
    """Return e.g.:

    YoungAlpha, WiseZulu
    """
    return f"{random.choice(adjectives).capitalize()}{random.choice(nato_alphabet)}"


if __name__ == "__main__":
    pass
