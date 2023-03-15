import argparse
import inspect
import os
import random
import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
import torch
import yaml

T = TypeVar("T")


def parse_kwargs(kwargs_strs: list[str], list_sep=",", key_value_sep="="):
    """

    Example:
        kwargs_str = stretch_factors=0.8,1.2 freq_mask_param=30
        returns {"stretch_factors": [0.8, 1.2], "freq_mask_param": 30}

    Args:
        kwargs_str: _description_
        list_sep: _description_..
        arg_sep: _description_..
    """

    def parse_value(value: str):
        if isint(value):
            return int(value)
        if isfloat(value):
            return float(value)
        return value

    kwargs = {}
    for key_value in kwargs_strs:
        _kv = key_value.split(key_value_sep)
        assert (
            len(_kv) == 2
        ), f"Exactly one {key_value_sep} should appear in {key_value}"
        key, value = _kv
        value = [parse_value(v) for v in value.split(list_sep)]
        value = value if len(value) > 1 else value[0]
        kwargs[key] = value
    return kwargs


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


def isfloat(x: str):
    try:
        a = float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True


def isint(x: str):
    try:
        a = float(x)
        b = int(x)
    except (TypeError, ValueError):
        return False
    else:
        return a == b


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


def serialize_functions(*rest):
    current = rest[len(rest) - 1]
    rest = rest[:-1]
    return lambda x: current(serialize_functions(*rest)(x) if rest else x)


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
    def from_string(cls, s, do_except=False):
        try:
            return cls(s)
        except Exception:
            if do_except:
                raise ValueError(s)
            else:
                print(f"Skipping enum parsing {s}")

    def __str__(self) -> str:
        first = super().__str__()
        return f"{first}: '{self.value}'"


def to_yaml(data):
    return yaml.dump(data, allow_unicode=True, default_flow_style=False)


def function_kwargs(func):
    return inspect.getfullargspec(func)


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
