import argparse
import os
import shutil

import numpy as np
import pandas as pd


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="The path to load data that will be sampled",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="The path to save sampled data .csv file",
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="The number of examples in total that will be sampled",
    )
    parser.add_argument(
        "--info_filename",
        type=str,
        default="info.csv",
        help="The filename for the information dataframe",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed used for sampling (important for reproducibility)",
    )
    args = parser.parse_args()
    return args


def set_seed(seed):
    np.random.seed(seed)


def move_uniform_sample(save_path, load_path, info_filename, n):
    os.makedirs(save_path, exist_ok=True)

    sample = []
    instruments = os.listdir(load_path)
    for i in instruments:
        path_to_old_dir = os.path.join(load_path, i)

        all_ex_for_class = np.array(os.listdir(path_to_old_dir))
        num_for_class = np.random.choice(
            len(all_ex_for_class), size=n // len(instruments), replace=False
        )
        sample.append(all_ex_for_class[num_for_class])

        for ex in all_ex_for_class[num_for_class]:
            src = os.path.join(*(load_path, i, ex))
            dst = os.path.join(save_path, ex)
            shutil.copy(src, dst)

    stacked = np.stack(sample, axis=1)

    info_df = pd.DataFrame(stacked, columns=instruments)
    info_df.to_csv(os.path.join(save_path, info_filename), index=False)


def main():
    args = parse()
    set_seed(args.seed)
    move_uniform_sample(args.save_path, args.load_path, args.info_filename, args.n)
    print(f"Saved sample to {args.save_path}.")


if __name__ == "__main__":
    main()
