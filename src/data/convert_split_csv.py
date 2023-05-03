import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path, help="Path to the input CSV file")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.2,
        help="Fraction of rows to be used for the test set",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input_csv)

    # Calculate the number of rows to be used for the test set
    num_test_rows = int(args.fraction * len(data))

    # Randomly shuffle the DataFrame
    data = data.sample(frac=1, random_state=42)

    # Split the data into train and test sets
    train_data = data[num_test_rows:]
    test_data = data[:num_test_rows]

    # Save the train and test sets to CSV files in the same directory as the input file
    input_dir = args.input_csv.parent
    original_name = args.input_csv.stem
    train_file = Path(input_dir, f"{original_name}_trainn_{len(train_data)}.csv")
    test_file = Path(input_dir, f"{original_name}_testn_{len(test_data)}.csv")
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)


if __name__ == "__main__":
    main()
