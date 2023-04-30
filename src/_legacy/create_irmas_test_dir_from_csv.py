"""Copies."""
import argparse
import csv
import os
import shutil


def copy_files(csv_file, output_dir):
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_path = row["file"]
            txt_file_path = row["txt_file"]
            label = None
            for col in reader.fieldnames:
                if col != "file" and col != "txt_file" and int(row[col]) == 1:
                    label = col
                    break
            if label is None:
                continue
            output_label_dir = os.path.join(output_dir, label)
            if not os.path.exists(output_label_dir):
                os.makedirs(output_label_dir)
            file_name = os.path.basename(file_path)
            output_path = os.path.join(output_label_dir, file_name)
            shutil.copyfile(file_path, output_path)
            txt_file_name = os.path.basename(txt_file_path)
            txt_output_path = os.path.join(output_label_dir, txt_file_name)
            shutil.copyfile(txt_file_path, txt_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Path to input CSV file")
    parser.add_argument("output_dir", help="Path to output directory")
    args = parser.parse_args()

    copy_files(args.csv_file, args.output_dir)
