"""Convert IRMAS dataset to sum of three audio files dataset"""

import argparse
import os
import random
from itertools import chain
from math import ceil
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm
import soundfile as sf

from src.config.config_defaults import AUDIO_EXTENSIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=Path("data/irmas/test"), type=Path)
    parser.add_argument("--sampling-rate", default=16_000, type=int)
    parser.add_argument("--sum-n", default=3, type=int)
    args = parser.parse_args()

    num_sum = args.sum_n
    sr = args.sampling_rate
    directory = args.dir
    directory_out = Path(
        directory.parent, f"{directory.stem}_sum{num_sum}_sr_{args.sampling_rate}"
    )

    directory_out.mkdir(exist_ok=True, parents=True)

    glob_expressions = [f"*.{ext}" for ext in AUDIO_EXTENSIONS]
    glob_generators = chain(
        *[directory.rglob(glob_exp) for glob_exp in glob_expressions]
    )
    audio_files = [str(filename) for filename in glob_generators]
    random.shuffle(audio_files)

    num_iter = 0

    while len(audio_files) > 0:
        print(num_iter)
        chosen_files = []
        for _ in range(num_sum):
            if len(audio_files) < num_sum - 1:
                exit(1)
            chosen_files.append(audio_files.pop())

        audios = []
        for f in chosen_files:
            audio = librosa.load(f, mono=True, sr=sr)[0]
            meter = pyloudnorm.Meter(sr)
            loudness = meter.integrated_loudness(audio)
            audio = pyloudnorm.normalize.loudness(audio, loudness, -12)
            audio = librosa.util.normalize(audio)
            audios.append(audio)

        # Make all audios equally long
        max_len = max(len(a) for a in audios)
        for i, a in enumerate(audios):
            num_repeats = max(1, ceil(max_len / len(a)))
            a = np.tile(a, num_repeats)[:max_len]
            audios[i] = a

        assert all([len(a) == max_len for a in audios])

        # sum audios
        summed_audio = sum(audios)

        # create a new filename for the combined audio file
        new_filename = f"track{num_iter + 1}.wav"

        # save the combined audio file
        out_name = os.path.join(directory_out, new_filename)
        sf.write(out_name, summed_audio, samplerate=sr)

        # create a new label for the combined audio file
        labels = set()
        for audio_path in chosen_files:
            path_without_ext = os.path.splitext(audio_path)[0]
            txt_path = path_without_ext + ".txt"
            print(txt_path)
            with open(txt_path) as f:
                for line in f:
                    instrument = line.rstrip("\n").replace("\t", "")
                    labels.add(instrument)
        label = "\n".join(labels)
        print(label)

        # create a new filename for the label file
        label_filename = f"track{num_iter + 1}.txt"

        # save the label file
        with open(os.path.join(directory_out, label_filename), "w") as f:
            f.write(label)
        num_iter = num_iter + 1


if __name__ == "__main__":
    main()
