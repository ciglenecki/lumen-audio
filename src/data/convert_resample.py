import argparse
import os
from itertools import chain
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm

from src.config.config_defaults import AUDIO_EXTENSIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=Path("data/irmas/train"), type=Path)
    parser.add_argument("--sampling-rate", default=16_000, type=int)
    args = parser.parse_args()

    directory: Path = args.dir
    sampling_rate: int = args.sampling_rate
    directory_out = Path(directory.parent, f"{directory.stem}_{sampling_rate}_hz")
    directory_out.mkdir(exist_ok=True)

    glob_expressions = [f"*.{ext}" for ext in AUDIO_EXTENSIONS]
    glob_generators = chain(
        *[directory.rglob(glob_exp) for glob_exp in glob_expressions]
    )

    audio_paths = [filename for filename in glob_generators]
    print("Saving to", str(directory_out))

    for audio_path in tqdm(audio_paths):
        audio_base_path = audio_path.relative_to(directory)
        audio, _ = librosa.load(audio_path, sr=sampling_rate, mono=True)
        output = Path(directory_out, audio_base_path)
        parent_audio_path = output.parent
        parent_audio_path.mkdir(exist_ok=True)
        sf.write(output, audio, samplerate=sampling_rate)


if __name__ == "__main__":
    main()
