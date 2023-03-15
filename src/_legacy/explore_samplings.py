"""Checks if all audio files are indeed sampled at 44_100."""
import wave

import config_defaults as cd

val_dict = {}

for file_name in cd.PATH_VAL.rglob("*.wav"):
    with wave.open(str(file_name), "rb") as wave_file:
        frame_rate = wave_file.getframerate()
        if frame_rate not in val_dict:
            val_dict[frame_rate] = 0
        val_dict[frame_rate] += 1

train_dict = {}

for file_name in cd.PATH_TRAIN.rglob("*.wav"):
    with wave.open(str(file_name), "rb") as wave_file:
        frame_rate = wave_file.getframerate()
        if frame_rate not in train_dict:
            train_dict[frame_rate] = 0
        train_dict[frame_rate] += 1
