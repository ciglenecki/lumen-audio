from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio


def stereo_to_mono(audio: torch.Tensor | np.ndarray):
    if isinstance(audio, torch.Tensor):
        return torch.mean(audio, dim=0).unsqueeze(0)
    elif isinstance(audio, np.ndarray):
        return librosa.to_mono(audio)


def time_stretch(audio: np.ndarray, min_stretch, max_stretch, trim=True):
    """Audio stretch with random offset + trimming.

    A------B-------C        original
    A---------B---------C   streched
    ---------B------        trim

    Args:
        audio: _description_
        min_stretch: minimal stretch factor
        max_stretch: maximum stretch factor
        trim: trim the audio to original legnth
    """

    stretch_rate = np.random.uniform(min_stretch, max_stretch)
    size_before = len(audio)
    audio = librosa.effects.time_stretch(y=audio, rate=stretch_rate)

    if not trim:
        return audio

    diff = max(len(audio) - size_before, 0)
    offset = np.random.randint(0, diff + 1)
    audio_offset = audio[offset:]
    audio_trimmed = librosa.util.fix_length(audio_offset, size=size_before)
    return audio_trimmed


def load_audio_from_file(
    audio_path: Path | str, method: str = "torch", normalize=True
) -> tuple[torch.Tensor | np.ndarray, int]:
    if method == "librosa":
        waveform, original_sr = librosa.load(audio_path, sr=None)
    elif method == "torch":
        waveform, original_sr = torchaudio.load(audio_path, normalize=normalize)
    return waveform, original_sr


def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


def plot_spectrogram_librosa(spec):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spec, ax=ax)
    fig.colorbar(img, ax=ax)
    plt.show()
