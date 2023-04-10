import os
import subprocess
from pathlib import Path, PurePath
from tempfile import NamedTemporaryFile

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pydub
import torch
import torch_audiomentations
import torchaudio
from pydub.utils import get_player_name

from src.config import config_defaults
from src.config.config import config
from src.utils.utils_functions import print_tensor


def caculate_spectrogram_width_for_one_second(sampling_rate: int, hop_size: int):
    return (sampling_rate / hop_size) + 1


def caculate_audio_max_seconds_for_image_width(
    sampling_rate: int, hop_size: int, image_width: int
) -> float:
    audio_seconds = image_width / ((sampling_rate / hop_size) + 1)
    return audio_seconds


def stereo_to_mono(audio: torch.Tensor | np.ndarray):
    if isinstance(audio, torch.Tensor):
        return torch.mean(audio, dim=0).unsqueeze(0)
    elif isinstance(audio, np.ndarray):
        return librosa.to_mono(audio)


def time_stretch(audio: np.ndarray, min_stretch, max_stretch, trim=True):
    """Audio stretch with random offset + trimming which ensures that the stretched waveform will
    be the same length as the original.

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
    audio_path: Path | str,
    method: str = "librosa",
    normalize=True,
    target_sr: int | None = config.sampling_rate,
) -> tuple[torch.Tensor | np.ndarray, int]:
    """Performs loading of the audio file.

    Depending on the arguments, the function will normalize, mono and resample the audio.
    """

    if method == "librosa":
        waveform, original_sr = librosa.load(audio_path, sr=target_sr, mono=True)
        if normalize:
            waveform = librosa.util.normalize(waveform)

    elif method == "torch":
        # default normalize for torch is True
        waveform, original_sr = torchaudio.load(audio_path, normalize=normalize)
        torch.mean(waveform, dim=0, keepdim=waveform)
        if target_sr is not None:
            torchaudio.functional.resample(
                waveform, orig_freq=original_sr, new_freq=target_sr
            )

    return_sr = target_sr if target_sr is not None else original_sr
    return waveform, return_sr


def spec_to_npy(spectrogram: torch.Tensor):
    assert (
        spectrogram.dim() <= 3
    ), "Shape can't be larger than 3, if it can, implement it"
    if spectrogram.dim() == 3 and spectrogram.shape[0] == 1:
        # single spectrogram with extra dimension
        spectrogram = spectrogram.squeeze(dim=0)
    return spectrogram.numpy()


# TODO: FIX all plots and add comments
def plot_spec_general(spectrogram: np.ndarray, sr: int, type="mel", fmax=8000):
    spectrogram = spec_to_npy(spectrogram)
    if len(spectrogram.shape) == 3:
        spectrograms = [spectrogram[i] for i in spectrogram]
    else:
        spectrograms = [spectrogram]

    for s in spectrograms:
        # s = librosa.power_to_db(s, ref=np.max)
        img = librosa.display.specshow(s, x_axis="time", y_axis=type, sr=sr, fmax=fmax)
        plt.title("Mel spectrogram display")
        plt.colorbar(img)
    plt.show()


def plot_spec_general_no_scale(mel_spectrogram: np.ndarray, sr: int):
    # mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    img = librosa.display.specshow(mel_spectrogram, x_axis="time", sr=sr)
    plt.title("Mel spectrogram display")
    plt.colorbar(img)
    plt.show()


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


def plot_mel_spectrogram(mel_spectrogram: np.ndarray, sr: int, fmax=16_000):
    # mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    img = librosa.display.specshow(
        mel_spectrogram, y_axis="mel", x_axis="time", sr=sr, fmax=fmax
    )
    plt.title("Mel spectrogram display")
    plt.colorbar(img, format="%+2.f dB")
    plt.show()


def librosa_to_pydub(waveform: np.ndarray, sr: int) -> pydub.AudioSegment:
    waveform_int = np.array(waveform * (1 << 15), dtype=np.int16)
    audio_segment = pydub.AudioSegment(
        data=waveform_int.tobytes(),
        frame_rate=sr,
        sample_width=waveform_int.dtype.itemsize,
        channels=1,
    )
    return audio_segment


def play_with_ffplay_suppress(seg, max_seconds: float | None = None):
    # https://stackoverflow.com/questions/37028671/pydub-and-aplay-suppress-verbose-output
    PLAYER = get_player_name()

    with NamedTemporaryFile("w+b", suffix=".wav") as f:
        seg.export(f.name, "wav")
        devnull = open(os.devnull, "w")
        args = [PLAYER]
        if max_seconds:
            args.extend(["-t", str(max_seconds)])
        args.extend(["-nodisp", "-autoexit", "-hide_banner", f.name])
        subprocess.call(args, stdout=devnull, stderr=devnull)


def play_audio(
    audio: torch.Tensor | np.ndarray | Path | str,
    sr: int = None,
    max_seconds: float | None = None,
):
    print("Playing audio...")
    if isinstance(audio, PurePath) or type(audio) is str:
        waveform, sr = librosa.load(audio, sr=None)
    elif isinstance(audio, torch.Tensor):
        assert sr is not None, "Provide sr argument"
        waveform = audio.numpy()
    elif type(audio) is np.ndarray:
        assert sr is not None, "Provide sr argument"
        waveform = audio
    audio_segment = librosa_to_pydub(waveform, sr)
    play_with_ffplay_suppress(audio_segment, max_seconds)


def example_audio_mel_audio():
    fname = "data/irmas/test/1992 - Blind Guardian - Somewhere Far Beyond - The Bard's Song (In The Forest)-10.wav"
    audio, original_sr = load_audio_from_file(
        fname,
        method="librosa",
        normalize=True,
    )
    target_sr = 16_000

    audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    print_tensor(audio, "audio")
    spec = librosa.feature.melspectrogram(
        y=audio, sr=target_sr, n_mels=128, n_fft=400, hop_length=160
    )
    print_tensor(spec, "spec")

    audio_reconstructed = librosa.feature.inverse.mel_to_audio(
        spec, sr=16_000, n_fft=400, hop_length=160
    )
    print_tensor(audio_reconstructed, "audio_reconstructed")

    play_audio(audio, sr=16_000, max_seconds=3)
    play_audio(audio_reconstructed, sr=16_000, max_seconds=3)


def ast_feature_inverse(spectrogram: torch.Tensor):
    n_fft = config.n_fft
    hop = config.hop_length
    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        sample_rate=config.sampling_rate,
    )
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, hop_length=hop, n_iter=54
    )

    with torch.enable_grad():
        spectrogram = spectrogram.clone().cpu()[0, :, :].T
        tmp = inverse_mel(spectrogram)
        audio = griffin_lim(tmp)

    return audio.squeeze(0).numpy()


if __name__ == "__main__":
    fname = "data/irmas_sample/[gel][jaz_blu]0907__2.wav"
    waveform, sr = librosa.load(fname, sr=22100)
    # play_audio(waveform, sr, 2)
    waveform = torch.tensor(waveform).unsqueeze(0).unsqueeze(0)
    a = torch_audiomentations.TimeInversion(p=1)
    b = a(waveform, sr).squeeze(0).squeeze(0).numpy()
    play_audio(b, sr)
