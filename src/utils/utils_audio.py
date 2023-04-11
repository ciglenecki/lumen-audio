import math
import os
import subprocess
from pathlib import Path, PurePath
from tempfile import NamedTemporaryFile
from typing import Literal

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pydub
import torch
import torch_audiomentations
import torchaudio
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from pydub.utils import get_player_name
from torch.nn.utils.rnn import pad_sequence

from src.config.config import config
from src.utils.utils_functions import print_tensor


def caculate_spectrogram_width_for_one_second(sampling_rate: int, hop_size: int):
    return (sampling_rate / hop_size) + 1


def caculate_spectrogram_duration_in_seconds(
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


def spectrogram_batchify(
    spectrograms: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
) -> np.ndarray:
    """Send one or multiple spectrograms [height, weight] and return as [batch, height, weight]"""
    if isinstance(spectrograms, list):
        spectrograms = np.array(spectrograms)
    if isinstance(spectrograms, torch.Tensor):
        spectrograms = spectrograms.detach().cpu().numpy()
    if not isinstance(spectrograms, np.ndarray):
        assert False, "Invalid type"
    if len(spectrograms.shape) == 2:
        spectrograms = [spectrograms]
    elif len(spectrograms.shape) > 3:
        assert False, "spectrograms has to be 1D or 2D (batch) {spectrograms.shape}"
    # Make all spectrograms equal size
    spectrograms = torch.tensor(spectrograms)
    spectrograms = pad_sequence(spectrograms, batch_first=True)

    # Make sure it's [batch, height, width]
    if spectrograms.shape[1] != config.n_mels:
        spectrograms = torch.permute(spectrograms, (0, 2, 1))
    if spectrograms.shape[1] != config.n_mels:
        assert False, f"Check spectrogram dimensions {spectrograms.shape}"
    return spectrograms.numpy()


def plot_spectrograms(
    spectrograms: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    sr=config.sampling_rate,
    titles: list[str] | None = None,
    y_axis: Literal[None, "linear", "fft", "hz", "log", "mel"] = "mel",
    hop_length=config.hop_length,
    n_fft=config.n_fft,
    use_power_to_db=True,
):
    """Plot one or multiple spectrograms."""
    spectrograms = spectrogram_batchify(spectrograms)

    # Prepare sizes, nrows, ncols
    batch_size = len(spectrograms)
    if titles is not None and len(titles) != batch_size:
        assert False, "There should be n titles or None"
    sqrt = math.ceil(math.sqrt(batch_size))
    n_rows = sqrt
    n_cols = sqrt
    fig = plt.figure(figsize=(13, 9))

    # AST scale
    norm = plt.Normalize(-1.25, 1.25) if y_axis is None else None
    # Plot each spectrogram
    for i, spec in enumerate(spectrograms):
        title = titles[i] if titles is not None else ""
        ax = plt.subplot(n_rows, n_cols, i + 1)
        if use_power_to_db:
            spec = librosa.power_to_db(spec, ref=np.max)
        img = librosa.display.specshow(
            spec,
            y_axis=y_axis,
            x_axis="time",
            sr=sr,
            hop_length=hop_length,
            n_fft=n_fft,
            norm=norm,
        )

        # Add an Axes to the right of the main Axes.
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="2%", pad="2%")
        fig.colorbar(img, cax=cax, format="%+2.f dB")

        plt.title(title)
    plt.tight_layout()
    plt.show()


def audios_to_mel_spectrograms(
    audio: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    sr=config.sampling_rate,
    hop_length=config.hop_length,
    n_fft=config.n_fft,
):
    """Convert audio(s) to mel spectrogram(s)"""
    if isinstance(audio, list):
        audio = np.array(audio)
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    if not isinstance(audio, np.ndarray):
        assert False, "Invalid type"
    if len(audio.shape) == 1:
        audio = [audio]
    elif len(audio.shape) > 2:
        assert False, "Audio has to be 1D or 2D (batch)"
    spectrograms = [
        librosa.feature.melspectrogram(y=a, sr=sr, hop_length=hop_length, n_fft=n_fft)
        for a in audio
    ]
    batched_spectrograms = np.stack(spectrograms)
    return batched_spectrograms


def audio_melspectrogram_plot(
    audio: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    sr=config.sampling_rate,
    hop_length=config.hop_length,
    n_fft=config.n_fft,
    titles: list[str] | None = None,
):
    """Plot spectrogram(s) from audio signal(s)"""
    spectrograms = audios_to_mel_spectrograms(
        audio, sr, hop_length=hop_length, n_fft=n_fft
    )
    plot_spectrograms(spectrograms, sr=sr, titles=titles)


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


def ast_spec_to_audio(
    spectrogram: torch.Tensor,
    n_fft=config.n_fft,
    sampling_rate=config.sampling_rate,
    hop_length=config.hop_length,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_fft = n_fft
    hop = hop_length
    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        sample_rate=sampling_rate,
    )
    inverse_mel = inverse_mel
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, hop_length=hop, n_iter=32
    )
    griffin_lim = griffin_lim

    with torch.enable_grad():
        spectrogram = torch.permute(spectrogram.clone().detach().cpu(), (0, 2, 1))
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
