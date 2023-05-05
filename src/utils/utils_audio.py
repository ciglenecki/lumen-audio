import math
import os
import subprocess
from math import ceil
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

from src.utils.utils_functions import print_tensor


def caculate_spectrogram_width_for_waveform(num_audio_samples: int, hop_size: int):
    return math.floor((num_audio_samples / hop_size) + 1)


def caculate_spectrogram_duration_in_seconds(
    sampling_rate: int, hop_size: int, image_width: int
) -> float:
    audio_seconds = ((image_width - 1) * hop_size) / sampling_rate
    return audio_seconds


def spec_width_to_num_samples(image_width: int, hop_size: int):
    return (image_width - 1) * hop_size


def stereo_to_mono(audio: torch.Tensor | np.ndarray):
    if isinstance(audio, torch.Tensor):
        return torch.mean(audio, dim=0).unsqueeze(0)
    elif isinstance(audio, np.ndarray):
        return librosa.to_mono(audio)


def iron_audios(audios: list[np.ndarray], target_width: int):
    """Add first audio to last audio so that last audio has equal length as all audios. Assumption:
    all audios in list have the same size except the last one. If there's only one audio it will
    repeat itself.

        target_width = 100
        audios: |100|100|55|
        returns |100|100|100|

        target_width = 100
        audios: |55|
        returns |100|

    Args:
        audios: _description_
    """
    last_chunk = audios[-1]
    last_chunk_length = len(last_chunk)
    diff = target_width - last_chunk_length

    # Find how many times should first chunk be repeated (usually >1 for first chunk == first chunk)
    first_chunk: torch.Tensor = audios[0]
    first_chunk_width = first_chunk.shape[-1]
    num_first_chunk_repeats = max(1, ceil(diff / first_chunk_width))
    repeated_first_chunk = np.concatenate(
        [first_chunk] * num_first_chunk_repeats, axis=-1
    )

    # Remove remove excess width caused by repeating
    repeated_first_chunk = repeated_first_chunk[..., :diff]
    audios[-1] = np.concatenate((audios[-1], repeated_first_chunk), axis=-1)
    return audios


def repeat_self_to_length(features: torch.Tensor, new_width: int):
    """
    Resizes the width of a batch of features to a new width.

    Args:
        features (torch.Tensor): Batch of features with shape (B, C, H, W).
        new_width (int): Desired new width of features.

    Returns:
        torch.Tensor: Batch of resized features with shape (B, C, H, new_width).
    """
    # Get the current width of the features

    current_width = features.shape[-1]

    if current_width == new_width:
        return features

    num_other_dims = features.ndim - 1

    # If the new width is larger, repeat the first part of the feature to fill the width
    if new_width > current_width:
        diff = new_width - current_width
        num_repeats = max(1, ceil(diff / current_width))
        ones = [1] * num_other_dims
        repeated_features = features.repeat(*ones, num_repeats)[..., :diff]
        resized_features = torch.cat([features, repeated_features], dim=-1)

    # If the new width is smaller, cut the width
    elif new_width < current_width:
        resized_features = features[..., :new_width]

    return resized_features


def time_mask_audio(
    audio: np.ndarray | torch.Tensor, percentage: float, fill_value: float = 0
):
    """Sets random percentage of audio to zeros but zeros are consecutive.

    Args:
        audio: original audio
        percentage: percentage of audio which will be set to 0

    Returns:
        _type_: _description_
    """
    audio_len = audio.shape[-1]
    num_zero = int(audio_len * percentage)
    # Choose a random starting index for the sequence of zeros
    start_index = np.random.randint(audio_len - num_zero)
    # Set the selected indices to zero
    audio[..., start_index : start_index + num_zero] = fill_value
    return audio


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
    target_sr: int | None,
    method: str = "librosa",
    normalize=True,
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
        waveform = torch.mean(waveform, dim=0, keepdim=False)
        if target_sr is not None:
            torchaudio.functional.resample(
                waveform, orig_freq=original_sr, new_freq=target_sr
            )

    return_sr = target_sr if target_sr is not None else original_sr
    return waveform, return_sr


def spectrogram_to_list(
    spectrograms: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    n_mels: int,
) -> np.ndarray:
    """Send one or multiple spectrograms [height, weight] and return as [batch, height, weight]"""
    if isinstance(spectrograms, np.ndarray):
        spectrograms = torch.tensor(spectrograms)

    if isinstance(spectrograms, torch.Tensor) and len(spectrograms.shape) == 2:
        spectrograms = spectrograms.unsqueeze(0)

    if isinstance(spectrograms, torch.Tensor) and len(spectrograms.shape) == 3:
        spectrograms = [s for s in spectrograms]

    if isinstance(spectrograms, np.ndarray) and len(spectrograms.shape) == 3:
        spectrograms = [torch.tensor(s) for s in spectrograms]

    if isinstance(spectrograms, np.ndarray):
        spectrograms = torch.tensor(spectrograms)

    if isinstance(spectrograms, list) and isinstance(spectrograms[0], np.ndarray):
        spectrograms = [torch.tensor(s) for s in spectrograms]

    spectrograms = [s.T for s in spectrograms]  # [w, h]
    spectrograms = pad_sequence(spectrograms, batch_first=True)  # [w, h]
    spectrograms = [s.T for s in spectrograms]  # [h, w]
    spectrograms = torch.stack(spectrograms)  # [b, h, w]
    spectrograms = spectrograms.cpu()

    return spectrograms.numpy()


def plot_spectrograms(
    spectrograms: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    sampling_rate: int,
    hop_length: int,
    n_fft: int,
    n_mels: int,
    titles: list[str] | None = None,
    y_axis: Literal[None, "linear", "fft", "hz", "log", "mel"] = "mel",
    use_power_to_db=True,
    block_plot=True,
):
    """Plot one or multiple spectrograms."""
    spectrograms = spectrogram_to_list(spectrograms, n_mels)

    # Prepare sizes, nrows, ncols
    batch_size = len(spectrograms)
    if titles is not None and len(titles) != batch_size:
        assert False, "There should be n titles or None"
    sqrt = math.ceil(math.sqrt(batch_size))
    n_rows = sqrt
    n_cols = sqrt
    fig = plt.figure(figsize=(13, 9))

    # AST scale
    norm = plt.Normalize(-1.25, 1) if y_axis is None else None

    format_str = "%+2.f dB" if y_axis is None else None
    # y_axis_name = "%+2.f dB" if y_axis is None else None
    # Plot each spectrogram
    for i, spec in enumerate(spectrograms):
        title = titles[i] if titles is not None else ""
        ax = plt.subplot(n_rows, n_cols, i + 1)
        if use_power_to_db and y_axis is not None:
            spec = librosa.power_to_db(spec, ref=np.max)
        img = librosa.display.specshow(
            spec,
            y_axis=y_axis,
            x_axis="s",
            sr=sampling_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            norm=norm,
        )
        plt.title(title, loc="left")

        # Add an Axes to the right of the main Axes.
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="2%", pad="2%")
        fig.colorbar(img, cax=cax, format=format_str)

    plt.tight_layout()
    plt.show(block=block_plot)
    # plt.close()


def audios_to_mel_spectrograms(
    audio: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    sampling_rate: int,
    hop_length: int,
    n_fft: int,
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
        librosa.feature.melspectrogram(
            y=a, sr=sampling_rate, hop_length=hop_length, n_fft=n_fft
        )
        for a in audio
    ]
    batched_spectrograms = np.stack(spectrograms)
    return batched_spectrograms


def audio_melspectrogram_plot(
    audio: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    sampling_rate: int,
    hop_length: int,
    n_fft: int,
    n_mels: int,
    titles: list[str] | None = None,
):
    """Plot spectrogram(s) from audio signal(s)"""
    spectrograms = audios_to_mel_spectrograms(
        audio=audio,
        sampling_rate=sampling_rate,
        hop_length=hop_length,
        n_fft=n_fft,
    )
    plot_spectrograms(
        spectrograms,
        sampling_rate=sampling_rate,
        hop_length=hop_length,
        n_fft=n_fft,
        n_mels=n_mels,
        titles=titles,
    )


def librosa_to_pydub(waveform: np.ndarray, sampling_rate: int) -> pydub.AudioSegment:
    waveform_int = np.array(waveform * (1 << 15), dtype=np.int16)
    audio_segment = pydub.AudioSegment(
        data=waveform_int.tobytes(),
        frame_rate=sampling_rate,
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
    sampling_rate: int = None,
    max_seconds: float | None = None,
):
    print("Playing audio...")
    if isinstance(audio, PurePath) or type(audio) is str:
        waveform, sampling_rate = librosa.load(audio, sr=None)
    elif isinstance(audio, torch.Tensor):
        assert sampling_rate is not None, "Provide sampling_rate argument"
        waveform = audio.cpu().numpy()
    elif type(audio) is np.ndarray:
        assert sampling_rate is not None, "Provide sampling_rate argument"
        waveform = audio
    audio_segment = librosa_to_pydub(waveform, sampling_rate)
    play_with_ffplay_suppress(audio_segment, max_seconds)


def example_audio_mel_audio():
    fname = "data/irmas/test/1992 - Blind Guardian - Somewhere Far Beyond - The Bard's Song (In The Forest)-10.wav"
    audio, original_sr = load_audio_from_file(
        fname,
        target_sr=None,
        method="librosa",
        normalize=True,
    )
    target_sr = 16_000

    audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    print_tensor(audio, "audio")
    spec = librosa.feature.melspectrogram(
        y=audio, sampling_rate=target_sr, n_mels=128, n_fft=400, hop_length=160
    )
    print_tensor(spec, "spec")

    audio_reconstructed = librosa.feature.inverse.mel_to_audio(
        spec, sr=16_000, n_fft=400, hop_length=160
    )
    print_tensor(audio_reconstructed, "audio_reconstructed")

    play_audio(audio, sampling_rate=16_000, max_seconds=3)
    play_audio(audio_reconstructed, sampling_rate=16_000, max_seconds=3)


def ast_spec_to_audio(
    spectrogram: torch.Tensor,
    n_fft: int,
    sampling_rate: int,
    hop_length: int,
):
    # spectrogram = (spectrogram * 4.5689974 * 2) - 4.2677393

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
    waveform, sampling_rate = librosa.load(fname, sr=22100)
    # play_audio(waveform, sampling_rate, 2)
    waveform = torch.tensor(waveform).unsqueeze(0).unsqueeze(0)
    a = torch_audiomentations.TimeInversion(p=1)
    b = a(waveform, sampling_rate).squeeze(0).squeeze(0).numpy()
    play_audio(b, sampling_rate)
