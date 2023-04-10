from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from src.features.augmentations import SpectrogramAugmentation, WaveformAugmentation
from src.utils.utils_audio import load_audio_from_file
from src.utils.utils_functions import EnumStr


class UnsupportedAudioTransforms(Exception):
    pass


class AudioTransforms(EnumStr):
    """List of supported AudioTransforms we use."""

    AST = "ast"
    MEL_SPECTROGRAM_RESIZE_REPEAT = "mel_spectrogram_resize_repeat"
    MEL_SPECTROGRAM_FIXED_REPEAT = "mel_spectrogram_fixed_repeat"
    WAV2VEC = "wav2vec"
    MFCC_FIXED_REPEAT = "mfcc_fixed_repeat"
    WAV2VECCNN = "wav2veccnn"


class AudioTransformBase(ABC):
    """Base class for all audio transforms. Ideally, each audio transform class should be self
    contained and shouldn't depened on the outside context.

    Audio transfrom can be model dependent. We can create audio transforms which work only for one
    model and that's fine.
    """

    def __init__(
        self,
        sampling_rate: int,
        spectrogram_augmentation: SpectrogramAugmentation | None,
        waveform_augmentation: WaveformAugmentation | None,
    ) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.spectrogram_augmentation = spectrogram_augmentation
        self.waveform_augmentation = waveform_augmentation

    @abstractmethod
    def process(
        self,
        audio: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor]:
        """Function which prepares everything for model's .forward() function. It creates the
        spectrogram from audio.

        Args:
            audio: audio data

        Returns:
            tuple[torch.Tensor, torch.Tensor]: _description_
        """

    def process_from_file(
        self,
        audio_file_path: Path,
        method: Literal["torch", "librosa"],
        normalize: bool,
    ) -> tuple[torch.Tensor]:
        """Calls the process() but loads the file beforehand.

        Args:
            audio_path: audio file path

        Returns:
            tuple[torch.Tensor, torch.Tensor]: _description_
        """
        audio, _ = load_audio_from_file(
            audio_file_path,
            method=method,
            normalize=normalize,
            target_sr=self.sampling_rate,
        )
        return self.process(audio)
