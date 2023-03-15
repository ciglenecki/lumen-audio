from abc import ABC, abstractmethod

import librosa
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchaudio.transforms import (
    FrequencyMasking,
    MelScale,
    Spectrogram,
    TimeMasking,
    TimeStretch,
)
from transformers import ASTFeatureExtractor

import src.config_defaults as config_defaults
from src.utils_functions import EnumStr, MultiEnum


def stereo_to_mono(audio: torch.Tensor | np.ndarray):
    if isinstance(audio, torch.Tensor):
        return audio.sum(dim=1) / 2
    elif isinstance(audio, np.ndarray):
        return audio.sum(axis=-1) / 2


class AudioTransformBase(ABC):
    """Base class for all audio transforms. Ideally, each audio transform class should be self
    contained and shouldn't depened on the outside context.

    Audio transfrom can be model dependent. We can create audio transforms which work only for one
    model and that's fine.
    
    Sampling rate defines the target sampling rate.
    """
    
    @abstractmethod
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    @abstractmethod
    def process(
        self,
        audio: np.ndarray,
        labels: torch.Tensor | np.ndarray,
        orig_sampling_rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Function which prepares everything for model's .forward() function. It creates the
        spectrogram from audio and prepares the labels.

        Args:
            audio: audio data
            labels: _description_
            orig_sampling_rate: _description_

        Returns:
            tuple[torch.Tensor, torch.Tensor]: _description_
        """


class AudioTransformAST(AudioTransformBase):

    """Resamples audio, converts it to mono, does AST feature extraction which extracts spectrogram
    (mel filter banks) from audio."""

    def __init__(
        self,
        ast_pretrained_tag=config_defaults.DEFAULT_AST_PRETRAINED_TAG,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(ast_pretrained_tag)

    def process(
        self,
        audio: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        orig_sampling_rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels)

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().numpy()

        audio_mono = librosa.to_mono(audio)
        audio_resampled = librosa.resample(
            audio_mono,
            orig_sr=orig_sampling_rate,
            target_sr=self.sampling_rate,
        )
        features = self.feature_extractor(
            audio_resampled,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )

        # mel filter banks
        spectrogram = features["input_values"].squeeze(dim=0)
        return spectrogram, labels
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#############################################################################################################################################################

class AudioTransformMelSpectrogram(AudioTransformBase):
    """Resamples audio and extracts melspectrogram from audio."""

    def __init__(
        self,
        n_fft: int = config_defaults.DEFAULT_N_FFT,
        hop_length: int = config_defaults.DEFAULT_HOP_LENGTH,
        n_mels: int = config_defaults.DEFAULT_N_MELS,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def process(
        self,
        audio: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        orig_sampling_rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels)

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().numpy()

        audio_mono = librosa.to_mono(audio)
        audio_resampled = librosa.resample(
            audio,
            orig_sr=orig_sampling_rate,
            target_sr=self.sampling_rate,
        )

        spectrogram = librosa.feature.melspectrogram(
            y=audio_resampled,
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        return spectrogram, labels


class AudioTransformMelSpectrogramResize(AudioTransformMelSpectrogram):
    """Resamples audio, extracts melspectrogram from audio, resizes it to the given dimensions."""

    def __init__(
        self,
        dim: tuple[int, int],
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def process(
        self,
        audio: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        orig_sampling_rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        spectrogram, labels = super().process(
            audio, labels, orig_sampling_rate
        )

        spectrogram = spectrogram.reshape(1, 1, *spectrogram.shape)
        spectrogram = F.resize(torch.tensor(spectrogram), size=self.dim, antialias=True)
        spectrogram = spectrogram.reshape(1, *self.dim)
        
        return spectrogram, labels

class AudioTransformMelSpectrogramFixed(AudioTransformMelSpectrogram):
    """Resamples audio, extracts melspectrogram from audio and pads the original spectrogram to dimension of spectrogram
    for max_len sequence."""

    def __init__(
        self,
        max_len: int, 
        dim: tuple[int, int],
        *args,  **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.max_len = max_len
        self.dim = dim
        
        FAKE_SAMPLE_RATE = 44_100
        dummy_audio = np.random.random(size=(max_len * FAKE_SAMPLE_RATE,))
        audio_resampled = librosa.resample(
            dummy_audio,
            orig_sr=FAKE_SAMPLE_RATE,
            target_sr=self.sampling_rate,
        )

        spectrogram = librosa.feature.melspectrogram(
            y=audio_resampled,
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        
        self.seq_dim = spectrogram.shape
        

    def process(
        self,
        audio: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        orig_sampling_rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        spectrogram, labels = super().process(
            audio, labels, orig_sampling_rate
        )
        
        spectrogram_padded = np.zeros(self.seq_dim)
        w, h = spectrogram.shape
        spectrogram_padded[:w, :h] = spectrogram
        spectrogram_padded = spectrogram_padded.reshape(1, 1, *self.seq_dim)
        spectrogram_padded = F.resize(torch.tensor(spectrogram_padded), size=self.dim, antialias=True)
        spectrogram_padded = spectrogram_padded.reshape(1, *self.dim)

        return spectrogram_padded.type(torch.float32), labels


class AudioTransformMelSpectrogramFixedRepeated(AudioTransformMelSpectrogramFixed):
    """Calls AudioTransformMelSpectrogramFixed and repeats the output 3 times. This is useful for mocking RGB channels."""

    def __init__(self, repeat=3, **kwargs):
        super().__init__(**kwargs)
        self.repeat = repeat

    def process(
        self,
        audio: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        orig_sampling_rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        spectrogram, labels = super().process(
            audio, labels, orig_sampling_rate
        )
        spectrogram = spectrogram.repeat(1, self.repeat, 1, 1)[0]

        return spectrogram, labels
    
    
class AudioTransformMelSpectrogramResizedRepeated(AudioTransformMelSpectrogramFixed):
    """Calls AudioTransformMelSpectrogramResize and repeats the output 3 times. This is useful for mocking RGB channels."""

    def __init__(self, repeat=3, **kwargs):
        super().__init__(**kwargs)
        self.repeat = repeat

    def process(
        self,
        audio: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        orig_sampling_rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        spectrogram, labels = super().process(
            audio, labels, orig_sampling_rate
        )
        spectrogram = spectrogram.repeat(1, self.repeat, 1, 1)[0]

        return spectrogram, labels
    
    
#############################################################################################################################################################























class UnsupportedAudioTransforms(ValueError):
    pass


class AudioTransforms(EnumStr):
    """List of supported AudioTransforms we use."""

    AST = "ast"
    MEL_SPECTROGRAM_RESIZE_REPEAT = "mel_spectrogram_resize_repeat"
    MEL_SPECTROGRAM_FIXED_REPEAT = "mel_spectrogram_fixed_repeat"


def get_audio_transform(sampling_rate: int, dim: tuple[int, int], audio_transform_enum: AudioTransforms) -> AudioTransformBase:
    # TODO: check if everyone has 3.10 and switch all ifs to case
    if audio_transform_enum is AudioTransforms.AST:
        return AudioTransformAST(
            ast_pretrained_tag=config_defaults.DEFAULT_AST_PRETRAINED_TAG
        )
    
    elif audio_transform_enum is AudioTransforms.MEL_SPECTROGRAM_FIXED_REPEAT:
        return AudioTransformMelSpectrogramFixedRepeated(sampling_rate=sampling_rate, dim=dim, repeat=3, max_len=20)
    
    elif audio_transform_enum is AudioTransforms.MEL_SPECTROGRAM_RESIZE_REPEAT:
        return AudioTransformMelSpectrogramResizedRepeated(sampling_rate=sampling_rate, dim=dim, repeat=3)
    
    raise UnsupportedAudioTransforms(f"Unsupported transform {audio_transform_enum}")


if __name__ == "__main__": # for testing only
    from src.dataset import IRMASDatasetTrain
    audio_tf = AudioTransformMelSpectrogramFixed(max_len=20, dim=(384, 384), sampling_rate=22050)
    ds = IRMASDatasetTrain(audio_transform=audio_tf)
    
    x, y = ds[100]

    import matplotlib.pyplot as plt
    librosa.display.specshow(x[0].cpu().detach().numpy(), x_axis="time", y_axis="mel", sr=22050)
    plt.show()
    