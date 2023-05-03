import torch
import torch.nn as nn
import librosa
import src.features.wavelet as wavelet
from src.features.audio_transform_base import AudioTransformBase


class MelChroWavelet(AudioTransformBase):
    def __init__(self,
        n_fft: int,
        hop_length: int,
        image_size: tuple[int, int],
        n_mels: int,
        use_rgb: bool = True,
        normalize_audio=True,
        normalize_image=True,
        *args,
        **kwargs,
                 ) -> None:

        super().__init__(*args, **kwargs)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.image_size = image_size
        self.use_rgb = use_rgb
        self.normalize_audio = normalize_audio
        self.normalize_image = normalize_image
        self.wavelet = wavelet.WaveletConv(
                sampling_rate=self.sampling_rate,
                n_scales=self.n_mels
        )


    def __call__(self, audio):
        mel= librosa.feature.melspectrogram(
            y=audio,
            n_mels = 128,
            sr = self.sampling_rate
        )
        mel = librosa.amplitude_to_db(mel)
        mel = (mel-mel.mean())/mel.std()
        mel = torch.tensor(mel)#.unsqueeze(0)
        chroma = librosa.feature.chroma_stft(
            y=audio,
            n_chroma=self.n_mels,
            sr = self.sampling_rate
        )
        chroma = (chroma-chroma.mean())/chroma.std()
        chroma = torch.tensor(chroma)#.unsqueeze(0)
        scalogram = self.wavelet(
            torch.tensor(audio).view(1,-1),
        ).squeeze(0)
        scalogram = (scalogram - scalogram.mean())/scalogram.std()
        diff = chroma.shape[-1] - scalogram.shape[-1]
        scalogram = nn.functional.pad(
            scalogram,pad=(0,diff)
        )
        #print("scale: ",scalogram.shape)
        #print("mel: ",mel.shape)
        #print("chro: ",chroma.shape)
        
        spec = torch.stack([mel,chroma,scalogram],dim = 0)
        return spec