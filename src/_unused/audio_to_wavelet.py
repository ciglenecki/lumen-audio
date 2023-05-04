import abc

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import gausspulse, ricker


class GeneralWaveletForm(abc.ABC):
    @abc.abstractmethod
    def __init__(self, n_pts, time_res):
        pass

    @abc.abstractmethod
    def __call__(self, scale):
        pass


class GaussianWavelet(GeneralWaveletForm):
    def __init__(self, n_pts, time_res):
        self.n_pts = n_pts
        self.time_res = time_res
        self.t = time_res * np.linspace(-1, 1, n_pts)

    def __call__(self, scale):
        return gausspulse(self.t, fc=1 / scale)


class DerivativeGaussWavelet(GeneralWaveletForm):
    def __init__(self, n_pts, time_res):
        self.n_pts = n_pts
        self.time_res = time_res
        self.t = time_res * np.linspace(-1, 1, n_pts)

    def __call__(self, scale):
        return (
            -self.t
            * np.exp(-0.5 * self.t**2 / scale**2)
            / (np.sqrt(2 * np.pi))
            / scale**2
        )


class RickerWavelet(GeneralWaveletForm):
    def __init__(self, n_pts, time_res):
        self.n_pts = n_pts
        self.time_res = time_res
        self.t = time_res * np.linspace(-1, 1, n_pts)

    def __call__(self, scale):
        a = int(self.n_pts * (scale / self.time_res))
        return ricker(self.n_pts, a=a)


class WaveletConv(nn.Module):
    def __init__(
        self,
        sampling_rate: int = 22050,
        n_scales: int = 128,
        min_sf: float = 1e-2,
        max_sf: float = 2,
        wavelet_form: GeneralWaveletForm = GaussianWavelet,
        time_resolution=0.256,
    ):
        """
        time_resolution: sliding window length in seconds
        sample_rate: sampling rate in Hz
        min_sf: used to define minimum scale of wavelet: min_scale = min_sf*time_resolution
        max_sf: used to define minimum scale of wavelet: max_scale = max_sf*time_resolution
        stride: number of points to stride the sliding window
        wavelet_form: GeneralWavletfFrom, needs args n_pts and time_res
        """

        super().__init__()
        self.sampling_rate = sampling_rate
        self.kernel_size = 2048
        self.time_resolution = time_resolution
        self.min_s = self.time_resolution * min_sf
        self.max_s = self.time_resolution * max_sf
        self.n_scales = n_scales
        self.stride = (
            512  # int(2*self.sampling_rate/129)#int(stride_in_time * sampling_rate)
        )
        self.wavelet_form = wavelet_form(
            n_pts=self.kernel_size, time_res=self.time_resolution
        )

        # Create the wavelet filters
        self.filters = nn.Parameter(
            torch.zeros((self.n_scales, 1, self.kernel_size), dtype=torch.float32)
        )
        scales = torch.logspace(
            np.log10(self.min_s), np.log10(self.max_s), self.n_scales
        )
        self.filters_data = []
        for scale in scales:
            gaussian = self.wavelet_form(scale)
            self.filters_data.append(gaussian)

        self.filters_data = torch.stack(self.filters_data, dim=0).unsqueeze(1)
        self.conv = nn.Conv1d(
            1,
            self.n_scales,
            kernel_size=self.kernel_size,
            stride=self.stride,
            bias=False,
        )
        self.conv.weight.data = self.filters_data.float()

    def forward(self, x):
        # Add a dimension to the input tensor for the channel dimension
        with torch.no_grad():
            x = x.unsqueeze(1)

            # Convolve the input with the wavelet filters
            x = self.conv(x)

            # Take the absolute value of the output to get the magnitude of the wavelet coefficients
            x = torch.abs(x)

            # Remove the channel dimension from the output
            x = x.squeeze(1)

        return x
