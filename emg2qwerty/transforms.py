# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar
import torch.nn.functional as F

import numpy as np
import torch
import torchaudio
from time import sleep


TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]


@dataclass
class ToTensor:
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    fields: Sequence[str] = ("emg_left", "emg_right")
    stack_dim: int = 1

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(data[f]) for f in self.fields], dim=self.stack_dim
        )


@dataclass
class Lambda:
    """Applies a custom lambda function as a transform.

    Args:
        lambd (lambda): Lambda to wrap within.
    """

    lambd: Transform[Any, Any]

    def __call__(self, data: Any) -> Any:
        return self.lambd(data)


@dataclass
class ForEach:
    """Applies the provided ``transform`` over each item of a batch
    independently. By default, assumes the input is of shape (T, N, ...).

    Args:
        transform (Callable): The transform to apply to each batch item of
            the input tensor.
        batch_dim (int): The bach dimension, i.e., the dim along which to
            unstack/unbind the input tensor prior to mapping over
            ``transform`` and restacking. (default: 1)
    """

    transform: Transform[torch.Tensor, torch.Tensor]
    batch_dim: int = 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.transform(t) for t in tensor.unbind(self.batch_dim)],
            dim=self.batch_dim,
        )


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclass
class RandomBandRotation:
    """Applies band rotation augmentation by shifting the electrode channels
    by an offset value randomly chosen from ``offsets``. By default, assumes
    the input is of shape (..., C).

    NOTE: If the input is 3D with batch dim (TNC), then this transform
    applies band rotation for all items in the batch with the same offset.
    To apply different rotations each batch item, use the ``ForEach`` wrapper.

    Args:
        offsets (list): List of integers denoting the offsets by which the
            electrodes are allowed to be shift. A random offset from this
            list is chosen for each application of the transform.
        channel_dim (int): The electrode channel dimension. (default: -1)
    """

    offsets: Sequence[int] = (-1, 0, 1)
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        offset = np.random.choice(self.offsets) if len(self.offsets) > 0 else 0
        return tensor.roll(offset, dims=self.channel_dim)


@dataclass
class TemporalAlignmentJitter:
    """Applies a temporal jittering augmentation that randomly jitters the
    alignment of left and right EMG data by up to ``max_offset`` timesteps.
    The input must be of shape (T, ...).

    Args:
        max_offset (int): The maximum amount of alignment jittering in terms
            of number of timesteps.
        stack_dim (int): The dimension along which the left and right data
            are stacked. See ``ToTensor()``. (default: 1)
    """

    max_offset: int
    stack_dim: int = 1

    def __post_init__(self) -> None:
        assert self.max_offset >= 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[self.stack_dim] == 2
        left, right = tensor.unbind(self.stack_dim)

        offset = np.random.randint(-self.max_offset, self.max_offset + 1)
        if offset > 0:
            left = left[offset:]
            right = right[:-offset]
        if offset < 0:
            left = left[:offset]
            right = right[-offset:]

        return torch.stack([left, right], dim=self.stack_dim)


@dataclass
class LogSpectrogram:
    """Creates log10-scaled spectrogram from an EMG signal. In the case of
    multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned spectrogram
    is of shape (T, ..., freq).

    Args:
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 frequency bins.
            (default: 64)
        hop_length (int): Number of samples to stride between consecutive
            STFT windows. (default: 16)
    """

    n_fft: int = 64
    hop_length: int = 16

    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            # Disable centering of FFT windows to avoid padding inconsistencies
            # between train and test (due to differing window lengths), as well
            # as to be more faithful to real-time/streaming execution.
            center=False,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)  # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)  # (..., C, freq, T)
        return logspec.movedim(-1, 0)  # (T, ..., C, freq)


@dataclass
class SpecAugment:
    """Applies time and frequency masking as per the paper
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech
    Recognition, Park et al" (https://arxiv.org/abs/1904.08779).

    Args:
        n_time_masks (int): Maximum number of time masks to apply,
            uniformly sampled from 0. (default: 0)
        time_mask_param (int): Maximum length of each time mask,
            uniformly sampled from 0. (default: 0)
        iid_time_masks (int): Whether to apply different time masks to
            each band/channel (default: True)
        n_freq_masks (int): Maximum number of frequency masks to apply,
            uniformly sampled from 0. (default: 0)
        freq_mask_param (int): Maximum length of each frequency mask,
            uniformly sampled from 0. (default: 0)
        iid_freq_masks (int): Whether to apply different frequency masks to
            each band/channel (default: True)
        mask_value (float): Value to assign to the masked columns (default: 0.)
    """

    n_time_masks: int = 0
    time_mask_param: int = 0
    iid_time_masks: bool = True
    n_freq_masks: int = 0
    freq_mask_param: int = 0
    iid_freq_masks: bool = True
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        self.time_mask = torchaudio.transforms.TimeMasking(
            self.time_mask_param, iid_masks=self.iid_time_masks
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            self.freq_mask_param, iid_masks=self.iid_freq_masks
        )

    def __call__(self, specgram: torch.Tensor) -> torch.Tensor:
        # (T, ..., C, freq) -> (..., C, freq, T)
        x = specgram.movedim(0, -1)

        # Time masks
        n_t_masks = np.random.randint(self.n_time_masks + 1)
        for _ in range(n_t_masks):
            x = self.time_mask(x, mask_value=self.mask_value)

        # Frequency masks
        n_f_masks = np.random.randint(self.n_freq_masks + 1)
        for _ in range(n_f_masks):
            x = self.freq_mask(x, mask_value=self.mask_value)

        # (..., C, freq, T) -> (T, ..., C, freq)
        return x.movedim(-1, 0)


# Jimmy's custom transforms:
@dataclass
class MelSpectrogram:
    """Creates mel-scaled spectrogram from an EMG signal. In the case of
    multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned spectrogram
    is of shape (T, ..., freq).

    Args:
        n_fft (int): Size of FFT. (default: 64)
        hop_length (int): Number of samples to stride between consecutive
            STFT windows. (default: 16)
        n_mels (int): Number of mel filterbanks. (default: 33)
        sample_rate (int): Sampling rate of the input signal. (default: 2000)
        f_min (float): Minimum frequency. (default: 0.0)
        f_max (float | None): Maximum frequency. (default: None)
        log (bool): Whether to apply log10 to the mel spectrogram. (default: False)
    """

    n_fft: int = 64
    hop_length: int = 16
    n_mels: int = 33  # Same number of features as LogSpectrogram (n_fft//2+1)
    sample_rate: int = 2000  # EMG sampling rate is 2kHz
    f_min: float = 0.0
    f_max: float | None = None
    log: bool = False

    def __post_init__(self) -> None:
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            normalized=True,
            center=False,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        mel_spec = self.mel_spectrogram(x)  # (..., C, n_mels, T)
        if self.log:
            mel_spec = torch.log10(mel_spec + 1e-6)  # Apply log10 to compress dynamic range
        return mel_spec.movedim(-1, 0)  # (T, ..., C, n_mels)

@dataclass
class WaveletTransform:
    """Creates a wavelet-based time-frequency representation of an EMG signal.
    In the case of multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned wavelet coefficients
    are of shape (T, ..., freq).

    Args:
        num_freq_bins (int): Number of frequency bins to output (default: 33)
        wavelet (str): Wavelet to use (default: 'morl')
        sample_rate (int): Sampling rate of the input signal (default: 2000)
        hop_length (int): Number of samples between consecutive frames (default: 16)
        min_freq (float): Minimum frequency to analyze (default: 10.0)
        max_freq (float): Maximum frequency to analyze (default: 500.0)
        log (bool): Whether to apply log10 to the wavelet scalogram (default: False)
    """

    num_freq_bins: int = 33  # Same number of features as LogSpectrogram
    wavelet: str = 'morl'  # Morlet wavelet is good for time-frequency analysis
    sample_rate: int = 2000  # EMG sampling rate is 2kHz
    hop_length: int = 16
    min_freq: float = 10.0  # Lower frequency bound for EMG
    max_freq: float = 500.0  # Upper frequency bound for EMG
    log: bool = False

    def __post_init__(self) -> None:
        try:
            import pywt
            import numpy as np
            self.pywt = pywt
            self.np = np
        except ImportError:
            raise ImportError("PyWavelets (pywt) is required for WaveletTransform")
        
        # Ensure min_freq is greater than 0 to avoid log10(0)
        if self.min_freq <= 0:
            self.min_freq = 0.1  # Set a safe minimum value
        
        # Ensure max_freq is a numeric value
        if self.max_freq is None or not isinstance(self.max_freq, (int, float)):
            self.max_freq = self.sample_rate / 2  # Use Nyquist frequency as default
        
        # Get central frequency of the wavelet
        if self.wavelet == 'morl':
            central_freq = 0.8125  # For Morlet wavelet
        else:
            wavelet = self.pywt.ContinuousWavelet(self.wavelet)
            central_freq = wavelet.center_frequency
        
        # Calculate scales from frequencies: scale = central_freq * sample_rate / freq
        freqs = self.np.logspace(
            self.np.log10(self.min_freq), 
            self.np.log10(self.max_freq), 
            self.num_freq_bins
        )
        self.scales = central_freq * self.sample_rate / freqs



    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        original_device = tensor.device
        
        # Move time dimension to the end for processing
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        
        # Convert to numpy for pywt processing
        x_np = x.detach().cpu().numpy()
        
        # Get output shape, preserving original dimensions
        shape = list(x_np.shape)
        time_dim = shape[-1]
        
        # Calculate output time dimension after downsampling
        output_time = (time_dim - 1) // self.hop_length + 1
        output_shape = shape[:-1] + [self.num_freq_bins, output_time]
        
        # Allocate output array
        wavelet_coeffs = self.np.zeros(output_shape, dtype=np.float32)
        
        # Process each channel independently
        for idx in self.np.ndindex(tuple(shape[:-1])):
            # Get signal for this specific channel
            signal = x_np[idx]
            
            # Perform wavelet transform (CWT)
            coeffs, _ = self.pywt.cwt(signal, self.scales, self.wavelet, 
                                    1.0/self.sample_rate)
            
            # Downsample in time to match hop_length
            coeffs_downsampled = coeffs[:, ::self.hop_length]
            
            # Take magnitude (like a spectrogram)
            scalogram = self.np.abs(coeffs_downsampled)
            
            # Store in output array
            wavelet_coeffs[idx] = scalogram[:, :output_time]  # Ensure exact output size
        
        # Convert back to torch tensor and move to original device
        result = torch.from_numpy(wavelet_coeffs).to(original_device)
        
        # Apply log transformation if requested
        if self.log:
            result = torch.log10(result + 1e-6)
        
        # Move time dimension back to the front
        return result.movedim(-1, 0)  # (..., freq, T) -> (T, ..., freq)





# MY CUSTOM CLASSES@dataclass


@dataclass
class FFTTransform:
    """Computes the FFT of the input tensor and normalizes the result across all channels and bins.

    Args:
        n_bins (int): Number of frequency bins to compute.
    """

    n_bins: int

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Compute the FFT
        fft_result = torch.fft.fft(tensor, n=self.n_bins, dim=0)
        
        # Compute the magnitude of the FFT result
        magnitude = torch.abs(fft_result)
        
        # Normalize the magnitude across all channels and bins
        mean = magnitude.mean()
        std = magnitude.std()
        normalized_magnitude = (magnitude - mean) / std
        
        return normalized_magnitude


@dataclass
class Normalize:
    """Normalizes the input tensor to have zero mean and unit variance across all channels.

    Args:
        mean (float): The mean value for normalization. If None, it will be computed from the data.
        std (float): The standard deviation value for normalization. If None, it will be computed from the data.
    """

    mean: float = None
    std: float = None

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            self.mean = tensor.mean().item()
        if self.std is None:
            self.std = tensor.std().item()
        return (tensor - self.mean) / self.std


@dataclass
class NormalizePerChannel:
    """Normalizes the input tensor to have zero mean and unit variance for each channel independently.

    Args:
        mean (torch.Tensor): The mean values for normalization. If None, they will be computed from the data.
        std (torch.Tensor): The standard deviation values for normalization. If None, they will be computed from the data.
    """

    mean: torch.Tensor = None
    std: torch.Tensor = None
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(tensor)

        if self.mean is None:
            self.mean = torch.where(mask, tensor, torch.tensor(0.0)).sum(dim=0, keepdim=True) / mask.sum(dim=0, keepdim=True)
            print(f"Self.mean",self.mean)
            sleep(1)
        if self.std is None:
            self.std = torch.sqrt(torch.where(mask, (tensor - self.mean) ** 2, torch.tensor(0.0)).sum(dim=0, keepdim=True) / mask.sum(dim=0, keepdim=True))
            print(f"Self.std",self.std)
            sleep(1)
        normalized_tensor = (tensor - self.mean) / self.std
        if torch.isnan(normalized_tensor).any():
            raise ValueError("NaN values found in normalized tensor")
        return normalized_tensor


    
@dataclass
class TimeStretch:
    """Stretches or compresses the time axis of the signal."""

    stretch_factor: float  # > 1 stretches, < 1 compresses
    stack_dim: int = 1     # The dimension along which the left and right data are stacked (default: 1)

    def __post_init__(self) -> None:
        assert self.stretch_factor > 0  # Stretch factor should be greater than 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Ensure tensor is in 5D shape (batch, channels, segments, length)
        if tensor.ndimension() == 5:
            # Flatten the batch and segment dimensions to apply interpolation
            tensor = tensor.view(-1, tensor.shape[2], tensor.shape[3])  # (batch * segments, channels, length)

        # Apply time stretching or compression
        stretched_tensor = F.interpolate(tensor, scale_factor=self.stretch_factor, mode='linear', align_corners=False)

        # Reshape back to original 5D shape
        if stretched_tensor.ndimension() == 3 and stretched_tensor.shape[0] != tensor.shape[0]:
            stretched_tensor = stretched_tensor.view(tensor.shape[0], tensor.shape[1], -1, stretched_tensor.shape[-1])

        return stretched_tensor
    
@dataclass
class RandomFrequencyShift:
    """Randomly shifts the frequency of the input signal."""
    
    shift_range: int = 10  # range in Hz for shifting the frequency

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Simulating frequency shift by adding/subtracting random noise in the frequency domain
        shift = np.random.randint(-self.shift_range, self.shift_range)
        return tensor.roll(shift, dims=-1)  # Assuming the last dimension represents the frequency


@dataclass
class AdditiveNoise:
    """Adds random noise to the input signal."""
    
    noise_factor: float = 0.1  # The strength of the noise

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Correct noise generation with mean and std
        noise = torch.randn_like(tensor) * self.noise_factor
        return tensor + noise

@dataclass
class RandomTimeShift:
    """Randomly shifts the signal along the time axis."""
    
    max_shift: int  # Maximum number of time steps to shift

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        return tensor.roll(shift, dims=0)  # Shifting along the time axis (T dimension)

@dataclass
class RandomCrop:
    """Randomly crops a segment of the signal."""
    
    crop_percentage: float  # Percentage of the tensor to crop (0.0 - 1.0)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        crop_size = int(tensor.shape[0] * self.crop_percentage)
        
        crop_size = max(crop_size, 1)
        
        if tensor.shape[0] <= crop_size:
            raise ValueError(f"Tensor size ({tensor.shape[0]}) is smaller than the calculated crop size ({crop_size})")
        
        start = np.random.randint(0, tensor.shape[0] - crop_size)
        
        return tensor[start:start + crop_size]
    
@dataclass
class TimeWarp:
    """Applies a time-warping transformation to the signal."""
    
    max_warp: float = 0.1  # Max warp factor (in percentage)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Create a time warping function and apply to tensor
        warp_factor = np.random.uniform(1 - self.max_warp, 1 + self.max_warp)
        time_axis = np.arange(tensor.shape[0])
        warped_time_axis = np.interp(time_axis, time_axis * warp_factor, time_axis)
        return torch.tensor(np.interp(warped_time_axis, time_axis, tensor.numpy()))


@dataclass
class RandomDropout:
    """Randomly zeroes out a percentage of the input signal."""
    
    dropout_rate: float = 0.1  # Fraction of the input signal to drop

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mask = (torch.rand_like(tensor) > self.dropout_rate).float()
        return tensor * mask
    

@dataclass
class GaussianSmoothing:
    """Applies Gaussian smoothing to the signal."""
    
    kernel_size: int = 3
    sigma: float = 1.0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        from scipy.ndimage import gaussian_filter1d
        smoothed_tensor = gaussian_filter1d(tensor.numpy(), sigma=self.sigma, axis=0)
        return torch.tensor(smoothed_tensor)
    

@dataclass
class RandomElectrodeDropout:
    """Randomly drops one or more electrode channels."""
    
    drop_rate: float = 0.2  # Percentage of electrodes to drop

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        num_channels = tensor.shape[-1]
        drop_count = int(num_channels * self.drop_rate)
        drop_indices = np.random.choice(num_channels, drop_count, replace=False)
        tensor[..., drop_indices] = 0
        return tensor
    
@dataclass
class RandomSignalScaling:
    """Randomly scales the signal."""
    
    scaling_factor_range: tuple = (0.8, 1.2)  # Min and max scaling factor

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        scale = np.random.uniform(*self.scaling_factor_range)
        return tensor * scale
