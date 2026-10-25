"""Training-only critics: an official DAC adapter and the legacy waveform model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor, nn
from torch.nn import functional as F


class WaveformDiscriminator(nn.Module):
    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("Discriminator channels must be positive")
        dimensions = (1, channels, channels * 2, channels * 4, channels * 4)
        self.layers = nn.ModuleList([
            nn.Conv1d(left, right, kernel_size=15, stride=stride, padding=7)
            for left, right, stride in zip(dimensions[:-1], dimensions[1:], (2, 2, 4, 4))
        ])
        self.logits = nn.Conv1d(dimensions[-1], 1, kernel_size=3, padding=1)

    def forward(self, waveform: Tensor) -> list[Tensor]:
        features = []
        x = waveform
        for layer in self.layers:
            x = F.leaky_relu(layer(x), negative_slope=0.2)
            features.append(x)
        return features + [self.logits(x)]


class MultiScaleDiscriminator(nn.Module):
    """Independent critics of the waveform and successively downsampled copies."""

    def __init__(self, channels: int = 16, scales: int = 3) -> None:
        super().__init__()
        if scales < 1 or scales > 5:
            raise ValueError("Discriminator scales must be between one and five")
        self.discriminators = nn.ModuleList([WaveformDiscriminator(channels) for _ in range(scales)])

    def forward(self, waveform: Tensor) -> list[list[Tensor]]:
        if waveform.ndim != 2 or waveform.shape[-1] < 1:
            raise ValueError("Discriminator requires [batch, samples] audio")
        x = waveform.unsqueeze(1)
        outputs = []
        for index, discriminator in enumerate(self.discriminators):
            if index:
                x = F.avg_pool1d(x, kernel_size=2, stride=2, ceil_mode=True)
            outputs.append(discriminator(x))
        return outputs


class DACMultiResolutionDiscriminator(nn.Module):
    """Use DAC's published complex multiband STFT critic at each resolution.

    The upstream MRD owns its layers and feature outputs. This adapter only
    chooses bands for a fixed sample rate and pads very short valid crops.
    It does not use DAC's waveform normalization or periodic discriminators,
    and is not the unreleased adaptive discriminator from the RE-USE papers.
    """

    def __init__(self, sample_rate: int, fft_sizes: Sequence[int] = (512, 1024, 2048),
                 band_width_hz: int = 4000, sample_rates: Sequence[int] = (),
                 sample_rate_scaling: bool = False) -> None:
        super().__init__()
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate < 1:
            raise ValueError("Discriminator sample_rate must be a positive integer")
        if isinstance(band_width_hz, bool) or not isinstance(band_width_hz, int) or band_width_hz < 1:
            raise ValueError("Discriminator band_width_hz must be a positive integer")
        self.fft_sizes = tuple(fft_sizes)
        if not self.fft_sizes or any(isinstance(size, bool) or not isinstance(size, int) or size < 8
                                     for size in self.fft_sizes):
            raise ValueError("Discriminator fft_sizes must contain integers >= 8")
        self.sample_rate = sample_rate
        self.band_width_hz = band_width_hz
        self.sample_rates = tuple(sample_rates) or (sample_rate,)
        if (sample_rate not in self.sample_rates or len(set(self.sample_rates)) != len(self.sample_rates)
                or any(isinstance(rate, bool) or not isinstance(rate, int) or rate < 1
                       for rate in self.sample_rates)):
            raise ValueError("sample_rates must be unique positive integers including sample_rate")
        if not isinstance(sample_rate_scaling, bool):
            raise ValueError("sample_rate_scaling must be boolean")
        self.sample_rate_scaling = sample_rate_scaling
        # Ratios are relative to Nyquist; the last band may be narrower.
        maximum_rate = max(self.sample_rates)
        boundaries = [*range(0, maximum_rate, 2 * band_width_hz), maximum_rate]
        self.bands = [(left / maximum_rate, right / maximum_rate)
                      for left, right in zip(boundaries[:-1], boundaries[1:])]
        self._rate_geometry = {}
        for rate in self.sample_rates:
            boundaries = [*range(0, rate, 2 * band_width_hz), rate]
            fractions = [(left / rate, right / rate) for left, right in zip(boundaries[:-1], boundaries[1:])]
            sizes = self.fft_sizes_for_rate(rate)
            geometry = []
            for size in sizes:
                bins = size // 2 + 1
                bands = [(int(left * bins), int(right * bins)) for left, right in fractions]
                if size < 8 or any(right <= left for left, right in bands):
                    raise ValueError("Discriminator FFT resolution leaves an empty frequency band")
                geometry.append((size, bands))
            self._rate_geometry[rate] = geometry
        try:
            from dac.model.discriminator import MRD
        except (ImportError, OSError) as error:
            raise RuntimeError("DAC discriminator requires the gan extra and compatible torch/torchaudio; "
                               "install realtime-use[gan]") from error
        self.discriminators = nn.ModuleList([
            MRD(window_length=size, sample_rate=maximum_rate, bands=self.bands)
            for size in self.fft_sizes_for_rate(maximum_rate)
        ])

    def fft_sizes_for_rate(self, sample_rate: int) -> tuple[int, ...]:
        if sample_rate not in self.sample_rates:
            raise ValueError(f"Unsupported discriminator sample rate: {sample_rate}")
        return tuple(round(size * sample_rate / self.sample_rate) if self.sample_rate_scaling else size
                     for size in self.fft_sizes)

    def forward(self, waveform: Tensor, sample_rate: int | None = None) -> list[list[Tensor]]:
        if waveform.ndim != 2 or min(waveform.shape) < 1 or not waveform.is_floating_point():
            raise ValueError("Discriminator requires nonempty floating-point [batch, samples] audio")
        rate = self.sample_rate if sample_rate is None else sample_rate
        if isinstance(rate, bool) or rate not in self._rate_geometry:
            raise ValueError(f"Unsupported discriminator sample rate: {rate}")
        outputs = []
        for (size, bands), discriminator in zip(self._rate_geometry[rate], self.discriminators):
            # AudioSignal's match_stride STFT adds reflection padding on both
            # sides, plus right alignment. A full window makes those pads legal
            # even for a one-sample crop; ordinary inputs are passed unchanged.
            padded = F.pad(waveform, (0, max(0, size - waveform.shape[-1])))
            # All band CNNs are registered before DDP. Lower-rate batches use
            # the available low-frequency bands; the official forward's zip
            # leaves higher-band parameters unused. No weights are recreated.
            original = (discriminator.sample_rate, discriminator.window_length,
                        discriminator.stft_params, discriminator.bands)
            discriminator.sample_rate, discriminator.window_length = rate, size
            discriminator.stft_params = original[2]._replace(window_length=size, hop_length=size // 4)
            discriminator.bands = bands
            try:
                outputs.append(discriminator(padded.unsqueeze(1)))
            finally:
                (discriminator.sample_rate, discriminator.window_length,
                 discriminator.stft_params, discriminator.bands) = original
        return outputs


def build_discriminator(settings: Mapping[str, Any], sample_rate: int,
                        sample_rates: Sequence[int] = ()) -> nn.Module:
    """Missing implementation preserves the architecture of older checkpoints."""
    implementation = settings.get("implementation", "legacy")
    if implementation == "legacy":
        return MultiScaleDiscriminator(channels=settings.get("channels", 16), scales=settings.get("scales", 3))
    if implementation == "dac_mrd":
        return DACMultiResolutionDiscriminator(sample_rate=sample_rate,
            fft_sizes=settings.get("fft_sizes", (512, 1024, 2048)),
            band_width_hz=settings.get("band_width_hz", 4000), sample_rates=sample_rates,
            sample_rate_scaling=settings.get("sample_rate_scaling", False))
    raise ValueError("adversarial.implementation must be legacy or dac_mrd")
