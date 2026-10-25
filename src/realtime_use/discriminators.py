"""Small multi-scale waveform discriminators used only during training."""

from __future__ import annotations

import torch
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
