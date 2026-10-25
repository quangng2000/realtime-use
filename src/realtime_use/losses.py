"""Independently chosen training objectives, not a reproduction of NVIDIA losses."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def valid_length_groups(waveform: Tensor, lengths: Tensor | None = None) -> list[tuple[Tensor, int]]:
    """Indices grouped by valid sample count, with padding excluded from objectives."""
    if waveform.ndim != 2 or min(waveform.shape) < 1:
        raise ValueError("Audio must be nonempty [batch, samples]")
    if lengths is None:
        return [(torch.arange(len(waveform), device=waveform.device), waveform.shape[-1])]
    lengths = torch.as_tensor(lengths).detach().cpu()
    if lengths.ndim != 1 or len(lengths) != len(waveform):
        raise ValueError("lengths must contain one valid sample count per utterance")
    if lengths.dtype == torch.bool or lengths.is_floating_point() or lengths.is_complex():
        raise ValueError("lengths must contain integers")
    if (lengths < 1).any() or (lengths > waveform.shape[-1]).any():
        raise ValueError("Valid lengths must be positive and no greater than the padded audio length")
    return [(torch.nonzero(lengths == size, as_tuple=True)[0].to(waveform.device), int(size))
            for size in torch.unique(lengths, sorted=True)]


class MultiResolutionSTFTLoss(nn.Module):
    """Magnitude spectral convergence and log-magnitude error at several scales."""

    def __init__(self, fft_sizes: Sequence[int] = (256, 512, 1024)) -> None:
        super().__init__()
        self.fft_sizes = tuple(int(size) for size in fft_sizes)
        if not self.fft_sizes or any(size < 8 for size in self.fft_sizes):
            raise ValueError("fft_sizes must contain integers >= 8")

    def forward(self, estimate: Tensor, target: Tensor) -> Tensor:
        if estimate.ndim != 2 or estimate.shape != target.shape:
            raise ValueError("STFT loss requires matching [batch, samples] tensors")
        # FFT and logarithms remain FP32 even when the network uses mixed precision.
        with torch.autocast(device_type=estimate.device.type, enabled=False):
            estimate, target = estimate.float(), target.float()
            total = estimate.new_zeros(())
            for size in self.fft_sizes:
                padding = max(0, size - estimate.shape[-1])
                window = torch.hann_window(size, device=estimate.device)
                kwargs = dict(n_fft=size, hop_length=size // 4, window=window,
                              center=False, return_complex=True)
                predicted = torch.stft(F.pad(estimate, (0, padding)), **kwargs).abs()
                reference = torch.stft(F.pad(target, (0, padding)), **kwargs).abs()
                difference = torch.linalg.vector_norm(predicted - reference, dim=(1, 2))
                scale = torch.linalg.vector_norm(reference, dim=(1, 2)).clamp_min(1e-5)
                convergence = (difference / scale).mean()
                log_error = F.l1_loss(predicted.clamp_min(1e-5).log(),
                                      reference.clamp_min(1e-5).log())
                total = total + convergence + log_error
            return total / len(self.fft_sizes)


class ReconstructionLoss(nn.Module):
    """Clean-reference or output-distillation waveform and spectral objective."""

    def __init__(self, waveform: float = 1.0, stft: float = 1.0,
                 fft_sizes: Sequence[int] = (256, 512, 1024)) -> None:
        super().__init__()
        if (not math.isfinite(waveform) or not math.isfinite(stft)
                or waveform < 0 or stft < 0 or waveform + stft <= 0):
            raise ValueError("Reconstruction weights must be nonnegative and nonzero")
        self.waveform_weight, self.stft_weight = float(waveform), float(stft)
        self.spectral = MultiResolutionSTFTLoss(fft_sizes)

    def forward(self, estimate: Tensor, target: Tensor,
                lengths: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        if estimate.ndim != 2 or estimate.shape != target.shape:
            raise ValueError("Reconstruction loss requires matching [batch, samples]")
        waveform = estimate.new_zeros((), dtype=torch.float32)
        spectral = estimate.new_zeros((), dtype=torch.float32)
        for indices, length in valid_length_groups(estimate, lengths):
            predicted = estimate[indices, :length]
            reference = target[indices, :length]
            weight = len(indices) / len(estimate)
            waveform = waveform + weight * F.l1_loss(predicted.float(), reference.float())
            if self.stft_weight:
                spectral = spectral + weight * self.spectral(predicted, reference)
        total = self.waveform_weight * waveform + self.stft_weight * spectral
        return total, {"waveform_l1": waveform.detach(), "stft_loss": spectral.detach()}


def discriminator_hinge_loss(real: list[list[Tensor]], fake: list[list[Tensor]]) -> Tensor:
    if not real or len(real) != len(fake):
        raise ValueError("Discriminator outputs must have the same nonzero scale count")
    return torch.stack([F.relu(1 - r[-1]).mean() + F.relu(1 + f[-1]).mean()
                        for r, f in zip(real, fake)]).mean()


def generator_hinge_loss(fake: list[list[Tensor]]) -> Tensor:
    return torch.stack([-scale[-1].mean() for scale in fake]).mean()


def feature_matching_loss(real: list[list[Tensor]], fake: list[list[Tensor]]) -> Tensor:
    if not real or len(real) != len(fake):
        raise ValueError("Feature outputs must have the same nonzero scale count")
    terms = []
    for reference, predicted in zip(real, fake):
        if len(reference) != len(predicted) or len(reference) < 2:
            raise ValueError("Each discriminator must expose features and logits")
        terms.extend(F.l1_loss(p, r.detach()) for r, p in zip(reference[:-1], predicted[:-1]))
    return torch.stack(terms).mean()


def discriminator_audio_loss(discriminator: nn.Module, reference: Tensor, prediction: Tensor,
                             lengths: Tensor | None = None) -> Tensor:
    """Hinge critic loss averaged over utterances after removing batch padding."""
    if reference.shape != prediction.shape:
        raise ValueError("Real and enhanced audio must have matching shapes")
    total = prediction.new_zeros(())
    for indices, length in valid_length_groups(prediction, lengths):
        real = discriminator(reference[indices, :length])
        fake = discriminator(prediction[indices, :length].detach())
        total = total + (len(indices) / len(prediction)) * discriminator_hinge_loss(real, fake)
    return total


def generator_audio_losses(discriminator: nn.Module, reference: Tensor, prediction: Tensor,
                           lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Adversarial and feature losses on valid audio; caller freezes the critic."""
    if reference.shape != prediction.shape:
        raise ValueError("Real and enhanced audio must have matching shapes")
    adversarial = prediction.new_zeros(())
    matching = prediction.new_zeros(())
    for indices, length in valid_length_groups(prediction, lengths):
        with torch.no_grad():
            real = discriminator(reference[indices, :length])
        fake = discriminator(prediction[indices, :length])
        weight = len(indices) / len(prediction)
        adversarial = adversarial + weight * generator_hinge_loss(fake)
        matching = matching + weight * feature_matching_loss(real, fake)
    return adversarial, matching
