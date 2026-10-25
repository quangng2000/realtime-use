"""Differentiable, uncentered transforms and bounded-state streaming inference.

Past zero padding initializes overlap-add. The first valid output requires a
full analysis window of input. Right zero padding occurs only at end-of-stream.
Retained samples have complete overlap, avoiding singular window endpoints.
"""

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from .model import Enhancer


class SpectralTransform(nn.Module):
    def __init__(self, sample_rate: int, window_ms: float, hop_ms: float):
        super().__init__()
        self.window_samples = round(sample_rate * window_ms / 1000)
        self.hop_samples = round(sample_rate * hop_ms / 1000)
        if self.hop_samples < 1 or self.window_samples != 2 * self.hop_samples:
            raise ValueError("Transform requires window = 2 * hop, with positive integer samples")
        self.register_buffer("window", torch.hann_window(self.window_samples).sqrt())

    def frame_count(self, length: int) -> int:
        if length < 1:
            raise ValueError("Audio must contain at least one sample")
        return math.ceil(length / self.hop_samples) + 1

    def analysis(self, waveform: Tensor) -> Tensor:
        if waveform.ndim != 2 or waveform.shape[-1] == 0 or not waveform.is_floating_point():
            raise ValueError("Expected nonempty real floating audio [batch, samples]")
        count = self.frame_count(waveform.shape[-1])
        size = (count - 1) * self.hop_samples + self.window_samples
        padded = F.pad(waveform.float(), (self.hop_samples, size - self.hop_samples - waveform.shape[-1]))
        frames = padded.unfold(-1, self.window_samples, self.hop_samples)
        return torch.fft.rfft(frames * self.window.float(), dim=-1)

    def synthesis(self, spectrum: Tensor, length: int) -> Tensor:
        if spectrum.ndim != 3 or not spectrum.is_complex() or spectrum.shape[1] == 0:
            raise ValueError("Expected nonempty complex spectrum [batch, frames, bins]")
        if spectrum.shape[-1] != self.window_samples // 2 + 1:
            raise ValueError("Spectrum bin count does not match transform")
        batch, frames, _ = spectrum.shape
        size = (frames - 1) * self.hop_samples + self.window_samples
        if not 0 < length <= size - 2 * self.hop_samples:
            raise ValueError("Requested length must fit the synthesized frames")
        window = self.window.float()
        signal = torch.fft.irfft(spectrum, n=self.window_samples, dim=-1) * window
        output = F.fold(signal.transpose(1, 2), (1, size), (1, self.window_samples), stride=(1, self.hop_samples))
        weights = window.square().view(1, -1, 1).expand(1, -1, frames)
        denominator = F.fold(weights, (1, size), (1, self.window_samples), stride=(1, self.hop_samples))
        restored = (output / denominator.clamp_min(1e-8)).reshape(batch, size)
        return restored[:, self.hop_samples:self.hop_samples + length]


class StreamingEnhancer:
    """One independent stream batch. Keep this object between successive chunks.

    Input is [B,N] at the model sample rate, on the model's device. Output may
    be empty until a window is available. Concatenate push results and flush
    once to recover exactly the original sample count. Reset between clips.
    """

    def __init__(self, model: "Enhancer", exit_depth: int):
        if model.training:
            raise ValueError("StreamingEnhancer requires model.eval()")
        if exit_depth not in model.config.exits:
            raise ValueError("Exit is not configured in this model")
        self.model = model
        self.exit_depth = exit_depth
        self.transform = model.transform
        self.reset()

    def reset(self) -> None:
        self._input = None
        self._overlap = None
        self._weights = None
        self._state = None
        self._received = 0
        self._emitted = 0
        self._frames = 0
        self._finished = False

    def _frame(self) -> Tensor:
        width, hop = self.transform.window_samples, self.transform.hop_samples
        window = self.transform.window.float()
        spectrum = torch.fft.rfft(self._input[:, :width] * window, dim=-1)
        enhanced, self._state = self.model.step(spectrum, self._state, exit_depth=self.exit_depth)
        frame = torch.fft.irfft(enhanced, n=width, dim=-1) * window
        self._overlap = self._overlap + frame
        self._weights = self._weights + window.square()
        output = self._overlap[:, :hop] / self._weights[:hop].clamp_min(1e-8)
        self._overlap = F.pad(self._overlap[:, hop:], (0, hop))
        self._weights = F.pad(self._weights[hop:], (0, hop))
        self._input = self._input[:, hop:]
        self._frames += 1
        # The first emitted hop is artificial past padding, not user audio.
        return output[:, :0] if self._frames == 1 else output

    @torch.no_grad()
    def push(self, chunk: Tensor) -> Tensor:
        if self._finished:
            raise RuntimeError("Stream is finished; call reset before pushing more audio")
        if self.model.training:
            raise ValueError("StreamingEnhancer requires model.eval()")
        if chunk.ndim != 2 or not chunk.is_floating_point():
            raise ValueError("Expected floating audio [batch, samples]")
        if chunk.device != next(self.model.parameters()).device:
            raise ValueError("Audio and model must use the same device")
        chunk = chunk.float()
        width = self.transform.window_samples
        if self._input is None:
            self._input = chunk.new_zeros(chunk.shape[0], self.transform.hop_samples)
            self._overlap = chunk.new_zeros(chunk.shape[0], width)
            self._weights = chunk.new_zeros(width)
        elif chunk.shape[0] != self._input.shape[0]:
            raise ValueError("Batch size changed within a stream; reset first")
        self._received += chunk.shape[-1]
        self._input = torch.cat([self._input, chunk], dim=-1)
        pieces = []
        while self._input.shape[-1] >= width:
            pieces.append(self._frame())
        # A residual slice otherwise retains a large caller chunk's storage.
        self._input = self._input.clone()
        output = torch.cat(pieces, dim=-1) if pieces else chunk[:, :0]
        self._emitted += output.shape[-1]
        return output

    @torch.no_grad()
    def flush(self) -> Tensor:
        if self._finished:
            raise RuntimeError("Stream already flushed; reset before reuse")
        self._finished = True
        if self._input is None:
            raise ValueError("Push at least one chunk before flushing")
        if self._received == 0:
            return self._input[:, :0]
        needed = self._received - self._emitted
        count = self.transform.frame_count(self._received)
        pieces = []
        while self._frames < count:
            self._input = F.pad(self._input, (0, self.transform.window_samples - self._input.shape[-1]))
            pieces.append(self._frame())
        pieces.append(self._overlap / self._weights.clamp_min(1e-8))
        output = torch.cat(pieces, dim=-1)[:, :needed]
        self._emitted += output.shape[-1]
        return output
