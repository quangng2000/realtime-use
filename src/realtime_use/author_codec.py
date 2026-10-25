"""Causal adaptation of the licensed, pinned NVIDIA NemotronLabs codec.

The upstream encoder/decoder forward methods are used unchanged. Child modules
adapt their temporal geometry and normalization to zero-lookahead streaming.
Each step receives and returns external state; no stream history is kept on the
model. The scoped context only routes those dictionaries through the unchanged
upstream forward signatures and is restored even if a call fails.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ._vendor.nemotron_codec import codec_module_time_d4 as author


_STEP_STATE: ContextVar[tuple[dict[str, Tensor], dict[str, Tensor]] | None] = ContextVar(
    "author_codec_step_state", default=None,
)


class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each time-frequency position."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class CausalConv(nn.Module):
    """Native Conv2d with past-only padding and bounded external input history."""

    def __init__(self, conv: nn.Conv2d, *, time_axis: int = 2) -> None:
        super().__init__()
        self.conv = conv
        self.time_axis = time_axis
        dimension = time_axis - 2
        if conv.stride[dimension] != 1:
            raise ValueError("Causal codec convolutions must preserve the time dimension")
        self.history = (conv.kernel_size[dimension] - 1) * conv.dilation[dimension]
        self.frequency_padding = conv.padding[1 - dimension]
        self.conv.padding = (0, 0)
        self.state_key = ""

    def _pad(self, x: Tensor, past: int) -> Tensor:
        frequency = self.frequency_padding
        padding = (frequency, frequency, past, 0) if self.time_axis == 2 else (past, 0, frequency, frequency)
        return F.pad(x, padding)

    def forward(self, x: Tensor) -> Tensor:
        context = _STEP_STATE.get()
        if context is None:
            return self.conv(self._pad(x, self.history))
        if x.shape[self.time_axis] != 1:
            raise ValueError("Codec step requires exactly one time frame")
        previous, following = context
        shape = list(x.shape)
        shape[self.time_axis] = self.history
        cache = previous.get(self.state_key)
        if cache is None:
            cache = x.new_zeros(shape)
        elif tuple(cache.shape) != tuple(shape) or cache.device != x.device or cache.dtype != x.dtype:
            raise ValueError("Codec stream state shape/device/dtype changed; reset state")
        window = torch.cat((cache, x), dim=self.time_axis)
        following[self.state_key] = window.narrow(self.time_axis, 1, self.history)
        return self.conv(self._pad(window, 0))


def _adapt_children(module: nn.Module) -> None:
    for name, child in tuple(module.named_children()):
        if isinstance(child, CausalConv):
            continue
        if isinstance(child, nn.InstanceNorm2d):
            module.add_module(name, ChannelLayerNorm(child.num_features))
        elif isinstance(child, nn.Conv2d) and child.kernel_size[0] > 1:
            module.add_module(name, CausalConv(child))
        else:
            _adapt_children(child)


def _assign_state_keys(module: nn.Module) -> None:
    for name, child in module.named_modules():
        if isinstance(child, CausalConv):
            child.state_key = name


def _configuration(width: int) -> dict:
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("Codec width must be a positive integer")
    # n_fft/beta are stored but unused by these mapping-based author classes.
    return {"model_cfg": {"input_channel": 2, "hid_feature": width, "output_channel": 1, "beta": 2.0},
            "stft_cfg": {"n_fft": 2}}


@contextmanager
def _stream_context(state: dict | None, shape: tuple[int, ...]) -> Iterator[dict]:
    if state is not None and state.get("shape") != shape:
        raise ValueError("Codec stream shape changed; reset state")
    following: dict = {"shape": shape, "convolutions": {}}
    token = _STEP_STATE.set(({} if state is None else state["convolutions"], following["convolutions"]))
    try:
        yield following
    finally:
        _STEP_STATE.reset(token)


class AuthorEncoder(nn.Module):
    """Complex B,T,F input to B,T,F',C features using the author's encoder."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.codec = author.DenseEncoder(_configuration(width))
        self.codec.dense_conv_1[0] = nn.Conv2d(2, width, (3, 3), padding=(1, 1))
        self.codec.dense_conv_2[0].stride = (1, 2)
        _adapt_children(self.codec)
        _assign_state_keys(self)

    def forward(self, spectrum: Tensor) -> Tensor:
        if spectrum.ndim != 3 or not spectrum.is_complex() or min(spectrum.shape) < 1:
            raise ValueError("AuthorEncoder expects nonempty complex [batch, time, frequency]")
        compressed = spectrum.abs().log1p()
        inputs = torch.stack((compressed, spectrum.angle()), dim=1)
        # The author generator also appends two frequency bins. This makes both
        # odd/even FFT-bin counts compatible with the stride-two valid convolution.
        return self.codec(F.pad(inputs, (0, 2))).permute(0, 2, 3, 1)

    def step(self, frame: Tensor, state: dict | None = None) -> tuple[Tensor, dict]:
        if frame.ndim != 2 or not frame.is_complex():
            raise ValueError("AuthorEncoder.step expects complex [batch, frequency]")
        with _stream_context(state, tuple(frame.shape)) as following:
            output = self(frame.unsqueeze(1))[:, 0]
        return output, following


class AuthorDecoder(nn.Module):
    """Separate author magnitude/phase heads, preserving the noisy input shape."""

    def __init__(self, width: int) -> None:
        super().__init__()
        cfg = _configuration(width)
        self.magnitude = author.MagDecoder(cfg)
        self.phase = author.PhaseDecoder(cfg)
        for head in (self.magnitude, self.phase):
            # The upstream forward transposes T/F around this second stage.
            # Its factor-four temporal upsampling becomes factor one; the
            # remaining width-three convolution refines past time frames only.
            temporal = author.SPConvTranspose2d(width, width, (1, 3), r=1)
            temporal.pad1 = nn.Identity()
            temporal.conv = CausalConv(temporal.conv, time_axis=3)
            head.up_conv2[0] = temporal
            _adapt_children(head)
        _assign_state_keys(self)

    def forward(self, features: Tensor, noisy: Tensor) -> Tensor:
        if features.ndim != 4 or noisy.ndim != 3 or not noisy.is_complex() or min(noisy.shape) < 1:
            raise ValueError("AuthorDecoder expects features [B,T,F',C] and complex noisy [B,T,F]")
        if features.shape[:2] != noisy.shape[:2]:
            raise ValueError("Decoder features and noisy input must have matching batch/time dimensions")
        x = features.permute(0, 3, 1, 2)
        mapped = self.magnitude(x)[:, 0, :, :noisy.shape[-1]]
        phase = self.phase(x)[:, 0, :, :noisy.shape[-1]]
        if mapped.shape != noisy.shape:
            raise ValueError("Decoder frequency geometry does not match the noisy input")
        if mapped.dtype not in (torch.float32, torch.float64):
            mapped, phase = mapped.float(), phase.float()
        # Both the commercial and real-time recipes select relu_log1p. The
        # commercial release's stfts.py defines this inverse as expm1(ReLU(x)).
        magnitude = F.relu(mapped).expm1()
        return torch.complex(magnitude * phase.cos(), magnitude * phase.sin())

    def step(self, features: Tensor, noisy: Tensor, state: dict | None = None) -> tuple[Tensor, dict]:
        if features.ndim != 3 or noisy.ndim != 2:
            raise ValueError("AuthorDecoder.step expects features [B,F',C] and noisy [B,F]")
        shape = (*features.shape, noisy.shape[-1])
        with _stream_context(state, shape) as following:
            output = self(features.unsqueeze(1), noisy.unsqueeze(1))[:, 0]
        return output, following
