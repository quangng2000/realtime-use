"""A new causal time-frequency enhancer with selectable block exits.

This baseline follows the published ideas, not NVIDIA's exact codec or weights.
Frequency context is bidirectional within an available frame. Temporal context
is strictly causal. Parameter counts must be measured for this implementation.
"""

import copy

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig
from .ssm import SelectiveMixer


class CausalFeatureConv(nn.Module):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__()
        self.conv = nn.Conv2d(incoming, outgoing, (3, 3))

    def forward(self, x: Tensor) -> Tensor:
        # Features are [B,T,F,C]. Pad only past time and both frequency edges.
        return self.conv(F.pad(x.permute(0, 3, 1, 2), (1, 1, 2, 0))).permute(0, 2, 3, 1)

    def step(self, x: Tensor, state: Tensor | None) -> tuple[Tensor, Tensor]:
        if state is None:
            state = x.new_zeros(x.shape[0], 2, x.shape[1], x.shape[2])
        window = torch.cat([state, x.unsqueeze(1)], dim=1)
        output = self.conv(F.pad(window.permute(0, 3, 1, 2), (1, 1, 0, 0)))
        return output[:, :, 0, :].transpose(1, 2), window[:, 1:]


class TimeFrequencyBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.frequency_norm = nn.LayerNorm(cfg.width)
        self.time_norm = nn.LayerNorm(cfg.width)
        self.frequency_forward = SelectiveMixer(cfg.width, cfg.state_dim, cfg.expand, cfg.conv_kernel, cfg.scan_backend)
        self.frequency_backward = SelectiveMixer(cfg.width, cfg.state_dim, cfg.expand, cfg.conv_kernel, cfg.scan_backend)
        self.frequency_merge = nn.Linear(2 * cfg.width, cfg.width)
        self.temporal = SelectiveMixer(cfg.width, cfg.state_dim, cfg.expand, cfg.conv_kernel, cfg.scan_backend)

    def _frequency(self, x: Tensor) -> Tensor:
        values = self.frequency_norm(x)
        front = self.frequency_forward(values)
        back = self.frequency_backward(values.flip(1)).flip(1)
        return x + self.frequency_merge(torch.cat([front, back], dim=-1))

    def forward(self, x: Tensor) -> Tensor:
        batch, time, frequency, channels = x.shape
        x = self._frequency(x.reshape(batch * time, frequency, channels)).reshape(batch, time, frequency, channels)
        temporal = self.time_norm(x).permute(0, 2, 1, 3).reshape(batch * frequency, time, channels)
        residual = self.temporal(temporal).reshape(batch, frequency, time, channels).permute(0, 2, 1, 3)
        return x + residual

    def step(self, x: Tensor, state: dict | None) -> tuple[Tensor, dict]:
        x = self._frequency(x)
        residual, state = self.temporal.step(self.time_norm(x).flatten(0, 1), state)
        return x + residual.reshape_as(x), state


class SpectralDecoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.context = CausalFeatureConv(width, width)
        self.output = nn.Linear(width, 2)
        # Start close to identity while retaining nonzero encoder gradients.
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def _decode(self, features: Tensor, noisy: Tensor) -> Tensor:
        components = self.output(F.silu(features)).float()
        # An additive complex residual can restore bins absent from the input.
        # This decoder is an independent design, not the authors' exact codec.
        return noisy + torch.complex(components[..., 0], components[..., 1])

    def forward(self, x: Tensor, noisy: Tensor) -> Tensor:
        return self._decode(self.context(self.norm(x)), noisy)

    def step(self, x: Tensor, noisy: Tensor, state: Tensor | None) -> tuple[Tensor, Tensor]:
        features, state = self.context.step(self.norm(x), state)
        return self._decode(features, noisy), state


class Enhancer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.encoder = CausalFeatureConv(3, config.width)
        self.encoder_norm = nn.LayerNorm(config.width)
        self.blocks = nn.ModuleList([TimeFrequencyBlock(config) for _ in range(config.depth)])
        self.shared_decoder: SpectralDecoder | None = None
        self.decoders = nn.ModuleDict()
        if config.decoder_mode == "shared":
            self.shared_decoder = SpectralDecoder(config.width)
        else:
            self.decoders.update({str(depth): SpectralDecoder(config.width) for depth in config.exits})
        from .audio import SpectralTransform
        self.transform = SpectralTransform(config.sample_rate, config.window_ms, config.hop_ms)

    @staticmethod
    def _features(spectrum: Tensor) -> Tensor:
        magnitude = spectrum.abs()
        compressed = spectrum / magnitude.clamp_min(1e-7).pow(0.7)
        return torch.stack([compressed.real, compressed.imag, magnitude.log1p()], dim=-1)

    def _exit_list(self, exits: list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
        result = self.config.exits if exits is None else tuple(exits)
        if not result or len(set(result)) != len(result) or any(x not in self.config.exits for x in result):
            raise ValueError(f"Requested exits must be a nonempty unique subset of {self.config.exits}")
        return tuple(sorted(result))

    def _decoder(self, exit_depth: int) -> SpectralDecoder:
        if self.config.decoder_mode == "shared":
            if self.shared_decoder is None:
                raise RuntimeError("Shared decoder is not initialized")
            return self.shared_decoder
        return self.decoders[str(exit_depth)]

    def forward(self, spectrum: Tensor, exit_depths=None) -> dict[int, Tensor]:
        if spectrum.ndim != 3 or not spectrum.is_complex() or spectrum.shape[1] == 0:
            raise ValueError("Expected nonempty complex [batch, time, frequency]")
        exits = self._exit_list(exit_depths)
        x = F.silu(self.encoder_norm(self.encoder(self._features(spectrum))))
        result = {}
        for depth, block in enumerate(self.blocks, 1):
            x = block(x)
            if depth in exits:
                result[depth] = self._decoder(depth)(x, spectrum)
            if depth == max(exits):
                break
        return result

    def enhance(self, waveform: Tensor, exit_depths=None) -> dict[int, Tensor]:
        if waveform.ndim != 2 or waveform.shape[-1] < 1:
            raise ValueError("Expected nonempty [batch, samples]")
        # FFT remains FP32 under mixed-precision training; learned layers use autocast.
        spectrum = self.transform.analysis(waveform.float())
        return {depth: self.transform.synthesis(value, waveform.shape[-1]) for depth, value in self(spectrum, exit_depths).items()}

    def split_decoders(self) -> None:
        if self.config.decoder_mode == "per_exit":
            return
        self.decoders.update({str(depth): copy.deepcopy(self.shared_decoder) for depth in self.config.exits})
        self.shared_decoder = None
        self.config.decoder_mode = "per_exit"

    def prepare_decoder_transition(self) -> None:
        """Register both head layouts before a training strategy sets up parameters."""
        if self.shared_decoder is None:
            self.shared_decoder = SpectralDecoder(self.config.width)
        for depth in self.config.exits:
            if str(depth) not in self.decoders:
                self.decoders[str(depth)] = copy.deepcopy(self.shared_decoder)

    def activate_prepared_decoders(self) -> None:
        """Copy the trained shared head without replacing any registered parameters."""
        if self.config.decoder_mode == "per_exit":
            return
        if self.shared_decoder is None or any(str(depth) not in self.decoders for depth in self.config.exits):
            raise RuntimeError("Call prepare_decoder_transition before optimizer construction")
        state = self.shared_decoder.state_dict()
        for decoder in self.decoders.values():
            decoder.load_state_dict(state, strict=True)
        self.config.decoder_mode = "per_exit"

    def step(self, frame: Tensor, state: dict | None = None, exit_depth: int | None = None) -> tuple[Tensor, dict]:
        if frame.ndim != 2 or not frame.is_complex():
            raise ValueError("Expected complex [batch, frequency]")
        exit_depth = max(self.config.exits) if exit_depth is None else exit_depth
        self._exit_list((exit_depth,))
        if state is None:
            state = {"encoder": None, "blocks": [None] * exit_depth, "decoder": None, "exit_depth": exit_depth, "shape": tuple(frame.shape)}
        if state["exit_depth"] != exit_depth or state["shape"] != tuple(frame.shape):
            raise ValueError("Exit or stream shape changed; reset state before switching")
        x, encoder = self.encoder.step(self._features(frame), state["encoder"])
        x = F.silu(self.encoder_norm(x))
        block_states = []
        for i in range(exit_depth):
            x, block_state = self.blocks[i].step(x, state["blocks"][i])
            block_states.append(block_state)
        output, decoder = self._decoder(exit_depth).step(x, frame, state["decoder"])
        return output, {"encoder": encoder, "blocks": block_states, "decoder": decoder, "exit_depth": exit_depth, "shape": tuple(frame.shape)}

    def active_parameter_count(self, exit_depth: int) -> int:
        self._exit_list((exit_depth,))
        modules = [self.encoder, self.encoder_norm, *self.blocks[:exit_depth], self._decoder(exit_depth)]
        return sum(p.numel() for module in modules for p in module.parameters())
