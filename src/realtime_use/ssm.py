"""Selective state-space mixer, implemented from the Mamba equations.

The portable recurrence is a correctness reference, not an optimized trainer.
An optional Apache-2.0 mamba-ssm selective-scan kernel accelerates sequence
training on CUDA. No NVIDIA RE-USE implementation or checkpoint is used.
"""

import math
import warnings

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SelectiveMixer(nn.Module):
    def __init__(self, width: int, state_dim: int, expand: int, kernel: int, backend: str):
        super().__init__()
        self.width, self.inner, self.state_dim = width, width * expand, state_dim
        self.kernel, self.backend = kernel, backend
        self.in_projection = nn.Linear(width, 2 * self.inner)
        self.local_conv = nn.Conv1d(self.inner, self.inner, kernel, groups=self.inner)
        self.rank = max(1, math.ceil(width / 16))
        self.selection = nn.Linear(self.inner, self.rank + 2 * state_dim, bias=False)
        self.delta_projection = nn.Linear(self.rank, self.inner)
        nn.init.constant_(self.delta_projection.bias, -4.0)
        self.log_decay = nn.Parameter(torch.arange(1, state_dim + 1).float().log().repeat(self.inner, 1))
        self.skip = nn.Parameter(torch.ones(self.inner))
        self.out_projection = nn.Linear(self.inner, width)
        self._warned_fallback = False

    def _parameters_for(self, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        delta_low, drive, readout = self.selection(value).split(
            [self.rank, self.state_dim, self.state_dim], dim=-1
        )
        delta = F.softplus(self.delta_projection(delta_low).float())
        return delta, drive.float(), readout.float()

    def _reference_scan(self, u: Tensor, delta: Tensor, drive: Tensor, readout: Tensor) -> Tensor:
        batch, length, _ = u.shape
        state = torch.zeros(batch, self.inner, self.state_dim, device=u.device, dtype=torch.float32)
        decay = -self.log_decay.float().exp()
        values = []
        for t in range(length):
            step = delta[:, t, :, None]
            state = (step * decay).exp() * state + step * drive[:, t, None, :] * u[:, t, :, None].float()
            values.append((state * readout[:, t, None, :]).sum(-1) + self.skip.float() * u[:, t].float())
        return torch.stack(values, dim=1).to(u.dtype)

    def _scan(self, u: Tensor, delta: Tensor, drive: Tensor, readout: Tensor) -> Tensor:
        if self.backend == "reference":
            return self._reference_scan(u, delta, drive, readout)
        if not u.is_cuda:
            if self.backend == "mamba_ssm":
                raise RuntimeError("mamba_ssm scan requires CUDA; use reference for CPU correctness tests")
            return self._reference_scan(u, delta, drive, readout)
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        except ImportError as error:
            if self.backend == "mamba_ssm":
                raise RuntimeError("Install the cuda extra and compatible PyTorch/CUDA build") from error
            if not self._warned_fallback:
                warnings.warn("CUDA reference scan is slow: install mamba-ssm before a long training run", stacklevel=2)
                self._warned_fallback = True
            return self._reference_scan(u, delta, drive, readout)
        # The kernel requires B/C in the input dtype and A/D in float32.
        scanned = selective_scan_fn(
            u.transpose(1, 2).contiguous(),
            delta.to(u.dtype).transpose(1, 2).contiguous(),
            -self.log_decay.float().exp(),
            drive.to(u.dtype).transpose(1, 2).contiguous(),
            readout.to(u.dtype).transpose(1, 2).contiguous(),
            self.skip.float(),
            delta_softplus=False,
        )
        return scanned.transpose(1, 2)

    def forward(self, sequence: Tensor) -> Tensor:
        if sequence.ndim != 3 or sequence.shape[1] < 1:
            raise ValueError("Expected nonempty [batch, length, width]")
        raw, gate = self.in_projection(sequence).chunk(2, dim=-1)
        u = self.local_conv(F.pad(raw.transpose(1, 2), (self.kernel - 1, 0)))
        u = F.silu(u.transpose(1, 2))
        delta, drive, readout = self._parameters_for(u)
        y = self._scan(u, delta, drive, readout)
        return self.out_projection(y * F.silu(gate))

    def step(self, token: Tensor, state: dict[str, Tensor] | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        if token.ndim != 2:
            raise ValueError("Expected [batch, width]")
        raw, gate = self.in_projection(token).chunk(2, dim=-1)
        if state is None:
            history = raw.new_zeros(raw.shape[0], self.inner, self.kernel - 1)
            memory = torch.zeros(raw.shape[0], self.inner, self.state_dim, device=raw.device, dtype=torch.float32)
        else:
            history, memory = state["conv"], state["ssm"]
            if history.shape[:2] != (raw.shape[0], self.inner):
                raise ValueError("Stream shape changed; reset its state")
        window = torch.cat([history, raw.unsqueeze(-1)], dim=-1)
        u = F.silu(F.conv1d(window, self.local_conv.weight, self.local_conv.bias, groups=self.inner).squeeze(-1))
        delta, drive, readout = self._parameters_for(u)
        step = delta.unsqueeze(-1)
        memory = (step * -self.log_decay.float().exp()).exp() * memory + step * drive.unsqueeze(1) * u.float().unsqueeze(-1)
        y = (memory * readout.unsqueeze(1)).sum(-1) + self.skip.float() * u.float()
        output = self.out_projection(y.to(u.dtype) * F.silu(gate))
        history = window[:, :, 1:] if self.kernel > 1 else window[:, :, :0]
        return output, {"conv": history, "ssm": memory}
