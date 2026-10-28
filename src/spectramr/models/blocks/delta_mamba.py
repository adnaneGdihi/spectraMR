"""DeltaMamba: a Mamba selective scan whose step size takes an external scale and shift.

Mamba computes ``delta = softplus(dt_proj(x) + bias)`` per token and channel. Here
the caller supplies a per-token, per-channel ``delta_scale`` and ``delta_shift``:

    delta = softplus(dt_proj(x) + bias + delta_shift) * delta_scale

The parameter names match ``mamba_ssm.Mamba`` so its state dict loads unchanged.

Backends:
    ``cuda``: the official ``mamba_ssm`` selective-scan kernel. Raises at
        construction when ``mamba_ssm`` is absent, even if
        ``SPECTRAMR_ALLOW_MAMBA_FALLBACK`` is set, because the GRU fallback has no Δ.
    ``reference``: a pure-torch ZOH scan for CPU tests. It loops over L, so it is
        not a training path.
"""

from __future__ import annotations

import importlib.util
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from spectramr.models.blocks.grad_checkpointing import GradCheckpointingMixin

_BACKENDS = ("cuda", "reference")


def _load_selective_scan_fn():
    """Return ``mamba_ssm``'s selective-scan kernel, or raise (no fallback)."""
    if importlib.util.find_spec("mamba_ssm") is None:
        raise ImportError(
            "DeltaMamba backend 'cuda' requires the mamba_ssm selective-scan kernel "
            "(pip install -e '.[mamba]'). The GRU fallback has no Δ, so it is not used."
        )
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    return selective_scan_fn


def _reference_selective_scan(u, delta, A, b_mat, c_mat, D, z):
    """ZOH selective scan in pure torch.

    Shapes: ``u``, ``delta``, ``z`` are ``[B, d_in, L]``; ``A`` is ``[d_in, N]``;
    ``b_mat``, ``c_mat`` are ``[B, N, L]``; ``D`` is ``[d_in]``.
    """
    d_a = torch.exp(delta.unsqueeze(-1) * A[None, :, None, :])  # [B, d_in, L, N]
    d_bu = delta.unsqueeze(-1) * b_mat.permute(0, 2, 1).unsqueeze(1) * u.unsqueeze(-1)
    h = torch.zeros(u.shape[0], u.shape[1], A.shape[1], device=u.device, dtype=u.dtype)
    ys = []
    for t in range(u.shape[-1]):
        h = d_a[:, :, t] * h + d_bu[:, :, t]
        ys.append((h * c_mat[:, :, t].unsqueeze(1)).sum(-1))
    y = torch.stack(ys, dim=-1) + D[None, :, None] * u
    return y * F.silu(z)


class DeltaMamba(nn.Module):
    """Mamba block with an external Δ scale and shift. Input and output are ``[B, L, d_model]``."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        backend: str = "cuda",
    ) -> None:
        super().__init__()
        if backend not in _BACKENDS:
            raise ValueError(f"DeltaMamba backend must be one of {_BACKENDS}, got {backend!r}.")
        self.backend = backend
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self._kernel = _load_selective_scan_fn() if backend == "cuda" else None

        d_in, N, R = self.d_inner, d_state, self.dt_rank
        self.in_proj = nn.Linear(d_model, 2 * d_in, bias=False)
        self.conv1d = nn.Conv1d(
            d_in, d_in, kernel_size=d_conv, groups=d_in, padding=d_conv - 1, bias=True
        )
        self.x_proj = nn.Linear(d_in, R + 2 * N, bias=False)
        self.dt_proj = nn.Linear(R, d_in, bias=True)
        # Mamba's init: uniform weight, and a bias that puts softplus(bias) in [dt_min, dt_max].
        std = R**-0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(
            torch.rand(d_in) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, N + 1).float().repeat(d_in, 1)))
        self.D = nn.Parameter(torch.ones(d_in))
        self.out_proj = nn.Linear(d_in, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        delta_scale: torch.Tensor | None = None,
        delta_shift: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the scan with an optional per-token, per-channel Δ scale and shift.

        Args:
            x: ``[B, L, d_model]``.
            delta_scale: broadcastable to ``[B, L, d_inner]``; multiplies Δ after softplus.
            delta_shift: broadcastable to ``[B, L, d_inner]``; added before softplus.
        """
        # The scan runs in fp32 with autocast off, as in Mamba's own kernels.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            L = x.shape[1]
            xs, z = self.in_proj(x).transpose(1, 2).chunk(2, dim=1)
            xs = F.silu(self.conv1d(xs)[..., :L])
            dt, b_mat, c_mat = torch.split(
                self.x_proj(xs.transpose(1, 2)), [self.dt_rank, self.d_state, self.d_state], -1
            )
            pre = F.linear(dt, self.dt_proj.weight) + self.dt_proj.bias
            if delta_shift is not None:
                pre = pre + delta_shift
            delta = F.softplus(pre)
            if delta_scale is not None:
                delta = delta * delta_scale
            delta = delta.transpose(1, 2)
            A = -torch.exp(self.A_log.float())
            y = self._scan(xs, delta, A, b_mat.transpose(1, 2), c_mat.transpose(1, 2), z)
            return self.out_proj(y.transpose(1, 2))

    def _scan(self, u, delta, A, b_mat, c_mat, z):
        D = self.D.float()
        if self._kernel is not None:
            # delta is already softplus-ed and scaled, so the kernel gets no bias and no softplus.
            return self._kernel(
                u, delta, A, b_mat, c_mat, D, z=z, delta_bias=None, delta_softplus=False
            )
        return _reference_selective_scan(u, delta, A, b_mat, c_mat, D, z)


def _per_layer(value, n: int) -> list:
    """Broadcast one shared value to ``n`` layers, or check a per-layer list."""
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"expected {n} per-layer delta shifts, got {len(value)}.")
        return list(value)
    return [value] * n


class DeltaMambaStack(GradCheckpointingMixin, nn.Module):
    """``num_layers`` DeltaMamba blocks with residuals and a final LayerNorm, as ``_MambaEncoder``."""

    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        backend: str = "cuda",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DeltaMamba(d_model, d_state, d_conv, expand, backend=backend) for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def freeze_dt_bias(self) -> None:
        """Stop training the Δ bias, so a fixed Δ scale cannot be absorbed into it."""
        for layer in self.layers:
            layer.dt_proj.bias.requires_grad_(False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        delta_scale: torch.Tensor | None = None,
        delta_shift: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply every layer; ``delta_shift`` is one tensor for all layers, or one per layer."""
        shifts = _per_layer(delta_shift, len(self.layers))
        ckpt = self._checkpointing_active()
        for layer, shift in zip(self.layers, shifts, strict=True):
            kw = {"delta_scale": delta_scale, "delta_shift": shift}
            x = x + (checkpoint(layer, x, use_reentrant=False, **kw) if ckpt else layer(x, **kw))
        return self.norm(x)


__all__ = ["DeltaMamba", "DeltaMambaStack"]
