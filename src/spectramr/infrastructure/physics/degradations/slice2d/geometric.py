"""2-D geometric kernels: low-frequency warp (D20) and gradient non-linearity (D21)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from spectramr.infrastructure.physics.degradations._common import (
    _ensure_complex,
    _generator,
    _isolated_rng,
)
from spectramr.infrastructure.physics.digital_twin_simulator import simulate_gradient_nonlinearity


def degrade_geometric_warp(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_amplitude_px: float = 16.0,
    n_modes: int = 5,
) -> torch.Tensor:
    """D20 — Low-frequency geometric warp via random Fourier modes.

    Mimics registration error / motion residue via a smooth bilinear
    deformation. The mode sum is normalised so the peak displacement magnitude
    is exactly ``theta * max_amplitude_px`` pixels for every seed and grid size.
    """
    x = _ensure_complex(x)
    B, _, H, W = x.shape
    device = x.device
    if float(theta) <= 0.0:
        return x
    g = _generator(device, seed)

    rdt = x.real.dtype
    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=rdt),
        torch.linspace(-1, 1, W, device=device, dtype=rdt),
        indexing="ij",
    )
    params = torch.rand(3, n_modes, 1, 1, generator=g, device=device, dtype=rdt)
    fxs, fys, phs = params[0] * 3.0, params[1] * 3.0, params[2] * (2 * math.pi)
    dx = torch.sin(fxs * math.pi * X + fys * math.pi * Y + phs).sum(0)
    dy = torch.cos(fxs * math.pi * X - fys * math.pi * Y + phs).sum(0)
    # Peak |d| in pixels (align_corners=True: one pixel is 2 / (N - 1) grid units).
    px_x, px_y = 2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)
    peak = torch.sqrt((dx / px_x) ** 2 + (dy / px_y) ** 2).amax().clamp(min=1e-12)
    scale = float(theta) * max_amplitude_px / peak
    dx, dy = dx * scale, dy * scale

    grid = torch.stack([X + dx, Y + dy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    real = F.grid_sample(x.real, grid, mode="bilinear", padding_mode="border", align_corners=True)
    imag = F.grid_sample(x.imag, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return torch.complex(real, imag)


def _delegate_gradient_nl(x, theta, seed=0, **kw):
    x = _ensure_complex(x)
    with _isolated_rng(x.device, seed):
        torch.manual_seed(int(seed))
        if x.device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        # The primitive scales severity to normalised [-1, 1] coefficients at a fixed
        # reference grid, so the warp is a fixed FOV fraction: see the SeveritySpec.
        severity = float(kw.get("severity", float(theta) * 7e-5))
        out = simulate_gradient_nonlinearity(x, x.shape[-2:], severity=severity)
    return out
