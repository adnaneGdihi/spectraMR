"""Spin-history corruption from through-plane motion (twin stage for arm m6).

Voxels pushed out of the slice are replaced by fresh tissue at equilibrium, so after the next
excitation they are brighter than the steady state. :class:`ContinuousBlochODE` integrates that
per voxel; this stage turns its signal into a gain map ``g = S_moving / S_static`` and the
caller multiplies the anatomy by it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.bloch_simulation import ContinuousBlochODE


class _ThroughPlaneMotion:
    """``dz[t]`` computed on demand, so no ``[n_steps, B, 1, H, W]`` tensor is held."""

    def __init__(
        self,
        profile: torch.Tensor,
        phase: torch.Tensor,
        amplitude: float,
        dt: float,
        period_ms: float,
        n: int,
    ) -> None:
        self._profile, self._phase = profile, phase
        self._amplitude, self._dt, self._period, self._n = amplitude, dt, period_ms, n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, step: int) -> torch.Tensor:
        wave = torch.sin(2 * math.pi * step * self._dt / self._period + self._phase)  # [B, 1, 1, 1]
        return self._amplitude * self._profile * wave


class SpinHistoryStage(nn.Module):
    """Gain map of the spin-history artefact for a batch of magnitude anatomies."""

    def __init__(
        self,
        *,
        dt: float,
        slice_thickness: float,
        n_steps: int,
        tr: float,
        t1_ms: float,
        t2_ms: float,
        amplitude: float,
        period_ms: float,
    ) -> None:
        super().__init__()
        if n_steps * dt < 2.0 * tr:
            raise ValueError(
                f"n_steps * dt = {n_steps * dt:g} ms covers fewer than 2 TR ({2 * tr:g} ms): the "
                "first excitation would be the last sample and no steady state is reached."
            )
        self.ode = ContinuousBlochODE(dt=dt, slice_thickness=slice_thickness)
        self.n_steps, self.tr, self.dt = n_steps, tr, dt
        self.t1_ms, self.t2_ms = t1_ms, t2_ms
        self.amplitude, self.period_ms = amplitude, period_ms

    def _maps(self, pd: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [pd, torch.full_like(pd, self.t1_ms), torch.full_like(pd, self.t2_ms)], dim=1
        )

    def _window_mean(self, signal: torch.Tensor) -> torch.Tensor:
        """Mean over the last TR, where the steady state has been reached."""
        return signal[-max(1, int(self.tr / self.dt)) :].mean(dim=0)

    @torch.no_grad()
    def gain(self, anatomy: torch.Tensor) -> torch.Tensor:
        """``[B, 1, H, W]`` multiplicative spin-history gain (1 where no signal)."""
        mag = anatomy.abs().mean(dim=1, keepdim=True)
        b, _, h, w = mag.shape
        pd = mag / mag.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)

        grid = torch.linspace(-1.0, 1.0, h, device=mag.device)
        gy, gx = torch.meshgrid(
            grid, torch.linspace(-1.0, 1.0, w, device=mag.device), indexing="ij"
        )
        fx, fy = (
            torch.rand(b, 1, 1, 1, device=mag.device) * 2,
            torch.rand(b, 1, 1, 1, device=mag.device) * 2,
        )
        spatial = torch.sin(math.pi * (fx * gx + fy * gy) + torch.rand_like(fx) * 2 * math.pi)
        phase = torch.rand(b, 1, 1, 1, device=mag.device) * 2 * math.pi
        motion = _ThroughPlaneMotion(
            spatial, phase, self.amplitude, self.dt, self.period_ms, self.n_steps
        )

        moving = self._window_mean(self.ode(self._maps(pd), self.n_steps, self.tr, motion))
        # Signal is linear in PD, so the static steady state per unit PD is one 1x1 integration.
        unit = torch.ones(1, 1, 1, 1, device=mag.device)
        static = self._window_mean(self.ode(self._maps(unit), self.n_steps, self.tr, None))
        ref = pd * static
        return torch.where(ref > 1e-6, moving / ref.clamp_min(1e-6), torch.ones_like(ref))
