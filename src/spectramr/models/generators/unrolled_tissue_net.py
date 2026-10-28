r"""Unrolled tissue-parameter network: learned gradient descent through the TSE render.

The unknowns are the tissue maps :math:`\theta = (\rho, T_1, T_2)`, not an image. Each of
``iterations`` steps renders every acquired contrast at its header protocol
(:func:`~spectramr.infrastructure.physics.tse_forward.tse_signal`), fits the per-image
gain in closed form, and hands the gain-scaled residuals to a shared prior network that
proposes the update of :math:`\theta` (learned gradient descent, Adler & Öktem 2017,
here in tissue-parameter space through weighted-contrast renders).

A contrast marked unavailable never reaches the maps: its input channel is zeroed, its
gain is not fitted and its residual is zeroed, so it drops out of the data term. That is
what makes leave-one-contrast-out synthesis an honest test.

The gain absorbs the scanner scale per image, so the maps carry the two gauge directions
an unknown gain leaves open (``DESIGN_tissue_map_cohort.md`` §1); a tissue prior closes
them in the loss, not here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from spectramr.infrastructure.physics.tse_forward import tse_signal
from spectramr.models.generators.contrast_token_encoder import ContrastTokenEncoder
from spectramr.models.registry import register_model

__all__ = ["UnrolledTissueNet"]

_N_CONTRASTS = 3
_PROTOCOL_KEYS = ("family", "tr", "te", "ti", "esp", "etl", "refocus_deg")


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.GroupNorm(min(8, c_out), c_out),
        nn.LeakyReLU(0.2),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.GroupNorm(min(8, c_out), c_out),
        nn.LeakyReLU(0.2),
    )


class _PriorNet(nn.Module):
    """Two-level U-Net mapping (state, residuals, availability, step) to an update."""

    def __init__(self, c_in: int, base: int) -> None:
        super().__init__()
        self.enc = _block(c_in, base)
        self.mid = _block(base, 2 * base)
        self.up = nn.ConvTranspose2d(2 * base, base, 2, stride=2)
        self.dec = _block(2 * base, base)
        self.head = nn.Conv2d(base, _N_CONTRASTS, 1)
        nn.init.zeros_(self.head.weight)  # start as the identity: no update before training
        nn.init.zeros_(self.head.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        e = self.enc(h)
        m = self.mid(nn.functional.avg_pool2d(e, 2))
        return self.head(self.dec(torch.cat([self.up(m), e], dim=1)))


class _ConvInit(nn.Sequential):
    """The default initialiser: one conv block on the masked triplet and its availability."""

    def __init__(self, base_channels: int) -> None:
        super().__init__(
            _block(2 * _N_CONTRASTS, base_channels), nn.Conv2d(base_channels, _N_CONTRASTS, 1)
        )

    def forward(self, x: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        planes = available.view(*available.shape, 1, 1).expand(-1, -1, *x.shape[2:])
        return super().forward(torch.cat([x, planes.to(x.dtype)], dim=1))


#: ``model_kwargs.initializer`` -> builder(base_channels, initializer_kwargs). B1 of the cohort
#: is the contrast-token transformer, used as B2's amortised initialiser.
_INITIALISERS: dict[str, Any] = {
    "conv": lambda base, kw: _ConvInit(base, **kw),
    "contrast_transformer": lambda base, kw: ContrastTokenEncoder(**kw),
}


@register_model(
    name="unrolled_tissue_net",
    training_mode="reconstruction",
    input_domain="image",
)
class UnrolledTissueNet(nn.Module):
    """Learned gradient descent on (rho, T1, T2) through the per-contrast TSE render.

    Args:
        in_channels: Must be 3, the [T1, T2, FLAIR] triplet in noise units.
        out_channels: Accepted for the registry contract; the output is the maps.
        base_channels: Width of the shared prior network.
        iterations: Unrolled steps.
        signal_floor: Noise-unit threshold of the voxels a gain is fitted on.
        rho_range, t1_range_ms, t2_range_ms: Bounds the maps are squashed into.
        initializer: ``conv`` (default) or ``contrast_transformer`` (B1: contrast
            tokens, mask tokens, factorised attention), keyed in ``_INITIALISERS``.
        initializer_kwargs: Passed to the initialiser (dim, depth, heads, window).
        checkpoint_iterations: Recompute each step in the backward pass to save memory.
    """

    #: The strategy passes each contrast's protocol and an availability mask.
    takes_protocol: ClassVar[bool] = True

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        iterations: int = 4,
        signal_floor: float = 3.0,
        rho_range: tuple[float, float] = (0.0, 1.0),
        t1_range_ms: tuple[float, float] = (50.0, 5000.0),
        t2_range_ms: tuple[float, float] = (5.0, 2000.0),
        checkpoint_iterations: bool = True,
        initializer: str = "conv",
        initializer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if in_channels != _N_CONTRASTS:
            raise ValueError(f"in_channels={in_channels}; the input is the 3-contrast triplet")
        if iterations < 1:
            raise ValueError(f"iterations={iterations}; at least one step is needed")
        self.iterations = int(iterations)
        self.signal_floor = float(signal_floor)
        self.checkpoint_iterations = bool(checkpoint_iterations)
        self._ranges = (tuple(rho_range), tuple(t1_range_ms), tuple(t2_range_ms))
        if initializer not in _INITIALISERS:
            raise ValueError(f"initializer={initializer!r}; expected one of {sorted(_INITIALISERS)}")
        # Input: the masked, scale-normalised triplet and the availability mask.
        self.init_net = _INITIALISERS[initializer](base_channels, dict(initializer_kwargs or {}))
        # State (3) + residuals (3) + availability (3) + step plane (1).
        self.prior = _PriorNet(3 * _N_CONTRASTS + 1, base_channels)
        self.step_embedding = nn.Parameter(torch.zeros(self.iterations))
        #: Mean squared data residual entering each step (detached), the per-step diagnostic.
        self.last_residual_norms: list[torch.Tensor] = []

    def _maps(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        for k, (name, (lo, hi)) in enumerate(zip(("rho", "t1", "t2"), self._ranges, strict=True)):
            out[name] = lo + (hi - lo) * torch.sigmoid(z[:, k : k + 1])
        return out

    @staticmethod
    def _render(maps: dict[str, torch.Tensor], protocol: Sequence[dict]) -> list[torch.Tensor]:
        return [tse_signal(maps["rho"], maps["t1"], maps["t2"], **p) for p in protocol]

    def _residuals(
        self,
        renders: list[torch.Tensor],
        x: torch.Tensor,
        avail: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Gain-scaled residuals; unavailable contrasts contribute zeros, and no gain."""
        out = []
        for k, r in enumerate(renders):
            y = x[:, k : k + 1]
            w = ((y > self.signal_floor) & (avail[:, k : k + 1, None, None] > 0)).to(r.dtype)
            with torch.no_grad():
                num = (r * y * w).flatten(1).sum(1)
                gain = (num / (r * r * w).flatten(1).sum(1).clamp_min(1e-12)).view(-1, 1, 1, 1)
            out.append((gain * r - y) * w / scale)
        return torch.cat(out, dim=1)

    def _step(self, z, x, avail, avail_planes, scale, protocol, k):
        maps = self._maps(z)
        res = self._residuals(self._render(maps, protocol), x, avail, scale)
        step = self.step_embedding[k].expand(z.shape[0], 1, *z.shape[2:])
        return z + self.prior(torch.cat([z, res, avail_planes, step], dim=1)), res

    def unroll(
        self,
        x: torch.Tensor,
        protocol: Sequence[dict[str, Any]],
        available: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
        """Return the maps and the unscaled render of every contrast.

        Args:
            x: ``(B, 3, H, W)`` triplet in noise units.
            protocol: one dict per contrast with ``family, tr, te, ti, esp, etl,
                refocus_deg`` (ms), in [T1, T2, FLAIR] order.
            available: ``(B, 3)`` 0/1 mask; ``None`` means every contrast is acquired.
        """
        if x.ndim != 4 or x.shape[1] != _N_CONTRASTS:
            raise ValueError(f"expected (B, 3, H, W), got {tuple(x.shape)}")
        if len(protocol) != _N_CONTRASTS:
            raise ValueError(f"one protocol per contrast is needed, got {len(protocol)}")
        protocol = [{k: p[k] for k in _PROTOCOL_KEYS} for p in protocol]
        b = x.shape[0]
        avail = (
            torch.ones(b, _N_CONTRASTS, device=x.device)
            if available is None
            else available.to(x.device, x.dtype)
        )
        if (avail.sum(1) < 1).any():
            raise ValueError("every sample needs at least one available contrast")
        a4 = avail.view(b, _N_CONTRASTS, 1, 1)
        x = x * a4  # a dropped contrast is never read below this line
        with torch.no_grad():
            scale = (
                torch.quantile(x.flatten(1).float(), 0.99, dim=1).clamp_min(1.0).view(b, 1, 1, 1)
            )
        avail_planes = a4.expand(-1, -1, *x.shape[2:])
        z = self.init_net(x / scale, avail)
        self.last_residual_norms = []
        for k in range(self.iterations):
            if self.training and self.checkpoint_iterations:
                z, res = checkpoint(
                    self._step, z, x, avail, avail_planes, scale, protocol, k, use_reentrant=False
                )
            else:
                z, res = self._step(z, x, avail, avail_planes, scale, protocol, k)
            self.last_residual_norms.append(res.detach().pow(2).mean())  # no host sync
        maps = self._maps(z)
        return maps, self._render(maps, protocol)

    def render_triplet(
        self,
        x: torch.Tensor,
        protocol: Sequence[dict[str, Any]],
        available: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
        """The strategy hook shared with the cohort's other models; see ``unroll``."""
        return self.unroll(x, protocol, available)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Registry contract: without a protocol there is nothing to render."""
        raise RuntimeError(
            "unrolled_tissue_net renders through each contrast's header protocol; call "
            "render_triplet(x, protocol, available) from CalibratedQMapStrategy (takes_protocol)."
        )
