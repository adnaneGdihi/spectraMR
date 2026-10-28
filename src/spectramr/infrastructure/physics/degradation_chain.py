"""An ordered, compounded degradation chain with a fixed severity per axis.

The chain is the fitted object: ``[(axis, theta), ...]``. Its severities are emitted
as DEGENERATE ``degradation_ranges`` -- ``(theta, theta)`` -- which is what pins an
axis to a constant regardless of the diffusion corruption factor, because
:meth:`DigitalTwinSimulator._get_effective_cf` returns ``vmin + (vmax - vmin) * eff``.
That is the whole reason this module introduces no new runtime degradation path: a
fitted chain replays through the already-wired simulator as ordinary config.

Chain ORDER is not free. The simulator runs its native pipeline first and registry
axes afterwards in list order, so a chain is a *declared axis set* whose severities
are fitted; orderings are never searched.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from spectramr.infrastructure.physics.degradations.registry import (
    DEGRADATION_REGISTRY,
    apply_degradation,
    derive_axis_seed,
)
from spectramr.infrastructure.physics.digital_twin_simulator import (
    NATIVE_DEGRADATION_AXES,
    known_degradation_axes,
)

__all__ = [
    "ChainLink",
    "DegradationChain",
    "UnapplicableAxisError",
    "UnreplayableAxisError",
]

#: Axes present in BOTH banks. The simulator handles these natively and filters
#: them out of its registry loop, so the operator a fit SCORED is not the one a
#: replay would run -- and the native branch is gated on an enable flag that
#: defaults False, so in practice the axis contributes nothing at all (measured
#: on a nested-block phantom: b0 moves the volume 0.59 through the registry op
#: and 0.00 on replay). Derived from the two banks rather than written out, so a
#: newly dual-listed axis is covered the day it is added.
_DUAL_BANK_AXES: frozenset[str] = NATIVE_DEGRADATION_AXES & frozenset(DEGRADATION_REGISTRY)


class UnapplicableAxisError(ValueError):
    """An axis that cannot be applied as a standalone registry operator."""


class UnreplayableAxisError(ValueError):
    """An axis the fit can score but the simulator would not reproduce."""


@dataclass(frozen=True, slots=True)
class ChainLink:
    """One degradation at one severity."""

    axis: str
    theta: float


@dataclass(frozen=True, slots=True)
class DegradationChain:
    """An ordered compound of registry degradations at fixed severities."""

    links: tuple[ChainLink, ...]

    def __post_init__(self) -> None:
        if not self.links:
            raise ValueError("a DegradationChain must have at least one link")
        for link in self.links:
            if link.axis not in DEGRADATION_REGISTRY:
                if link.axis in NATIVE_DEGRADATION_AXES:
                    raise UnapplicableAxisError(
                        f"{link.axis!r} is a native DigitalTwinSimulator axis, not a "
                        "DEGRADATION_REGISTRY operator, so it cannot be applied "
                        "standalone. Use the registry equivalent instead (e.g. "
                        "'rigid_motion' rather than 'motion')."
                    )
                raise UnapplicableAxisError(
                    f"{link.axis!r} is not a known degradation. Valid: "
                    f"{sorted(known_degradation_axes())}"
                )
            if link.axis in _DUAL_BANK_AXES:
                raise UnreplayableAxisError(
                    f"{link.axis!r} is in BOTH the simulator's native bank and "
                    "DEGRADATION_REGISTRY. The fit would score the REGISTRY "
                    "operator, but the simulator filters dual-bank names out of "
                    "its registry loop and defers to a native branch whose enable "
                    "flag defaults False -- so the axis contributes nothing on "
                    "replay and the calibration would not reproduce. Dual-bank "
                    f"axes: {sorted(_DUAL_BANK_AXES)}."
                )
            if not 0.0 <= link.theta <= 1.0:
                raise ValueError(f"theta for {link.axis!r} must lie in [0, 1]; got {link.theta}")

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(link.axis for link in self.links)

    @property
    def thetas(self) -> tuple[float, ...]:
        return tuple(link.theta for link in self.links)

    def with_thetas(self, thetas: Sequence[float]) -> DegradationChain:
        """Same axes, new severities -- the fitter's per-iteration constructor."""
        if len(thetas) != len(self.links):
            raise ValueError(
                f"thetas length {len(thetas)} does not match the chain's {len(self.links)} links"
            )
        return DegradationChain(
            links=tuple(
                ChainLink(axis=link.axis, theta=float(t))
                for link, t in zip(self.links, thetas, strict=True)
            )
        )

    def apply(self, x: torch.Tensor, *, seed: int) -> torch.Tensor:
        """Compound every link onto ``x`` in order. ``x`` is ``(B, C, H, W)``.

        Per-axis seeds are derived so distinct axes never share one artefact
        realisation, while each axis's seed stays independent of its theta -- theta
        must remain the only free variable, or a severity sweep also sweeps noise.
        """
        out = x
        for link in self.links:
            out = apply_degradation(
                link.axis,
                out,
                theta=link.theta,
                seed=derive_axis_seed(seed, link.axis),
            )
        return out

    def to_digital_twin_config(self) -> dict[str, Any]:
        """Emit this chain as ``DigitalTwinConfig`` fields, faithful on replay.

        EVERY axis is listed in ``degradation_ranges``: an omitted axis defaults to
        ``(0.0, 1.0)``, which would silently track the corruption factor instead of
        holding the fitted severity.

        Every native stage is pinned to ``(0.0, 0.0)``, ``noise`` included, although
        none is a chain axis. The
        simulator's AWGN stage has no enable flag, and ``_get_effective_cf`` holds
        any feature *absent* from a non-empty ``progressive_degradations`` at 1.0 --
        so a block naming only the fitted axes replays them PLUS a fresh 10-25 dB
        scanner-noise draw the fit never saw. Measured on a nested-block phantom:
        with every theta at 0 the unpinned block moved the volume by 0.169 relative
        L2 where the chain itself moves it by 0.0087, and pinning brings that to
        0.0083. The chain owns the noise budget through its own noise axis; the
        native stage has to be silent or the replayed volume is not the fitted one.

        ``enabled`` is emitted because a chain that is not enabled is a
        contradiction -- without it the block is inert wherever the twin is gated on
        it. ``apply_as_transform`` deliberately is NOT: it selects the ROUTE (a
        data-pipeline transform versus a VF strategy corrupting internally), and
        forcing it on would double-corrupt a VF arm.
        """
        return {
            "enabled": True,
            "progressive_degradations": list(self.axes),
            "degradation_ranges": {
                # Every native stage is pinned to 0, FIRST so a chain axis can never be
                # overwritten by it: a stage the consumer enables but the fit never saw
                # (b1, motion, ...) is otherwise held at full strength on replay.
                # Native names are rejected as links by __post_init__.
                **dict.fromkeys(sorted(NATIVE_DEGRADATION_AXES), (0.0, 0.0)),
                **{link.axis: (link.theta, link.theta) for link in self.links},
            },
        }
