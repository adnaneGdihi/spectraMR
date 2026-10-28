"""``apply_degradation`` for tests that sweep the whole registry.

An axis that declares ``requires`` (inputs the image cannot supply) would raise ``MissingInput``
in a registry-wide test, so this wrapper adds the inputs the golden gate builds for it
(``check_degradation_golden.required_kwargs``) unless the caller passed them.
"""

from __future__ import annotations

import torch

from scripts.ci.degradation_golden_inputs import required_kwargs
from spectramr.infrastructure.physics.degradations import apply_degradation as _apply

__all__ = ["apply_degradation"]


def apply_degradation(
    name: str, x: torch.Tensor, theta: float, seed: int = 0, **kwargs
) -> torch.Tensor:
    """``registry.apply_degradation`` with the axis's required inputs filled in."""
    for key, value in required_kwargs(name, x).items():
        kwargs.setdefault(key, value)
    return _apply(name, x, theta, seed, **kwargs)
