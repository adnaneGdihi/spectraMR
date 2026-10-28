"""Regression tests for the 2026-06 audit fixes to ``ReconstructionTrainingStrategy``.

The shared reconstruction base is exercised only through full training/validation
forwards (it needs a built environment), so these tests pin the audit fixes by
introspecting the control flow rather than spinning up a generator:

* an enabled PINN module that errors must re-raise (not silently drop the term);
* a model advertising ``forward_with_intermediates`` that errors must re-raise
  (not silently lose its deep-supervision path);
* validation must use a deterministic timestep (random draw was non-reproducible).
"""
from __future__ import annotations

import inspect
import re

from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionStepMixin,
    ReconstructionTrainingStrategy,
)

# The step body lives on the shared mixin; the concrete strategy composes it.
_SRC = inspect.getsource(ReconstructionStepMixin)


def test_the_concrete_strategy_composes_the_step() -> None:
    assert ReconstructionStepMixin in ReconstructionTrainingStrategy.__mro__


def _reraises_after(marker: str, window: int = 400) -> bool:
    """True if a bare ``raise`` follows ``marker`` within ``window`` chars."""
    idx = _SRC.find(marker)
    assert idx != -1, f"marker not found: {marker!r}"
    segment = _SRC[idx : idx + window]
    return re.search(r"\n\s+raise\b", segment) is not None


def test_pinn_loss_failure_reraises() -> None:
    assert _reraises_after("PINN loss computation failed")


def test_intermediate_outputs_failure_reraises() -> None:
    assert _reraises_after("Failed to get intermediate outputs from generator")


def test_validation_uses_deterministic_timesteps() -> None:
    # The val path threads validation=True and draws a deterministic timestep.
    assert "validation: bool = False" in _SRC
    assert "validation=True" in _SRC
    # #2759: one fixed t on every split, so training has no random draw either;
    # the behaviour itself is pinned in test_reconstruction.py.
    assert "torch.rand((B,)" not in _SRC
    assert 'if _callable_accepts_kwarg(generator_forward, "timesteps"):' in _SRC
