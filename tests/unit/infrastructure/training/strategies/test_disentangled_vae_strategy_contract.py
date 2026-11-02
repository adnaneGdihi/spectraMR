"""Regression: DisentangledVAETrainingStrategy enforces its generator API contract.

The 2026-05-10 cluster smoke ran ``experiment_55_disentangled_representation``
with ``model_type: sparse_vae`` — which returns a 4-tuple
``(reconstruction, mu, log_var, sparse_z)`` and lacks the
``encode_content`` / ``decode`` API. The strategy used to crash at the
first forward with ``ValueError: too many values to unpack (expected 3)``
at line 229's ``recon_64, mu_64, logvar_64 = generator(...)``.

The hardened contract check raises ``TypeError`` *before* the unpack,
naming both the offending model and the required API — so:

* The error is loud and actionable (CLAUDE.md #9 "no silent fallbacks").
* It fires at strategy setup time conceptually, not after iter 1 of
  training, saving GPU time on the cluster.
* Anyone copy-pasting an old sparse_vae YAML into this strategy now sees
  exactly what to change.

This test exercises the check directly via the bound method on a real
strategy instance built from a minimal mock config — no actual training
takes place.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")


def _strategy_with_mock_generator(generator: Any) -> Any:
    """Build a real DisentangledVAETrainingStrategy whose ``env.generator``
    is the supplied mock, skipping the heavy DI bootstrap.
    """
    from spectramr.infrastructure.training.strategies import disentangled_vae_strategy as mod

    strategy = mod.DisentangledVAETrainingStrategy.__new__(
        mod.DisentangledVAETrainingStrategy
    )
    strategy.config = SimpleNamespace(model=SimpleNamespace(model_type="mock"))
    strategy.env = SimpleNamespace(generator=generator)
    strategy.device = torch.device("cpu")
    return strategy


def test_strategy_rejects_generator_without_encode_content() -> None:
    """sparse_vae-shaped generator (4-tuple, no encode_content) is rejected."""
    bad_gen = MagicMock(spec=[])  # no methods at all
    strategy = _strategy_with_mock_generator(bad_gen)

    img = torch.randn(2, 1, 16, 16)
    with pytest.raises(TypeError, match=r"requires a generator.*encode_content"):
        # patch the physics-vector registry so the function runs far enough
        # to hit the contract check at line 215-224.
        with patch(
            "spectramr.infrastructure.training.strategies.disentangled_vae_physics."
            "AcquisitionRegistry.get_physics_vector",
            return_value=torch.zeros(1, 4),
        ):
            strategy._compute_losses_impl(img, img, epoch=0)


def test_strategy_rejects_generator_without_decode() -> None:
    """A generator that has ``encode_content`` but not ``decode`` still fails."""
    half_gen = MagicMock(spec=["encode_content"])
    half_gen.encode_content = MagicMock(return_value=torch.randn(2, 32, 4, 4))
    strategy = _strategy_with_mock_generator(half_gen)

    img = torch.randn(2, 1, 16, 16)
    with pytest.raises(TypeError, match=r"requires a generator.*encode_content"):
        with patch(
            "spectramr.infrastructure.training.strategies.disentangled_vae_physics."
            "AcquisitionRegistry.get_physics_vector",
            return_value=torch.zeros(1, 4),
        ):
            strategy._compute_losses_impl(img, img, epoch=0)
