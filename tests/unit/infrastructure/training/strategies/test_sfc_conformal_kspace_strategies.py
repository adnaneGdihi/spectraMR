"""Unit tests for the k-space SFC / conformal training strategies.

Covers :class:`AdaptiveSFCHSSCStrategy` and
:class:`ConformalDiffusionReconStrategy` in
``spectramr.infrastructure.training.strategies.sfc_conformal_kspace_strategies``.

Regression focus (SFC #3): ``AdaptiveSFCHSSCStrategy._compute_losses_impl`` must
introspect the generator's ``forward`` signature before passing the
``sfc_indices`` kwarg. A strict ``forward(self, x)`` generator (no ``**kwargs``)
must still be trained — passing an unexpected kwarg would raise ``TypeError``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from spectramr.infrastructure.training.strategies.sfc_conformal_kspace_strategies import (
    AdaptiveSFCHSSCStrategy,
    ConformalDiffusionReconStrategy,
)
from spectramr.models.blocks.space_filling_curves import BeltramiSFCBlock


class _StubEnv:
    """Minimal stand-in for ``TrainingEnvironment`` exposing ``.generator``.

    ``BaseTrainingStrategy.generator_model`` is a read-only property that does
    ``return self.env.generator``. Rather than fight the class-level property
    (its metaclass is ``ABCMeta`` and rejects ``setattr``), we hand the strategy
    a tiny env object carrying the generator, exercising the real property path
    while staying free of the DI container.
    """

    def __init__(self, generator: object) -> None:
        self.generator = generator


def _bare_strategy(cls: type) -> Any:
    """Construct a strategy without running ``__init__`` (bypasses DI).

    The two strategies are plain ``BaseTrainingStrategy`` subclasses (not
    ``nn.Module``), so we can allocate an instance and attach only the
    attributes ``_compute_losses_impl`` reads.
    """
    return object.__new__(cls)


class _StrictGenerator(torch.nn.Module):
    """Generator with a STRICT ``forward(self, x)`` — no ``**kwargs``.

    Passing an unexpected ``sfc_indices=`` kwarg to this generator raises
    ``TypeError``; it models a registered reconstruction backbone that does
    not advertise the SFC kwarg. The strategy must still train it (the
    docstring promise on :class:`AdaptiveSFCHSSCStrategy`).
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102 - strict sig
        return self.conv(x)


class _SFCAwareGenerator(torch.nn.Module):
    """Generator that NAMES ``sfc_indices`` and records what it received."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=1)
        self.received_sfc_indices: object = "__unset__"

    def forward(
        self, x: torch.Tensor, sfc_indices: object = None
    ) -> torch.Tensor:  # noqa: D102
        self.received_sfc_indices = sfc_indices
        return self.conv(x)


def _make_sfc_strategy(generator: torch.nn.Module) -> AdaptiveSFCHSSCStrategy:
    """Build an ``AdaptiveSFCHSSCStrategy`` with the SFC head + stub env."""
    strategy = _bare_strategy(AdaptiveSFCHSSCStrategy)
    strategy.sfc = BeltramiSFCBlock(grid_size=16, in_channels=1)
    strategy.lambda_mu = 0.01
    strategy.lambda_traj = 0.01
    strategy.device = None
    # What BaseTrainingStrategy.__init__ provides for owned modules (#2629).
    strategy.config = SimpleNamespace(parallel=None)
    strategy._owned_modules = {}
    strategy._owned_wrappers = {}
    strategy.register_owned_module("sfc", strategy.sfc)
    # ``generator_model`` is a read-only property delegating to
    # ``self.env.generator`` — drive it through a stub env.
    strategy.env = _StubEnv(generator)
    return strategy


class TestAdaptiveSFCStrictGeneratorForward:
    """Regression: SFC strategy must introspect before passing ``sfc_indices``.

    Pre-fix, ``_compute_losses_impl`` called
    ``self.generator_model(input_batch, sfc_indices=...)`` unconditionally,
    which raised ``TypeError`` for any generator whose ``forward`` did not
    accept the kwarg — contradicting the docstring promise that models which
    ignore the kwarg are still trained.
    """

    def test_strict_no_kwarg_generator_trains_without_typeerror(self) -> None:
        """A strict ``forward(self, x)`` generator must NOT raise."""
        gen = _StrictGenerator()
        strategy = _make_sfc_strategy(gen)
        # Pre-fix this raised: TypeError: forward() got an unexpected
        # keyword argument 'sfc_indices'.
        out = strategy._compute_losses_impl(
            torch.randn(2, 1, 16, 16),
            torch.randn(2, 1, 16, 16),
            epoch=0,
        )
        assert "g_total_loss" in out
        assert out["g_total_loss"].requires_grad

    def test_sfc_aware_generator_receives_indices(self) -> None:
        """A generator that names ``sfc_indices`` must still receive them."""
        gen = _SFCAwareGenerator()
        strategy = _make_sfc_strategy(gen)
        strategy._compute_losses_impl(
            torch.randn(2, 1, 16, 16),
            torch.randn(2, 1, 16, 16),
            epoch=0,
        )
        # The kwarg must have been forwarded (not the "__unset__" sentinel and
        # not left at the default None — the SFC head produces real indices).
        assert gen.received_sfc_indices is not None
        assert gen.received_sfc_indices != "__unset__"
        assert isinstance(gen.received_sfc_indices, torch.Tensor)


class TestConformalDiffusionReconStrategy:
    """The conformal DC step is the strategy's whole point — it must fail loud
    when the batch lacks the keys it needs, not silently run plain MSE."""

    def _strategy(self) -> Any:
        from spectramr.infrastructure.physics.data_consistency import (
            ConformalDataConsistency,
        )

        strategy = _bare_strategy(ConformalDiffusionReconStrategy)
        strategy.sigma_min = 1e-2
        strategy.sigma_max = 1.0
        strategy.dc = ConformalDataConsistency(alpha=1.0)
        strategy.env = _StubEnv(torch.nn.Conv2d(1, 1, kernel_size=1))
        return strategy

    def test_missing_mask_and_jacobian_raises(self) -> None:
        """No mask + no conformal_jacobian → ValueError (was: warn + plain MSE,
        the conformal paradigm silently inert, pitfall #16)."""
        strategy = self._strategy()
        with pytest.raises(ValueError, match="conformal_jacobian"):
            strategy._compute_losses_impl(
                torch.randn(2, 1, 16, 16),
                torch.randn(2, 1, 16, 16),
                epoch=0,
                batch={},
            )

    def test_missing_jacobian_only_raises(self) -> None:
        """A mask without the Jacobian still can't run conformal DC → raise."""
        strategy = self._strategy()
        with pytest.raises(ValueError, match="conformal_jacobian"):
            strategy._compute_losses_impl(
                torch.randn(2, 1, 16, 16),
                torch.randn(2, 1, 16, 16),
                epoch=0,
                batch={"mask": torch.ones(2, 1, 16, 16)},
            )

    def test_loss_runs_with_mask_and_jacobian(self) -> None:
        """Happy path: both keys present → the DC projection runs and the loss
        is a differentiable scalar."""
        strategy = self._strategy()
        out = strategy._compute_losses_impl(
            torch.randn(2, 1, 16, 16),
            torch.randn(2, 1, 16, 16),
            epoch=0,
            batch={
                "mask": torch.ones(2, 1, 16, 16),
                "conformal_jacobian": torch.ones(2, 1, 16, 16),
            },
        )
        assert "g_total_loss" in out
        assert out["g_total_loss"].requires_grad
        assert "conformal_diffusion_dsm" in out
