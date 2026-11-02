r"""DP-SGD mixin for federated / privacy-preserving training strategies.

Implements PR-7 (H5) of
``TODO/backlog_paradigm_expansion_roadmap.md``. Wraps a strategy's
backward pass with the canonical DP-SGD update from Abadi et al. 2016:

1. **Per-sample gradient clipping** to bound the L2 sensitivity to
   ``max_grad_norm``.
2. **Gaussian noise injection** scaled by ``noise_multiplier``.
3. **Privacy accounting** via :class:`RDPAccountant` so the strategy
   exposes a live ``(ε, δ)`` budget.

Usage
-----
.. code-block:: python

    class FederatedDPStrategy(BaseTrainingStrategy, DPMixin):
        def __init__(self, ...):
            super().__init__(...)
            self.configure_dp(
                noise_multiplier=1.1,
                max_grad_norm=1.0,
                sample_rate=batch_size / dataset_size,
                target_delta=1e-5,
            )

The mixin is intentionally minimal — it neither installs autograd
hooks nor depends on opacus, so it works in any environment that has
the optional ``[dp]`` extra *and* in environments that don't. For
production-grade per-sample gradients use ``opacus.PrivacyEngine``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from spectramr.domain.interfaces import IPrivacyAccountant
from spectramr.infrastructure.services.privacy.dp_sgd import (
    add_gaussian_noise,
    clip_per_sample_grad,
)
from spectramr.infrastructure.services.privacy.rdp_accountant import RDPAccountant

logger = logging.getLogger(__name__)


class DPMixin:
    """Adds DP-SGD post-processing to a training strategy's backward pass.

    The mixin assumes the host strategy already produces a gradient
    via its standard backward pass; the mixin then clips, noises, and
    accounts on top.

    Concrete strategies must call :py:meth:`configure_dp` before the
    first training step. They should also call :py:meth:`record_dp_step`
    after each optimiser update so the accountant stays in sync.
    """

    # Set up by configure_dp; defaults make the mixin a no-op until
    # explicitly configured.
    _dp_enabled: bool = False
    _noise_multiplier: float = 0.0
    _max_grad_norm: float = 0.0
    _sample_rate: float = 1.0
    _target_delta: float = 1e-5
    _accountant: IPrivacyAccountant | None = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_dp(
        self,
        noise_multiplier: float,
        max_grad_norm: float,
        sample_rate: float,
        target_delta: float = 1e-5,
        accountant: IPrivacyAccountant | None = None,
    ) -> None:
        """Activate the DP-SGD post-processor.

        Args:
            noise_multiplier: ``σ`` in ``N(0, σ² Δ²)``. Must be > 0.
            max_grad_norm: Per-sample L2 clip threshold. Must be > 0.
            sample_rate: Per-sample inclusion probability (typically
                ``batch_size / dataset_size``). Must lie in ``(0, 1]``.
            target_delta: Target ``δ`` of the ``(ε, δ)``-DP guarantee.
            accountant: Optional pre-built privacy accountant. If
                ``None``, an :class:`RDPAccountant` is constructed.

        Raises:
            ValueError: invalid arguments.
        """
        if noise_multiplier <= 0:
            raise ValueError(f"noise_multiplier must be > 0; got {noise_multiplier}")
        if max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be > 0; got {max_grad_norm}")
        if not 0.0 < sample_rate <= 1.0:
            raise ValueError(f"sample_rate must lie in (0, 1]; got {sample_rate}")
        if not 0.0 < target_delta < 1.0:
            raise ValueError(f"target_delta must lie in (0, 1); got {target_delta}")
        self._dp_enabled = True
        self._noise_multiplier = float(noise_multiplier)
        self._max_grad_norm = float(max_grad_norm)
        self._sample_rate = float(sample_rate)
        self._target_delta = float(target_delta)
        self._accountant = accountant or RDPAccountant()
        logger.info(
            "[DPMixin] configured: σ=%.3f, C=%.3f, q=%.4f, δ=%.1e",
            noise_multiplier,
            max_grad_norm,
            sample_rate,
            target_delta,
        )

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def dp_enabled(self) -> bool:
        """``True`` iff :py:meth:`configure_dp` has been called."""
        return self._dp_enabled

    @property
    def accountant(self) -> IPrivacyAccountant:
        """The (Rényi-)DP accountant tracking the privacy budget."""
        if self._accountant is None:
            raise RuntimeError("DPMixin not yet configured — call configure_dp() first.")
        return self._accountant

    def get_epsilon(self) -> float:
        """Current ``ε`` of the ``(ε, δ)``-DP guarantee."""
        return self.accountant.get_epsilon(delta=self._target_delta)

    # ------------------------------------------------------------------
    # Backward-pass hooks
    # ------------------------------------------------------------------

    def dp_post_process_per_sample_grads(
        self,
        per_sample_grads: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Apply per-sample clipping then add noise, returning the *averaged* gradient.

        The host strategy is responsible for providing per-sample
        gradients (e.g. via ``torch.func.vmap`` or ``opacus``'s hooks).
        The mixin then clips, averages, and noises.

        Args:
            per_sample_grads: Tensor ``[B, *param_shape]``.
            generator: Optional ``torch.Generator`` for reproducibility.

        Returns:
            Single noised gradient tensor ``[*param_shape]``.

        Raises:
            RuntimeError: if :py:meth:`configure_dp` was not called.
        """
        if not self._dp_enabled:
            raise RuntimeError("dp_post_process_per_sample_grads called before configure_dp().")
        clipped = clip_per_sample_grad(per_sample_grads, max_grad_norm=self._max_grad_norm)
        batch_size = clipped.shape[0]
        averaged = clipped.mean(dim=0)
        return add_gaussian_noise(
            averaged,
            noise_multiplier=self._noise_multiplier,
            max_grad_norm=self._max_grad_norm,
            batch_size=batch_size,
            generator=generator,
        )

    def record_dp_step(self) -> None:
        """Tell the accountant one DP-SGD round has been executed.

        Call this once per optimiser step (after the gradient has been
        clipped + noised + applied). The accountant updates its Rényi
        divergences with the configured ``noise_multiplier`` /
        ``sample_rate``.
        """
        if not self._dp_enabled:
            return
        self.accountant.step(
            noise_multiplier=self._noise_multiplier,
            sample_rate=self._sample_rate,
        )

    def restore_dp_steps(self, steps: int) -> None:
        """Compose ``steps`` saved rounds onto a fresh accountant (checkpoint resume).

        Raises:
            RuntimeError: DP is off or the accountant already holds steps, so the
                restored ledger would not match the saved one.
        """
        held = self.accountant.steps if self._dp_enabled else 0
        if steps == held:
            return
        if not self._dp_enabled or held != 0:
            raise RuntimeError(
                f"Cannot restore {steps} DP steps onto an accountant that is "
                f"{'disabled' if not self._dp_enabled else f'already at {held} steps'}."
            )
        self.accountant.step(self._noise_multiplier, self._sample_rate, num_steps=int(steps))

    # ------------------------------------------------------------------
    # Diagnostics for the metrics dict
    # ------------------------------------------------------------------

    def dp_metrics(self) -> dict[str, Any]:
        """Lightweight key-value snapshot for the loss-logging CSV."""
        if not self._dp_enabled:
            return {}
        return {
            "dp/epsilon": self.get_epsilon(),
            "dp/delta": self._target_delta,
            "dp/steps": float(self.accountant.steps),
            "dp/noise_multiplier": self._noise_multiplier,
            "dp/max_grad_norm": self._max_grad_norm,
        }
