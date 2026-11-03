"""Privacy-accountant protocol.

Implements ``CC`` interface for ``PR-7 / H5`` of
``TODO/backlog_paradigm_expansion_roadmap.md``: a Rényi-DP accountant
for tracking the cumulative privacy loss across DP-SGD rounds.

A privacy accountant tracks a *privacy budget*: each call to
:py:meth:`step` records a sub-sampled Gaussian mechanism with the
given noise multiplier and sample rate; :py:meth:`get_epsilon` then
converts the accumulated Rényi divergences into an ``(ε, δ)``-DP
guarantee.

Implementations live under ``src/spectramr/infrastructure/services/privacy/``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class IPrivacyAccountant(Protocol):
    """Track DP-SGD privacy budget across training rounds.

    Implementations should:

    - Be additive: each :py:meth:`step` records *one* application of
      the sub-sampled Gaussian mechanism.
    - Permit different noise multipliers / sample rates across calls
      (heterogeneous schedules are common).
    - Fail loud (no silent saturation) when the budget would exceed
      the ``target_epsilon`` configured by the caller.
    """

    @property
    def steps(self) -> int:
        """Number of mechanism applications recorded so far."""
        ...

    def step(
        self,
        noise_multiplier: float,
        sample_rate: float,
        num_steps: int = 1,
    ) -> None:
        """Record ``num_steps`` applications of the sub-sampled Gaussian mechanism.

        Args:
            noise_multiplier: ``sigma`` in ``N(0, sigma^2 Delta^2)``. Must be > 0.
            sample_rate: Probability that a given sample is included
                in the mini-batch. Must lie in ``(0, 1]``.
            num_steps: Identical rounds to compose at once (e.g. on resume). Must be >= 1.
        """
        ...

    def get_epsilon(self, delta: float) -> float:
        """Convert the accumulated Rényi divergences into ``ε``-DP at ``δ``.

        Args:
            delta: Target ``δ`` of the ``(ε, δ)``-DP guarantee. Must
                lie in ``(0, 1)``.

        Returns:
            Tightest ``ε`` consistent with the accountant's history.
        """
        ...
