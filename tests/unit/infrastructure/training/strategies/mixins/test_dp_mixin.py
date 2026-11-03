"""Tests for ``DPMixin``.

Targets ``spectramr.infrastructure.training.strategies.mixins.dp_mixin``.

PR-7 (H5) — DP-SGD wraps a strategy's backward pass with per-sample
clipping + Gaussian noise + Rényi-DP accounting.

Categories:

- Mixin defaults to *disabled*; calling DP hooks before
  ``configure_dp`` raises
- ``configure_dp`` validates noise_multiplier / max_grad_norm /
  sample_rate / target_delta
- ``dp_post_process_per_sample_grads`` returns a single-sample-shaped
  averaged + noised gradient
- ``record_dp_step`` increments the accountant's step counter
- ``get_epsilon`` returns 0 before first step, > 0 after
- ``dp_metrics`` snapshots empty when disabled / populated when enabled
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.strategies.mixins.dp_mixin import DPMixin


class _Host(DPMixin):
    """Minimal host class for the mixin under test."""


# ---------------------------------------------------------------------------
# Disabled / configured contract
# ---------------------------------------------------------------------------


def test_default_disabled() -> None:
    """A fresh host has ``dp_enabled = False``."""
    host = _Host()
    assert host.dp_enabled is False


def test_post_process_before_configure_raises() -> None:
    """Calling the post-processor before ``configure_dp`` is an error."""
    host = _Host()
    with pytest.raises(RuntimeError, match="configure_dp"):
        host.dp_post_process_per_sample_grads(torch.zeros(4, 8))


def test_record_dp_step_silent_when_disabled() -> None:
    """``record_dp_step`` is a no-op while disabled (no exception)."""
    host = _Host()
    host.record_dp_step()  # must not raise


def test_dp_metrics_empty_when_disabled() -> None:
    """``dp_metrics`` returns an empty dict before configuration."""
    host = _Host()
    assert host.dp_metrics() == {}


# ---------------------------------------------------------------------------
# configure_dp validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwarg,value",
    [
        ("noise_multiplier", 0.0),
        ("noise_multiplier", -1.0),
        ("max_grad_norm", 0.0),
        ("sample_rate", 0.0),
        ("sample_rate", 1.5),
        ("target_delta", 0.0),
        ("target_delta", 1.0),
    ],
)
def test_configure_dp_invalid_args_rejected(kwarg: str, value: float) -> None:
    """Invalid arguments raise ``ValueError``."""
    base = dict(
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        sample_rate=0.1,
        target_delta=1e-5,
    )
    base[kwarg] = value
    with pytest.raises(ValueError, match=kwarg):
        _Host().configure_dp(**base)


def test_configure_dp_enables_mixin() -> None:
    """After valid configuration, ``dp_enabled`` flips to True."""
    host = _Host()
    host.configure_dp(noise_multiplier=1.0, max_grad_norm=1.0, sample_rate=0.1)
    assert host.dp_enabled is True


# ---------------------------------------------------------------------------
# Post-process gradients
# ---------------------------------------------------------------------------


def test_post_process_returns_param_shape() -> None:
    """Output shape collapses the per-sample batch dim."""
    host = _Host()
    host.configure_dp(
        noise_multiplier=0.5, max_grad_norm=1.0, sample_rate=0.1
    )
    per_sample = torch.randn(8, 4, 4)
    out = host.dp_post_process_per_sample_grads(per_sample)
    assert out.shape == (4, 4)


def test_post_process_with_zero_noise_equals_clipped_mean() -> None:
    """``σ → 0+`` recovers the deterministic clipped-mean gradient."""
    host = _Host()
    host.configure_dp(
        noise_multiplier=1e-9, max_grad_norm=1.0, sample_rate=0.1
    )
    per_sample = torch.randn(8, 4)
    out = host.dp_post_process_per_sample_grads(per_sample)
    # Compute the same operation manually for comparison.
    from spectramr.infrastructure.services.privacy.dp_sgd import clip_per_sample_grad

    expected_mean = clip_per_sample_grad(per_sample, 1.0).mean(dim=0)
    # Loose tolerance: a tiny bit of noise is added.
    assert torch.allclose(out, expected_mean, atol=1e-6)


def test_post_process_seeded_generator_reproducible() -> None:
    """Same generator seed → identical noised output."""
    host = _Host()
    host.configure_dp(noise_multiplier=1.0, max_grad_norm=1.0, sample_rate=0.1)
    per_sample = torch.randn(4, 6)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    out1 = host.dp_post_process_per_sample_grads(per_sample, generator=g1)
    out2 = host.dp_post_process_per_sample_grads(per_sample, generator=g2)
    assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


def test_get_epsilon_zero_before_step() -> None:
    """Before any ``record_dp_step``, ε = 0 by convention."""
    host = _Host()
    host.configure_dp(noise_multiplier=1.0, max_grad_norm=1.0, sample_rate=0.1)
    assert host.get_epsilon() == 0.0


def test_record_dp_step_increments_accountant() -> None:
    """Each call adds one entry to the accountant ledger."""
    host = _Host()
    host.configure_dp(noise_multiplier=1.0, max_grad_norm=1.0, sample_rate=0.1)
    for _ in range(3):
        host.record_dp_step()
    assert host.accountant.steps == 3


def test_get_epsilon_grows_after_steps() -> None:
    """ε is strictly positive after at least one DP step."""
    host = _Host()
    host.configure_dp(noise_multiplier=1.0, max_grad_norm=1.0, sample_rate=0.1)
    host.record_dp_step()
    assert host.get_epsilon() > 0.0


def test_dp_metrics_includes_canonical_keys() -> None:
    """``dp_metrics`` snapshot exposes the documented surface."""
    host = _Host()
    host.configure_dp(noise_multiplier=1.0, max_grad_norm=1.0, sample_rate=0.1)
    host.record_dp_step()
    metrics = host.dp_metrics()
    for key in ("dp/epsilon", "dp/delta", "dp/steps", "dp/noise_multiplier", "dp/max_grad_norm"):
        assert key in metrics


def test_restore_dp_steps_matches_live_ledger_and_refuses_overwrite() -> None:
    """A restored step count reproduces the live epsilon; restoring onto a used ledger raises."""
    live, resumed = _Host(), _Host()
    for host in (live, resumed):
        host.configure_dp(noise_multiplier=1.1, max_grad_norm=1.0, sample_rate=0.05)
    for _ in range(5):
        live.record_dp_step()
    resumed.restore_dp_steps(5)
    assert resumed.accountant.steps == 5
    assert resumed.get_epsilon() == pytest.approx(live.get_epsilon(), rel=1e-12)
    resumed.restore_dp_steps(5)  # idempotent at the same count
    with pytest.raises(RuntimeError, match="already at 5"):
        resumed.restore_dp_steps(9)
    with pytest.raises(RuntimeError, match="disabled"):
        _Host().restore_dp_steps(1)
