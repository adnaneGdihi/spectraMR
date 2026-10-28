"""Regression tests for the 2026-06 strategy-audit HIGH-tier fixes.

Each test pins a specific defect surfaced by the strategy audit
(``TODO/strategy_audit_2026_05_31_unreviewed_findings.txt``) so it cannot
silently regress. The fixes are deliberately small and self-contained to the
strategy files; these tests exercise the fixed seams with minimal fakes rather
than spinning up a full training environment.
"""
from __future__ import annotations

import inspect

import pytest
import torch

# ---------------------------------------------------------------------------
# guided_sr_strategy
# ---------------------------------------------------------------------------


def test_guided_sr_to_4d_projects_volume_and_passes_4d_through() -> None:
    from spectramr.infrastructure.training.strategies.guided_sr_strategy import (
        GuidedSuperResolutionStrategy,
    )

    vol = torch.zeros(2, 1, 3, 8, 8)  # [B, C, D, H, W]
    out = GuidedSuperResolutionStrategy._to_4d(vol)
    assert out.shape == (6, 1, 8, 8)  # depth folded into batch

    img = torch.zeros(4, 1, 8, 8)
    assert GuidedSuperResolutionStrategy._to_4d(img).shape == (4, 1, 8, 8)


def test_ssim_hfen_construct_without_channel_kwarg() -> None:
    """The old guided_sr code called ``SSIMLoss(window_size=7, channel=...)``
    (an unsupported kwarg) behind a bare ``except: pass`` so the term was always
    silently dropped. The fixed call must construct and run cleanly on 4-D."""
    from spectramr.models.losses.hfen_loss import HFENLoss
    from spectramr.models.losses.ssim_loss import SSIMLoss

    pred = torch.rand(2, 1, 16, 16)
    target = torch.rand(2, 1, 16, 16)

    ssim = SSIMLoss(window_size=7)
    hfen = HFENLoss()
    assert torch.isfinite(ssim(pred, target)).all()
    assert torch.isfinite(hfen(pred, target)).all()

    # The buggy signature must still be rejected loudly (not silently swallowed).
    with pytest.raises(TypeError):
        SSIMLoss(window_size=7, channel=1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# hssc_strategy
# ---------------------------------------------------------------------------


def test_hssc_dead_hilbert_helper_removed() -> None:
    from spectramr.infrastructure.training.strategies.hssc_strategy import HSSCStrategy

    assert not hasattr(HSSCStrategy, "_resolve_hilbert")
    assert not hasattr(HSSCStrategy, "_hilbert")


def test_hssc_data_consistency_trusts_observed_inside_mask() -> None:
    from spectramr.infrastructure.training.strategies.hssc_strategy import HSSCStrategy

    k_obs = torch.ones(1, 1, 4, 4)
    k_hat = torch.zeros(1, 1, 4, 4)
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask[..., 0, :] = True  # one observed row

    # self is unused on this path → pass None.
    out = HSSCStrategy._data_consistency(None, k_hat, k_obs, mask)
    assert torch.equal(out[..., 0, :], k_obs[..., 0, :])  # observed kept
    assert torch.equal(out[..., 1:, :], k_hat[..., 1:, :])  # model elsewhere


# ---------------------------------------------------------------------------
# ib_active_acquisition_strategy
# ---------------------------------------------------------------------------


def test_ib_active_select_next_line_rejects_unknown_mode() -> None:
    from spectramr.infrastructure.training.strategies.ib_active_acquisition_strategy import (
        IBActiveAcquisitionStrategy,
    )

    assert frozenset(
        {"linear_gaussian", "nonlinear"}
    ) == IBActiveAcquisitionStrategy.VALID_MODES

    observed = torch.zeros(4, dtype=torch.bool)
    est = torch.zeros(1, 4)
    with pytest.raises(ValueError, match="unknown mode"):
        IBActiveAcquisitionStrategy.select_next_line(
            observed, est, mode="linear-gaussian"  # typo
        )

    # The valid linear-Gaussian path still returns an unobserved index.
    idx = IBActiveAcquisitionStrategy.select_next_line(
        observed, est, mode="linear_gaussian"
    )
    assert 0 <= idx < 4


# ---------------------------------------------------------------------------
# n2n_strategy
# ---------------------------------------------------------------------------


def test_n2n_single_repetition_raises_explicitly() -> None:
    from spectramr.infrastructure.training.strategies.n2n_strategy import pair_from_stack

    x = torch.zeros(2, 1, 4, 4)
    with pytest.raises(ValueError, match="at least two"):
        pair_from_stack(x, x, "random_pair")  # one slot: nothing to pair with


def test_n2n_reads_iteration_kwarg_not_step() -> None:
    """The schedule was frozen at iteration 0 because the strategy read the
    nonexistent ``step`` kwarg. N2N now runs the shared reconstruction step,
    which reads the canonical ``iteration``."""
    from spectramr.infrastructure.training.strategies.n2n_strategy import (
        NoiseToNoiseStrategy,
    )
    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionStepMixin,
    )

    step = NoiseToNoiseStrategy._compute_losses_impl
    assert step is ReconstructionStepMixin._compute_losses_impl
    src = inspect.getsource(step)
    assert 'kwargs.get("iteration"' in src
    assert 'kwargs.get("step"' not in src


# ---------------------------------------------------------------------------
# standard_strategy
# ---------------------------------------------------------------------------


def test_standard_strategy_requires_env() -> None:
    from spectramr.infrastructure.training.strategies.standard_strategy import (
        StandardTrainingStrategy,
    )

    criterion = torch.nn.MSELoss()
    # base raises TypeError on a None env; the previous env=None default could
    # only ever crash at construction.
    with pytest.raises(TypeError):
        StandardTrainingStrategy(criterion=criterion, env=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# base._resolve_legacy_batch (truthiness on tensor)
# ---------------------------------------------------------------------------


def test_resolve_legacy_batch_accepts_tensor_without_ambiguous_bool() -> None:
    from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

    multi = torch.zeros(2, 2)  # bool(multi) would raise "ambiguous"
    out = BaseTrainingStrategy._resolve_legacy_batch(None, {"batch": multi})
    assert out is multi

    out2 = BaseTrainingStrategy._resolve_legacy_batch(
        None, {"input_batch": multi}
    )
    assert out2 is multi
