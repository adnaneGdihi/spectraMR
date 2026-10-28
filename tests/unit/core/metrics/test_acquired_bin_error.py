"""Acquired / unacquired k-space bin error: the hard-DC noise floor beside the model's work."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.core.metrics.acquired_bin_error import (
    ACQUIRED_NAME,
    BIN_METRICS,
    UNACQUIRED_NAME,
    bin_errors,
)
from spectramr.core.metrics.computer import ValidationMetricsComputer
from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.registry import MetricsRegistry
from spectramr.core.metrics.types import MetricSpec, ValidationMetricsConfig

pytestmark = pytest.mark.unit

SIZE = 16
COILS = 2


def _target(zero_cols: int = 0) -> torch.Tensor:
    """Complex ``[1, COILS, H, W]`` k-space; the last ``zero_cols`` columns never acquired."""
    gen = torch.Generator().manual_seed(0)
    k = torch.complex(
        torch.randn(1, COILS, SIZE, SIZE, generator=gen),
        torch.randn(1, COILS, SIZE, SIZE, generator=gen),
    )
    if zero_cols:
        k[..., SIZE - zero_cols :] = 0
    return k


def _mask(zero_cols: int = 0) -> torch.Tensor:
    """Every other column acquired, never on the padded columns."""
    mask = torch.zeros(1, 1, SIZE, SIZE)
    mask[..., ::2] = 1.0
    if zero_cols:
        mask[..., SIZE - zero_cols :] = 0.0
    return mask


def _score(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    ctx = MetricContext(mask=mask)
    return {name: cls()(pred, target, context=ctx) for name, cls in BIN_METRICS.items()}


def test_both_metrics_are_registered_lower_is_better_and_need_the_mask() -> None:
    for name in (ACQUIRED_NAME, UNACQUIRED_NAME):
        assert MetricsRegistry.is_registered(name)
        assert BIN_METRICS[name]().higher_is_better is False
        assert MetricsRegistry.needs_context(name)


def test_hard_dc_output_reports_the_noise_floor_on_acquired_and_zero_on_unacquired() -> None:
    """Acquired bins = the noisy input, unacquired = perfect: acquired error is the noise level."""
    target = _target()
    mask = _mask()
    noise = 0.25 * torch.complex(torch.randn_like(target.real), torch.randn_like(target.real))
    noisy_input = target + noise
    pred = torch.where(mask > 0, noisy_input, target)
    got = _score(pred, target, mask)
    sel = (mask > 0).expand_as(target)
    expected = float((noise[sel].abs().pow(2).sum() / target[sel].abs().pow(2).sum()).sqrt())
    assert got[ACQUIRED_NAME] == pytest.approx(expected, rel=1e-5)
    assert got[ACQUIRED_NAME] > 0.1
    assert got[UNACQUIRED_NAME] == pytest.approx(0.0, abs=1e-6)


def test_a_perfect_output_scores_zero_on_both() -> None:
    target = _target()
    got = _score(target.clone(), target, _mask())
    assert got[ACQUIRED_NAME] == pytest.approx(0.0, abs=1e-6)
    assert got[UNACQUIRED_NAME] == pytest.approx(0.0, abs=1e-6)


def test_the_zero_filled_output_scores_one_on_the_unacquired_bins() -> None:
    target = _target()
    mask = _mask()
    got = _score(target * mask, target, mask)
    assert got[ACQUIRED_NAME] == pytest.approx(0.0, abs=1e-6)
    assert got[UNACQUIRED_NAME] == pytest.approx(1.0, abs=1e-5)


def test_a_call_without_a_mask_is_nan_like_its_siblings() -> None:
    target = _target()
    for cls in BIN_METRICS.values():
        assert math.isnan(cls()(target, target))
        assert math.isnan(cls()(target, target, context=MetricContext()))


def test_padded_zero_lines_do_not_move_either_value() -> None:
    """The same acquirable content, with and without never-acquired columns appended."""
    base_target = _target()[..., : SIZE - 4]
    base_mask = _mask()[..., : SIZE - 4]
    noise = 0.3 * torch.complex(
        torch.randn_like(base_target.real), torch.randn_like(base_target.real)
    )
    pred = torch.where(base_mask > 0, base_target + noise, base_target * 0.5)

    def padded(x: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
        pad = torch.full((*x.shape[:-1], 4), fill, dtype=x.dtype)
        return torch.cat([x, pad], dim=-1)

    plain = bin_errors(pred, base_target, base_mask)
    # The padded columns are zero in the target and the mask; a prediction that is
    # non-zero there must not count as error either.
    with_padding = bin_errors(padded(pred, 5.0), padded(base_target), padded(base_mask))
    assert with_padding == pytest.approx(plain, rel=1e-6)
    assert plain[0] > 0 and plain[1] > 0


def test_no_target_energy_on_the_set_is_zero_not_nan() -> None:
    target = _target()
    full = torch.ones(1, 1, SIZE, SIZE)
    got = _score(target * 2, target, full)
    assert got[ACQUIRED_NAME] == pytest.approx(1.0, abs=1e-5)
    assert got[UNACQUIRED_NAME] == 0.0


def test_real_stacked_kspace_is_read_as_interleaved_complex() -> None:
    target = _target()
    pair = torch.view_as_real(target).permute(0, 1, 4, 2, 3).reshape(1, 2 * COILS, SIZE, SIZE)
    got = _score(pair, pair, _mask())
    assert got[ACQUIRED_NAME] == pytest.approx(0.0, abs=1e-6)


def test_the_validation_computer_reaches_both_metrics_with_a_context() -> None:
    """Through the real computer, the way the diffusion seam calls it."""
    computer = ValidationMetricsComputer(
        config=ValidationMetricsConfig(
            metrics=[MetricSpec.from_name(n) for n in (ACQUIRED_NAME, UNACQUIRED_NAME)]
        )
    )
    target = _target()
    mask = _mask()
    pred = torch.where(mask > 0, target + 0.1, target)
    out = computer.compute(pred, target, context=MetricContext(mask=mask))
    assert out[ACQUIRED_NAME] > 0
    assert out[UNACQUIRED_NAME] == pytest.approx(0.0, abs=1e-6)
