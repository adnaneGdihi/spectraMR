"""The test pass refuses input that is not degraded the way the arm was trained."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.data.transforms.tio_physics import REALISED_ACCELERATION_KEY
from spectramr.infrastructure.reporting.test_split.degradation_guard import (
    DegradationGuard,
    declared_acceleration,
    realised_rates,
)

pytestmark = pytest.mark.unit


def _block(base, no_acc=False):
    return SimpleNamespace(
        undersampling=SimpleNamespace(base_acceleration=base, declares_no_acceleration=no_acc)
    )


def _xy():
    return torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8)


def test_declared_acceleration_reads_the_undersampling_block() -> None:
    assert declared_acceleration(_block(4.0)) == 4.0
    assert declared_acceleration(_block(1.0, no_acc=True)) is None
    assert declared_acceleration(SimpleNamespace(undersampling=None)) is None


def test_the_loader_stamp_owns_the_rate_and_the_mask_is_only_a_fallback() -> None:
    mask = torch.zeros(2, 1, 4, 4)
    mask[..., :1] = 1
    rates, stamped = realised_rates(
        {REALISED_ACCELERATION_KEY: torch.tensor([6.6, 6.6]), "mask": mask}
    )
    assert stamped and rates.tolist() == pytest.approx([6.6, 6.6])
    rates, stamped = realised_rates({"mask": mask})
    assert not stamped and rates.tolist() == [4.0, 4.0]
    assert realised_rates({}) == (None, False)


def test_input_equal_to_target_is_poisoned_whatever_the_arm_declares() -> None:
    x = torch.rand(1, 1, 8, 8)
    assert "equals the target" in DegradationGuard(None).verdict(x, x.clone(), None)


def test_a_stamped_rate_of_one_is_poisoned_without_any_declaration() -> None:
    """The sweep arms declare R in a transform, not an undersampling block."""
    verdict = DegradationGuard(None).verdict(*_xy(), torch.ones(1), stamped=True)
    assert "fully sampled" in verdict


def test_a_full_mask_on_an_r4_arm_is_poisoned() -> None:
    assert "R=4" in DegradationGuard(4.0).verdict(*_xy(), torch.ones(1))


def test_radial_r16_with_calibration_lines_is_not_flagged() -> None:
    """Realised R counts the ACS square, so it sits far below the declared 16 by design."""
    acs = 0.094
    realised = 1.0 / (acs + (1 - acs) / 16)
    assert DegradationGuard(16.0).verdict(*_xy(), torch.tensor([realised]), stamped=True) is None


def test_a_mask_derived_rate_without_a_declaration_is_not_judged() -> None:
    assert DegradationGuard(None).verdict(*_xy(), torch.ones(1)) is None
