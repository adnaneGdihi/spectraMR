"""Tests for the calibrated-gain Bloch loss (exp 32, gain-calibrated maps)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.calibrated_gain_bloch_loss import CalibratedGainBlochLoss

_RENDER = torch.ones(1, 1, 4, 4)
_MASK = torch.ones(1, 1, 4, 4, dtype=torch.bool)


def test_calibrated_mode_penalises_a_wrong_scale() -> None:
    loss = CalibratedGainBlochLoss("calibrated")
    ctx = {"render": _RENDER, "gain": torch.tensor(2.0), "mask": _MASK}
    assert loss(None, 2 * _RENDER, context=ctx) < 1e-12
    assert loss(None, 3 * _RENDER, context=ctx) > 0.5


def test_projected_mode_absorbs_any_scale() -> None:
    loss = CalibratedGainBlochLoss("projected")
    assert loss(None, 3 * _RENDER, context={"render": _RENDER, "gain": None, "mask": _MASK}) < 1e-12


def test_only_masked_voxels_count() -> None:
    loss = CalibratedGainBlochLoss("calibrated")
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask[..., 0, 0] = True
    target = torch.full((1, 1, 4, 4), 9.0)
    target[..., 0, 0] = 1.0
    assert (
        loss(None, target, context={"render": _RENDER, "gain": torch.tensor(1.0), "mask": mask})
        < 1e-12
    )


def test_calibrated_mode_without_a_gain_raises() -> None:
    with pytest.raises(ValueError, match="gain"):
        CalibratedGainBlochLoss("calibrated")(
            None, _RENDER, context={"render": _RENDER, "gain": None, "mask": _MASK}
        )


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="gain_mode"):
        CalibratedGainBlochLoss("bounded")


def test_missing_context_raises() -> None:
    with pytest.raises(ValueError, match="context"):
        CalibratedGainBlochLoss("projected")(None, _RENDER)
