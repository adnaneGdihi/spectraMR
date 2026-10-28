"""The undersampling process removes lines along the declared phase-encode axis (#2160).

Non-square planes throughout: on a square plane a swapped H/W is invisible.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.line_axis_gate import LineAxisMismatchError
from spectramr.models.diffusion.kspace_process import (
    KSpaceUndersamplingProcess,
    resolve_undersampling_kwargs,
)

ARM = {
    "acceleration_type": "density_nested",
    "base_acceleration": 1.0,
    "max_acceleration": 8.0,
    "center_fraction": 0.08,
    "min_center_fraction": 0.02,
    "acceleration_range": [1.0, 2.0, 4.0, 8.0],
    "mask_direction": "phase",
    "schedule_type": "step",
    "mask_seed": 42,
    "enforce_nested": True,
    "enable_dynamic_mask": True,
}


def _process(axis: int | None, **over: object) -> KSpaceUndersamplingProcess:
    arm = {**ARM, **over}
    return KSpaceUndersamplingProcess(
        num_timesteps=4, **resolve_undersampling_kwargs(arm, phase_encode_axis=axis)
    )


def _whole_lines(mask: torch.Tensor) -> tuple[int, int, int, int]:
    """(whole rows, touched rows, whole columns, touched columns) of a ``[H, W]`` mask."""
    return (
        int(mask.all(-1).sum()),
        int(mask.any(-1).sum()),
        int(mask.all(-2).sum()),
        int(mask.any(-2).sum()),
    )


@pytest.mark.parametrize(("h", "w"), [(48, 80), (40, 64)])
@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
def test_axis_minus_one_removes_columns(h: int, w: int, training: bool) -> None:
    """M4Raw / fastMRI brain: the mask keeps whole columns and touches every row."""
    process = _process(-1)
    process.train(training)
    _, masks = process.q_sample(torch.randn(2, 4, h, w), torch.tensor([3, 3]))
    whole_rows, touched_rows, whole_cols, touched_cols = _whole_lines(masks[0, 0].bool())
    assert whole_rows == 0 and touched_rows == h
    assert 0 < whole_cols == touched_cols < w
    assert process.line_axis == "x"


@pytest.mark.parametrize(("h", "w"), [(48, 80), (40, 64)])
def test_axis_minus_two_removes_rows(h: int, w: int) -> None:
    process = _process(-2)
    _, masks = process.q_sample(torch.randn(2, 4, h, w), torch.tensor([3, 3]))
    whole_rows, touched_rows, whole_cols, touched_cols = _whole_lines(masks[0, 0].bool())
    assert whole_cols == 0 and touched_cols == w
    assert 0 < whole_rows == touched_rows < h
    assert process.line_axis == "y"


@pytest.mark.parametrize("family", ["density_nested", "equispaced", "uniform_cartesian"])
@pytest.mark.parametrize("axis", [-1, -2])
def test_the_stored_line_axis_is_the_one_the_mask_realises(family: str, axis: int) -> None:
    """``line_axis`` (read by the decay envelope) and the realised mask (read by the
    gate) are two readers of one fact; a family that stores an axis it ignores fails."""
    from spectramr.models.diffusion.reveal_attribution import resolve_line_axis

    process = _process(axis, acceleration_type=family, enforce_nested=False)
    assert resolve_line_axis(process.mask_at(3, (48, 80))) == process.line_axis


def test_readout_is_the_other_axis() -> None:
    assert _process(-1, mask_direction="readout").line_axis == "y"


def test_phase_without_a_declared_axis_raises_at_resolution() -> None:
    """PLANTED: the arm shape every M4Raw/fastMRI arm had before #2160."""
    with pytest.raises(ValueError, match="measure_phase_encode_axis"):
        resolve_undersampling_kwargs(ARM)


def test_the_axis_declared_twice_must_agree() -> None:
    with pytest.raises(ValueError, match="declared twice"):
        resolve_undersampling_kwargs(ARM, {"phase_encode_axis": -2}, phase_encode_axis=-1)
    agreed = resolve_undersampling_kwargs(ARM, {"phase_encode_axis": -1}, phase_encode_axis=-1)
    assert agreed["phase_encode_axis"] == -1


def test_the_generator_spelling_reaches_the_process() -> None:
    """The generator passes the axis inside its kwargs (``overrides``)."""
    resolved = resolve_undersampling_kwargs(ARM, {"phase_encode_axis": -1})
    assert KSpaceUndersamplingProcess(num_timesteps=4, **resolved).line_axis == "x"


class TestFirstBatchGate:
    """``q_sample`` checks its first batch per mode against the mask's line axis."""

    @staticmethod
    def _partial_fov(h: int, w: int) -> torch.Tensor:
        x = torch.randn(2, 4, h, w)
        x[..., :9] = 0  # never-acquired phase-encode columns
        return x

    @pytest.mark.parametrize(("h", "w"), [(48, 80), (40, 64)])
    def test_a_row_mask_on_column_partial_data_raises(self, h: int, w: int) -> None:
        """PLANTED: the pre-#2160 cohort -- partial phase FOV on dim -1, rows removed."""
        process = _process(-2)
        process.train(True)
        with pytest.raises(LineAxisMismatchError):
            process.q_sample(self._partial_fov(h, w), torch.tensor([3, 3]))

    def test_agreement_is_recorded_once_per_mode(self) -> None:
        process = _process(-1)
        process.train(True)
        process.q_sample(self._partial_fov(48, 80), torch.tensor([3, 3]))
        assert process.line_axis_gate["train"]["verdict"] == "agree"
        assert process.line_axis_gate["train"]["declared_phase_encode_axis"] == -1
        # A later batch is not re-read: zero-free data would otherwise say inconclusive.
        process.q_sample(torch.randn(2, 4, 48, 80), torch.tensor([3, 3]))
        assert process.line_axis_gate["train"]["verdict"] == "agree"
        assert "eval" not in process.line_axis_gate

    def test_zero_free_data_is_inconclusive(self) -> None:
        process = _process(-2)
        process.eval()
        process.q_sample(torch.randn(2, 4, 40, 64), torch.tensor([3, 3]))
        assert process.line_axis_gate["eval"]["verdict"] == "inconclusive"

    def test_a_five_dimensional_batch_is_gated_on_its_plane(self) -> None:
        """``q_sample`` takes ``[B, D, C, H, W]``; the plane is still the last two."""
        process = _process(-2)
        x = torch.randn(1, 3, 4, 40, 64)
        x[..., :9] = 0
        with pytest.raises(LineAxisMismatchError):
            process.q_sample(x, torch.tensor([3]))
