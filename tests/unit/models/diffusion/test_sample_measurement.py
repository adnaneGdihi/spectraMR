"""Contract tests for the non-Cartesian measurement carrier."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.diffusion.sample_measurement import SampleMeasurement


class _Projector:
    """Records what it was asked to project, so a test can watch the delegation."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def forward_project(self, image: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
        self.calls.append((tuple(image.shape), tuple(trajectory.shape)))
        return torch.zeros(image.shape[0], image.shape[1], trajectory.shape[-1])


def _measurement(batch: int = 2, coils: int = 3, n: int = 16) -> SampleMeasurement:
    return SampleMeasurement(
        samples=torch.randn(batch, coils, n, dtype=torch.complex64),
        mask=torch.ones(batch, n),
        trajectory=torch.zeros(2, n),
        projector=_Projector(),
    )


def test_it_projects_through_its_own_operator() -> None:
    """The measurement owns the operator, so a consumer never rebuilds one."""
    m = _measurement()
    m.project(torch.randn(2, 3, 8, 8, dtype=torch.complex64))
    assert m.projector.calls == [((2, 3, 8, 8), (2, 16))]


def test_a_mask_that_does_not_cover_the_samples_raises() -> None:
    """A mask of the wrong length would silently score a spoke subset."""
    with pytest.raises(ValueError, match="mask covers"):
        SampleMeasurement(
            samples=torch.randn(1, 1, 16, dtype=torch.complex64),
            mask=torch.ones(1, 8),
            trajectory=torch.zeros(2, 16),
            projector=_Projector(),
        )


def test_samples_must_carry_a_coil_axis() -> None:
    """``[B, N]`` samples would broadcast against a ``[B, C, N]`` prediction."""
    with pytest.raises(ValueError, match=r"samples must be \[B, C, N\]"):
        SampleMeasurement(
            samples=torch.randn(1, 16, dtype=torch.complex64),
            mask=torch.ones(1, 16),
            trajectory=torch.zeros(2, 16),
            projector=_Projector(),
        )


def test_it_is_frozen() -> None:
    """A consumer must not be able to edit the measurement it is scored against."""
    with pytest.raises(Exception, match=r"(?i)frozen|cannot assign"):
        _measurement().samples = torch.zeros(1)


def test_to_moves_the_tensors_and_keeps_the_operator() -> None:
    """``.to`` is a device hop, not a rebuild: the operator survives identically."""
    m = _measurement()
    moved = m.to("cpu")
    assert moved.projector is m.projector
    assert moved.samples.device.type == "cpu"
