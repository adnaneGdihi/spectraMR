"""KSpaceGeometryGain: geometry planes as a per-bin gain, not an additive k-space term."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.kspace_attention_kernels import crop_to_level_grid
from spectramr.models.blocks.kspace_geometry_gain import KSpaceGeometryGain

pytestmark = pytest.mark.unit


def _randomised(gain: KSpaceGeometryGain) -> KSpaceGeometryGain:
    torch.manual_seed(0)
    for p in gain.parameters():
        p.data.normal_(0, 0.5)
    return gain


def _planes(n: int = 2, k: int = 3, size: int = 16) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.rand(n, k, size, size) + 0.5


def test_a_fresh_gain_is_the_identity() -> None:
    gain = KSpaceGeometryGain(3, (4,))
    x = torch.randn(2, 8, 16, 16)
    torch.testing.assert_close(gain(x, _planes(), 0), x)


def test_zero_features_stay_zero_whatever_the_planes() -> None:
    """The defect the input-channel planes caused: a non-zero output from zero k-space."""
    gain = _randomised(KSpaceGeometryGain(3, (4,)))
    out = gain(torch.zeros(2, 8, 16, 16), _planes(), 0)
    assert out.abs().max() == 0


def test_the_gain_commutes_with_a_global_phase() -> None:
    gain = _randomised(KSpaceGeometryGain(3, (4,)))
    z = torch.randn(2, 4, 16, 16, dtype=torch.complex64)
    rot = torch.exp(torch.tensor(0.9j)).to(torch.complex64)
    torch.testing.assert_close(gain(z * rot, _planes(), 0), gain(z, _planes(), 0) * rot)


def test_interleaved_and_complex_layouts_agree() -> None:
    gain = _randomised(KSpaceGeometryGain(3, (4,)))
    z = torch.randn(2, 4, 16, 16, dtype=torch.complex64)
    interleaved = torch.view_as_real(z).permute(0, 1, 4, 2, 3).reshape(2, 8, 16, 16)
    out = gain(interleaved, _planes(), 0).reshape(2, 4, 2, 16, 16).permute(0, 1, 3, 4, 2)
    torch.testing.assert_close(torch.view_as_complex(out.contiguous()), gain(z, _planes(), 0))


def test_a_coarser_stage_reads_the_centre_crop_of_the_planes() -> None:
    gain = _randomised(KSpaceGeometryGain(3, (4, 4)))
    planes = _planes()
    x = torch.randn(2, 8, 8, 8)
    expected = gain(x, crop_to_level_grid(planes, 8, 8), 1)
    torch.testing.assert_close(gain(x, planes, 1), expected)


@pytest.mark.parametrize(
    ("features", "planes", "match"),
    [
        (torch.randn(2, 6, 16, 16), _planes(), "built for 4 complex channels"),
        (torch.randn(2, 8, 16, 16), _planes(k=2), r"\[N, 3, H, W\] planes"),
        (torch.randn(3, 8, 16, 16), _planes(), "plane batch"),
        (torch.randn(2, 8, 32, 32), _planes(), "larger than the plane"),
    ],
)
def test_a_mismatched_input_raises(features, planes, match) -> None:
    with pytest.raises(ValueError, match=match):
        KSpaceGeometryGain(3, (4,))(features, planes, 0)


@pytest.mark.parametrize(("k", "stages"), [(0, (4,)), (3, ())])
def test_an_empty_build_raises(k, stages) -> None:
    with pytest.raises(ValueError, match="num_planes > 0"):
        KSpaceGeometryGain(k, stages)
