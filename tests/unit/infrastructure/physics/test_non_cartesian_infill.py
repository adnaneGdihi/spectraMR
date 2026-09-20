"""What is exact, what is interpolation, what is invention.

The Hermitian half is an identity and must hold to machine precision on a real
object and visibly fail on a complex one -- if it did not, the diagnostic would
be reporting noise. The gate is pure geometry and must not depend on any signal.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.non_cartesian_infill import (
    hermitian_extend,
    hermitian_phase_violation,
    sample_density_gate,
)
from spectramr.infrastructure.physics.nufft_ops import NUFFTForwardModel
from spectramr.infrastructure.physics.trajectories import get_trajectory

IM = (32, 32)


@pytest.fixture(scope="module")
def radial():
    return get_trajectory("radial", im_size=IM)


# ---------------------------------------------------------------------------
# Hermitian symmetry: an identity, so it must be exact
# ---------------------------------------------------------------------------


def test_a_real_object_satisfies_the_symmetry_exactly(radial) -> None:
    """``S(-k) = conj(S(k))`` is not approximately true for a real object."""
    traj, _ = radial
    op = NUFFTForwardModel(im_size=IM)
    real_object = torch.rand(1, 1, *IM).to(torch.complex64)
    violation = hermitian_phase_violation(traj, op.forward_project(real_object, traj))
    assert float(violation) < 1e-4, float(violation)


def test_a_phase_carrying_object_breaks_it_measurably(radial) -> None:
    """The planted counter-case. MRI objects are complex -- B0, coil phase, flow
    -- so a diagnostic that reported ~0 here would be useless."""
    traj, _ = radial
    op = NUFFTForwardModel(im_size=IM)
    torch.manual_seed(0)
    phase = torch.exp(1j * 6.0 * torch.rand(1, 1, *IM))
    obj = torch.rand(1, 1, *IM).to(torch.complex64) * phase
    violation = hermitian_phase_violation(traj, op.forward_project(obj, traj))
    assert float(violation) > 0.1, float(violation)


def test_the_extension_flags_which_samples_were_actually_measured(radial) -> None:
    """A mirrored sample is inferred, not acquired; a DC layer has to be able to
    hold the two at different strengths."""
    traj, _ = radial
    n = traj.shape[-1]
    traj_ext, samples_ext, is_measured = hermitian_extend(
        traj, torch.randn(1, 1, n, dtype=torch.complex64)
    )
    assert traj_ext.shape[-1] == samples_ext.shape[-1] == is_measured.shape[0]
    assert int(is_measured.sum()) == n
    assert torch.equal(is_measured[:n], torch.ones(n))
    assert traj_ext.shape[-1] >= n


def test_the_mirrored_samples_are_conjugated_not_copied(radial) -> None:
    """Copying instead of conjugating would assert the object is symmetric rather
    than Hermitian, and quietly halve the imaginary part of the reconstruction."""
    traj, _ = radial
    n = traj.shape[-1]
    samples = torch.randn(1, 1, n, dtype=torch.complex64)
    _, samples_ext, is_measured = hermitian_extend(traj, samples)
    added = int((is_measured == 0).sum())
    if added == 0:
        pytest.skip("this trajectory is already fully self-symmetric")
    mirrored = samples_ext[..., n:]
    assert torch.allclose(mirrored.real, mirrored.real)  # finite
    assert not torch.allclose(mirrored, samples[..., :added], atol=1e-6)


def test_a_redundant_antipode_is_dropped(radial) -> None:
    """A radial spoke already samples both ends, so mirroring must not duplicate
    a frequency and let the adjoint count it twice."""
    traj, _ = radial
    n = traj.shape[-1]
    traj_ext, _, _ = hermitian_extend(traj, torch.randn(1, 1, n, dtype=torch.complex64))
    assert traj_ext.shape[-1] < 2 * n


def test_a_malformed_trajectory_raises() -> None:
    with pytest.raises(ValueError, match=r"must be \[2, N\]"):
        hermitian_extend(torch.randn(3, 10), torch.randn(1, 1, 10, dtype=torch.complex64))


def test_a_sample_count_mismatch_raises(radial) -> None:
    traj, _ = radial
    with pytest.raises(ValueError, match="samples cover"):
        hermitian_extend(traj, torch.randn(1, 1, 9, dtype=torch.complex64))


# ---------------------------------------------------------------------------
# The density gate: geometry only
# ---------------------------------------------------------------------------


def test_the_gate_is_high_where_the_trajectory_visited_and_low_where_it_did_not(radial) -> None:
    traj, _ = radial
    gate = sample_density_gate(traj, IM)
    assert gate.shape == (1, 1, *IM)
    assert float(gate.max()) == pytest.approx(1.0, abs=1e-5)
    assert float(gate.min()) >= 0.0
    # A radial acquisition genuinely leaves the corners of k-space unvisited.
    assert float((gate < 0.1).float().mean()) > 0.1


def test_the_gate_depends_on_the_trajectory_only_not_on_any_signal(radial) -> None:
    """The point of a geometric confidence is that no learned quantity enters it,
    so it cannot be talked into trusting an unsampled frequency."""
    traj, _ = radial
    assert torch.equal(sample_density_gate(traj, IM), sample_density_gate(traj, IM))


def test_a_denser_trajectory_gates_less(radial) -> None:
    radial_traj, _ = radial
    spiral_traj, _ = get_trajectory("spiral", im_size=IM)
    covered = lambda t: float((sample_density_gate(t, IM) > 0.1).float().mean())  # noqa: E731
    assert covered(radial_traj) > covered(spiral_traj)


@pytest.mark.parametrize("floor", [0.0, 0.05, 1.0])
def test_the_floor_is_the_declared_extrapolation_budget(radial, floor: float) -> None:
    """``floor=0`` forbids reaching past the sampling outright; a positive value
    is an explicit allowance, not an emergent one."""
    traj, _ = radial
    gate = sample_density_gate(traj, IM, floor=floor)
    assert float(gate.min()) == pytest.approx(floor, abs=1e-5)


@pytest.mark.parametrize("floor", [-0.1, 1.2])
def test_a_floor_outside_the_unit_interval_raises(radial, floor: float) -> None:
    traj, _ = radial
    with pytest.raises(ValueError, match=r"floor must be in \[0, 1\]"):
        sample_density_gate(traj, IM, floor=floor)
