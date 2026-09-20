"""Sample-domain data consistency for acquisitions that never touch a grid.

The Cartesian layers enforce fidelity at masked ``fft2c`` bins. This one enforces
it at off-grid samples, so the two things it must get right are the mask layout
(indexed by readout, not by coil) and the step scale (the NUFFT is not a
unit-gain adjoint pair). Each test plants one way of getting those wrong.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.non_cartesian_dc import (
    NC_DC_MODES,
    NonCartesianDataConsistency,
    align_sample_mask,
)
from spectramr.infrastructure.physics.trajectories import get_trajectory

IM = (32, 32)


@pytest.fixture(scope="module")
def spiral():
    return get_trajectory("spiral", im_size=IM)


def _layer(**kw) -> NonCartesianDataConsistency:
    opts = {"im_size": IM, "learn_step": False}
    opts.update(kw)
    return NonCartesianDataConsistency(**opts)


# ---------------------------------------------------------------------------
# align_sample_mask: the readout axis, not the coil axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda n: None, id="none-means-all-acquired"),
        pytest.param(lambda n: torch.ones(n), id="[N]"),
        pytest.param(lambda n: torch.ones(2, n), id="[B,N]"),
        pytest.param(lambda n: torch.ones(2, 1, n), id="[B,1,N]"),
        pytest.param(lambda n: torch.ones(n) > 0.5, id="bool"),
    ],
)
def test_accepted_sample_mask_layouts(build) -> None:
    assert align_sample_mask(build(64), 2, 64).shape == (2, 1, 64)


def test_a_cartesian_grid_mask_is_refused() -> None:
    """The planted confusion: a [B, 1, H, W] mask means the caller still thinks
    this acquisition is Cartesian, and silently broadcasting it would enforce
    consistency at frequencies the scanner never visited."""
    with pytest.raises(ValueError, match="indexed by readout"):
        align_sample_mask(torch.ones(2, 1, 32, 32), 2, 64)


def test_a_mask_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(ValueError, match="readouts but the trajectory"):
        align_sample_mask(torch.ones(48), 2, 64)


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_an_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown non-Cartesian DC mode"):
        _layer(mode="hard")


@pytest.mark.parametrize("lam", [-0.1, 1.5])
def test_lambda_outside_the_unit_interval_raises(lam: float) -> None:
    with pytest.raises(ValueError, match=r"lambda_dc must be in \[0, 1\]"):
        _layer(lambda_dc=lam)


def test_both_modes_are_reachable() -> None:
    assert set(NC_DC_MODES) == {"gradient", "replace"}
    for mode in sorted(NC_DC_MODES):
        assert _layer(mode=mode).mode == mode


# ---------------------------------------------------------------------------
# The forward must never no-op
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["measured_samples", "trajectory"])
def test_a_missing_measurement_raises_rather_than_passing_through(spiral, missing) -> None:
    """Returning the input would report a data-consistent reconstruction that
    never saw its measurement -- the shape the Cartesian AdaptiveDataConsistency
    still has at ``data_consistency.py:257``."""
    traj, _ = spiral
    n = traj.shape[-1]
    args = {
        "measured_samples": torch.randn(1, 1, n, dtype=torch.complex64),
        "trajectory": traj,
    }
    args[missing] = None
    with pytest.raises(ValueError, match="reached without"):
        _layer()(torch.randn(1, 1, *IM, dtype=torch.complex64), **args)


def test_a_real_image_raises_instead_of_being_coerced(spiral) -> None:
    traj, _ = spiral
    with pytest.raises(ValueError, match="needs a complex image"):
        _layer()(
            torch.randn(1, 1, *IM),
            torch.randn(1, 1, traj.shape[-1], dtype=torch.complex64),
            traj,
        )


@pytest.mark.parametrize("mode", ["gradient", "replace"])
def test_the_output_keeps_the_input_layout(spiral, mode: str) -> None:
    traj, dcf = spiral
    n = traj.shape[-1]
    out = _layer(mode=mode)(
        torch.randn(2, 1, *IM, dtype=torch.complex64),
        torch.randn(2, 1, n, dtype=torch.complex64),
        traj,
        sample_mask=torch.ones(n),
        dcf=dcf,
    )
    assert out.shape == (2, 1, *IM)
    assert torch.is_complex(out)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# The step is normalised, which is what makes it portable
# ---------------------------------------------------------------------------


def test_the_gradient_step_reduces_the_sample_domain_residual(spiral) -> None:
    """With a consistent measurement, repeated DC must descend, not diverge.

    The planted violation is an un-normalised step: before the Gram norm was
    divided out, ``step_size=1.0`` grew the residual by twenty orders of
    magnitude in six iterations on this exact trajectory.
    """
    traj, dcf = spiral
    layer = _layer(mode="gradient", step_size=1.0)
    truth = torch.randn(1, 1, *IM, dtype=torch.complex64)
    measured = layer.operator.forward_project(truth, traj)

    x = torch.zeros_like(truth)
    residuals = []
    for _ in range(6):
        x = layer(x, measured, traj, sample_mask=torch.ones(traj.shape[-1]), dcf=dcf)
        residuals.append(float((layer.operator.forward_project(x, traj) - measured).abs().mean()))

    assert residuals == sorted(residuals, reverse=True), residuals
    assert residuals[-1] < 0.5 * residuals[0]


def test_the_gram_norm_is_measured_per_trajectory_not_assumed(spiral) -> None:
    """A constant would be wrong on the next matrix size: measured 9.0e3 at
    32x32 and 2.3e4 at 64x64 for the same spiral family."""
    traj, dcf = spiral
    small = _layer(mode="gradient")
    small(
        torch.randn(1, 1, *IM, dtype=torch.complex64),
        torch.randn(1, 1, traj.shape[-1], dtype=torch.complex64),
        traj,
        dcf=dcf,
    )
    assert small._gram_norm, "no Gram norm was estimated"
    assert next(iter(small._gram_norm.values())) > 1.0


def test_the_gram_norm_is_cached_not_recomputed_per_step(spiral) -> None:
    """The power iteration does a host sync; per-step it would violate
    non-negotiable 9."""
    traj, dcf = spiral
    layer = _layer(mode="gradient")
    args = (
        torch.randn(1, 1, *IM, dtype=torch.complex64),
        torch.randn(1, 1, traj.shape[-1], dtype=torch.complex64),
        traj,
    )
    layer(*args, dcf=dcf)
    first = dict(layer._gram_norm)
    layer(*args, dcf=dcf)
    assert layer._gram_norm == first


def test_a_learned_step_is_a_parameter_and_a_fixed_one_is_not(spiral) -> None:
    assert any(p.requires_grad for p in _layer(learn_step=True).parameters())
    layer = _layer(learn_step=False)
    assert layer.step_size is None
    assert not [p for p in layer.parameters() if p.requires_grad]


def test_an_unmasked_sample_leaves_the_image_alone(spiral) -> None:
    """Gradient mode must touch the image only where a measurement disagrees."""
    traj, dcf = spiral
    layer = _layer(mode="gradient", step_size=1.0)
    x = torch.randn(1, 1, *IM, dtype=torch.complex64)
    zeros = torch.zeros(traj.shape[-1])
    out = layer(
        x,
        torch.randn(1, 1, traj.shape[-1], dtype=torch.complex64),
        traj,
        sample_mask=zeros,
        dcf=dcf,
    )
    assert torch.allclose(out, x, atol=1e-5)


def test_a_dcf_of_the_wrong_length_raises(spiral) -> None:
    traj, _ = spiral
    with pytest.raises(ValueError, match="dcf covers"):
        _layer()(
            torch.randn(1, 1, *IM, dtype=torch.complex64),
            torch.randn(1, 1, traj.shape[-1], dtype=torch.complex64),
            traj,
            dcf=torch.ones(7),
        )


# ---------------------------------------------------------------------------
# The sample-domain path: no transform, so it must be EXACT
#
# The Cartesian layers' ``is_kspace_domain`` saves a round trip that is free and
# lossless. This one saves a round trip that is 54x an FFT and corrupts the
# samples by 85 %, because ``A A^H W`` is not the identity -- so these tests pin
# exactness, not just shape.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["gradient", "replace"])
def test_sample_domain_dc_reproduces_the_measurement_exactly(spiral, mode: str) -> None:
    """Going through the image domain cannot do this: the adjoint would rewrite
    the measurements on the way through."""
    traj, dcf = spiral
    n = traj.shape[-1]
    measured = torch.randn(1, 4, n, dtype=torch.complex64)
    predicted = torch.randn(1, 4, n, dtype=torch.complex64)
    mask = torch.zeros(n)
    mask[: n // 2] = 1

    out = _layer(mode=mode, lambda_dc=1.0)(
        predicted, measured, traj, sample_mask=mask, dcf=dcf, is_sample_domain=True
    )

    assert out.shape == predicted.shape
    assert torch.allclose(out[..., : n // 2], measured[..., : n // 2], atol=1e-6)


@pytest.mark.parametrize("mode", ["gradient", "replace"])
def test_sample_domain_dc_leaves_unmeasured_readouts_alone(spiral, mode: str) -> None:
    """Whatever filled the gaps is the network's business, not DC's."""
    traj, dcf = spiral
    n = traj.shape[-1]
    predicted = torch.randn(1, 4, n, dtype=torch.complex64)
    mask = torch.zeros(n)
    mask[: n // 2] = 1

    out = _layer(mode=mode, lambda_dc=1.0)(
        predicted,
        torch.randn(1, 4, n, dtype=torch.complex64),
        traj,
        sample_mask=mask,
        dcf=dcf,
        is_sample_domain=True,
    )
    assert torch.allclose(out[..., n // 2 :], predicted[..., n // 2 :], atol=1e-6)


def test_sample_domain_dc_does_no_transform_at_all(spiral) -> None:
    """The planted regression: if either NUFFT ran, the Gram norm cache would be
    populated and the measurements would no longer be exact."""
    traj, dcf = spiral
    n = traj.shape[-1]
    layer = _layer(mode="gradient")
    layer(
        torch.randn(1, 4, n, dtype=torch.complex64),
        torch.randn(1, 4, n, dtype=torch.complex64),
        traj,
        dcf=dcf,
        is_sample_domain=True,
    )
    assert layer._gram_norm == {}, "a NUFFT ran on the transform-free path"


def test_an_image_with_the_sample_flag_set_raises(spiral) -> None:
    """Silently treating [B, C, H, W] as samples would index the wrong axis."""
    traj, _ = spiral
    with pytest.raises(ValueError, match="expects a \\[B, C, N\\] prediction"):
        _layer()(
            torch.randn(1, 4, *IM, dtype=torch.complex64),
            torch.randn(1, 4, traj.shape[-1], dtype=torch.complex64),
            traj,
            is_sample_domain=True,
        )
