"""Coil self-consistency projection (Algorithm E): calibration, the operator and its guard."""

from __future__ import annotations

import pytest
import torch

import spectramr.models.diffusion.reverse_spirit as spirit
from spectramr.models.diffusion.reverse_domain import linear_kspace
from spectramr.models.diffusion.reverse_integrity import ReverseIntegrityError
from tests.unit.models.diffusion import _reverse_phantom as P

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def setting():
    proc = P.process(8.0)
    target = P.kspace()
    mask = P.mask(proc)
    return proc, target, target * mask, mask


def _sampler(proc, target):
    return P.sampler(
        P.Oracle(target, proc), "replace_freeze_dc", projection="spirit", integrity_check="raise"
    )


def _error(x, target, free):
    return float((linear_kspace(x - target, log_scaled=False).abs() * free).pow(2).sum().sqrt())


def test_calibration_takes_the_longest_run_of_measured_rows() -> None:
    rows = torch.tensor(
        [[0, 1, 1, 0, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0, 0, 1, 1]], dtype=torch.bool
    )
    assert spirit.calibration_rows(rows) == [(4, 8), (0, 3)]


def test_calibration_below_the_kernel_height_is_refused() -> None:
    rows = torch.tensor([[0, 1, 1, 0, 1, 0]], dtype=torch.bool)
    with pytest.raises(ValueError, match="contiguous"):
        spirit.calibration_rows(rows)


def test_the_kernel_never_reads_the_target_coils_own_centre_tap(setting) -> None:
    _, target, _, _ = setting
    core = linear_kspace(target, log_scaled=False)[0, :, 28:36]
    kernel = spirit.calibrate_kernel(core)
    coils = torch.arange(core.shape[0])
    assert kernel.shape == (core.shape[0], core.shape[0], *spirit.KERNEL)
    assert torch.equal(kernel[coils, coils, 1, 2], torch.zeros_like(kernel[coils, coils, 1, 2]))


def test_the_residual_adjoint_is_exact() -> None:
    gen = torch.Generator().manual_seed(0)
    kernels = torch.complex(*torch.randn(2, 2, 3, 3, *spirit.KERNEL, generator=gen))
    forward, adjoint = spirit._residual_op(kernels, 2)
    k = torch.complex(*torch.randn(2, 2, 3, 16, 16, generator=gen))
    r = torch.complex(*torch.randn(2, 2, 3, 16, 16, generator=gen))
    lhs = (forward(k) * r.conj()).sum()
    rhs = (k * adjoint(r).conj()).sum()
    assert torch.allclose(lhs, rhs, rtol=1e-4)


def test_the_projection_fills_toward_the_target_and_holds_the_measured_bins(setting) -> None:
    proc, target, meas, mask = setting
    project = spirit.build_spirit(_sampler(proc, target), meas, mask)
    out = project(meas, mask, where="test")
    free = (mask == 0).float()
    assert _error(out, target, free) < 0.9 * _error(meas, target, free)
    held = mask.bool().expand_as(out)
    assert torch.equal(out[held], meas[held])


def test_a_step_past_the_lipschitz_bound_is_caught(setting, monkeypatch) -> None:
    """PLANTED: ten times the admissible step makes the residual rise, and the check raises."""
    proc, target, meas, mask = setting
    monkeypatch.setattr(spirit, "_LIPSCHITZ_MARGIN", 0.1)
    project = spirit.build_spirit(_sampler(proc, target), meas, mask)
    with pytest.raises(ReverseIntegrityError) as info:
        project(meas, mask, where="test")
    assert info.value.invariant == "projection_residual"


def test_too_few_measured_rows_to_calibrate_is_refused(setting) -> None:
    proc, target, _, _ = setting
    mask = torch.zeros(target.shape[0], 1, *target.shape[-2:])
    mask[..., [20, 32, 44], :] = 1.0
    with pytest.raises(ValueError, match="contiguous"):
        spirit.build_spirit(_sampler(proc, target), target * mask, mask)
