"""Hardware catalogue invariants for every (axis, mode), plus the shared line-shift plumbing."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    Realisation,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.catalogue.hardware_common import (
    line_taus,
    shift_lines,
)
from spectramr.infrastructure.physics.degradations.spec import ActivationGate
from spectramr.infrastructure.physics.degradations.volume.motion_schedules import (
    readout_line_shift,
)
from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T
from spectramr.infrastructure.physics.scanner_profile import Sequence, active_axes

M = AcquisitionMode
NAMES = ("b0_drift", "gradient_delay", "gradient_scaling_drift", "mechanical_vibration")
VOXEL = (5.0, 1.0, 2.0)
GEOMETRY = {
    M.SLICE_2D: None,
    M.MULTISLICE_2D: AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=VOXEL),
    M.VOLUME_3D: AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=VOXEL),
}
CASES = [(n, m) for n in NAMES for m in M]


def _x(mode: M, batch: int = 2, seed: int = 0) -> torch.Tensor:
    shape = (batch, 1, 16, 16) if mode is M.SLICE_2D else (batch, 1, 4, 16, 16)
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))


def _run(name: str, mode: M, x: torch.Tensor, theta: float, seed: int = 3, **kw) -> torch.Tensor:
    return apply_degradation(name, x, theta, seed=seed, geometry=GEOMETRY[mode], **kw)


@pytest.mark.parametrize("name", NAMES)
def test_declared_native_in_every_mode_with_no_requirement(name: str) -> None:
    spec = DEGRADATION_REGISTRY[name]
    assert spec.supported_modes == frozenset(M)
    assert set(spec.realisation.values()) == {Realisation.NATIVE}
    assert spec.requires == frozenset()
    assert spec.severity.params[0].at_theta_min == 0.0


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_theta_zero_is_the_identity(name: str, mode: M) -> None:
    x = _x(mode)
    torch.testing.assert_close(_run(name, mode, x, 0.0), x, rtol=0, atol=0)


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_effect_grows_with_theta(name: str, mode: M) -> None:
    x = _x(mode)
    err = [float((_run(name, mode, x, t) - x).abs().pow(2).mean()) for t in (0.0, 0.3, 0.6, 1.0)]
    assert err[0] == 0.0
    assert err[0] < err[1] < err[2] < err[3], err


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_batch_invariant_finite_shape_and_deterministic(name: str, mode: M) -> None:
    both = _x(mode)
    alone = both[:1]
    neighbour = torch.cat([alone, 10.0 * both[1:]])
    out = _run(name, mode, alone, 0.7)
    assert out.shape == alone.shape
    assert torch.isfinite(torch.view_as_real(out)).all()
    torch.testing.assert_close(_run(name, mode, neighbour, 0.7)[:1], out, rtol=0, atol=1e-5)
    torch.testing.assert_close(_run(name, mode, alone, 0.7), out, rtol=0, atol=0)


@pytest.mark.parametrize("name", NAMES)
def test_one_slice_stack_is_the_slice_2d_result(name: str) -> None:
    x = _x(M.SLICE_2D)
    stack = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=VOXEL)
    ref = apply_degradation(name, x, 0.6, seed=4)
    out = apply_degradation(
        name, x.unsqueeze(2), 0.6, seed=4, geometry=replace(stack, voxel_mm=None)
    )
    torch.testing.assert_close(out.squeeze(2), ref, rtol=0, atol=1e-6)


def test_a_slice_geometry_reaches_the_2d_kernel() -> None:
    x = _x(M.SLICE_2D)
    geom = AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=VOXEL)
    a = apply_degradation("mechanical_vibration", x, 1.0, 1, geometry=geom, axis="ro")
    b = apply_degradation("mechanical_vibration", x, 1.0, 1, axis="ro")
    assert (a - b).abs().max() > 1e-3  # 2 mm readout voxels halve the displacement in px


@pytest.mark.parametrize("name", NAMES)
def test_two_dimensional_kernel_rejects_a_volume(name) -> None:
    with pytest.raises(ValueError, match="expects"):
        DEGRADATION_REGISTRY[name].fn(_x(M.VOLUME_3D), 0.5)


def test_shift_lines_is_an_exact_fourier_shift_per_line() -> None:
    n = 16
    g = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=None)
    idx = torch.arange(n, dtype=torch.float32)
    wave = torch.exp(2j * torch.pi * 3 * idx / n)
    x = (wave[None, :] * wave[:, None]).expand(1, 1, 2, n, n).clone()
    d = torch.tensor([[[0.25] * n, [-0.5] * n]])
    out = shift_lines(x, g, pe=d, ro=2 * d)
    expect = x * torch.exp(-2j * torch.pi * 3 / n * (d + 2 * d))[..., None]
    torch.testing.assert_close(out, expect[:1].reshape(out.shape), rtol=0, atol=1e-5)
    torch.testing.assert_close(
        shift_lines(x, g, ro=d), readout_line_shift(x, d, g), rtol=0, atol=1e-6
    )
    with pytest.raises(ValueError, match="kz"):
        shift_lines(x, g, kz=d)


@pytest.mark.parametrize("mode", [M.MULTISLICE_2D, M.VOLUME_3D])
def test_line_taus_run_from_zero_to_one(mode: M) -> None:
    tau = line_taus(AcquisitionGeometry(mode=mode, voxel_mm=None), 4, 16, torch.device("cpu"))
    assert tau.shape == (4, 16)
    assert float(tau.min()) == 0.0
    assert float(tau.max()) == 1.0


def test_gradient_delay_is_gated_to_epi_not_required() -> None:
    spec = DEGRADATION_REGISTRY["gradient_delay"]
    assert spec.active_when == frozenset({ActivationGate.SEQUENCE_EPI})
    epi = replace(FASTMRI_3T, name="epi_3T", sequence=Sequence.EPI, esp_ms=0.5)
    assert "gradient_delay" not in active_axes(FASTMRI_3T, NAMES)
    assert "gradient_delay" in active_axes(epi, NAMES)
    apply_degradation("gradient_delay", _x(M.SLICE_2D), 0.5)  # never blocks a call


def test_vibration_takes_its_tr_from_the_profile_and_a_profile_conflict_raises() -> None:
    timed = replace(FASTMRI_3T, name="timed", tr_ms=12.5)
    x = _x(M.SLICE_2D)
    a = apply_degradation("mechanical_vibration", x, 1.0, 2, profile=timed, axis="ro")
    b = apply_degradation(
        "mechanical_vibration", x, 1.0, 2, geometry=timed.geometry, tr_s=0.0125, axis="ro"
    )
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="set by scanner profile"):
        apply_degradation("mechanical_vibration", x, 1.0, 2, profile=timed, tr_s=0.02)
