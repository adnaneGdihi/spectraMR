"""Physical properties of the registry degradations that were wrong and are now pinned:
per-sample independence, declared ranges equal to realised ones, and the forward models."""

from __future__ import annotations

import math

import pytest
import torch

import spectramr.infrastructure.physics.degradations as ext
from spectramr.infrastructure.physics.degradations.registry_queries import (
    requirement_free_degradations,
    slice_capable_degradations,
)
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fft2c, ifft2c
from tests.utils.degradation_inputs import apply_degradation


def _img(n: int = 64, b: int = 1, c: int = 1, dtype=torch.complex64) -> torch.Tensor:
    return torch.randn(b, c, n, n, generator=torch.Generator().manual_seed(0), dtype=dtype)


# These run on a 2-D slice; volume-only axes get their batch test under degradations/catalogue/.
# The input has 2 coils; q-map axes get the gate's synthetic maps.
@pytest.mark.parametrize("name", sorted(slice_capable_degradations()))
def test_a_sample_does_not_depend_on_its_batch_neighbours(name: str) -> None:
    x = _img(b=2, c=2)
    x[1] *= 10
    alone = apply_degradation(name, x[:1], 0.7, seed=3)
    batched = apply_degradation(name, x, 0.7, seed=3)
    assert torch.allclose(batched[:1], alone, atol=1e-5, rtol=1e-5)


# Multi-coil axes get their double-precision case in catalogue/test_receive_axes.py.
@pytest.mark.parametrize(
    "name", sorted(set(slice_capable_degradations()) & set(requirement_free_degradations()))
)
def test_double_precision_input_is_accepted(name: str) -> None:
    ext.apply_degradation(name, _img(dtype=torch.complex128), 0.6, seed=2)
    ext.apply_degradation(name, torch.randn(1, 1, 64, 64, dtype=torch.float64), 0.6, seed=2)


def test_complex_gaussian_snr_is_per_sample_and_shared_by_coils() -> None:
    x = _img(b=2, c=2)
    x[1] *= 10
    k = fft2c(x)
    n = fft2c(ext.apply_degradation("complex_gaussian", x, 0.5, seed=1)) - k
    snr = 10 * torch.log10(k.abs().pow(2).mean(dim=(1, 2, 3)) / n.abs().pow(2).mean(dim=(1, 2, 3)))
    assert torch.allclose(snr, torch.full_like(snr, 21.0), atol=0.2)


def test_rician_at_theta_zero_returns_complex_input_unchanged() -> None:
    x = _img()
    assert torch.equal(ext.apply_degradation("rician", x, 0.0), x)


def test_spike_lands_on_the_same_k_samples_in_every_coil() -> None:
    x = _img(n=128, c=4)
    k = fft2c(x)
    added = fft2c(ext.apply_degradation("spike", x, 1.0, seed=2)) - k
    hit = added.abs() > 1e-4 * k.abs().max()
    assert torch.equal(hit, hit[:, :1].expand_as(hit))
    rms = added[hit].abs().pow(2).mean().sqrt() / k.abs().pow(2).mean().sqrt()
    assert rms.item() == pytest.approx(5.0, rel=0.05)


def test_t2star_weights_readout_only_by_the_spin_echo_decay() -> None:
    x = _img()
    w = fft2c(ext.degrade_t2star_blur(x, 0.5)) / fft2c(x)
    want = torch.exp(-2.0 * centred_freqs(64).abs()).view(1, 1, 1, -1).expand_as(w)
    assert torch.allclose(w, want.to(w.dtype), atol=1e-4)


def test_t2star_rejects_an_unknown_echo() -> None:
    with pytest.raises(ValueError, match="echo"):
        ext.degrade_t2star_blur(_img(), 0.5, echo="stimulated")


def test_slice_pve_does_not_darken_the_borders() -> None:
    ones = torch.ones(1, 1, 64, 64, dtype=torch.complex64)
    out = ext.degrade_slice_thickness_pve(ones, 1.0)
    assert torch.allclose(out.abs(), torch.ones(1, 1, 64, 64), atol=1e-5)


def test_pulsatile_peak_displacement_is_the_declared_one() -> None:
    n = 128
    _, xx = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
    ramp = xx.view(1, 1, n, n).to(torch.complex64)
    # base_freq 0: every line sees one cardiac phase, so the output is one static pose.
    phase0 = (torch.rand(1, generator=torch.Generator().manual_seed(5)) * 2 * math.pi).item()
    centre = (int(phase0 / (2 * math.pi) * 16) + 0.5) * 2 * math.pi / 16
    out = ext.degrade_pulsatile_motion(ramp, 1.0, seed=5, base_freq_hz=0.0)
    dx = ((out.real - ramp.real) * (n - 1) / 2).abs().max().item()
    yy, xg = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
    env = torch.exp(-((xg**2 + yy**2) / 0.5)) / (0.5 * math.exp(-0.5))
    pred = (9.0 * abs(math.sin(centre)) * env * xg.abs()).max().item()  # px
    assert dx == pytest.approx(pred, rel=1e-3)


@pytest.mark.parametrize("n", [64, 128])
def test_geometric_warp_peak_displacement_is_the_declared_one(n: int) -> None:
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
    ox = ext.degrade_geometric_warp(xx.view(1, 1, n, n).to(torch.complex64), 0.5, seed=0)
    oy = ext.degrade_geometric_warp(yy.view(1, 1, n, n).to(torch.complex64), 0.5, seed=0)
    d = torch.sqrt((ox.real - xx) ** 2 + (oy.real - yy) ** 2) * (n - 1) / 2
    assert d.max().item() == pytest.approx(8.0, rel=0.02)


def test_chemical_shift_theta_scales_the_physical_shift() -> None:
    x = _img()
    phys = 42.577e6 * 1.5 * 3.5e-6 / 200.0
    out = ext.apply_degradation(
        "chemical_shift", x, 0.5, b0_T=1.5, bw_per_pixel_hz=200.0, fat_fraction=1.0
    )
    ramp = torch.exp(-2j * math.pi * centred_freqs(64).view(1, 1, 1, -1) * 0.5 * phys)
    assert torch.allclose(out, ifft2c(fft2c(x) * ramp), atol=1e-5)


@pytest.mark.parametrize(
    "kw", [{"b0_T": 1.5}, {"b0_T": 1.5, "bw_per_pixel_hz": 200.0, "shift_pixels": 3.0}]
)
def test_chemical_shift_rejects_an_incomplete_or_conflicting_spec(kw) -> None:
    with pytest.raises(ValueError, match="chemical_shift"):
        ext.apply_degradation("chemical_shift", _img(), 0.5, **kw)


def test_susceptibility_slice_plane_selects_the_kernel() -> None:
    x = _img()
    coronal = ext.degrade_susceptibility(x, 1.0, seed=1)
    sagittal = ext.degrade_susceptibility(x, 1.0, seed=1, slice_plane="sagittal")
    axial = ext.degrade_susceptibility(x, 1.0, seed=1, slice_plane="axial")
    assert torch.equal(coronal, sagittal)  # both contain B0 along image y
    assert not torch.allclose(axial, coronal)


def test_a_uniform_off_resonance_moves_content_the_same_way_in_b0_drift_and_b0_geometric(
    monkeypatch,
) -> None:
    from spectramr.infrastructure.physics.degradations import (
        AcquisitionGeometry,
        AcquisitionMode,
    )
    from spectramr.infrastructure.physics.degradations.catalogue.hardware_field import (
        off_resonance_lines,
    )
    from spectramr.infrastructure.physics.degradations.volume import fields_b0

    n, bw, hz = 32, 250.0, 500.0  # +2 px of readout offset
    geometry = AcquisitionGeometry(mode=AcquisitionMode.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 1.0))
    x = torch.zeros(1, 1, 2, n, n, dtype=torch.complex64)
    x[..., 16] = 1.0
    drift = off_resonance_lines(
        x, torch.full((2, n), hz), geometry, bw_hz_per_px=bw, sequence="spin_warp"
    )

    def uniform(batch, shape, *args, dtype, **kw):
        return torch.full((batch, 1, *shape), hz, dtype=dtype)

    monkeypatch.setattr(fields_b0, "b0_offset_hz", uniform)
    geo = fields_b0.b0_geometric_volume(x, 1.0, 0, geometry, dwell_time=1.0 / (n * bw))
    peaks = [int(t.abs()[0, 0, 0, 0].argmax()) for t in (drift, geo)]
    assert peaks == [14, 14]  # both move content by -df / BW_px
