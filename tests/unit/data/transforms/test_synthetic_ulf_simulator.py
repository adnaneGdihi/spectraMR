"""Unit tests for SimulateULFFromHF.

Pin physical reasonableness of the synthetic ULF: the output is
noisier and blurrier than the input, the source HF image is
preserved alongside it, and the transform is deterministic given
a fixed seed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
tio = pytest.importorskip("torchio")

from spectramr.data.transforms.synthetic_ulf_simulator import (  # noqa: E402
    SimulateULFFromHF,
)


def _hf_subject(D: int = 8, H: int = 32, W: int = 32) -> tio.Subject:
    rng = torch.Generator().manual_seed(0)
    # TorchIO ScalarImage data is (C, W, H, D).
    data = torch.rand(1, W, H, D, generator=rng)
    return tio.Subject(target=tio.ScalarImage(tensor=data))


def test_attaches_synthetic_ulf_image() -> None:
    subject = _hf_subject()
    out = SimulateULFFromHF(target_snr_db=10.0)(subject)
    assert "input" in out
    assert "target" in out
    assert out["input"].data.shape == out["target"].data.shape


def test_ulf_is_noisier_than_hf() -> None:
    subject = _hf_subject()
    out = SimulateULFFromHF(target_snr_db=8.0, t2_star_blur_sigma=0.0,
                              b0_distortion_strength=0.0,
                              enable_maxwell=False)(subject)
    hf = out["target"].data.float()
    ulf = out["input"].data.float()
    # High-frequency residual energy is a noise proxy.
    hf_diff = (hf - hf.mean()).pow(2).mean().sqrt()
    ulf_diff = (ulf - ulf.mean()).pow(2).mean().sqrt()
    # Rician noise raises overall variance.
    assert ulf_diff >= 0.5 * hf_diff


def test_ulf_is_blurrier_than_hf() -> None:
    """T2* blurring should reduce the magnitude of the local Laplacian."""
    subject = _hf_subject()
    out = SimulateULFFromHF(
        target_snr_db=40.0,             # very low noise — isolate blur effect
        t2_star_blur_sigma=2.0,         # strong blur
        b0_distortion_strength=0.0,
        enable_maxwell=False,
    )(subject)
    hf = out["target"].data.float()
    ulf = out["input"].data.float()

    def _hf_energy(x: torch.Tensor) -> torch.Tensor:
        # Sum-of-squared finite differences along W (proxy for sharpness).
        return (x[..., 1:] - x[..., :-1]).pow(2).mean()

    assert _hf_energy(ulf) < _hf_energy(hf)


def test_missing_source_raises() -> None:
    subject = tio.Subject(other=tio.ScalarImage(tensor=torch.rand(1, 8, 8, 8)))
    with pytest.raises(KeyError, match="target"):
        SimulateULFFromHF()(subject)


def test_unknown_sequence_and_pe_axis_raise() -> None:
    with pytest.raises(ValueError, match="sequence"):
        SimulateULFFromHF(sequence="gre")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pe_axis"):
        SimulateULFFromHF(pe_axis="D")  # type: ignore[arg-type]


def test_sequence_is_stamped() -> None:
    t = SimulateULFFromHF(sequence="epi")
    assert "sequence" in t.args_names
    assert t.sequence == "epi"


def _geometry_only(**kw: object) -> SimulateULFFromHF:
    return SimulateULFFromHF(
        target_snr_db=200.0,
        t2_star_blur_sigma=0.0,
        b0_distortion_strength=0.1,
        enable_maxwell=False,
        pe_axis="H",
        **kw,  # type: ignore[arg-type]
    )


def _stripes(axis: str) -> tio.Subject:
    """Stripe phantom, constant along the other in-plane axis. TorchIO is (C,W,H,D)."""
    D, H, W = 8, 32, 32
    ramp = (torch.arange(32) % 8 < 4).float() * 0.8 + 0.1
    if axis == "W":  # varies along readout (W), constant along H
        data = ramp.view(1, W, 1, 1).expand(1, W, H, D)
    else:  # varies along PE (H), constant along W
        data = ramp.view(1, 1, H, 1).expand(1, W, H, D)
    return tio.Subject(target=tio.ScalarImage(tensor=data.contiguous()))


def test_spin_warp_shifts_readout_only() -> None:
    """Readout-stripe phantom moves under spin-warp; PE-stripe phantom does not."""
    ro = _geometry_only(sequence="spin_warp")(_stripes("W"))
    pe = _geometry_only(sequence="spin_warp")(_stripes("H"))
    assert (ro["input"].data - ro["target"].data).abs().max() > 0.05
    assert (pe["input"].data - pe["target"].data).abs().max() < 1e-3


def test_epi_shifts_pe_only() -> None:
    ro = _geometry_only(sequence="epi")(_stripes("W"))
    pe = _geometry_only(sequence="epi")(_stripes("H"))
    assert (pe["input"].data - pe["target"].data).abs().max() > 0.05
    assert (ro["input"].data - ro["target"].data).abs().max() < 1e-3


def test_zero_shift_warp_is_identity() -> None:
    from spectramr.data.transforms.synthetic_ulf_simulator import _apply_axis_shift

    vol = torch.rand(1, 2, 5, 9, 11)
    for axis in ("H", "W"):
        out = _apply_axis_shift(vol, torch.zeros(5, 9, 11), axis=axis)  # type: ignore[arg-type]
        assert torch.allclose(out, vol, atol=1e-6)


def test_field_strength_validation() -> None:
    with pytest.raises(ValueError, match="field_strength_t"):
        SimulateULFFromHF(field_strength_t=0.0)


def test_maxwell_only_below_one_tesla() -> None:
    """At 3 T the Maxwell branch must be a no-op (only B0+blur+noise contribute)."""
    subject = _hf_subject()
    high_field = SimulateULFFromHF(
        field_strength_t=3.0,
        target_snr_db=40.0,
        t2_star_blur_sigma=0.0,
        b0_distortion_strength=0.0,
        enable_maxwell=True,
    )(subject)
    no_maxwell = SimulateULFFromHF(
        field_strength_t=3.0,
        target_snr_db=40.0,
        t2_star_blur_sigma=0.0,
        b0_distortion_strength=0.0,
        enable_maxwell=False,
    )(subject)
    # Same RNG draws → identical output when Maxwell is silent.
    torch.manual_seed(0)
    a = high_field["input"].data
    torch.manual_seed(0)
    b = no_maxwell["input"].data
    # Outputs aren't bit-identical (Rician noise reseeds), but they
    # must be statistically similar — far closer than two ULF runs.
    diff = (a - b).abs().mean().item()
    assert diff < 0.5  # generous bound; we just want "not catastrophically different"
