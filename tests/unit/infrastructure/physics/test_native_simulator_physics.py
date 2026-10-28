"""Physical properties of the native DigitalTwinSimulator pipeline and the ULF operator."""

from __future__ import annotations

import pytest
import torch

import spectramr.infrastructure.physics.digital_twin_simulator as sim
from spectramr.config.schemas.physics import DigitalTwinConfig
from spectramr.infrastructure.physics.degradation_chain import ChainLink, DegradationChain
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.ulf_forward_operator import DifferentiableULFForwardOperator


def _phantom(n: int = 64, c: int = 1) -> torch.Tensor:
    v = torch.zeros(1, c, n, n)
    v[..., n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = 0.5
    v[..., 3 * n // 8 : 5 * n // 8, 3 * n // 8 : 5 * n // 8] = 1.0
    return v.to(torch.complex64)


def test_noise_lands_only_on_acquired_lines_with_one_sigma_for_all_coils() -> None:
    twin = sim.DigitalTwinSimulator(
        im_size=(64, 64),
        enable_motion=False,
        enable_b0=False,
        enable_b1=False,
        enable_undersampling=True,
        acceleration=4.0,
        snr_range=(10.0, 10.0),
    )
    x = _phantom(c=2)
    x[:, 1] *= 0.1  # a weakly coupled coil sees the same receiver noise
    out, _, clean = twin(x, degradation_only=True, seed=3)
    acquired = twin.last_undersampling_mask.squeeze() > 0
    k_out = fft2c(out)
    assert k_out[..., ~acquired, :].abs().max().item() < 1e-5
    noise = (k_out - fft2c(clean))[..., acquired, :]
    assert noise[:, 0].std().item() == pytest.approx(noise[:, 1].std().item(), rel=0.1)


@pytest.mark.parametrize("R", [2.0, 4.0, 8.0])
def test_native_undersampling_realises_the_declared_acceleration(R: float) -> None:
    k = torch.ones(1, 1, 128, 8, dtype=torch.complex64)
    for s in range(10):
        torch.manual_seed(s)
        _, mask = sim.simulate_undersampling(k, acceleration=R, return_mask=True)
        assert 128 / mask.sum().item() == pytest.approx(R)


def test_rf_zipper_is_a_line_at_its_fe_location() -> None:
    n = 128
    k = fft2c(_phantom(n))
    res = ifft2c(sim.simulate_rf_zipper(k, fe_location=0.75, intensity_ratio=0.3) - k).abs()
    col = res.squeeze().pow(2).sum(dim=0)
    assert int(col.argmax()) == 96
    assert (col[95:98].sum() / col.sum()).item() > 0.99


def test_rf_crosstalk_conserves_the_mean_intensity() -> None:
    im = _phantom()
    out = sim.simulate_rf_crosstalk(im, crosstalk_fraction=0.2)
    assert out.real.mean().item() == pytest.approx(im.real.mean().item(), rel=1e-5)


def test_chain_replay_is_the_fitted_realisation_under_the_same_seed() -> None:
    chain = DegradationChain(
        links=(ChainLink("complex_gaussian", 0.5), ChainLink("rigid_motion", 0.3))
    )
    twin = sim.DigitalTwinSimulator.from_config(
        DigitalTwinConfig(**chain.to_digital_twin_config()), (64, 64)
    )
    replayed, _, clean = twin(_phantom(), degradation_only=True, seed=7)
    fitted = chain.apply(clean, seed=7)
    assert ((replayed - fitted).abs().norm() / fitted.abs().norm()).item() < 1e-4


def _ulf(**kw) -> DifferentiableULFForwardOperator:
    return DifferentiableULFForwardOperator(image_shape=(32, 32), enable_gnl=False, **kw)


def _qmaps():
    shape = (1, 1, 32, 32)
    return torch.ones(shape), torch.full(shape, 1000.0), torch.full(shape, 80.0)


def test_ulf_maxwell_is_zero_at_isocentre_and_wired_off_it() -> None:
    rho, t1, t2 = _qmaps()
    iso = ifft2c(_ulf(enable_noise=False)(rho, t1, t2))
    off = ifft2c(_ulf(enable_noise=False, gz=15.0, slice_offset_m=0.05)(rho, t1, t2))
    assert torch.angle(iso).abs().max().item() < 1e-6
    assert torch.angle(off).abs().max().item() > 0.1


def test_ulf_cartesian_emi_is_a_line_at_f_times_t_read() -> None:
    rho, t1, t2 = _qmaps()
    op = _ulf(
        enable_maxwell=False, thermal_scale=0.0, emi_frequencies=(300.0,), emi_amplitudes=(1.0,)
    )
    col = ifft2c(op(torch.zeros_like(rho), t1, t2)).abs().squeeze().pow(2).sum(dim=0)
    assert int(col.argmax()) == 16 + 9  # 300 Hz x 30 ms readout = 9 px right of centre
    with pytest.raises(ValueError, match="readout timestamps"):
        op(rho, t1, t2, time_vector=torch.linspace(0, 0.03, 32 * 32))
