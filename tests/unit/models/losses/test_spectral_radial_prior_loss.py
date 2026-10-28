"""``SpectralRadialPriorLoss``: the default comparison is frozen, and ``log_ratio`` is scale-free.

``log1p_difference`` (the default) compares ``log1p(E)`` per radial ring. With ``E << 1`` that is
``E`` itself, so a 2x power deficit is worth ``E/2`` squared: the term shrinks with the image's
absolute scale and a deficit in a faint ring is invisible. ``log_ratio`` compares
``log(E + eps)`` with ``eps`` a multiple of the target's outer-band level, so a 2x deficit costs
``(log 2)**2`` in every ring whatever the scale, and rings below ``min_ring_snr`` are dropped.
"""

import math

import pytest
import torch

from spectramr.models.losses.registry import LossRegistry
from spectramr.models.losses.spectral_radial_prior_loss import (
    SpectralRadialPriorLoss,
    compute_radial_spectrum,
)


def _pair(seed: int = 1234) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    pred = torch.rand(2, 3, 64, 64, generator=g)
    tgt = torch.rand(2, 3, 64, 64, generator=g)
    return pred, tgt


def test_default_comparison_is_bit_identical_to_the_pre_change_formula():
    """Golden values captured on the pre-change code; the default must not move."""
    pred, tgt = _pair()
    assert SpectralRadialPriorLoss()(pred, tgt).item() == float.fromhex("0x1.82a8040000000p-14")
    lin = SpectralRadialPriorLoss(log_space=False)(pred, tgt).item()
    assert lin == float.fromhex("0x1.c72f6c0000000p+1")


def test_default_formula_matches_the_explicit_log1p_expression():
    pred, tgt = _pair()
    p = torch.log1p(compute_radial_spectrum(pred, 32))
    t = torch.log1p(compute_radial_spectrum(tgt, 32))
    expected = (p - t).pow(2).mean()
    assert torch.equal(SpectralRadialPriorLoss()(pred, tgt), expected)


def test_log_space_defaults_follow_the_comparison():
    assert SpectralRadialPriorLoss().log_space is True
    assert SpectralRadialPriorLoss(comparison="log_ratio").log_space is False


def test_log_ratio_is_zero_for_identical_inputs():
    _, tgt = _pair()
    loss = SpectralRadialPriorLoss(comparison="log_ratio")
    assert loss(tgt, tgt).item() == pytest.approx(0.0, abs=1e-12)


def test_log_ratio_costs_log2_squared_per_ring_for_a_half_power_deficit():
    """A 2x power deficit (image x 1/sqrt 2) is (log 2)^2 per ring, at any absolute scale."""
    torch.manual_seed(0)
    tgt = torch.rand(1, 1, 64, 64) + 0.1
    pred = tgt / math.sqrt(2.0)
    loss = SpectralRadialPriorLoss(comparison="log_ratio", eps_noise_mult=1e-3, min_ring_snr=0.0)
    value = loss(pred, tgt).item()
    assert value == pytest.approx(math.log(2.0) ** 2, rel=2e-2)


def test_log_ratio_is_scale_invariant_and_log1p_is_not():
    """The property log1p lacked: the same deficit, measured at three absolute scales."""
    torch.manual_seed(0)
    base_t = torch.rand(1, 1, 64, 64) + 0.1
    base_p = base_t / math.sqrt(2.0)
    ratio = SpectralRadialPriorLoss(comparison="log_ratio", eps_noise_mult=1e-3, min_ring_snr=0.0)
    log1p = SpectralRadialPriorLoss()
    ratio_vals, log1p_vals = [], []
    for scale in (1.0, 1e-3, 1e3):
        t, p = base_t * scale, base_p * scale
        ratio_vals.append(ratio(p, t).item())
        log1p_vals.append(log1p(p, t).item())
    assert max(ratio_vals) / min(ratio_vals) == pytest.approx(1.0, rel=1e-3)
    # At scale 1e-3 the log1p term is ~E^2 and collapses by orders of magnitude.
    assert log1p_vals[0] / log1p_vals[1] > 1e6


def test_low_snr_rings_are_excluded():
    """Only rings with target SNR >= min_ring_snr enter the mean.

    A smooth blob has energy in the first bins only; the rest of the spectrum is white noise
    at the noise level. Re-drawing the noise in the prediction must register only in the
    rings that pass the SNR cut, so the strict loss stays below the loose one.
    """
    torch.manual_seed(0)
    yy, xx = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    blob = torch.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / (2 * 10.0**2))[None, None]
    tgt = blob + 0.1 * torch.randn(1, 1, 64, 64)
    pred = blob + 0.1 * torch.randn(1, 1, 64, 64)
    strict = SpectralRadialPriorLoss(comparison="log_ratio", min_ring_snr=4.0)
    loose = SpectralRadialPriorLoss(comparison="log_ratio", min_ring_snr=0.0)
    assert strict(pred, tgt).item() < loose(pred, tgt).item()
    assert strict.empty_ring_batches == 0


def test_log_ratio_keeps_the_graph_and_gives_zero_when_no_ring_passes_later():
    pred, tgt = _pair()
    pred = pred.clone().requires_grad_(True)
    loss = SpectralRadialPriorLoss(comparison="log_ratio", min_ring_snr=0.0)
    loss(pred, tgt)
    loss.min_ring_snr = 1e12  # every ring now fails the SNR cut on the next batch
    out = loss(pred, tgt)
    assert out.item() == 0.0
    assert loss.empty_ring_batches == 1
    out.backward()
    assert pred.grad is not None and torch.count_nonzero(pred.grad).item() == 0


def test_log_ratio_raises_when_the_first_batch_has_no_ring():
    pred, tgt = _pair()
    loss = SpectralRadialPriorLoss(comparison="log_ratio", min_ring_snr=1e12)
    with pytest.raises(ValueError, match="keeps no ring"):
        loss(pred, tgt)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"comparison": "log_difference"}, "Unknown comparison"),
        ({"comparison": "log_ratio", "log_space": True}, "log_space"),
        ({"comparison": "log_ratio", "eps_noise_mult": 0.0}, "eps_noise_mult"),
        ({"comparison": "log_ratio", "eps_noise_mult": float("nan")}, "eps_noise_mult"),
        ({"comparison": "log_ratio", "min_ring_snr": -1.0}, "min_ring_snr"),
    ],
)
def test_invalid_configs_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SpectralRadialPriorLoss(**kwargs)


def test_log_ratio_is_reachable_through_the_registry_kwargs_path():
    loss = LossRegistry.create("spectral_radial_prior", comparison="log_ratio", min_ring_snr=2.0)
    assert loss.comparison == "log_ratio"
    assert loss.min_ring_snr == 2.0
