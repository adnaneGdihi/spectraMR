import numpy as np
import pytest

from spectramr.infrastructure.reporting.metrics import iqm


def _head(noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((64, 64))
    img[16:48, 16:48] = 1.0
    return img + noise * np.abs(rng.normal(size=img.shape))


def test_otsu_separates_a_bright_square_from_the_background():
    mask = iqm.foreground_mask(_head(0.02))
    assert mask[32, 32] and not mask[2, 2]
    assert mask.sum() == pytest.approx(32 * 32, rel=0.05)


def test_snr_falls_and_background_sigma_rises_with_noise():
    quiet, loud = iqm.iqms(_head(0.01)), iqm.iqms(_head(0.1))
    assert loud["bg_sigma"] > quiet["bg_sigma"]
    assert loud["snr"] < quiet["snr"]
    assert loud["fber"] < quiet["fber"]


def test_efc_is_higher_for_a_spread_image_than_a_compact_one():
    spread = np.ones((64, 64))
    assert iqm.efc(spread) > iqm.efc(_head())


def test_empty_regions_give_nan_not_a_guess():
    out = iqm.iqms(np.ones((8, 8)), np.ones((8, 8), dtype=bool))
    assert np.isnan(out["snr"]) and np.isnan(out["bg_sigma"])
