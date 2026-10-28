import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.catalog.examples_payloads import make_image_cases
from spectramr.infrastructure.reporting.plotters.images import image_embedding


def test_a_sharper_image_has_more_high_frequency_power():
    rng = np.random.default_rng(0)
    smooth = np.outer(np.hanning(64), np.hanning(64)).astype(np.float32)
    noisy = smooth + 0.2 * rng.standard_normal(smooth.shape).astype(np.float32)
    d_smooth, d_noisy = image_embedding.descriptor(smooth), image_embedding.descriptor(noisy)
    assert d_smooth.shape == d_noisy.shape == (32,)
    assert d_noisy[16:24].mean() > d_smooth[16:24].mean()


def test_renders_for_one_run_and_for_several_arms(tmp_path):
    one = image_embedding.make(
        pd.DataFrame(), tmp_path / "a.pdf", cases=make_image_cases(8), formats=("pdf",), dpi=72
    )
    arms = {"x": make_image_cases(4), "y": make_image_cases(4)}
    many = image_embedding.make(
        pd.DataFrame(), tmp_path / "b.pdf", arm_cases=arms, formats=("pdf",), dpi=72
    )
    assert one is not None and one.exists() and many is not None and many.exists()


def test_soft_skips_with_too_few_cases_or_mismatched_shapes(tmp_path):
    assert (
        image_embedding.make(pd.DataFrame(), tmp_path / "f.pdf", cases=make_image_cases(7)) is None
    )
    bad = [{"prediction": np.ones((8, 8)), "target": np.ones((9, 9))}] * 6
    assert image_embedding.make(pd.DataFrame(), tmp_path / "f.pdf", cases=bad) is None
