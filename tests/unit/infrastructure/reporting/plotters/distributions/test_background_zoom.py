import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.catalog.examples_payloads import make_image_cases
from spectramr.infrastructure.reporting.plotters.distributions import background_zoom


def _noisy_cases():
    rng = np.random.default_rng(0)
    case = make_image_cases(1)[0]
    case["target"] = (case["target"] + rng.normal(0, 0.02, case["target"].shape)).astype(np.float32)
    return [case]


def test_renders_and_writes_the_iqm_table_beside_the_figure(
    tmp_path,
):
    out = background_zoom.make(
        pd.DataFrame(), tmp_path / "f.pdf", cases=_noisy_cases(), formats=("pdf",), dpi=72
    )
    assert out is not None and out.exists()
    table = pd.read_csv(out.with_suffix(".iqm.csv"))
    assert list(table["image"]) == ["prediction", "target"]
    assert {"efc", "fber", "snr", "bg_sigma"} <= set(table.columns)


def test_soft_skips_without_cases_or_a_noise_free_background(tmp_path):
    assert background_zoom.make(pd.DataFrame(), tmp_path / "f.pdf", cases=None) is None
    flat = [{"prediction": np.zeros((8, 8)), "target": np.zeros((8, 8))}]
    assert background_zoom.make(pd.DataFrame(), tmp_path / "f.pdf", cases=flat) is None
