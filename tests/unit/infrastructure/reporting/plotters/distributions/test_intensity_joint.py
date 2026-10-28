import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.catalog.examples_payloads import make_image_cases
from spectramr.infrastructure.reporting.plotters.distributions import intensity_joint


def test_renders_from_a_case(tmp_path):
    out = intensity_joint.make(
        pd.DataFrame(), tmp_path / "f.pdf", cases=make_image_cases(), formats=("pdf",), dpi=72
    )
    assert out is not None and out.exists()


def test_soft_skips_without_cases_or_on_a_shape_mismatch(tmp_path):
    assert intensity_joint.make(pd.DataFrame(), tmp_path / "f.pdf", cases=None) is None
    bad = [{"prediction": np.ones((4, 4)), "target": np.ones((5, 5))}]
    assert intensity_joint.make(pd.DataFrame(), tmp_path / "f.pdf", cases=bad) is None
