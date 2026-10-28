"""Every reporting figure is written by ``style.save_figure``.

Ten plotters called ``fig.savefig`` with a fixed DPI and one format, so a run
declaring ``reporting.formats: [pdf, tiff]`` silently got whatever each plotter
hard-coded (#2504). ``style.save_plot`` is the one way a plotter ends.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPORTING = Path("src/spectramr/infrastructure/reporting")
_OWNER = _REPORTING / "style.py"
_SAVEFIG = re.compile(r"\.savefig\s*\(")


def _offenders(text: str) -> int:
    return len(_SAVEFIG.findall(text))


@pytest.mark.parametrize(
    "planted",
    [
        'fig.savefig(out_path, dpi=200, bbox_inches="tight")',
        "plt.savefig(out)",
        "figure.savefig (path)",
        "self.fig.savefig(\n    out_path,\n)",
    ],
    ids=["figure", "pyplot", "space-before-paren", "multiline-attribute"],
)
def test_the_detector_sees_every_savefig_shape(planted):
    assert _offenders(planted) == 1


def test_the_detector_passes_the_sanctioned_call():
    assert _offenders("save_plot(fig, out_path, formats=formats, dpi=dpi)") == 0


def test_no_reporting_module_saves_around_the_style_owner():
    hits = {
        str(p): _offenders(p.read_text(encoding="utf-8"))
        for p in _REPORTING.rglob("*.py")
        if p != _OWNER and _offenders(p.read_text(encoding="utf-8"))
    }
    assert not hits, f"figures saved outside style.save_figure: {hits}"


@pytest.mark.parametrize("fmt", [("png",), ("pdf", "png", "svg")])
def test_a_former_bypasser_writes_every_requested_format(tmp_path, fmt):
    import numpy as np
    import pandas as pd

    from spectramr.infrastructure.reporting.plotters import get

    rng = np.random.default_rng(0)
    target = rng.uniform(0, 1, (16, 16))
    cases = [{"prediction": target * 0.9, "target": target, "case_id": "c0"}]
    out = get("fig_1_12_failure_gallery")(
        pd.DataFrame(), tmp_path / f"g.{fmt[0]}", cases=cases, formats=fmt, dpi=50
    )
    assert out is not None
    assert all((tmp_path / f"g.{f}").exists() for f in fmt)

def test_the_acquisition_pareto_honours_the_requested_formats(tmp_path):
    import pandas as pd

    from spectramr.infrastructure.reporting.catalog.examples_payloads import make_acquisition_arms
    from spectramr.infrastructure.reporting.plotters import get

    out = get("learnable_acquisition_pareto")(
        pd.DataFrame(), tmp_path / "p.png", acquisition_arms=make_acquisition_arms(),
        formats=("png",), dpi=40,
    )
    assert out is not None and out.suffix == ".png" and not (tmp_path / "p.pdf").exists()
