"""Image-domain terms decompress (``expm1``) before the iFFT, so they pull on low bins hardest."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load():
    path = REPO / "scripts/diagnostics/image_term_band_pull.py"
    spec = importlib.util.spec_from_file_location("image_term_band_pull", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_image_term_gradient_falls_off_with_radius() -> None:
    pull = _load().band_pull(size=64, power=1.0, scale=20.0)
    low, mid, high = (pull[b][0] for b in ("low", "mid", "high"))
    assert low > 2.0 * mid > 0.0
    assert mid > high > 0.0
