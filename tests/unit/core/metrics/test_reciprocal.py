"""``core.metrics.reciprocal`` is a leaf, and ``import spectramr.pipelines.train`` stays light."""

import subprocess
import sys

import pytest


def _loaded(*modules: str) -> list[str]:
    code = (
        f"import sys, spectramr.pipelines.train;print([m for m in {modules!r} if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return eval(out.stdout.strip().splitlines()[-1])


def test_is_reciprocal_strips_val_prefix_and_case():
    from spectramr.core.metrics.reciprocal import RECIPROCAL, is_reciprocal

    assert frozenset({"nex_equivalent"}) == RECIPROCAL
    assert is_reciprocal("val_NEX_equivalent")
    assert not is_reciprocal("val_psnr")


def test_per_slice_reexports_the_same_objects():
    from spectramr.core.metrics import reciprocal
    from spectramr.infrastructure.reporting import per_slice

    assert per_slice.is_reciprocal is reciprocal.is_reciprocal
    assert per_slice.RECIPROCAL is reciprocal.RECIPROCAL


@pytest.mark.parametrize(
    "module", ["torchmetrics", "transformers", "spectramr.infrastructure.reporting.per_slice"]
)
def test_importing_train_does_not_load(module):
    assert _loaded(module) == []
