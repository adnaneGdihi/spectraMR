"""``scripts/n2n_paper_mosaics.py``: like-for-like grids from the validation PNGs.

A grid is only a comparison if every column shows the same slice, so the reference
check is planted red: an arm whose reference tile differs from the others refuses
to render. The foreground/background split is pinned on a synthetic tile where the
floor level is known, since that split is what explains the cohort's PSNR reading.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from tests.unit.scripts.n2n_paper_fixture import SIZE, build_runs, reference_tile

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "n2n_paper_mosaics.py"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("n2n_paper_mosaics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mosaics() -> Any:
    return _module()


def test_build_writes_grids_split_table_and_index(mosaics: Any, tmp_path: Path) -> None:
    runs = build_runs(tmp_path / "runs")
    out = tmp_path / "out"
    index = mosaics.build(runs, out, ("common", "final"), 0.2, 2)
    for view in ("common", "final"):
        for stem in (
            f"mosaic_n2n_{view}_predictions",
            f"mosaic_n2n_{view}_difference",
            f"fig_n2n_{view}_fg_bg_error",
        ):
            assert (out / "mosaics" / f"{stem}.pdf").exists()
            assert (out / "mosaics" / f"{stem}.png").exists()
        assert (out / "tables" / f"tab_n2n_fg_bg_{view}.md").exists()
    assert index["common_iteration"] == 2000
    on_disk = json.loads((out / "mosaics" / "mosaic_index.json").read_text())
    assert on_disk["views"]["common"]["arms"]["A"]["iteration"] == 2000
    assert on_disk["views"]["final"]["arms"]["A"]["iteration"] == 3000
    assert on_disk["views"]["common"]["difference_block"] == 2
    assert len(on_disk["views"]["common"]["arms"]["A"]["tiles"]) == 2
    with (out / "tables" / "tab_n2n_fg_bg_common.csv").open(newline="") as handle:
        rows = {r["Arm"]: r for r in csv.DictReader(handle)}
    assert rows["A"]["iteration"] == "2000"
    assert rows["A"]["bg mean ref / pred"].startswith("0.08")


def test_drop_paths_are_in_slice_order_and_refuse_a_missing_pass(
    mosaics: Any, tmp_path: Path
) -> None:
    arms = mosaics.discover_arms(build_runs(tmp_path))
    a = arms[0]
    paths = mosaics.drop_paths(a, 2000, "fake")
    assert [p.name[-7:-4] for p in paths] == ["000", "001"]
    assert all("step002000" in p.name for p in paths)
    with pytest.raises(FileNotFoundError, match="iteration 9000"):
        mosaics.drop_paths(a, 9000, "real")


def test_a_differing_reference_refuses_to_render(mosaics: Any, tmp_path: Path) -> None:
    runs = build_runs(tmp_path)
    arms = mosaics.discover_arms(runs)
    iterations = {a.key: 2000 for a in arms}
    assert len(mosaics.reference_tiles(arms, iterations)) == 2
    victim = mosaics.drop_paths(arms[-1], 2000, "real")[1]
    Image.fromarray(np.zeros((SIZE, SIZE), dtype=np.uint8), mode="L").save(victim)
    with pytest.raises(ValueError, match="reference slice 1 differs"):
        mosaics.reference_tiles(arms, iterations)


def test_foreground_mask_and_error_split_read_the_floor(mosaics: Any) -> None:
    reference = reference_tile(0).astype(np.float32) / 255.0
    mask = mosaics.foreground_mask(reference)
    assert mask[16, 16] and not mask[0, 31]
    prediction = np.where(mask, reference, 0.0).astype(np.float32)
    split = mosaics.split_error(prediction, reference, mask)
    assert split["fg_mse"] == pytest.approx(0.0)
    assert split["bg_mean_reference"] == pytest.approx(20 / 255)
    assert split["bg_mean_prediction"] == pytest.approx(0.0)
    assert split["bg_mse"] == pytest.approx((20 / 255) ** 2)
    assert 0.0 < split["bg_fraction"] < 1.0


def test_block_rms_pools_the_signed_difference(mosaics: Any) -> None:
    difference = np.array([[1.0, -1.0, 3.0], [1.0, -1.0, 3.0], [9.0, 9.0, 9.0]], dtype=np.float32)
    pooled = mosaics.block_rms(difference, 2)
    assert pooled.shape == (1, 1)
    assert pooled[0, 0] == pytest.approx(1.0)
    assert np.array_equal(mosaics.block_rms(difference, 1), np.abs(difference))
