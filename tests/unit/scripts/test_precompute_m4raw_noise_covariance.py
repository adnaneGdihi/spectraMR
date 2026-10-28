"""``scripts/precompute_m4raw_noise_covariance.py``: producer of the store the loader reads (#3117).

The oracle is the consumer: the real ``M4RawRepetitionDataset`` is driven over the
directory the script wrote with the loader's own fit forbidden, so a key, a slice
window or a format the two disagree on is a red test, not a worker that refits.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tests.fixtures import m4raw_slice_volumes as vols

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/precompute_m4raw_noise_covariance.py"


def _load(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("precompute_m4raw_noise_covariance", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def corpus(tmp_path) -> tuple[Path, Path, list[Path]]:
    data = tmp_path / "data"
    data.mkdir()
    files = vols.write_group(data, "2022061001", seed=1) + vols.write_group(
        data, "2022061002", seed=2
    )
    records = [{"relative_path": p.name, "file_id": p.stem} for p in files]
    manifest = tmp_path / "m4raw_test.json"
    manifest.write_text(json.dumps({"data_root": str(data), "records": records}))
    return manifest, tmp_path / "sigma", files


def _r2r(files, root=None, **kw):
    from spectramr.data.datasets.m4raw_dataset import M4RawRepetitionDataset

    return M4RawRepetitionDataset(
        files,
        single_contrast=True,
        coil_processing_mode="none",
        target_mode="r2r",
        r2r_covariance_source="self_calibrated",
        slice_level_records=True,
        noise_covariance_root=root,
        **kw,
    )


def test_the_dataset_reads_what_it_wrote_without_fitting(corpus, monkeypatch) -> None:
    import spectramr.data.datasets.m4raw_dataset as module

    manifest, out, files = corpus
    own = _r2r(files[:3])
    _ = own[0]
    assert _load(monkeypatch).main(["--manifest", str(manifest), "--out", str(out)]) == 0
    assert len(json.loads((out / "noise_covariance.json").read_text())["groups"]) == 2
    monkeypatch.setattr(module, "fit_scan_noise_covariance", lambda *a, **k: pytest.fail("refit"))
    stored = _r2r(files[:3], out)
    _ = stored[0]
    key = str(files[0])
    assert torch.equal(stored._r2r_cov_cache[key], own._r2r_cov_cache[key].to(torch.complex64))


def test_a_second_run_fits_nothing(corpus, monkeypatch) -> None:
    manifest, out, _ = corpus
    mod = _load(monkeypatch)
    assert mod.main(["--manifest", str(manifest), "--out", str(out)]) == 0
    monkeypatch.setattr(mod, "compute_one", lambda *a: pytest.fail("refit"))
    assert mod.main(["--manifest", str(manifest), "--out", str(out)]) == 0


def test_a_missing_group_raises_unless_fit(corpus, monkeypatch) -> None:
    from spectramr.data.datasets.m4raw_noise_covariance_store import NoiseCovarianceMissingError

    manifest, out, files = corpus
    mod = _load(monkeypatch)
    assert mod.main(["--manifest", str(manifest), "--out", str(out), "--limit", "1"]) == 0
    with pytest.raises(NoiseCovarianceMissingError):
        _ = _r2r(files[3:], out)[0]
    _ = _r2r(files[3:], out, noise_covariance_missing="fit")[0]


def test_dry_run_writes_nothing(corpus, monkeypatch) -> None:
    manifest, out, _ = corpus
    assert (
        _load(monkeypatch).main(["--manifest", str(manifest), "--out", str(out), "--dry-run"]) == 0
    )
    assert not out.exists()
