"""``scripts/precompute_m4raw_nex_shifts.py``: producer of the store the loader reads (#2977).

The oracle is the consumer: the real ``M4RawRepetitionDataset`` is driven over the
directory the script wrote, with the loader's own estimator forbidden, so a key or
format the two disagree on is a red test rather than a worker that re-estimates.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/precompute_m4raw_nex_shifts.py"
_S, _C, _H, _W = 4, 2, 32, 32


def _load(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("precompute_m4raw_nex_shifts", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_h5(path: Path, seed: int, shift: int) -> None:
    rng = np.random.default_rng(seed)
    image = np.zeros((_S, _C, _H, _W), dtype=np.complex128)
    image[..., 10:20, 12:22] = 5.0
    image = np.roll(image, shift, axis=-1) + 0.05 * rng.normal(size=image.shape)
    k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1))), axes=(-2, -1))
    with h5py.File(str(path), "w") as f:
        f.create_dataset("kspace", data=(k / _H).astype(np.complex64))


@pytest.fixture()
def corpus(tmp_path) -> tuple[Path, Path, list[Path]]:
    """Two groups of three repetitions (the second and third moved) and a manifest."""
    data = tmp_path / "data"
    data.mkdir()
    files, records = [], []
    for g, patient in enumerate(("2022061001", "2022061002")):
        for r, shift in enumerate((0, 2, -1)):
            path = data / f"{patient}_T10{r + 1}.h5"
            _write_h5(path, seed=10 * g + r, shift=shift)
            files.append(path)
            records.append({"relative_path": path.name, "file_id": path.stem})
    manifest = tmp_path / "m4raw_test.json"
    manifest.write_text(json.dumps({"data_root": str(data), "records": records}))
    return manifest, tmp_path / "shifts", files


def _dataset(files, root=None, **kw):
    from spectramr.data.datasets.m4raw_dataset import M4RawRepetitionDataset

    return M4RawRepetitionDataset(
        files,
        single_contrast=True,
        use_repetitions=True,
        coil_processing_mode="none",
        target_mode="phase_aligned_mean",
        nex_registration="translation",
        nex_shift_root=root,
        **kw,
    )


def _group_shifts(dataset, files):
    from spectramr.data.datasets.m4raw_dataset import _load_reps_or_skip

    group = files[:3]
    return dataset._group_shifts(group, _load_reps_or_skip(group, "t"), {"slice_index": None})


def test_it_writes_one_entry_per_group(corpus, monkeypatch) -> None:
    manifest, out, _ = corpus
    assert _load(monkeypatch).main(["--manifest", str(manifest), "--out", str(out)]) == 0
    groups = json.loads((out / "nex_shifts.json").read_text())["groups"]
    assert len(groups) == 2
    first = next(iter(groups.values()))
    assert len(first) == 3 and first[0] == [0.0, 0.0]


def test_the_dataset_reads_what_it_wrote_without_estimating(corpus, monkeypatch) -> None:
    import spectramr.data.datasets.m4raw_dataset as module

    manifest, out, files = corpus
    estimated = _group_shifts(_dataset(files), files)
    assert _load(monkeypatch).main(["--manifest", str(manifest), "--out", str(out)]) == 0
    monkeypatch.setattr(module, "group_shifts", lambda *a: pytest.fail("re-estimated"))
    assert _group_shifts(_dataset(files, out), files) == estimated


def test_a_second_run_estimates_nothing(corpus, monkeypatch) -> None:
    manifest, out, _ = corpus
    mod = _load(monkeypatch)
    assert mod.main(["--manifest", str(manifest), "--out", str(out)]) == 0
    monkeypatch.setattr(mod, "compute_one", lambda *a: pytest.fail("recomputed"))
    assert mod.main(["--manifest", str(manifest), "--out", str(out)]) == 0


def test_dry_run_and_limit(corpus, monkeypatch) -> None:
    manifest, out, _ = corpus
    mod = _load(monkeypatch)
    assert mod.main(["--manifest", str(manifest), "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()
    assert mod.main(["--manifest", str(manifest), "--out", str(out), "--limit", "1"]) == 0
    assert len(json.loads((out / "nex_shifts.json").read_text())["groups"]) == 1


def test_a_group_missing_from_the_store_raises_at_load_unless_estimate(corpus, monkeypatch) -> None:
    from spectramr.data.datasets.m4raw_nex_shift_store import NexShiftMissingError

    manifest, out, files = corpus
    assert (
        _load(monkeypatch).main(["--manifest", str(manifest), "--out", str(out), "--limit", "1"])
        == 0
    )
    second = files[3:]
    from spectramr.data.datasets.m4raw_dataset import _load_reps_or_skip

    reps = _load_reps_or_skip(second, "t")
    with pytest.raises(NexShiftMissingError):
        _dataset(files, out)._group_shifts(second, reps, {"slice_index": None})
    ds = _dataset(files, out, nex_shifts_missing="estimate")
    assert len(ds._group_shifts(second, reps, {"slice_index": None})) == 3


def test_a_shift_root_without_registration_is_refused(corpus) -> None:
    from spectramr.data.datasets.m4raw_dataset import M4RawRepetitionDataset

    _, out, files = corpus
    with pytest.raises(ValueError, match="nex_shift_root"):
        M4RawRepetitionDataset(
            files,
            single_contrast=True,
            use_repetitions=True,
            coil_processing_mode="none",
            target_mode="phase_aligned_mean",
            nex_registration="none",
            nex_shift_root=out,
        )
