"""``scripts/precompute_m4raw_coil_maps.py``: producer of the files the loader reads.

The oracle is the consumer. Every assertion that matters drives the real
``M4RawRepetitionDataset`` over the directory the script wrote, so a format or
key the two disagree on is a red test rather than a run that recomputes (or
raises) on the cluster. Also pinned: skip-if-exists (the job is resumable),
``--dry-run`` writes nothing, ``--limit`` bounds the work, and a failing group
is reported through the exit code instead of stopping the rest.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/precompute_m4raw_coil_maps.py"
_S, _C, _H, _W = 2, 4, 48, 48


def _load():
    spec = importlib.util.spec_from_file_location("precompute_m4raw_coil_maps", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_h5(path: Path, seed: int, phase: float) -> None:
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(-1, 1, _H), np.linspace(-1, 1, _W), indexing="ij")
    centres = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
    sens = np.stack([np.exp(-((xx - a) ** 2 + (yy - b) ** 2)) for a, b in centres])
    anatomy = ((xx**2 + yy**2) < 0.5) * 20.0
    noise = rng.normal(size=(_S, _C, _H, _W)) + 1j * rng.normal(size=(_S, _C, _H, _W))
    image = sens[None] * anatomy * np.exp(1j * phase) + noise
    k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1))), axes=(-2, -1))
    k = (k / np.sqrt(_H * _W)).astype(np.complex64)
    k[..., -8:] = 0
    with h5py.File(str(path), "w") as f:
        f.create_dataset("kspace", data=k)


@pytest.fixture()
def corpus(tmp_path) -> tuple[Path, Path, list[Path]]:
    """Two groups of three repetitions and a v3 JSON manifest naming them."""
    data = tmp_path / "data"
    data.mkdir()
    files, records = [], []
    for g, patient in enumerate(("2022061001", "2022061002")):
        for r, phase in enumerate((0.0, 1.9, -2.4)):
            path = data / f"{patient}_T10{r + 1}.h5"
            _write_h5(path, seed=10 * g + r, phase=phase)
            files.append(path)
            records.append({"relative_path": path.name, "file_id": path.stem})
    manifest = tmp_path / "m4raw_test.json"
    manifest.write_text(json.dumps({"data_root": str(data), "records": records}))
    return manifest, tmp_path / "maps", files


def _dataset(files, root, **kw):
    from spectramr.data.datasets.m4raw_dataset import M4RawRepetitionDataset

    kw.setdefault("target_mode", "nullspace_r2r")
    return M4RawRepetitionDataset(
        files,
        single_contrast=True,
        use_repetitions=True,
        coil_processing_mode="none",
        coil_maps_root=root,
        **kw,
    )


class TestTheFilesItWrites:
    def test_one_file_per_group_with_its_metadata(self, corpus) -> None:
        manifest, out, _ = corpus
        assert _load().main(["--manifest", str(manifest), "--out", str(out)]) == 0
        assert sorted(p.name for p in out.iterdir()) == ["2022061001_T1.pt", "2022061002_T1.pt"]
        payload = torch.load(out / "2022061001_T1.pt", weights_only=True)
        meta = payload["meta"]
        assert payload["maps"].shape == (_S, _C, _H, _W)
        assert payload["maps"].dtype == torch.complex64
        assert meta["acs_size"] == 24 and meta["num_repetitions_averaged"] == 3
        assert meta["pe_support_width"] == _W - 8
        assert meta["file_id"] == "2022061001_T1"
        assert meta["source_files"] == [f"2022061001_T10{r}.h5" for r in (1, 2, 3)]
        assert meta["source_commit"]

    def test_the_dataset_reads_what_it_wrote(self, corpus, monkeypatch) -> None:
        """Round trip through the consumer, on both split kinds, with ESPIRiT forbidden."""
        import spectramr.infrastructure.physics.scan_coil_calibration as calibration

        manifest, out, files = corpus
        assert _load().main(["--manifest", str(manifest), "--out", str(out)]) == 0
        written = torch.load(out / "2022061002_T1.pt", weights_only=True)["maps"]
        monkeypatch.setattr(
            calibration, "estimate_csm_espirit", lambda *a, **k: pytest.fail("re-estimated")
        )
        for kw in ({}, {"target_mode": "phase_aligned_mean", "nex_target_exclude_input": True}):
            subject = _dataset(files, out, **kw)[1]
            assert subject["coil_maps_source"] == f"file:{out / '2022061002_T1.pt'}"
            torch.testing.assert_close(
                subject["sensitivity_complex"].data, written.permute(1, 2, 3, 0)
            )

    def test_an_acs_the_arm_does_not_use_is_refused_at_load(self, corpus) -> None:
        from spectramr.data.datasets.m4raw_coil_maps import CoilMapMismatchError

        manifest, out, files = corpus
        assert (
            _load().main(["--manifest", str(manifest), "--out", str(out), "--acs-size", "28"]) == 0
        )
        with pytest.raises(CoilMapMismatchError, match="acs_size=28"):
            _dataset(files, out)._load_item(0)


class TestItIsResumable:
    def test_an_existing_file_is_skipped(self, corpus, monkeypatch) -> None:
        manifest, out, _ = corpus
        mod = _load()
        assert mod.main(["--manifest", str(manifest), "--out", str(out), "--limit", "1"]) == 0
        first = (out / "2022061001_T1.pt").stat().st_mtime_ns
        assert sorted(p.name for p in out.iterdir()) == ["2022061001_T1.pt"]
        assert mod.main(["--manifest", str(manifest), "--out", str(out)]) == 0
        assert (out / "2022061001_T1.pt").stat().st_mtime_ns == first
        assert (out / "2022061002_T1.pt").is_file()

    def test_dry_run_writes_nothing(self, corpus, capsys) -> None:
        manifest, out, _ = corpus
        assert _load().main(["--manifest", str(manifest), "--out", str(out), "--dry-run"]) == 0
        assert not out.exists()
        assert capsys.readouterr().out.count("<- 3 repetition(s)") == 2


class TestFailuresAreCounted:
    def test_a_missing_manifest_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="regenerate_cluster_manifests"):
            _load().main(["--manifest", str(tmp_path / "absent.json"), "--out", str(tmp_path)])

    def test_one_unreadable_group_fails_the_exit_code_not_the_rest(self, corpus) -> None:
        manifest, out, files = corpus
        for path in files[:3]:
            path.write_bytes(b"not hdf5")
        assert _load().main(["--manifest", str(manifest), "--out", str(out)]) == 1
        assert sorted(p.name for p in out.iterdir()) == ["2022061002_T1.pt"]

    def test_a_relative_out_resolves_where_the_loader_looks(self, corpus, monkeypatch) -> None:
        """The loader resolves data.coil_maps_root through PathResolver; so must the writer."""
        from spectramr.data.metadata.path_resolver import PathResolver

        manifest, out, _ = corpus
        monkeypatch.setattr(
            PathResolver,
            "resolve",
            staticmethod(lambda p: str(out) if p == "databases/m4raw/coil_maps" else p),
        )
        assert _load().main(["--manifest", str(manifest), "--limit", "1"]) == 0
        assert (out / "2022061001_T1.pt").is_file()

    def test_the_help_states_the_cost(self, capsys) -> None:
        with pytest.raises(SystemExit):
            _load().main(["--help"])
        assert "0.8-2.3 s per 256x256" in capsys.readouterr().out


class TestTheInputRepetitionRecipe:
    """``--recipe input_repetition``: maps from the served input alone (#2510)."""

    _ARGS = ("--recipe", "input_repetition", "--acs-size", "20")

    def test_the_dataset_reads_it_only_when_it_declares_the_same_recipe(self, corpus) -> None:
        from spectramr.data.datasets.m4raw_coil_maps import CoilMapMismatchError

        manifest, out, files = corpus
        assert _load().main(["--manifest", str(manifest), "--out", str(out), *self._ARGS]) == 0
        meta = torch.load(out / "2022061001_T1.pt", weights_only=True)["meta"]
        assert meta["recipe"] == "input_repetition" and meta["acs_size"] == 20
        assert meta["num_repetitions_averaged"] == 1
        kw = {"target_mode": "phase_aligned_mean", "nex_target_exclude_input": True}
        subject = _dataset(
            files, out, coil_maps_recipe="input_repetition", coil_maps_acs_size=20, **kw
        )[0]
        assert subject["coil_maps_source"].startswith("file:")
        # Planted: the default recipe reading this store would serve maps it did not ask for.
        with pytest.raises(CoilMapMismatchError, match="recipe 'input_repetition'"):
            _dataset(files, out, coil_maps_acs_size=20, **kw)._load_item(0)

    def test_the_maps_are_the_input_repetitions_alone(self, corpus) -> None:
        """Rewriting the other repetitions leaves the input_repetition maps unchanged."""
        manifest, out, files = corpus
        assert _load().main(["--manifest", str(manifest), "--out", str(out), *self._ARGS]) == 0
        before = torch.load(out / "2022061001_T1.pt", weights_only=True)["maps"]
        for path, seed in zip(files[1:3], (91, 92), strict=True):
            _write_h5(path, seed=seed, phase=0.7)
        (out / "2022061001_T1.pt").unlink()
        assert _load().main(["--manifest", str(manifest), "--out", str(out), *self._ARGS]) == 0
        after = torch.load(out / "2022061001_T1.pt", weights_only=True)["maps"]
        torch.testing.assert_close(after, before)

    def test_an_existing_file_of_another_recipe_is_refused_not_skipped(self, corpus) -> None:
        """Planted: the default store re-used for input_repetition used to read as done."""
        manifest, out, _ = corpus
        mod = _load()
        assert mod.main(["--manifest", str(manifest), "--out", str(out), "--limit", "1"]) == 0
        assert mod.main(["--manifest", str(manifest), "--out", str(out), *self._ARGS]) == 1
        assert sorted(p.name for p in out.iterdir()) == ["2022061001_T1.pt"]

    def test_an_unknown_recipe_is_refused_by_the_parser(self) -> None:
        with pytest.raises(SystemExit):
            _load().build_parser().parse_args(["--recipe", "all_reps_median"])

    def test_the_input_is_the_first_file_of_the_group_on_both_sides(self, corpus) -> None:
        """The leak check: the precompute's repetition 0 is the file the dataset serves."""
        from spectramr.data.datasets.m4raw_coil_maps import group_repetition_files
        from spectramr.data.datasets.m4raw_dataset import M4RawRepetitionDataset

        _, _, files = corpus
        shuffled = [files[i] for i in (2, 0, 4, 1, 5, 3)]
        script_side = [group[0] for group in group_repetition_files(shuffled).values()]
        dataset_side = [group[0] for group in M4RawRepetitionDataset._build_groups(shuffled)]
        assert sorted(script_side) == sorted(dataset_side)
        assert all(p.stem.endswith("01") for p in script_side)
