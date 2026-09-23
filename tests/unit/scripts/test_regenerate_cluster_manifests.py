"""Unit tests for scripts/data/regenerate_cluster_manifests.py.

This builder is the only shipped way to produce the v3 JSON manifests the
exemplar arms name in ``data.index_path``. ``audit --probe`` is a *synthetic*
forward probe -- it builds the model and runs a forward pass but never
constructs the dataset -- so an arm can pass 150 checks while naming an index
file that exists nowhere. These tests are what stands in for that gap.

Two things are pinned:

* the **round trip**. The assertion is not that the builder emits keys some
  docstring names, but that ``parse_fastmri_index`` -- the function the dataset
  actually calls -- resolves the emitted manifest to files that exist on disk,
  with ``shape`` retained. A fixture agreeing with a docstring is green and
  blind; the consumer is the oracle.
* the **failure paths**. Every one of them used to exit 0. A build that indexed
  nothing printed "Total: Indexed 0 files" and returned success, which reads as
  "the manifests are built" to anything downstream, including a human.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/data/regenerate_cluster_manifests.py"


def _load():
    spec = importlib.util.spec_from_file_location("regen_cluster_manifests", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RCM = _load()

M4RAW_SPLITS = (
    ("multicoil_train", ("T1_sub1", "T2_sub2")),
    ("multicoil_val", ("FLAIR_sub3",)),
)


def _build_h5_tree(root: Path) -> Path:
    """A minimal stand-in for the M4Raw layout the builder's subpaths expect."""
    h5py = pytest.importorskip("h5py")
    databases = root / "databases"
    for split, subjects in M4RAW_SPLITS:
        d = databases / "m4raw/data" / split / split
        d.mkdir(parents=True, exist_ok=True)
        for subject in subjects:
            with h5py.File(d / f"{subject}.h5", "w") as f:
                f.create_dataset(
                    "kspace", data=np.zeros((2, 4, 16, 16), dtype=np.complex64)
                )
    return databases


def _run(monkeypatch, cwd: Path, *argv: str) -> int:
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(
        "sys.argv", ["regenerate_cluster_manifests.py", *argv], raising=False
    )
    return RCM.main()


# ── the round trip ────────────────────────────────────────────────────


def test_emitted_manifest_resolves_through_the_real_loader(tmp_path, monkeypatch):
    """The consumer is the oracle, not the builder's own docstring."""
    from spectramr.data.datasets.universal_dataset import parse_fastmri_index

    databases = _build_h5_tree(tmp_path)
    rc = _run(
        monkeypatch,
        tmp_path,
        "--data-base",
        str(databases),
        "--datasets",
        "m4raw_multicoil_train",
        "m4raw_multicoil_val",
    )
    assert rc == 0

    records = parse_fastmri_index(str(tmp_path / "data/manifests/m4raw_train.json"))
    assert len(records) == 2
    for record in records:
        # ``primary_path`` is the key FastMRISubjectBuilder reads, remapped from
        # the builder's ``relative_path`` by parse_fastmri_index.
        assert Path(record["primary_path"]).exists()
        # ``shape`` lets the queue builder's patch-compatibility filter check
        # spatial extent WITHOUT loading the volume; a manifest missing it makes
        # that filter materialise the whole corpus (universal_dataset.py:447).
        assert record["shape"] == [2, 4, 16, 16]


def test_val_manifest_is_written_under_its_own_name(tmp_path, monkeypatch):
    databases = _build_h5_tree(tmp_path)
    _run(monkeypatch, tmp_path, "--data-base", str(databases), "--datasets",
         "m4raw_multicoil_val")
    payload = json.loads(
        (tmp_path / "data/manifests/m4raw_multicoil_val.json").read_text()
    )
    assert payload["manifest_version"] == "3.0"
    assert payload["total_records"] == 1
    # Relative, so the manifest survives being copied between machines.
    assert not Path(payload["data_root"]).is_absolute()


# ── the failure paths that used to exit 0 ─────────────────────────────


def test_missing_data_base_raises_instead_of_searching_for_one(tmp_path, monkeypatch):
    """A fallback search writes a manifest naming a tree the caller never asked
    for, and nothing downstream can tell that from a correct one."""
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tmp_path, "--data-base", str(tmp_path / "nope"))
    assert "does not exist" in str(exc.value)


def test_indexing_nothing_is_a_failure(tmp_path, monkeypatch):
    """The directory exists but holds no dataset trees."""
    empty = tmp_path / "databases"
    empty.mkdir()
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tmp_path, "--data-base", str(empty))
    assert "no files indexed" in str(exc.value)


def test_unknown_dataset_name_raises(tmp_path, monkeypatch):
    """No silent fallback: an unrecognised --datasets value must not select
    nothing and report success (non-negotiable 3)."""
    databases = _build_h5_tree(tmp_path)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tmp_path, "--data-base", str(databases), "--datasets",
             "m4raw_train")  # the manifest's name, not the dataset key
    message = str(exc.value)
    assert "unknown dataset" in message
    assert "m4raw_multicoil_train" in message  # names what to use instead


def test_dry_run_over_an_empty_base_also_fails(tmp_path, monkeypatch):
    empty = tmp_path / "databases"
    empty.mkdir()
    with pytest.raises(SystemExit):
        _run(monkeypatch, tmp_path, "--data-base", str(empty), "--dry-run")


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    databases = _build_h5_tree(tmp_path)
    assert _run(monkeypatch, tmp_path, "--data-base", str(databases), "--dry-run") == 0
    assert not (tmp_path / "data/manifests").exists()


# ── the public-tree ratchet ───────────────────────────────────────────


def test_source_names_no_cluster_or_personal_identifiers():
    """This file ships in the public tree (Workstream C). It previously carried
    a hardcoded ``/project/<pi>/<user>/`` fallback list."""
    source = SCRIPT.read_text(encoding="utf-8")
    for identifier in ("johnsson", "agdihi", "/project/", "/scratch/", "uh.edu",
                       "carya", "sabine"):
        assert identifier not in source, f"{identifier!r} leaked into a shipped file"


def test_a_requested_dataset_that_produces_nothing_fails_the_whole_run(
    tmp_path, monkeypatch
):
    """A PARTIAL success is the dangerous shape, not an empty one.

    Train indexes fine, val's tree is missing, the total is non-zero -- so
    without this guard the run exits 0 having written only half of what was
    asked for, and the arm's ``validation_index_path`` names a file that was
    never created. Nothing downstream can tell that from a complete build.
    """
    databases = _build_h5_tree(tmp_path)
    val_dir = databases / "m4raw/data/multicoil_val/multicoil_val"
    for stale in val_dir.iterdir():
        stale.unlink()
    val_dir.rmdir()

    with pytest.raises(SystemExit) as excinfo:
        _run(
            monkeypatch,
            tmp_path,
            "--data-base",
            str(databases),
            "--datasets",
            "m4raw_multicoil_train",
            "m4raw_multicoil_val",
        )

    message = str(excinfo.value)
    assert "m4raw_multicoil_val" in message
    assert "m4raw_multicoil_train" not in message
    # It really is the partial shape: the train half was written and the run
    # still failed. A test that only asserted the raise would also pass if the
    # guard fired on an all-zero run, which the next test already covers.
    assert (tmp_path / "data/manifests/m4raw_train.json").exists()


def test_an_unrequested_missing_dataset_is_still_skipped_quietly(
    tmp_path, monkeypatch
):
    """The discrimination leg: the guard keys on being ASKED for, not on absence.

    A public checkout holds one corpus and not the other eleven. If the default
    sweep raised on every tree it does not have, the builder would be unusable
    for everyone outside the cluster -- so this asserts the fix above did not
    widen into that.
    """
    databases = _build_h5_tree(tmp_path)
    val_dir = databases / "m4raw/data/multicoil_val/multicoil_val"
    for stale in val_dir.iterdir():
        stale.unlink()
    val_dir.rmdir()

    rc = _run(monkeypatch, tmp_path, "--data-base", str(databases))

    assert rc == 0
    assert (tmp_path / "data/manifests/m4raw_train.json").exists()
    assert not (tmp_path / "data/manifests/m4raw_multicoil_val.json").exists()


def test_every_dataset_name_in_the_usage_docstring_is_a_real_key():
    """An unknown name is now a hard error, so a stale example is a broken command.

    The docstring shipped ``--datasets m4raw_multicoil m4raw_motion``; the first
    has never been a key (``m4raw_multicoil_train`` is), so copying the
    documented line exited non-zero the moment the unknown-name guard landed.
    """
    named: list[str] = []
    for line in (RCM.__doc__ or "").splitlines():
        if "--datasets" not in line:
            continue
        for token in line.split("--datasets", 1)[1].split():
            if not token.replace("_", "").isalnum():
                break
            named.append(token)

    assert named, "no --datasets example found in the docstring -- test is vacuous"
    unknown = sorted(set(named) - set(RCM.DATASET_CONFIGS))
    assert not unknown, f"docstring names dataset(s) that do not exist: {unknown}"


# ── --min-reps: the repetition-group filter ───────────────────────────
#
# A leave-one-out NEX target needs 3 repetitions. Where a group ships fewer,
# the dataset's two escapes both change the experiment, so the group is dropped
# at manifest time instead. These pin that it drops the right thing, writes
# somewhere else, and fails loudly rather than emitting an empty manifest.


def _build_nex_tree(root: Path) -> Path:
    """M4Raw's real naming: ``<patient>_<contrast><NN>``, repetitions in NN."""
    h5py = pytest.importorskip("h5py")
    d = root / "databases/m4raw/data/multicoil_train/multicoil_train"
    d.mkdir(parents=True, exist_ok=True)
    stems = (
        # a 3-repetition group -- survives --min-reps 3
        "2022061007_T101", "2022061007_T102", "2022061007_T103",
        # a 2-repetition group -- the shape that blocks n2n arm C
        "2022061008_T101", "2022061008_T102",
    )
    for stem in stems:
        with h5py.File(d / f"{stem}.h5", "w") as f:
            f.create_dataset("kspace", data=np.zeros((2, 4, 16, 16), dtype=np.complex64))
    return root / "databases"


def test_group_key_matches_the_dataset_convention():
    """``stem[:-2]`` is what M4RawRepetitionDataset groups on; mirror it exactly."""
    assert RCM.repetition_group_key("2022061007_T101") == "2022061007_T1"
    assert RCM.repetition_group_key("2022061007_FLAIR01") == "2022061007_FLAIR"


def test_min_reps_drops_only_the_short_group(tmp_path, monkeypatch):
    databases = _build_nex_tree(tmp_path)
    rc = _run(monkeypatch, tmp_path, "--data-base", str(databases),
              "--datasets", "m4raw_multicoil_train", "--min-reps", "3")
    assert rc == 0

    payload = json.loads((tmp_path / "data/manifests/m4raw_train_nex3.json").read_text())
    kept = {r["file_id"] for r in payload["records"]}
    assert kept == {"2022061007_T101", "2022061007_T102", "2022061007_T103"}
    assert payload["total_records"] == 3


def test_min_reps_never_overwrites_the_shared_manifest(tmp_path, monkeypatch):
    """``m4raw_train.json`` is shared by reconstruction, physics_driven and others.

    Filtering in place would silently change the corpus every one of them trains
    on, with nothing downstream able to tell.
    """
    databases = _build_nex_tree(tmp_path)
    _run(monkeypatch, tmp_path, "--data-base", str(databases),
         "--datasets", "m4raw_multicoil_train", "--min-reps", "3")
    assert (tmp_path / "data/manifests/m4raw_train_nex3.json").exists()
    assert not (tmp_path / "data/manifests/m4raw_train.json").exists()


def test_min_reps_below_two_raises(tmp_path, monkeypatch):
    databases = _build_nex_tree(tmp_path)
    with pytest.raises(SystemExit, match="at least 2"):
        _run(monkeypatch, tmp_path, "--data-base", str(databases),
             "--datasets", "m4raw_multicoil_train", "--min-reps", "1")


def test_min_reps_that_drops_everything_is_a_failure(tmp_path, monkeypatch):
    """Exiting 0 with no manifest reads as 'the manifests are built'."""
    databases = _build_nex_tree(tmp_path)
    with pytest.raises(SystemExit, match="produced no manifest"):
        _run(monkeypatch, tmp_path, "--data-base", str(databases),
             "--datasets", "m4raw_multicoil_train", "--min-reps", "9")
    assert not (tmp_path / "data/manifests/m4raw_train_nex9.json").exists()


def test_dry_run_reports_the_filtered_count_and_destination(tmp_path, monkeypatch, capsys):
    """A dry run that printed the unfiltered pair would describe a different run."""
    databases = _build_nex_tree(tmp_path)
    _run(monkeypatch, tmp_path, "--data-base", str(databases),
         "--datasets", "m4raw_multicoil_train", "--min-reps", "3", "--dry-run")
    out = capsys.readouterr().out
    assert "Would index 3 files" in out
    assert "m4raw_train_nex3.json" in out
    assert not (tmp_path / "data/manifests").exists()
