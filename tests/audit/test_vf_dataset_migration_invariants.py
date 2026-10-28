"""Anti-facade invariants for the VF cohort dataset migration (spec sub-project 3).

The ~17 physics-specialised VF arms select an external dataset via
``data.index_path: data/manifests/external/<id>.json`` (Mechanism A reads it). A
schema-valid YAML can still point an arm at the *wrong* dataset — a B1⁺ arm graded
against a trajectory set, or an ``index_path`` whose manifest does not exist. None
of those is a schema error (pitfall #18: metric/claim ↔ data mismatch), so only a
contract test catches them. This driver pins:

* every external ``index_path`` resolves to a real manifest with a ``data_root``
  and a declared ``source.needs`` (capability);
* the manifest's capability covers the arm's physics need, derived from the arm
  name (B1⁺→need 2, B0/bSSFP→need 1, trajectory/spiral→need 4, MRF→need 5);
* completeness — no ``bart_kspace`` / ``ismrmrd_kspace`` arm was left on a bare
  ``data_root`` without its manifest selector (``oracle_bssfp`` is excluded by
  design until the TWIX reader lands — its manifest data_root differs).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from tests.utils.corpus import tracked_yamls

_REPO = Path(__file__).resolve().parents[2]
_ARM_DIRS = [
    _REPO / "experiments" / "inprogress" / "vf",
    _REPO / "experiments" / "inprogress" / "multi_acquisition",
    _REPO / "experiments" / "inprogress" / "trajectory_recon",
]
_yaml = YAML(typ="safe")

# need keyword (in the arm filename) → the dataset capability need-id it requires.
# Order matters: the most specific physics keyword wins.
_NEED_BY_KEYWORD: list[tuple[str, int]] = [
    ("b1plus", 2),
    ("b1minus", 2),
    ("bloch_siegert", 2),
    ("double_angle", 2),
    ("afi", 2),
    ("trajectory", 4),
    ("spiral", 4),
    ("mrf", 5),
    ("b0", 1),
    ("bssfp", 1),
    ("phase_unwrap", 1),
]


def _load_data_block(path: Path) -> dict:
    doc = _yaml.load(path.read_text())
    data = doc.get("data") if isinstance(doc, dict) else None
    return data if isinstance(data, dict) else {}


def _migrated_arms() -> list[Path]:
    out: list[Path] = []
    for d in _ARM_DIRS:
        if not d.is_dir():
            continue
        for p in tracked_yamls(d, recursive=False):
            data = _load_data_block(p)
            ip = data.get("index_path", "") or ""
            if "data/manifests/external/" in ip:
                out.append(p)
    return out


def _expected_need(arm_name: str) -> int | None:
    name = arm_name.lower()
    for kw, need in _NEED_BY_KEYWORD:
        if kw in name:
            return need
    return None


_MIGRATED = _migrated_arms()
_IDS = [p.stem for p in _MIGRATED]

# vf_22/23/24 derive a REAL B1⁺ map from qmri_bloch_inversion's double-angle
# b1map.cfl (BartKspaceDataset.b1_from_dam, Insko & Bolinger 1993) — the repo's
# loader + arm config assert that capability. But datasets.json catalogs that set
# as T1/T2-only (needs 7/5/4), so the capability ⊆ need invariant cannot yet be
# proven from the catalog. OPEN (cluster-confirmable): confirm b1map.cfl is present
# on the cluster → add need 2 to datasets.json (regen the manifest) and drop this
# exception; else re-point these arms at traveling_heads_7t / kirby B1⁺ MAP.
_B1_PENDING_CONFIRMATION = {
    "exp_vf_22_bloch_siegert_b1plus_sim_v2",
    "exp_vf_23_afi_b1plus_sim_v2",
    "exp_vf_24_double_angle_b1plus_sim_v2",
}


def test_some_arms_were_migrated() -> None:
    """Guard against a glob/path regression silently emptying the cohort."""
    assert len(_MIGRATED) >= 10, f"expected the migrated VF cohort, found {_MIGRATED}"


@pytest.mark.parametrize("arm", _MIGRATED, ids=_IDS)
def test_index_path_resolves_to_a_valid_external_manifest(arm: Path) -> None:
    ip = _load_data_block(arm)["index_path"]
    manifest_path = _REPO / ip
    assert manifest_path.is_file(), f"{arm.name}: manifest {ip} does not exist"
    manifest = json.loads(manifest_path.read_text())
    assert manifest.get("data_root"), f"{ip}: external manifest has no data_root"
    assert manifest.get("source", {}).get(
        "needs"
    ), f"{ip}: manifest declares no source.needs (capability)"


@pytest.mark.parametrize("arm", _MIGRATED, ids=_IDS)
def test_dataset_capability_matches_arm_physics(arm: Path) -> None:
    """The arm's physics need must be in the chosen dataset's capability — a B1⁺
    arm cannot be graded against a trajectory-only dataset (pitfall #18)."""
    need = _expected_need(arm.stem)
    if need is None:
        pytest.skip(f"{arm.stem}: no physics-need keyword to assert")
    manifest = json.loads((_REPO / _load_data_block(arm)["index_path"]).read_text())
    needs = manifest["source"]["needs"]
    if need not in needs and arm.stem in _B1_PENDING_CONFIRMATION:
        pytest.xfail(
            f"{arm.stem}: derives B1⁺ via b1_from_dam from qmri_bloch_inversion's "
            f"b1map.cfl, but datasets.json catalogs needs={needs} (T1/T2) without #2. "
            "Confirm b1map.cfl on cluster → add need 2 to datasets.json, or re-point "
            "at traveling_heads/kirby B1⁺ MAP (see _B1_PENDING_CONFIRMATION)."
        )
    assert need in needs, (
        f"{arm.name}: physics need #{need} not in dataset capability {needs} "
        f"({manifest['dataset_name']}) — likely the wrong dataset (pitfall #18)."
    )


# A migrated main arm whose Δ=0 baseline sibling sits on a DIFFERENT dataset is a
# confound (pitfall #17): no delta between them is attributable to the tested knob.
# vf_30's baseline is on M4Raw (val_bland_altman_bias, but M4Raw has no real B0)
# while its main is on multiecho_radial — a confound + metric mismatch the migration
# exposed. Aligning the whole baseline to its main is a design-level change (deferred);
# tracked here so it cannot ship silently.
_BASELINE_CONFOUND_PENDING = {"exp_vf_30_phase_unwrap_seed_b0_baseline_v2"}


def _baseline_for(main: Path) -> Path | None:
    cand = main.with_name(main.name.replace("_v2.yaml", "_baseline_v2.yaml"))
    return cand if cand.is_file() and cand != main else None


@pytest.mark.parametrize("arm", _MIGRATED, ids=_IDS)
def test_baseline_sibling_shares_dataset_with_main(arm: Path) -> None:
    """A Δ=0 baseline must sit on the same dataset family as its main (else the
    comparison is confounded — pitfall #17)."""
    if "baseline" in arm.stem:
        pytest.skip("arm is itself a baseline")
    baseline = _baseline_for(arm)
    if baseline is None:
        pytest.skip("no baseline sibling")
    main_dt = _load_data_block(arm).get("dataset_type")
    base_dt = _load_data_block(baseline).get("dataset_type")
    if main_dt != base_dt and baseline.stem in _BASELINE_CONFOUND_PENDING:
        pytest.xfail(
            f"{baseline.stem}: dataset_type={base_dt} (M4Raw) != main {main_dt} — "
            "confound + B0-metric-on-no-B0 mismatch; align baseline to main's "
            "dataset (deferred design change, see _BASELINE_CONFOUND_PENDING)."
        )
    assert main_dt == base_dt, (
        f"{baseline.name}: dataset_type={base_dt} != main {arm.name} {main_dt} "
        "— Δ=0 baseline on a different dataset is a confound (pitfall #17)."
    )


def test_no_bart_or_ismrmrd_arm_left_unmigrated() -> None:
    """Completeness: every bart/ismrmrd arm carries its manifest selector
    (oracle_bssfp excluded by design — the TWIX reader has landed, but oracle still
    needs the cluster TWIX->NIfTI extraction before its arms migrate)."""
    stragglers = []
    for d in _ARM_DIRS:
        if not d.is_dir():
            continue
        for p in tracked_yamls(d, recursive=False):
            data = _load_data_block(p)
            if data.get("dataset_type") in (
                "bart_kspace",
                "ismrmrd_kspace",
            ) and "data/manifests/external/" not in (data.get("index_path") or ""):
                stragglers.append(p.name)
    assert not stragglers, f"bart/ismrmrd arms missing index_path selector: {stragglers}"


def _all_arm_yamls() -> list[Path]:
    out: list[Path] = []
    for d in _ARM_DIRS:
        if d.is_dir():
            out.extend(tracked_yamls(d, recursive=False))
    return out


def _has_enabled_loss(losses: object) -> bool:
    """True if the losses block carries at least one enabled loss in any domain."""
    if not isinstance(losses, dict):
        return False
    for key in ("image_losses", "kspace_losses", "complex_losses"):
        for item in losses.get(key) or []:
            if isinstance(item, dict) and item.get("enabled", True):
                return True
    # legacy ``reconstruction: {lambda_*: ...}`` style also counts
    return any(
        isinstance(losses.get(k), dict) and losses[k]
        for k in ("reconstruction", "physics")
    )


@pytest.mark.parametrize("arm", _all_arm_yamls(), ids=[p.stem for p in _all_arm_yamls()])
def test_every_arm_declares_a_loss(arm: Path) -> None:
    """Every VF / multi-acquisition / trajectory arm must declare a non-empty
    ``losses:`` block. ``exp_vf_07`` was the lone arm with none, so its
    LossBuilder built nothing and the pipeline aborted with "No losses were
    built" (smoke 2026-06-15) — a crash the schema cannot catch because the key
    is optional."""
    doc = _yaml.load(arm.read_text())
    losses = doc.get("losses") if isinstance(doc, dict) else None
    assert _has_enabled_loss(losses), (
        f"{arm.name}: no enabled loss in `losses:` — LossBuilder will raise "
        "'No losses were built'. Add an image/kspace/complex loss list."
    )
