"""Cohort-consistency guard for the three n2n NEX-synthesis arms.

The cohort's whole design is that the arms differ in ONE knob --
``data.target_mode`` (``r2r`` / ``rep_pair`` / ``phase_aligned_mean``) -- so a
PSNR difference is attributable to the training target and nothing else. A
sharding block added to one arm and not the others silently converts that into a
two-knob comparison, which is why the block is pinned identical here rather than
merely present.

The precision clause is not style. ``complex_unet`` accepts complex tensors, and
DeepSpeed casts weights to half from INSIDE the engine, where
``get_autocast_context``'s complex+fp16 guard cannot see it; fp16 weights against
complex64 activations give NaNs rather than a slowdown. bfloat16 is the other
safe spelling, but not on the sm_70 V100s this cohort targets, which emulate it.

Each predicate below is exercised against a planted violation as well as the
real corpus (non-negotiable 15): a guard that has only ever been seen to pass is
not known to fail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

_COHORT = Path(__file__).resolve().parents[3] / "experiments" / "inprogress" / "n2n"
_ARMS = sorted(_COHORT.glob("*.yaml")) if _COHORT.is_dir() else []


def _arm_id(path: Path) -> str:
    return path.stem


def _parallel(cfg: dict[str, Any]) -> dict[str, Any]:
    return (cfg.get("parallel") or {}) if isinstance(cfg, dict) else {}


def _deepspeed_declaration_error(cfg: dict[str, Any]) -> str | None:
    """Why ``cfg``'s sharding declaration is incoherent, or None."""
    parallel = _parallel(cfg)
    deepspeed = parallel.get("deepspeed") or {}
    strategy = parallel.get("strategy")
    if strategy != "deepspeed":
        return f"parallel.strategy must be 'deepspeed', got {strategy!r}"
    if deepspeed.get("enabled") is not (strategy == "deepspeed"):
        return "parallel.deepspeed.enabled must equal (strategy == 'deepspeed')"
    if deepspeed.get("zero_stage") != 2:
        return (
            f"zero_stage must be 2, got {deepspeed.get('zero_stage')!r}. Stage 3 "
            "partitions parameters for ~50% more communication and a lone arm on "
            "3 is no longer comparable to its siblings."
        )
    return None


def _precision_error(cfg: dict[str, Any]) -> str | None:
    """Why ``cfg``'s AMP setting is unsafe under DeepSpeed on a complex arm."""
    precision = ((cfg.get("optimization") or {}).get("precision")) or {}
    if not precision.get("enabled", False):
        return None  # AMP off -> fp32; the engine emits neither an fp16 nor a bf16 block
    if precision.get("dtype") not in {"bfloat16", "float32"}:
        return (
            f"AMP enabled with dtype={precision.get('dtype')!r} under "
            "parallel.strategy: deepspeed on a complex arm -- fp16 weights against "
            "complex64 activations give NaNs."
        )
    return None


def _compile_error(cfg: dict[str, Any]) -> str | None:
    """Why ``cfg`` compiles a model torchinductor cannot generate code for."""
    torch_compile = ((cfg.get("optimization") or {}).get("compile")) or {}
    deepcompile = (_parallel(cfg).get("deepspeed") or {}).get("compile") or {}
    if torch_compile.get("enabled", False) or deepcompile.get("enabled", False):
        return (
            "compilation is enabled on a complex_unet arm; torchinductor cannot "
            "generate code for complex operators and falls back to eager silently."
        )
    return None


@pytest.mark.skipif(not _ARMS, reason="n2n cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_arm_declares_deepspeed_zero2(arm: Path) -> None:
    error = _deepspeed_declaration_error(yaml.safe_load(arm.read_text()) or {})
    assert error is None, f"{_arm_id(arm)}: {error}"


@pytest.mark.skipif(not _ARMS, reason="n2n cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_arm_is_not_fp16_under_deepspeed(arm: Path) -> None:
    error = _precision_error(yaml.safe_load(arm.read_text()) or {})
    assert error is None, f"{_arm_id(arm)}: {error}"


@pytest.mark.skipif(not _ARMS, reason="n2n cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_arm_does_not_compile_a_complex_model(arm: Path) -> None:
    error = _compile_error(yaml.safe_load(arm.read_text()) or {})
    assert error is None, f"{_arm_id(arm)}: {error}"


@pytest.mark.skipif(len(_ARMS) < 2, reason="n2n cohort not present")
def test_sharding_is_identical_across_the_cohort() -> None:
    """The one-knob design: sharding must not become a second axis."""
    blocks = {_arm_id(a): _parallel(yaml.safe_load(a.read_text()) or {}) for a in _ARMS}
    reference_name, reference = next(iter(blocks.items()))
    for name, block in blocks.items():
        assert block == reference, (
            f"{name}'s parallel block differs from {reference_name}'s. The cohort "
            f"compares target modes; a sharding difference makes it two knobs.\n"
            f"  {reference_name}: {reference}\n  {name}: {block}"
        )


# --- planted violations: each predicate is watched failing (non-negotiable 15) ---

_SHARDED = {"parallel": {"strategy": "deepspeed", "deepspeed": {"enabled": True, "zero_stage": 2}}}


@pytest.mark.parametrize(
    ("planted", "expected"),
    [
        ({"parallel": {}}, "parallel.strategy"),
        (
            {"parallel": {"strategy": "ddp", "deepspeed": {"enabled": True, "zero_stage": 2}}},
            "parallel.strategy",
        ),
        (
            {"parallel": {"strategy": "deepspeed", "deepspeed": {"enabled": False}}},
            "must equal",
        ),
        (
            {
                "parallel": {
                    "strategy": "deepspeed",
                    "deepspeed": {"enabled": True, "zero_stage": 3},
                }
            },
            "zero_stage must be 2",
        ),
    ],
    ids=["no-parallel-block", "strategy-is-ddp", "enabled-disagrees", "stage-3"],
)
def test_deepspeed_predicate_rejects_planted_violation(planted: dict, expected: str) -> None:
    error = _deepspeed_declaration_error(planted)
    assert error is not None and expected in error


@pytest.mark.parametrize(
    ("dtype", "rejected"),
    [("float16", True), (None, True), ("bfloat16", False), ("float32", False)],
    ids=["fp16", "unset-defaults-to-fp16", "bf16", "fp32"],
)
def test_precision_predicate_rejects_fp16(dtype: str | None, rejected: bool) -> None:
    planted = {**_SHARDED, "optimization": {"precision": {"enabled": True, "dtype": dtype}}}
    assert (_precision_error(planted) is not None) is rejected


@pytest.mark.parametrize(
    "planted",
    [
        {"optimization": {"compile": {"enabled": True}}},
        {"parallel": {"strategy": "deepspeed", "deepspeed": {"compile": {"enabled": True}}}},
    ],
    ids=["torch-compile", "deepcompile"],
)
def test_compile_predicate_rejects_planted_violation(planted: dict) -> None:
    assert _compile_error(planted) is not None


# ---------------------------------------------------------------------------
# Arm D: the coil-subspace barrier.
#
# Arm D's claim is narrower than the cohort's. A/B/C differ in
# ``data.target_mode``; D is arm A with ONE addition -- a barrier term and the
# transform that feeds it -- so a D-vs-A difference is attributable to the
# barrier. That holds only while every other knob matches, and the two halves
# must arrive together: the term raises without the maps, and the transform
# without the term is a per-subject eigendecomposition nothing reads.
# ---------------------------------------------------------------------------

_ARM_A = _COHORT / "n2n_a_r2r_single_ex_m4raw.yaml"
_ARM_D = _COHORT / "n2n_d_coil_subspace_barrier_m4raw.yaml"
_BOTH = _ARM_A.is_file() and _ARM_D.is_file()

#: Knobs D is allowed to differ from A on: the two halves of the barrier, and
#: the per-arm identity/path fields every arm necessarily spells for itself.
#: The map source of an arm whose objective reads the maps (the split, the barrier,
#: the projection): maps from the input repetition alone, and the all-repetitions
#: store kept as the scoring maps (audit 2026-10-09).
_MAP_SOURCE_DIFFS = frozenset(
    {"data.coil_maps_root", "data.coil_maps_recipe", "data.coil_maps_reference_root"}
)
_SCORING_STORE = "databases/m4raw/coil_maps"


_D_EXPECTED_DIFFS = frozenset(
    {
        "losses.complex_losses",
        "data.processing.transforms",
        "metadata.name",
        "metadata.description",
        "metadata.hypothesis",
        "metadata.baseline",
        "metadata.expected_outcome",
        "training.output_dir",
        "reporting.method_name",
        "logging.identity.experiment",
        "logging.sinks.dir",
    }
) | _MAP_SOURCE_DIFFS


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """Leaf paths of a config, with lists kept whole (order is meaning here)."""
    if not isinstance(node, dict):
        return {prefix: node}
    out: dict[str, Any] = {}
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out


def _coil_subspace_entry(cfg: dict[str, Any]) -> dict[str, Any] | None:
    for entry in (cfg.get("losses") or {}).get("complex_losses") or []:
        if isinstance(entry, dict) and entry.get("name") == "coil_subspace_residual":
            return entry
    return None


def _espirit_entry(cfg: dict[str, Any]) -> dict[str, Any] | None:
    processing = ((cfg.get("data") or {}).get("processing")) or {}
    for entry in processing.get("transforms") or []:
        if isinstance(entry, dict) and entry.get("name") == "espirit_sensitivity":
            return entry
    return None


def _reads_maps_in_its_objective(cfg: dict[str, Any]) -> bool:
    data = cfg.get("data") or {}
    scoring = (cfg.get("validation") or {}).get("scoring") or {}
    return (
        data.get("target_mode") == "nullspace_r2r"
        or _coil_subspace_entry(cfg) is not None
        or bool(scoring.get("coil_manifold_projection"))
    )


def _map_leak_error(cfg: dict[str, Any]) -> str | None:
    """Why the maps an arm reads (objective or null-space certificate) could contain the reference, or None."""
    data = cfg.get("data") or {}
    if not data.get("coil_maps_root"):
        return None
    if data.get("coil_maps_recipe") != "input_repetition":
        return (
            f"coil_maps_recipe={data.get('coil_maps_recipe')!r} (unset is all_reps_mean): "
            "the objective reads maps averaged over the leave-one-out reference"
        )
    if data.get("coil_maps_reference_root") != _SCORING_STORE:
        return f"coil_maps_reference_root is not the shared scoring store {_SCORING_STORE!r}"
    return None


@pytest.mark.parametrize("arm", _ARMS, ids=[a.stem for a in _ARMS])
def test_no_objective_reads_maps_that_contain_the_reference(arm: Path) -> None:
    error = _map_leak_error(yaml.safe_load(arm.read_text()) or {})
    assert error is None, f"{arm.stem}: {error}"


def test_the_map_leak_predicate_rejects_a_planted_store() -> None:
    cfg = yaml.safe_load(_ARM_G.read_text()) or {}
    assert _map_leak_error(cfg) is None
    cfg["data"].pop("coil_maps_recipe")
    assert (_map_leak_error(cfg) or "").startswith("coil_maps_recipe=None")
    cfg = yaml.safe_load(_ARM_G.read_text()) or {}
    cfg["data"].pop("coil_maps_reference_root")
    assert (_map_leak_error(cfg) or "").startswith("coil_maps_reference_root")


@pytest.mark.skipif(not _BOTH, reason="arms A and D not both present")
def test_arm_d_differs_from_arm_a_only_in_the_barrier() -> None:
    """One knob, or the D-vs-A comparison measures something else too."""
    a = _flatten(yaml.safe_load(_ARM_A.read_text()) or {})
    d = _flatten(yaml.safe_load(_ARM_D.read_text()) or {})
    differing = {k for k in set(a) | set(d) if a.get(k) != d.get(k)}
    unexpected = differing - _D_EXPECTED_DIFFS
    assert not unexpected, (
        f"arm D differs from arm A on {sorted(unexpected)}, which is outside the "
        "barrier. A D-vs-A PSNR difference would no longer be attributable to the "
        "coil-subspace term."
    )


@pytest.mark.skipif(not _BOTH, reason="arms A and D not both present")
def test_the_barrier_and_its_maps_arrive_together() -> None:
    """Either half alone is a defect, in opposite directions.

    The term raises at the first step without the maps; the transform without
    the term is a per-subject eigendecomposition nothing reads.
    """
    cfg = yaml.safe_load(_ARM_D.read_text()) or {}
    assert _coil_subspace_entry(cfg) is not None, "arm D declares no coil_subspace_residual"
    assert _espirit_entry(cfg) is not None, (
        "arm D declares the barrier but not `espirit_sensitivity`; nothing else in "
        "the loader produces coil maps, so the term raises at the first step"
    )


@pytest.mark.skipif(not _BOTH, reason="arms A and D not both present")
def test_the_barrier_takes_the_image_domain_under_a_kspace_output() -> None:
    """`input_domain: kspace` here would be inverse-transformed twice (#467).

    `output_domain: kspace` makes LossBuilder wrap every `complex_losses` entry
    in its own ifft_complex bridge. The loss advertises `use_fourier_bridge`
    under `kspace`, so the builder refuses that combination -- but the arm must
    not be written to depend on being refused.
    """
    cfg = yaml.safe_load(_ARM_D.read_text()) or {}
    entry = _coil_subspace_entry(cfg)
    assert entry is not None
    output_domain = ((cfg.get("losses") or {}).get("policy") or {}).get("output_domain")
    if output_domain == "kspace":
        assert (entry.get("kwargs") or {}).get("input_domain") == "image", (
            "under losses.policy.output_domain: kspace the builder already bridges, "
            "so the term must declare input_domain: image"
        )


@pytest.mark.skipif(not _BOTH, reason="arms A and D not both present")
def test_the_maps_describe_the_physical_array() -> None:
    """ESPIRiT calibrates the physical coils; compressing after that forks them.

    Maps estimated for 4 physical coils against virtual coils produced by a
    later compression describe different arrays, and nothing downstream can
    detect the mismatch.
    """
    cfg = yaml.safe_load(_ARM_D.read_text()) or {}
    mode = ((cfg.get("data") or {}).get("coils") or {}).get("processing_mode")
    assert mode == "none", (
        f"arm D estimates ESPIRiT maps for the physical array but declares "
        f"data.coils.processing_mode={mode!r}"
    )


@pytest.mark.parametrize(
    ("planted", "reason"),
    [
        ({"losses": {"complex_losses": []}}, "barrier removed"),
        ({"data": {"processing": {"transforms": []}}}, "maps removed"),
    ],
)
def test_the_pairing_predicate_rejects_a_planted_half(planted: dict, reason: str) -> None:
    """Half an arm must be rejected, whichever half is missing."""
    cfg: dict[str, Any] = {
        "losses": {"complex_losses": [{"name": "coil_subspace_residual"}]},
        "data": {"processing": {"transforms": [{"name": "espirit_sensitivity"}]}},
    }
    cfg.update(planted)
    assert (_coil_subspace_entry(cfg) is None) or (_espirit_entry(cfg) is None), reason


def test_the_single_knob_predicate_rejects_a_planted_drift() -> None:
    """A second knob must be caught, not absorbed by the allow-list."""
    a = {"data": {"r2r_alpha": 1.0}, "training": {"epochs": 50}}
    d = {"data": {"r2r_alpha": 0.5}, "training": {"epochs": 50}}
    flat_a, flat_d = _flatten(a), _flatten(d)
    differing = {k for k in set(flat_a) | set(flat_d) if flat_a.get(k) != flat_d.get(k)}
    assert differing - _D_EXPECTED_DIFFS == {"data.r2r_alpha"}


# ---------------------------------------------------------------------------
# Arm E: the self-calibrated R2R draw.
#
# Also read against A rather than the A/B/C ladder: same target, same alpha,
# one knob -- where Sigma_n comes from. R2R's decorrelation is exact only where
# the drawn covariance matches the scan's own, so this is the cohort's one
# remaining external assumption made internal.
# ---------------------------------------------------------------------------

_ARM_E = _COHORT / "n2n_e_selfcal_sigma_m4raw.yaml"
_A_AND_E = _ARM_A.is_file() and _ARM_E.is_file()

#: The per-arm identity/path fields every arm necessarily spells for itself.
_IDENTITY_DIFFS = frozenset(
    {
        "metadata.name",
        "metadata.description",
        "metadata.hypothesis",
        "metadata.baseline",
        "metadata.expected_outcome",
        "training.output_dir",
        "reporting.method_name",
        "logging.identity.experiment",
        "logging.sinks.dir",
    }
)

#: ``noise_covariance_root`` is a cache of the same fit (#3117), not a second knob:
#: the store holds what ``fit_scan_noise_covariance`` returns on the same slices.
_E_EXPECTED_DIFFS = _IDENTITY_DIFFS | {"data.r2r_covariance_source", "data.noise_covariance_root"}


@pytest.mark.skipif(not _A_AND_E, reason="arms A and E not both present")
def test_arm_e_differs_from_arm_a_only_in_the_covariance_source() -> None:
    a = _flatten(yaml.safe_load(_ARM_A.read_text()) or {})
    e = _flatten(yaml.safe_load(_ARM_E.read_text()) or {})
    differing = {k for k in set(a) | set(e) if a.get(k) != e.get(k)}
    unexpected = differing - _E_EXPECTED_DIFFS
    assert not unexpected, (
        f"arm E differs from arm A on {sorted(unexpected)}, which is outside the "
        "covariance source. An A-vs-E difference would no longer be attributable "
        "to where Sigma_n came from."
    )


@pytest.mark.skipif(not _A_AND_E, reason="arms A and E not both present")
def test_arm_e_actually_selects_the_self_calibrated_source() -> None:
    """An arm named for the fit that declares the default is the facade shape."""
    cfg = yaml.safe_load(_ARM_E.read_text()) or {}
    assert (cfg.get("data") or {}).get("r2r_covariance_source") == "self_calibrated"


@pytest.mark.skipif(not _A_AND_E, reason="arms A and E not both present")
def test_arm_a_keeps_the_committed_source_as_the_control() -> None:
    """A must NOT declare the knob: its absence is the default and the control."""
    cfg = yaml.safe_load(_ARM_A.read_text()) or {}
    declared = (cfg.get("data") or {}).get("r2r_covariance_source")
    assert declared in (None, "committed"), (
        f"arm A declares r2r_covariance_source={declared!r}; it is arm E's control "
        "and must draw from the committed matrix"
    )


@pytest.mark.skipif(not _ARM_E.is_file(), reason="arm E not present")
def test_the_self_calibrated_fit_needs_uncompressed_coils() -> None:
    """The null space it fits in is `C - 1` dimensional; compression shrinks it.

    Estimating Sigma_n for virtual coils and then drawing noise the physical
    array never produced is a mismatch nothing downstream can detect.
    """
    cfg = yaml.safe_load(_ARM_E.read_text()) or {}
    mode = ((cfg.get("data") or {}).get("coils") or {}).get("processing_mode")
    assert mode == "none", f"arm E fits Sigma_n per coil but declares processing_mode={mode!r}"


def test_the_covariance_source_predicate_rejects_a_planted_default() -> None:
    """An arm that silently declares the default must be caught."""
    planted = {"data": {"r2r_covariance_source": "committed"}}
    assert (planted["data"]).get("r2r_covariance_source") != "self_calibrated"


# --- arm A at alpha 0.5: the drawn-noise twin of the theta sweep ---------------

_ARM_A05 = _COHORT / "n2n_a_alpha05_r2r_single_ex_m4raw.yaml"
_A_AND_A05 = _ARM_A.is_file() and _ARM_A05.is_file()
_A05_EXPECTED_DIFFS = _IDENTITY_DIFFS | {"data.r2r_alpha"}


def _differing(left: Path, right: Path) -> set[str]:
    a = _flatten(yaml.safe_load(left.read_text()) or {})
    b = _flatten(yaml.safe_load(right.read_text()) or {})
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


@pytest.mark.skipif(not _A_AND_A05, reason="arms A and A_alpha05 not both present")
def test_arm_a_alpha05_differs_from_arm_a_only_in_alpha() -> None:
    unexpected = _differing(_ARM_A, _ARM_A05) - _A05_EXPECTED_DIFFS
    assert not unexpected, f"A_alpha05 differs from A outside r2r_alpha: {sorted(unexpected)}"
    cfg = yaml.safe_load(_ARM_A05.read_text()) or {}
    assert cfg["data"]["r2r_alpha"] == 0.5


def test_the_alpha_predicate_rejects_a_planted_second_knob(tmp_path: Path) -> None:
    planted = tmp_path / "planted.yaml"
    cfg = yaml.safe_load(_ARM_A.read_text()) or {}
    cfg["data"]["r2r_alpha"] = 0.5
    cfg["optimization"]["optimizer"]["learning_rate"] = 2e-4
    planted.write_text(yaml.safe_dump(cfg))
    unexpected = _differing(_ARM_A, planted) - _A05_EXPECTED_DIFFS
    assert unexpected == {"optimization.optimizer.learning_rate"}


# --- arm G: the null-space split, one knob group away from arm A ---------------

_ARM_G = _COHORT / "n2n_g_nullspace_r2r_m4raw.yaml"
_A_AND_G = _ARM_A.is_file() and _ARM_G.is_file()

#: The split replaces the drawn recorruption: the target mode itself, the two
#: knobs the split reads, the draw's alpha that the schema refuses under it, the
#: second repetition the repetition-variance metric consumes, and that metric.
_G_EXPECTED_DIFFS = _IDENTITY_DIFFS | _MAP_SOURCE_DIFFS | {
    "data.target_mode",
    "data.r2r_alpha",
    "data.r2r_split_angle_deg",
    "data.r2r_whitening",
}


@pytest.mark.skipif(not _A_AND_G, reason="arms A and G not both present")
def test_arm_g_differs_from_arm_a_only_in_the_split() -> None:
    a = _flatten(yaml.safe_load(_ARM_A.read_text()) or {})
    g = _flatten(yaml.safe_load(_ARM_G.read_text()) or {})
    differing = {k for k in set(a) | set(g) if a.get(k) != g.get(k)}
    unexpected = differing - _G_EXPECTED_DIFFS
    assert not unexpected, (
        f"arm G differs from arm A on {sorted(unexpected)}, which is outside the "
        "null-space split. An A-vs-G difference would no longer be attributable "
        "to where the recorruption came from."
    )


@pytest.mark.skipif(not _ARM_G.is_file(), reason="arm G not present")
def test_arm_g_actually_selects_the_null_space_split() -> None:
    cfg = yaml.safe_load(_ARM_G.read_text()) or {}
    assert (cfg.get("data") or {}).get("target_mode") == "nullspace_r2r"


@pytest.mark.skipif(not _ARM_G.is_file(), reason="arm G not present")
def test_arm_g_states_its_whitener() -> None:
    """`none` is the planted violation on M4Raw's coupled array; the arm must say
    which whitener it runs under rather than inherit a default a reader has to
    look up."""
    cfg = yaml.safe_load(_ARM_G.read_text()) or {}
    assert (cfg.get("data") or {}).get("r2r_whitening") == "committed"


@pytest.mark.skipif(not _ARM_G.is_file(), reason="arm G not present")
def test_the_split_needs_uncompressed_coils() -> None:
    """The null space it splits along is `C - 1` dimensional; compression shrinks it."""
    cfg = yaml.safe_load(_ARM_G.read_text()) or {}
    mode = ((cfg.get("data") or {}).get("coils") or {}).get("processing_mode")
    assert mode == "none", f"arm G splits the physical coil vector but declares processing_mode={mode!r}"


def test_the_split_predicate_rejects_a_planted_drift() -> None:
    """A G that also changed the learning rate would no longer be a one-knob arm."""
    a = _flatten({"data": {"target_mode": "r2r"}, "optimization": {"optimizer": {"learning_rate": 1e-4}}})
    g = _flatten(
        {"data": {"target_mode": "nullspace_r2r"}, "optimization": {"optimizer": {"learning_rate": 3e-4}}}
    )
    differing = {k for k in set(a) | set(g) if a.get(k) != g.get(k)}
    assert differing - _G_EXPECTED_DIFFS == {"optimization.optimizer.learning_rate"}


# --- arm F: arm A scored after projection onto the coil manifold ---------------

_ARM_F = _COHORT / "n2n_f_projection_m4raw.yaml"
_A_AND_F = _ARM_A.is_file() and _ARM_F.is_file()

#: The projection needs maps at validation (the transform), the knob that
#: applies it, the metrics that read the projected output, and the second
#: repetition the Rep-BV metrics consume.
_F_EXPECTED_DIFFS = _IDENTITY_DIFFS | _MAP_SOURCE_DIFFS | {
    "data.processing.transforms",
    "validation.scoring.coil_manifold_projection",
}


@pytest.mark.skipif(not _A_AND_F, reason="arms A and F not both present")
def test_arm_f_differs_from_arm_a_only_in_the_projection() -> None:
    a = _flatten(yaml.safe_load(_ARM_A.read_text()) or {})
    f = _flatten(yaml.safe_load(_ARM_F.read_text()) or {})
    differing = {k for k in set(a) | set(f) if a.get(k) != f.get(k)}
    unexpected = differing - _F_EXPECTED_DIFFS
    assert not unexpected, (
        f"arm F differs from arm A on {sorted(unexpected)}, which is outside the "
        "validation-time projection. An A-vs-F difference would no longer be "
        "attributable to the projection."
    )


@pytest.mark.skipif(not _ARM_F.is_file(), reason="arm F not present")
def test_arm_f_projects_and_carries_no_barrier() -> None:
    """F is D's constraint applied at scoring time: projection on, barrier off."""
    cfg = yaml.safe_load(_ARM_F.read_text()) or {}
    scoring = ((cfg.get("validation") or {}).get("scoring")) or {}
    assert scoring.get("coil_manifold_projection") is True
    assert _coil_subspace_entry(cfg) is None
    assert _espirit_entry(cfg) is not None, "the projection needs maps at validation"


def test_the_projection_predicate_rejects_a_planted_barrier() -> None:
    planted = {"losses": {"complex_losses": [{"name": "coil_subspace_residual", "weight": 1.0}]}}
    assert _coil_subspace_entry(planted) is not None


# --- the inline L1 term is off on every arm ------------------------------------


def _l1_is_pinned_off(cfg: dict[str, Any]) -> bool:
    """`lambda_l1` is 10.0 by schema default and read by the strategy outside
    the weight table, so an arm that does not declare it trains on 10*L1 + L2."""
    recon = (cfg.get("losses") or {}).get("reconstruction") or {}
    return recon.get("lambda_l1") == 0.0


@pytest.mark.parametrize("arm", _ARMS, ids=lambda p: p.stem)
def test_arm_pins_the_inline_l1_term_off(arm: Path) -> None:
    cfg = yaml.safe_load(arm.read_text()) or {}
    assert _l1_is_pinned_off(cfg), (
        f"{arm.name} does not declare losses.reconstruction.lambda_l1: 0.0, so it "
        "trains on 10*L1 + L2 while its kspace_losses declare L2 alone"
    )


@pytest.mark.parametrize(
    "planted",
    [{"losses": {"reconstruction": {"warmup_losses": []}}}, {"losses": {"reconstruction": {"lambda_l1": 10.0}}}, {}],
)
def test_the_l1_predicate_rejects_a_planted_default(planted: dict) -> None:
    assert not _l1_is_pinned_off(planted)


# --- the theta sweep: arm G at four other split angles -------------------------

_SWEEP = sorted(_COHORT.glob("n2n_g_theta*_nullspace_r2r_m4raw.yaml"))

#: A sweep point differs from G in the angle, in its own identity fields, and in
#: naming G (not A) as the arm it is read against.
_SWEEP_EXPECTED_DIFFS = _IDENTITY_DIFFS | {"data.r2r_split_angle_deg", "metadata.expected_outcome"}


def _angle_in_name(arm: Path) -> float:
    return float(arm.stem.split("theta")[1].split("_")[0])


@pytest.mark.skipif(not (_SWEEP and _ARM_G.is_file()), reason="sweep arms or arm G not present")
@pytest.mark.parametrize("arm", _SWEEP, ids=lambda p: p.stem)
def test_a_sweep_point_differs_from_arm_g_only_in_the_angle(arm: Path) -> None:
    g = _flatten(yaml.safe_load(_ARM_G.read_text()) or {})
    p = _flatten(yaml.safe_load(arm.read_text()) or {})
    differing = {k for k in set(g) | set(p) if g.get(k) != p.get(k)}
    unexpected = differing - _SWEEP_EXPECTED_DIFFS
    assert not unexpected, (
        f"{arm.stem} differs from arm G on {sorted(unexpected)}, which is outside the split "
        "angle. A G-vs-sweep difference would no longer be attributable to tau."
    )


@pytest.mark.skipif(not _SWEEP, reason="sweep arms not present")
@pytest.mark.parametrize("arm", _SWEEP, ids=lambda p: p.stem)
def test_a_sweep_point_declares_the_angle_its_name_promises(arm: Path) -> None:
    cfg = yaml.safe_load(arm.read_text()) or {}
    data = cfg.get("data") or {}
    assert data.get("r2r_split_angle_deg") == _angle_in_name(arm), arm.name
    assert data.get("r2r_split_angle_deg") != 45.0, "45 degrees is arm G itself"
    assert (cfg.get("metadata") or {}).get("baseline") == "n2n_g_nullspace_r2r_m4raw"


def test_the_sweep_predicate_rejects_a_planted_angle_drift() -> None:
    """A sweep point whose YAML angle disagrees with its filename is the facade shape."""
    planted = Path("n2n_g_theta30_nullspace_r2r_m4raw.yaml")
    assert _angle_in_name(planted) == 30.0
    assert {"data": {"r2r_split_angle_deg": 45.0}}["data"]["r2r_split_angle_deg"] != _angle_in_name(planted)


# ---------------------------------------------------------------------------
# Protocol v2 (after the first run, PAPER.md 8): the Rep-BV metrics, the maps and
# the second repetition on EVERY arm, the NEX-equivalent as the one selector, and
# per-slice validation rows. Cohort-wide and identical, so the one-knob contrasts
# above still hold; a drift on any arm makes its contrast measure the protocol.
# ---------------------------------------------------------------------------

_REP_BV_METRICS = (
    "null_space_residual",
    "null_space_residual_fraction",
    "rep_variance",
    "reference_noise_corrected_mse",
    "nex_equivalent",
    "bias_squared",
)


#: SENSE-combined PSNR/SSIM beside the RSS pair (PAPER.md 8). Columns only:
#: nothing selects on them while the paper's combination is undecided.
_SENSE_COLUMNS = ("sense_psnr", "sense_ssim")

_SELECTOR = "val_reference_noise_corrected_mse"

#: A pass costs 2J forwards per slice (about 31 min on a V100), so the cohort
#: validates ten times, the last on the final iteration, and stops after a
#: 20,480-iteration horizon without improvement.
_VALIDATION_INTERVAL = 5_120
_PATIENCE = 4


def _protocol_v2_error(cfg: dict[str, Any]) -> str | None:
    validation = cfg.get("validation") or {}
    compute = list(((validation.get("scoring") or {}).get("compute")) or [])
    missing = [m for m in _REP_BV_METRICS + _SENSE_COLUMNS if m not in compute]
    if missing:
        return f"validation.scoring.compute lacks {missing}"
    early = cfg.get("early_stopping") or {}
    # The pass value of val_nex_equivalent is a mean of per-slice ratios whose
    # denominator can be zero or negative; the corrected MSE is linear in slices
    # and, the floor being fixed by the input, selects the pooled-N_eq optimum.
    if early.get("metric") != _SELECTOR or early.get("mode") != "min":
        return f"early_stopping selects {early.get('metric')!r}/{early.get('mode')!r}"
    if (cfg.get("metadata") or {}).get("primary_metric") != "nex_equivalent":
        return f"metadata.primary_metric is {(cfg.get('metadata') or {}).get('primary_metric')!r}"
    metrics = cfg.get("metrics") or {}
    if metrics.get("best_metric_name") != _SELECTOR:
        declared = metrics.get("best_metric_name")
        return f"metrics.best_metric_name {declared!r} contradicts the selector"
    interval = (validation.get("schedule") or {}).get("interval_steps")
    if (interval, early.get("patience")) != (_VALIDATION_INTERVAL, _PATIENCE):
        return f"validation every {interval!r} with patience {early.get('patience')!r}"
    if validation.get("per_batch_rows") is not True:
        return "validation.per_batch_rows is off"
    if (validation.get("loader") or {}).get("batch_size") != 1:
        return "validation.loader.batch_size is not 1, so a row is not a slice"
    data = cfg.get("data") or {}
    if data.get("emit_alternate_input") is not True or not data.get("coil_maps_root"):
        return "data lacks emit_alternate_input / coil_maps_root"
    reference = (data.get("processing") or {}).get("kspace_scale_reference")
    if reference != "input":
        return f"kspace_scale_reference is {reference!r}; prediction scales on the input"
    return _slice_route_budget_error(cfg)


#: Protocol v2's budget: 100 group-route epochs of 512 batches (#2476).
_PROTOCOL_V2_ITERATIONS = 51_200


def _slice_route_budget_error(cfg: dict[str, Any]) -> str | None:
    """Why ``cfg`` is not on the slice-level route at protocol v2's budget, or None.

    The route (#2673) changes what an epoch is: every train slice once (~4,608
    records, ~2,304 batches) instead of 4 random slices per group (512 batches).
    An arm that kept ``epochs: 100`` would train 4.5x longer, so the budget is
    declared in iterations and no epoch count is left beside it to disagree.
    """
    training = cfg.get("training") or {}
    if "epochs" in training:
        return (
            f"training.epochs={training['epochs']!r} is declared; on the slice route an "
            "epoch is every train slice, so the budget is training.max_iterations"
        )
    if training.get("max_iterations") != _PROTOCOL_V2_ITERATIONS:
        return f"training.max_iterations is {training.get('max_iterations')!r}"
    data = cfg.get("data") or {}
    if data.get("slice_level_records") is not True:
        return f"data.slice_level_records is {data.get('slice_level_records')!r}"
    sampling = data.get("sampling") or {}
    if sampling.get("samples_per_volume") != 1:
        return f"samples_per_volume is {sampling.get('samples_per_volume')!r}"
    patch = list(sampling.get("patch_size") or [])
    if len(patch) != 3 or patch[2] != 1:
        return f"patch_size {patch!r} is not one slice deep"
    return None


def test_the_last_validation_pass_lands_on_the_final_iteration() -> None:
    # Otherwise restore_best_weights discards the unscored tail of the budget.
    assert _PROTOCOL_V2_ITERATIONS % _VALIDATION_INTERVAL == 0


@pytest.mark.skipif(not _ARMS, reason="n2n cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_arm_follows_protocol_v2(arm: Path) -> None:
    cfg = yaml.safe_load(arm.read_text()) or {}
    assert _protocol_v2_error(cfg) is None, _protocol_v2_error(cfg)


@pytest.mark.parametrize(
    ("planted", "reason"),
    [
        ({"early_stopping": {"metric": "val_psnr"}}, "selects 'val_psnr'"),
        (
            {"early_stopping": {"metric": "val_nex_equivalent", "mode": "max"}},
            "selects 'val_nex_equivalent'/'max'",
        ),
        ({"early_stopping": {"mode": "max"}}, "/'max'"),
        ({"validation": {"scoring": {"compute": ["psnr", "ssim"]}}}, "lacks"),
        (
            {"validation": {"scoring": {"compute": ["psnr", "ssim", *_REP_BV_METRICS]}}},
            "lacks ['sense_psnr', 'sense_ssim']",
        ),
        ({"validation": {"loader": {"batch_size": 2}}}, "not a slice"),
        ({"validation": {"per_batch_rows": False}}, "per_batch_rows is off"),
        ({"validation": {"schedule": {"interval_steps": 2000}}}, "validation every 2000"),
        ({"early_stopping": {"patience": 10}}, "with patience 10"),
        ({"metrics": {"best_metric_name": "val_psnr"}}, "contradicts"),
        ({"metadata": {"primary_metric": "psnr"}}, "primary_metric is 'psnr'"),
        ({"data": {"emit_alternate_input": False}}, "emit_alternate_input"),
        ({"data": {"processing": {"kspace_scale_reference": "target"}}}, "is 'target'"),
        ({"data": {"processing": {"kspace_scale_reference": None}}}, "is None"),
        ({"training": {"epochs": 100}}, "training.epochs=100 is declared"),
        ({"training": {"max_iterations": 25_600}}, "max_iterations is 25600"),
        ({"training": {"max_iterations": None}}, "max_iterations is None"),
        ({"data": {"slice_level_records": False}}, "slice_level_records is False"),
        ({"data": {"sampling": {"samples_per_volume": 4}}}, "samples_per_volume is 4"),
        ({"data": {"sampling": {"patch_size": [256, 256, 3]}}}, "not one slice deep"),
    ],
)
def test_the_protocol_predicate_rejects_a_planted_drift(planted: dict, reason: str) -> None:
    cfg = yaml.safe_load(_ARM_A.read_text()) or {}
    assert _protocol_v2_error(cfg) is None
    for block, values in planted.items():
        node = cfg.setdefault(block, {})
        for key, value in values.items():
            if isinstance(value, dict):
                node.setdefault(key, {}).update(value)
            else:
                node[key] = value
    error = _protocol_v2_error(cfg)
    assert error is not None and reason in error, error


#: Which pairing forms the training target from what each target_mode builds.
_PAIRING_FOR = {
    "r2r": "ordered",
    "nullspace_r2r": "ordered",
    "rep_pair": "random_pair",
    "phase_aligned_mean": "one_vs_rest_mean",
}


def _n2n_error(cfg: dict) -> str | None:
    """Why an arm is not a Noise2Noise arm, or None."""
    training, data = cfg.get("training") or {}, cfg.get("data") or {}
    if not str(training.get("strategy_class", "")).endswith("n2n_strategy.NoiseToNoiseStrategy"):
        return f"strategy_class={training.get('strategy_class')!r} is not NoiseToNoiseStrategy"
    if training.get("training_mode") != "n2n":
        return f"training_mode={training.get('training_mode')!r}, not 'n2n'"
    expected = _PAIRING_FOR.get(data.get("target_mode"))
    if data.get("n2n_pairing") != expected:
        return f"n2n_pairing={data.get('n2n_pairing')!r} but target_mode wants {expected!r}"
    return None


@pytest.mark.parametrize("arm", _ARMS, ids=lambda p: p.stem)
def test_arm_trains_through_the_noise2noise_strategy(arm: Path) -> None:
    """The training target is formed by the strategy from a repetition stack, not served."""
    error = _n2n_error(yaml.safe_load(arm.read_text()) or {})
    assert error is None, f"{arm.name}: {error}"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["training"].update(
            strategy_class="spectramr.infrastructure.training.strategies.reconstruction."
            "ReconstructionTrainingStrategy"
        ),
        lambda c: c["training"].pop("training_mode"),
        lambda c: c["data"].update(n2n_pairing="random_pair"),
        lambda c: c["data"].pop("n2n_pairing"),
    ],
    ids=["reconstruction-strategy", "no-training-mode", "wrong-pairing", "no-pairing"],
)
def test_the_noise2noise_predicate_rejects_planted_violations(mutate) -> None:
    cfg = yaml.safe_load(_ARM_A.read_text()) or {}
    assert _n2n_error(cfg) is None
    mutate(cfg)
    assert _n2n_error(cfg) is not None


_ARM_D_LAMBDA01 = _COHORT / "n2n_d_lambda01_coil_subspace_barrier_m4raw.yaml"

#: The identity keys an arm owns, plus the one knob the lambda arm varies.
_D_LAMBDA_EXPECTED_DIFFS = {
    "metadata.name",
    "metadata.description",
    "training.output_dir",
    "reporting.method_name",
    "logging.identity.experiment",
    "logging.sinks.dir",
}


@pytest.mark.skipif(not _ARM_D_LAMBDA01.exists(), reason="lambda arm not present")
def test_the_lambda_arm_differs_from_arm_d_only_in_the_barrier_weight() -> None:
    """The D-vs-D_lambda01 gap is read as the barrier's map bias, so nothing else may move."""
    d = yaml.safe_load(_ARM_D.read_text()) or {}
    dl = yaml.safe_load(_ARM_D_LAMBDA01.read_text()) or {}
    assert _coil_subspace_entry(d)["weight"] == 1.0
    assert _coil_subspace_entry(dl)["weight"] == 0.1
    fd, fdl = _flatten(d), _flatten(dl)
    differing = {k for k in set(fd) | set(fdl) if fd.get(k) != fdl.get(k)}
    unexpected = {k for k in differing - _D_LAMBDA_EXPECTED_DIFFS if "complex_losses" not in k}
    assert not unexpected, f"the lambda arm also differs from arm D on {sorted(unexpected)}"


def test_the_map_leak_predicate_rejects_an_arm_whose_only_use_is_the_certificate() -> None:
    cfg = yaml.safe_load(_ARM_A.read_text()) or {}
    assert not _reads_maps_in_its_objective(cfg)
    assert _map_leak_error(cfg) is None
    cfg["data"].pop("coil_maps_recipe")
    assert (_map_leak_error(cfg) or "").startswith("coil_maps_recipe=None")


# ---------------------------------------------------------------------------
# Repetition registration (#2849, #2882).
#
# Every arm is graded against the leave-one-out mean of the repetitions other than
# its input, and B and C train on repetitions too. Subjects move between
# repetitions, so an unregistered reference scores that motion as bias and
# favours arm A (whose halves share one acquisition). The knob has to be the same
# on every arm or the cohort compares two references.
# ---------------------------------------------------------------------------

_SHIFT_STORE = "databases/m4raw/nex_shifts"


def _registration_error(cfg: dict[str, Any]) -> str | None:
    """Why ``cfg`` does not register its repetitions from the shared shift store, or None."""
    data = cfg.get("data") or {}
    if data.get("nex_registration") != "translation":
        return f"data.nex_registration is {data.get('nex_registration')!r}, not 'translation'"
    if data.get("nex_shift_root") != _SHIFT_STORE:
        return f"data.nex_shift_root is {data.get('nex_shift_root')!r}, not {_SHIFT_STORE!r}"
    if data.get("nex_shifts_missing", "raise") != "raise":
        return "data.nex_shifts_missing must stay 'raise': an estimate per worker is the #2977 stall"
    return None


@pytest.mark.skipif(not _ARMS, reason="n2n cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_arm_registers_its_repetitions(arm: Path) -> None:
    error = _registration_error(yaml.safe_load(arm.read_text()) or {})
    assert error is None, f"{_arm_id(arm)}: {error}"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({}, "nex_registration"),
        ({"nex_registration": "none", "nex_shift_root": _SHIFT_STORE}, "nex_registration"),
        ({"nex_registration": "translation"}, "nex_shift_root"),
        (
            {
                "nex_registration": "translation",
                "nex_shift_root": _SHIFT_STORE,
                "nex_shifts_missing": "estimate",
            },
            "nex_shifts_missing",
        ),
    ],
    ids=["undeclared", "none", "no-store", "estimate-on-miss"],
)
def test_registration_predicate_rejects_planted_violation(data: dict, expected: str) -> None:
    error = _registration_error({"data": data})
    assert error is not None and expected in error


# ---------------------------------------------------------------------------
# The slice-chunked tree. The released M4Raw files chunk k-space one column across
# every slice, so a one-slice read decodes the volume: on sabine (2026-10-21) the
# read was 41% of arm B's training phase. The loader reads the manifest's
# ``data_root``, so the root and all three manifests must name the new tree.
# ---------------------------------------------------------------------------

_SLICECHUNKED_ROOT = "databases/m4raw/data_slicechunked"
_SLICECHUNKED_KEYS = ("index_path", "validation_index_path", "test_index_path")


def _slicechunked_error(cfg: dict[str, Any]) -> str | None:
    """Why ``cfg`` reads the released one-chunk-per-column M4Raw files, or None."""
    source = (cfg.get("data") or {}).get("source") or {}
    if source.get("root") != _SLICECHUNKED_ROOT:
        return f"data.source.root is {source.get('root')!r}, not {_SLICECHUNKED_ROOT!r}"
    for key in _SLICECHUNKED_KEYS:
        if not str(source.get(key, "")).endswith("_nex3_slicechunked.json"):
            return f"data.source.{key} is {source.get(key)!r}, not a *_nex3_slicechunked.json manifest"
    return None


@pytest.mark.skipif(not _ARMS, reason="n2n cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_arm_reads_the_slice_chunked_tree(arm: Path) -> None:
    error = _slicechunked_error(yaml.safe_load(arm.read_text()) or {})
    assert error is None, f"{_arm_id(arm)}: {error}"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ({"root": "databases/m4raw/data"}, "source.root"),
        ({"root": _SLICECHUNKED_ROOT, "index_path": "m_nex3.json"}, "index_path"),
        (
            {
                "root": _SLICECHUNKED_ROOT,
                "index_path": "a_nex3_slicechunked.json",
                "validation_index_path": "b_nex3_slicechunked.json",
                "test_index_path": "c_nex3.json",
            },
            "test_index_path",
        ),
    ],
    ids=["released-root", "released-manifest", "one-manifest-left"],
)
def test_slicechunked_predicate_rejects_planted_violation(source: dict, expected: str) -> None:
    error = _slicechunked_error({"data": {"source": source}})
    assert error is not None and expected in error
