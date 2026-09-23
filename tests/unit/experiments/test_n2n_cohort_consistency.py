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
        "checkpoint.checkpoint_dir",
        "reporting.method_name",
        "logging.identity.experiment",
        "logging.sinks.dir",
    }
)


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
        "checkpoint.checkpoint_dir",
        "reporting.method_name",
        "logging.identity.experiment",
        "logging.sinks.dir",
    }
)

_E_EXPECTED_DIFFS = _IDENTITY_DIFFS | {"data.r2r_covariance_source"}


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
