"""Cohort-consistency guard for the self_supervised reference-free arms.

Three properties, each of which was violated when this guard was written.

**The equivariance ablation needs its control.** `ei_unet_m4raw_r4`'s own
metadata says to compare "ONLY against a zero-filled or
measurement-consistency-only reconstruction of the SAME UNet
(alpha_equivariance=0)". No such arm existed, so the cohort could not support
its own claim. The pair is pinned here by shared `metadata.group` and by the
knob being the only substantive difference.

**Validation must not run every other step.** Both arms declared
`validation.schedule.interval_steps: 2` against `max_iterations: 100000` --
roughly 50,000 validation passes, and low enough that the `TRAIN_ITERS` smoke
cap could never lower it, which silently removes the one affordance that
exercises the data path cheaply.

**The unread normalization key.** `data.normalization` is not a schema field,
is not in `RENAMES`, and a nonsense value loads clean, so both arms ran in raw
scanner units while declaring `minmax`. Scale is load-bearing for equivariant
imaging: the measurement-consistency anchor carries squared scanner units while
the equivariance term is scale-free, so their 1:1 balance drifts with receiver
gain unless normalization is on.

Each predicate is exercised against a planted violation as well as the real
corpus (non-negotiable 15).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

_COHORT = Path(__file__).resolve().parents[3] / "experiments" / "inprogress" / "self_supervised"
_ARMS = sorted(_COHORT.glob("*.yaml")) if _COHORT.is_dir() else []
_EI_PAIR = ("ei_unet_m4raw_r4", "ei_control_mc_only_m4raw_r4")


def _arm_id(path: Path) -> str:
    return path.stem


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _validation_cadence_error(cfg: dict[str, Any], floor: int = 100) -> str | None:
    """Why ``cfg``'s validation cadence is too tight to run or to cap."""
    steps = ((cfg.get("validation") or {}).get("schedule") or {}).get("interval_steps")
    if steps is None:
        return None  # inherits the schema default, which is not this defect
    if steps < floor:
        return (
            f"validation.schedule.interval_steps={steps} is below {floor}: the run "
            "spends its wall clock validating, and the TRAIN_ITERS smoke cap can "
            "never lower an interval already beneath it."
        )
    return None


def _unread_normalization_error(cfg: dict[str, Any]) -> str | None:
    """``data.normalization`` is read by nothing; the live surface is processing."""
    if "normalization" in (cfg.get("data") or {}):
        return (
            "data.normalization is not a schema field and is not in RENAMES, so it "
            "is read by nothing and a nonsense value loads clean. Declare "
            "data.processing.enable_kspace_normalization instead."
        )
    return None


def _val_shuffle_error(cfg: dict[str, Any]) -> str | None:
    """An unshuffled val loader scores the leading background slices of a volume."""
    loader = ((cfg.get("validation") or {}).get("loader")) or {}
    if loader.get("shuffle") is False:
        return "validation.loader.shuffle is false: the first-N-slices pathology."
    return None


@pytest.mark.skipif(not _ARMS, reason="self_supervised cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_validation_cadence_is_runnable(arm: Path) -> None:
    error = _validation_cadence_error(_load(arm))
    assert error is None, f"{_arm_id(arm)}: {error}"


@pytest.mark.skipif(not _ARMS, reason="self_supervised cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_no_arm_declares_the_unread_normalization_key(arm: Path) -> None:
    error = _unread_normalization_error(_load(arm))
    assert error is None, f"{_arm_id(arm)}: {error}"


@pytest.mark.skipif(not _ARMS, reason="self_supervised cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_validation_loader_is_shuffled(arm: Path) -> None:
    error = _val_shuffle_error(_load(arm))
    assert error is None, f"{_arm_id(arm)}: {error}"


@pytest.mark.skipif(not _ARMS, reason="self_supervised cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_every_arm_declares_a_held_out_test_split(arm: Path) -> None:
    """Otherwise every reported number is computed on the selection set."""
    source = ((_load(arm).get("data") or {}).get("source")) or {}
    assert source.get("test_index_path"), (
        f"{_arm_id(arm)}: no data.source.test_index_path. M4Raw ships a real "
        "multicoil_test split and regenerate_cluster_manifests.py builds it."
    )


@pytest.mark.skipif(not _ARMS, reason="self_supervised cohort not present")
def test_the_equivariance_ablation_has_its_control() -> None:
    present = {_arm_id(a) for a in _ARMS}
    missing = [name for name in _EI_PAIR if name not in present]
    assert not missing, (
        f"the equivariance ablation is missing {missing}. ei_unet's own metadata "
        "names alpha_equivariance=0 as the only admissible comparison."
    )

    cfgs = {name: _load(_COHORT / f"{name}.yaml") for name in _EI_PAIR}
    groups = {name: (cfg.get("metadata") or {}).get("group") for name, cfg in cfgs.items()}
    assert len(set(groups.values())) == 1 and all(groups.values()), (
        f"the pair must share metadata.group or audit_paired_arms raises: {groups}"
    )

    alphas = {
        name: ((cfg.get("training") or {}).get("equivariant_imaging") or {}).get(
            "alpha_equivariance"
        )
        for name, cfg in cfgs.items()
    }
    assert alphas[_EI_PAIR[0]] and not alphas[_EI_PAIR[1]], (
        f"the control must set alpha_equivariance to 0 and the treatment above it: {alphas}"
    )


# --- planted violations: each predicate is watched failing (non-negotiable 15) ---


@pytest.mark.parametrize(
    ("steps", "rejected"),
    [(2, True), (99, True), (100, False), (2000, False), (None, False)],
    ids=["two", "just-below", "at-floor", "shipped", "unset"],
)
def test_cadence_predicate_rejects_a_tight_interval(steps: int | None, rejected: bool) -> None:
    planted = {"validation": {"schedule": {"interval_steps": steps}}} if steps else {}
    assert (_validation_cadence_error(planted) is not None) is rejected


@pytest.mark.parametrize(
    ("planted", "rejected"),
    [
        ({"data": {"normalization": "minmax"}}, True),
        ({"data": {"normalization": "not_a_real_normalizer"}}, True),
        ({"data": {"processing": {"enable_kspace_normalization": True}}}, False),
    ],
    ids=["the-shipped-value", "a-nonsense-value", "the-live-surface"],
)
def test_normalization_predicate_rejects_the_unread_key(planted: dict, rejected: bool) -> None:
    assert (_unread_normalization_error(planted) is not None) is rejected


@pytest.mark.parametrize(
    ("planted", "rejected"),
    [
        ({"validation": {"loader": {"shuffle": False}}}, True),
        ({"validation": {"loader": {"shuffle": True}}}, False),
        ({"validation": {"loader": {}}}, False),
    ],
    ids=["unshuffled", "shuffled", "unset"],
)
def test_shuffle_predicate_rejects_an_unshuffled_loader(planted: dict, rejected: bool) -> None:
    assert (_val_shuffle_error(planted) is not None) is rejected


@pytest.mark.skipif(not _ARMS, reason="self_supervised cohort not present")
def test_no_arm_uses_the_status_field_as_a_comment() -> None:
    """``metadata.status`` is a LAUNCH GATE, not documentation.

    ``_refuse_unlaunchable_status`` (``pipelines/train.py``) refuses the run
    before seeding, so recording a caveat there costs the cohort the arm's
    result entirely. A caveat belongs in prose; a status means "do not run me".
    Learned by putting one on ``robust_ssdu`` and watching the cluster refuse it.
    """
    blocked = {"needs_implementation", "inert", "blocked"}
    offenders = {
        _arm_id(arm): (_load(arm).get("metadata") or {}).get("status")
        for arm in _ARMS
        if (_load(arm).get("metadata") or {}).get("status") in blocked
    }
    assert not offenders, (
        f"these arms will not launch: {offenders}. If that is intended, say so here; "
        "if it was meant as a note, move it to the description."
    )


@pytest.mark.skipif(not _ARMS, reason="self_supervised cohort not present")
def test_the_inert_ssdu_treatment_is_still_recorded_somewhere() -> None:
    """The declared-sigma caveat must survive the correction being implemented.

    ``noise_std_estimate`` is a constant never measured on this data. Where it
    falls below the scan's sigma the corruption is too weak for the correction to
    denoise, and on the acquired lines the arm behaves as vanilla SSDU does. That
    has to stay visible or the control gets reported as a Robust SSDU result.
    """
    arm = _COHORT / "robust_ssdu_m4raw_r4.yaml"
    if not arm.exists():
        pytest.skip("robust_ssdu arm not present")
    text = arm.read_text().lower()
    assert "vanilla ssdu" in text, "the inert-treatment caveat was lost"


# ---------------------------------------------------------------------------
# ei_multicoil: the operator the arm inverts.
#
# EI recovers what a SINGLE operator leaves unidentifiable, so the operator is
# what decides whether the arm has a problem to solve. Built explicitly at R4,
# one virtual coil leaves 12 of 16 directions unrecoverable and four physical
# coils leave 0 -- so an arm that compresses its coils away manufactures the
# under-determination its assumed dihedral symmetry then repairs (#2207).
# ---------------------------------------------------------------------------

_EI_MC = _COHORT / "ei_multicoil_m4raw_r4.yaml"
_EI_SVD = _COHORT / "ei_unet_m4raw_r4.yaml"


def _ei_block(cfg: dict[str, Any]) -> dict[str, Any]:
    return ((cfg.get("training") or {}).get("equivariant_imaging")) or {}


def _compression(cfg: dict[str, Any]) -> dict[str, Any]:
    return (((cfg.get("physics") or {}).get("coil_processing")) or {}).get("compression") or {}


def _multicoil_coherence_error(cfg: dict[str, Any]) -> str | None:
    """Why this arm's operator declaration is incoherent, or None.

    The two halves are not independent: the multi-coil operator has no coils to
    use if they were compressed away, and estimating maps for the physical array
    after compressing to virtual ones leaves the two describing different arrays.
    """
    if not _ei_block(cfg).get("multicoil_operator"):
        return None
    method = _compression(cfg).get("method")
    if method not in (None, "none"):
        return (
            f"multicoil_operator=true with coil compression {method!r}: the "
            "operator would use maps estimated for an array the data no longer "
            "describes."
        )
    if "svd_coil_compression" in _transform_names(cfg):
        return (
            "multicoil_operator=true with svd_coil_compression: the operator needs "
            "the physical coils the transform compresses away."
        )
    if not _declares_maps(cfg):
        return (
            "multicoil_operator=true without coil maps (data.coil_maps_root or the "
            "`espirit_sensitivity` transform): the strategy raises."
        )
    return None


def _transform_names(cfg: dict[str, Any]) -> set[str]:
    return {
        t.get("name")
        for t in (((cfg.get("data") or {}).get("processing")) or {}).get("transforms") or []
        if isinstance(t, dict)
    }


def _declares_maps(cfg: dict[str, Any]) -> bool:
    """The loader's two map sources: the precomputed store, or per-load ESPIRiT."""
    store = (cfg.get("data") or {}).get("coil_maps_root") is not None
    return store or "espirit_sensitivity" in _transform_names(cfg)


@pytest.mark.skipif(not _EI_MC.is_file(), reason="ei_multicoil arm not present")
def test_the_multicoil_arm_declares_a_coherent_operator() -> None:
    error = _multicoil_coherence_error(yaml.safe_load(_EI_MC.read_text()) or {})
    assert error is None, error


@pytest.mark.skipif(not _EI_MC.is_file(), reason="ei_multicoil arm not present")
def test_the_multicoil_arm_keeps_the_network_unchanged() -> None:
    """The coils enter through the OPERATOR; the model still predicts one image.

    An arm that also widened `in_channels` would be testing two knobs, and the
    comparison against the compressed sibling would no longer isolate the
    operator.
    """
    mc = yaml.safe_load(_EI_MC.read_text()) or {}
    svd = yaml.safe_load(_EI_SVD.read_text()) or {}
    assert (mc.get("model") or {}).get("in_channels") == (svd.get("model") or {}).get("in_channels")
    assert (mc.get("model") or {}).get("out_channels") == (svd.get("model") or {}).get(
        "out_channels"
    )
    assert _ei_block(mc).get("group") == _ei_block(svd).get("group")
    assert _ei_block(mc).get("alpha_equivariance") == _ei_block(svd).get("alpha_equivariance")


@pytest.mark.skipif(not _EI_SVD.is_file(), reason="ei_unet arm not present")
def test_the_compressed_sibling_stays_on_the_combined_operator() -> None:
    """It is the control; declaring the knob would collapse the comparison."""
    assert _ei_block(yaml.safe_load(_EI_SVD.read_text()) or {}).get("multicoil_operator") in (
        None,
        False,
    )


@pytest.mark.parametrize(
    ("planted", "rejected"),
    [
        ({}, False),
        ({"training": {"equivariant_imaging": {"multicoil_operator": True}}}, True),
        (
            {
                "training": {"equivariant_imaging": {"multicoil_operator": True}},
                "physics": {"coil_processing": {"compression": {"method": "svd"}}},
                "data": {"processing": {"transforms": [{"name": "espirit_sensitivity"}]}},
            },
            True,
        ),
        (
            {
                "training": {"equivariant_imaging": {"multicoil_operator": True}},
                "physics": {"coil_processing": {"compression": {"method": "none"}}},
                "data": {"processing": {"transforms": [{"name": "espirit_sensitivity"}]}},
            },
            False,
        ),
        (
            {
                "training": {"equivariant_imaging": {"multicoil_operator": True}},
                "data": {"coil_maps_root": "databases/m4raw/coil_maps_input_rep_acs24"},
            },
            False,
        ),
        (
            {
                "training": {"equivariant_imaging": {"multicoil_operator": True}},
                "data": {
                    "coil_maps_root": "databases/m4raw/coil_maps_input_rep_acs24",
                    "processing": {"transforms": [{"name": "svd_coil_compression"}]},
                },
            },
            True,
        ),
    ],
    ids=["knob-off", "maps-missing", "coils-compressed", "coherent", "store", "svd-transform"],
)
def test_multicoil_predicate_rejects_each_planted_shape(planted: dict, rejected: bool) -> None:
    assert (_multicoil_coherence_error(planted) is not None) is rejected


# ---------------------------------------------------------------------------
# robust_ssdu_selfcal: where the Noisier2Noise sigma comes from.
#
# `inject_noisier_kspace` is the IDENTITY at sigma -> 0, so sigma is not a
# tuning knob -- it is the arm. After per-subject normalization no constant can
# be right for the whole corpus: normalization removes SCALE but turns SNR
# differences into sigma differences, and the correct value spans 0.018 (high
# SNR) to 0.268 (low) through this cohort's own image-domain quantile.
# ---------------------------------------------------------------------------

_SSDU_SC = _COHORT / "robust_ssdu_selfcal_m4raw_r4.yaml"
_SSDU_DECLARED = _COHORT / "robust_ssdu_m4raw_r4.yaml"


def _ssdu_block(cfg: dict[str, Any]) -> dict[str, Any]:
    return ((cfg.get("training") or {}).get("ssdu")) or {}


def _selfcal_sigma_error(cfg: dict[str, Any]) -> str | None:
    """Why this arm's sigma declaration is incoherent, or None."""
    ssdu = _ssdu_block(cfg)
    if ssdu.get("noise_std_source") != "self_calibrated":
        return None
    if ssdu.get("noise_std_estimate") is not None:
        return (
            "noise_std_source: self_calibrated IGNORES noise_std_estimate, but it "
            f"is declared as {ssdu['noise_std_estimate']} -- a number nothing reads "
            "that a reader would take for the sigma used."
        )
    if not ssdu.get("noisier2noise_correction"):
        return (
            "noise_std_source: self_calibrated with the Noisier2Noise correction "
            "off: the measured sigma would reach nothing."
        )
    compression = (
        (((cfg.get("physics") or {}).get("coil_processing")) or {}).get("compression") or {}
    ).get("method")
    if compression not in (None, "none"):
        return (
            f"noise_std_source: self_calibrated with coil compression "
            f"{compression!r}: the null space sigma is measured in is C-1 "
            "dimensional, and compressing to one virtual coil empties it."
        )
    if not _declares_maps(cfg):
        return (
            "noise_std_source: self_calibrated without coil maps (data.coil_maps_root "
            "or `espirit_sensitivity`): the null space the estimate reads has no basis."
        )
    return None


@pytest.mark.skipif(not _SSDU_SC.is_file(), reason="selfcal SSDU arm not present")
def test_the_selfcal_ssdu_arm_is_coherent() -> None:
    error = _selfcal_sigma_error(yaml.safe_load(_SSDU_SC.read_text()) or {})
    assert error is None, error


@pytest.mark.skipif(not _SSDU_SC.is_file(), reason="selfcal SSDU arm not present")
def test_the_selfcal_arm_carries_no_launch_gate() -> None:
    """`metadata.status` is a LAUNCH GATE, not documentation.

    The sibling declares `needs_implementation` because its treatment is inert.
    This arm IS that implementation, so inheriting the gate would refuse to
    launch the fix.
    """
    status = (yaml.safe_load(_SSDU_SC.read_text()) or {}).get("metadata", {}).get("status")
    assert status is None, f"arm declares metadata.status={status!r}, which refuses launch"


@pytest.mark.skipif(not _SSDU_DECLARED.is_file(), reason="declared SSDU arm not present")
def test_the_declared_sibling_keeps_its_constant_and_says_it_is_the_control() -> None:
    """It is the selfcal arm's control: its sigma stays declared, and its status says
    what it is without refusing the launch. The Robust SSDU correction is now in the
    code (#2327), so `needs_implementation` would be false and would stop the run
    that the selfcal comparison needs."""
    cfg = yaml.safe_load(_SSDU_DECLARED.read_text()) or {}
    assert _ssdu_block(cfg).get("noise_std_source") in (None, "declared")
    assert _ssdu_block(cfg).get("noise_std_estimate") is not None
    assert (cfg.get("metadata") or {}).get("status") == "baseline_control"


@pytest.mark.parametrize(
    ("planted", "rejected"),
    [
        ({}, False),
        (
            {"training": {"ssdu": {"noise_std_source": "self_calibrated"}}},
            True,
        ),
        (
            {
                "training": {
                    "ssdu": {
                        "noise_std_source": "self_calibrated",
                        "noisier2noise_correction": True,
                        "noise_std_estimate": 0.02,
                    }
                }
            },
            True,
        ),
        (
            {
                "training": {
                    "ssdu": {
                        "noise_std_source": "self_calibrated",
                        "noisier2noise_correction": True,
                    }
                },
                "physics": {"coil_processing": {"compression": {"method": "svd"}}},
                "data": {"processing": {"transforms": [{"name": "espirit_sensitivity"}]}},
            },
            True,
        ),
        (
            {
                "training": {
                    "ssdu": {
                        "noise_std_source": "self_calibrated",
                        "noisier2noise_correction": True,
                    }
                },
                "physics": {"coil_processing": {"compression": {"method": "none"}}},
                "data": {"processing": {"transforms": [{"name": "espirit_sensitivity"}]}},
            },
            False,
        ),
        (
            {
                "training": {
                    "ssdu": {
                        "noise_std_source": "self_calibrated",
                        "noisier2noise_correction": True,
                    }
                },
                "data": {"coil_maps_root": "databases/m4raw/coil_maps_input_rep_acs24"},
            },
            False,
        ),
        (
            {
                "training": {
                    "ssdu": {
                        "noise_std_source": "self_calibrated",
                        "noisier2noise_correction": True,
                    }
                },
            },
            True,
        ),
    ],
    ids=[
        "declared",
        "correction-off",
        "stale-constant",
        "coils-compressed",
        "coherent",
        "store",
        "maps-missing",
    ],
)
def test_selfcal_sigma_predicate_rejects_each_planted_shape(planted: dict, rejected: bool) -> None:
    assert (_selfcal_sigma_error(planted) is not None) is rejected


# ---------------------------------------------------------------------------
# #2324 / #2325 / #2327: what the arms declare so the fixed code does the right
# thing. Each predicate has planted shapes below it (non-negotiable 15).
# ---------------------------------------------------------------------------

#: Paths that name an arm rather than specify it: never a knob.
_IDENTITY_PREFIXES = ("metadata.", "logging.", "checkpoint.", "reporting.method_name")
_IDENTITY_LEAVES = ("output_dir",)


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            out.update(_flatten(value, f"{prefix}{key}."))
        return out
    return {prefix[:-1]: node}


def _knob_diff(a: dict[str, Any], b: dict[str, Any]) -> set[str]:
    """Dotted paths whose values differ, identity paths excluded."""
    fa, fb = _flatten(a), _flatten(b)
    return {
        path
        for path in set(fa) | set(fb)
        if fa.get(path) != fb.get(path)
        and not path.startswith(_IDENTITY_PREFIXES)
        and path.rsplit(".", 1)[-1] not in _IDENTITY_LEAVES
    }


def _image_network(cfg: dict[str, Any]) -> bool:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import get_model_capabilities

    populate_model_registry()
    caps = get_model_capabilities(str((cfg.get("model") or {}).get("model_type")))
    return caps is not None and caps.output_domain == "image"


def _scored_target_error(cfg: dict[str, Any]) -> str | None:
    """An image network over a k-space loader must bring its target to the image domain."""
    if (cfg.get("data") or {}).get("dataset_type") not in ("kspace", "m4raw"):
        return None
    if not _image_network(cfg):
        return None
    declared = ((cfg.get("validation") or {}).get("scoring") or {}).get("output_transform")
    if declared not in ("ifft_target_magnitude", "sense_magnitude"):
        return (
            f"validation.scoring.output_transform={declared!r}: the prediction is an image "
            "and the target is k-space. Undeclared scores |k| (peak at DC, #2325); "
            "ifft_magnitude would transform the image too. Declare sense_magnitude "
            "(ifft_target_magnitude for an arm without coil maps)."
        )
    if declared == "sense_magnitude" and not (cfg.get("data") or {}).get("coil_maps_root"):
        return "sense_magnitude without data.coil_maps_root: there are no maps to combine with."
    return None


def _espirit_reference_error(cfg: dict[str, Any]) -> str | None:
    """ESPIRiT must calibrate from the measurement, inside its fully sampled centre."""
    data = cfg.get("data") or {}
    transforms = [
        t for t in (data.get("processing") or {}).get("transforms") or [] if isinstance(t, dict)
    ]
    espirit = [t for t in transforms if t.get("name") == "espirit_sensitivity"]
    if not espirit:
        return None
    kwargs = espirit[0].get("kwargs") or {}
    if list(kwargs.get("keys") or []) != ["input"]:
        return (
            f"espirit_sensitivity keys={kwargs.get('keys')!r}: the default order reads "
            "`kspace`, the fully sampled reference (#2324). Declare keys: [input]."
        )
    fraction = (cfg.get("undersampling") or {}).get("center_fraction")
    for t in transforms:
        if t.get("name") == "complementary_rep_masking":
            fraction = (t.get("kwargs") or {}).get("center_fraction", fraction)
    # The ACS band runs along the phase-encode axis (dim -1, the patch WIDTH) and
    # is rounded the way complementary_rep_masking._acs rounds it.
    width = ((data.get("sampling") or {}).get("patch_size") or [256, 256])[1]
    centre = round(width * float(fraction or 0.0))
    if int(kwargs.get("acs_size", 24)) > centre:
        return (
            f"espirit_sensitivity acs_size={kwargs.get('acs_size', 24)} exceeds the "
            f"{centre} fully sampled centre lines: the window reaches zero-filled lines."
        )
    return None


@pytest.mark.skipif(not _ARMS, reason="self_supervised cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_an_image_network_scores_an_image(arm: Path) -> None:
    error = _scored_target_error(_load(arm))
    assert error is None, f"{_arm_id(arm)}: {error}"


@pytest.mark.skipif(not _ARMS, reason="self_supervised cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_coil_maps_come_from_the_measurement(arm: Path) -> None:
    error = _espirit_reference_error(_load(arm))
    assert error is None, f"{_arm_id(arm)}: {error}"


_ROBUST_ONE_KNOB = {"training.ssdu.noise_std_estimate", "training.ssdu.noise_std_source"}
_REP_EI = _COHORT / "rep_ei_m4raw_r2.yaml"
_SAMEMASK = _COHORT / "conex_r2_samemask_m4raw.yaml"
#: The EI term and what it needs: the strategy, its block, and the coil maps from
#: the input-repetition store. Activation checkpointing pays for the EI term's
#: second forward on a 16 GB GPU and does not change the objective.
_REP_EI_KNOB = {
    "training.strategy_class",
    "training.training_mode",
    "data.coil_maps_root",
    "data.coil_maps_recipe",
    "data.coil_maps_acs_size",
    "optimization.gradient.enable_checkpointing",
}
#: Both select on the corrected MSE and score the repetition decomposition; the
#: metric list and the scoring store differ from the same-mask arm's.
_PROTOCOL_V2_RECONSTRUCTION_ONLY = {
    "validation.scoring.compute",
    "early_stopping.metric",
    "early_stopping.mode",
    "metrics.best_metric_name",
    "metrics.best_metric_mode",
    "metadata.primary_metric",
    "data.emit_alternate_input",
    "data.coil_maps_reference_root",
}


def _rep_ei_diff_error(rep_ei: dict[str, Any], samemask: dict[str, Any]) -> str | None:
    extra = {
        path
        for path in _knob_diff(rep_ei, samemask)
        if path not in _REP_EI_KNOB
        and path not in _PROTOCOL_V2_RECONSTRUCTION_ONLY
        and not path.startswith("training.equivariant_imaging.")
    }
    return (
        f"Rep-EI differs from its baseline beyond the EI term: {sorted(extra)}" if extra else None
    )


@pytest.mark.skipif(not _SSDU_SC.is_file(), reason="selfcal SSDU arm not present")
def test_the_robust_ssdu_pair_differs_in_one_knob() -> None:
    """#2327: the selfcal arm used to differ in sigma source, maps and target_channels."""
    diff = _knob_diff(_load(_SSDU_SC), _load(_SSDU_DECLARED))
    assert diff == _ROBUST_ONE_KNOB, sorted(diff)


@pytest.mark.skipif(not _REP_EI.is_file(), reason="rep_ei arm not present")
def test_rep_ei_differs_from_the_same_mask_arm_only_by_the_ei_term() -> None:
    error = _rep_ei_diff_error(_load(_REP_EI), _load(_SAMEMASK))
    assert error is None, error
    meta = _load(_REP_EI).get("metadata") or {}
    assert meta.get("baseline") == "conex_r2_samemask_m4raw"
    assert meta.get("group") == (_load(_SAMEMASK).get("metadata") or {}).get("group")


@pytest.mark.skipif(not _ARMS, reason="self_supervised cohort not present")
def test_the_robust_arms_leave_their_data_consistency_uncapped() -> None:
    """A tanh after the last DC step clips the image above 1 (#2327)."""
    for arm in (_SSDU_DECLARED, _SSDU_SC):
        kwargs = (_load(arm).get("model") or {}).get("model_kwargs") or {}
        assert kwargs.get("output_activation") == "none", _arm_id(arm)


@pytest.mark.parametrize(
    ("planted", "rejected"),
    [
        ({"data": {"dataset_type": "kspace"}, "model": {"model_type": "standard_unet"}}, True),
        (
            {
                "data": {"dataset_type": "kspace"},
                "model": {"model_type": "standard_unet"},
                "validation": {"scoring": {"output_transform": "ifft_magnitude"}},
            },
            True,
        ),
        (
            {
                "data": {"dataset_type": "kspace"},
                "model": {"model_type": "standard_unet"},
                "validation": {"scoring": {"output_transform": "ifft_target_magnitude"}},
            },
            False,
        ),
        (
            {
                "data": {"dataset_type": "m4raw", "coil_maps_root": "databases/m4raw/coil_maps"},
                "model": {"model_type": "standard_unet"},
                "validation": {"scoring": {"output_transform": "sense_magnitude"}},
            },
            False,
        ),
        (
            {
                "data": {"dataset_type": "m4raw"},
                "model": {"model_type": "standard_unet"},
                "validation": {"scoring": {"output_transform": "sense_magnitude"}},
            },
            True,
        ),
        ({"data": {"dataset_type": "m4raw"}, "model": {"model_type": "complex_unet"}}, False),
    ],
    ids=["undeclared", "symmetric", "declared", "sense", "sense-without-maps", "kspace-network"],
)
def test_scored_target_predicate_rejects_each_planted_shape(planted: dict, rejected: bool) -> None:
    assert (_scored_target_error(planted) is not None) is rejected


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        ("ei_multicoil_m4raw_r4", "sense_magnitude"),
        ("ssdu_m4raw_r4", "sense_magnitude"),
        ("ei_unet_m4raw_r4", "ifft_target_magnitude"),
    ],
)
def test_the_zero_filled_baseline_resolves_the_arms_transform(arm: str, expected: str) -> None:
    """The EI/SSDU val_zf_* follow this name; a resolver miss would leave them on RSS."""
    from spectramr.infrastructure.training.strategies.equivariant_imaging_ops import (
        scoring_transform,
    )
    from tests.fixtures.self_supervised_batches import load_arm

    assert scoring_transform(load_arm(arm)) == expected


@pytest.mark.parametrize(
    ("kwargs", "rejected"),
    [
        ({"acs_size": 20}, True),
        ({"keys": ["kspace_raw", "kspace", "input"], "acs_size": 20}, True),
        ({"keys": ["input"], "acs_size": 24}, True),
        ({"keys": ["input"], "acs_size": 20}, False),
    ],
    ids=["default-keys", "reference-first", "window-too-wide", "coherent"],
)
def test_espirit_predicate_rejects_each_planted_shape(kwargs: dict, rejected: bool) -> None:
    planted = {
        "undersampling": {"center_fraction": 0.08},
        "data": {
            "sampling": {"patch_size": [256, 256, 1]},
            "processing": {"transforms": [{"name": "espirit_sensitivity", "kwargs": kwargs}]},
        },
    }
    assert (_espirit_reference_error(planted) is not None) is rejected


def test_the_one_knob_diff_sees_a_planted_second_knob() -> None:
    base = {"data": {"domain": {"target_channels": 8}}, "training": {"ssdu": {"x": 1}}}
    planted = {"data": {"domain": {"target_channels": 1}}, "training": {"ssdu": {"x": 1}}}
    assert _knob_diff(planted, base) == {"data.domain.target_channels"}
    rep_ei = {"model": {"in_channels": 2}, "training": {"strategy_class": "ei"}}
    assert _rep_ei_diff_error(rep_ei, {"model": {"in_channels": 8}}) is not None


def test_the_espirit_window_is_checked_against_the_phase_encode_width() -> None:
    """Planted: a tall, narrow patch. The band runs along W; H used to be read."""
    planted = {
        "undersampling": {"center_fraction": 0.08},
        "data": {
            "sampling": {"patch_size": [320, 200, 1]},
            "processing": {
                "transforms": [
                    {"name": "espirit_sensitivity", "kwargs": {"keys": ["input"], "acs_size": 20}}
                ]
            },
        },
    }
    assert _espirit_reference_error(planted) is not None  # round(200 * 0.08) = 16 < 20


# ---------------------------------------------------------------------------
# The R4 arms on the m4raw route (#2509, #2382, #2327b, #2510): one route, one
# reference, one way of counting R, maps that never see the reference, and a
# selector each arm can actually emit.
# ---------------------------------------------------------------------------

#: The eight R4 arms: five migrated, three controls added with them.
_R4_ARMS = (
    "ei_unet_m4raw_r4",
    "ei_control_mc_only_m4raw_r4",
    "ei_multicoil_m4raw_r4",
    "ei_multicoil_control_m4raw_r4",
    "supervised_bound_mc_m4raw_r4",
    "robust_ssdu_m4raw_r4",
    "robust_ssdu_selfcal_m4raw_r4",
    "ssdu_m4raw_r4",
    "cs_l1wavelet_sense_m4raw_r4",
)
_R4_PATHS = [_COHORT / f"{name}.yaml" for name in _R4_ARMS]
#: Keys of the retired `kspace` route. `trajectory` + `undersampling` on this
#: route would mask the target too (conex_r2_complementary_m4raw.yaml's note).
_RETIRED_ROUTE_KEYS = ("trajectory", "caching", "datasets", "data_type", "augmentation")
_MANIFESTS = ("index_path", "validation_index_path", "test_index_path")
_CORRECTED_MSE = "reference_noise_corrected_mse"
#: Metrics that read the model's output on `input_alt` (`pred_alt`).
_PRED_ALT_METRICS = ("rep_variance", "bias_squared")
_NULL_SPACE_METRICS = ("null_space_residual", "null_space_residual_fraction")


def _masking_kwargs(cfg: dict[str, Any]) -> dict[str, Any] | None:
    for t in (((cfg.get("data") or {}).get("processing")) or {}).get("transforms") or []:
        if isinstance(t, dict) and t.get("name") == "complementary_rep_masking":
            return t.get("kwargs") or {}
    return None


def _route_error(cfg: dict[str, Any]) -> str | None:
    """Why an R4 arm is not on the m4raw route with R counted on the support, or None."""
    data = cfg.get("data") or {}
    if data.get("dataset_type") != "m4raw":
        return f"data.dataset_type={data.get('dataset_type')!r}, not m4raw"
    retired = [key for key in _RETIRED_ROUTE_KEYS if key in data]
    if retired or "undersampling" in cfg:
        return f"retired kspace-route keys declared: {retired or ['undersampling']}"
    expected = {
        "target_mode": "phase_aligned_mean",
        "val_target_mode": "phase_aligned_mean",
        "nex_target_exclude_input": True,
        "nex_fallback": "error",
        "nex_registration": "translation",
        "use_repetitions": True,
    }
    wrong = {k: data.get(k) for k, v in expected.items() if data.get(k) != v}
    if wrong:
        return f"not the leave-one-out reference: {wrong}"
    if ((data.get("coils") or {}).get("processing_mode")) != "none":
        return "data.coils.processing_mode must be declared none (the preset injects svd)"
    if _compression(cfg).get("method") not in (None, "none"):
        return f"physics compression {_compression(cfg).get('method')!r} on the m4raw route"
    masking = _masking_kwargs(cfg)
    if masking is None:
        return "no complementary_rep_masking: the input is not undersampled"
    if (masking.get("acceleration"), masking.get("center_fraction")) != (4.0, 0.094):
        return f"complementary_rep_masking {masking}: the cohort is R4 with a 0.094 ACS"
    source = data.get("source") or {}
    stale = [
        k
        for k in _MANIFESTS
        if not str(source.get(k, "")).endswith(("_nex3.json", "_nex3_slicechunked.json"))
    ]
    if stale:
        return f"{stale} are not the `_nex3` manifests: the cohort's one subject set"
    return None


_SCORING_STORE = "databases/m4raw/coil_maps"


def _map_leak_error(cfg: dict[str, Any]) -> str | None:
    """Why an arm's coil maps could carry the reference, or None."""
    data = cfg.get("data") or {}
    reference = data.get("coil_maps_reference_root")
    if reference is not None and reference == data.get("coil_maps_root"):
        return "coil_maps_reference_root is the training store: the scoring maps would train."
    if reference is not None and reference != _SCORING_STORE:
        return f"coil_maps_reference_root={reference!r}; every family scores on {_SCORING_STORE!r}"
    if data.get("coil_maps_root") is not None:
        if data.get("coil_maps_recipe") != "input_repetition":
            return (
                f"coil_maps_recipe={data.get('coil_maps_recipe')!r} (unset is "
                "all_reps_mean): maps averaged over every repetition contain the "
                "leave-one-out reference (#2510)."
            )
        masking = _masking_kwargs(cfg) or {}
        width = ((data.get("sampling") or {}).get("patch_size") or [256, 256])[1]
        band = round(width * float(masking.get("center_fraction", 0.0)))
        if int(data.get("coil_maps_acs_size") or 24) > band:
            return (
                f"coil_maps_acs_size={data.get('coil_maps_acs_size')} exceeds the "
                f"{band}-line ACS band: the stored maps read lines the masked input lacks."
            )
    return _espirit_reference_error(cfg)


def _selector_error(cfg: dict[str, Any]) -> str | None:
    """Multi-coil arms select on the corrected MSE; one virtual coil on val_mse.

    The reference's noise is independent of the output, so the MSE against it is
    about the true MSE plus a constant (the magnitude reference is Rician-biased, so
    not exactly); SSIM against it rewards a noisy output (the control's best
    checkpoint was step 2000 of 32000).
    """
    stop = cfg.get("early_stopping") or {}
    metrics = cfg.get("metrics") or {}
    if (stop.get("metric"), stop.get("mode")) != (
        metrics.get("best_metric_name"),
        metrics.get("best_metric_mode"),
    ):
        return f"early_stopping {stop.get('metric')} != metrics.best_metric_name"
    if _declares_maps(cfg):
        compute = ((cfg.get("validation") or {}).get("scoring") or {}).get("compute") or []
        if (stop.get("metric"), stop.get("mode")) != (f"val_{_CORRECTED_MSE}", "min"):
            return (
                f"selects on {stop.get('metric')!r}: an RSS image of the noisy reference "
                "is a target no S x_hat reaches (#2509); select on the corrected MSE."
            )
        if _CORRECTED_MSE not in compute:
            return f"validation.scoring.compute does not list {_CORRECTED_MSE}"
    elif (stop.get("metric"), stop.get("mode")) != ("val_mse", "min"):
        return "one virtual coil has no null space for Rep-BV: select on val_mse (min)"
    return None


def _pred_alt_error(cfg: dict[str, Any]) -> str | None:
    """The repetition metrics come with their second repetition, and an image
    network reports no null-space certificate (``S x_hat`` has none)."""
    compute = ((cfg.get("validation") or {}).get("scoring") or {}).get("compute") or []
    alternate = bool((cfg.get("data") or {}).get("emit_alternate_input"))
    wants = [m for m in compute if m in _PRED_ALT_METRICS]
    if wants and not alternate:
        return f"declares {wants} without data.emit_alternate_input"
    if alternate and not wants:
        return "emits input_alt that no declared metric reads"
    if _image_network(cfg):
        certificate = [m for m in compute if m in _NULL_SPACE_METRICS]
        if certificate:
            return f"an image-output arm declares {certificate}: S x_hat has no null-space component."
    return None


def test_every_r4_arm_is_present() -> None:
    missing = [p.stem for p in _R4_PATHS if not p.is_file()]
    assert not missing, f"missing R4 arms: {missing}"


@pytest.mark.parametrize("arm", _R4_PATHS, ids=_R4_ARMS)
def test_the_r4_arms_share_one_route(arm: Path) -> None:
    error = _route_error(_load(arm))
    assert error is None, f"{arm.stem}: {error}"


@pytest.mark.parametrize("arm", _R4_PATHS, ids=_R4_ARMS)
def test_no_coil_map_sees_the_reference(arm: Path) -> None:
    error = _map_leak_error(_load(arm))
    assert error is None, f"{arm.stem}: {error}"


@pytest.mark.parametrize("arm", _R4_PATHS, ids=_R4_ARMS)
def test_each_r4_arm_selects_on_what_it_can_emit(arm: Path) -> None:
    error = _selector_error(_load(arm))
    assert error is None, f"{arm.stem}: {error}"


_MAP_ARMS = [a for a in _ARMS if _declares_maps(_load(a))]


@pytest.mark.parametrize("arm", _MAP_ARMS, ids=[a.stem for a in _MAP_ARMS])
def test_every_multicoil_arm_selects_on_the_corrected_mse(arm: Path) -> None:
    """CoNEX and Rep-EI too: Rep-EI selected on val_psnr until 2026-10-09."""
    error = _selector_error(_load(arm))
    assert error is None, f"{arm.stem}: {error}"


def test_rep_ei_is_among_the_multicoil_arms() -> None:
    assert _REP_EI in _MAP_ARMS


@pytest.mark.parametrize("arm", _ARMS, ids=[a.stem for a in _ARMS])
def test_the_repetition_metrics_come_with_their_second_repetition(arm: Path) -> None:
    error = _pred_alt_error(_load(arm))
    assert error is None, f"{arm.stem}: {error}"


#: The four-coil EI and SSDU arms report the full decomposition since #2593.
_DECOMPOSED_R4 = [
    p for p in _R4_PATHS if p.stem not in ("ei_unet_m4raw_r4", "ei_control_mc_only_m4raw_r4")
]


_SCORED_ON_THE_STORE = [*_DECOMPOSED_R4, _REP_EI]


@pytest.mark.parametrize(
    "arm", _SCORED_ON_THE_STORE, ids=[p.stem for p in _SCORED_ON_THE_STORE]
)
def test_sense_columns_score_on_the_shared_store(arm: Path) -> None:
    """#2664: the arms whose training maps are not that store score on it."""
    assert (_load(arm).get("data") or {}).get("coil_maps_reference_root") == _SCORING_STORE


@pytest.mark.parametrize(
    ("root", "reason"),
    [
        ("databases/m4raw/coil_maps_input_rep_acs24", "coil_maps_reference_root is the training"),
        ("databases/m4raw/other_store", "coil_maps_reference_root="),
    ],
    ids=["the-training-store", "another-store"],
)
def test_the_scoring_store_predicate_rejects_a_planted_root(root: str, reason: str) -> None:
    cfg = _planted("data.coil_maps_reference_root", root)
    error = _map_leak_error(cfg)
    assert error is not None and error.startswith(reason), error


@pytest.mark.parametrize("arm", _DECOMPOSED_R4, ids=[p.stem for p in _DECOMPOSED_R4])
def test_the_four_coil_r4_arms_report_variance_and_bias(arm: Path) -> None:
    compute = (_load(arm).get("validation") or {}).get("scoring", {}).get("compute") or []
    assert {"rep_variance", "bias_squared"} <= set(compute), compute


#: (treatment, control) -> the only dotted paths they may differ on.
_ONE_KNOB_PAIRS = {
    ("ei_unet_m4raw_r4", "ei_control_mc_only_m4raw_r4"): {
        "training.equivariant_imaging.alpha_equivariance",
    },
    ("ei_multicoil_m4raw_r4", "ei_multicoil_control_m4raw_r4"): {
        "training.equivariant_imaging.alpha_equivariance",
    },
    # The CS baseline swaps the network for the solver and nothing else: its rows
    # pair with the EI arms' only while route, mask, maps and scoring are shared.
    ("cs_l1wavelet_sense_m4raw_r4", "ei_multicoil_m4raw_r4"): {
        "model.model_type",
        "model.model_kwargs.features",
        "model.model_kwargs.lambda_reg",
        "model.model_kwargs.num_iterations",
        "model.model_kwargs.wavelet",
        "model.model_kwargs.levels",
    },
    ("supervised_bound_mc_m4raw_r4", "ei_multicoil_control_m4raw_r4"): {
        "training.equivariant_imaging.measurement_target",
        "data.processing.transforms",
        "losses.kspace_losses",
        # The control's measured anchor scores the structural zeros per scored entry;
        # the bound's alternate-repetition anchor is masked_kspace_l2, which refuses both.
        "training.equivariant_imaging.anchor_mask",
        "training.equivariant_imaging.anchor_normalization",
        # Pins on the bound alone: its anchor resolves from the loss table.
        "losses.reconstruction.lambda_l1",
        "losses.reconstruction.warmup_losses",
    },
    ("robust_ssdu_m4raw_r4", "ssdu_m4raw_r4"): {
        "training.training_mode",
        "training.ssdu.noisier2noise_correction",
        "training.ssdu.alpha",
        "training.ssdu.noise_std_estimate",
        "training.ssdu.noise_model",
        # The injection's coil shape: read only by the Noisier2Noise step.
        "training.ssdu.noise_covariance",
        # Vanilla SSDU scores the structural zeros with the normalised L1-L2; Robust
        # SSDU scores all of Omega with Eq. 35 weights, which belong to the squared loss.
        "training.ssdu.score_structural_zeros",
        "losses.kspace_losses",
    },
}


@pytest.mark.parametrize("pair", list(_ONE_KNOB_PAIRS), ids=lambda p: f"{p[0]}-vs-{p[1]}")
def test_each_control_pair_differs_in_its_knob_only(pair: tuple[str, str]) -> None:
    a, b = (_load(_COHORT / f"{name}.yaml") for name in pair)
    assert _knob_diff(a, b) == _ONE_KNOB_PAIRS[pair]
    assert (a.get("metadata") or {}).get("group") == (b.get("metadata") or {}).get("group")


def test_the_supervised_bound_scores_every_line_against_the_reference() -> None:
    cfg = _load(_COHORT / "supervised_bound_mc_m4raw_r4.yaml")
    assert (_masking_kwargs(cfg) or {}).get("mode") == "full_target"
    assert _ei_block(cfg).get("measurement_target") == "alternate_repetition"
    assert not _ei_block(cfg).get("alpha_equivariance")
    names = [e.get("name") for e in (cfg.get("losses") or {}).get("kspace_losses") or []]
    assert names == ["masked_kspace_l2"]


# --- planted violations for the four R4 predicates (non-negotiable 15) ---


def _good_r4() -> dict[str, Any]:
    return _load(_COHORT / "ei_multicoil_m4raw_r4.yaml")


def _planted(path: str, value: Any) -> dict[str, Any]:
    """``_good_r4()`` with the dotted ``path`` set to ``value`` (None deletes it)."""
    import copy

    cfg = copy.deepcopy(_good_r4())
    *parents, leaf = path.split(".")
    node = cfg
    for key in parents:
        node = node.setdefault(key, {})
    if value is None:
        node.pop(leaf, None)
    else:
        node[leaf] = value
    return cfg


def _masking(**kwargs: Any) -> list[dict[str, Any]]:
    base = {"acceleration": 4.0, "center_fraction": 0.094, "mode": "same"}
    return [{"name": "complementary_rep_masking", "kwargs": {**base, **kwargs}}]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("data.trajectory", "cartesian"),
        ("undersampling", {"base_acceleration": 4.0}),
        ("data.processing.transforms", _masking(acceleration=2.0)),
        ("data.source.index_path", "data/manifests/m4raw_train.json"),
        ("physics.coil_processing.compression", {"method": "svd", "num_virtual_coils": 1}),
        ("data.dataset_type", "kspace"),
        ("data.nex_target_exclude_input", False),
        ("data.nex_registration", "none"),
        ("data.coils", None),
        ("data.processing.transforms", []),
    ],
    ids=[
        "trajectory",
        "undersampling",
        "r2",
        "not-nex3",
        "physics-svd",
        "kspace-route",
        "loo-off",
        "unregistered",
        "preset-svd",
        "unmasked",
    ],
)
def test_the_route_predicate_rejects_each_planted_shape(path: str, value: Any) -> None:
    assert _route_error(_good_r4()) is None
    assert _route_error(_planted(path, value)) is not None


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("data.coil_maps_recipe", None),
        ("data.coil_maps_recipe", "all_reps_mean"),
        ("data.coil_maps_acs_size", 28),
    ],
    ids=["all-reps-store-root", "all-reps-recipe", "window-too-wide"],
)
def test_the_map_leak_predicate_rejects_each_planted_shape(path: str, value: Any) -> None:
    assert _map_leak_error(_good_r4()) is None
    planted = _planted(path, value)
    if path.endswith("recipe") and value is None:
        planted["data"]["coil_maps_root"] = "databases/m4raw/coil_maps"
    assert _map_leak_error(planted) is not None


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("early_stopping", {"metric": "val_ssim", "mode": "max"}),
        ("validation.scoring.compute", ["psnr", "ssim"]),
        ("metrics.best_metric_name", "val_ssim"),
    ],
    ids=["ssim-selector", "metric-not-computed", "provenance-disagrees"],
)
def test_the_selector_predicate_rejects_each_planted_shape(path: str, value: Any) -> None:
    assert _selector_error(_good_r4()) is None
    assert _selector_error(_planted(path, value)) is not None


def test_the_selector_predicate_holds_the_virtual_coil_pair_to_mse() -> None:
    vc = _load(_COHORT / "ei_unet_m4raw_r4.yaml")
    assert _selector_error(vc) is None
    ssim = {**vc, "early_stopping": {"metric": "val_ssim", "mode": "max"},
            "metrics": {"best_metric_name": "val_ssim", "best_metric_mode": "max"}}
    assert _selector_error(ssim) is not None, "planted: SSIM against a noisy reference"
    vc["early_stopping"] = {"metric": f"val_{_CORRECTED_MSE}", "mode": "min"}
    vc["metrics"] = {"best_metric_name": f"val_{_CORRECTED_MSE}", "best_metric_mode": "min"}
    assert _selector_error(vc) is not None


@pytest.mark.parametrize(
    ("path", "value", "rejected"),
    [
        ("validation.scoring.compute", ["psnr", _CORRECTED_MSE, "null_space_residual"], True),
        ("data.emit_alternate_input", False, True),
        ("validation.scoring.compute", ["psnr", _CORRECTED_MSE], True),
        ("validation.scoring.compute", ["psnr", _CORRECTED_MSE, "rep_variance"], False),
    ],
    ids=["null-space", "metrics-without-input-alt", "input-alt-without-metrics", "coherent"],
)
def test_the_pred_alt_predicate_rejects_each_planted_shape(
    path: str, value: Any, rejected: bool
) -> None:
    assert (_pred_alt_error(_planted(path, value)) is not None) is rejected


def test_the_pred_alt_predicate_leaves_a_kspace_network_alone() -> None:
    """complex_unet reads `input_alt` itself, so CoNEX may declare rep_variance."""
    assert _pred_alt_error(_load(_COHORT / "conex_r2_samemask_m4raw.yaml")) is None


def test_the_one_knob_pins_see_a_planted_second_knob() -> None:
    pair = ("ei_multicoil_m4raw_r4", "ei_multicoil_control_m4raw_r4")
    a = _good_r4()
    b = _planted("training.equivariant_imaging.alpha_equivariance", 0.0)
    assert _knob_diff(a, b) == _ONE_KNOB_PAIRS[pair]
    b["data"]["coil_maps_acs_size"] = 28
    assert _knob_diff(a, b) != _ONE_KNOB_PAIRS[pair]


# ---------------------------------------------------------------------------
# Every m4raw arm is on the slice-level route at an iteration budget (#2673).
# The route makes an epoch every train slice (~2,304 batches at batch 2, ~1,152
# at batch 4) instead of 4 random slices per group, so a budget derived from
# `epochs` would grow 4.5x; each arm declares `max_iterations` instead.
# ---------------------------------------------------------------------------

_M4RAW_ARMS = [a for a in _ARMS if (_load(a).get("data") or {}).get("dataset_type") == "m4raw"]
#: The budget each family already ran: CoNEX and Rep-EI doubled with the n2n
#: cohort (100 group-route epochs of 512 batches); the R4 arms always declared it.
_ITERATIONS_R2 = 51_200
_ITERATIONS_R4 = 100_000
#: The EI alpha sweep around ``ei_multicoil_m4raw_r4`` runs at its sibling's budget.
_ALPHA_SWEEP = ("ei_multicoil_m4raw_r4_alpha01", "ei_multicoil_m4raw_r4_alpha10")
#: The trained iterative-CS arms, siblings of ``ei_multicoil_m4raw_r4``.
_CS_LEARNED = ("cs_pnp_unet_sense_m4raw_r4", "cs_lista_wavelet_sense_m4raw_r4")


def _slice_route_error(cfg: dict[str, Any], expected_iterations: int) -> str | None:
    """Why an m4raw arm is not on the slice route at its budget, or None."""
    data = cfg.get("data") or {}
    if data.get("slice_level_records") is not True:
        return f"data.slice_level_records is {data.get('slice_level_records')!r}"
    sampling = data.get("sampling") or {}
    if sampling.get("samples_per_volume") != 1:
        return f"samples_per_volume is {sampling.get('samples_per_volume')!r}"
    patch = list(sampling.get("patch_size") or [])
    if len(patch) != 3 or patch[2] != 1:
        return f"patch_size {patch!r} is not one slice deep"
    iterations = (cfg.get("training") or {}).get("max_iterations")
    if iterations != expected_iterations:
        return f"training.max_iterations is {iterations!r}, expected {expected_iterations}"
    return None


def _expected_iterations(arm: Path) -> int:
    """The sweep cells, CS cells included, inherit their R4 sibling's budget."""
    sweep = {name for name, *_ in (*_SWEEP, *_CS_SWEEP)} | set(_ALPHA_SWEEP) | set(_CS_LEARNED)
    return _ITERATIONS_R4 if arm.stem in _R4_ARMS or arm.stem in sweep else _ITERATIONS_R2


def test_the_m4raw_arms_are_the_thirteen_plus_the_cs_baseline_plus_the_sweep() -> None:
    """Thirteen moved for #2673, the CS baseline that arrived on it, and the sweeps."""
    expected = 14 + len(_SWEEP) + len(_CS_SWEEP) + len(_ALPHA_SWEEP) + len(_CS_LEARNED)
    assert len(_M4RAW_ARMS) == expected, [a.stem for a in _M4RAW_ARMS]


@pytest.mark.parametrize("arm", _M4RAW_ARMS, ids=[_arm_id(a) for a in _M4RAW_ARMS])
def test_every_m4raw_arm_is_on_the_slice_route_at_its_budget(arm: Path) -> None:
    error = _slice_route_error(_load(arm), _expected_iterations(arm))
    assert error is None, f"{arm.stem}: {error}"


@pytest.mark.parametrize(
    ("arm", "path", "value", "reason"),
    [
        ("conex_r2_samemask_m4raw", "data.slice_level_records", None, "is None"),
        ("conex_r2_samemask_m4raw", "data.slice_level_records", False, "is False"),
        ("conex_r2_samemask_m4raw", "data.sampling.samples_per_volume", 4, "is 4"),
        ("conex_r2_samemask_m4raw", "data.sampling.patch_size", [256, 256, 3], "deep"),
        ("conex_r2_samemask_m4raw", "training.max_iterations", None, "is None"),
        ("rep_ei_m4raw_r2", "training.max_iterations", 25_600, "is 25600"),
        ("ssdu_m4raw_r4", "training.max_iterations", 51_200, "expected 100000"),
    ],
    ids=["flag-absent", "flag-off", "four-per-record", "deep-patch", "epoch-budget", "v1", "r4"],
)
def test_the_slice_route_predicate_rejects_each_planted_shape(
    arm: str, path: str, value: Any, reason: str
) -> None:
    import copy

    source = _COHORT / f"{arm}.yaml"
    cfg = copy.deepcopy(_load(source))
    assert _slice_route_error(cfg, _expected_iterations(source)) is None
    *parents, leaf = path.split(".")
    node = cfg
    for key in parents:
        node = node.setdefault(key, {})
    if value is None:
        node.pop(leaf, None)
    else:
        node[leaf] = value
    error = _slice_route_error(cfg, _expected_iterations(source))
    assert error is not None and reason in error, error


# ---------------------------------------------------------------------------
# SENSE-combined PSNR/SSIM (n2n PAPER.md 8): extra columns on every arm whose
# batch carries complex maps of the physical coils, and on no other. One virtual
# coil has no maps, so the columns would raise there on the first validation.
# ---------------------------------------------------------------------------

_SENSE_COLUMNS = ("sense_psnr", "sense_ssim")
#: The one-virtual-coil pair keeps RSS only: nothing to combine, no maps.
_RSS_ONLY_ARMS = ("ei_unet_m4raw_r4", "ei_control_mc_only_m4raw_r4")


def _validation_compute(cfg: dict[str, Any]) -> list[str]:
    """``validation.scoring.compute``, or ``metrics.compute`` it inherits when unset."""
    scoring = ((cfg.get("validation") or {}).get("scoring") or {}).get("compute")
    return list(scoring if scoring is not None else (cfg.get("metrics") or {}).get("compute") or [])


def _sense_columns_error(cfg: dict[str, Any]) -> str | None:
    """Why an arm's SENSE columns are missing or cannot be scored, or None."""
    compute = _validation_compute(cfg)
    declared = [m for m in _SENSE_COLUMNS if m in compute]
    compressed = _compression(cfg).get("method") not in (None, "none")
    if not _declares_maps(cfg) or compressed:
        if declared:
            return (
                f"declares {declared} without complex maps of the physical coils "
                "(no coil_maps_root / espirit_sensitivity, or coils compressed): "
                "the SENSE combine has nothing to combine and raises."
            )
        return None
    missing = [m for m in _SENSE_COLUMNS if m not in compute]
    return f"validation.scoring.compute lacks {missing}" if missing else None


@pytest.mark.skipif(not _ARMS, reason="self_supervised cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_every_mapped_arm_reports_the_sense_columns(arm: Path) -> None:
    error = _sense_columns_error(_load(arm))
    assert error is None, f"{_arm_id(arm)}: {error}"


def test_the_one_virtual_coil_arms_keep_rss_only() -> None:
    for name in _RSS_ONLY_ARMS:
        compute = _validation_compute(_load(_COHORT / f"{name}.yaml"))
        assert "psnr" in compute and not set(_SENSE_COLUMNS) & set(compute), name


@pytest.mark.parametrize(
    ("arm", "compute", "rejected"),
    [
        ("ei_unet_m4raw_r4", ["psnr", "ssim", "sense_psnr"], True),
        ("ei_multicoil_m4raw_r4", ["psnr", "ssim", _CORRECTED_MSE], True),
        ("ei_multicoil_m4raw_r4", ["psnr", "ssim", "sense_psnr", _CORRECTED_MSE], True),
        ("ei_unet_m4raw_r4", ["psnr", "ssim"], False),
    ],
    ids=["single-coil-declares", "mapped-omits", "mapped-half", "single-coil-rss"],
)
def test_the_sense_predicate_rejects_each_planted_shape(
    arm: str, compute: list[str], rejected: bool
) -> None:
    import copy

    cfg = copy.deepcopy(_load(_COHORT / f"{arm}.yaml"))
    cfg["validation"]["scoring"]["compute"] = compute
    assert (_sense_columns_error(cfg) is not None) is rejected


def _uncomputed_report_metrics(cfg: dict[str, Any]) -> set[str]:
    """``reporting.metrics`` the run never computes: the main-results table leaves them empty."""
    reported = set((cfg.get("reporting") or {}).get("metrics") or [])
    computed = set(((cfg.get("validation") or {}).get("scoring") or {}).get("compute") or [])
    computed |= set((cfg.get("metrics") or {}).get("compute") or [])
    return reported - computed


@pytest.mark.parametrize("arm", _ARMS, ids=[a.stem for a in _ARMS])
def test_every_reported_metric_is_computed(arm: Path) -> None:
    missing = _uncomputed_report_metrics(_load(arm))
    assert not missing, f"{arm.stem}: reporting.metrics names {sorted(missing)}, never computed"


@pytest.mark.parametrize(
    ("cfg", "rejected"),
    [
        ({"reporting": {"metrics": ["rmse"]}, "validation": {"scoring": {"compute": ["mse"]}}}, True),
        ({"reporting": {"metrics": ["nmse"]}, "metrics": {"compute": ["psnr"]}}, True),
        ({"reporting": {"metrics": ["mse"]}, "metrics": {"compute": ["mse"]}}, False),
    ],
    ids=["scoring-lacks-it", "metrics-lacks-it", "computed"],
)
def test_the_report_predicate_rejects_a_planted_metric(cfg: dict, rejected: bool) -> None:
    assert bool(_uncomputed_report_metrics(cfg)) is rejected


# ---------------------------------------------------------------------------
# Registered repetitions (#2849, #2882): every arm whose reference is built from
# more than one repetition registers them onto the input first, so the R4 arms,
# CoNEX and Rep-EI grade against references with the same motion treatment.
# ---------------------------------------------------------------------------

#: Target modes that combine repetitions, and therefore read nex_registration.
_MULTI_REP_TARGETS = ("complex_mean", "phase_aligned_mean", "rep_pair")


def _registration_error(cfg: dict[str, Any]) -> str | None:
    """Why an arm's multi-repetition reference is averaged unregistered, or None."""
    data = cfg.get("data") or {}
    if data.get("dataset_type") != "m4raw" or data.get("use_repetitions") is False:
        return None
    if data.get("target_mode", "complex_mean") not in _MULTI_REP_TARGETS:
        return None
    if data.get("nex_registration") != "translation":
        return (
            f"data.nex_registration={data.get('nex_registration')!r}: the repetitions "
            "are averaged as acquired, and 28 of 60 M4Raw groups move more than 0.5 px "
            "between them (#2849)."
        )
    return None


@pytest.mark.skipif(not _ARMS, reason="self_supervised cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_every_multi_repetition_reference_is_registered(arm: Path) -> None:
    error = _registration_error(_load(arm))
    assert error is None, f"{arm.stem}: {error}"


@pytest.mark.parametrize(
    ("path", "value"),
    [("data.nex_registration", "none"), ("data.nex_registration", None)],
    ids=["declared-none", "undeclared"],
)
def test_the_registration_predicate_rejects_each_planted_shape(path: str, value: Any) -> None:
    rep_pair = _load(_COHORT / "conex_r2_samemask_m4raw.yaml")
    assert _registration_error(_good_r4()) is None
    assert _registration_error(rep_pair) is None
    assert _registration_error(_planted(path, value)) is not None
    rep_pair["data"].pop("nex_registration", None)
    assert _registration_error(rep_pair) is not None


# ---------------------------------------------------------------------------
# The acceleration sweep (PAPER.md, "Acceleration sweep"): each cell is a copy of
# its R4 sibling that differs in the masking transform's pattern and
# acceleration only, plus identity paths. Cartesian R8 and R16 are absent on
# purpose: 24 and 12 lines on ~195 supported columns cannot hold the 24-line ACS
# that the ESPIRiT window (coil_maps_acs_size: 24) needs, and the draw raises. A
# 20-line window dropped up to 62% of the T1 anatomy from the maps' support.
# ---------------------------------------------------------------------------

_SWEEP_SIBLINGS = {"ei_multicoil_m4raw": "ei_multicoil_m4raw_r4"}
_SWEEP_SIBLINGS["supervised_bound_mc_m4raw"] = "supervised_bound_mc_m4raw_r4"
_SWEEP_CELLS = (
    ("poisson", "poisson_disk", 4.0),
    ("poisson", "poisson_disk", 8.0),
    ("poisson", "poisson_disk", 16.0),
    ("radial", "radial", 4.0),
    ("radial", "radial", 8.0),
    ("radial", "radial", 16.0),
)
_SWEEP = [
    (f"{prefix}_{tag}_r{int(r)}", sibling, pattern, r)
    for prefix, sibling in _SWEEP_SIBLINGS.items()
    for tag, pattern, r in _SWEEP_CELLS
]
_SWEEP_CAMPAIGN = (
    _COHORT.parents[2] / "experiments" / "campaigns" / "self_supervised_acceleration_sweep_m4raw.yaml"
)
#: M4Raw's acquired share of the phase-encode axis (partial FOV), as measured
#: on the coil-map store (pe_support_fraction ~0.76: 195 of 256 columns).
_M4RAW_SUPPORT = 195


#: The CS baseline at every sweep cell, plus spiral, where only CS runs. A CS
#: cell is cs_l1wavelet_sense_m4raw_r4 with the mask changed, graded by
#: --validate-only, so it is not in the sweep campaign (whose array runs train).
_CS_SWEEP = [
    (f"cs_l1wavelet_sense_m4raw_{tag}_r{int(r)}", "cs_l1wavelet_sense_m4raw_r4", pattern, r)
    for tag, pattern, r in (*_SWEEP_CELLS, ("spiral", "spiral", 4.0), ("spiral", "spiral", 8.0), ("spiral", "spiral", 16.0))
]


def _sweep_diff_error(cell: dict[str, Any], sibling: dict[str, Any], pattern: str, r: float) -> str | None:
    """Why a sweep cell is not its sibling with only the mask changed, or None.

    A 2-D mask keeps a calibration window, not whole lines, so its scale is
    measured on that window (``acs_window``) and it needs no phase-encode axis;
    those follow from the mask and are the only other knobs allowed to move.
    """
    diff = _knob_diff(cell, sibling)
    follows_mask = set()
    if pattern != "cartesian_1d":
        follows_mask = {p for p in diff if p.startswith("data.acquisition")}
        follows_mask.add("data.processing.kspace_scale_domain")
    if diff - follows_mask != {"data.processing.transforms"}:
        return f"differs from its sibling in {sorted(diff - follows_mask)}"
    domain = cell["data"]["processing"].get("kspace_scale_domain")
    if domain != ("acs" if pattern == "cartesian_1d" else "acs_window"):
        return f"a {pattern} mask with kspace_scale_domain {domain!r}"
    ours, theirs = (c["data"]["processing"]["transforms"] for c in (cell, sibling))
    rest = [t for t in ours if t.get("name") != "complementary_rep_masking"]
    if rest != [t for t in theirs if t.get("name") != "complementary_rep_masking"]:
        return "a transform other than complementary_rep_masking changed"
    mine, base = _masking_kwargs(cell) or {}, _masking_kwargs(sibling) or {}
    changed = {k for k in set(mine) | set(base) if mine.get(k) != base.get(k)}
    if not changed <= {"acceleration", "pattern"}:
        return f"masking kwargs {sorted(changed - {'acceleration', 'pattern'})} changed"
    if (mine.get("pattern"), mine.get("acceleration")) != (pattern, r):
        return f"masking declares {mine.get('pattern')} R{mine.get('acceleration')}, cell is {pattern} R{r}"
    return None


def _budget_error(cfg: dict[str, Any]) -> str | None:
    """Why an arm's calibration region fills its R budget on M4Raw's support, or None."""
    masking = _masking_kwargs(cfg)
    if masking is None:
        return None
    height, width = ((cfg.get("data") or {}).get("sampling") or {}).get("patch_size", [256, 256])[:2]
    r = float(masking.get("acceleration", 2.0))
    n = round(width * float(masking.get("center_fraction", 0.08)))
    if masking.get("pattern", "cartesian_1d") == "cartesian_1d":
        budget, calib = round(_M4RAW_SUPPORT / r), n
    else:
        budget, calib = round(height * _M4RAW_SUPPORT / r), n * n
    if calib >= budget:
        return f"the {calib}-bin calibration region fills the {budget}-bin budget of R{r:g}"
    return None


def test_every_sweep_cell_is_present() -> None:
    missing = [name for name, *_ in _SWEEP if not (_COHORT / f"{name}.yaml").is_file()]
    assert not missing, f"missing sweep cells: {missing}"


@pytest.mark.parametrize(("name", "sibling", "pattern", "r"), _SWEEP, ids=[s[0] for s in _SWEEP])
def test_each_sweep_cell_differs_from_its_sibling_in_the_mask_only(
    name: str, sibling: str, pattern: str, r: float
) -> None:
    cell = _load(_COHORT / f"{name}.yaml")
    error = _sweep_diff_error(cell, _load(_COHORT / f"{sibling}.yaml"), pattern, r)
    assert error is None, f"{name}: {error}"
    assert cell["workflow"] == _load(_COHORT / f"{sibling}.yaml")["workflow"]
    assert cell["config_version"] == "1.0"


@pytest.mark.parametrize(("name", "sibling", "pattern", "r"), _SWEEP, ids=[s[0] for s in _SWEEP])
def test_each_sweep_cell_pairs_with_its_own_bound(name: str, sibling: str, pattern: str, r: float) -> None:
    cell = _load(_COHORT / f"{name}.yaml")
    partner = name.replace("ei_multicoil_m4raw", "supervised_bound_mc_m4raw")
    if partner == name:
        partner = name.replace("supervised_bound_mc_m4raw", "ei_multicoil_m4raw")
    other = _load(_COHORT / f"{partner}.yaml")
    assert cell["metadata"]["group"] == other["metadata"]["group"]
    assert cell["metadata"]["group"] != "ei_multicoil_ablation_m4raw_r4"


@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_no_calibration_region_fills_its_budget(arm: Path) -> None:
    error = _budget_error(_load(arm))
    assert error is None, f"{arm.stem}: {error}"


def test_the_campaign_lists_every_sweep_cell() -> None:
    campaign = _load(_SWEEP_CAMPAIGN)
    listed = {Path(e["config"]).stem for e in campaign["experiments"]}
    assert listed == {name for name, *_ in _SWEEP}


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"acceleration": 16.0}, "masking declares"),  # R16 in a cell named R8
        ({"center_fraction": 0.04}, "center_fraction"),
        ({"mode": "independent"}, "mode"),
    ],
    ids=["wrong-r", "acs-moved", "mode-moved"],
)
def test_the_sweep_diff_predicate_rejects_each_planted_shape(change: dict, reason: str) -> None:
    import copy

    sibling = _load(_COHORT / "ei_multicoil_m4raw_r4.yaml")
    cell = _load(_COHORT / "ei_multicoil_m4raw_poisson_r8.yaml")
    assert _sweep_diff_error(cell, sibling, "poisson_disk", 8.0) is None
    planted = copy.deepcopy(cell)
    planted["data"]["processing"]["transforms"][0]["kwargs"].update(change)
    error = _sweep_diff_error(planted, sibling, "poisson_disk", 8.0)
    assert error is not None and reason in error, error
    second_knob = copy.deepcopy(cell)
    second_knob["data"]["coil_maps_acs_size"] = 16
    assert _sweep_diff_error(second_knob, sibling, "poisson_disk", 8.0) is not None


def test_a_2d_cell_must_measure_its_scale_on_the_window() -> None:
    """Planted: a Poisson cell left on the line band scales on lines its mask lacks."""
    import copy

    sibling = _load(_COHORT / "ei_multicoil_m4raw_r4.yaml")
    cell = _load(_COHORT / "ei_multicoil_m4raw_poisson_r8.yaml")
    assert _sweep_diff_error(cell, sibling, "poisson_disk", 8.0) is None
    planted = copy.deepcopy(cell)
    planted["data"]["processing"]["kspace_scale_domain"] = "acs"
    assert "kspace_scale_domain" in (_sweep_diff_error(planted, sibling, "poisson_disk", 8.0) or "")


@pytest.mark.parametrize(
    ("pattern", "r", "rejected"),
    [
        ("cartesian_1d", 16.0, True),
        ("cartesian_1d", 8.0, True),
        ("cartesian_1d", 4.0, False),
        ("poisson_disk", 16.0, False),
    ],
    ids=["cartesian-r16", "cartesian-r8", "cartesian-r4", "poisson-r16"],
)
def test_the_budget_predicate_rejects_cartesian_r8_and_r16(pattern: str, r: float, rejected: bool) -> None:
    cfg = _load(_COHORT / "ei_multicoil_m4raw_poisson_r8.yaml")
    cfg["data"]["processing"]["transforms"][0]["kwargs"].update(pattern=pattern, acceleration=r)
    assert (_budget_error(cfg) is not None) is rejected


@pytest.mark.parametrize(("name", "sibling", "pattern", "r"), _CS_SWEEP, ids=[s[0] for s in _CS_SWEEP])
def test_each_cs_sweep_cell_differs_from_the_r4_cs_arm_in_the_mask_only(
    name: str, sibling: str, pattern: str, r: float
) -> None:
    cell = _load(_COHORT / f"{name}.yaml")
    error = _sweep_diff_error(cell, _load(_COHORT / f"{sibling}.yaml"), pattern, r)
    assert error is None, f"{name}: {error}"
    assert cell["workflow"] == _load(_COHORT / f"{sibling}.yaml")["workflow"]
    assert cell["config_version"] == "1.0"


@pytest.mark.parametrize(("name", "sibling", "pattern", "r"), _CS_SWEEP, ids=[s[0] for s in _CS_SWEEP])
def test_each_cs_sweep_cell_pairs_with_the_learned_arms_of_its_cell(
    name: str, sibling: str, pattern: str, r: float
) -> None:
    group = _load(_COHORT / f"{name}.yaml")["metadata"]["group"]
    learned = name.replace("cs_l1wavelet_sense_m4raw", "ei_multicoil_m4raw")
    if pattern == "spiral":
        assert not (_COHORT / f"{learned}.yaml").is_file()
        assert group == f"cs_sweep_m4raw_spiral_r{int(r)}"
    else:
        assert group == _load(_COHORT / f"{learned}.yaml")["metadata"]["group"]


def test_no_cs_cell_is_in_the_sweep_campaign() -> None:
    listed = {Path(e["config"]).stem for e in _load(_SWEEP_CAMPAIGN)["experiments"]}
    assert not listed & {name for name, *_ in _CS_SWEEP}


_SHIFT_STORE = "databases/m4raw/nex_shifts"


def _shift_store_error(cfg: dict[str, Any]) -> str | None:
    """Why ``cfg`` estimates its registration shifts in every loader worker, or None."""
    data = cfg.get("data") or {}
    if data.get("nex_registration", "none") == "none":
        return None
    if data.get("nex_shift_root") != _SHIFT_STORE:
        return f"data.nex_shift_root is {data.get('nex_shift_root')!r}, not {_SHIFT_STORE!r}"
    if data.get("nex_shifts_missing", "raise") != "raise":
        return "data.nex_shifts_missing must stay 'raise': an estimate per worker is the #2977 stall"
    return None


@pytest.mark.skipif(not _ARMS, reason="self_supervised cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_registered_arm_reads_the_shift_store(arm: Path) -> None:
    error = _shift_store_error(_load(arm))
    assert error is None, f"{_arm_id(arm)}: {error}"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"nex_registration": "translation"}, "nex_shift_root"),
        ({"nex_registration": "translation", "nex_shift_root": "elsewhere"}, "nex_shift_root"),
        (
            {
                "nex_registration": "translation",
                "nex_shift_root": _SHIFT_STORE,
                "nex_shifts_missing": "estimate",
            },
            "nex_shifts_missing",
        ),
    ],
    ids=["no-store", "wrong-store", "estimate-on-miss"],
)
def test_shift_store_predicate_rejects_planted_violation(data: dict, expected: str) -> None:
    error = _shift_store_error({"data": data})
    assert error is not None and expected in error


_SLICECHUNKED_ROOT = "databases/m4raw/data_slicechunked"
_SLICECHUNKED_KEYS = ("index_path", "validation_index_path", "test_index_path")


def _slicechunked_error(cfg: dict[str, Any]) -> str | None:
    """Why ``cfg`` reads the released one-chunk-per-column M4Raw files, or None.

    A slice read from the released layout decodes the whole volume, and the loader reads the
    manifest's ``data_root``, not ``data.source.root``, so the root and the three manifests
    must name the rechunked tree together.
    """
    source = (cfg.get("data") or {}).get("source") or {}
    if source.get("root") != _SLICECHUNKED_ROOT:
        return f"data.source.root is {source.get('root')!r}, not {_SLICECHUNKED_ROOT!r}"
    for key in _SLICECHUNKED_KEYS:
        if not str(source.get(key, "")).endswith("_slicechunked.json"):
            return f"data.source.{key} is {source.get(key)!r}, not a *_slicechunked.json manifest"
    return None


@pytest.mark.skipif(not _ARMS, reason="self_supervised cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_arm_reads_the_slice_chunked_tree(arm: Path) -> None:
    error = _slicechunked_error(_load(arm))
    assert error is None, f"{_arm_id(arm)}: {error}"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ({"root": "databases/m4raw/data"}, "source.root"),
        ({"root": _SLICECHUNKED_ROOT, "index_path": "m_nex3.json"}, "index_path"),
        (
            {
                "root": _SLICECHUNKED_ROOT,
                "index_path": "a_slicechunked.json",
                "validation_index_path": "b_slicechunked.json",
                "test_index_path": "c_nex3.json",
            },
            "test_index_path",
        ),
    ],
    ids=["released-root", "released-train-manifest", "released-test-manifest"],
)
def test_slice_chunked_predicate_rejects_planted_violation(source: dict, expected: str) -> None:
    error = _slicechunked_error({"data": {"source": source}})
    assert error is not None and expected in error


def _scale_window_error(cfg: dict[str, Any]) -> str | None:
    """Why an arm's scale is not measured inside the region every one of its masks keeps."""
    processing = (cfg.get("data") or {}).get("processing") or {}
    domain = processing.get("kspace_scale_domain")
    masking = _masking_kwargs(cfg)
    if domain not in ("acs", "acs_window") or masking is None:
        return None
    scale = float(processing.get("log_scaling_center_fraction", 0.25))
    mask = float(masking.get("center_fraction", 0.08))
    if scale > mask:
        return f"the scale reads a {scale:g} calibration region, the masks keep only {mask:g}"
    return None


@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_the_scale_is_measured_inside_what_every_mask_keeps(arm: Path) -> None:
    error = _scale_window_error(_load(arm))
    assert error is None, f"{arm.stem}: {error}"


def test_the_scale_window_predicate_rejects_a_wider_region() -> None:
    """Planted: the schema default 0.25 would scale on lines the input never has."""
    cfg = _load(_COHORT / "ssdu_m4raw_r4.yaml")
    assert _scale_window_error(cfg) is None
    cfg["data"]["processing"]["log_scaling_center_fraction"] = 0.25
    assert _scale_window_error(cfg) is not None
