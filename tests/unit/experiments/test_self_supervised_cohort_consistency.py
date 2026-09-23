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
    """Removing the status must not lose the caveat it carried.

    ``noise_std_estimate`` sits far below the measured M4Raw sigma, so the
    Noisier2Noise correction is numerically inert and the arm trains as vanilla
    SSDU. That has to stay visible or the arm gets reported as Robust SSDU.
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
    names = {
        t.get("name")
        for t in (((cfg.get("data") or {}).get("processing")) or {}).get("transforms") or []
        if isinstance(t, dict)
    }
    if "espirit_sensitivity" not in names:
        return (
            "multicoil_operator=true without the `espirit_sensitivity` transform: "
            "nothing else in the loader produces coil maps, so the strategy raises."
        )
    return None


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
    ],
    ids=["knob-off", "maps-missing", "coils-compressed", "coherent"],
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
    names = {
        t.get("name")
        for t in (((cfg.get("data") or {}).get("processing")) or {}).get("transforms") or []
        if isinstance(t, dict)
    }
    if "espirit_sensitivity" not in names:
        return (
            "noise_std_source: self_calibrated without the `espirit_sensitivity` "
            "transform: nothing else produces the coil maps the estimate needs."
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
def test_the_declared_sibling_keeps_its_gate_and_its_constant() -> None:
    """It is the control, and its treatment really is inert; both must stay."""
    cfg = yaml.safe_load(_SSDU_DECLARED.read_text()) or {}
    assert _ssdu_block(cfg).get("noise_std_source") in (None, "declared")
    assert (cfg.get("metadata") or {}).get("status") == "needs_implementation"


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
    ],
    ids=["declared", "correction-off", "stale-constant", "coils-compressed", "coherent"],
)
def test_selfcal_sigma_predicate_rejects_each_planted_shape(planted: dict, rejected: bool) -> None:
    assert (_selfcal_sigma_error(planted) is not None) is rejected
