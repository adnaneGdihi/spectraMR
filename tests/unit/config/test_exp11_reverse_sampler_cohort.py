"""Regression: every exp11 kspace_filling arm declares the reverse loop it samples with.

Two properties keep the cohort's comparisons valid:

1. **No cold-diffusion arm inherits a reverse loop by omission.** Every
   ``kspace_cold_diffusion`` arm sets ``model.model_kwargs.reverse_sampling_mode``
   to a registered mode and, on a bounded loop, a positive ``reverse_clip_ratio``
   (``additive`` reads none, #2428). The constructor
   default is ``additive`` (Bansal Algorithm 2, unbounded), so an unset knob
   silently selects it; an arm may run ``additive`` only as the declared
   reproduction baseline (``metadata.tags.reverse: additive``).

2. **Per-sub-cohort uniformity (non-negotiable 17).** Arms compared against each
   other live in one directory and share one ``(mode, clip_ratio, decay_envelope)``
   triple, so a shootout is not confounded by its reverse loop. An arm whose
   variable *is* the reverse loop declares it in ``metadata.tags.reverse`` and is
   compared against its sibling explicitly, so it is exempt from the uniformity check.

3. **The freeze loop runs under the decay envelope that matches its mask.** Every
   undeclared ``replace_freeze_dc`` arm of the three ``kspace_filling*`` cohorts
   and of ``exp_11_variant`` sets ``reverse_decay_envelope: per_sample``, or
   ``per_sample_radial`` when its mask samples 2-D points (random, radial,
   spiral): a row envelope fits the rows such a mask only half measures, and
   left loud inventions at 1.6-2.1x where the shell envelope holds 1.24-1.51x
   (backlog A5.4.1). Without an envelope the 1.3 ``band_local``
   ceiling passes an invented loud texture at 2.8-2.9x the true unmeasured
   energy, and F holds it to about 1x
   (``docs/explanation/kspace_reverse_processes.md``, "Invented content"). The
   ``ablations/reverse`` control declares ``metadata.tags.reverse`` and stays
   without F, because it is the loop F is measured against.

Arms of the upstream baselines (``UpstreamProcessStrategy``) do not use the
cold-MRI sampler and are out of scope. The arms are read with ``yaml.safe_load``
and no model is built; the registered modes come from ``kspace_process``. The
planted configs at the bottom are the shapes each rule must reject.
"""

from __future__ import annotations

import collections
from pathlib import Path

import pytest
import yaml

from tests.utils.corpus import tracked_yamls
from tests.utils.repo_scripts import skip_if_public_export

_COHORT_DIR = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "inprogress"
    / "kspace_filling"
)

#: The mode the constructor falls back to, allowed only as a declared baseline.
_UNBOUNDED_DEFAULT = "additive"


def _cohort_yamls() -> list[Path]:
    return tracked_yamls(_COHORT_DIR)


def _ids(paths: list[Path]) -> list[str]:
    return [p.relative_to(_COHORT_DIR).as_posix() for p in paths]


def _sub_cohort(path: Path) -> str:
    """Comparison group = the immediate parent dir under kspace_filling.

    Loose top-level variant files (sampling-pattern family) share the ``.``
    group; each sub-directory is its own group.
    """
    return path.relative_to(_COHORT_DIR).parent.as_posix()


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _reverse_knobs(cfg: dict) -> tuple[object, object, object]:
    model_kwargs = (cfg.get("model") or {}).get("model_kwargs") or {}
    return (
        model_kwargs.get("reverse_sampling_mode"),
        model_kwargs.get("reverse_clip_ratio"),
        model_kwargs.get("reverse_decay_envelope"),
    )


#: Mask geometries whose envelope is fitted over shells of radial |k|. Variable
#: density and gaussian stay on rows until their noise estimate is fixed (#2475).
_RADIAL_ENVELOPE_GEOMETRIES = frozenset({"random", "radial", "spiral"})


def _expected_envelope(cfg: dict) -> str:
    geometry = (cfg.get("undersampling") or {}).get("acceleration_type")
    return "per_sample_radial" if geometry in _RADIAL_ENVELOPE_GEOMETRIES else "per_sample"


def _uniformity_key(cfg: dict) -> tuple[object, object, object]:
    """The reverse knobs, with an envelope that follows the arm's own mask read as one value."""
    mode, ratio, envelope = _reverse_knobs(cfg)
    if envelope is not None and envelope == _expected_envelope(cfg):
        envelope = "geometry-matched"
    return mode, ratio, envelope


def _declared_reverse_variable(cfg: dict) -> object:
    tags = (cfg.get("metadata") or {}).get("tags")
    return tags.get("reverse") if isinstance(tags, dict) else None


def _uses_cold_sampler(cfg: dict) -> bool:
    return (cfg.get("model") or {}).get("model_type") == "kspace_cold_diffusion"


def _mode_problem(cfg: dict) -> str | None:
    """Why this arm's reverse declaration is refused, or ``None``."""
    from spectramr.models.diffusion.kspace_process import VALID_REVERSE_MODES
    from spectramr.models.diffusion.reverse_physics import resolve_reverse_physics_kwargs

    mode, ratio, _ = _reverse_knobs(cfg)
    if mode not in VALID_REVERSE_MODES:
        return (
            f"reverse_sampling_mode={mode!r} is unset or unregistered; an unset knob "
            f"silently selects {_UNBOUNDED_DEFAULT!r}. Registered: {sorted(VALID_REVERSE_MODES)}."
        )
    if mode == _UNBOUNDED_DEFAULT and _declared_reverse_variable(cfg) != _UNBOUNDED_DEFAULT:
        return (
            f"reverse_sampling_mode={mode!r} is allowed only on the declared "
            f"reproduction baseline (metadata.tags.reverse: {_UNBOUNDED_DEFAULT})."
        )
    if mode == _UNBOUNDED_DEFAULT:
        # The bound-free loop takes no ratio; the resolver the build calls owns that refusal.
        try:
            resolve_reverse_physics_kwargs(mode, "hard", (cfg.get("model") or {})["model_kwargs"])
        except ValueError as exc:
            return str(exc)
        return None
    if not (isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and ratio > 0):
        return f"reverse_clip_ratio={ratio!r} must be a positive number."
    return None


def _mixed_groups(entries: list[tuple[str, dict]]) -> dict[str, set]:
    """Comparison groups whose non-ablation arms disagree on the reverse loop."""
    groups: dict[str, set] = collections.defaultdict(set)
    for group, cfg in entries:
        if _declared_reverse_variable(cfg) is None:
            groups[group].add(_uniformity_key(cfg))
    return {g: settings for g, settings in groups.items() if len(settings) > 1}


_YAMLS = [p for p in _cohort_yamls() if _uses_cold_sampler(_load(p))]

#: The freeze loop that must run under the decay envelope.
_ENVELOPED_MODE = "replace_freeze_dc"

_ENVELOPE_COHORT_DIRS = [
    _COHORT_DIR.parent / name
    for name in (
        "kspace_filling",
        "kspace_filling_fastmri_brain",
        "kspace_filling_fastmri_knee",
        "exp_11_variant",
    )
]
_ENVELOPE_YAMLS = [
    p
    for d in _ENVELOPE_COHORT_DIRS
    for p in tracked_yamls(d)
    if _uses_cold_sampler(_load(p))
    and _reverse_knobs(_load(p))[0] == _ENVELOPED_MODE
    and _declared_reverse_variable(_load(p)) is None
]


def _envelope_problem(cfg: dict) -> str | None:
    """Why an undeclared freeze-loop arm is refused for its decay envelope, or ``None``."""
    mode, _, envelope = _reverse_knobs(cfg)
    if mode != _ENVELOPED_MODE or _declared_reverse_variable(cfg) is not None:
        return None
    expected = _expected_envelope(cfg)
    if envelope != expected:
        return (
            f"reverse_decay_envelope={envelope!r}; an undeclared {_ENVELOPED_MODE!r} arm on this "
            f"mask runs under {expected!r}, or declares metadata.tags.reverse as an ablation."
        )
    return None


def test_cohort_is_non_empty() -> None:
    skip_if_public_export("experiments/ does not ship; the exp11 cohort is empty here")
    # Floor below the always-present tracked count so a fresh worktree is not a false red.
    assert len(_YAMLS) >= 25, f"expected the full exp11 cohort, found {len(_YAMLS)}"


@pytest.mark.parametrize("yaml_path", _YAMLS, ids=_ids(_YAMLS))
def test_arm_declares_a_registered_reverse_mode(yaml_path: Path) -> None:
    problem = _mode_problem(_load(yaml_path))
    assert problem is None, f"{yaml_path.name}: {problem}"


def test_each_sub_cohort_is_reverse_sampler_uniform() -> None:
    """Every non-ablation arm in a comparison group shares one (mode, clip_ratio) pair."""
    mixed = _mixed_groups([(_sub_cohort(p), _load(p)) for p in _YAMLS])
    assert not mixed, (
        "mixed reverse-sampler settings within a comparison sub-cohort; change a "
        "sub-cohort uniformly, or declare the arm's variable in "
        f"metadata.tags.reverse: {mixed}"
    )


def test_envelope_cohort_is_non_empty() -> None:
    skip_if_public_export("experiments/ does not ship; the exp11 cohorts are empty here")
    count = len(_ENVELOPE_YAMLS)
    assert count >= 150, f"expected the four freeze cohorts, found {count}"


@pytest.mark.parametrize(
    "yaml_path",
    _ENVELOPE_YAMLS,
    ids=[p.relative_to(_COHORT_DIR.parent).as_posix() for p in _ENVELOPE_YAMLS],
)
def test_freeze_arm_runs_under_the_decay_envelope(yaml_path: Path) -> None:
    problem = _envelope_problem(_load(yaml_path))
    assert problem is None, f"{yaml_path.name}: {problem}"


def _arm(
    mode: object, ratio: object = 1.3, reverse_tag: object = None, envelope: object = None
) -> dict:
    kwargs = {} if ratio is None else {"reverse_clip_ratio": ratio}
    if mode is not None:
        kwargs["reverse_sampling_mode"] = mode
    if envelope is not None:
        kwargs["reverse_decay_envelope"] = envelope
    tags = {"reverse": reverse_tag} if reverse_tag is not None else {}
    return {
        "metadata": {"tags": tags},
        "model": {"model_type": "kspace_cold_diffusion", "model_kwargs": kwargs},
    }


@pytest.mark.parametrize(
    ("cfg", "fragment"),
    [
        (_arm(None), "unset or unregistered"),
        (_arm("bogus"), "unset or unregistered"),
        (_arm("additive"), "declared"),
        (_arm("additive", reverse_tag="terminal_t0"), "declared"),
        (_arm("replace_freeze_dc", ratio=0), "positive"),
        (_arm("replace_freeze_dc", ratio=True), "positive"),
        (_arm("replace_freeze_dc", ratio=None), "positive"),
        (_arm("additive", reverse_tag="additive"), "not read"),
    ],
)
def test_planted_declarations_are_refused(cfg: dict, fragment: str) -> None:
    problem = _mode_problem(cfg)
    assert problem is not None and fragment in problem


def test_declared_baselines_and_ablations_are_accepted() -> None:
    assert _mode_problem(_arm("additive", ratio=None, reverse_tag="additive")) is None
    assert _mode_problem(_arm("reveal_refine_dc", reverse_tag="refine")) is None


def test_planted_mixed_group_is_reported_and_a_declared_ablation_is_not() -> None:
    base = _arm("replace_freeze_dc", envelope="per_sample")
    assert _mixed_groups([("g", base), ("g", _arm("replace_freeze", envelope="per_sample"))]) != {}
    assert (
        _mixed_groups([("g", base), ("g", _arm("replace_freeze_dc", 4.0, envelope="per_sample"))])
        != {}
    )
    assert _mixed_groups([("g", base), ("g", _arm("replace_freeze_dc"))]) != {}
    assert _mixed_groups([("g", base), ("g", _arm("additive", reverse_tag="additive"))]) == {}


@pytest.mark.parametrize("envelope", [None, "off", "batch_pooled"])
def test_planted_freeze_arm_without_the_cohort_envelope_is_refused(envelope: object) -> None:
    problem = _envelope_problem(_arm("replace_freeze_dc", envelope=envelope))
    assert problem is not None and "reverse_decay_envelope" in problem


def test_envelope_rule_exempts_the_declared_control_and_other_loops() -> None:
    assert _envelope_problem(_arm("replace_freeze_dc", envelope="per_sample")) is None
    assert _envelope_problem(_arm("replace_freeze_dc", reverse_tag="replace_freeze_dc")) is None
    assert _envelope_problem(_arm("replace_freeze_dc_t0")) is None


def _on_mask(cfg: dict, geometry: str) -> dict:
    return {**cfg, "undersampling": {"acceleration_type": geometry}}


@pytest.mark.parametrize("geometry", sorted(_RADIAL_ENVELOPE_GEOMETRIES))
def test_a_point_mask_arm_on_the_row_envelope_is_refused(geometry: str) -> None:
    """PLANTED: the pre-A5.4.1 shape, a 2-D point mask under the row envelope."""
    arm = _on_mask(_arm("replace_freeze_dc", envelope="per_sample"), geometry)
    problem = _envelope_problem(arm)
    assert problem is not None and "per_sample_radial" in problem
    assert _envelope_problem(_on_mask(_arm("replace_freeze_dc", envelope="per_sample_radial"), geometry)) is None


@pytest.mark.parametrize("geometry", ["density_nested", "variable_density", "variable_density_2d_gaussian"])
def test_a_row_or_variable_density_arm_on_the_radial_envelope_is_refused(geometry: str) -> None:
    """Variable density stays on rows until #2475; a Cartesian mask has nothing to gain."""
    arm = _on_mask(_arm("replace_freeze_dc", envelope="per_sample_radial"), geometry)
    assert _envelope_problem(arm) is not None


def test_a_group_mixing_masks_is_uniform_when_each_arm_matches_its_mask() -> None:
    rows = _on_mask(_arm("replace_freeze_dc", envelope="per_sample"), "density_nested")
    points = _on_mask(_arm("replace_freeze_dc", envelope="per_sample_radial"), "random")
    assert _mixed_groups([(".", rows), (".", points)]) == {}
    mismatched = _on_mask(_arm("replace_freeze_dc", envelope="per_sample"), "random")
    assert _mixed_groups([(".", rows), (".", points), (".", mismatched)]) != {}
