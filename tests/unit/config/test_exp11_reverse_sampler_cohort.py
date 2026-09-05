"""Regression: every exp11 kspace_filling arm samples via a corrected reverse loop.

Guards the two properties that keep the cohort scientifically valid, without
pinning a single global value (which drifts as sub-cohorts are added):

1. **No arm inherits the broken ``additive`` default.** Every arm must set
   ``model.model_kwargs.reverse_sampling_mode`` to a corrected mode
   (``replace_freeze`` or ``replace_freeze_dc``) and a positive
   ``reverse_clip_ratio``. The legacy ``additive`` reverse loop re-adds
   ``x0 * (M_{t-1} - M_t)`` onto bins that are already nonzero and its strided
   ``t-1`` reveal leaves holes, so magnitudes compound across the trajectory
   (the pinned train-21 / val-~3 PSNR collapse). An unset knob silently falls
   back to that default (``kspace_cold_diffusion_generator.py:1966``).

2. **Per-sub-cohort uniformity (pitfall #17).** Arms compared against each other
   live in one directory; every arm in a directory must share one
   ``(reverse_sampling_mode, reverse_clip_ratio)`` pair, so a shootout is not
   confounded by a mixed reverse loop. The cohort is deliberately **not**
   globally uniform: ``dc_shootout`` and ``attention_shootout`` sample via
   ``replace_freeze_dc`` (observed lines honour ``dc_method`` every step) while
   the rest stay on ``replace_freeze``; ``attention_shootout`` additionally caps
   at ``reverse_clip_ratio: 1.3`` (2026-07-19 val-band fix — the 4.0 ceiling let
   unobserved-line predictions reach 4x|DC| and dominate as a saturated central
   k-space band). Comparability is enforced within a directory, not across.

Pure-config test (``yaml.safe_load`` only, no torch / no model construction);
schema round-tripping of the same files is covered by
``test_exp11_variant_dc_consistency.py``.
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

# The corrected reverse loops (both write-once/freeze the unobserved support and
# magnitude-cap at ``reverse_clip_ratio × max|observed|``). ``additive`` is the
# broken legacy default and is intentionally excluded.
_ALLOWED_MODES = frozenset({"replace_freeze", "replace_freeze_dc"})


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


def _reverse_knobs(path: Path) -> tuple[object, object]:
    cfg = yaml.safe_load(path.read_text())
    model_kwargs = (cfg.get("model") or {}).get("model_kwargs") or {}
    return (
        model_kwargs.get("reverse_sampling_mode"),
        model_kwargs.get("reverse_clip_ratio"),
    )


_YAMLS = _cohort_yamls()


def test_cohort_is_non_empty() -> None:
    skip_if_public_export("experiments/ does not ship; the exp11 cohort is empty here")
    # Same floor rationale as test_exp11_variant_dc_consistency.py: floor below
    # the always-present tracked count so a fresh worktree is not a false red.
    assert len(_YAMLS) >= 25, f"expected the full exp11 cohort, found {len(_YAMLS)}"


@pytest.mark.parametrize("yaml_path", _YAMLS, ids=_ids(_YAMLS))
def test_arm_sets_an_explicit_corrected_reverse_mode(yaml_path: Path) -> None:
    mode, ratio = _reverse_knobs(yaml_path)
    assert mode in _ALLOWED_MODES, (
        f"{yaml_path.name}: model_kwargs.reverse_sampling_mode={mode!r} — every "
        f"kspace_filling arm must set one of {sorted(_ALLOWED_MODES)}; an unset "
        f"knob silently inherits the broken legacy 'additive' reverse loop and "
        f"confounds cross-arm comparisons."
    )
    assert (
        isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and ratio > 0
    ), (
        f"{yaml_path.name}: reverse_clip_ratio={ratio!r} — must be a positive "
        f"number (the fixed per-sample magnitude ceiling ratio*max|observed|)."
    )


def test_each_sub_cohort_is_reverse_sampler_uniform() -> None:
    """Every arm in a comparison group shares one (mode, clip_ratio) pair."""
    groups: dict[str, set[tuple[object, object]]] = collections.defaultdict(set)
    for path in _YAMLS:
        groups[_sub_cohort(path)].add(_reverse_knobs(path))

    mixed = {g: settings for g, settings in groups.items() if len(settings) > 1}
    assert not mixed, (
        "mixed reverse-sampler settings within a comparison sub-cohort "
        "(confounded ablation, pitfall #17); change a sub-cohort's cap "
        f"uniformly, not per-arm: {mixed}"
    )


def test_allowed_modes_are_valid_reverse_modes() -> None:
    """Fail loudly if a mode string drifts from the source-of-truth set."""
    src = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "spectramr"
        / "models"
        / "diffusion"
        / "kspace_process.py"
    ).read_text()
    for mode in sorted(_ALLOWED_MODES):
        assert f'"{mode}"' in src, (
            f"{mode!r} no longer present in kspace_process.py "
            f"(VALID_REVERSE_MODES) — the whole cohort would fail at build; "
            f"update the configs and this test together."
        )
