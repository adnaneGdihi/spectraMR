"""Regression: every exp11 kspace_filling cold arm declares the training-time head clamp.

#869 found ``experiment_11_kan_dual_domain`` diverging in validation because it
carried none of the cohort's scale guards. ``test_exp11_reverse_sampler_cohort``
pins two of them, the reverse mode and ``reverse_clip_ratio``; those bound only the
reverse loop. This module pins the other two, which bound the generator's output
during training: ``output_kspace_clip_ratio`` (absent means the clamp never runs)
and ``output_kspace_clip_reference`` (absent means ``global_max``, a ceiling
referenced to the DC peak rather than to the band). It also pins the KAN arm's
``kspace_feature_norm: rms``. That pin is per-arm on purpose: whether ``rms`` should
stay cohort-wide is open on #2117, whose proposed resolution is a variant arm
declaring ``none``.

The arms are read with ``yaml.safe_load`` and no model is built. The planted
configs at the bottom are the shapes the rule must reject.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.utils.corpus import tracked_yamls
from tests.utils.repo_scripts import skip_if_public_export

_COHORT_DIR = Path(__file__).resolve().parents[3] / "experiments" / "inprogress" / "kspace_filling"

_HEAD_CLAMP = {"output_kspace_clip_ratio": 1.3, "output_kspace_clip_reference": "band_local"}

#: Cold arms that declare no head clamp on dev at the time of writing. They are
#: recorded so the rule applies to every new arm; whether these six should carry
#: the clamp is open on #869. The set is exact: an arm that gains the clamp must
#: leave it, or the stale-entry test fails.
_HEAD_CLAMP_EXEMPT = frozenset(
    {
        "experiment_130_ti_ccd.yaml",
        "experiment_cross_contrast_kspace_diffusion.yaml",
        "experiment_cross_contrast_kspace_diffusion_noprior_baseline.yaml",
        "experiment_cross_contrast_kspace_diffusion_noprior_flair.yaml",
        "experiment_cross_contrast_kspace_diffusion_refenc.yaml",
        "experiment_cross_contrast_kspace_diffusion_t2_to_flair.yaml",
    }
)

_KAN_ARM = "experiment_11_kan_dual_domain.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _model_kwargs(cfg: dict) -> dict:
    return (cfg.get("model") or {}).get("model_kwargs") or {}


def _arm_id(path: Path) -> str:
    return path.relative_to(_COHORT_DIR).as_posix()


def _uses_cold_sampler(cfg: dict) -> bool:
    return (cfg.get("model") or {}).get("model_type") == "kspace_cold_diffusion"


def _head_clamp_problem(cfg: dict) -> str | None:
    """Why a cold arm's training-time head clamp is refused, or ``None``."""
    kwargs = _model_kwargs(cfg)
    wrong = {k: kwargs.get(k) for k, v in _HEAD_CLAMP.items() if kwargs.get(k) != v}
    if wrong:
        return f"head clamp {wrong} differs from the cohort's {_HEAD_CLAMP} (#869)."
    return None


_COLD_YAMLS = [p for p in tracked_yamls(_COHORT_DIR) if _uses_cold_sampler(_load(p))]
_CLAMPED_YAMLS = [p for p in _COLD_YAMLS if _arm_id(p) not in _HEAD_CLAMP_EXEMPT]


def test_cohort_is_non_empty() -> None:
    skip_if_public_export("experiments/ does not ship; the exp11 cohort is empty here")
    assert len(_CLAMPED_YAMLS) >= 25, f"expected the exp11 cohort, found {len(_CLAMPED_YAMLS)}"


@pytest.mark.parametrize("yaml_path", _CLAMPED_YAMLS, ids=[_arm_id(p) for p in _CLAMPED_YAMLS])
def test_cold_arm_declares_the_training_time_head_clamp(yaml_path: Path) -> None:
    problem = _head_clamp_problem(_load(yaml_path))
    assert problem is None, f"{yaml_path.name}: {problem}"


def test_head_clamp_exemptions_are_exact() -> None:
    skip_if_public_export("experiments/ does not ship; the exp11 cohort is empty here")
    by_id = {_arm_id(p): _load(p) for p in _COLD_YAMLS}
    missing = sorted(_HEAD_CLAMP_EXEMPT - by_id.keys())
    assert not missing, f"exempt arms are no longer cold arms of the cohort: {missing}"
    stale = sorted(n for n in _HEAD_CLAMP_EXEMPT if _head_clamp_problem(by_id[n]) is None)
    assert not stale, f"exempt arms now declare the head clamp; remove them from the set: {stale}"


def test_kan_arm_carries_the_inter_layer_norm() -> None:
    skip_if_public_export("experiments/ does not ship; the exp11 cohort is empty here")
    norm = _model_kwargs(_load(_COHORT_DIR / _KAN_ARM)).get("kspace_feature_norm")
    assert norm == "rms", (
        f"{_KAN_ARM}: kspace_feature_norm={norm!r}; its 2026-07-29 run diverged without the "
        "norm its attention_shootout sibling carries (#869)."
    )


def _clamped_arm() -> dict:
    return {
        "model": {
            "model_type": "kspace_cold_diffusion",
            "model_kwargs": {"reverse_sampling_mode": "replace_freeze_dc", **_HEAD_CLAMP},
        }
    }


def test_the_planted_baseline_is_accepted() -> None:
    assert _head_clamp_problem(_clamped_arm()) is None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("output_kspace_clip_ratio", None),
        ("output_kspace_clip_ratio", 4.0),
        ("output_kspace_clip_reference", None),
        ("output_kspace_clip_reference", "global_max"),
    ],
)
def test_planted_arm_without_the_head_clamp_is_refused(key: str, value: object) -> None:
    """PLANTED: the #869 shape, a cold arm whose head clamp is absent, loose or global."""
    arm = _clamped_arm()
    kwargs = arm["model"]["model_kwargs"]
    if value is None:
        kwargs.pop(key)
    else:
        kwargs[key] = value
    problem = _head_clamp_problem(arm)
    assert problem is not None and key in problem
