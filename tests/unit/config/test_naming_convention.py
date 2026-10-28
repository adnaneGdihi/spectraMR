"""The config-schema naming convention, enforced.

A reader cannot skim a config whose keys follow no rule. The census behind these
rules (1,888 unique field names across 295 classes) found a clear plurality for
each, so this ratifies existing practice rather than imposing a new style:

===================  =========================  =========================
Kind                 Rule                       Basis
===================  =========================  =========================
Boolean switch       ``enable_<thing>``         165 vs 20 ``use_*`` / 4 ``*_enabled``
A block's own gate   bare ``enabled``           reserved; never a feature flag
Count                ``num_<thing>``            36 vs 30 ``n_*``
Fraction in [0,1]    ``<thing>_fraction``       12 vs 5 ``*_ratio``
Negation             forbidden                  invert the sense instead
===================  =========================  =========================

**The gate is the deliverable, not the sweep.** A convention nobody can violate
by accident outlives any one-time rename, so this ships green: today's violators
are grandfathered in ``KNOWN_NAMING_EXCEPTIONS`` and each decomposition phase
drains the ones it already touches. New fields get no such grace.

Two rules from the plan's table are documented but deliberately NOT gated here.
The registry-selector rule (``<thing>_type`` over ``_mode`` / ``_strategy``)
cannot be mechanically separated from a genuine *mode*, and the path rule
(``_path`` file / ``_dir`` directory / ``_root`` tree) is a 16-16 split that has
to be decided per field by what it points at. Gating either on a name alone would
produce false failures, which is how a ratchet gets disabled.
"""

from __future__ import annotations

import importlib
import pkgutil
import re

import pytest
from pydantic import BaseModel


def _schema_field_names() -> set[str]:
    """Every field name declared anywhere under ``spectramr.config``."""
    import spectramr.config.schemas as schemas_pkg

    for mod in pkgutil.walk_packages(schemas_pkg.__path__, schemas_pkg.__name__ + "."):
        try:
            importlib.import_module(mod.name)
        except Exception:  # pragma: no cover - optional extras
            continue
    importlib.import_module("spectramr.config.settings")

    seen: set[type] = set()
    stack: list[type] = [BaseModel]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    names: set[str] = set()
    for cls in seen:
        if cls.__module__.startswith("spectramr.config"):
            names |= set(cls.model_fields)
    return names


#: name -> why it violates the convention.
RULES = {
    "use_*": ("boolean switches are `enable_<thing>`", lambda n: n.startswith("use_")),
    "*_enabled": (
        "boolean switches are `enable_<thing>`; bare `enabled` is reserved for a "
        "block's own gate",
        lambda n: n.endswith("_enabled"),
    ),
    "n_*": ("counts are `num_<thing>`", lambda n: bool(re.match(r"^n_", n))),
    "disable_*/no_*": (
        "negated booleans are forbidden — invert the sense",
        lambda n: n.startswith(("disable_", "no_")),
    ),
    "*_ratio": (
        "fractions in [0,1] are `<thing>_fraction`",
        lambda n: n.endswith("_ratio"),
    ),
}

#: Grandfathered violations, so the gate is green on day one and can only shrink.
#: Do NOT add to this list — rename the field instead. Each decomposition phase
#: drains the entries for the block it touches.
KNOWN_NAMING_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "auto_ratio",
        "background_suppression_threshold_ratio",
        "concomitant_enabled",
        "mask_ratio",
        "mlp_ratio",
        "multislice_enabled",
        "n_bootstrap",
        "n_calibration",
        "n_classes",
        "n_coils",
        "n_contrasts",
        "n_embeddings",
        "n_features",
        "n_frames",
        "n_group_samples",
        "n_iter",
        "n_mamba_layers",
        "n_orientations",
        "n_phase_cycles",
        "n_power_iterations",
        "n_real",
        "n_render_fields",
        "n_report_cases",
        "n_samples_per_tissue",
        "n_scales",
        "n_seeds_per_setting",
        "n_sites",
        "n_steps",
        "n_strata",
        "n_sub_bands",
        "n_super_bands",
        "n_synthetic",
        "n_synthetic_pairs",
        "n_timepoints",
        "n_vendors",
        "progress_bar_enabled",
        "topk_ratio",
        "use_adaptive_threshold",
        "use_async_dataloader",
        "use_bn_in_head",
        "use_capacity_scheduling",
        "use_compile",
        "use_curriculum_scheduling",
        "use_distributed",
        "use_field_conditioning",
        "use_fp16",
        "use_gradient_checkpointing",
        "use_latent_regularization",
        "use_magnitude_scaling",
        "use_memory_bank",
        "use_momentum",
        "use_orig_params",
        "use_real_stack",
        "use_repetitions",
    }
)

_ALL_NAMES = _schema_field_names()


def _violations() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _ALL_NAMES:
        for label, (why, pred) in RULES.items():
            if pred(name):
                out[name] = f"{label}: {why}"
                break
    return out


def test_discovery_is_non_trivial() -> None:
    """Guard against an import regression that empties the sweep and makes every
    assertion below vacuously true."""
    assert len(_ALL_NAMES) > 1000


def test_no_new_naming_violations() -> None:
    """Any field breaking a rule must be grandfathered. New ones are rejected."""
    new = sorted(set(_violations()) - KNOWN_NAMING_EXCEPTIONS)
    assert not new, (
        "these field names break the naming convention and are not "
        "grandfathered:\n  "
        + "\n  ".join(f"{n} — {_violations()[n]}" for n in new)
        + "\n\nRename the field. Do not add it to KNOWN_NAMING_EXCEPTIONS."
    )


def test_exception_list_has_no_stale_entries() -> None:
    """The ratchet must tighten: an entry whose field was renamed or deleted has
    to leave the list, or the list stops measuring anything."""
    stale = sorted(KNOWN_NAMING_EXCEPTIONS - set(_violations()))
    assert not stale, (
        "these names are grandfathered but no longer violate anything (renamed "
        "or removed). Delete them from KNOWN_NAMING_EXCEPTIONS:\n  "
        + "\n  ".join(stale)
    )


@pytest.mark.parametrize("label", sorted(RULES))
def test_every_rule_can_fire(label: str) -> None:
    """A rule that matches nothing is untested. Each must reject a sample name,
    so a typo in a predicate cannot make the gate silently permissive."""
    samples = {
        "use_*": "use_amp",
        "*_enabled": "tracking_enabled",
        "n_*": "n_coils",
        "disable_*/no_*": "disable_defaults",
        "*_ratio": "split_ratio",
    }
    _why, pred = RULES[label]
    assert pred(samples[label]), f"rule {label} no longer matches its own sample"
