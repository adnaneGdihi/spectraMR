"""Registry-contract tests: every entry must satisfy the dispatch shape.

Targets ``spectramr.models.registry``. We don't try to instantiate all 400+ models
here (that lives in a future ``@pytest.mark.slow`` exhaustive-instantiation
test, see plan D.3 follow-up). Instead, we verify the structural contract that
``ModelBuilder`` / ``ModelFactory`` / config-audit relies on:

1. The registry is non-empty after auto-discovery has run.
2. Every entry exposes a ``"class"`` key whose value is a Python class.
3. Every entry exposes a ``"mode"`` key (string).
4. ``get_model_class(name)`` returns the registered class for every name.
5. ``get_model_class("definitely-not-a-model")`` raises ``ValueError`` with
   the available-names list (the audit relies on this for misconfigured
   YAMLs — silent fallbacks are forbidden per CLAUDE.md pitfall #9).

Future work (gated ``@pytest.mark.slow``, opt-in nightly): instantiate every
class and run a forward pass. Out of scope here because per-model minimal
config construction is multi-day work.
"""

from __future__ import annotations

import inspect

import pytest

# Trigger auto-discovery. Importing ``init_registry`` only *defines*
# ``populate_model_registry`` — it must be *called* to walk the model tree
# and fill ``MODEL_REGISTRY``. Calling it at module load makes the registry
# populated both for the runtime assertions below and for the collection-time
# ``parametrize(... MODEL_REGISTRY.keys())``. (Previously this relied on some
# earlier test in the session having populated the global registry, so the
# file failed when run in isolation.)
from spectramr.models.init_registry import populate_model_registry
from spectramr.models.registry import MODEL_REGISTRY, get_model_class

populate_model_registry()


def test_registry_is_non_empty_after_discovery() -> None:
    """Auto-discovery should have populated the registry with many models."""
    assert len(MODEL_REGISTRY) >= 50, (
        f"Expected ≥50 registered models post-discovery, got {len(MODEL_REGISTRY)}. "
        "Likely cause: auto-discovery in spectramr.models.init_registry failed silently."
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY.keys()))
def test_entry_exposes_class_field(name: str) -> None:
    """Every registered entry must have a callable class under the 'class' key."""
    entry = MODEL_REGISTRY[name]
    assert "class" in entry, f"Entry '{name}' missing 'class' key: {list(entry.keys())}"
    cls = entry["class"]
    assert inspect.isclass(cls), (
        f"'class' for '{name}' is {type(cls).__name__}, not a class"
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY.keys()))
def test_entry_has_mode_string(name: str) -> None:
    """Every entry must declare a training mode (used by strategy dispatch)."""
    entry = MODEL_REGISTRY[name]
    assert "mode" in entry, f"Entry '{name}' missing 'mode' key"
    assert isinstance(entry["mode"], str) and entry["mode"], (
        f"Entry '{name}' has empty/non-string mode: {entry['mode']!r}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY.keys()))
def test_get_model_class_returns_registered_class(name: str) -> None:
    """``get_model_class(name)`` must round-trip the registered class."""
    expected = MODEL_REGISTRY[name]["class"]
    got = get_model_class(name)
    assert got is expected, f"get_model_class({name!r}) returned {got}, expected {expected}"


def test_unknown_model_raises_with_available_names() -> None:
    """Misspelled / dropped model names must fail loud, not silently fallback."""
    with pytest.raises(ValueError, match="not found in registry"):
        get_model_class("definitely-not-a-real-model-name-xyz123")


def test_unknown_model_error_lists_alternatives() -> None:
    """The error message must include the available names so a YAML author can fix the typo."""
    try:
        get_model_class("nonexistent_model_zzz")
    except ValueError as e:
        msg = str(e)
        # At least one real model name should appear in the error.
        sample = next(iter(MODEL_REGISTRY))
        assert sample in msg, (
            f"Expected available-models list in error, got: {msg[:200]}"
        )
    else:
        pytest.fail("Expected ValueError for unknown model name")


# ── 6. One owner for capability flags (#1916) ──────────────────────
#
# The helper tests in ``test_registry_helpers.py`` register their own models
# into a cleared registry, so they can only prove the readers agree about a
# model the test itself just wrote. That is the easy shape (non-negotiable
# 15). The invariant below is checked against the REAL registry, which is
# where #1916 lived: two reader families each answered confidently and shared
# **zero** models on every flag.

# Everything a registry entry is allowed to carry at the top level.
# ``class``/``mode`` are the dispatch shape asserted above; ``role`` (#1932)
# is a routing key -- which factory bucket the model belongs in -- and
# ``capabilities`` is the nested dataclass that owns every capability flag.
_ALLOWED_ENTRY_KEYS = frozenset({"class", "mode", "role", "capabilities"})


def test_no_entry_carries_a_top_level_capability_key() -> None:
    """Capability flags live on the nested dataclass and nowhere else.

    This is the invariant #1916 elected, and it is deliberately NOT phrased as
    "the two readers agree". ``model_supports`` is *implemented as* a read of
    ``get_model_capabilities``, so comparing them is a tautology today -- and
    it would stay green if a future change re-pointed both readers at the same
    wrong surface together. Asserting the loser surface is *absent* cannot be
    satisfied that way.

    What it caught: ``register_model`` used to fan
    ``supports_contrast_conditioning`` / ``supports_vendor_conditioning`` out
    to top-level keys *as well as* into ``ModelCapabilities``, so
    ``model_supports`` read the top-level half and ``get_model_capabilities``
    the nested half. Measured on ``dev`` over 588 models: ``accepts_complex``
    20 nested vs 0 top-level, ``requires_paired_data`` 71 vs 0,
    ``expects_real_imag_interleaved`` 11 vs 0,
    ``supports_contrast_conditioning`` 0 vs 28 -- 130 disagreements, every
    fixture-scoped test green throughout.
    """
    assert MODEL_REGISTRY, "registry is empty; this test would pass vacuously"
    offenders = {
        name: sorted(set(entry) - _ALLOWED_ENTRY_KEYS)
        for name, entry in MODEL_REGISTRY.items()
        if isinstance(entry, dict) and not set(entry) <= _ALLOWED_ENTRY_KEYS
    }
    assert not offenders, (
        f"{len(offenders)} registry entries carry top-level keys outside "
        f"{sorted(_ALLOWED_ENTRY_KEYS)}. A capability flag at the top level is "
        f"a second owner: it is invisible to get_model_capabilities and to "
        f"every audit check that reads it (#1916). Declare it on "
        f"ModelCapabilities instead. Offenders: "
        f"{dict(sorted(offenders.items())[:10])}"
    )


def test_the_nested_owner_actually_holds_declarers() -> None:
    """Non-vacuity for the test above: the surviving owner is populated.

    Moving a flag from a dict key to a dataclass field turns a loud
    ``KeyError`` into a quiet ``None``, so a census can read **zero** and every
    assertion over it passes trivially. That happened twice while #1916 was
    being fixed -- once in production (``_contrast_aware_critics`` reported 0
    aware critics) and once in the test policing it. A key-absence assertion
    alone is satisfied by a registry that declares nothing at all.

    Counts drift as models are added; this asserts only that each flag has
    real declarers, and records the reading for the next person.
    """
    from spectramr.models.registry import _boolean_capability_fields, model_supports

    declared = {
        f: sum(model_supports(n, f) for n in MODEL_REGISTRY) for f in _boolean_capability_fields()
    }
    # Measured 2026-09-08 on 588 models: accepts_complex 20,
    # requires_paired_data 71, expects_real_imag_interleaved 11,
    # supports_contrast_conditioning 28, supports_vendor_conditioning 0.
    # ``supports_vendor_conditioning`` is honestly 0 -- the capability is
    # unbuilt (#1941) -- so it is excluded rather than asserted non-zero.
    expected_nonzero = set(declared) - {"supports_vendor_conditioning"}
    empty = sorted(f for f in expected_nonzero if declared[f] == 0)
    assert not empty, (
        f"{empty} have zero declarers on ModelCapabilities over "
        f"{len(MODEL_REGISTRY)} models. Either register_model stopped "
        f"forwarding them to the dataclass, or the flag moved again -- and "
        f"every check reading it is now silently answering 'no' for "
        f"everything (#1916). Full reading: {declared}"
    )
