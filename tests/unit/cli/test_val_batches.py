"""``--val-batches``: the one owner of the key that caps a validation pass.

The tests that matter here are the ones that would go red on a *plausible*
mistake rather than a typo (non-negotiable 15): re-pointing the flag at an inert
key, lowering it onto ``num_samples`` (which the readers consult only as a
fallback), and appending a cap on top of an operator's own ``-O`` for the same
key.
"""

from __future__ import annotations

import argparse

import pytest

from spectramr.cli.val_batches import (
    VAL_BATCHES_OVERRIDE_KEY,
    add_val_batches_argument,
    resolve_val_batches_overrides,
    val_batches_override,
)

# ------------------------------------------------------- the key it lowers onto


def test_the_key_exists_on_the_real_schema():
    """A rename of the field must fail here, not silently at the terminal.

    Walks the declared path through the Pydantic models rather than a loaded
    config, so the assertion is about the SCHEMA and cannot be satisfied by an
    arm that happens to set the key.
    """
    from spectramr.config.schemas.validation import ValidationConfigSchema

    model: type = ValidationConfigSchema
    for part in VAL_BATCHES_OVERRIDE_KEY.split(".")[1:]:
        assert part in model.model_fields, f"{part!r} missing from {model.__name__}"
        model = model.model_fields[part].annotation


def test_the_key_is_canonical():
    """Lowering onto a legacy spelling would work until the fold is drained."""
    from spectramr.config.schemas.renames import canonical_override_path

    assert canonical_override_path(VAL_BATCHES_OVERRIDE_KEY) == VAL_BATCHES_OVERRIDE_KEY


def test_it_caps_num_batches_and_not_num_samples():
    """`num_samples` is read ONLY when `num_batches` is None (pipelines/train.py).

    A flag lowered onto it is therefore inert on every arm whose YAML already
    declares `num_batches` -- the silent-no-op shape of pitfall 15.
    """
    assert val_batches_override(3) == "validation.loader.num_batches=3"
    assert "num_samples" not in val_batches_override(3)


def test_it_does_not_target_the_inert_enabled_knob():
    """Issue #673: `validation.enabled` is set by 1006 arms and read by nothing."""
    assert "validation.enabled" not in val_batches_override(1)


# ----------------------------------------------------------------- the budget


@pytest.mark.parametrize("n", [0, -1])
def test_a_sub_one_budget_raises_rather_than_meaning_no_cap(n):
    """Both readers treat a sub-1 budget as 'no cap', i.e. the whole split."""
    with pytest.raises(ValueError, match=r"must be >= 1"):
        val_batches_override(n)


def test_one_is_accepted():
    assert val_batches_override(1).endswith("=1")


# -------------------------------------------------------------- the lowering


def test_absent_flag_leaves_the_overrides_untouched():
    """A run that does not pass the flag must be identical to a pre-flag run."""
    existing = ["training.max_iterations=10"]
    assert resolve_val_batches_overrides(existing, None) is existing
    assert resolve_val_batches_overrides(None, None) is None


def test_it_appends_rather_than_replacing():
    out = resolve_val_batches_overrides(["training.max_iterations=10"], 2)
    assert out == ["training.max_iterations=10", "validation.loader.num_batches=2"]


def test_it_works_from_an_empty_override_list():
    assert resolve_val_batches_overrides(None, 2) == ["validation.loader.num_batches=2"]


def test_conflict_with_an_explicit_override_raises():
    """Appending would win on precedence and discard the operator's own -O."""
    with pytest.raises(ValueError, match=r"conflicts with the override"):
        resolve_val_batches_overrides(["validation.loader.num_batches=9"], 2)


def test_conflict_is_detected_through_the_legacy_spelling():
    """A textual match would miss this: the legacy path still folds onto the key."""
    with pytest.raises(ValueError, match=r"conflicts with the override"):
        resolve_val_batches_overrides(["validation.num_validation_batches=9"], 2)


def test_an_unrelated_override_is_not_mistaken_for_a_conflict():
    out = resolve_val_batches_overrides(["validation.loader.batch_size=2"], 2)
    assert out[-1] == "validation.loader.num_batches=2"


def test_a_malformed_override_does_not_mask_the_cap():
    """A bare token has no `=`; it is the loader's error to report, not ours."""
    out = resolve_val_batches_overrides(["not-an-override"], 2)
    assert out[-1] == "validation.loader.num_batches=2"


# --------------------------------------------------------------- the argument


def test_the_argument_parses_and_defaults_to_none():
    parser = argparse.ArgumentParser()
    add_val_batches_argument(parser)
    assert parser.parse_args([]).val_batches is None
    assert parser.parse_args(["--val-batches", "4"]).val_batches == 4


def test_the_argument_rejects_a_non_integer():
    parser = argparse.ArgumentParser()
    add_val_batches_argument(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--val-batches", "all"])
