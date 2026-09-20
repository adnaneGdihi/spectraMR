"""Tests for ``ValidationConfigSchema``.

Targets ``spectramr.config.schemas.validation``. Focused on the
``empty_cache_before_validation`` knob added by the wasted-compute audit
(backlog_wasted_compute_audit_2026_05_29 PIPE-2): the per-validation
``torch.cuda.empty_cache()`` is now wired to this flag so memory-headroom runs
can skip the allocator re-grow on the next train step. The default must remain
``True`` (preserving the prior OOM-safe behavior).
"""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.validation import ValidationConfigSchema


def test_empty_cache_before_validation_defaults_true() -> None:
    """Default preserves the prior always-on OOM-safe behavior."""
    cfg = ValidationConfigSchema()
    assert cfg.empty_cache_before_validation is True


def test_empty_cache_before_validation_can_be_disabled() -> None:
    """Memory-headroom runs can opt out of the per-validation empty_cache."""
    cfg = ValidationConfigSchema(empty_cache_before_validation=False)
    assert cfg.empty_cache_before_validation is False


def test_multistep_cold_sampling_defaults_false() -> None:
    """2026-06-08: the opt-in true-multi-step cold sampler must default OFF so
    every existing arm keeps the single-forward validation behaviour unchanged.

    This pins a CHOSEN default, not pydantic's behaviour -- flipping it would
    silently change what every arm validates. It does not, on its own, exercise
    the rename; the test below does that.
    """
    assert ValidationConfigSchema().sampling.enable_multistep_cold is False


def test_the_legacy_spelling_raises_and_names_its_destination() -> None:
    """The rename's posture was promoted ``fold`` -> ``raise``.

    Under ``fold`` the honest test of this rename was legacy-in / canonical-out,
    and that is exactly what this file used to assert. Promoting the record
    inverted it: the legacy declaration must now REFUSE, and say where the key
    went. Same intent, opposite assertion -- which is why a posture promotion
    breaks the tests written to exercise the fold it replaced (2 records in
    cluster job 8012333).

    Matching on the destination path, not just on "raises": a bare raise would
    also pass if the field had simply been deleted, which is a different and
    much worse outcome for a user holding an old config.
    """
    with pytest.raises(ValidationError, match=r"validation\.sampling\.enable_multistep_cold"):
        ValidationConfigSchema(multistep_cold_sampling=True)


def test_multistep_cold_sampling_can_be_enabled() -> None:
    """The _sampled arm opts into the genuine cold_mri reverse loop."""
    cfg = ValidationConfigSchema(sampling={"enable_multistep_cold": True})
    assert cfg.sampling.enable_multistep_cold is True


def _flag_tokens(text: str) -> set[str]:
    """Every CLI-flag-looking token in ``text`` (``-O``, ``--override``, ...)."""
    return set(re.findall(r"(?<![\w-])(--?[A-Za-z][\w-]*)", text))


def _real_cli_flags() -> set[str]:
    """Every option string the real CLI parser accepts, across all subcommands."""
    from spectramr.cli.app import build_parser

    flags: set[str] = set()
    stack = [build_parser()]
    while stack:
        parser = stack.pop()
        for action in parser._actions:
            flags.update(action.option_strings)
            # Only a subparsers action carries a dict of parsers; an ordinary
            # `choices=` is a plain sequence of values and must not be walked.
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                stack.extend(c for c in choices.values() if hasattr(c, "_actions"))
    return flags


def test_the_interval_steps_help_names_a_flag_the_cli_actually_accepts() -> None:
    """A schema description that names a non-existent flag is worse than one
    that names none.

    It is read at exactly the moment an operator is trying to fix a config, and
    it sends them to type something that fails. This exact defect shipped: the
    description told readers to use ``--set training.max_iterations=...`` while
    the CLI only ever accepted ``--override``/``-O``. Nothing caught it because
    no gate crosses prose against argparse -- so this test does.
    """
    from spectramr.config.schemas.validation import ValidationScheduleConfigSchema

    description = ValidationScheduleConfigSchema.model_fields["interval_steps"].description or ""
    assert description, "the interval_steps description went missing"

    real = _real_cli_flags()
    assert "--override" in real and "-O" in real, (
        "the override flag was renamed; this test's own anchor is stale"
    )

    named = _flag_tokens(description)
    unknown = named - real
    assert not unknown, (
        f"interval_steps' description names CLI flags that do not exist: "
        f"{sorted(unknown)}. An operator reading it would type a command that "
        f"fails. Real flags include: {sorted(f for f in real if 'over' in f)}"
    )


class TestValidationCascadeBlock:
    """`validation.cascade.levels` — the declared validation ladder (#1394)."""

    def test_absent_by_default_meaning_the_framework_ladder(self):
        """`None`, not a copy of (2, 8, 32).

        Presence is what distinguishes "the author chose this ladder" from
        "the author said nothing", and a defaulted copy collapses the two —
        a later change to the framework default would then silently move an
        arm that had made no decision, or fail to move one that had.
        """
        from spectramr.config.schemas.validation import ValidationConfigSchema

        assert ValidationConfigSchema().cascade.levels is None

    def test_a_declared_ladder_is_stored_deduped_and_ascending(self):
        from spectramr.config.schemas.validation import ValidationConfigSchema

        cfg = ValidationConfigSchema(cascade={"levels": [8, 2, 8, 4]})
        assert cfg.cascade.levels == [2.0, 4.0, 8.0]

    def test_the_resolver_and_the_schema_agree_on_a_declared_ladder(self):
        """One owner for "is this ladder legal" (non-negotiable 17)."""
        from spectramr.config.schemas.validation import ValidationConfigSchema
        from spectramr.core.cascading_validation import resolve_cascade_levels

        cfg = ValidationConfigSchema(cascade={"levels": [4, 16]})
        assert resolve_cascade_levels(cfg) == (4, 16)

    def test_a_boolean_rung_is_refused_before_pydantic_coerces_it(self):
        """`levels: [true]` would otherwise become the legal ladder `[1.0]`.

        Nothing downstream could tell that apart from a declared R=1 sweep —
        the silent substitution non-negotiable 3 forbids.
        """
        import pytest as _pytest

        from spectramr.config.schemas.validation import ValidationConfigSchema

        with _pytest.raises(ValueError, match="boolean"):
            ValidationConfigSchema(cascade={"levels": [True]})

    @pytest.mark.parametrize("bad", [[], [0.5], [float("inf")]])
    def test_an_illegal_ladder_is_refused_at_load_time(self, bad):
        from spectramr.config.schemas.validation import ValidationConfigSchema

        with pytest.raises(ValueError):
            ValidationConfigSchema(cascade={"levels": bad})

    def test_the_cascade_sub_block_forbids_an_unknown_key(self):
        """New sub-blocks are born strict (`_VAL_SUBBLOCK`)."""
        from spectramr.config.schemas.validation import ValidationConfigSchema

        with pytest.raises(ValueError):
            ValidationConfigSchema(cascade={"levls": [2, 8]})


class TestValidationEnsembleKnobs:
    """``validation.sampling.ensemble_samples`` / ``coverage_k`` (review follow-up item 2).

    Planted refusals first: every rule the schema claims is watched going red.
    """

    def test_defaults_are_a_single_deterministic_sample(self) -> None:
        sampling = ValidationConfigSchema().sampling
        assert sampling.ensemble_samples == 1
        assert sampling.coverage_k == 2.0

    def test_an_ensemble_needs_the_multistep_sampler(self) -> None:
        """N > 1 on the single-step forward would be N identical members."""
        with pytest.raises(ValidationError, match="enable_multistep_cold"):
            ValidationConfigSchema(sampling={"ensemble_samples": 2})

    def test_five_members_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="less than or equal to 4"):
            ValidationConfigSchema(sampling={"enable_multistep_cold": True, "ensemble_samples": 5})

    def test_zero_members_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            ValidationConfigSchema(sampling={"enable_multistep_cold": True, "ensemble_samples": 0})

    def test_a_nonpositive_coverage_k_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="greater than 0"):
            ValidationConfigSchema(sampling={"coverage_k": 0.0})

    def test_an_ensemble_of_four_loads_with_the_multistep_sampler(self) -> None:
        cfg = ValidationConfigSchema(
            sampling={"enable_multistep_cold": True, "ensemble_samples": 4, "coverage_k": 1.5}
        )
        assert cfg.sampling.ensemble_samples == 4
        assert cfg.sampling.coverage_k == 1.5


class TestRevealAttributionKnob:
    """``validation.sampling.reveal_attribution`` (#2067 follow-up).

    Planted refusal first: the knob partitions the reconstruction by reverse
    step, which a single deterministic forward does not have. Accepting it there
    would emit an all-``nan`` ``val_band_*`` block, which reads as "the bands are
    unmeasurable" rather than "this arm cannot run the diagnostic".
    """

    def test_default_is_off_so_existing_arms_are_unchanged(self) -> None:
        assert ValidationConfigSchema().sampling.reveal_attribution is False

    def test_attribution_without_the_multistep_sampler_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="enable_multistep_cold"):
            ValidationConfigSchema(sampling={"reveal_attribution": True})

    def test_attribution_loads_with_the_multistep_sampler(self) -> None:
        sampling = ValidationConfigSchema(
            sampling={"enable_multistep_cold": True, "reveal_attribution": True}
        ).sampling
        assert sampling.reveal_attribution is True

    def test_the_multistep_sampler_alone_does_not_enable_attribution(self) -> None:
        """Opt-in: the oracle reads the target, so it is never implied."""
        sampling = ValidationConfigSchema(sampling={"enable_multistep_cold": True}).sampling
        assert sampling.reveal_attribution is False
