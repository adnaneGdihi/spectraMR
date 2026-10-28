"""Tests for the config-layer override utilities (``config/overrides.py``).

These moved out of ``spectramr.main`` (the CLI/entry layer) so ``pipelines/`` can
import them rightward instead of leftward (CLAUDE.md #13). Behavioural coverage
of ``apply_overrides`` against a real ``TrainingSettings`` lives in
``tests/unit/config/test_settings.py``; here we pin the pure ``_parse_value``
type coercion and the module's re-export contract.
"""

from __future__ import annotations

from contextlib import suppress

import pytest

from spectramr.config.overrides import (
    _parse_value,
    applied_override_paths,
    apply_overrides,
)
from tests.unit.config.test_settings import _minimal_config


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True),
        ("YES", True),
        ("on", True),
        ("false", False),
        ("no", False),
        ("off", False),
        ("none", None),
        ("null", None),
        ("42", 42),
        ("-10", -10),
        ("3.14", 3.14),
        ("1e-4", 0.0001),
        ("standard_unet", "standard_unet"),
    ],
)
def test_parse_value_type_coercion(raw, expected):
    result = _parse_value(raw)
    assert result == expected
    assert type(result) is type(expected)


def test_parse_value_int_before_float():
    # "10" must land as int, not float (integer branch runs first).
    assert _parse_value("10") == 10
    assert isinstance(_parse_value("10"), int)


def test_main_reexports_same_objects():
    """``spectramr.main`` must re-export the SAME callables (backward compat)."""
    from spectramr import main

    assert main.apply_overrides is apply_overrides
    assert main._parse_value is _parse_value


def test_apply_overrides_malformed_raises():
    with pytest.raises(ValueError, match="expected 'key.subkey=value'"):
        apply_overrides(_DummySettings(), ["no_equals_sign"])


class _DummySettings:
    """Minimal stand-in exercising the malformed-format guard before any dump."""

    def model_dump(self, **_kwargs):  # pragma: no cover - the '=' guard fires first
        return {}


# --------------------------------------------------------------------------- #
# Authorship preservation across the round-trip (2026-07-25)
# --------------------------------------------------------------------------- #
def test_override_preserves_model_fields_set():
    """``--override`` must not turn schema defaults into author declarations.

    ``apply_overrides`` round-trips through ``model_dump()``, which serialises
    defaults indistinguishably from author-set values; reconstructing then put
    EVERY field into ``model_fields_set``. Downstream that is not cosmetic —
    ``models/losses/weights.py`` documents "a schema default is not a
    declaration" and reads ``model_fields_set`` to decide it, so the poisoned
    set made two *defaulted* aliased lambdas (``lambda_perceptual`` = 10.0,
    ``lambda_content`` = 0.0, both canonicalising to ``perceptual``) look like
    conflicting author declarations and raised ConfigurationError at build time.

    Net effect: ``spectramr audit`` passed (no override) and ``spectramr train
    --override ...`` died — and SMOKE mode always injects an iteration cap, so
    three ldm_two_stage_ulf_to_hf arms failed on 2026-07-25 (SLURM 7796517).
    Same disease as the 2026-05-29 ``model_domain`` round-trip break, which was
    worked around per-field; this fixes the mechanism.
    """
    from spectramr.config.settings import TrainingSettings

    cfg = _minimal_config()
    # Mirror the production shape: ONE authored lambda, the rest defaulted.
    cfg["losses"] = {"reconstruction": {"enable_ssim": True, "lambda_ssim": 0.5}}
    settings = TrainingSettings(**cfg)

    before = set(settings.losses.reconstruction.model_fields_set)
    assert "lambda_ssim" in before, "precondition: authored"
    assert "lambda_content" not in before, "precondition: default, not authored"
    assert "lambda_perceptual" not in before, "precondition: default, not authored"

    updated = apply_overrides(settings, ["training.max_iterations=3000"])

    after = set(updated.losses.reconstruction.model_fields_set)
    assert after == before, (
        "an override on an unrelated key must not mark loss lambdas as authored; "
        f"newly-marked: {sorted(after - before)}"
    )
    # The override itself IS an authored value on its own section.
    assert "max_iterations" in updated.training.model_fields_set
    assert updated.training.max_iterations == 3000


def test_override_marks_only_the_overridden_path_as_authored():
    """A nested override marks that field authored without touching siblings."""
    from spectramr.config.settings import TrainingSettings

    settings = TrainingSettings(**_minimal_config())
    before = set(settings.optimization.model_fields_set)
    updated = apply_overrides(settings, ["optimization.optimizer.learning_rate=1e-4"])

    # Phase 8: the authored field on `optimization` is now the SUB-BLOCK, and
    # the leaf's authorship lives one level down. Both halves matter -- if only
    # the sub-block were marked, every knob inside it would read as authored.
    assert "optimizer" in updated.optimization.model_fields_set
    assert "learning_rate" in updated.optimization.optimizer.model_fields_set
    assert updated.optimization.optimizer.learning_rate == 1e-4
    # Siblings that were never authored stay unauthored.
    assert set(updated.optimization.model_fields_set) == before | {"optimizer"}


def test_override_preserves_previously_authored_fields():
    """Fields the AUTHOR set (or a migration set) survive the round-trip."""
    from spectramr.config.settings import TrainingSettings

    cfg = _minimal_config()
    cfg["optimization"] = {"learning_rate": 3e-4}  # the FLAT spelling, folded
    settings = TrainingSettings(**cfg)
    assert "learning_rate" in settings.optimization.optimizer.model_fields_set

    updated = apply_overrides(settings, ["training.max_iterations=10"])
    assert (
        "learning_rate" in updated.optimization.optimizer.model_fields_set
    ), "an authored value must not be demoted to 'default' by the round-trip"
    assert updated.optimization.optimizer.learning_rate == 3e-4


class TestDeepSpeedRoundTrip:
    """No override was possible on ANY DeepSpeed arm (issue #1113).

    ``parallel.deepspeed.compile`` and ``.zenflow`` guard against a knob
    declared under a disabled block -- the block is omitted from the ds_config
    entirely, so such a knob reaches nothing (pitfall #15). They implement that
    as an AUTHORSHIP test (``model_fields_set - {"enabled"}``), which is right
    against a YAML document.

    Against the old COMPLETE ``model_dump()`` it was fatal: the rebuild received
    all 8 compile / 9 zenflow fields as present keys, Pydantic marked them
    author-set, and the validator raised on knobs nobody wrote. Every DeepSpeed
    arm therefore died on ``train --override ...`` while ``audit`` passed --
    and smoke mode always injects ``--override training.max_iterations=<cap>``.
    Measured at 7 of 120 randomly sampled ``experiments/inprogress`` arms, all
    DeepSpeed.

    Both halves are pinned here. Dropping the guard would also "fix" the first
    test; only the second says the guard must still catch a real dead knob.
    """

    @staticmethod
    def _deepspeed_config():
        # strategy and deepspeed.enabled must agree -- the parallel block
        # rejects `strategy: deepspeed` with `deepspeed.enabled: false`.
        cfg = _minimal_config()
        cfg["parallel"] = {
            "strategy": "deepspeed",
            "deepspeed": {"enabled": True, "zero_stage": 2},
        }
        return cfg

    def test_override_round_trips_a_deepspeed_arm(self):
        from spectramr.config.settings import TrainingSettings

        settings = TrainingSettings(**self._deepspeed_config())
        # Precondition: the arm declares NEITHER sub-block. Without it the test
        # would pass for the uninteresting reason that there is nothing to flag.
        assert settings.parallel.deepspeed.compile.model_fields_set == set()
        assert settings.parallel.deepspeed.zenflow.model_fields_set == set()
        assert settings.parallel.deepspeed.compile.enabled is False

        updated = apply_overrides(settings, ["training.max_iterations=10"])

        assert updated.training.max_iterations == 10
        # ...and the round-trip must not have invented declarations either.
        assert updated.parallel.deepspeed.compile.model_fields_set == set()
        assert updated.parallel.deepspeed.zenflow.model_fields_set == set()

    def test_override_that_declares_a_dead_knob_still_raises(self):
        """The guard survives the fix: an override CAN author a dead knob.

        This is the anti-regression for "simplify" refactors that would make
        the validators skip the check on a rebuild. Here the author really did
        write ``topk_ratio``, under ``zenflow.enabled: false``, so it really
        does reach nothing -- and must fail loud (pitfall #9), not silently.
        """
        from spectramr.config.settings import TrainingSettings

        settings = TrainingSettings(**self._deepspeed_config())
        with pytest.raises(ValueError, match="topk_ratio"):
            apply_overrides(settings, ["parallel.deepspeed.zenflow.topk_ratio=0.2"])


def test_override_does_not_promote_defaults_to_declarations():
    """The general form of #1113, on a block with no DeepSpeed involvement.

    ``model_fields_set`` is the ONLY thing separating "the author asked for
    this" from "the schema defaulted it", and a complete-dump round-trip erased
    it wholesale. Assert on the root model, where the blast radius was widest:
    an unrelated override must not mark every top-level section as declared.
    """
    from spectramr.config.settings import TrainingSettings

    settings = TrainingSettings(**_minimal_config())
    before = set(settings.model_fields_set)
    undeclared = set(type(settings).model_fields) - before
    assert undeclared, "precondition: _minimal_config leaves some sections defaulted"

    updated = apply_overrides(settings, ["training.max_iterations=7"])

    newly = set(updated.model_fields_set) - before - {"training"}
    assert not newly, f"round-trip promoted defaulted sections to authored: {sorted(newly)}"


class TestLegacySpellingOverrides:
    """A staged (``fold``) rename must keep its legacy ``--override`` working.

    Regression for a break introduced by phase 8 and live until phase 9b.
    ``apply_overrides`` re-validates a dump of the settings, so whenever the arm
    authored the canonical path it is already present. Writing the legacy
    spelling beside it looked to the fold validator like two spellings that
    disagree, and it raised. (The dump became ``exclude_unset`` in #1113, which
    narrows the collision to authored keys but does not remove it -- every
    override below targets a key ``_minimal_config`` authors.)

    That is correct for a YAML document -- there both keys are in the source and
    only a human can say which was meant -- and wrong here, where one of the two
    is an artefact of the dump. The fix translates the path before the write.

    These are not hypothetical spellings: ``optimization.learning_rate`` is the
    example in ``config/overrides.py``'s own docstring and in
    ``.claude/rules/commands.md``, and ``data.max_train_subjects`` is injected by
    ``scripts/ci/smoke_test_vf_configs.sh`` into every cluster smoke run.
    """

    # Every override below must differ from the value ``_minimal_config`` already
    # declares (lr=1e-4, batch_size=2). Overriding a key to the value it already
    # holds makes the fold see AGREEING duplicates, which it drops silently --
    # so the test passes with the translation removed and proves nothing.

    def test_legacy_flat_override_lands_on_the_canonical_field(self):
        from spectramr.config.settings import TrainingSettings

        settings = TrainingSettings(**_minimal_config())
        assert (
            settings.optimization.optimizer.learning_rate == 1e-4
        ), "fixture drifted -- pick an override value that differs from it"
        updated = apply_overrides(settings, ["optimization.learning_rate=7e-4"])
        assert updated.optimization.optimizer.learning_rate == 7e-4

    def test_canonical_override_still_works(self):
        """Both spellings resolve to the same field -- neither shadows the other."""
        from spectramr.config.settings import TrainingSettings

        settings = TrainingSettings(**_minimal_config())
        legacy = apply_overrides(settings, ["optimization.learning_rate=2e-4"])
        canon = apply_overrides(settings, ["optimization.optimizer.learning_rate=2e-4"])
        assert (
            legacy.optimization.optimizer.learning_rate
            == canon.optimization.optimizer.learning_rate
            == 2e-4
        )

    def test_legacy_override_marks_the_canonical_field_authored(self):
        """Authorship must follow the value, not the spelling.

        ``models/losses/weights.py`` reads ``model_fields_set`` to tell a
        declaration from a default; crediting the legacy leaf would leave the
        real field looking undeclared.
        """
        from spectramr.config.settings import TrainingSettings

        settings = TrainingSettings(**_minimal_config())
        updated = apply_overrides(settings, ["optimization.learning_rate=7e-4"])
        assert "optimizer" in updated.optimization.model_fields_set
        assert "learning_rate" in updated.optimization.optimizer.model_fields_set

    def test_data_block_legacy_override_lands(self):
        """The same break existed in ``data:`` after phase 9a."""
        from spectramr.config.settings import TrainingSettings

        settings = TrainingSettings(**_minimal_config())
        assert settings.data.loader.batch_size == 2, "fixture drifted"
        updated = apply_overrides(settings, ["data.batch_size=8"])
        assert updated.data.loader.batch_size == 8

    def test_every_fold_record_is_overridable(self):
        """Totality, not a sample.

        A fold record the CLI cannot express is a key that YAML accepts and
        ``--override`` rejects -- the two surfaces disagreeing about what the
        config language is. Asserting per-record keeps new folds honest without
        anyone remembering to extend this file.
        """
        from spectramr.config.schemas.renames import RENAMES, canonical_override_path

        folds = [r for r in RENAMES.values() if r.posture == "fold"]
        assert folds, "no fold records -- this test needs a new anchor"
        for rec in folds:
            assert canonical_override_path(rec.legacy) == rec.canonical, (
                f"--override {rec.legacy}=... would write the legacy path and "
                "collide with the dumped canonical default"
            )

    def test_a_retired_raise_record_still_raises(self):
        """Translation is for ``fold`` only.

        A ``raise`` record must fall through untranslated so the owning block's
        reject validator produces the message that names the replacement --
        silently rewriting it would hide the fact that the key is gone.
        """
        from spectramr.config.schemas.renames import RENAMES, canonical_override_path

        retired = [r for r in RENAMES.values() if r.posture == "raise"]
        assert retired, "no raise records left -- this test needs a new anchor"
        for rec in retired:
            assert canonical_override_path(rec.legacy) == rec.legacy


def _settings():
    """A validated ``TrainingSettings`` — ``apply_overrides`` takes the object,
    not the dict ``_minimal_config`` returns."""
    from spectramr.config.settings import TrainingSettings

    return TrainingSettings(**_minimal_config())


@pytest.mark.parametrize("spelling", ["none", "None", "NONE"])
def test_none_reaches_a_literal_field_that_rejects_none(spelling):
    """#1480: `parallel.strategy` declares the string 'none' and refuses None."""
    updated = apply_overrides(_settings(), [f"parallel.strategy={spelling}"])
    assert updated.parallel.strategy == "none"


def test_none_still_clears_an_optional_field():
    updated = apply_overrides(_settings(), ["data.source.test_index_path=none"])
    assert updated.data.source.test_index_path is None


def test_null_is_never_rewritten_to_the_string():
    with pytest.raises(ValueError, match=r"parallel\.strategy"):
        apply_overrides(_settings(), ["parallel.strategy=null"])


class TestOverrideEcho:
    """An applied ``-O`` must be visible in a normally-configured run's log.

    It was echoed only at DEBUG, so the most consequential thing a caller does
    to a run left no trace at any level anyone actually runs at -- and nothing
    downstream re-states it either.
    """

    def test_applied_overrides_are_echoed_at_info(self, caplog):
        config = _settings()
        with caplog.at_level("INFO", logger="spectramr.config.overrides"):
            apply_overrides(config, ["training.max_iterations=40"])
        info = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
        assert any("Overrides applied (1)" in m for m in info)
        assert any("training.max_iterations=40" in m for m in info)

    def test_echo_names_the_canonical_destination_after_a_fold(self, caplog):
        """A legacy spelling lands somewhere else; the echo has to say where,
        or it confirms a write to a path that was never written."""
        from spectramr.config.schemas.renames import RENAMES

        folded = [r for r in RENAMES.values() if r.posture == "fold"]
        assert folded, "no fold records left -- this test needs a new anchor"
        rec = folded[0]
        config = _settings()
        # The echo precedes re-validation, so a value that does not validate for
        # this particular key still exercises the path under test.
        with caplog.at_level("INFO", logger="spectramr.config.overrides"), suppress(
            ValueError
        ):
            apply_overrides(config, [f"{rec.legacy}=1"])
        info = " ".join(r.getMessage() for r in caplog.records)
        assert f"{rec.legacy}→{rec.canonical}" in info

    def test_no_echo_when_no_overrides_are_passed(self, caplog):
        config = _settings()
        with caplog.at_level("INFO", logger="spectramr.config.overrides"):
            apply_overrides(config, [])
        assert not [r for r in caplog.records if "Overrides applied" in r.getMessage()]


class TestOverrideProvenance:
    """``apply_overrides`` must leave a MACHINE-READABLE trace, not just a log line.

    Until this landed, the only record an override left was the human INFO echo
    covered by ``TestOverrideEcho`` above. Nothing downstream could ask "did the
    caller move this field, or did the arm declare it?" -- and
    ``model_fields_set`` cannot answer it, because the ``exclude_unset`` dump
    documented in ``apply_overrides`` marks a YAML declaration and an override
    identically *by construction*.

    The consumer is the launch banner in ``pipelines/training_loop.py``: a
    4-GPU run on 2026-08-21 was launched with ``-O training.max_iterations=5000``
    while sanity-check mode independently forces the budget to a hardcoded 5000,
    and the log printed the number without its origin.
    """

    def test_an_overridden_path_is_recorded(self):
        updated = apply_overrides(_settings(), ["training.max_iterations=40"])
        assert "training.max_iterations" in applied_override_paths(updated)

    def test_an_untouched_settings_object_records_nothing(self):
        """The negative half. If this ever returned a non-empty tuple the banner
        would label every declared budget an override -- worse than silence,
        because it is confidently wrong."""
        assert applied_override_paths(_settings()) == ()
        assert applied_override_paths(apply_overrides(_settings(), [])) == ()

    def test_the_record_is_the_canonical_path_not_the_typed_one(self):
        """A fold moves the key. Recording the spelling the caller TYPED would
        make the banner miss an override that did land, because every consumer
        asks about the canonical path."""
        from spectramr.config.schemas.renames import RENAMES

        folded = [
            r
            for r in RENAMES.values()
            if r.posture == "fold" and r.canonical == "optimization.optimizer.learning_rate"
        ]
        assert folded, "no fold record for the anchor path -- pick a new anchor"
        updated = apply_overrides(_settings(), [f"{folded[0].legacy}=7e-4"])
        paths = applied_override_paths(updated)
        assert folded[0].canonical in paths
        assert folded[0].legacy not in paths

    def test_the_record_survives_a_second_apply_overrides_call(self):
        """Private attrs do NOT survive the dump/re-validate round trip inside
        ``apply_overrides``, so a chained call would silently drop everything the
        first one recorded unless it merges explicitly."""
        once = apply_overrides(_settings(), ["training.max_iterations=40"])
        twice = apply_overrides(once, ["data.loader.batch_size=8"])
        paths = applied_override_paths(twice)
        assert "training.max_iterations" in paths
        assert "data.loader.batch_size" in paths

    def test_the_source_object_is_not_mutated(self):
        """``apply_overrides`` returns a NEW object (CLAUDE.md #1: config is
        frozen). Stamping the record must not reach back into the caller's."""
        base = _settings()
        apply_overrides(base, ["training.max_iterations=40"])
        assert applied_override_paths(base) == ()

    def test_the_record_is_invisible_to_serialisation(self):
        """A private attr, not a field, precisely so it stays out of
        ``model_dump`` / the JSON Schema / provenance. If it leaked into the dump
        it would round-trip back through ``apply_overrides`` as config, and with
        ``extra="forbid"`` an arm could spell it in YAML."""
        updated = apply_overrides(_settings(), ["training.max_iterations=40"])
        assert "_override_paths" not in updated.model_dump()
        assert "_override_paths" not in updated.model_json_schema()["properties"]

    def test_a_non_settings_object_reports_nothing_rather_than_raising(self):
        """``applied_override_paths`` is called from the training loop against
        whatever config object it was handed. "Nothing was overridden" is the
        truthful answer for an object no override ever touched; an
        ``AttributeError`` at the launch banner would not be."""
        assert applied_override_paths(object()) == ()
