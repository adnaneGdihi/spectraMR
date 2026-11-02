"""Tests for configuration enumeration types.

Targets ``spectramr.config.schemas.enums``. Every enum is a ``str`` Enum so
the value can flow through Pydantic validation as either the enum
member or the bare string.

Categories:

- Each enum exposes documented values
- ``str`` mixin: ``EnumMember.value == "string"`` and `EnumMember == "string"`
- Enum coercion from valid string succeeds
- Coercion from invalid string raises ``ValueError``
"""

from __future__ import annotations

import pytest

from spectramr.config.schemas.enums import (
    AccelerationSchedule,
    ActivationFunction,
    ArtifactOrigin,
    Axis,
    DataFormatTypes,
    DatasetType,
    GANLossType,
    GeneratorTypes,
    GradientClipMethod,
    InputType,
    KANType,
    LRScheduler,
    LogLevel,
    LossTypes,
    Maturity,
    MetricMode,
    Modality,
    NoiseSchedule,
    NormalizationType,
    OptimizationTypes,
    OptimizerType,
    PredictionType,
    ReportTask,
    Regime,
    SchedulerTypes,
    Task,
    TrainingMode,
    TrainingModeTypes,
    VolumeFormat,
)

# ---------------------------------------------------------------------------
# Each enum should be a str-Enum (Pydantic-friendly)
# ---------------------------------------------------------------------------

ALL_ENUMS = [
    TrainingMode,
    GANLossType,
    LRScheduler,
    NoiseSchedule,
    AccelerationSchedule,
    OptimizerType,
    DatasetType,
    VolumeFormat,
    PredictionType,
    KANType,
    InputType,
    LogLevel,
    GradientClipMethod,
    ActivationFunction,
    NormalizationType,
    MetricMode,
    LossTypes,
    SchedulerTypes,
    GeneratorTypes,
    TrainingModeTypes,
    DataFormatTypes,
    OptimizationTypes,
    # Workflow vocabulary (imaging-regime × task contract)
    Modality,
    Regime,
    Task,
    Axis,
    ArtifactOrigin,
    Maturity,
]


@pytest.mark.parametrize("enum_class", ALL_ENUMS, ids=lambda e: e.__name__)
def test_enum_is_str_subclass(enum_class) -> None:
    """All enums inherit from ``str`` for Pydantic round-trip compatibility."""
    assert issubclass(enum_class, str)


@pytest.mark.parametrize("enum_class", ALL_ENUMS, ids=lambda e: e.__name__)
def test_enum_members_are_string_equal(enum_class) -> None:
    """``EnumMember == 'value_string'`` (str-Enum equality)."""
    for member in enum_class:
        assert member == member.value


@pytest.mark.parametrize("enum_class", ALL_ENUMS, ids=lambda e: e.__name__)
def test_enum_string_values_are_unique(enum_class) -> None:
    """No duplicate string values within an enum."""
    values = [m.value for m in enum_class]
    assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# TrainingMode — content checks
# ---------------------------------------------------------------------------


def test_training_mode_covers_main_paradigms() -> None:
    """All main paradigms covered: GAN, diffusion, VAE, SSL, MAE, …"""
    expected = {"gan", "diffusion", "vae", "reconstruction", "ssl", "mae"}
    actual = {m.value for m in TrainingMode}
    assert expected <= actual


def test_report_task_includes_calibration() -> None:
    """The conformal-calibration arms set ``reporting.task: calibration``; the
    enum must accept it or those configs raise ValidationError at load time and
    cannot run at all (exp_p1 / exp_vf_conformal_calib / eval_c5 regression)."""
    assert ReportTask("calibration") == ReportTask.CALIBRATION
    assert "calibration" in {m.value for m in ReportTask}


def test_training_mode_coercion_from_string() -> None:
    """Valid string can be coerced to enum via ``TrainingMode('value')``."""
    assert TrainingMode("gan") == TrainingMode.GAN


def test_training_mode_invalid_string_raises() -> None:
    """Invalid value raises ``ValueError``."""
    with pytest.raises(ValueError):
        TrainingMode("not_a_paradigm")


# ---------------------------------------------------------------------------
# GANLossType
# ---------------------------------------------------------------------------


def test_gan_loss_types_include_wgan_gp_and_hinge() -> None:
    """``wgan-gp`` and ``hinge`` are in the GAN loss enum."""
    values = {m.value for m in GANLossType}
    assert "wgan-gp" in values
    assert "hinge" in values


# ---------------------------------------------------------------------------
# LRScheduler
# ---------------------------------------------------------------------------


def test_lr_scheduler_includes_documented_strategies() -> None:
    """Cosine, step, linear, warmup are present."""
    values = {m.value for m in LRScheduler}
    assert {"cosine", "step", "linear", "warmup"} <= values


# ---------------------------------------------------------------------------
# OptimizerType
# ---------------------------------------------------------------------------


def test_optimizer_type_includes_adam_adamw_sgd() -> None:
    """The three default optimisers are registered."""
    values = {m.value for m in OptimizerType}
    assert {"adam", "adamw", "sgd"} <= values


# ---------------------------------------------------------------------------
# PredictionType (diffusion)
# ---------------------------------------------------------------------------


def test_prediction_type_three_modes() -> None:
    """Diffusion supports epsilon, sample, v_prediction."""
    values = {m.value for m in PredictionType}
    assert values == {"epsilon", "sample", "v_prediction"}


# ---------------------------------------------------------------------------
# NormalizationType
# ---------------------------------------------------------------------------


def test_normalization_type_includes_none() -> None:
    """``NONE`` is a valid normalisation strategy."""
    assert NormalizationType.NONE == "none"


# ---------------------------------------------------------------------------
# MetricMode
# ---------------------------------------------------------------------------


def test_metric_mode_min_max() -> None:
    """``MetricMode`` is exactly ``{min, max}``."""
    assert {m.value for m in MetricMode} == {"min", "max"}


# ---------------------------------------------------------------------------
# Round-trip: each enum can be created from its own value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("enum_class", ALL_ENUMS, ids=lambda e: e.__name__)
def test_enum_round_trip(enum_class) -> None:
    """``EnumClass(member.value) is member`` for every member."""
    for member in enum_class:
        assert enum_class(member.value) is member


def test_maturity_docstring_describes_the_enforced_live_rule() -> None:
    """The Maturity docstring must match what the ledger actually asserts.

    A docstring describing a rule nothing enforces is the documentation form of a
    facade: it reads as a guarantee and guarantees nothing. This has drifted
    twice already — it once claimed LIVE required "strategy + loss + metric",
    which nothing checked and which ``mri_structural`` contradicted.

    The rule enforced by test_maturity_ledger.py, since the 2026-07-16 widening:
    a registered forward model (operator OR signal model), a regime-tagged
    strategy, AND regime-tagged metrics. Pinned on the clauses, not exact prose.
    """
    from spectramr.config.schemas.enums import Maturity

    doc = Maturity.__doc__ or ""
    lines = doc.splitlines()
    start = next(i for i, line in enumerate(lines) if "``LIVE``" in line)
    # The entry wraps over several INDENTED lines, so collapse whitespace after
    # joining: a naive " ".join leaves the docstring's indentation inside the
    # text and "signal\n      model" then never matches "signal model".
    live_entry = " ".join(" ".join(lines[start : start + 4]).lower().split())

    for clause in ("forward model", "strategy", "metrics"):
        assert clause in live_entry, (
            f"LIVE's documented rule does not name the {clause!r} clause the "
            f"ledger enforces. Got: {live_entry.strip()!r}"
        )
    assert "signal model" in live_entry, (
        "LIVE accepts a signal model as well as a linear operator — that "
        "widening is what let mri_perfusion reach LIVE without registering a "
        "fake adjoint. A docstring naming only the operator sends the next "
        "reader back to the rule that rewarded the facade."
    )


def test_maturity_docstring_records_why_losses_are_not_a_live_clause() -> None:
    """The most likely 'helpful' regression is re-adding a losses clause.

    It looks like a tightening and is actually a lie: mri_structural trains on
    agnostic L1/SSIM, which is correct rather than a gap, so the clause could
    only be satisfied by mis-tagging L1 as structural-only. The docstring has to
    carry the reason, because the rule looks arbitrary without it.
    """
    from spectramr.config.schemas.enums import Maturity

    doc = (Maturity.__doc__ or "").lower()
    assert "does not require tagged **losses**".lower() in doc or (
        "not require tagged" in doc and "losses" in doc
    ), "the docstring must say, and justify, that LIVE excludes a losses clause"
    assert "structural" in doc, "the justification names mri_structural as the case"


# ---------------------------------------------------------------------------
# OptimizerType — the advertised optimizer vocabulary
#
# LARS and LAMB sat in this enum with no implementation anywhere from its
# introduction until 2026-07-30; test_optimizer_type_coverage.py xfailed them as
# "needs apex / custom". An enum member with no dispatch entry is an advertised
# knob with no reader (pitfall #15) wearing a type annotation.
# ---------------------------------------------------------------------------


class TestOptimizerTypeVocabulary:
    """The enum is the SSOT the registry must cover."""

    def test_values_are_lowercase_and_unique(self) -> None:
        from spectramr.config.schemas.enums import OptimizerType

        values = [m.value for m in OptimizerType]
        assert values == [v.lower() for v in values]
        assert len(values) == len(set(values))

    def test_names_frozenset_tracks_the_enum(self) -> None:
        """OPTIMIZER_NAMES is derived, not restated — a copy could drift."""
        from spectramr.config.schemas.enums import OPTIMIZER_NAMES, OptimizerType

        assert OPTIMIZER_NAMES == {m.value for m in OptimizerType}

    def test_covers_the_three_declared_tiers(self) -> None:
        from spectramr.config.schemas.enums import OPTIMIZER_NAMES

        torch_tier = {"adam", "adamw", "sgd", "rmsprop", "adamax", "nadam", "radam"}
        in_repo_tier = {"lars", "lamb", "lion"}
        extra_tier = {"adam8bit", "adamw8bit", "schedulefree_adamw", "schedulefree_sgd"}
        assert torch_tier <= OPTIMIZER_NAMES
        assert in_repo_tier <= OPTIMIZER_NAMES
        assert extra_tier <= OPTIMIZER_NAMES

    def test_every_alias_resolves_to_a_real_name(self) -> None:
        """An alias pointing at a nonexistent canonical name would make the
        validator reject a spelling it advertises."""
        from spectramr.config.schemas.enums import OPTIMIZER_ALIASES, OPTIMIZER_NAMES

        for alias, canonical in OPTIMIZER_ALIASES.items():
            assert canonical in OPTIMIZER_NAMES, (
                f"alias {alias!r} maps to {canonical!r}, which is not an "
                "OptimizerType value"
            )

    def test_no_alias_shadows_a_canonical_name(self) -> None:
        """A canonical name appearing as an alias key is a silent redirect."""
        from spectramr.config.schemas.enums import OPTIMIZER_ALIASES, OPTIMIZER_NAMES

        assert not (set(OPTIMIZER_ALIASES) & OPTIMIZER_NAMES)

    def test_sam_is_absent_until_its_two_pass_stepper_exists(self) -> None:
        """SAM needs two forward+backward passes per step. Against the current
        single-backward seam a lone ``step()`` would perturb the weights and
        never restore them, so the run would train on w + rho*g and report
        success -- strictly worse than the name being absent."""
        from spectramr.config.schemas.enums import OPTIMIZER_ALIASES, OPTIMIZER_NAMES

        assert "sam" not in OPTIMIZER_NAMES
        assert "sam" not in OPTIMIZER_ALIASES.values()

    def test_lookahead_is_absent_because_it_is_a_wrapper(self) -> None:
        """Lookahead wraps a base optimizer; as a standalone name it would be
        meaningless. It lives at optimization.optimizer.lookahead instead."""
        from spectramr.config.schemas.enums import OPTIMIZER_NAMES

        assert "lookahead" not in OPTIMIZER_NAMES

    def test_weight_decay_free_optimizers_are_named_deliberately(self) -> None:
        """lbfgs and sparseadam accept no weight_decay. They were reachable via
        the reflective getattr(torch.optim, ...) path and TypeError'd on the
        unconditionally-forwarded weight_decay default; naming them makes that a
        designed, tested drop instead of a crash."""
        from spectramr.config.schemas.enums import OPTIMIZER_NAMES

        assert {"lbfgs", "sparseadam"} <= OPTIMIZER_NAMES


def test_acceleration_schedule_has_log() -> None:
    assert AccelerationSchedule("log") is AccelerationSchedule.LOG
