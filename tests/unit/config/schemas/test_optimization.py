"""Tests for ``OptimizationConfigSchema``.

Targets ``spectramr.config.schemas.optimization``. v6.0 nested optimisation
config: learning rates, optimiser kwargs, AMP, gradient clipping,
scheduler, torch.compile knobs.

Categories:

- Defaults match documented values
- Constraints: ``learning_rate > 0``, ``weight_decay >= 0``, betas in
  ``[0, 1]``, ``gradient_accumulation_steps > 0``
- ``frozen=True`` (mutation raises)
- ``extra="forbid"`` (unknown keys REJECTED — see ``test_extra_keys_rejected``;
  this line used to claim ``extra="ignore"``, contradicting the test below)
- ``optimizer`` property aliases ``optimizer_type``
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.optimization import (
    CompileConfigSchema,
    OptimizationConfigSchema,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults() -> None:
    """Default field values match the documented contract."""
    cfg = OptimizationConfigSchema()
    assert cfg.optimizer.learning_rate == 1e-5
    assert cfg.optimizer.type == "adamw"
    assert cfg.optimizer.weight_decay == 1e-4
    assert cfg.optimizer.beta1 == 0.5
    assert cfg.optimizer.beta2 == 0.999
    assert cfg.precision.enabled is False
    assert cfg.gradient.accumulation_steps == 1
    assert cfg.lr_scheduler_strategy == "cosine"
    assert cfg.warmup_steps == 0


def test_the_legacy_accumulate_spelling_raises() -> None:
    """``accumulate_grad_steps`` was a second spelling of one number.

    It used to be folded into the canonical field by a hand-written validator,
    so which one won depended on declaration order rather than intent. The
    rename retired it; this pins the raise, and that the message names the
    replacement rather than only rejecting the key.
    """
    with pytest.raises(ValidationError) as exc:
        OptimizationConfigSchema(accumulate_grad_steps=4)
    msg = str(exc.value)
    assert "optimization.gradient.accumulation_steps" in msg
    assert "migrate_config_keys.py" in msg


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def test_zero_learning_rate_rejected() -> None:
    """``gt=0`` constraint on ``learning_rate``."""
    with pytest.raises(ValidationError):
        OptimizationConfigSchema(learning_rate=0.0)


def test_negative_weight_decay_rejected() -> None:
    """``ge=0`` constraint on ``weight_decay``."""
    with pytest.raises(ValidationError):
        OptimizationConfigSchema(weight_decay=-0.1)


def test_beta1_above_one_rejected() -> None:
    """``le=1`` constraint on ``beta1``."""
    with pytest.raises(ValidationError):
        OptimizationConfigSchema(beta1=1.5)


def test_beta2_below_zero_rejected() -> None:
    """``ge=0`` constraint on ``beta2``."""
    with pytest.raises(ValidationError):
        OptimizationConfigSchema(beta2=-0.1)


def test_zero_accumulate_grad_steps_rejected() -> None:
    """``gt=0`` constraint on ``accumulate_grad_steps``."""
    with pytest.raises(ValidationError):
        OptimizationConfigSchema(accumulate_grad_steps=0)


def test_zero_memory_monitoring_interval_rejected() -> None:
    """``gt=0`` constraint on ``memory_monitoring_interval``."""
    with pytest.raises(ValidationError):
        OptimizationConfigSchema(memory_monitoring_interval=0)


def test_negative_warmup_steps_rejected() -> None:
    """``ge=0`` constraint on ``warmup_steps``."""
    with pytest.raises(ValidationError):
        OptimizationConfigSchema(warmup_steps=-5)


def test_memory_safety_margin_above_one_rejected() -> None:
    """``le=1`` constraint on ``memory_safety_margin``."""
    with pytest.raises(ValidationError):
        OptimizationConfigSchema(memory_safety_margin=1.5)


# ---------------------------------------------------------------------------
# Optional fields
# ---------------------------------------------------------------------------


def test_separate_generator_discriminator_lrs() -> None:
    """Optional generator/discriminator LRs accepted when set."""
    cfg = OptimizationConfigSchema(
        generator_learning_rate=2e-4,
        discriminator_learning_rate=1e-4,
    )
    assert cfg.optimizer.generator_learning_rate == 2e-4
    assert cfg.optimizer.discriminator_learning_rate == 1e-4


def test_zero_gen_lr_rejected() -> None:
    """Optional ``generator_learning_rate`` still has ``gt=0``."""
    with pytest.raises(ValidationError):
        OptimizationConfigSchema(generator_learning_rate=0.0)


def test_gradient_clip_value_can_be_none() -> None:
    """``gradient_clip_value=None`` is permitted (legacy YAML compat)."""
    cfg = OptimizationConfigSchema(gradient_clip_value=None)
    assert cfg.gradient.clip.value is None


# ---------------------------------------------------------------------------
# `optimizer` is a sub-block, not a string alias
# ---------------------------------------------------------------------------


def test_optimizer_is_the_sub_block_not_a_string() -> None:
    """`OptimizationConfigSchema.optimizer` used to be a @property returning
    `optimizer_type`. Phase 8 gives the name to the sub-block, so a caller that
    still expects a string gets an object -- which is why the two log sites in
    pipelines/make.py had to move to `.optimizer.type` in the same change."""
    cfg = OptimizationConfigSchema(optimizer_type="sgd")
    assert not isinstance(cfg.optimizer, str)
    assert cfg.optimizer.type == "sgd"


# ---------------------------------------------------------------------------
# Frozen + extra
# ---------------------------------------------------------------------------


def test_frozen() -> None:
    """Mutation raises ``ValidationError`` (Pydantic frozen)."""
    cfg = OptimizationConfigSchema()
    with pytest.raises(ValidationError):
        cfg.optimizer.learning_rate = 1e-3  # type: ignore[misc]


def test_extra_keys_rejected() -> None:
    """Unknown keys raise ``ValidationError`` (``extra='forbid'``, NN#1).

    OptimizationConfigSchema sets ``model_config = {"extra": "forbid", ...}`` so a
    typo'd / orphan key must fail loudly rather than be silently dropped — a
    silent drop is exactly the kind of unread-knob hazard NN#1/NN#3 forbid.
    """
    with pytest.raises(ValidationError):
        OptimizationConfigSchema(some_unknown_key="value")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Flat scheduler knobs are CONSUMED (issue #533)
# ---------------------------------------------------------------------------


def test_flat_scheduler_knobs_reach_the_resolver() -> None:
    """``scheduler_type``/``T_max``/``eta_min`` were documented as inert.

    They now feed ``resolve_scheduler_spec``, so a config that declares them at
    the optimization-block top level gets the schedule it asked for instead of
    ``CosineAnnealingLR(T_max=100)``.
    """
    from spectramr.infrastructure.training.scheduler_resolution import (
        resolve_scheduler_spec,
    )

    cfg = OptimizationConfigSchema(
        scheduler_type="cosine", T_max=4242, eta_min=1e-7, scheduler={}
    )
    spec = resolve_scheduler_spec(cfg, max_iterations=30000)
    assert spec is not None
    assert spec.name == "cosine_annealing"
    assert spec.params == {"T_max": 4242, "eta_min": 1e-7}


def test_scheduler_none_means_no_scheduler() -> None:
    from spectramr.infrastructure.training.scheduler_resolution import (
        resolve_scheduler_spec,
    )

    cfg = OptimizationConfigSchema()
    assert cfg.scheduler is None
    assert resolve_scheduler_spec(cfg, max_iterations=100) is None


# ---------------------------------------------------------------------------
# optimizer_type — the closed vocabulary
#
# Was a bare ``str``. Three disagreeing name lists existed (the 6-member
# OptimizerType enum, a Literal["adam","sgd","adamw"] in domain/, and the
# 7-name OptimizerRegistry), and the field consulted none of them, so a typo
# reached ``getattr(torch.optim, ...)`` and failed at build time.
# ---------------------------------------------------------------------------


class TestOptimizerTypeVocabulary:
    @pytest.mark.parametrize(
        "value", ["adam", "adamw", "sgd", "rmsprop", "lars", "lamb", "lion"]
    )
    def test_accepts_advertised_names(self, value: str) -> None:
        assert OptimizationConfigSchema(optimizer_type=value).optimizer.type == value

    @pytest.mark.parametrize(
        "given,expected",
        [
            ("AdamW", "adamw"),
            ("ADAM", "adam"),
            ("  sgd  ", "sgd"),
            ("adam_w", "adamw"),
            ("adamw_8bit", "adamw8bit"),
        ],
    )
    def test_normalises_case_whitespace_and_aliases(
        self, given: str, expected: str
    ) -> None:
        assert OptimizationConfigSchema(optimizer_type=given).optimizer.type == expected

    def test_rejects_an_unknown_name_at_load_time(self) -> None:
        with pytest.raises(ValidationError, match="Unknown optimizer type"):
            OptimizationConfigSchema(optimizer_type="wumbo")

    def test_stays_a_plain_str_after_validation(self) -> None:
        """Deliberately NOT annotated as the enum: three live sites consume this
        as a bare string, and ``str()`` on a (str, Enum) member yields
        'OptimizerType.ADAMW', not 'adamw'."""
        value = OptimizationConfigSchema(optimizer_type="adamw").optimizer.type
        assert type(value) is str

    def test_the_enum_and_the_field_agree(self) -> None:
        """Every enum member must be an accepted field value — otherwise the
        advertised vocabulary and the validated one have re-forked."""
        from spectramr.config.schemas.enums import OptimizerType

        for member in OptimizerType:
            assert (
                OptimizationConfigSchema(optimizer_type=member.value).optimizer.type
                == member.value
            )


# ---------------------------------------------------------------------------
# torch.compile knobs — closed so a bad value cannot become an eager run
#
# An unknown compile_mode was accepted here, raised inside torch.compile, and
# was then swallowed by ModelBuilder.compile's blanket ``except`` into an
# eager-mode run reporting success (#619 F1+F2).
# ---------------------------------------------------------------------------


class TestCompileVocabulary:
    @pytest.mark.parametrize(
        "mode",
        ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
    )
    def test_accepts_real_modes(self, mode: str) -> None:
        assert OptimizationConfigSchema(compile_mode=mode).compile.mode == mode

    def test_rejects_an_unknown_mode(self) -> None:
        with pytest.raises(ValidationError):
            OptimizationConfigSchema(compile_model=True, compile_mode="turbo")

    def test_accepts_the_default_backend(self) -> None:
        assert (
            OptimizationConfigSchema(compile_backend="inductor").compile.backend
            == "inductor"
        )

    def test_rejects_an_unknown_backend(self) -> None:
        pytest.importorskip("torch")
        with pytest.raises(ValidationError):
            OptimizationConfigSchema(compile_model=True, compile_backend="wumbo")


# ---------------------------------------------------------------------------
# param_group_overrides — was dict[str, Any] and read by nothing
# ---------------------------------------------------------------------------


class TestParamGroupOverrides:
    def test_coerces_to_a_typed_sub_schema(self) -> None:
        cfg = OptimizationConfigSchema(
            optimizer={"param_groups": {"residual_head": {"learning_rate": 1e-5}}}
        )
        assert cfg.optimizer.param_groups is not None
        override = cfg.optimizer.param_groups["residual_head"]
        assert override.learning_rate == 1e-5
        assert override.freeze is False

    def test_rejects_a_typo_inside_the_override(self) -> None:
        """The whole point of typing it: ``dict[str, Any]`` validated nothing."""
        # `match=` for the same reason as the two-homes case below: the flat
        # spelling was promoted to `raise` (2026-08-18) and a bare
        # `raises(ValidationError)` would catch the rename, not the typo.
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            OptimizationConfigSchema(
                optimizer={"param_groups": {"head": {"learnng_rate": 1e-5}}}
            )

    def test_rejects_two_homes_for_one_learning_rate(self) -> None:
        with pytest.raises(ValidationError, match="learning_rate or lr_multiplier"):
            OptimizationConfigSchema(
                optimizer={
                    "param_groups": {
                        "head": {"learning_rate": 1e-5, "lr_multiplier": 2.0}
                    }
                }
            )

    def test_freeze_is_distinct_from_a_zero_learning_rate(self) -> None:
        """A zero-LR group still accrues optimizer state and is still
        decoupled-weight-decayed by AdamW, so freeze cannot be lr=0."""
        cfg = OptimizationConfigSchema(
            optimizer={"param_groups": {"encoder": {"freeze": True}}}
        )
        assert cfg.optimizer.param_groups is not None
        override = cfg.optimizer.param_groups["encoder"]
        assert override.freeze is True
        assert override.learning_rate is None


# ---------------------------------------------------------------------------
# Newly-typed knobs that previously existed only via the optimizer_kwargs hatch
# ---------------------------------------------------------------------------


class TestPreviouslyUntypedOptimizerKnobs:
    def test_nesterov_and_amsgrad_are_declarable_and_default_absent(self) -> None:
        cfg = OptimizationConfigSchema()
        assert cfg.optimizer.nesterov is None and cfg.optimizer.amsgrad is None
        assert OptimizationConfigSchema(optimizer={"nesterov": True}).optimizer.nesterov is True
        assert OptimizationConfigSchema(optimizer={"amsgrad": True}).optimizer.amsgrad is True

    def test_lookahead_is_a_wrapper_sub_block_not_an_optimizer_name(self) -> None:
        """A Lookahead with no inner optimizer is meaningless, so it cannot be an
        ``optimizer_type`` value."""
        from spectramr.config.schemas.enums import OPTIMIZER_NAMES

        assert "lookahead" not in OPTIMIZER_NAMES
        cfg = OptimizationConfigSchema(
            optimizer={
                "lookahead": {"enabled": True, "sync_period": 6, "alpha": 0.8}
            }
        )
        assert cfg.optimizer.lookahead.enabled
        assert cfg.optimizer.lookahead.sync_period == 6
        assert cfg.optimizer.lookahead.alpha == 0.8

    def test_lookahead_defaults_are_disabled(self) -> None:
        assert OptimizationConfigSchema().optimizer.lookahead.enabled is False

    def test_lookahead_period_is_not_named_k(self) -> None:
        """A one-letter field can never be shown to have a consumer.

        The unconsumed-knob scanner matches bare identifiers of 3+ characters, so
        a field named ``k`` sits in the ratchet permanently as a false positive no
        matter how thoroughly it is wired. Naming is load-bearing here.
        """
        from spectramr.config.schemas.optimization import LookaheadConfigSchema

        fields = set(LookaheadConfigSchema.model_fields)
        assert "k" not in fields
        assert "sync_period" in fields

    def test_lookahead_rejects_out_of_range_alpha(self) -> None:
        with pytest.raises(ValidationError):
            OptimizationConfigSchema(lookahead={"alpha": 1.5})

    def test_sam_block_is_not_declarable_while_its_stepper_does_not_exist(
        self,
    ) -> None:
        """``optimization.sam`` must stay UNMOUNTED until SAM actually runs.

        The block previously shipped with three knobs and no reader --
        ``SharpnessAwareStepper``, the consumer its own docstring named, was
        never written. So ``sam.rho: 0.1`` validated, was stamped into
        resolved_config.json, and changed nothing about the run.

        The superseded version of this test asserted the block's *defaults*, so
        it passed for the entire lifetime of the facade: asserting that an
        inert knob holds its default value confirms only that Pydantic works.
        This asserts the property that actually matters -- declaring it FAILS
        loudly -- which is what makes the removal self-enforcing if someone
        re-adds the schema without the stepper.
        """
        assert "sam" not in OptimizationConfigSchema.model_fields
        with pytest.raises(ValidationError):
            OptimizationConfigSchema(sam={"rho": 0.1, "adaptive": True})

    def test_sam_is_not_a_legal_optimizer_type(self) -> None:
        """The v6.1 reference template used to advertise 'sam' in its in-repo
        list while the enum rejected it -- the canonical copy-from surface
        documenting an illegal value."""
        with pytest.raises(ValidationError):
            OptimizationConfigSchema(optimizer_type="sam")


def test_no_docstring_still_names_the_phantom_optimizer_setup_mixin() -> None:
    """``OptimizerSetupMixin._build_optimizer`` was named as the consumer of
    betas / eps / momentum / param_group_overrides. That class has never existed
    in this repo, and ``betas`` was in fact read by nothing while 27 experiment
    YAMLs set it. A docstring naming a nonexistent consumer is worse than none:
    it makes an unwired knob look wired."""
    from spectramr.config.schemas.optimization import OptimizerConfigSchema

    fields = OptimizerConfigSchema.model_fields
    for name in ("betas", "eps", "momentum", "param_groups"):
        description = fields[name].description or ""
        assert (
            "OptimizerSetupMixin" not in description
        ), f"optimization.optimizer.{name} still cites the nonexistent mixin"


# ---------------------------------------------------------------------------
# Phase 8: the block is six groups, not 44 scalars
# ---------------------------------------------------------------------------
class TestSubBlockDecomposition:
    def test_the_flat_spelling_still_loads(self) -> None:
        """826 arms declare `optimization.learning_rate`. They must keep working
        while the corpus migrates on its own schedule."""
        cfg = OptimizationConfigSchema(learning_rate=3e-4, optimizer_type="adam")
        assert cfg.optimizer.learning_rate == 3e-4
        assert cfg.optimizer.type == "adam"

    def test_the_flat_spelling_is_unreachable_from_python(self) -> None:
        """The fold is NOT a forwarding property. If this ever passes as an
        attribute read, there are two read paths again and the phase bought
        nothing."""
        cfg = OptimizationConfigSchema(learning_rate=3e-4)
        assert not hasattr(cfg, "learning_rate")
        assert "learning_rate" not in OptimizationConfigSchema.model_fields

    def test_the_nested_spelling_is_what_a_new_arm_writes(self) -> None:
        cfg = OptimizationConfigSchema(
            optimizer={"type": "sgd", "learning_rate": 0.1, "momentum": 0.8},
            gradient={"accumulation_steps": 4, "clip": {"enabled": True, "value": 2.0}},
            precision={"enabled": True, "dtype": "bfloat16"},
        )
        assert cfg.optimizer.type == "sgd"
        assert cfg.gradient.clip.value == 2.0
        assert cfg.precision.dtype == "bfloat16"

    def test_mixing_the_two_spellings_disagreeably_raises(self) -> None:
        with pytest.raises(ValidationError, match="disagree"):
            OptimizationConfigSchema(
                learning_rate=1e-4, optimizer={"learning_rate": 9e-9}
            )

    def test_the_surviving_checkpointing_spelling_still_folds(self) -> None:
        """`use_gradient_checkpointing` folds — 35 arms still declare it.

        Its sibling `gradient_checkpointing` no longer does; see below. The two
        spellings of one knob drain **independently**, which is the whole point
        of a per-record posture: promoting the pair together would break those
        35 arms at load.
        """
        assert OptimizationConfigSchema(
            use_gradient_checkpointing=True
        ).gradient.enable_checkpointing

    def test_the_drained_checkpointing_spelling_now_raises(self) -> None:
        """`gradient_checkpointing` is promoted to `raise` (Wave 3.3, #919).

        Corpus declarations: 0, per `check_no_legacy_config_keys.py`. The
        remaining Python sites named by #919's AST walk do **not** declare this
        key — `performance_optimizer.py` uses its own
        `{"gradient_checkpointing": {"enabled": ...}}` dict shape,
        `test_pipeline_schema.py` reads a `PipelineStage` field, and the HPO
        script holds a dict of variant names. A leaf-name walk cannot tell those
        apart from a declaration, which is why the record looked pinned.
        """
        with pytest.raises(ValidationError, match="gradient_checkpointing"):
            OptimizationConfigSchema(gradient_checkpointing=True)

    def test_the_canonical_spelling_is_unaffected(self) -> None:
        """Both spellings pointed here; the canonical path must still work."""
        assert OptimizationConfigSchema(
            gradient={"enable_checkpointing": True}
        ).gradient.enable_checkpointing

    def test_a_sub_block_rejects_a_typo(self) -> None:
        """New blocks are born extra='forbid': a typo inside `precision:` has
        never worked, so there is no legacy corpus to protect."""
        with pytest.raises(ValidationError):
            OptimizationConfigSchema(precision={"enabld": True})

    def test_amp_float32_is_still_the_third_state(self) -> None:
        """dtype float32 disables AMP even with enabled=true. Preserved, not
        redesigned -- resolve_amp_precision owns that decision.

        Declared through the canonical `precision` block: `amp_dtype` was
        promoted to `raise` (#919), while its partner `use_amp` still folds --
        the two halves of one AMP decision drain independently.
        """
        cfg = OptimizationConfigSchema(use_amp=True, precision={"dtype": "float32"})
        assert cfg.precision.enabled is True
        assert cfg.precision.dtype == "float32"

    def test_the_drained_amp_dtype_spelling_now_raises(self) -> None:
        """`amp_dtype` is promoted to `raise` (Wave 3.3, #919).

        Corpus declarations: 0. Of the 28 Python sites #919's AST walk counted,
        only 3 were declarations into this schema (this one and two in
        `test_optimization_schema.py`); the other 25 are `amp_dtype` PARAMETERS
        of local test helpers (`_config`, `_settings`, `block_stub`,
        `_diffusion_config`) and of `resolve_amp_precision`, none of which
        reaches the schema under that name.
        """
        with pytest.raises(ValidationError, match="amp_dtype"):
            OptimizationConfigSchema(amp_dtype="float32")

    def test_its_partner_use_amp_still_folds(self) -> None:
        """AMP is one decision with two parts, and only one part is drained."""
        assert OptimizationConfigSchema(use_amp=True).precision.enabled is True


class TestSchedulerStaysFlat:
    """Phase 8 did NOT create a `scheduler:` sub-block — #662.

    These tests exist so the omission is a recorded decision rather than an
    oversight someone "fixes" later without reading the issue. The *reason*
    changed on 2026-08-08: the fold was blocked because declaring a strategy
    and declaring a dict meant different things, and creating the dict would
    have started annealing 531 constant-LR arms. They now mean the same thing,
    so the fold is safe — it is simply not done yet.
    """

    def test_the_strategy_is_still_a_top_level_key(self) -> None:
        cfg = OptimizationConfigSchema(lr_scheduler_strategy="step")
        assert cfg.lr_scheduler_strategy == "step"

    def test_declaring_a_strategy_does_not_conjure_a_scheduler_dict(self) -> None:
        """The schema must not synthesise the dict; the RESOLVER handles absence.

        Keeping these separate is what lets ``model_fields_set`` distinguish a
        declared family from the ``"cosine"`` default — the distinction the
        #662 fix is built on.
        """
        cfg = OptimizationConfigSchema(lr_scheduler_strategy="cosine")
        assert cfg.scheduler is None
        assert "lr_scheduler_strategy" in cfg.model_fields_set

    def test_the_default_is_distinguishable_from_a_declaration(self) -> None:
        """Undeclared still presents ``"cosine"``; only ``model_fields_set`` differs.

        If this ever stops holding, ``resolve_scheduler_spec`` loses its only
        way to tell "wants cosine" from "said nothing", and 305 arms that asked
        for no schedule start annealing.
        """
        cfg = OptimizationConfigSchema()
        assert cfg.lr_scheduler_strategy == "cosine"
        assert "lr_scheduler_strategy" not in cfg.model_fields_set


class TestUnwiredKnobStaysVisible:
    def test_num_steps_has_no_sub_block(self) -> None:
        """It has no consumer, so it has no group to belong to. Giving it a home
        under `optimizer:` would imply it works (pitfall #15/#16)."""
        assert "num_steps" in OptimizationConfigSchema.model_fields
        assert OptimizationConfigSchema().num_steps is None


class TestRegionalAndComplexOptOut:
    """The two knobs regional compilation adds, and why each validator exists."""

    def test_defaults_preserve_every_existing_arm(self):
        from spectramr.config.schemas.optimization import CompileConfigSchema

        cfg = CompileConfigSchema()
        assert cfg.regional is False
        assert cfg.allow_complex is False

    def test_regional_is_accepted_with_compile_on(self):
        from spectramr.config.schemas.optimization import CompileConfigSchema

        assert CompileConfigSchema(enabled=True, regional=True).regional is True

    @pytest.mark.parametrize("knob", ["regional", "allow_complex"])
    def test_a_knob_under_disabled_compile_raises(self, knob):
        """Planted: a knob declared against a disabled mechanism never runs, and
        is indistinguishable from the outside from one that works (pitfall 15)."""
        from spectramr.config.schemas.optimization import CompileConfigSchema

        with pytest.raises(ValidationError, match="never be read"):
            CompileConfigSchema(enabled=False, **{knob: True})

    def test_allow_complex_without_regional_raises(self):
        """Planted: the opt-out rests on the physics ops being fenced out of the
        graph. Whole-model compilation puts them back, where Inductor silently
        falls back to eager -- the false-throughput claim the check prevents."""
        from spectramr.config.schemas.optimization import CompileConfigSchema

        with pytest.raises(ValidationError, match="requires regional"):
            CompileConfigSchema(enabled=True, allow_complex=True)

    def test_allow_complex_with_fullgraph_raises(self):
        """Planted: the fences are torch._dynamo.disable, which forces a graph
        break, and a graph break under fullgraph raises. Rejecting the pair at
        load time costs 100ms instead of failing at the first forward pass."""
        from spectramr.config.schemas.optimization import CompileConfigSchema

        with pytest.raises(ValidationError, match="cannot be combined with"):
            CompileConfigSchema(
                enabled=True, regional=True, allow_complex=True, fullgraph=True
            )

    def test_the_legal_combination_is_accepted(self):
        from spectramr.config.schemas.optimization import CompileConfigSchema

        cfg = CompileConfigSchema(enabled=True, regional=True, allow_complex=True)
        assert cfg.allow_complex and cfg.regional and not cfg.fullgraph


# --------------------------------------------------------------------------- #
# compile.apply_to — which models compilation touches
#
# Before this existed, `regional` was unusable by every GAN arm: it refuses an
# arm whose models do not all expose a uniform block list, and a patch critic is
# a plain nn.Sequential. The knob is what makes "compile the generator only" a
# thing the config can say.
# --------------------------------------------------------------------------- #
def _compile(**kwargs) -> CompileConfigSchema:
    return CompileConfigSchema(**kwargs)


def test_apply_to_defaults_to_every_model() -> None:
    """None, not a list of all four -- the default must keep meaning "all" if a
    fifth model is ever built."""
    assert _compile().apply_to is None


def test_apply_to_accepts_the_built_model_names() -> None:
    assert _compile(enabled=True, apply_to=["generator", "discriminator"]).apply_to == [
        "generator",
        "discriminator",
    ]


def test_an_unknown_model_name_is_rejected_at_load() -> None:
    """Planted: a closed vocabulary, so a misspelling raises here rather than
    selecting nothing at build time (non-negotiable 3)."""
    with pytest.raises(ValidationError):
        _compile(enabled=True, apply_to=["critic"])


def test_an_empty_selection_is_rejected() -> None:
    """Planted: `apply_to: []` is `enabled: false` spelled so nobody notices."""
    with pytest.raises(ValidationError, match="empty list"):
        _compile(enabled=True, apply_to=[])


# --------------------------------------------------------------------------- #
# compile.recompile_limit / fail_on_recompile_limit
#
# torch drops a frame to eager once it exhausts its recompilation budget, and
# `fail_on_recompile_limit_hit` is OFF by default -- so a run can stop being
# compiled partway through and still report a compiled configuration.
# --------------------------------------------------------------------------- #
def test_failing_on_the_recompile_limit_is_the_default() -> None:
    """Torch defaults this off. This block's whole policy is that a run must not
    report a configuration it stopped executing, so the default is inverted."""
    assert _compile().fail_on_recompile_limit is True


def test_recompile_limit_defaults_to_torchs_own() -> None:
    """None, not 8: hard-coding torch's default here would silently pin it
    across a version bump that changed it."""
    assert _compile().recompile_limit is None


def test_a_declared_budget_must_be_enforced() -> None:
    """Planted: declaring how much recompilation you tolerate and then declining
    to act on it means exceeding it drops to eager in silence."""
    with pytest.raises(ValidationError, match="fail_on_recompile_limit"):
        _compile(enabled=True, recompile_limit=16, fail_on_recompile_limit=False)


@pytest.mark.parametrize("value", [0, -1])
def test_a_non_positive_budget_is_rejected(value: int) -> None:
    """Zero would disable compilation by a side door."""
    with pytest.raises(ValidationError, match="recompile_limit"):
        _compile(enabled=True, recompile_limit=value)


# --------------------------------------------------------------------------- #
# Inert knobs under `enabled: false`
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"apply_to": ["generator"]},
        {"recompile_limit": 16},
        {"regional": True},
        {"fail_on_recompile_limit": False},
    ],
    ids=["apply_to", "recompile_limit", "regional", "fail_on_recompile_limit"],
)
def test_a_knob_under_disabled_compilation_is_rejected(kwargs) -> None:
    """An advertised knob nothing reads is indistinguishable from one that works
    (pitfall 15)."""
    with pytest.raises(ValidationError, match="enabled is false"):
        _compile(enabled=False, **kwargs)


def test_plain_eager_is_still_valid() -> None:
    """Planted against the obvious way to get the above wrong: the inert-knob
    check compares against each field's DEFAULT, not truthiness. A truthiness
    test would call `fail_on_recompile_limit=True` "set" and reject every eager
    arm in the corpus.
    """
    assert CompileConfigSchema().enabled is False
    assert CompileConfigSchema(enabled=False).fail_on_recompile_limit is True
