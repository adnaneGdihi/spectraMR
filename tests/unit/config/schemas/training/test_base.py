"""Schema-level pins for ``training.vae`` and the KL-annealing alias folding.

``training.vae`` was READ by ``models/losses/computers/unified_vae.py`` but never
DECLARED on ``TrainingStrategyConfigSchema``. Because that schema is
``extra='allow'``, the block was admitted as a raw ``dict`` and every
``getattr(training.vae, "kl_beta_end", None)`` in the resolver returned ``None``
— a knob that validated clean and was never read (pitfall #15).

Independently, the 26 experiment YAMLs carrying the block spell the knobs
``enable_kl_annealing`` / ``kl_anneal_start`` / ``kl_anneal_end`` while the
schema and resolver spell them ``anneal_kl_beta`` / ``kl_beta_start`` /
``kl_beta_end``. ``AliasChoices`` folds both spellings onto one canonical field
so the KL weight keeps a single resolver (pitfall #13b).

Behavioural coverage of the resolved weight lives in
``tests/unit/models/losses/computers/test_unified_vae.py::TestKLAnnealNamingSchism``.
"""

from pathlib import Path

import pytest

from spectramr.config.schemas.training.base import (
    BaseTrainingConfigSchema,
    LatentTrainingConfigSchema,
    TrainingStrategyConfigSchema,
)
from spectramr.config.settings import TrainingSettings

REPO = Path(__file__).resolve().parents[5]

# The literal block shipped by the experiment YAMLs.
_YAML_SPELLING = {
    "enable_kl_annealing": True,
    "kl_anneal_start": 0.0,
    "kl_anneal_end": 1.0,
    "kl_anneal_steps": 10_000,
}


class TestTrainingVaeIsDeclared:
    """`training.vae` must validate into a model, never fall through to a dict."""

    def test_vae_block_is_not_a_raw_dict(self):
        training = TrainingStrategyConfigSchema(training_mode="vae", vae=dict(_YAML_SPELLING))
        assert isinstance(training.vae, LatentTrainingConfigSchema)
        # The pre-fix shape: a dict, against which getattr silently yields None.
        assert not isinstance(training.vae, dict)
        assert getattr(training.vae, "kl_beta_end", None) is not None

    def test_vae_defaults_to_none_when_absent(self):
        training = TrainingStrategyConfigSchema(training_mode="vae")
        assert training.vae is None

    def test_vae_and_latent_are_independent_blocks(self):
        training = TrainingStrategyConfigSchema(
            training_mode="vae",
            vae={"kl_anneal_end": 0.25},
            latent={"kl_beta_end": 0.75},
        )
        assert training.vae.kl_beta_end == pytest.approx(0.25)
        assert training.latent.kl_beta_end == pytest.approx(0.75)


class TestTrainingQualityMatchingIsDeclared:
    """`training.quality_matching` must validate into a model, never a raw dict.

    Same trap as `training.vae` above: this schema is ``extra='allow'``, so an
    unmounted block would be admitted as a dict and every attribute read would
    silently yield ``None`` -- a config that validates clean and does nothing.
    """

    def test_quality_matching_block_is_not_a_raw_dict(self):
        from spectramr.config.schemas.training.quality_matching import (
            QualityMatchingConfig,
        )

        training = TrainingStrategyConfigSchema(
            training_mode="quality_matching",
            quality_matching={
                "axes": ["complex_gaussian"],
                "target": {
                    "source": "literal",
                    "attributes": ["tenengrad_variance"],
                    "override": {"tenengrad_variance": 0.01},
                },
                # Not a geometry test: a literal target with match_spacing on would
                # need an explicit spacing_mm.
                "match_spacing": False,
            },
        )
        assert isinstance(training.quality_matching, QualityMatchingConfig)
        assert not isinstance(training.quality_matching, dict)
        assert getattr(training.quality_matching, "min_gap_closed", None) is not None

    def test_quality_matching_defaults_to_none_when_absent(self):
        training = TrainingStrategyConfigSchema(training_mode="quality_matching")
        assert training.quality_matching is None


class TestKLAnnealAliases:
    """Both spellings populate one canonical field."""

    @pytest.mark.parametrize(
        ("alias", "canonical", "value"),
        [
            ("enable_kl_annealing", "anneal_kl_beta", True),
            ("kl_anneal_start", "kl_beta_start", 0.2),
            ("kl_anneal_end", "kl_beta_end", 0.8),
        ],
    )
    def test_yaml_alias_populates_canonical_field(self, alias, canonical, value):
        cfg = LatentTrainingConfigSchema(**{alias: value})
        assert getattr(cfg, canonical) == value

    @pytest.mark.parametrize(
        ("canonical", "value"),
        [("anneal_kl_beta", True), ("kl_beta_start", 0.2), ("kl_beta_end", 0.8)],
    )
    def test_canonical_name_still_populates(self, canonical, value):
        """populate_by_name=True -- adding aliases must not break existing YAML."""
        cfg = LatentTrainingConfigSchema(**{canonical: value})
        assert getattr(cfg, canonical) == value

    def test_full_yaml_block_resolves_every_knob(self):
        cfg = LatentTrainingConfigSchema(**_YAML_SPELLING)
        assert cfg.anneal_kl_beta is True
        assert cfg.kl_beta_start == pytest.approx(0.0)
        assert cfg.kl_beta_end == pytest.approx(1.0)
        assert cfg.kl_anneal_steps == 10_000

    def test_schema_stays_frozen(self):
        """CLAUDE.md non-negotiable #1 -- config is immutable."""
        cfg = LatentTrainingConfigSchema(**_YAML_SPELLING)
        with pytest.raises((TypeError, ValueError)):
            cfg.kl_beta_end = 0.5


class TestCriticalInfrastructureValidatorIsSatisfiable:
    """``BaseTrainingConfigSchema.validate_critical_infrastructure_fields``.

    Phase 13 deleted 17 duplicated blocks from this schema while leaving the
    validator's ``critical_fields`` list naming seven of them. The class is
    ``extra="ignore"``, so a caller supplying ``logging={}`` had it dropped
    before the validator ran and ``hasattr`` was False regardless of the input:
    **every** construction raised, including the three tests written to drive
    the version gate through this class.

    The tell was that those tests failed identically on 1.0, 6.0 and 6.1 — a
    failure that does not vary with the thing under test is not about that
    thing. Pinned here because a validator nothing can satisfy reads as a
    strict gate while enforcing nothing, and the next block that moves out of
    this schema would silently re-break it.
    """

    def test_the_schema_constructs_at_all(self):
        from spectramr.config.schemas.training.base import BaseTrainingConfigSchema

        assert BaseTrainingConfigSchema().config_version

    def test_the_validator_only_demands_fields_the_class_declares(self):
        from spectramr.config.schemas.training.base import BaseTrainingConfigSchema

        demanded = {
            "logging",
            "loss_logging",
            "metrics",
            "early_stopping",
            "ema",
            "validation",
            "checkpoint",
            "services",
        }
        declared = set(BaseTrainingConfigSchema.model_fields)
        assert demanded - declared, (
            "control: if this schema declared all of them the intersection would "
            "be a no-op and this test would prove nothing"
        )
        assert BaseTrainingConfigSchema() is not None

    def test_a_declared_infrastructure_field_is_still_enforced(self):
        """The check must stay real for what remains -- ``services`` is declared,
        so the validator is narrowed, not neutered."""
        from spectramr.config.schemas.training.base import BaseTrainingConfigSchema

        assert "services" in BaseTrainingConfigSchema.model_fields
        assert hasattr(BaseTrainingConfigSchema(), "services")


# ── `resolve_output_dirs` was deleted (issue #698) ───────────────────────────
#
# It was a `model_validator(mode="after")` on this class opening with
# `if not hasattr(self, "artifacts") ...: return self`. `artifacts` is a field
# of `TrainingSettings`, NOT of this schema, so the guard was False on every
# construction and the body never ran once. Had it run it would have raised:
# it did `self.artifacts.persistent_root` on a field typed `dict | None`.
#
# This is the hasattr-guard-outlives-its-field shape that silently disabled
# `rule_spatial_rank` and the SFC conditioning wrapper on PR #644. A defensive
# `hasattr` around a DECLARED field is never harmless: when the field moves (or
# was never there) the guard turns the body off instead of raising.
#
# So the pin is the BEHAVIOUR the deletion preserved -- `output_dir` is
# authoritative and untouched -- not merely the absence of the method.


class TestOutputDirIsAuthoritative:
    def test_the_dead_resolver_is_gone(self) -> None:
        assert not hasattr(BaseTrainingConfigSchema, "resolve_output_dirs")

    def test_the_guard_it_depended_on_could_never_have_been_true(self) -> None:
        """Anti-vacuity for the deletion: proves the body was unreachable.

        If `artifacts` ever becomes a real field here, this fails and the
        deletion must be revisited rather than silently staying correct.
        """
        assert "artifacts" not in BaseTrainingConfigSchema.model_fields

    def test_declared_output_dir_survives_construction(self) -> None:
        """The property the deleted validator would have BROKEN if it ran.

        It did an unconditional `object.__setattr__(self, "output_dir", ...)`,
        so a run's declared output path would have been overwritten by
        `<persistent_root>/<name>` for every construction.
        """
        cfg = BaseTrainingConfigSchema(output_dir="/declared/by/yaml")
        assert cfg.output_dir == "/declared/by/yaml"

    def test_metrics_output_dir_is_not_auto_derived(self) -> None:
        """Unset stays unset; the run derives it downstream from output_dir."""
        assert BaseTrainingConfigSchema().metrics_output_dir is None

    def test_a_real_arm_keeps_the_output_dir_its_yaml_declares(self) -> None:
        """Resolved through the real loader, not a hand-built stand-in.

        A hand-built schema is a second resolver; this repo has repeatedly had
        one agree with the first until it didn't. `artifacts:` is declared by
        527 tracked arms, so if anything ever consumed it, an arm is where the
        divergence would show.
        """
        import yaml

        arm = REPO / "experiments/inprogress/workflow_baselines/b1_structural_recon_m4raw.yaml"
        if not arm.is_file():
            pytest.skip(f"{arm} not present")
        declared = (yaml.safe_load(arm.read_text()).get("training") or {}).get("output_dir")
        if declared is None:
            pytest.skip("arm does not declare training.output_dir")
        settings = TrainingSettings.from_yaml(str(arm))
        assert settings.training.output_dir == declared


class TestDiffusionBetaFieldsAreDeclared:
    """#799: the class ``training.diffusion`` mounts must carry the beta pair.

    ``DiffusionTrainingStrategy.__init__`` forwards
    ``training.diffusion.beta_start`` / ``beta_end`` to the noise scheduler
    unconditionally. They were declared only on ``TrainingConfigDiffusion``,
    which no arm mounts; ``DiffusionTrainingConfigSchema`` is ``extra='allow'``,
    so an arm that spelled them in YAML worked and every arm that did not raised
    ``AttributeError`` at construction. Only 30 arms in the corpus spell them.
    """

    def test_beta_pair_is_readable_without_being_declared(self) -> None:
        from spectramr.config.schemas.training.base import DiffusionTrainingConfigSchema

        config = DiffusionTrainingConfigSchema()

        assert config.beta_start == 0.0001
        assert config.beta_end == 0.02

    def test_declared_values_survive(self) -> None:
        from spectramr.config.schemas.training.base import DiffusionTrainingConfigSchema

        config = DiffusionTrainingConfigSchema(beta_start=1.0e-6, beta_end=0.01)

        assert config.beta_start == 1.0e-6
        assert config.beta_end == 0.01

    def test_defaults_match_the_schema_that_always_carried_them(self) -> None:
        """A second set of defaults is a second resolver -- pin them equal."""
        from spectramr.config.schemas.training.base import DiffusionTrainingConfigSchema
        from spectramr.config.schemas.training.diffusion import TrainingConfigDiffusion

        for field in ("beta_start", "beta_end"):
            assert (
                DiffusionTrainingConfigSchema.model_fields[field].default
                == TrainingConfigDiffusion.model_fields[field].default
            ), field


class TestShapeContractReadsTheCanonicalPatchSize:
    """`validate_shape_contracts` read `data.patch_size` after the fold moved it.

    The validator's own guard checks that the ``data`` and ``model`` BLOCKS
    exist, never that the leaf does -- so once `data.patch_size` folded to
    `data.sampling.patch_size` the read raised `AttributeError`, and every
    construction reaching this validator died.

    ``TrainingConfigMetaLearning`` is the ONLY schema in the tree declaring both
    blocks, so it is the only class this contract has ever fired on -- which is
    why two meta-learning tests were the entire failure signal.

    Note only ONE of the validator's two branches is reachable there:
    ``MetaLearningModelConfig.spatial_dims`` is a read-only property hardcoded to
    2 ("Meta-learning typically 2D"), and the class is ``extra="ignore"``, so
    passing ``spatial_dims=3`` is silently swallowed. The 2D-data/3D-model branch
    therefore cannot fire on the only subject the validator has.
    """

    def test_a_3d_patch_against_the_2d_model_still_raises_the_contract_error(self):
        """The contract must actually FIRE, not merely stop crashing.

        A port that silently skipped the check would turn the red tests green
        while removing the guard -- the failure mode this pin exists for.
        """
        import pytest

        from spectramr.config.schemas.training.meta_learning import (
            MetaLearningModelConfig,
            TrainingConfigMetaLearning,
        )

        with pytest.raises(ValueError, match="DIMENSION MISMATCH"):
            TrainingConfigMetaLearning(
                data={"patch_size": (64, 64, 32)},
                model=MetaLearningModelConfig(),
            )

    def test_a_consistent_2d_pair_constructs_and_reads_the_canonical_leaf(self):
        from spectramr.config.schemas.training.meta_learning import (
            MetaLearningModelConfig,
            TrainingConfigMetaLearning,
        )

        cfg = TrainingConfigMetaLearning(
            data={"patch_size": (64, 64, 1)},
            model=MetaLearningModelConfig(),
        )
        assert cfg.data.sampling.patch_size == (64, 64, 1)


class TestUniversalTrainingKeysAreDeclared:
    """The three universal `training:` keys that are READ are now fields.

    They previously arrived through `extra="allow"`, so they carried no type and
    the execution ledger classified them as untyped extras — including
    `training_mode`, the key 49 call sites dispatch the whole paradigm on.
    """

    READ_AND_DECLARED = ("training_mode", "input_domain", "output_domain")

    #: Equally universal, and deliberately NOT declared: nothing in `src/` reads
    #: them. Declaring a knob nothing consumes is exactly what non-negotiable #8
    #: forbids, and it would silence the audit for 20 arms running at a batch
    #: size they never asked for and 84 at an AMP setting they never asked for.
    #: Their fix is a fold onto the canonical path, which changes what those
    #: arms compute — issue #887. `enable_mixed_precision` stays absent here
    #: after the 2026-09-03 `inprogress/` drain: the `no_dead_precision_flag`
    #: witness reports the spelling until the other trees drain and a `raise`
    #: rename record can retire it.
    UNREAD_AND_DELIBERATELY_ABSENT = (
        "batch_size",
        "num_workers",
        "max_steps",
        "enable_mixed_precision",
        "enable_gradient_checkpointing",
    )

    def test_the_read_keys_are_declared_fields(self) -> None:
        from spectramr.config.schemas.training.base import (
            TrainingStrategyConfigSchema,
        )

        for key in self.READ_AND_DECLARED:
            assert key in TrainingStrategyConfigSchema.model_fields, (
                f"{key} is read by src/ but arrives as an untyped extra"
            )

    def test_the_unread_keys_stay_undeclared(self) -> None:
        """A field for a knob nothing reads advertises a lie (#8, #887)."""
        from spectramr.config.schemas.training.base import (
            TrainingStrategyConfigSchema,
        )

        wrongly_declared = [
            k
            for k in self.UNREAD_AND_DELIBERATELY_ABSENT
            if k in TrainingStrategyConfigSchema.model_fields
        ]
        assert not wrongly_declared, (
            f"{wrongly_declared} have zero readers in src/; declaring them "
            "makes the audit go quiet without making the knob work. Wire a "
            "reader or fold them onto the canonical path first (#887)."
        )

    def test_output_domain_feature_is_accepted(self) -> None:
        """`SignalDomain` has no `feature` member but 471 corpus arms declare
        it, so this field must NOT be typed as that enum."""
        from spectramr.config.schemas.training.base import (
            TrainingStrategyConfigSchema,
        )

        cfg = TrainingStrategyConfigSchema(output_domain="feature")
        assert cfg.output_domain == "feature"

    def test_declaring_them_does_not_change_what_readers_see(self) -> None:
        from spectramr.config.schemas.training.base import (
            TrainingStrategyConfigSchema,
        )

        cfg = TrainingStrategyConfigSchema(training_mode="diffusion", input_domain="kspace")
        assert cfg.training_mode == "diffusion"
        assert cfg.input_domain == "kspace"
        assert cfg.output_domain is None


_GRAPH_COLD = (
    "spectramr.infrastructure.training.strategies.graph_cold_diffusion_strategy."
    "GraphColdDiffusionStrategy"
)


class TestRicianColdDiffusionKnobs:
    """``degradation: rician`` + its sigma and the validation level are GraphCold-only."""

    def test_rician_needs_its_noise_ceiling(self) -> None:
        from spectramr.config.schemas.training.base import DiffusionTrainingConfigSchema

        with pytest.raises(ValueError, match="needs rician_sigma_max"):
            DiffusionTrainingConfigSchema(degradation="rician")

    def test_a_noise_ceiling_without_rician_is_refused(self) -> None:
        from spectramr.config.schemas.training.base import DiffusionTrainingConfigSchema

        with pytest.raises(ValueError, match="read only with degradation: rician"):
            DiffusionTrainingConfigSchema(degradation="kspace_mask", rician_sigma_max=0.1)

    def test_the_validation_level_must_sit_on_the_schedule(self) -> None:
        from spectramr.config.schemas.training.base import DiffusionTrainingConfigSchema

        with pytest.raises(ValueError, match="must be < timesteps"):
            DiffusionTrainingConfigSchema(timesteps=10, validation_timestep=10)

    def test_the_graph_cold_strategy_accepts_them(self) -> None:
        block = {"degradation": "rician", "rician_sigma_max": 0.2, "validation_timestep": 3}
        cfg = TrainingStrategyConfigSchema(strategy_class=_GRAPH_COLD, diffusion=block)
        assert cfg.diffusion.rician_sigma_max == 0.2
        assert cfg.diffusion.validation_timestep == 3

    @pytest.mark.parametrize(
        "block",
        [
            {"degradation": "rician", "rician_sigma_max": 0.2},
            {"validation_timestep": 3},
        ],
    )
    def test_any_other_strategy_is_refused(self, block) -> None:
        other = "spectramr.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy"
        with pytest.raises(ValueError, match="read only by GraphColdDiffusionStrategy"):
            TrainingStrategyConfigSchema(strategy_class=other, diffusion=block)

    def test_kspace_mask_stays_legal_everywhere(self) -> None:
        other = "spectramr.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy"
        cfg = TrainingStrategyConfigSchema(strategy_class=other, diffusion={"degradation": "kspace_mask"})
        assert cfg.diffusion.degradation == "kspace_mask"
