"""Tests for ``config/schemas/base.py`` — the schema-version SSOT.

These assert the **seam**, not the constant. A test that restates the accepted
set as a literal passes whether or not any gate actually consults it, which is
the condition the constant exists to rule out. The set was three independent
literals (``config/settings.py`` twice, ``config/schemas/training/base.py``
once) and nothing tied them together, so each could drift while agreeing with
itself.

The three tiers introduced on 2026-08-03 have since collapsed to two.
``LEGACY_CONFIG_VERSIONS`` was the middle tier — versions accepted only because
the loader *folded* them — and PR #891 deleted the fold and emptied the set. So
``ACCEPTED_CONFIG_VERSIONS`` is now exactly ``{CANONICAL_CONFIG_VERSION}``:
"accepted" and "current" mean the same thing again, and a declared ``6.0``/
``6.1`` is **refused**, not rewritten.

``TestVersionRetirement`` below pins that emptiness directly, because an empty
tier is the one thing a parametrised test cannot pin — iterating the set would
silently run zero cases and pass. (It replaced ``TestVersionFold``, which pinned
the fold that no longer exists.)

So every test below drives a **real gate** with a real value and asserts the
behaviour tracks the constant. The accept-side tests are parametrised over
``ACCEPTED_CONFIG_VERSIONS`` itself, so adding a version to the SSOT and
forgetting to make a gate honour it fails here rather than at a user's config.
"""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.base import (
    ACCEPTED_CONFIG_VERSIONS,
    CANONICAL_CONFIG_VERSION,
    LEGACY_CONFIG_VERSIONS,
)
from spectramr.config.schemas.training.base import BaseTrainingConfigSchema
from spectramr.config.settings import TrainingSettings

#: A version no future release will take, and the one 39 *committed* YAMLs carry —
#: all under ``experiments/ablation/`` and ``experiments/hpo/``, none under
#: ``inprogress/`` (``TODO/backlog_config_version_5_0_corpus.md``). Every gate must
#: refuse it. 613 files on disk declare it; the other 574 are gitignored scratch,
#: which is why the count in this repo's prose drifted by ~15x.
_UNSUPPORTED = "5.0"


def _infrastructure_sections() -> dict:
    """The blocks ``BaseTrainingConfigSchema``'s model validator demands.

    Added 2026-08-03. The three gate-1 tests below constructed the schema with
    ``config_version`` alone; that stopped working when the CRITICAL
    INFRASTRUCTURE FIELDS validator landed, and they had been red for every
    version — 1.0, 6.0 and 6.1 alike — so the failure was not about versions at
    all. Supplying the blocks restores what the tests were built to assert (the
    version gate) instead of re-asserting the infrastructure gate.
    """
    return {
        "logging": {},
        "loss_logging": {},
        "metrics": {},
        "early_stopping": {},
        "ema": {},
        "validation": {},
        "checkpoint": {},
        "services": {},
    }


def _base_sections() -> dict:
    """The always-required sections, mirroring ``tests/unit/config/test_settings.py``."""
    return {
        "data": {"train_path": "/tmp/train", "val_path": "/tmp/val", "batch_size": 2},
        "optimization": {"learning_rate": 1e-4},
        "logging": {},
        "model": {"model_type": "unet"},
    }


def test_accepted_versions_is_a_nonempty_frozenset() -> None:
    """An empty set would silently disable every gate; a mutable one lets a caller edit it."""
    assert isinstance(ACCEPTED_CONFIG_VERSIONS, frozenset)
    assert ACCEPTED_CONFIG_VERSIONS
    assert all(isinstance(version, str) for version in ACCEPTED_CONFIG_VERSIONS)


def test_unsupported_is_actually_excluded() -> None:
    """Guards this file: every rejection test below is vacuous if 5.0 is in the set."""
    assert _UNSUPPORTED not in ACCEPTED_CONFIG_VERSIONS


# --- gate 1: the schema field validator (config/schemas/training/base.py) ---
#
# Driven through construction rather than by calling ``validate_config_version``
# directly. Pydantic does not guarantee the decorated classmethod stays reachable
# via ``getattr`` — on ``TrainingStrategyConfigSchema`` it is not — so calling it
# by name would test an implementation detail. Construction is the path configs
# actually take.


@pytest.mark.parametrize("version", sorted(ACCEPTED_CONFIG_VERSIONS))
def test_schema_gate_accepts_every_version_in_the_ssot(version: str) -> None:
    schema = BaseTrainingConfigSchema(
        config_version=version, **_infrastructure_sections()
    )
    assert schema.config_version == version


def test_schema_gate_rejects_a_version_outside_the_ssot() -> None:
    """No infrastructure blocks needed: the field validator runs first, so this
    still fails on the version rather than on a missing block."""
    with pytest.raises(ValidationError, match="Unsupported config_version"):
        BaseTrainingConfigSchema(config_version=_UNSUPPORTED)


# --- gate 2: TrainingSettings.settings_from_dict (config/settings.py) ---


@pytest.mark.parametrize("version", sorted(ACCEPTED_CONFIG_VERSIONS))
def test_settings_from_dict_accepts_every_version_in_the_ssot(version: str) -> None:
    """``config_version`` is optional here and stripped before construction."""
    config = _base_sections() | {"config_version": version}
    assert TrainingSettings.settings_from_dict(config).model.model_type == "unet"


def test_settings_from_dict_rejects_a_version_outside_the_ssot() -> None:
    config = _base_sections() | {"config_version": _UNSUPPORTED}
    with pytest.raises(ValueError, match="not supported"):
        TrainingSettings.settings_from_dict(config)


# --- gate 3: TrainingSettings.from_yaml (config/settings.py) ---
#
# The gate every committed arm hits.


def test_from_yaml_rejects_a_version_outside_the_ssot(tmp_path) -> None:
    """This gate raises *before* any field validator runs.

    That ordering is why a v5 arm's ``dataset_type`` is never reached, and why
    issue #355 ("42 arms carry a bad dataset_type") was a misdiagnosis: the
    dataset_type is not what stops them, the version is.
    """
    config = tmp_path / "arm.yaml"
    config.write_text(
        f"config_version: '{_UNSUPPORTED}'\ndata:\n  dataset_type: cardiac_cine\n"
    )

    with pytest.raises(ValueError, match="not supported"):
        TrainingSettings.from_yaml(config)


def test_from_yaml_names_the_ssot_versions_when_config_version_is_missing(
    tmp_path,
) -> None:
    """The 'required' message must be generated from the SSOT, not a hardcoded string.

    It read ``Must be set to '6.0' or '6.1'.`` — a fourth copy of the set, in prose,
    which would still have said "'6.0' or '6.1'" after a v6.2 landed.
    """
    config = tmp_path / "arm.yaml"
    config.write_text("model:\n  model_type: unet\n")

    with pytest.raises(ValueError) as excinfo:
        TrainingSettings.from_yaml(config)

    message = str(excinfo.value)
    assert "config_version is required" in message
    for version in ACCEPTED_CONFIG_VERSIONS:
        assert version in message, f"{version!r} missing from the required-message"


# --- the mutation test: does each gate actually CONSULT the constant? ---
#
# Everything above is parametrised over ACCEPTED_CONFIG_VERSIONS, which pins that the
# gates agree with the SSOT's CURRENT contents — but every assertion is equally
# satisfied by a gate that restates {"6.0", "6.1"} inline, because the literal and the
# constant happen to be equal today. That is exactly the condition this file's docstring
# says it rules out, and it did not: reverting either settings.py gate to an inline set
# left the whole module green.
#
# Injecting a version the literals cannot contain separates the two. Each gate is
# patched at ITS OWN module binding, because `from ..base import ACCEPTED_CONFIG_VERSIONS`
# copies the name into the importing module's namespace — patching only the definition
# site in config/schemas/base.py would leave every consumer reading the original.

_INJECTED = "6.2"


def test_schema_gate_reads_the_constant_not_a_private_literal(monkeypatch) -> None:
    import spectramr.config.schemas.training.base as training_base

    monkeypatch.setattr(
        training_base,
        "ACCEPTED_CONFIG_VERSIONS",
        frozenset(ACCEPTED_CONFIG_VERSIONS | {_INJECTED}),
    )
    schema = BaseTrainingConfigSchema(
        config_version=_INJECTED, **_infrastructure_sections()
    )
    assert schema.config_version == _INJECTED


def test_settings_from_dict_reads_the_constant_not_a_private_literal(
    monkeypatch,
) -> None:
    import spectramr.config.settings as settings_module

    monkeypatch.setattr(
        settings_module,
        "ACCEPTED_CONFIG_VERSIONS",
        frozenset(ACCEPTED_CONFIG_VERSIONS | {_INJECTED}),
    )
    settings = TrainingSettings.settings_from_dict(
        {"config_version": _INJECTED, **_base_sections()}
    )
    assert settings is not None


def test_from_yaml_reads_the_constant_not_a_private_literal(
    monkeypatch, tmp_path
) -> None:
    import yaml

    import spectramr.config.settings as settings_module

    monkeypatch.setattr(
        settings_module,
        "ACCEPTED_CONFIG_VERSIONS",
        frozenset(ACCEPTED_CONFIG_VERSIONS | {_INJECTED}),
    )
    path = tmp_path / "injected.yaml"
    path.write_text(yaml.safe_dump({"config_version": _INJECTED, **_base_sections()}))
    assert TrainingSettings.from_yaml(path) is not None


def test_injected_version_is_rejected_without_the_patch() -> None:
    """Guards the three tests above: they are vacuous if 6.2 is accepted anyway."""
    assert _INJECTED not in ACCEPTED_CONFIG_VERSIONS
    with pytest.raises(ValidationError, match="Unsupported config_version"):
        BaseTrainingConfigSchema(config_version=_INJECTED)


def test_the_field_description_tracks_the_ssot() -> None:
    """The user-visible copy is a gate too — it feeds JSON schema and Sphinx autodoc.

    It was the fifth copy of the set and the one a config author actually reads, left
    restating "'6.0', '6.1'" while the three code gates were hoisted.
    """
    description = BaseTrainingConfigSchema.model_fields["config_version"].description
    assert description is not None
    for version in ACCEPTED_CONFIG_VERSIONS:
        assert repr(version) in description


class TestVersionRetirement:
    """The legacy tier is EMPTY and the fold is gone (2026-08-08).

    This class used to pin the fold: that ``6.0``/``6.1`` were accepted and
    rewritten to canonical before binding, and that a legacy file and a bumped
    file resolved to identical documents. That contract no longer exists —
    ``LEGACY_CONFIG_VERSIONS`` is empty, ``_bind_config_version`` binds the
    declared version unchanged, and ``fold_config_version`` is deleted.

    What replaces it is the stronger claim: accepted and current are now the
    same set, and a legacy declaration is REFUSED rather than laundered.
    """

    def test_canonical_is_accepted(self) -> None:
        assert CANONICAL_CONFIG_VERSION in ACCEPTED_CONFIG_VERSIONS

    def test_the_legacy_tier_is_empty(self) -> None:
        assert frozenset() == LEGACY_CONFIG_VERSIONS

    def test_accepted_is_exactly_the_canonical_version(self) -> None:
        """No second way in. While the fold existed this read
        ``{canonical} | legacy``; with the legacy tier drained the union
        collapses, and that collapse IS the retirement."""
        assert frozenset({CANONICAL_CONFIG_VERSION}) == ACCEPTED_CONFIG_VERSIONS

    def test_the_fold_helper_is_gone(self) -> None:
        """A no-op fold left in place is a shim that reads as a contract."""
        import spectramr.config.schemas.base as base_mod

        assert not hasattr(base_mod, "fold_config_version")
        assert "fold_config_version" not in base_mod.__all__

    @pytest.mark.parametrize("version", ["6.0", "6.1"])
    def test_a_retired_version_is_now_refused(self, tmp_path, version: str) -> None:
        """The versions the fold used to accept must not load at all."""
        import yaml

        path = tmp_path / f"arm_{version.replace('.', '_')}.yaml"
        path.write_text(yaml.safe_dump({"config_version": version, **_base_sections()}))
        with pytest.raises(ValueError, match="not supported"):
            TrainingSettings.from_yaml(path)

    def test_a_canonical_file_still_loads_and_binds(self, tmp_path) -> None:
        """Anti-vacuity for the test above: refusal must be about the VERSION,
        not about `_base_sections()` being unloadable."""
        import yaml

        path = tmp_path / "arm.yaml"
        path.write_text(
            yaml.safe_dump(
                {"config_version": CANONICAL_CONFIG_VERSION, **_base_sections()}
            )
        )
        assert (
            TrainingSettings.from_yaml(path).run.config_version
            == CANONICAL_CONFIG_VERSION
        )

    def test_the_declared_version_is_bound_unchanged(self) -> None:
        """Nothing rewrites it now, so declared and resolved must agree."""
        settings = TrainingSettings.settings_from_dict(
            {"config_version": CANONICAL_CONFIG_VERSION, **_base_sections()}
        )
        assert settings.run.config_version == CANONICAL_CONFIG_VERSION

    def test_an_unsupported_version_is_still_refused(self, tmp_path) -> None:
        import yaml

        path = tmp_path / "arm.yaml"
        path.write_text(
            yaml.safe_dump({"config_version": _UNSUPPORTED, **_base_sections()})
        )
        with pytest.raises(ValueError, match="not supported"):
            TrainingSettings.from_yaml(path)


class TestExpectedOutcome:
    """``metadata.expected_outcome`` — the marker that stops triage re-flagging
    deliberate controls.

    ``b110_ablate_pin`` (SSIM 0.0013), ``b22_ablate_likelihood`` and
    ``b27_wiener_refiner_ablate`` (0.2069) are DESIGNED to score near zero, yet each
    diagnostics sweep listed them beside arms at 0.75 with no way to tell a confirmed
    hypothesis from a broken run.
    """

    def test_absent_by_default(self) -> None:
        from spectramr.config.schemas.base import ExperimentMetadataSchema

        assert ExperimentMetadataSchema().expected_outcome is None

    @pytest.mark.parametrize("value", ["comparable", "floor", "ceiling"])
    def test_accepts_the_advertised_set(self, value: str) -> None:
        from spectramr.config.schemas.base import ExperimentMetadataSchema

        assert (
            ExperimentMetadataSchema(expected_outcome=value).expected_outcome == value
        )

    def test_rejects_an_unadvertised_value(self) -> None:
        # Anti-#15: a free-text field would silently mean nothing to the reader.
        from pydantic import ValidationError

        from spectramr.config.schemas.base import ExperimentMetadataSchema

        with pytest.raises(ValidationError):
            ExperimentMetadataSchema(expected_outcome="expected_to_fail")


# ---------------------------------------------------------------------------
# ParallelismConfigSchema — the strategy vocabulary and the double-switch
#
# ``strategy`` was a bare ``str`` that only none/dp/ddp were dispatched for,
# while FSDP was reached through an INDEPENDENT ``fsdp.enabled`` flag read in a
# different builder. The two combined into a trap with opposite failure modes:
# ``strategy: 'fsdp'`` (what the reference template advertised) raised
# ValueError after the entire training environment had been built, while
# ``strategy: 'none'`` + ``fsdp.enabled: true`` silently sharded. Issue #620.
# ---------------------------------------------------------------------------


class TestParallelStrategyVocabulary:
    """The advertised set is now the accepted set, rejected at load time."""

    @pytest.mark.parametrize("value", ["none", "dp", "ddp"])
    def test_accepts_the_strategies_needing_no_subblock(self, value: str) -> None:
        from spectramr.config.schemas.base import ParallelismConfigSchema

        assert ParallelismConfigSchema(strategy=value).strategy == value

    def test_rejects_an_unknown_strategy_at_load_time(self) -> None:
        """Not at build time, on the cluster, after models+optimizers exist."""
        from spectramr.config.schemas.base import ParallelismConfigSchema

        with pytest.raises(ValidationError, match="none"):
            ParallelismConfigSchema(strategy="horovod")

    def test_rejects_an_unknown_backend(self) -> None:
        from spectramr.config.schemas.base import ParallelismConfigSchema

        with pytest.raises(ValidationError):
            ParallelismConfigSchema(backend="horovod")

    def test_default_is_single_process_and_constructs(self) -> None:
        """Every corpus config relies on this: no parallel block => plain training."""
        from spectramr.config.schemas.base import ParallelismConfigSchema

        cfg = ParallelismConfigSchema()
        assert cfg.strategy == "none"
        assert cfg.backend == "nccl"
        assert cfg.fsdp.enabled is False
        assert cfg.deepspeed.enabled is False


class TestStrategyAndSubBlockAreOneDeclaration:
    """``strategy`` and the sub-block flag can no longer disagree."""

    def test_fsdp_requires_both_halves(self) -> None:
        from spectramr.config.schemas.base import ParallelismConfigSchema

        cfg = ParallelismConfigSchema(strategy="fsdp", fsdp={"enabled": True})
        assert cfg.strategy == "fsdp" and cfg.fsdp.enabled

    def test_strategy_fsdp_without_the_flag_raises(self) -> None:
        """Previously: accepted here, then ValueError deep inside the build."""
        from spectramr.config.schemas.base import ParallelismConfigSchema

        with pytest.raises(ValidationError, match="disagrees"):
            ParallelismConfigSchema(strategy="fsdp")

    def test_flag_without_the_strategy_raises(self) -> None:
        """Previously: silently sharded while the config read 'no parallelism'."""
        from spectramr.config.schemas.base import ParallelismConfigSchema

        with pytest.raises(ValidationError, match="disagrees"):
            ParallelismConfigSchema(fsdp={"enabled": True})

    def test_deepspeed_requires_both_halves(self) -> None:
        from spectramr.config.schemas.base import ParallelismConfigSchema

        cfg = ParallelismConfigSchema(
            strategy="deepspeed", deepspeed={"enabled": True, "zero_stage": 3}
        )
        assert cfg.deepspeed.zero_stage == 3

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"strategy": "deepspeed"},
            {"deepspeed": {"enabled": True}},
        ],
        ids=["strategy-only", "flag-only"],
    )
    def test_half_declared_deepspeed_raises(self, kwargs: dict) -> None:
        from spectramr.config.schemas.base import ParallelismConfigSchema

        with pytest.raises(ValidationError, match="disagrees"):
            ParallelismConfigSchema(**kwargs)

    def test_sync_batchnorm_is_rejected_for_dataparallel(self) -> None:
        """SyncBatchNorm needs a process group; DataParallel creates none.

        The old code converted BN for BOTH dp and ddp, so a dp arm with
        sync_batch_norm built modules that could not run.
        """
        from spectramr.config.schemas.base import ParallelismConfigSchema

        with pytest.raises(ValidationError, match="sync_batch_norm"):
            ParallelismConfigSchema(strategy="dp", sync_batch_norm=True)

        assert ParallelismConfigSchema(
            strategy="ddp", sync_batch_norm=True
        ).sync_batch_norm


class TestFSDPTransformerWrapPolicyIsNoLongerAFacade:
    """``transformer_block`` used to wrap NOTHING while logging success.

    ``_build_auto_wrap_policy`` passed ``transformer_layer_cls=set()``, which
    matches no module class, so FSDP applied only at the root — effectively
    NO_SHARD — while ``fsdp_wrap`` logged "FSDP wrapped: sharding=full_shard".
    There was no config surface to populate the class set, so the option could
    not be made to work from YAML at all (#621 F1).
    """

    def test_transformer_block_without_layer_classes_raises(self) -> None:
        from spectramr.config.schemas.base import ParallelismConfigSchema

        with pytest.raises(ValidationError, match="transformer_layer_cls"):
            ParallelismConfigSchema(
                strategy="fsdp",
                fsdp={"enabled": True, "auto_wrap_policy": "transformer_block"},
            )

    def test_transformer_block_with_layer_classes_is_accepted(self) -> None:
        from spectramr.config.schemas.base import ParallelismConfigSchema

        cfg = ParallelismConfigSchema(
            strategy="fsdp",
            fsdp={
                "enabled": True,
                "auto_wrap_policy": "transformer_block",
                "transformer_layer_cls": ["TransformerBlock"],
            },
        )
        assert cfg.fsdp.transformer_layer_cls == ["TransformerBlock"]

    def test_the_reference_templates_misspelling_is_rejected(self) -> None:
        """v1.0_reference.yaml (then v6.1) advertised 'transformer_based'; code took
        'transformer_block'. The reachable spelling was the inert one and the
        documented spelling crashed."""
        from spectramr.config.schemas.base import ParallelismConfigSchema

        with pytest.raises(ValidationError):
            ParallelismConfigSchema(
                strategy="fsdp",
                fsdp={"enabled": True, "auto_wrap_policy": "transformer_based"},
            )

    def test_layer_classes_are_not_required_by_other_policies(self) -> None:
        from spectramr.config.schemas.base import ParallelismConfigSchema

        for policy in ("size_based", "none"):
            cfg = ParallelismConfigSchema(
                strategy="fsdp",
                fsdp={"enabled": True, "auto_wrap_policy": policy},
            )
            assert cfg.fsdp.auto_wrap_policy == policy


class TestDeepSpeedConfigSchema:
    """The ZeRO knobs, and the deliberate absence of a ds_config path."""

    def test_has_no_config_path_field(self) -> None:
        """A hand-edited ds_config.json is a cluster-critical file with no
        committed generator, and a divergence from the YAML would be SILENT
        (DeepSpeed accepts a micro-batch that contradicts data.batch_size).
        ``build_deepspeed_config`` is the generator; the dict is derived."""
        from spectramr.config.schemas.base import DeepSpeedConfigSchema

        fields = set(DeepSpeedConfigSchema.model_fields)
        assert not {"config_path", "ds_config_path", "config_file"} & fields

    def test_does_not_redeclare_keys_owned_by_other_ssots(self) -> None:
        """batch size / accumulation / clipping / precision come from their own
        blocks. Declaring them here is how a ds_config silently disagrees."""
        from spectramr.config.schemas.base import DeepSpeedConfigSchema

        fields = set(DeepSpeedConfigSchema.model_fields)
        forbidden = {
            "train_micro_batch_size_per_gpu",
            "batch_size",
            "gradient_accumulation_steps",
            "gradient_clipping",
            "fp16",
            "bf16",
            "amp_dtype",
            "optimizer",
            "scheduler",
        }
        assert not forbidden & fields

    @pytest.mark.parametrize("stage", [0, 1, 2, 3])
    def test_accepts_the_real_zero_stages(self, stage: int) -> None:
        from spectramr.config.schemas.base import DeepSpeedConfigSchema

        assert DeepSpeedConfigSchema(zero_stage=stage).zero_stage == stage

    def test_rejects_a_nonexistent_zero_stage(self) -> None:
        from spectramr.config.schemas.base import DeepSpeedConfigSchema

        with pytest.raises(ValidationError):
            DeepSpeedConfigSchema(zero_stage=4)

    def test_nvme_offload_requires_a_path(self) -> None:
        from spectramr.config.schemas.base import DeepSpeedConfigSchema

        with pytest.raises(ValidationError, match="nvme_path"):
            DeepSpeedConfigSchema(offload_optimizer="nvme")

        assert DeepSpeedConfigSchema(
            offload_optimizer="nvme", nvme_path="/scratch/nvme"
        ).nvme_path

    def test_multi_engine_is_off_by_default(self) -> None:
        """engine.step() issues collectives, so a GAN whose discriminator steps
        every k iterations DEADLOCKS rather than erroring. Opt-in only."""
        from spectramr.config.schemas.base import DeepSpeedConfigSchema

        assert DeepSpeedConfigSchema().allow_multi_engine is False

    def test_a_typo_is_rejected_not_swallowed(self) -> None:
        from spectramr.config.schemas.base import DeepSpeedConfigSchema

        with pytest.raises(ValidationError):
            DeepSpeedConfigSchema(zero_stge=3)


class TestDeepSpeedFeatureBlocks:
    """DeepCompile and ZenFlow constraints, caught at load rather than on a node.

    Every rule here mirrors one DeepSpeed enforces itself -- the point is WHERE.
    DeepSpeed raises from inside ``deepspeed.initialize``, i.e. after the model,
    optimizer, losses, physics and dataloaders have been built on a cluster node.
    """

    @staticmethod
    def _ds(**kw):
        from spectramr.config.schemas.base import DeepSpeedConfigSchema

        return DeepSpeedConfigSchema(enabled=True, **kw)

    def test_zenflow_requires_cpu_offload(self):
        """DeepSpeed raises 'Zenflow must be used with cpu offload'; ZenFlow's
        whole subject is the offloaded optimizer step."""
        import pytest

        with pytest.raises(ValueError, match="offload_optimizer"):
            self._ds(zero_stage=3, zenflow={"enabled": True})

    def test_zenflow_with_offload_is_accepted(self):
        ds = self._ds(zero_stage=3, offload_optimizer="cpu", zenflow={"enabled": True})
        assert ds.zenflow.enabled

    def test_zenflow_selective_offload_needs_overlap_comm(self):
        import pytest

        with pytest.raises(ValueError, match="overlap_comm"):
            self._ds(
                zero_stage=2,
                offload_optimizer="cpu",
                overlap_comm=False,
                zenflow={"enabled": True, "offload": True},
            )

    def test_auto_strategy_rejects_an_explicit_select_interval(self):
        """DeepSpeed forces it to 1 and *raises a Warning instance* -- an odd
        control flow to meet on a cluster node."""
        import pytest

        with pytest.raises(ValueError, match="select_interval"):
            self._ds(
                zero_stage=3,
                offload_optimizer="cpu",
                zenflow={"enabled": True, "select_interval": 5},
            )

    def test_non_auto_strategy_requires_an_integer_interval(self):
        import pytest

        with pytest.raises(ValueError, match="integer select_interval"):
            self._ds(
                zero_stage=3,
                offload_optimizer="cpu",
                zenflow={"enabled": True, "select_strategy": "step"},
            )

    def test_select_interval_must_not_be_below_update_interval(self):
        import pytest

        with pytest.raises(ValueError, match="must be >="):
            self._ds(
                zero_stage=3,
                offload_optimizer="cpu",
                zenflow={
                    "enabled": True,
                    "select_strategy": "step",
                    "select_interval": 2,
                    "update_interval": 8,
                },
            )

    def test_zenflow_is_off_by_default(self):
        assert self._ds(zero_stage=3).zenflow.enabled is False

    def test_z3_pass_requires_stage_three(self):
        """Pass names are dispatch keys, not labels: compile_zero_optimization_stage
        branches on exactly these strings."""
        import pytest

        with pytest.raises(ValueError, match="z3"):
            self._ds(zero_stage=2, compile={"enabled": True, "passes": ["z3"]})

    def test_z1_pass_requires_stage_one_or_two(self):
        import pytest

        with pytest.raises(ValueError, match="z1"):
            self._ds(zero_stage=3, compile={"enabled": True, "passes": ["z1"]})

    def test_matching_pass_and_stage_are_accepted(self):
        assert self._ds(
            zero_stage=3, compile={"enabled": True, "passes": ["z3"]}
        ).compile.passes == ["z3"]

    def test_compile_knobs_without_enabled_are_rejected(self):
        """DeepSpeed reads the compile block only when deepcompile is true, so
        these would render faithfully into the ds_config and be ignored."""
        import pytest

        with pytest.raises(ValueError, match="enabled is false"):
            self._ds(zero_stage=3, compile={"passes": ["z3"]})

    def test_deepcompile_is_off_by_default(self):
        assert self._ds(zero_stage=3).compile.enabled is False

    def test_unknown_keys_are_forbidden_in_both_blocks(self):
        import pytest

        with pytest.raises(ValueError):
            self._ds(zero_stage=3, compile={"enabled": True, "deepcompile": True})
        with pytest.raises(ValueError):
            self._ds(
                zero_stage=3,
                offload_optimizer="cpu",
                zenflow={"enabled": True, "topk": 1},
            )


class TestDisabledFeatureBlocksRejectDeclaredKnobs:
    """A knob under a disabled block never reaches DeepSpeed at all.

    The generator OMITS the whole block, so this is stronger than "ignored":
    for zenflow it must omit it (``ZenFlowConfig`` has no ``enabled`` field and
    DeepSpeed branches on ``None``, so ``{}`` would turn ZenFlow on with all
    defaults). Checked over every field rather than a curated subset -- a hand
    list goes stale the moment a field is added.
    """

    @staticmethod
    def _ds(**kw):
        from spectramr.config.schemas.base import DeepSpeedConfigSchema

        return DeepSpeedConfigSchema(enabled=True, zero_stage=3, **kw)

    def test_compile_non_boolean_knob_is_rejected(self):
        import pytest

        with pytest.raises(ValueError, match="enabled is false"):
            self._ds(compile={"double_buffer": False})

    def test_compile_debug_log_is_rejected_too(self):
        """Previously only a hand-listed subset raised; debug_log slipped through."""
        import pytest

        with pytest.raises(ValueError, match="enabled is false"):
            self._ds(compile={"debug_log": True})

    def test_zenflow_knob_without_enabled_is_rejected(self):
        import pytest

        with pytest.raises(ValueError, match="enabled is false"):
            self._ds(offload_optimizer="cpu", zenflow={"topk_ratio": 0.5})

    def test_an_explicit_enabled_false_alone_is_fine(self):
        ds = self._ds(compile={"enabled": False}, zenflow={"enabled": False})
        assert ds.compile.enabled is False
        assert ds.zenflow.enabled is False

    def test_omitting_the_blocks_entirely_is_fine(self):
        ds = self._ds()
        assert ds.compile.enabled is False and ds.zenflow.enabled is False


class TestParallelDocstringsNameModulesThatExist:
    """A docstring that names a consumer is a claim, and claims rot silently.

    ``DeepSpeedConfigSchema`` pointed at
    ``...distributed.deepspeed.config_builder`` -- the package was renamed to
    ``deepspeed_backend`` precisely so a call site could not be confused with
    the upstream ``deepspeed`` package, and this reference was left behind. It
    is the SSOT pointer for "where does the ds_config come from", so following
    it gave ``ModuleNotFoundError`` on the one path a reader most needs.

    Resolves every dotted ``spectramr.*`` path in these docstrings rather than
    pinning the one that was wrong: a hardcoded expected string is a second
    copy of the very fact that drifted.
    """

    import re as _re

    #: Matches ``spectramr.a.b.c`` inside prose/backticks. Trailing ``.name``
    #: segments that are attributes rather than modules are peeled off by the
    #: resolver below, so the pattern does not need to know the difference.
    _PATH = _re.compile(r"spectramr(?:\.[a-z_][a-z0-9_]*)+")

    @staticmethod
    def _schemas():
        from spectramr.config.schemas.base import (
            DeepSpeedCompileConfigSchema,
            DeepSpeedConfigSchema,
            DeepSpeedZenFlowConfigSchema,
            FSDPConfigSchema,
            ParallelismConfigSchema,
        )

        return [
            DeepSpeedCompileConfigSchema,
            DeepSpeedConfigSchema,
            DeepSpeedZenFlowConfigSchema,
            FSDPConfigSchema,
            ParallelismConfigSchema,
        ]

    @staticmethod
    def _resolve(path: str) -> str | None:
        """``None`` if *path* resolves, else the segment that broke it.

        Peels trailing segments until a module imports, then requires the
        peeled ones to resolve as ATTRIBUTES of it. Stopping at "some prefix
        imported" is not enough and is worth spelling out, because the first
        version of this test did exactly that and was a rubber stamp: the
        broken ``...distributed.deepspeed.config_builder.build_deepspeed_config``
        peels back to ``...distributed``, which imports fine, so the check
        passed on the very defect it was written for.
        """
        import importlib

        parts = path.split(".")
        tail: list[str] = []
        module = None
        while parts:
            try:
                module = importlib.import_module(".".join(parts))
                break
            except ModuleNotFoundError:
                tail.insert(0, parts.pop())
            except Exception:
                # Imported far enough to fail for its OWN reasons (a missing
                # optional extra); the path itself is sound.
                return None
        if module is None:
            return path

        obj = module
        for seg in tail:
            obj = getattr(obj, seg, None)
            if obj is None:
                return f"{'.'.join(parts)} has no {seg!r}"
        return None

    def test_every_referenced_module_path_resolves(self):
        unresolvable = []
        for schema in self._schemas():
            for path in sorted(set(self._PATH.findall(schema.__doc__ or ""))):
                broken = self._resolve(path)
                if broken:
                    unresolvable.append(f"{schema.__name__}: {path} ({broken})")

        assert (
            not unresolvable
        ), "docstrings name modules/attributes that do not exist: " + "; ".join(
            unresolvable
        )

    def test_the_check_rejects_the_path_that_actually_rotted(self):
        """Guard the guard: pin that a wrong path is detected, not just absent.

        Without this, a future refactor of ``_resolve`` could silently relax it
        back into the always-green form and nothing would notice.
        """
        assert self._resolve(
            "spectramr.infrastructure.distributed.deepspeed.config_builder"
        )
        assert (
            self._resolve(
                "spectramr.infrastructure.distributed.deepspeed_backend."
                "config_builder.build_deepspeed_config"
            )
            is None
        )



#: A config tier is a bare dotted numeric string: "1.0", "5.0", "6.1".
_CONFIG_TIER_SHAPE = re.compile(r"\d+(?:\.\d+)+")


def looks_like_a_config_tier(value: object) -> bool:
    """Whether ``value`` is spelled the way a ``config_version`` is.

    The predicate for "this free-form field is holding a schema version". Shape,
    not membership: ``LEGACY_CONFIG_VERSIONS`` is empty since PR #891, so every
    retired tier (``5.0``, ``6.0``, ``6.1``) belongs to no set the code still
    holds, and a membership test goes green on exactly the values that motivated
    it. Imported by ``test_defaults_provider.py``, which owns the same invariant
    for the defaults provider — one definition, two call sites (non-negotiable 17).
    """
    return isinstance(value, str) and bool(_CONFIG_TIER_SHAPE.fullmatch(value))


class TestExperimentMetadataVersionIsNotASchemaTier:
    """``ExperimentMetadataSchema.version`` versions the ARM, not the SCHEMA.

    The twin of the defect already pinned in
    ``tests/unit/config/schemas/test_defaults_provider.py``. That file fixed the
    *defaults provider*'s ``metadata: {"version": "6.0"}``; this schema declared
    the same retired tier as its own field default, and nothing tied the two
    together — exactly the "three independent literals, each agreeing with
    itself" shape this module's docstring opens with.

    Kept separate from the corpus drain deliberately. The production load path
    types ``TrainingSettings.metadata`` as a raw ``dict[str, Any]``, so this
    schema is a *latent* surface: it reached no arm, and a corpus that is clean
    today says nothing about whether this default is. Only a test on the field
    itself can fire while the defect is still latent.
    """

    def test_the_default_is_not_stated(self) -> None:
        """``None`` == "the author did not say", like every free-form neighbour."""
        from spectramr.config.schemas.base import ExperimentMetadataSchema

        assert ExperimentMetadataSchema().version is None

    def test_the_default_is_never_a_config_tier(self) -> None:
        """The regression that matters, stated independently of the value above.

        Membership is the wrong predicate here, and this was verified by planting
        the exact historical value: with ``default="6.0"`` restored, an
        ``assert version not in ACCEPTED | LEGACY`` **passes**. PR #891 emptied
        the legacy set, so the retired tier this test exists to catch is no longer
        a member of anything — the check would go green on the one input it was
        written for, which is also why the old default read as harmless for so
        long.

        So this asserts the SHAPE instead: a config tier is a bare dotted numeric
        string, which is what ``CANONICAL_CONFIG_VERSION`` looks like and what
        every retired tier looked like. An author revision may still *contain*
        digits (``"v2"``, ``"2026-08-rev3"``); it just must not be spelled the way
        a tier is.
        """
        from spectramr.config.schemas.base import (
            CANONICAL_CONFIG_VERSION,
            ExperimentMetadataSchema,
        )

        version = ExperimentMetadataSchema().version
        assert not looks_like_a_config_tier(version), (
            f"the schema declares {version!r} as author metadata, and it is "
            "spelled like a config tier; `metadata.version` is the author's "
            "experiment revision, not a schema version"
        )
        # Stated separately: even the *current* tier is wrong here — it would make
        # every arm claim an author revision it never wrote, and the next bump
        # would silently rewrite it.
        assert version != CANONICAL_CONFIG_VERSION

    def test_an_author_revision_still_round_trips(self) -> None:
        """Widening to ``str | None`` must not stop an author declaring one.

        ``experiment_12_image_cold_diffusion_v2.yaml`` carries ``'2.0'`` and has
        ``_v2`` in its filename — the corpus evidence that this field really is
        an author revision. 84 ``inprogress/`` arms declare such a value and the
        drain deliberately left every one of them alone.
        """
        from spectramr.config.schemas.base import ExperimentMetadataSchema

        assert ExperimentMetadataSchema(version="2.0").version == "2.0"


class TestComparisonNote:
    """``metadata.comparison_note``: typed, optional, declared where #2101 needs it."""

    def test_defaults_to_none_and_rejects_a_non_string(self) -> None:
        from spectramr.config.schemas.base import ExperimentMetadataSchema

        assert ExperimentMetadataSchema().comparison_note is None
        with pytest.raises(ValidationError):
            ExperimentMetadataSchema(comparison_note={"two": "blocks"})

    def test_the_dual_domain_arm_declares_its_second_attention_block(self) -> None:
        from pathlib import Path

        arm = (
            Path(__file__).resolve().parents[4]
            / "experiments/inprogress/kspace_filling/attention_shootout"
            / "experiment_11_attention_dual_domain.yaml"
        )
        meta = TrainingSettings.from_yaml(str(arm)).metadata
        note = meta.get("comparison_note") if isinstance(meta, dict) else meta.comparison_note
        assert note and "PhaseSafeDualAttention" in note


class TestExperimentMetadataTagsDefault:
    def test_each_instance_gets_its_own_empty_tags(self) -> None:
        from spectramr.config.schemas.base import ExperimentMetadataSchema

        a, b = ExperimentMetadataSchema.model_construct(), ExperimentMetadataSchema.model_construct()
        assert a.tags == {} and a.tags is not b.tags

    def test_a_list_of_tags_still_becomes_a_dict(self) -> None:
        from spectramr.config.schemas.base import ExperimentMetadataSchema

        assert ExperimentMetadataSchema.validate_tags(["a", "b"]) == {"a": "a", "b": "b"}


def test_the_mypy_hook_does_not_follow_imports_into_unrelated_packages() -> None:
    """#2706: without --follow-imports=silent the torch-less hook env reports errors in
    models/ for any commit under config/, core/ or domain/."""
    from pathlib import Path

    text = (Path(__file__).resolve().parents[4] / ".pre-commit-config.yaml").read_text()
    mypy_block = text[text.index("id: mypy") :].split("- repo:")[0]
    assert "--follow-imports=silent" in mypy_block


class TestFSDPPrecisionDefault:
    """sbatch pins no GPU type and the clusters mix Ada and V100 (no native bf16)."""

    def test_precision_is_float32_unless_an_arm_declares_one(self) -> None:
        from spectramr.config.schemas.base import FSDPConfigSchema

        assert FSDPConfigSchema(enabled=True).mixed_precision is None

    def test_an_arm_may_still_declare_bf16(self) -> None:
        from spectramr.config.schemas.base import FSDPConfigSchema

        assert FSDPConfigSchema(enabled=True, mixed_precision="bf16").mixed_precision == "bf16"
