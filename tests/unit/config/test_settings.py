"""Tests for the main TrainingSettings configuration class.

Covers the two config-hygiene guarantees added 2026-05-28:

* H2 (CLAUDE.md pitfall #15) — the top-level ``model_domain`` knob is no
  longer a silent no-op: it propagates into ``model.model_domain`` /
  ``model.target_domain`` (where consumers actually read it) and raises on
  conflict.
* H3 (CLAUDE.md pitfall #10) — deprecated top-level ``diffusion`` /
  ``artifacts`` blocks now emit a runtime ``DeprecationWarning`` on load
  (Pydantic's ``Field(deprecated=...)`` marker alone does not).

NOTE on the strict warning policy: ``pyproject.toml`` promotes any
``DeprecationWarning`` raised by ``spectramr.*`` to an error during tests, so
the H3 tests MUST consume the warning via ``pytest.warns(DeprecationWarning)``
(which it does) — otherwise the warning would fail the run.
"""

import copy
from typing import ClassVar

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.base import (
    ACCEPTED_CONFIG_VERSIONS,
    CANONICAL_CONFIG_VERSION,
    LEGACY_CONFIG_VERSIONS,
)
from spectramr.config.schemas.checkpoint import CheckpointConfigSchema
from spectramr.config.settings import TrainingSettings

#: Tiers that were once declarable and are now unloadable. They are NOT in
#: ``LEGACY_CONFIG_VERSIONS`` -- that set was emptied when the fold was deleted
#: (PR #891, 2026-08-08), so every guard written as "iterate the legacy set"
#: silently became a no-op on the very day the tiers it guarded became fatal.
#: Naming them keeps those guards meaningful.
RETIRED_CONFIG_TIERS: tuple[str, ...] = ("6.0", "6.1")


def _base_sections() -> dict:
    """Return the always-required sections (no ``model`` block)."""
    return {
        "data": {
            "train_path": "/tmp/train",
            "val_path": "/tmp/val",
            "batch_size": 2,
        },
        "optimization": {"learning_rate": 1e-4},
        "logging": {},
    }


def _minimal_config() -> dict:
    """Return a minimal valid configuration dictionary."""
    cfg = _base_sections()
    cfg["model"] = {"model_type": "unet"}
    return cfg


def test_minimal_config_loads():
    """A minimal config dict should construct a valid TrainingSettings."""
    assert TrainingSettings(**_minimal_config()).model.model_type == "unet"


def test_frozen_config_rejects_mutation():
    """TrainingSettings is frozen; attribute assignment must fail."""
    settings = TrainingSettings(**_minimal_config())
    with pytest.raises(ValidationError):
        settings.seed = 999


def test_provenance_snapshot_stamps_defaulted_knobs():
    """WS-1 round-2: get_validated_snapshot must NOT use exclude_unset, so a
    knob left at its default still appears in the run's provenance (pitfall
    #15c — provenance must prove which value drove the run)."""
    cfg = _minimal_config()  # never sets ``run.seed``
    settings = TrainingSettings(**cfg)
    snapshot = settings.get_validated_snapshot()
    # ``run.seed`` was left unset -> under exclude_unset it would be ABSENT.
    # (The seed lives under ``run:`` since phase 4b; it was a root scalar.)
    assert "run" in snapshot and "seed" in snapshot["run"], (
        "defaulted knob missing from provenance snapshot — exclude_unset regressed (pitfall #15c)"
    )
    assert snapshot["run"]["seed"] == settings.run.seed


# ---------------------------------------------------------------------------
# H2: top-level ``model_domain`` is a wired knob, not a silent no-op (#15)
# ---------------------------------------------------------------------------


def test_the_root_model_domain_spelling_raises_and_names_its_replacement():
    """Phase 4b retired the root ``model_domain``: it was a WRITE-ONLY alias
    that a ``mode="before"`` validator copied into ``model.model_domain`` /
    ``model.target_domain``, which are the fields consumers actually read.

    The four tests that used to sit here exercised that propagation, plus its
    conflict detection. They are deleted rather than inverted -- an inverted
    test keeps a name asserting what it no longer checks.
    """
    cfg = _base_sections()
    cfg["model"] = {"model_type": "unet"}
    with pytest.raises(ValidationError) as exc:
        TrainingSettings(model_domain="kspace", **cfg)
    msg = str(exc.value)
    assert "model.model_domain" in msg, "the error must name the replacement"
    assert "migrate_config_keys.py" in msg, "and the fixer that applies it"


def test_the_nested_model_domain_is_what_consumers_read():
    """The surviving path: declare it where it is read."""
    cfg = _base_sections()
    cfg["model"] = {"model_type": "unet", "model_domain": "kspace",
                    "target_domain": "kspace"}
    settings = TrainingSettings(**cfg)
    assert settings.model.model_domain == "kspace"
    assert settings.model_dump()["model"]["target_domain"] == "kspace"


def test_apply_overrides_raises_on_non_dict_traversal():
    """Regression (2026-07-01, pitfall #9): an override path that traverses an
    existing scalar node used to silently replace it with ``{}`` — destroying
    the config subtree on typo'd paths. On permissive fields the Pydantic
    re-validation cannot catch the loss, so the traversal itself must raise."""
    from spectramr.main import apply_overrides

    settings = TrainingSettings(**_minimal_config())
    with pytest.raises(ValueError, match="non-dict config node"):
        apply_overrides(settings, ["optimization.optimizer.learning_rate.nested=1"])


def test_apply_overrides_still_builds_absent_paths():
    """Building a nested path through absent/None nodes remains legal —
    only *existing non-dict* nodes are protected."""
    from spectramr.main import apply_overrides

    settings = TrainingSettings(**_minimal_config())
    updated = apply_overrides(settings, ["optimization.lr_scheduler_kwargs.step_size=5"])
    assert updated.optimization.lr_scheduler_kwargs["step_size"] == 5


def test_model_dump_round_trip_idempotent_for_kspace_config():
    """``model_dump()`` then reconstruct must NOT raise for a kspace config.

    Regression for the smoke-test break (2026-05-29): ``apply_overrides`` in
    ``main.py`` dumps the loaded settings and re-validates. With the top-level
    ``model_domain`` field defaulting to ``"image"``, ``model_dump()``
    materialized ``model_domain="image"`` as an explicit key, which the
    ``mode="before"`` reconcile validator then read back and flagged as
    conflicting with ``model.model_domain="kspace"`` — a non-idempotent
    round-trip that broke every non-image experiment the moment any
    ``--override`` was applied.
    """
    cfg = _base_sections()
    cfg["model"] = {"model_type": "unet", "model_domain": "kspace"}
    settings = TrainingSettings(**cfg)

    # No top-level model_domain was supplied, so the round-trip must be a no-op:
    # it must not raise, and the nested value must survive unchanged.
    rebuilt = TrainingSettings(**settings.model_dump())
    assert rebuilt.model.model_domain == "kspace"
    assert rebuilt.model.target_domain == settings.model.target_domain


def test_no_top_level_model_domain_leaves_nested_default():
    """Omitting the top-level knob leaves nested defaults untouched.

    ``_propagate_top_level_model_domain`` runs in ``mode="before"`` on the raw
    input dict, so it only sees an *explicitly supplied* ``model_domain``. With
    nothing supplied, the top-level field takes its ``None`` default (which the
    validator treats as "unspecified") and the nested ``ModelConfigSchema``
    fields keep their own schema default (``None``); they are NOT back-filled.
    """
    settings = TrainingSettings(**_minimal_config())
    assert settings.model.target_domain is None
    assert settings.model.model_domain is None


# ---------------------------------------------------------------------------
# H3: deprecated top-level blocks emit a runtime DeprecationWarning (#10)
# ---------------------------------------------------------------------------


def test_top_level_diffusion_emits_deprecation_warning():
    """Loading a config with top-level ``diffusion`` must warn (and is consumed)."""
    cfg = _minimal_config()
    cfg["diffusion"] = {"num_timesteps": 1000}
    with pytest.warns(DeprecationWarning, match="config.diffusion is unused"):
        settings = TrainingSettings(**cfg)
    # Field still loads (kept for legacy compat under extra='forbid').
    assert settings.diffusion == {"num_timesteps": 1000}


def test_top_level_artifacts_is_refused():
    """``artifacts`` was removed (#698); the refusal names the live output path."""
    cfg = _minimal_config()
    cfg["artifacts"] = {"persistent_root": "/tmp/artifacts"}
    with pytest.raises(ValidationError, match=r"`artifacts` was removed.*training\.output_dir"):
        TrainingSettings(**cfg)


def test_clean_config_emits_no_deprecation_warning():
    """A config without the deprecated blocks must not warn (strict policy).

    Regression for the descriptor-access footgun: Pydantic's
    ``Field(deprecated=...)`` emits its warning on every attribute *access*,
    not on *set*. The ``_warn_on_deprecated_top_level_blocks`` validator
    therefore must NOT read ``self.diffusion`` / ``self.artifacts`` through the
    normal attribute path (which would warn on every load, even for clean
    configs that never carried the block); it reads ``self.__dict__`` instead.
    If that regresses, this test fails because the validator's guard re-fires
    the warning and ``simplefilter("error")`` promotes it.
    """
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", DeprecationWarning)
        TrainingSettings(**_minimal_config())  # would raise if a warning fired


def test_clean_config_round_trip_via_from_yaml_does_not_warn(tmp_path):
    """The same no-warn guarantee must hold on the real ``from_yaml`` path.

    ``from_yaml`` runs the full validator stack (model-domain propagation,
    registry check, deprecated-block surfacing). A clean YAML — no top-level
    ``diffusion`` / ``artifacts`` — must round-trip without tripping the
    deprecation warning that the descriptor footgun used to raise.
    """
    import warnings as _warnings

    import yaml as _yaml

    cfg = _minimal_config()
    cfg["config_version"] = CANONICAL_CONFIG_VERSION
    path = tmp_path / "clean.yaml"
    path.write_text(_yaml.safe_dump(cfg), encoding="utf-8")

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", DeprecationWarning)
        settings = TrainingSettings.from_yaml(path)  # raises if a warning fired
    assert settings.model.model_type == "unet"


def test_from_yaml_accepts_only_the_canonical_version(tmp_path):
    """``from_yaml`` accepts the canonical version and nothing else.

    This guarded the opposite contract until 2026-08-08: the loader accepted
    ``{"6.0", "6.1"}`` and folded them. With the legacy tier drained, accepting
    them would be accepting a spelling no corpus file uses and no code reads.
    An unsupported version still raises, as it always did.
    """
    import yaml as _yaml

    for version in ("6.0", "6.1"):
        cfg = _minimal_config()
        cfg["config_version"] = version
        path = tmp_path / f"cfg_{version}.yaml"
        path.write_text(_yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(ValueError, match="not supported"):
            TrainingSettings.from_yaml(path)

    cfg = _minimal_config()
    cfg["config_version"] = CANONICAL_CONFIG_VERSION
    path = tmp_path / "cfg_canonical.yaml"
    path.write_text(_yaml.safe_dump(cfg), encoding="utf-8")
    assert TrainingSettings.from_yaml(path).model.model_type == "unet"

    # An unsupported version still raises (no silent acceptance).
    cfg = _minimal_config()
    cfg["config_version"] = "5.0"
    bad = tmp_path / "cfg_5.yaml"
    bad.write_text(_yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="not supported"):
        TrainingSettings.from_yaml(bad)


# ---------------------------------------------------------------------------
# physics.coil_processing: the new unified block drives the legacy compression
# machinery on the real ``from_yaml`` path (configurable coil-processing pipeline).
# ---------------------------------------------------------------------------


def test_legacy_coil_config_loads_silently(tmp_path):
    """A legacy ``data.coil_processing_mode`` config must load WITHOUT any
    warning.

    Regression: an over-eager legacy→new "migration" shim used to emit a
    ``DeprecationWarning`` here, which — promoted to an error in tests — broke
    every one of the ~200 YAMLs that use the (still first-class)
    ``coil_processing_mode`` knob. ``data.coil_processing_mode`` is NOT
    deprecated; the new block is an additive, opt-in front-end.
    """
    import warnings as _warnings

    import yaml as _yaml

    cfg = _minimal_config()
    cfg["config_version"] = CANONICAL_CONFIG_VERSION
    cfg["data"]["coil_processing_mode"] = "svd"
    cfg["data"]["num_virtual_coils"] = 6
    path = tmp_path / "legacy_coil.yaml"
    path.write_text(_yaml.safe_dump(cfg), encoding="utf-8")

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", DeprecationWarning)  # any → failure
        settings = TrainingSettings.from_yaml(path)

    # Legacy keys are preserved unchanged (the new block is opt-in, not
    # synthesized — physics stays at its default here since none was authored).
    assert settings.data.coils.processing_mode == "svd"
    assert settings.data.coils.num_virtual_coils == 6


def test_new_block_svd_drives_legacy_mode_via_from_yaml(tmp_path):
    """A user-authored physics.coil_processing.compression=svd must drive the
    legacy data.coil_processing_mode machinery (pitfall #15 — the new knob is
    not a silent no-op; it executes via the existing subject-builder path)."""
    import yaml as _yaml

    cfg = _minimal_config()
    cfg["config_version"] = CANONICAL_CONFIG_VERSION
    cfg["physics"] = {
        "coil_processing": {
            "compression": {
                "method": "svd",
                "num_virtual_coils": 6,
                "calibration_lines": 24,
            },
            "estimation": {"method": "none", "enabled": False},
        }
    }
    path = tmp_path / "new_svd.yaml"
    path.write_text(_yaml.safe_dump(cfg), encoding="utf-8")

    settings = TrainingSettings.from_yaml(path)

    # The new block is preserved AND projected onto the legacy compression knobs
    # that FastMRISubjectBuilder._apply_coil_processing / SVDCoilCompression read.
    assert settings.physics.coil_processing.compression.method == "svd"
    assert settings.data.coils.processing_mode == "svd"
    assert settings.data.coils.num_virtual_coils == 6
    assert settings.data.coils.svd_calibration_lines == 24


# ---------------------------------------------------------------------------
# plugins: top-level out-of-tree plugin block (custom components outside tree)
# ---------------------------------------------------------------------------


def test_plugins_block_accepted_and_typed():
    """A typed ``plugins:`` block must be accepted under ``extra='forbid'``."""
    cfg = _minimal_config()
    cfg["plugins"] = {"enabled": True, "paths": ["mypkg.models.my_unet"]}
    settings = TrainingSettings(**cfg)
    assert settings.plugins is not None
    assert settings.plugins.enabled is True
    assert settings.plugins.paths == ["mypkg.models.my_unet"]


def test_plugins_defaults_to_none_when_absent():
    """Omitting ``plugins`` leaves the field at its ``None`` default."""
    settings = TrainingSettings(**_minimal_config())
    assert settings.plugins is None


# ---------------------------------------------------------------------------
# settings_from_dict: in-memory construction without writing a YAML file
# ---------------------------------------------------------------------------


def test_settings_from_dict_minimal_without_config_version():
    """``settings_from_dict`` accepts a dict with NO ``config_version``.

    Unlike ``from_yaml`` (which requires it), the in-memory path injects the
    default so Python callers are not forced to supply a schema-version key.
    """
    settings = TrainingSettings.settings_from_dict(_minimal_config())
    assert settings.model.model_type == "unet"


def test_flat_data_reserved_cores_folds_into_the_loader_block():
    """`data` is extra="ignore": an unfolded flat knob would be silently discarded."""
    cfg = _minimal_config()
    cfg.setdefault("data", {})["reserved_cores"] = 0
    settings = TrainingSettings.settings_from_dict(cfg)
    assert settings.data.loader.reserved_cores == 0


def test_settings_from_dict_strips_config_version_if_present():
    cfg = _minimal_config()
    cfg["config_version"] = CANONICAL_CONFIG_VERSION
    settings = TrainingSettings.settings_from_dict(cfg)
    assert settings.model.model_type == "unet"


def test_settings_from_dict_rejects_unsupported_config_version():
    cfg = _minimal_config()
    cfg["config_version"] = "5.0"
    with pytest.raises(ValueError, match="not supported"):
        TrainingSettings.settings_from_dict(cfg)


def test_settings_from_dict_rejects_unregistered_model_type():
    """The registry check that guards ``from_yaml`` must also guard this path."""
    cfg = _minimal_config()
    cfg["model"] = {"model_type": "definitely_not_a_real_model_xyz"}
    with pytest.raises(ValueError, match="Invalid model_type"):
        TrainingSettings.settings_from_dict(cfg)


def test_disabled_plugins_with_paths_logs_skip(caplog):
    """``plugins.paths`` set while ``plugins.enabled`` is false → a VISIBLE log,
    not a silent skip (pitfall #15c). The import must not run (the path is
    bogus; if it were imported it would raise)."""
    import logging

    cfg = _minimal_config()  # model_type="unet" — a real in-tree name
    cfg["plugins"] = {"enabled": False, "paths": ["mypkg.never.imported"]}
    with caplog.at_level(logging.INFO, logger="spectramr.config.settings"):
        TrainingSettings.settings_from_dict(cfg)
    assert "plugins.enabled is false" in caplog.text, (
        "expected an INFO log surfacing the disabled-but-populated plugins block"
    )


def test_settings_from_dict_does_not_mutate_caller_dict():
    """G3: the caller's dict is untouched (no in-place ``del`` / bridge edits)."""
    cfg = _minimal_config()
    cfg["config_version"] = CANONICAL_CONFIG_VERSION
    before = copy.deepcopy(cfg)
    TrainingSettings.settings_from_dict(cfg)
    assert cfg == before


def test_settings_from_dict_applies_coil_bridge():
    """The coil-processing bridge that lives only in ``from_yaml`` must run here
    too (pitfall #15): a ``physics.coil_processing.compression=svd`` block must
    drive the legacy ``data.coil_processing_mode`` machinery."""
    cfg = _minimal_config()
    cfg["physics"] = {
        "coil_processing": {
            "compression": {
                "method": "svd",
                "num_virtual_coils": 6,
                "calibration_lines": 24,
            },
            "estimation": {"method": "none", "enabled": False},
        }
    }
    settings = TrainingSettings.settings_from_dict(cfg)
    assert settings.data.coils.processing_mode == "svd"
    assert settings.data.coils.num_virtual_coils == 6


# ---------------------------------------------------------------------------
# Optional ``checkpoint:`` block must default to a usable schema, not None
# ---------------------------------------------------------------------------
def test_checkpoint_defaults_to_schema_when_block_omitted():
    """Omitting the top-level ``checkpoint:`` block must yield a default
    ``CheckpointConfigSchema`` (not ``None``) so the DI container build
    (``bootstrap.build_container``) does not abort with
    'config.checkpoint is required but not provided'.

    Regression (2026-06-20): the entire 28-arm ``mrixfields2026`` cohort
    crashed at bootstrap — every config omits ``checkpoint:`` and the field
    used to default to ``None``. The passing cohorts (kspace_filling 14/14,
    VF) all declare the block, which is why only mrixfields broke. See
    ``TODO/audit/run_debug_kspace_vf_mrixfields_20260617.md``.
    """
    settings = TrainingSettings(**_minimal_config())  # no ``checkpoint:`` block
    assert isinstance(settings.checkpoint, CheckpointConfigSchema)


def test_invalid_model_type_names_the_failed_import(monkeypatch, tmp_path) -> None:
    """An incomplete catalog must say so, not blame the model name.

    Regression for the CI yaml-audit failure of 2026-07-12: the audit lane installed
    no torchvision, so `vf_reconstruction_generators` never imported, so the registry
    held 171 models instead of 588, so `spectramr audit` rejected `neural_complex_sum`
    -- a model that exists and carries @register_model -- as an "Invalid model_type".
    The error must distinguish "you typed a bad name" from "the catalog is short".
    """
    import spectramr.models.init_registry as init_registry
    from spectramr.config.settings import TrainingSettings

    victim = "spectramr.models.generators.vf_reconstruction_generators"
    monkeypatch.setitem(
        init_registry._IMPORT_FAILURES, victim, "ImportError: No module named 'torchvision'"
    )

    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"config_version: '{CANONICAL_CONFIG_VERSION}'\n"
        "model:\n  model_type: definitely_not_a_model\n"
    )

    with pytest.raises(ValueError) as exc:
        TrainingSettings.from_yaml(str(cfg))

    msg = str(exc.value)
    assert "catalog is INCOMPLETE" in msg, "an incomplete catalog must be declared"
    assert victim in msg, "the error must name the module that failed to import"
    assert "torchvision" in msg, "the error must carry the underlying cause"


class TestExecutionLedgerIntegration:
    """``_finalize_from_dict`` records what the declaration became.

    Sensitivity pair throughout: a clean config must stay silent and a config
    with a dropped key must fire. Issue #550 is the reason that polarity is
    asserted explicitly rather than assumed -- a gate whose tests only cover
    the clean case is indistinguishable from a gate that never fires.
    """

    BASE: ClassVar[dict] = {
        "model": {"model_type": "unet"},
        "data": {"batch_size": 2},
        "training": {"max_iterations": 10},
        "optimization": {"learning_rate": 1e-4},
        "logging": {},
    }

    @pytest.fixture(autouse=True)
    def _disarm(self):
        from spectramr.core.execution_ledger import ExecutionLedger

        ExecutionLedger.reset()
        yield
        ExecutionLedger.reset()

    def test_unarmed_load_records_nothing(self):
        """Audit, tests and notebooks must pay only one ContextVar read."""
        from spectramr.core.execution_ledger import ExecutionLedger

        TrainingSettings.settings_from_dict(copy.deepcopy(self.BASE))
        assert ExecutionLedger.recording() is False
        assert ExecutionLedger.current() is None

    def test_clean_config_records_no_dropped_key(self):
        """CONTROL: proves the diff is not simply always firing."""
        from spectramr.core.execution_ledger import ExecutionLedger, SubstitutionClass

        ledger = ExecutionLedger.begin_run(source="control")
        TrainingSettings.settings_from_dict(copy.deepcopy(self.BASE))

        dropped = [
            s
            for s in ledger.substitutions
            if s.class_id is SubstitutionClass.EXTRA_IGNORE_DROPPED
        ]
        assert dropped == [], f"clean config reported drops: {[s.path for s in dropped]}"

    def test_key_the_schema_ignores_is_recorded_at_its_dotted_path(self):
        """DEFECT: the issue #550 mechanism, through the real loader.

        ``AccelerationConfigSchema`` is ``extra="ignore"``, so a key it does not
        declare is discarded while the YAML still shows it. That is exactly how
        ``min_center_fraction`` went missing and made the #534 ladder fix inert.
        """
        from spectramr.core.execution_ledger import ExecutionLedger, SubstitutionClass

        data = copy.deepcopy(self.BASE)
        # `undersampling:`, the canonical block name since phase 11 renamed the
        # root `acceleration:`. See the sibling test below for why writing the
        # LEGACY name here does not merely rename the recorded path -- it makes
        # the drop vanish from the ledger entirely.
        data["undersampling"] = {"center_fraction": 0.08, "min_centre_fraction": 0.02}

        ledger = ExecutionLedger.begin_run(source="defect")
        TrainingSettings.settings_from_dict(data)

        dropped = [
            s
            for s in ledger.substitutions
            if s.class_id is SubstitutionClass.EXTRA_IGNORE_DROPPED
        ]
        hit = [s for s in dropped if s.path == "undersampling.min_centre_fraction"]
        assert hit, [s.path for s in dropped]
        assert hit[0].requested == 0.02
        assert hit[0].severity == "error"

    def test_a_root_renamed_block_hides_its_discarded_keys_from_the_ledger(self):
        """KNOWN GAP, pinned so it is not rediscovered as a surprise.

        `diff_declared_vs_resolved` descends only into `raw_keys & resolved_keys`.
        A ROOT rename moves the whole block before that walk, so the raw document
        names `acceleration:` while the resolved model carries `undersampling:`
        -- the intersection is empty and every key the block discarded is
        invisible. Same root cause as the `defaults_injected` miscount, one level
        up.

        This matters beyond a test: `check_declared_keys_are_not_discarded` (the
        extra="ignore" census, 2,139 declarations across 430 arms) is blind to
        exactly the block phase 11 renamed, and issue #681 counted 125 phantom
        declarations under `undersampling:` -- which the ledger cannot see for
        any arm still writing the legacy spelling. Flip this assertion when the
        walker learns to follow root renames.
        """
        from spectramr.core.execution_ledger import ExecutionLedger, SubstitutionClass

        data = copy.deepcopy(self.BASE)
        data["acceleration"] = {"center_fraction": 0.08, "min_centre_fraction": 0.02}

        ledger = ExecutionLedger.begin_run(source="root-rename-blindspot")
        TrainingSettings.settings_from_dict(data)

        dropped = [
            s.path
            for s in ledger.substitutions
            if s.class_id is SubstitutionClass.EXTRA_IGNORE_DROPPED
        ]
        assert not [p for p in dropped if p.endswith("min_centre_fraction")], (
            "the walker now follows root renames -- delete this test and tighten "
            "the sibling above"
        )

    def test_injected_defaults_are_counted_not_itemised(self):
        """A full config resolves hundreds of keys at their default.

        Itemising them would bury the handful of real findings, and the resolved
        value is already visible in ``resolved_config.json`` regardless.
        """
        from spectramr.core.execution_ledger import ExecutionLedger

        ledger = ExecutionLedger.begin_run(source="defaults")
        TrainingSettings.settings_from_dict(copy.deepcopy(self.BASE))

        assert ledger.defaults_injected > 100
        assert len(ledger.substitutions) < 20, (
            "per-key default records would drown the findings that matter"
        )

    def test_a_diff_failure_never_breaks_the_config_load(self, monkeypatch):
        """Instrumentation must not be able to stop a run from starting."""
        from spectramr.core import execution_ledger as mod

        ledger = mod.ExecutionLedger.begin_run(source="boom")

        def _explode(*_args, **_kwargs):
            raise RuntimeError("diff blew up")

        monkeypatch.setattr(mod, "diff_declared_vs_resolved", _explode)

        settings = TrainingSettings.settings_from_dict(copy.deepcopy(self.BASE))

        assert settings is not None, "a diagnostic failure took the run offline"
        assert any("diff failed" in n for n in ledger.notes), (
            "a swallowed diff failure must be declared in the artifact"
        )


class TestConfigVersionRetirement:
    """The fold is gone (2026-08-08); the retired spellings are now refused.

    Two classes lived here: `TestConfigVersionFoldAtTheLoader`, which pinned
    that a 6.x file bound `1.0`, and `TestVersionFoldIsRecorded`, which pinned
    the ledger substitution the fold emitted so a migrated arm could be told
    from an unmigrated one. Both described machinery that no longer exists.

    The fold's ledger record is not lost as a concept -- the equivalent for
    retired KEYS is alive and tested in
    `tests/unit/config/schemas/test_renames.py::TestAFoldedKeyLeavesEvidence`.
    It is only the VERSION fold that had nothing left to record.
    """

    @staticmethod
    def _write(tmp_path, version):
        import yaml as _yaml

        cfg = _minimal_config()
        cfg["config_version"] = version
        path = tmp_path / f"cfg_{version.replace('.', '_')}.yaml"
        path.write_text(_yaml.safe_dump(cfg), encoding="utf-8")
        return path

    @pytest.mark.parametrize("version", ["6.0", "6.1"])
    def test_a_retired_version_is_refused_at_the_loader(self, tmp_path, version):
        with pytest.raises(ValueError, match="not supported"):
            TrainingSettings.from_yaml(self._write(tmp_path, version))

    def test_the_canonical_version_still_loads(self, tmp_path):
        """Anti-vacuity: refusal must be about the version, not the fixture."""
        from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION

        settings = TrainingSettings.from_yaml(
            self._write(tmp_path, CANONICAL_CONFIG_VERSION)
        )
        assert settings.run.config_version == CANONICAL_CONFIG_VERSION

    def test_no_version_fold_record_is_emitted_any_more(self, tmp_path):
        """Nothing folds, so nothing may claim to have folded."""
        from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION
        from spectramr.core.execution_ledger import (
            ExecutionLedger,
            SubstitutionClass,
        )

        path = self._write(tmp_path, CANONICAL_CONFIG_VERSION)
        ledger = ExecutionLedger.begin_run(source=str(path))
        TrainingSettings.from_yaml(path)
        assert not [
            s
            for s in ledger.substitutions
            if s.path == "config_version"
            and s.class_id is SubstitutionClass.VALUE_CHANGED_ON_FINALIZE
        ]


def test_training_settings_docstring_does_not_present_legacy_as_current() -> None:
    """The class docstring said "Schema Version: 6.0 / 6.1 (both accepted)".

    Accepted is not current. 6.0/6.1 load ONLY through the fold, which rewrites
    them before anything downstream sees them, and `spectramr audit` reports a
    legacy declaration as an error. A docstring that presents them as the
    schema version is the stale-spelling class this repo keeps hitting: it never
    breaks a config, so nothing forces it to be corrected.
    """
    from spectramr.config.schemas.base import (
        CANONICAL_CONFIG_VERSION,
        LEGACY_CONFIG_VERSIONS,
    )
    from spectramr.config.settings import TrainingSettings

    doc = TrainingSettings.__doc__ or ""
    assert "CANONICAL_CONFIG_VERSION" in doc, (
        "the docstring must point at the constant, not restate a version literal"
    )
    # `LEGACY_CONFIG_VERSIONS` has been EMPTY since the fold was deleted on
    # 2026-08-08, so iterating it alone asserts nothing -- the loop below went
    # vacuous the moment the tiers it guards against became unloadable. The
    # retired literals are named explicitly so the guard survives that.
    for legacy in set(LEGACY_CONFIG_VERSIONS) | set(RETIRED_CONFIG_TIERS):
        assert f"Version**: {legacy}" not in doc
    # And it must not hardcode the canonical number either -- that is the same
    # defect one revision later.
    assert f"Version**: {CANONICAL_CONFIG_VERSION}" not in doc


# --------------------------------------------------------------------------
# The MACHINE-READABLE half of the same self-description
# --------------------------------------------------------------------------
#
# The docstring test above was written, and the prose fixed -- while
# `model_config["json_schema_extra"]` fifteen lines below the docstring still read
# `{"version": "6.0", "title": "... Schema v6.0", "description": "Strict v6.0
# schema ..."}`. So the class whose docstring records 6.0 as *deleted* exported a
# JSON schema announcing itself as the one version it now REFUSES to load. Nothing
# breaks: `json_schema_extra` is metadata, never validated against, which is
# exactly why the prose half got fixed and this half did not.


class TestSchemaSelfDescription:
    """The exported schema must name the tier it actually accepts."""

    @staticmethod
    def _extra() -> dict:
        return dict(TrainingSettings.model_config.get("json_schema_extra") or {})

    def test_the_exported_version_is_canonical(self) -> None:
        assert self._extra()["version"] == CANONICAL_CONFIG_VERSION

    def test_no_retired_tier_survives_anywhere_in_the_self_description(self) -> None:
        """Title and description too, not just the `version` key.

        All three strings said 6.0. Fixing only the machine-read one would leave a
        title that still tells a human reader the wrong thing.
        """
        blob = " ".join(str(value) for value in self._extra().values())
        for retired in set(LEGACY_CONFIG_VERSIONS) | set(RETIRED_CONFIG_TIERS):
            assert f"v{retired}" not in blob, (
                f"the schema still describes itself as v{retired}, a tier it "
                f"refuses to load: {blob!r}"
            )

    def test_the_self_description_tracks_the_constant_not_a_literal(self) -> None:
        """The seam, matching `test_defaults_provider`'s equivalent case.

        Asserting the value alone passes on a hard-coded "1.0" that goes stale at
        the next bump — which is how this drifted from 6.0 in the first place. The
        title and description must be *derived*, so they move with the constant.
        """
        extra = self._extra()
        assert CANONICAL_CONFIG_VERSION in extra["title"]
        assert CANONICAL_CONFIG_VERSION in extra["description"]
        assert str(sorted(ACCEPTED_CONFIG_VERSIONS)) in extra["description"], (
            "the accepted set must be enumerated from the constant, so a reader "
            "of the exported schema learns what actually loads"
        )

    def test_the_version_reaches_the_generated_json_schema(self) -> None:
        """`json_schema_extra` is only useful if it survives export — that is the
        surface a downstream consumer (editor, validator, docs build) reads."""
        schema = TrainingSettings.model_json_schema()
        assert schema["version"] == CANONICAL_CONFIG_VERSION
        assert schema["title"] == f"Training Configuration Schema v{CANONICAL_CONFIG_VERSION}"


class TestOverrideProvenancePrivateAttr:
    """``TrainingSettings._override_paths`` is a PRIVATE attr, and that choice
    carries the guarantees below. Behavioural coverage of the thing that writes
    it lives in ``tests/unit/config/test_overrides.py``; here we pin the
    declaration itself, because every property that makes it safe is a property
    of *not being a field*.
    """

    def test_it_defaults_to_empty_on_a_freshly_loaded_config(self):
        """A config nobody overrode must not look overridden. The launch banner
        reads this to decide whether to tell the operator their ``-O`` took
        effect."""
        assert TrainingSettings(**_minimal_config())._override_paths == ()

    def test_it_is_not_a_field(self):
        """As a field it would be in ``model_dump``, in the published JSON
        Schema, in the provenance snapshot -- and, with ``extra="forbid"``, a
        spellable YAML key an arm could assert about itself. It is a record of
        how the config was assembled, not configuration."""
        settings = TrainingSettings(**_minimal_config())
        assert "_override_paths" not in type(settings).model_fields
        assert "_override_paths" not in settings.model_dump()
        assert "_override_paths" not in settings.model_json_schema()["properties"]

    def test_frozen_still_rejects_field_assignment(self):
        """The whole design rests on a Pydantic v2 asymmetry: ``frozen=True``
        guards FIELD assignment but not private-attr assignment. If a future
        Pydantic tightened private attrs the writer would break loudly (covered
        next); if it ever LOOSENED fields, CLAUDE.md #1 would break silently --
        so pin the half that must not move."""
        settings = TrainingSettings(**_minimal_config())
        with pytest.raises(ValidationError):
            settings.training = None

    def test_the_private_attr_is_assignable_on_a_frozen_instance(self):
        """The other half of the asymmetry, pinned so a Pydantic upgrade that
        removes it fails here -- next to the explanation -- instead of at the
        first override-bearing run."""
        settings = TrainingSettings(**_minimal_config())
        settings._override_paths = ("training.max_iterations",)
        assert settings._override_paths == ("training.max_iterations",)


class TestValidationEnsembleNeedsAStochasticSampler:
    """The DECLARED half of the ensemble guard: ``validation.sampling.ensemble_samples > 1``
    against ``model.model_kwargs.sampler_sigma`` (the strategy re-checks the resolved
    generator). Planted refusals first."""

    @staticmethod
    def _cfg(sigma=None, n: int = 2) -> dict:
        cfg = _minimal_config()
        cfg["validation"] = {"sampling": {"enable_multistep_cold": True, "ensemble_samples": n}}
        if sigma is not None:
            cfg["model"]["model_kwargs"] = {"sampler_sigma": sigma}
        return cfg

    def test_sigma_zero_is_refused(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="sampler_sigma"):
            TrainingSettings(**self._cfg(sigma=0.0))

    def test_an_absent_sigma_is_refused(self) -> None:
        """Absent resolves to the deterministic default, so it is the same facade."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="sampler_sigma"):
            TrainingSettings(**self._cfg())

    def test_a_non_numeric_sigma_is_refused(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="not a number"):
            TrainingSettings(**self._cfg(sigma="loud"))

    def test_a_positive_sigma_loads(self) -> None:
        settings = TrainingSettings(**self._cfg(sigma=0.05))
        assert settings.validation.sampling.ensemble_samples == 2

    def test_a_single_sample_validation_never_reads_sigma(self) -> None:
        settings = TrainingSettings(**self._cfg(n=1))
        assert settings.validation.sampling.ensemble_samples == 1


class TestRecorruptedInferenceNeedsAnR2RTarget:
    """`validation.scoring.recorrupted_inference` is refused at load where nothing recorrupts."""

    @staticmethod
    def _settings(target_mode: str, **knob):
        cfg = _minimal_config()
        cfg["data"].update({"target_mode": target_mode, "coils": {"processing_mode": "none"}})
        if target_mode in ("r2r", "nullspace_r2r"):
            cfg["data"]["val_target_mode"] = "phase_aligned_mean"
        cfg["validation"] = {"scoring": {"recorrupted_inference": {"samples": 3, **knob}}}
        return TrainingSettings(**cfg)

    def test_r2r_and_nullspace_r2r_load(self) -> None:
        for mode in ("r2r", "nullspace_r2r"):
            assert self._settings(mode).validation.scoring.recorrupted_inference.samples == 3

    def test_an_averaging_target_is_refused_at_load(self) -> None:
        with pytest.raises(ValueError, match="recorrupted_inference is read only under"):
            self._settings("phase_aligned_mean")

    def test_the_other_modes_strength_is_refused(self) -> None:
        with pytest.raises(ValueError, match="angle_deg is the strength of the other"):
            self._settings("r2r", angle_deg=30.0)
        with pytest.raises(ValueError, match="alpha is the strength of the other"):
            self._settings("nullspace_r2r", alpha=0.5)


class TestTALLLocalityNeedsATALLModel:
    """``training.tall_sr.lambda_locality > 0`` is refused at load on any model but ``tall_sr``."""

    @staticmethod
    def _cfg(model_type: str, weight: float) -> dict:
        cfg = _minimal_config()
        cfg["model"] = {"model_type": model_type}
        cfg["training"] = {"tall_sr": {"lambda_locality": weight}}
        return cfg

    def test_positive_weight_on_a_non_tall_model_is_refused(self) -> None:
        with pytest.raises(ValidationError, match=r"requires model\.model_type 'tall_sr'"):
            TrainingSettings(**self._cfg("unet", 0.01))

    def test_zero_weight_on_a_non_tall_model_loads(self) -> None:
        settings = TrainingSettings(**self._cfg("unet", 0.0))
        assert settings.training.tall_sr.lambda_locality == 0.0

    def test_positive_weight_on_tall_sr_loads(self) -> None:
        settings = TrainingSettings(**self._cfg("tall_sr", 0.01))
        assert settings.training.tall_sr.lambda_locality == 0.01


class TestDomainGRLNeedsCycleGAN:
    """``losses.gan.domain_grl.enabled`` is refused at load on any strategy but ``cyclegan``."""

    @staticmethod
    def _cfg(strategy: str, enabled: bool) -> dict:
        cfg = _minimal_config()
        cfg["training"] = {"strategy_class": strategy}
        cfg["losses"] = {"gan": {"domain_grl": {"enabled": enabled}}}
        return cfg

    def test_enabled_on_cyclegan_loads(self) -> None:
        settings = TrainingSettings(**self._cfg("cyclegan", True))
        assert settings.losses.gan.domain_grl.enabled is True

    def test_enabled_on_another_strategy_is_refused(self) -> None:
        with pytest.raises(ValidationError, match=r"requires training\.strategy_class 'cyclegan'"):
            TrainingSettings(**self._cfg("cut", True))

    def test_disabled_on_another_strategy_loads(self) -> None:
        TrainingSettings(**self._cfg("cut", False))


class TestLevyDenoisingNeedsItsStrategy:
    """``training.levy_denoising`` is refused at load on a strategy that never reads it."""

    @staticmethod
    def _cfg(**training: object) -> dict:
        cfg = _minimal_config()
        cfg["training"] = {"levy_denoising": {"alpha": 1.7}, **training}
        return cfg

    def test_on_the_reconstruction_strategy_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="read only by LevyDenoisingStrategy"):
            TrainingSettings(**self._cfg(strategy_class="reconstruction"))

    def test_on_the_levy_mode_loads(self) -> None:
        settings = TrainingSettings(**self._cfg(strategy_class="levy_diffusion"))
        assert settings.training.levy_denoising.alpha == 1.7

    def test_on_the_dotted_strategy_path_loads(self) -> None:
        path = "spectramr.infrastructure.training.strategies.levy_denoising_strategy.LevyDenoisingStrategy"
        assert TrainingSettings(**self._cfg(strategy_class=path)).training.levy_denoising


class TestForwardNoiseExcludesSamplerSigma:
    """``model.model_kwargs.forward_noise_sigma`` and ``sampler_sigma`` are one owner of reverse noise."""

    @staticmethod
    def _cfg(**model_kwargs) -> dict:
        cfg = _minimal_config()
        cfg["model"]["model_kwargs"] = dict(model_kwargs)
        return cfg

    def test_forward_noise_alone_loads(self) -> None:
        settings = TrainingSettings(**self._cfg(forward_noise_sigma=0.2))
        assert settings.model.model_kwargs["forward_noise_sigma"] == 0.2

    def test_forward_noise_beside_sampler_sigma_is_refused(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="declare one"):
            TrainingSettings(**self._cfg(forward_noise_sigma=0.2, sampler_sigma=0.1))
