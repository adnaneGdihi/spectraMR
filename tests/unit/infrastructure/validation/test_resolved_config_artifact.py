"""The resolved-config artifact SSOT, shared by audit and train.

The point of one producer is that the pre-flight and the run cannot describe the
same config differently. Two surfaces hand-rolling the same JSON is how the two
disjoint validation stacks happened, so these tests assert the shapes are
identical, not merely similar.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from spectramr.core.execution_ledger import ExecutionLedger, SubstitutionClass
from spectramr.infrastructure.validation.resolved_config_artifact import (
    FAILURE_SENTINEL_NAME,
    RESOLVED_CONFIG_NAME,
    build_resolved_config_payload,
    resolved_config_run_name,
    write_ledger_failure_sentinel,
    write_resolved_config,
)

BASE = {
    "model": {"model_type": "unet"},
    "data": {"batch_size": 2},
    "training": {"max_iterations": 10},
    "optimization": {"learning_rate": 1e-4},
    "logging": {},
}


@pytest.fixture(autouse=True)
def _disarm():
    ExecutionLedger.reset()
    yield
    ExecutionLedger.reset()


def _settings(extra=None):
    import copy

    from spectramr.config.settings import TrainingSettings

    data = copy.deepcopy(BASE)
    if extra:
        data.update(extra)
    return TrainingSettings.settings_from_dict(data)


def test_payload_is_the_config_plus_an_additive_ledger_key():
    ExecutionLedger.begin_run(source="t")
    payload = build_resolved_config_payload(_settings(), run_id="r1")

    assert "_ledger" in payload
    assert payload["_ledger"]["run_id"] == "r1"
    # The config's own top-level keys survive untouched — the block is additive,
    # not an annotation tree, because cohort_ablation reads metadata.* as scalars.
    assert payload["model"]["model_type"] == "unet"


def test_the_same_config_yields_the_same_ledger_shape_on_both_surfaces():
    """Audit and train must not be able to disagree about the artifact."""
    ExecutionLedger.begin_run(source="audit")
    audit_payload = build_resolved_config_payload(
        _settings({"acceleration": {"min_centre_fraction": 0.02}}),
        ledger_source="audit",
    )
    ExecutionLedger.begin_run(source="train")
    train_payload = build_resolved_config_payload(
        _settings({"acceleration": {"min_centre_fraction": 0.02}}),
        ledger_source="run_training_pipeline",
    )

    def _shape(p):
        return sorted(p["_ledger"].keys()), [
            (s["class_id"], s["path"]) for s in p["_ledger"]["substitutions"]
        ]

    assert _shape(audit_payload) == _shape(train_payload)


def test_the_audit_surface_records_the_drop_before_any_gpu_time():
    """The whole point: the pre-flight now leaves the evidence behind."""
    ExecutionLedger.begin_run(source="audit")
    payload = build_resolved_config_payload(
        _settings({"acceleration": {"center_fraction": 0.08, "min_centre_fraction": 0.02}}),
        ledger_source="audit",
    )
    dropped = [
        s
        for s in payload["_ledger"]["substitutions"]
        if s["class_id"] == SubstitutionClass.EXTRA_IGNORE_DROPPED.value
    ]
    assert [s["path"] for s in dropped] == ["acceleration.min_centre_fraction"]


def test_clean_config_records_no_drop():
    """CONTROL: proves the artifact is not simply always reporting drops."""
    ExecutionLedger.begin_run(source="audit")
    payload = build_resolved_config_payload(_settings(), ledger_source="audit")
    dropped = [
        s
        for s in payload["_ledger"]["substitutions"]
        if s["class_id"] == SubstitutionClass.EXTRA_IGNORE_DROPPED.value
    ]
    assert dropped == []


def test_write_resolved_config_produces_readable_json(tmp_path):
    ExecutionLedger.begin_run(source="t")
    out = write_resolved_config(tmp_path, _settings(), run_id="r1")

    assert out.name == RESOLVED_CONFIG_NAME
    reloaded = json.loads(out.read_text())
    assert reloaded["_ledger"]["write_status"] == "ok"


def test_write_raises_so_the_audit_can_fail_loudly(tmp_path):
    """The audit must not exit 0 when it could not write its own record.

    Train catches this and falls back to the sentinel; the audit does not, because
    an audit that cannot report has failed at its only job.
    """
    ExecutionLedger.begin_run(source="t")
    missing = tmp_path / "nope" / "deeper"
    with pytest.raises(OSError):
        write_resolved_config(missing, _settings())


def test_failure_sentinel_preserves_the_records_collected_so_far(tmp_path):
    """A silently-failed ledger would BE the class this artifact detects."""
    ledger = ExecutionLedger.begin_run(source="t")
    ledger.record(
        class_id=SubstitutionClass.EXTRA_IGNORE_DROPPED,
        site="s",
        stage="config_finalize",
        path="acceleration.min_centre_fraction",
        requested=0.02,
    )
    out = write_ledger_failure_sentinel(tmp_path, RuntimeError("disk on fire"))

    assert out is not None and out.name == FAILURE_SENTINEL_NAME
    payload = json.loads(out.read_text())
    assert payload["write_status"] == "failed"
    assert "disk on fire" in payload["error"]
    assert payload["substitutions"][0]["path"] == "acceleration.min_centre_fraction"


def test_strict_mode_turns_a_write_failure_into_an_abort(tmp_path, monkeypatch):
    """ "A run without a ledger did not happen" — for CI and cluster audits."""
    monkeypatch.setenv("SPECTRAMR_LEDGER_STRICT", "1")
    ExecutionLedger.begin_run(source="t")
    with pytest.raises(RuntimeError, match="SPECTRAMR_LEDGER_STRICT"):
        write_ledger_failure_sentinel(tmp_path, RuntimeError("boom"))


# --------------------------------------------------------------------------
# The artifact must label which "version" is which
# --------------------------------------------------------------------------
#
# `_ledger.schema_version: 2` versions the LEDGER FILE FORMAT; `config_version`
# versions the config SCHEMA TIER; `metadata.version` is the author's revision.
# The audit's `--json` report keeps `_ledger` and discards the resolved config
# around it, so a reader who sees only this block used to have no way to tell the
# first apart from the two others — a `2` beside a `"6.0"` reads as a
# contradiction. Threading the tier in here is what makes the block self-describing.
#
# It is threaded from the payload rather than from the constant on purpose: this
# artifact is the SSOT shared by audit and train, so a tier read from anywhere but
# the config being described could make the two surfaces disagree about the same
# config — the exact failure this module's docstring exists to prevent.


def test_the_ledger_block_labels_what_its_schema_version_versions():
    ExecutionLedger.begin_run(source="t")
    block = build_resolved_config_payload(_settings(), run_id="r1")["_ledger"]

    assert block["schema_version"] == 2, "the ledger file-format version, unchanged"
    assert block["schema_version_of"] == "execution_ledger"
    assert "config_version" in block, "the schema tier must be present and named"


def test_the_declared_schema_tier_reaches_the_ledger_block():
    """Sensitivity pair, half 1: a config that states its tier is reported."""
    ExecutionLedger.begin_run(source="t")
    payload = build_resolved_config_payload(_settings({"config_version": "1.0"}), run_id="r1")
    assert payload["_ledger"]["config_version"] == "1.0"
    # And it is the config's own value, not a constant: same key, same answer.
    assert payload["run"]["config_version"] == "1.0"


def test_an_undeclared_schema_tier_is_none_rather_than_invented():
    """Half 2. A config that omits the key must not acquire one here.

    Substituting a default would be the very class of defect the ledger exists to
    record (non-negotiable 3b), and it would make an unversioned config
    indistinguishable from a canonical one in the one artifact that survives.
    """
    ExecutionLedger.begin_run(source="t")
    payload = build_resolved_config_payload(_settings(), run_id="r1")
    assert payload["_ledger"]["config_version"] is None
    assert payload["run"]["config_version"] is None


def test_the_tier_reader_does_not_guess_at_a_foreign_payload():
    """`_declared_config_version` is fail-soft by design: the artifact writer must
    not crash a run over a payload shape it did not expect."""
    from spectramr.infrastructure.validation.resolved_config_artifact import (
        _declared_config_version,
    )

    assert _declared_config_version({}) is None
    assert _declared_config_version({"anything": 1}) is None
    assert _declared_config_version({"run": "not-a-block"}) is None
    assert _declared_config_version({"run": {}}) is None
    # Stringified, so the JSON type is stable no matter how it was authored.
    assert _declared_config_version({"run": {"config_version": 1}}) == "1"


def test_both_surfaces_still_agree_once_the_tier_is_threaded():
    """The additive keys must not reintroduce the divergence this module prevents."""
    declared = {"config_version": "1.0"}
    ExecutionLedger.begin_run(source="audit")
    audit = build_resolved_config_payload(_settings(declared), ledger_source="audit")
    ExecutionLedger.begin_run(source="train")
    train = build_resolved_config_payload(
        _settings(declared), ledger_source="run_training_pipeline"
    )

    for key in ("schema_version", "schema_version_of", "config_version"):
        assert audit["_ledger"][key] == train["_ledger"][key]
    assert sorted(audit["_ledger"]) == sorted(train["_ledger"])


def test_run_id_also_lands_a_run_qualified_copy(tmp_path):
    """The canonical name is overwrite-on-launch, so the run's own record needs one too.

    #1299 gave ``provenance.json`` and the debug snapshots a run-id copy and left
    the config artifact on the old contract (#1379).
    """
    ExecutionLedger.begin_run(source="t")
    out = write_resolved_config(tmp_path, _settings(), run_id="arm-20260821_231431-c76ad2f38f58")

    copy = tmp_path / resolved_config_run_name("arm-20260821_231431-c76ad2f38f58")
    assert out.name == RESOLVED_CONFIG_NAME, "the canonical path is still what we return"
    assert copy.exists(), "a known run_id must leave a record the next launch cannot overwrite"
    assert copy.read_bytes() == out.read_bytes(), (
        "the copy must be byte-identical -- a run-id file that disagrees with the "
        "canonical one is worse than no copy at all"
    )


def test_no_run_id_writes_only_the_canonical_file(tmp_path):
    """The audit path passes no ``run_id``; it must not start emitting siblings."""
    ExecutionLedger.begin_run(source="audit")
    write_resolved_config(tmp_path, _settings(), ledger_source="audit")

    written = sorted(p.name for p in tmp_path.glob("*.json"))
    assert written == [RESOLVED_CONFIG_NAME]


def test_relaunching_into_one_directory_keeps_every_runs_config(tmp_path):
    """The violation shape the copy exists for, planted end to end.

    Two launches into one run directory is not hypothetical -- it is how
    ``experiment_11_attention_none`` ended up pairing a 40-iteration smoke config
    with a 4000-step run's ``validation_metrics.csv``. The canonical file is
    *expected* to be the newest launch; the point is that the older run stops
    being unrecoverable.
    """
    ExecutionLedger.begin_run(source="t")
    write_resolved_config(
        tmp_path, _settings({"training": {"max_iterations": 4000}}), run_id="run-B"
    )
    ExecutionLedger.reset()
    ExecutionLedger.begin_run(source="t")
    write_resolved_config(tmp_path, _settings({"training": {"max_iterations": 40}}), run_id="run-C")

    canonical = json.loads((tmp_path / RESOLVED_CONFIG_NAME).read_text())
    run_b = json.loads((tmp_path / resolved_config_run_name("run-B")).read_text())
    run_c = json.loads((tmp_path / resolved_config_run_name("run-C")).read_text())

    assert canonical["training"]["max_iterations"] == 40, "canonical is last-writer-wins"
    assert run_b["training"]["max_iterations"] == 4000, "the earlier run survives its relaunch"
    assert run_c["training"]["max_iterations"] == 40
    assert run_b["_ledger"]["run_id"] == "run-B"
    assert run_c["_ledger"]["run_id"] == "run-C"


# ---- the declared block, and rebuilding settings from the artifact (2026-09-03, #1379) ----


def _reference_settings():
    import spectramr.config.schemas as _schemas
    from spectramr.config.settings import TrainingSettings

    return TrainingSettings.from_yaml(
        str(pathlib.Path(_schemas.__file__).parent / "templates" / "v1.0_reference.yaml")
    )


def test_payload_carries_a_declared_block_that_revalidates_to_the_resolved_settings():
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.validation.resolved_config_artifact import (
        DECLARED_KEY,
        LR_SCHEDULE_KEY,
        build_resolved_config_payload,
    )

    settings = _reference_settings()
    payload = build_resolved_config_payload(settings, run_id="t")
    declared = payload[DECLARED_KEY]
    assert declared and set(declared) < set(payload)
    rebuilt = TrainingSettings.model_validate(declared)
    additive = ("_ledger", DECLARED_KEY, LR_SCHEDULE_KEY)
    resolved = {k: v for k, v in payload.items() if k not in additive}
    assert rebuilt.model_dump(mode="json") == resolved


def test_settings_from_resolved_config_rebuilds_the_run(tmp_path):
    from spectramr.infrastructure.validation.resolved_config_artifact import (
        settings_from_resolved_config,
        write_resolved_config,
    )

    settings = _reference_settings()
    path = write_resolved_config(tmp_path, settings, run_id="t")
    rebuilt = settings_from_resolved_config(path)
    assert rebuilt.model_dump(mode="json") == settings.model_dump(mode="json")


def test_an_artifact_without_the_declared_block_is_refused_with_guidance(tmp_path):
    from spectramr.infrastructure.validation.resolved_config_artifact import (
        settings_from_resolved_config,
    )

    old = tmp_path / "resolved_config.json"
    old.write_text(json.dumps({"model": {"model_type": "unet"}, "_ledger": {}}))
    with pytest.raises(ValueError, match="--from-yaml"):
        settings_from_resolved_config(old)


def test_resolved_config_beside_looks_in_the_checkpoint_dir_then_its_parent(tmp_path):
    from spectramr.infrastructure.validation.resolved_config_artifact import resolved_config_beside

    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    ckpt = run / "checkpoints" / "best.pt"
    ckpt.write_bytes(b"")
    assert resolved_config_beside(ckpt) is None
    (run / "resolved_config.json").write_text("{}")
    assert resolved_config_beside(ckpt) == run / "resolved_config.json"
    (run / "checkpoints" / "resolved_config.json").write_text("{}")
    assert resolved_config_beside(ckpt) == run / "checkpoints" / "resolved_config.json"
    assert resolved_config_beside(None) is None


def test_has_declared_block_reads_the_artifact_without_rebuilding(tmp_path):
    from spectramr.infrastructure.validation.resolved_config_artifact import (
        has_declared_block,
        write_resolved_config,
    )

    old = tmp_path / "resolved_config.json"
    old.write_text(json.dumps({"model": {}, "_ledger": {}}))
    assert has_declared_block(old) is False
    assert has_declared_block(tmp_path / "missing.json") is False
    (tmp_path / "run").mkdir()
    assert (
        has_declared_block(
            write_resolved_config(tmp_path / "run", _reference_settings(), run_id="t")
        )
        is True
    )


class TestTheDumpStatesTheBuiltSchedule:
    """#2760: a defaulted ``lr_scheduler_strategy`` dumped as ``"cosine"`` on runs
    that built no scheduler, so the artifact read as an annealed run."""

    def test_an_optimizer_only_arm_dumps_no_scheduler(self):
        from spectramr.infrastructure.validation.resolved_config_artifact import LR_SCHEDULE_KEY

        settings = _settings()
        assert settings.optimization.lr_scheduler_strategy == "cosine", "precondition"
        payload = build_resolved_config_payload(settings)
        assert payload["optimization"]["lr_scheduler_strategy"] is None
        assert payload[LR_SCHEDULE_KEY] == {"scheduler": None}
        assert "lr_scheduler_strategy" not in payload["_declared"]["optimization"]

    def test_a_declared_strategy_is_left_as_declared(self):
        from spectramr.infrastructure.validation.resolved_config_artifact import LR_SCHEDULE_KEY

        extra = {"optimization": {"learning_rate": 1e-4, "lr_scheduler_strategy": "cosine"}}
        payload = build_resolved_config_payload(_settings(extra))
        assert payload["optimization"]["lr_scheduler_strategy"] == "cosine"
        assert payload[LR_SCHEDULE_KEY]["scheduler"] == "cosine_annealing"

    def test_a_scheduler_declared_elsewhere_replaces_the_defaulted_family(self):
        extra = {
            "optimization": {
                "learning_rate": 1e-4,
                "scheduler_type": "step",
                "scheduler": {"step_size": 5},
            }
        }
        payload = build_resolved_config_payload(_settings(extra))
        assert payload["optimization"]["lr_scheduler_strategy"] == "step_lr"

    def test_an_unresolvable_scheduler_is_recorded_not_raised(self, monkeypatch):
        """The audit builds this payload for every arm, so the failure belongs to
        the check that owns it, not to the ledger block."""
        import spectramr.infrastructure.training.scheduler_resolution as _sr
        from spectramr.domain.exceptions import ConfigurationError
        from spectramr.infrastructure.validation.resolved_config_artifact import LR_SCHEDULE_KEY

        def boom(_config):
            raise ConfigurationError("planted")

        monkeypatch.setattr(_sr, "resolved_schedule_record", boom)
        payload = build_resolved_config_payload(_settings())
        assert payload[LR_SCHEDULE_KEY] == {"unresolved": "planted"}
        assert payload["optimization"]["lr_scheduler_strategy"] == "cosine"


# ----------------------------------------------------------------------
# A run's snapshot outlives the removal of a key it declared (#2842)
# ----------------------------------------------------------------------
#
# `early_stopping.check_interval` was removed on 2026-10-09 (#1555, #2805). 16 of
# the 24 n2n/self_supervised arms declared it until then, so every run trained
# from them carries it in its `_declared` block, and predict refused to rebuild
# those runs. Nothing read the key, so dropping it rebuilds the run that trained.


def _snapshot(tmp_path, declared):
    from spectramr.infrastructure.validation.resolved_config_artifact import DECLARED_KEY

    path = tmp_path / "resolved_config.json"
    path.write_text(json.dumps({DECLARED_KEY: declared, "_ledger": {}}))
    return path


def _declared_with(**blocks):
    import copy

    data = copy.deepcopy(BASE)
    for block, values in blocks.items():
        data.setdefault(block, {}).update(values)
    return data


def test_a_snapshot_with_a_retired_key_rebuilds_with_one_warning(tmp_path, caplog):
    import logging

    from spectramr.infrastructure.validation.resolved_config_artifact import (
        settings_from_resolved_config,
    )

    declared = _declared_with(early_stopping={"enabled": True, "check_interval": 5120})
    with caplog.at_level(logging.WARNING):
        rebuilt = settings_from_resolved_config(_snapshot(tmp_path, declared))

    assert rebuilt.early_stopping.enabled is True
    warned = [r.getMessage() for r in caplog.records if "removed since the run" in r.getMessage()]
    assert len(warned) == 1, warned
    assert "`early_stopping.check_interval` (removed 2026-10-09" in warned[0]


def test_the_same_key_in_an_authored_yaml_is_still_refused(tmp_path):
    import yaml

    from spectramr.config.settings import TrainingSettings

    path = tmp_path / "arm.yaml"
    authored = _declared_with(early_stopping={"enabled": True, "check_interval": 5120})
    path.write_text(yaml.safe_dump({"config_version": "1.0", **authored}))
    with pytest.raises(ValueError, match=r"early_stopping\.check_interval"):
        TrainingSettings.from_yaml(str(path))


@pytest.mark.parametrize(
    ("declared", "named"),
    [
        (_declared_with(no_such_block={"x": 1}), "no_such_block"),
        (_declared_with(training={"seed": 3}), "seed"),
    ],
    ids=["unknown-root-key", "moved-key"],
)
def test_a_key_that_was_not_removed_still_fails_the_snapshot(tmp_path, declared, named):
    """Only `removals.py` keys are dropped: an unknown key, or one that moved to a
    new path and whose value would matter there, still refuses the rebuild."""
    from spectramr.infrastructure.validation.resolved_config_artifact import (
        settings_from_resolved_config,
    )

    with pytest.raises(ValueError, match=named):
        settings_from_resolved_config(_snapshot(tmp_path, declared))
