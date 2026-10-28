"""The resolved-config artifact: one definition, owned by the audit layer.

``resolved_config.json`` answers "what did this config actually resolve to, and
what did the declaration lose on the way there?". That is an **audit** question,
so the artifact is defined here rather than inside the training pipeline, and both
surfaces call this module:

* ``cli/app.py::_audit_one`` emits it pre-flight, before any GPU time is spent,
* ``pipelines/train.py`` deposits a copy in the run directory, so a bundle stays
  self-describing without the audit having to have been run.

Why the definition moved out of ``train.py``. The audit resolves a config through
exactly the same ``_finalize_from_dict`` path as training, so it *computes* every
substitution the ledger records and then threw them all away, because the audit
had no output file: its report goes to stdout, and under ``--json`` to a pipe. The
one surface whose entire job is to catch problems before a run was the one leaving
no artifact, so a config's drops were only discoverable after committing to the
run. Meanwhile the shape lived in a hand-rolled dict inside the pipeline, where
the audit could not reach it.

Two surfaces hand-rolling the same artifact is how the two disjoint validation
stacks happened (``bootstrap`` runs one validator set, ``_audit_one`` another,
neither aware of the other). One producer cannot drift from itself.

**Train still writes its copy.** Making the audit the *only* writer would break
``infrastructure/reporting/cohort_ablation.py::_read_resolved_config``, which
reads ``<arm_dir>/resolved_config.json`` out of run output directories, and would
leave any un-audited run with no record of its own configuration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "FAILURE_SENTINEL_NAME",
    "LR_SCHEDULE_KEY",
    "RESOLVED_CONFIG_NAME",
    "build_resolved_config_payload",
    "resolved_config_run_name",
    "write_ledger_failure_sentinel",
    "write_resolved_config",
]

RESOLVED_CONFIG_NAME = "resolved_config.json"
FAILURE_SENTINEL_NAME = "_ledger_write_FAILED.json"


#: The additive top-level key holding the run's declared (unset-excluded) config.
DECLARED_KEY = "_declared"

LR_SCHEDULE_KEY = "_lr_schedule"  #: the LR schedule the run actually builds

#: The artifact's file name, as ``write_resolved_config`` writes it.
RESOLVED_CONFIG_FILENAME = "resolved_config.json"


def resolved_config_beside(checkpoint_path: str | Path | None) -> Path | None:
    """The run's ``resolved_config.json`` next to a checkpoint, or ``None``.

    Training writes the artifact into the run directory and the checkpoints
    into that directory or its ``checkpoints/`` child, so the checkpoint's own
    directory is tried first and its parent second.
    """
    if checkpoint_path is None:
        return None
    here = Path(checkpoint_path).expanduser().resolve().parent
    for directory in (here, here.parent):
        candidate = directory / RESOLVED_CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


def has_declared_block(path: str | Path) -> bool:
    """Whether the artifact at ``path`` carries a non-empty ``_declared`` block.

    Artifacts written before 2026-09-03 do not; the inference resolver falls
    back to the training YAML for them instead of refusing the run.
    """
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return False
    declared = payload.get(DECLARED_KEY) if isinstance(payload, dict) else None
    return isinstance(declared, dict) and bool(declared)


def settings_from_resolved_config(path: str | Path) -> Any:
    """Rebuild the run's ``TrainingSettings`` from the artifact's declared block.

    Raises ``ValueError`` when the artifact predates the block: the resolved
    dump it holds does not re-validate (see ``build_resolved_config_payload``),
    so the caller must fall back to the training YAML explicitly.

    This is the one place a run's snapshot becomes settings again, so it alone
    drops keys retired (``removals.py``) after the run was written, with one
    warning naming each. Nothing read such a key, so the rebuilt run is the run
    that trained; an authored YAML still has the key refused. Any other key the
    schema refuses, a moved one included, still raises.
    """
    from spectramr.config.schemas.removals import strip_removed_keys
    from spectramr.config.settings import TrainingSettings

    payload = json.loads(Path(path).read_text())
    declared = payload.get(DECLARED_KEY) if isinstance(payload, dict) else None
    if not isinstance(declared, dict) or not declared:
        raise ValueError(
            f"{path} carries no {DECLARED_KEY!r} block (written before 2026-09-03), so the "
            "run's settings cannot be rebuilt from it; pass the training YAML with "
            "--config and --from-yaml."
        )
    declared, dropped = strip_removed_keys(declared)
    if dropped:
        logger.warning(
            "[config] %s declares %d key(s) removed since the run was written; they are "
            "dropped, and the rebuilt settings are unchanged because nothing read them: %s",
            path,
            len(dropped),
            "; ".join(f"`{rec.legacy}` (removed {rec.since}: {rec.reason})" for rec in dropped),
        )
    return TrainingSettings.model_validate(declared)


def resolved_config_run_name(run_id: str) -> str:
    """The run-id-qualified sibling of :data:`RESOLVED_CONFIG_NAME`.

    A function rather than an f-string at the write site so the spelling has one
    owner: readers that want a specific run's config must be able to build the
    name without re-deriving it, and a second derivation is how two surfaces
    disagree about the same filename.
    """
    return f"resolved_config_run_{run_id}.json"


def build_resolved_config_payload(
    settings: Any,
    *,
    run_id: str | None = None,
    health_report: Any = None,
    ledger_source: str = "resolved_config_artifact",
) -> dict[str, Any]:
    """The canonical payload: the resolved config plus its ``_ledger`` block.

    ``_ledger`` is an additive top-level key, deliberately not a per-key
    annotation tree: ``cohort_ablation.py`` reads ``metadata.tags.*`` as raw
    scalars, and an inline shadow tree would break every existing consumer.

    Also the only place the config health report is persisted.
    ``validate_config_health`` runs on every training run and already classifies
    findings under ``category="silent_fallback"``, but its report was only
    ``log_summary()``-ed and then dropped, so a run's own fallback findings
    vanished the moment it started.
    """
    from spectramr.core.execution_ledger import ExecutionLedger

    payload: dict[str, Any] = (
        settings.model_dump(mode="json") if hasattr(settings, "model_dump") else {}
    )
    _stamp_resolved_schedule(settings, payload)
    ledger = ExecutionLedger.current_or_begin(source=ledger_source)
    if health_report is not None:
        ledger.attach_health_report(health_report)
    # Hand the ledger the config's own schema tier so `_ledger` carries a
    # LABELLED version fact beside its own file-format `schema_version`. The
    # audit's --json report keeps `_ledger` and drops the config around it
    # (cli/app.py::_audit_ledger_block), so without this the tier is absent from
    # that payload entirely and the bare `schema_version: 2` was the only
    # version in sight -- next to a `metadata.version` of "6.0" surfacing inside
    # a substitution, which read as the two contradicting each other.
    payload["_ledger"] = ledger.to_dict(
        run_id=run_id, config_version=_declared_config_version(payload)
    )
    if hasattr(settings, "model_dump"):
        # The run's DECLARED config: every key the YAML (plus any -O override)
        # set, none of the schema defaults the resolved dump above materialises.
        # ``TrainingSettings.model_validate`` of this block re-runs the same
        # validation the run ran and yields the resolved settings again -- the
        # full dump does not round-trip, because cross-field validators read a
        # materialised default as a declaration (``parallel.deepspeed.compile``
        # is the first to refuse). ``predict`` rebuilds from this block (#1379).
        payload[DECLARED_KEY] = settings.model_dump(mode="json", exclude_unset=True)
    return payload


def _stamp_resolved_schedule(settings: Any, payload: dict[str, Any]) -> None:
    """Replace a defaulted ``lr_scheduler_strategy`` (never built from, #662) by the
    scheduler actually built, so the dump cannot report a constant LR as cosine (#2760)."""
    from spectramr.domain.exceptions import ConfigurationError
    from spectramr.infrastructure.training.scheduler_resolution import resolved_schedule_record

    optimization, dumped = getattr(settings, "optimization", None), payload.get("optimization")
    if optimization is None or not isinstance(dumped, dict):
        return
    # The audit builds this payload for every arm: record an unresolvable scheduler, never raise.
    try:
        payload[LR_SCHEDULE_KEY] = record = resolved_schedule_record(settings)
    except ConfigurationError as exc:
        payload[LR_SCHEDULE_KEY] = {"unresolved": str(exc)}
        return
    if "lr_scheduler_strategy" not in getattr(optimization, "model_fields_set", ()):
        dumped["lr_scheduler_strategy"] = record["scheduler"]


def _declared_config_version(payload: dict[str, Any]) -> str | None:
    """Read the config's schema tier off a dumped ``TrainingSettings`` payload.

    ``run.config_version`` is the ONLY authored spelling: ``TrainingSettings``
    moves a root-level ``config_version`` into the ``run`` block before binding
    and raises if the file authored it inside ``run`` itself, so post-dump the
    tier lives at exactly one path. Returns ``None`` rather than guessing when
    the payload is not a settings dump (the in-memory-config test paths), so an
    absent tier reads as "not stated", never as a fabricated one.
    """
    run_block = payload.get("run")
    if isinstance(run_block, dict):
        version = run_block.get("config_version")
        if version is not None:
            return str(version)
    return None


def write_resolved_config(
    directory: str | Path,
    settings: Any,
    *,
    run_id: str | None = None,
    health_report: Any = None,
    ledger_source: str = "resolved_config_artifact",
) -> Path:
    """Write the artifact into ``directory`` and return the canonical path.

    Raises on failure. Callers that must never die from a stamping hiccup (the
    training pipeline) catch it and fall back to
    :func:`write_ledger_failure_sentinel`; callers that should fail loudly (the
    audit) let it propagate.

    When ``run_id`` is known the payload lands a second time under
    :func:`resolved_config_run_name`. The canonical name is overwrite-on-launch,
    so relaunching an arm into the same directory destroys the record of which
    config produced the artifacts already sitting there -- the failure #1299
    fixed for ``provenance.json`` and the debug snapshots, which left the config
    artifact behind. It is the same defect with the same blast radius:
    ``reporting/cohort_ablation.py`` pairs ``resolved_config.json`` with
    ``logs/validation_metrics.csv``, and after a relaunch those two describe
    different runs while reading as one. Measured on
    ``experiment_11_attention_none``, which holds three runs' artifacts and one
    config: a 40-iteration smoke config paired against a 4000-step run's metrics
    (#1379).

    Additive, not a rename: the canonical name keeps working for every existing
    reader and the copy costs a few KB. No rank gating here, unlike the
    ``provenance_run_*`` writer it mirrors -- the training call site is already
    ``_is_rank_zero``-gated and the audit path passes no ``run_id``, so this is a
    no-op by construction on every caller that must not fan out.
    """
    out = Path(directory) / RESOLVED_CONFIG_NAME
    payload = build_resolved_config_payload(
        settings,
        run_id=run_id,
        health_report=health_report,
        ledger_source=ledger_source,
    )
    # Serialized once and written twice: two ``json.dumps`` calls could diverge
    # on a value whose repr is not stable, and a run-id copy that differs from
    # the canonical file would be worse than no copy at all.
    rendered = json.dumps(payload, indent=2, default=str)
    out.write_text(rendered)
    if run_id:
        (Path(directory) / resolved_config_run_name(run_id)).write_text(rendered)
    return out


def write_ledger_failure_sentinel(directory: str | Path, exc: BaseException) -> Path | None:
    """Turn a failed artifact write into a disk artifact, not just a log line.

    Soft-failing the stamp is right, since a diagnostic must never kill GPU work,
    but soft-failing *silently* would reproduce the exact class this artifact
    exists to detect. ``SPECTRAMR_LEDGER_STRICT`` escalates to an abort for CI and
    cluster-audit contexts that want "a run without a ledger did not happen".
    """
    import traceback

    from spectramr.core.execution_ledger import ExecutionLedger

    if ExecutionLedger.strict():
        raise RuntimeError(
            f"resolved-config write failed and SPECTRAMR_LEDGER_STRICT is set: {exc}"
        ) from exc
    try:
        ledger = ExecutionLedger.current()
        out = Path(directory) / FAILURE_SENTINEL_NAME
        out.write_text(
            json.dumps(
                {
                    "write_status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "substitutions": (
                        [s.to_dict() for s in ledger.substitutions] if ledger else []
                    ),
                },
                indent=2,
                default=str,
            )
        )
        return out
    except Exception:  # pragma: no cover - last resort
        logger.error("could not write the resolved-config failure sentinel")
        return None
