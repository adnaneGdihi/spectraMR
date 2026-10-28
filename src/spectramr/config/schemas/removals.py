"""Config keys removed outright: nothing replaces them.

:mod:`renames` owns keys that MOVED: every record names a canonical path, and
the fixer can carry the value there. A removed key is a different fact. Its
value never reached any code, so carrying it anywhere would change what a run
does. ``training.num_workers: 8`` moved to ``data.loader.num_workers`` would
give a run that ran on one worker eight. The only correct migration is deletion.

Most records are keys nothing ever read. A few (``was_read=True``) are knobs that
WERE read and are retired because the behaviour they switched no longer exists;
their refusal says so rather than claim the key was inert. Deletion is still the
only migration, because no code reads the value any more.

Each record names, where one exists, the key that actually governs the behaviour
the removed one advertised (``live``), so the refusal tells the author where the
setting they wanted lives without moving their value there.

The refusal is served by the same mounts as a ``raise`` rename:
:func:`renames.reject_renamed_keys` includes this table, so a block that mounts
it refuses both kinds, and :func:`rename_mounts.audit_mounts` counts these
records when it checks that every retired key has a mount. Most blocks are
``extra="ignore"`` or ``"allow"``, where a deleted field is silently dropped or
kept as an untyped extra; the refusal is what makes the deletion loud.

Every record here was drained from every loadable config before it landed
(#568, #563, #557, #306, #698, #1135, #1555).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

#: Mount path for a key at the document root, matching :data:`renames.ROOT`.
ROOT = ""


@dataclass(frozen=True, slots=True)
class RemovalRecord:
    """One removed config key."""

    #: Dotted path of the removed key.
    legacy: str
    #: ISO date of the removal.
    since: str
    #: One line: what the key claimed, and the evidence that nothing read it.
    reason: str
    #: Dotted path of the key that governs that behaviour, if there is one.
    live: str | None = None
    #: The schema whose validator serves this record; ``None`` means the top-level
    #: block of ``legacy``. Set it for a nested block, as for :class:`RenameRecord`.
    mount: str | None = None
    #: True for a knob that code read until the behaviour it switched was removed.
    was_read: bool = False
    #: True when ``live`` takes the same value, so the refusal says to move it.
    carry_value: bool = False

    def __post_init__(self) -> None:
        if self.mount is not None and not self.legacy.startswith(f"{self.mount}."):
            raise ValueError(f"mount={self.mount!r} does not own legacy={self.legacy!r}")

    #: Removed keys are always refused; the posture of a ``raise`` rename.
    posture = "raise"

    @property
    def mount_path(self) -> str:
        """The schema this record's validator is mounted on."""
        if self.mount is not None:
            return self.mount
        return self.legacy.split(".", 1)[0] if "." in self.legacy else ROOT

    @property
    def legacy_key(self) -> str:
        """Leaf name of the removed key, as it appears in YAML."""
        return self.legacy.rsplit(".", 1)[-1]

    def message(self) -> str:
        """The error a user sees: what was removed, why, and where the live setting is."""
        if self.was_read:
            head = (
                f"`{self.legacy}` was retired ({self.since}): it was read, and its "
                f"behaviour is gone. {self.reason}"
            )
        else:
            head = f"`{self.legacy}` was removed ({self.since}): nothing read it. {self.reason}"
        if self.live is not None:
            head += f" The setting that takes effect is `{self.live}`."
        if self.carry_value:
            return head + f" Move the value to `{self.live}`."
        return head + " Delete the key; moving its value would change what the run does."


_SINCE = "2026-10-04"

_RECORDS = [
    RemovalRecord(
        "validation.use_training_loss",
        _SINCE,
        "Validation never chose its loss by it (#568).",
    ),
    RemovalRecord(
        "validation.visualization_dir",
        _SINCE,
        "No visualization was ever written there (#563).",
    ),
    RemovalRecord(
        "validation.num_visualizations",
        _SINCE,
        "No image logger read the count (#563).",
    ),
    RemovalRecord(
        "validation.enable_validation_augmentation",
        "2026-10-25",
        "The validation transform chain builds no augmentation step, so the flag "
        "switched nothing; all 136 arms that set it said false.",
    ),
    RemovalRecord(
        "checkpoint.save_on_epoch",
        _SINCE,
        "No epoch-boundary checkpoint was ever written (#568).",
        live="checkpoint.save_interval",
    ),
    RemovalRecord(
        "checkpoint.keep_best_n",
        "2026-10-09",
        "The best checkpoint is one file, `best.<format>`, replaced on "
        "each improvement, so there was never a set of best files to bound (#1972).",
    ),
    RemovalRecord(
        "metrics.track_best_metric",
        _SINCE,
        "Best-model tracking never consulted it (#568).",
    ),
    RemovalRecord(
        "training.diffusion.degradation_dynamics",
        _SINCE,
        "No degradation schedule consulted it (#568).",
        mount="training.diffusion",
    ),
    RemovalRecord(
        "physics.data_consistency.acs_mask_center_fraction",
        _SINCE,
        "No DC path masks a fixed ACS fraction (#563).",
        live="undersampling.center_fraction",
        mount="physics.data_consistency",
    ),
    RemovalRecord(
        "physics.data_consistency.enable_acs_replacement",
        _SINCE,
        "No DC path replaces ACS lines (#563).",
        mount="physics.data_consistency",
    ),
    RemovalRecord(
        "physics.coil_sensitivity.estimation_method",
        _SINCE,
        "The coil-map estimator never read it (#306).",
        live="physics.coil_processing.estimation.method",
        mount="physics.coil_sensitivity",
    ),
    RemovalRecord(
        "training.num_workers",
        _SINCE,
        "No DataLoader read it (#557).",
        live="data.loader.num_workers",
    ),
    RemovalRecord(
        "training.enable_gradient_checkpointing",
        _SINCE,
        "model_factory stripped it before any generator saw it (#1135).",
        live="optimization.gradient.enable_checkpointing",
    ),
    RemovalRecord(
        "artifacts",
        _SINCE,
        "`artifacts.persistent_root` never set an output path (#698).",
        live="training.output_dir",
    ),
    RemovalRecord(
        "early_stopping.check_interval",
        "2026-10-09",
        "Only `EarlyStoppingService.should_check` used it, and nothing called "
        "that; early stopping updates on every validation pass, so `patience` "
        "counts passes (#1555).",
        live="validation.schedule.interval_steps",
    ),
    RemovalRecord(
        "checkpoint.checkpoint_dir",
        "2026-10-10",
        "No checkpoint was written there: training saves to "
        "<training.output_dir>/checkpoints, and resume and run_summary look there; "
        "its one effect was an empty ./checkpoints under the launch directory (#2842).",
        live="training.output_dir",
    ),
    RemovalRecord(
        "checkpoint.save_dir",
        "2026-10-10",
        "It was the validation alias of `checkpoint.checkpoint_dir`, removed with it (#2842).",
        live="training.output_dir",
    ),
    RemovalRecord(
        "parallel.deepspeed.save_consolidated_best",
        "2026-10-10",
        "The best checkpoint now carries the gathered weights itself (a ZeRO-3 capture "
        "consolidates them on rank 0), so there is no separate consolidated file to "
        "switch off.",
        mount="parallel.deepspeed",
        was_read=True,
    ),
    RemovalRecord(
        "checkpoint.pretrained_path",
        "2026-10-10",
        "No reader loaded it; a warm-start is model.checkpoint_path.",
        live="model.checkpoint_path",
    ),
    RemovalRecord(
        "checkpoint.resume_training",
        "2026-10-10",
        "No reader consulted it; resuming is the --resume flag (a path, auto or if-present).",
    ),
    RemovalRecord(
        "checkpoint.resume_from",
        "2026-10-10",
        "The schema dropped it, so the distillation teacher it named never loaded.",
        live="training.distillation.teacher_checkpoint",
        carry_value=True,
    ),
    RemovalRecord(
        "training.tissue_prior_checkpoint",
        "2026-10-10",
        "Existence-checked, never loaded: no code read the weights (#647).",
    ),
    RemovalRecord(
        "training.corruption_calibration_checkpoint",
        "2026-10-10",
        "Existence-checked, never loaded: no code read the weights (#647).",
    ),
    RemovalRecord(
        "training.bloch_manifold_dps.pretrained_score_checkpoint",
        "2026-10-10",
        "Existence-checked, never loaded: no code read the weights (#647).",
        mount="training.bloch_manifold_dps",
    ),
    RemovalRecord(
        "training.equivariance_conformal.pretrained_reconstructor_checkpoint",
        "2026-10-10",
        "Existence-checked, never loaded: no code read the weights (#647).",
        mount="training.equivariance_conformal",
    ),
    RemovalRecord(
        "training.phys_residual_conformal.pretrained_reconstructor_checkpoint",
        "2026-10-10",
        "Existence-checked, never loaded: no code read the weights (#647).",
        mount="training.phys_residual_conformal",
    ),
]


def _index(records: list[RemovalRecord]) -> dict[str, RemovalRecord]:
    out: dict[str, RemovalRecord] = {}
    for rec in records:
        if rec.legacy in out:
            raise ValueError(f"duplicate removal record for {rec.legacy!r}")
        out[rec.legacy] = rec
    return out


#: Dotted legacy path -> record.
REMOVALS: dict[str, RemovalRecord] = _index(_RECORDS)


def removals_for_block(
    block: str, table: dict[str, RemovalRecord] | None = None
) -> dict[str, RemovalRecord]:
    """Leaf name -> record, for every removal whose validator is mounted on ``block``."""
    src = REMOVALS if table is None else table
    return {rec.legacy_key: rec for rec in src.values() if rec.mount_path == block}


def strip_removed_keys(
    doc: dict[str, Any], table: dict[str, RemovalRecord] | None = None
) -> tuple[dict[str, Any], list[RemovalRecord]]:
    """A copy of ``doc`` without any removed key, and the records of the keys it dropped.

    For a run's own snapshot only, never for an authored config: a snapshot written
    before a removal still carries the key, and since no current code reads it (a
    ``was_read`` knob's behaviour is gone too), dropping it rebuilds what this
    version of the code would run. An authored config keeps the refusal, which is what
    tells its author to delete the key.
    """
    src = REMOVALS if table is None else table
    out = copy.deepcopy(doc)
    dropped: list[RemovalRecord] = []
    for rec in src.values():
        *parents, leaf = rec.legacy.split(".")
        node: Any = out
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict) and leaf in node:
            del node[leaf]
            dropped.append(rec)
    return out, dropped


__all__ = ["REMOVALS", "RemovalRecord", "removals_for_block", "strip_removed_keys"]
