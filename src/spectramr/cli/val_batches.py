"""The one place that knows which config key caps a run's validation pass.

``--val-batches`` is offered by ``train``, ``sanity_check`` and ``profile``, and
each used to spell the target key itself. That is the shape non-negotiable 17
exists to stop: a rename of the key fixes whichever verb the author had open and
leaves the others lowering onto a path nothing reads, which reads at the terminal
exactly like a cap that worked.

The key is read twice on purpose, and both readers matter to what the flag is
worth. :mod:`spectramr.infrastructure.builders.directors.data_pipeline_director`
strides the validation dataset down to the capped budget at BUILD time, so the
volumes outside it are never decoded; ``pipelines/train.py`` caps the loop as
well, which is what covers a dataset with no ``__len__``. Capping therefore buys
real wallclock rather than merely a shorter progress bar.
"""

from __future__ import annotations

import argparse

#: The config path that caps validation. ``num_batches`` and NOT the sibling
#: ``num_samples``: both readers consult ``num_samples`` only when
#: ``num_batches`` is None, so a flag lowered onto ``num_samples`` is silently
#: inert on any arm whose YAML already declares ``num_batches``.
VAL_BATCHES_OVERRIDE_KEY = "validation.loader.num_batches"

#: Help text shared by every verb that offers the flag, so the three cannot
#: drift into describing the same key differently.
VAL_BATCHES_HELP = (
    "Cap validation at N batches for this run (--val-batches 2 applies "
    f"-O {VAL_BATCHES_OVERRIDE_KEY}=2). Use it for smoke runs and one-off "
    "commands, where walking the whole validation split dominates the "
    "wallclock. Volume-backed arms load one volume per batch, so N is also the "
    "number of volumes graded. The kept samples are spread across the split "
    "rather than taken from the front, so N=1 is not always the first one. "
    "Omit for a full validation pass, which is what a real run wants."
)


def val_batches_override(n_batches: int) -> str:
    """Render the ``KEY=VALUE`` override that caps validation at ``n_batches``.

    Raises:
        ValueError: If ``n_batches`` is below 1. Zero reads as "skip validation",
            which this key cannot express — both readers treat a sub-1 budget as
            "no cap" and grade the whole split, so accepting it would advertise
            the opposite of what happens (pitfall 9).
    """
    if n_batches < 1:
        raise ValueError(
            f"--val-batches must be >= 1, got {n_batches}. It caps the "
            "validation pass; it cannot switch validation off, and a value "
            "below 1 is read downstream as 'no cap' (the full split)."
        )
    return f"{VAL_BATCHES_OVERRIDE_KEY}={n_batches}"


def add_val_batches_argument(parser: argparse.ArgumentParser) -> None:
    """Register ``--val-batches`` on a subparser whose verb runs validation."""
    parser.add_argument(
        "--val-batches",
        dest="val_batches",
        type=int,
        default=None,
        metavar="N",
        help=VAL_BATCHES_HELP,
    )


def resolve_val_batches_overrides(
    overrides: list[str] | None, n_batches: int | None
) -> list[str] | None:
    """Return ``overrides`` with the ``--val-batches`` cap appended.

    ``overrides`` is returned unchanged when the flag is absent, so a run that
    does not pass it is byte-identical to one from before the flag existed.

    Raises:
        ValueError: If ``overrides`` already sets the same key. Appending would
            win on precedence and silently discard the operator's own ``-O``,
            and the two are just as likely to have been meant the other way
            round; naming the conflict is the only reading that cannot be wrong.
    """
    if n_batches is None:
        return overrides

    existing = list(overrides or [])
    # Compare CANONICAL paths: the legacy spelling `validation.num_validation_batches`
    # still folds onto this key, so a textual match would miss a real conflict.
    from spectramr.config.schemas.renames import canonical_override_path

    for entry in existing:
        key = entry.split("=", 1)[0].strip()
        try:
            canonical = canonical_override_path(key)
        except Exception:
            continue
        if canonical == VAL_BATCHES_OVERRIDE_KEY:
            raise ValueError(
                f"--val-batches {n_batches} conflicts with the override "
                f"{entry!r}, which sets the same key "
                f"({VAL_BATCHES_OVERRIDE_KEY}). Pass one or the other."
            )

    return [*existing, val_batches_override(n_batches)]


__all__ = [
    "VAL_BATCHES_HELP",
    "VAL_BATCHES_OVERRIDE_KEY",
    "add_val_batches_argument",
    "resolve_val_batches_overrides",
    "val_batches_override",
]
