r"""Audit that every :mod:`renames` record is served by a mounted validator.

:data:`~spectramr.config.schemas.renames.RENAMES` is the SSOT for retired config
spellings, but a record only does anything if some schema class mounts the
validator that serves it. Nothing connected the two halves, and the gap is
**silent in both directions**:

* A block owning records with **no mount** loses the record's guided message.
  ``reporting`` was in this state from 2026-08-05 until this module landed: its
  one record named the replacement, the reason, the issue (#503) and the
  one-line fixer, and a user declaring ``per_case_metrics`` saw pydantic's
  generic ``Extra inputs are not permitted`` instead. Where the block is
  ``extra="ignore"`` rather than ``extra="forbid"`` the same gap is worse — the
  key is dropped without a word (pitfall #15).
* A **mount owning no records** is a permanent no-op. Both builders return
  early on an empty table (``if not retired`` / ``if not foldable``), so a
  mount whose block argument is misspelled is indistinguishable from a correct
  one until the day a record is added and does not fire.

:meth:`RenameRecord.__post_init__` already refuses a record whose ``mount``
does not own its keys. This is the same rule from the other side: a mount that
owns no records, and a record that no mount owns, both become findings.

Runtime introspection, not source text. The mounts are read out of pydantic's
own ``__pydantic_decorators__`` registry, so the audit sees exactly what
pydantic sees — a mount added under a different attribute name, or inherited
from a base class (``training/base.py`` mounts one that seven diffusion
``*Params`` subclasses inherit), is still counted. A ``grep`` for the call site
sees 16 mounts where the model tree has 23 class-level appearances, and neither
number is the one that matters: the audit keys on ``(block, posture)``.

Why the ``raise`` half is a promotion precondition
--------------------------------------------------

57 of ~163 mounted schema classes are ``extra="ignore"``, and :mod:`renames`
opens with what that costs: "simply deleting a field is *worse* than leaving
the ambiguity in place -- the key stops working **and** stops being visible."
Promoting a drained ``fold`` record to ``raise`` in a block that cannot refuse
it converts a working fold into exactly that silent drop. When the first 31
records were promoted, **28** landed in blocks (``data``, ``logging``,
``losses``, ``validation``) mounting ``fold_renamed_keys`` and no
``reject_renamed_keys``: the retired key was accepted and discarded, and
``exclude_defaults`` came back ``[]`` from a config that declared ``["mse"]``.
So a record may only be promoted once its block's ``raise`` mount exists, and
this audit is what makes that checkable instead of remembered.
"""

from __future__ import annotations

import typing
from collections.abc import Iterator
from dataclasses import dataclass

from pydantic import BaseModel

from .removals import REMOVALS, RemovalRecord, removals_for_block
from .renames import RENAMES, Posture, RenameRecord, renames_for_block

__all__ = [
    "FORWARD_PROVISIONED",
    "MOUNT_FOR_POSTURE",
    "MountFinding",
    "audit_mounts",
    "discover_mounts",
    "schema_classes",
]

#: Attributes :func:`~spectramr.config.schemas.renames.reject_renamed_keys` and
#: :func:`~spectramr.config.schemas.renames.fold_renamed_keys` stamp on the
#: validator they build.
#:
#: The stamp is load-bearing, not decoration: ``reject_renamed_keys`` closes
#: over its RESOLVED record dict and nothing else, so the block it serves is
#: unrecoverable at runtime without it. Reading the block out of a closure
#: freevar by name would work for ``fold`` and silently return ``None`` for
#: every ``reject`` — the shape this module exists to catch.
BLOCK_ATTR = "__rename_block__"
POSTURE_ATTR = "__rename_posture__"

#: Which builder serves which posture. Named here so a finding can tell the
#: reader what to mount rather than only that something is missing.
MOUNT_FOR_POSTURE: dict[Posture, str] = {
    "raise": "reject_renamed_keys",
    "fold": "fold_renamed_keys",
}

#: ``(block, posture)`` mounts that serve zero records **on purpose**, with the
#: reason. An entry here is a claim someone has to defend at review time; the
#: default is that an inert mount is a defect.
FORWARD_PROVISIONED: dict[tuple[str, str], str] = {
    ("run", "raise"): (
        "`run` is a block the phase-4b split CREATED (d06df4d47, 2026-07-31), so "
        "no key has been retired out of it yet and RENAMES holds no record whose "
        "mount_path is `run`. Arming the mount ahead of the first retirement is "
        "the cheap side of the trade: an unmounted retirement is a silent drop, "
        "an unused mount costs one early return. Note the records that MOVED "
        "keys INTO `run` (`seed`, `device`) mount on ROOT and `training`, where "
        "the legacy keys lived -- `mount_path` is the legacy side, which is why "
        "this mount reads as covering them and does not."
    ),
}


@dataclass(frozen=True)
class MountFinding:
    """One way the mounts and the table disagree."""

    #: ``missing_mount`` (records with no validator) or ``inert_mount``
    #: (validator with no records).
    kind: str
    block: str
    posture: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] block={self.block!r} posture={self.posture!r}: {self.detail}"


def _models_in(annotation: object) -> Iterator[type[BaseModel]]:
    """Every ``BaseModel`` reachable through a (possibly generic) annotation.

    Descends ``X | None``, ``list[X]``, ``dict[str, X]`` and any nesting of
    them. Stops at the model rather than recursing into its fields — walking
    fields is :func:`schema_classes`' job, and doing it here would recurse
    forever on a self-referential schema.
    """
    stack: list[object] = [annotation]
    while stack:
        current = stack.pop()
        if isinstance(current, type) and issubclass(current, BaseModel):
            yield current
            continue
        stack.extend(typing.get_args(current))


def schema_classes(root: type[BaseModel]) -> set[type[BaseModel]]:
    """Every schema class reachable from ``root`` by following model fields.

    This is the surface a mount can actually protect: a validator on a class no
    config document ever builds guards nothing, so an unreachable mount is
    correctly invisible here and its block reads as unmounted.
    """
    seen: set[type[BaseModel]] = set()
    stack = [root]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        for field in cls.model_fields.values():
            stack.extend(_models_in(field.annotation))
    return seen


def discover_mounts(root: type[BaseModel]) -> dict[tuple[str, str], frozenset[str]]:
    """Map ``(block, posture)`` to the names of the classes mounting it.

    Reads pydantic's decorator registry rather than the class ``__dict__``, so
    the attribute a mount is bound to is irrelevant and an inherited mount is
    found on every subclass that inherits it.
    """
    found: dict[tuple[str, str], set[str]] = {}
    for cls in schema_classes(root):
        for decorator in cls.__pydantic_decorators__.model_validators.values():
            func = decorator.func
            block = getattr(func, BLOCK_ATTR, None)
            posture = getattr(func, POSTURE_ATTR, None)
            if block is None or posture is None:
                continue
            found.setdefault((block, posture), set()).add(cls.__name__)
    return {key: frozenset(value) for key, value in found.items()}


def audit_mounts(
    root: type[BaseModel],
    table: dict[str, RenameRecord] | None = None,
    allowed_inert: dict[tuple[str, str], str] | None = None,
    removals: dict[str, RemovalRecord] | None = None,
) -> list[MountFinding]:
    """Compare the mounts reachable from ``root`` against ``table``.

    Returns every disagreement, sorted, so a caller reports all of them at once
    instead of one per run.

    ``allowed_inert`` defaults to :data:`FORWARD_PROVISIONED`; pass ``{}`` to
    audit with no exemptions at all.
    """
    records = RENAMES if table is None else table
    # An injected rename table audits that table alone unless removals are injected too.
    removed = (REMOVALS if table is None else {}) if removals is None else removals
    exempt = FORWARD_PROVISIONED if allowed_inert is None else allowed_inert
    mounted = discover_mounts(root)
    # A removed key is refused by the `raise` mount of its block, like a raise rename.
    owed = {(record.mount_path, record.posture) for record in records.values()}
    owed |= {(record.mount_path, "raise") for record in removed.values()}

    findings: list[MountFinding] = []
    for block, posture in sorted(owed - set(mounted)):
        served = {
            **renames_for_block(block, records, posture=posture),
            **(removals_for_block(block, removed) if posture == "raise" else {}),
        }
        findings.append(
            MountFinding(
                kind="missing_mount",
                block=block,
                posture=posture,
                detail=(
                    f"{len(served)} `{posture}` record(s) name this block, but no "
                    f"schema reachable from {root.__name__} mounts "
                    f"{MOUNT_FOR_POSTURE[posture]}({block!r}). "
                    # `renames_for_block` keys by LEAF name, and a leaf is not
                    # findable: `old` appears in no file, `widget.old` is the
                    # RENAMES key. Report the path.
                    f"Affected: {sorted(r.legacy for r in served.values())}. "
                    "Mount it, or the records are dead text."
                ),
            )
        )

    for block, posture in sorted(set(mounted) - owed):
        if (block, posture) in exempt:
            continue
        findings.append(
            MountFinding(
                kind="inert_mount",
                block=block,
                posture=posture,
                detail=(
                    f"{sorted(mounted[(block, posture)])} mount "
                    f"{MOUNT_FOR_POSTURE[posture]}({block!r}), but no record has "
                    f"mount_path={block!r} and posture={posture!r}, so the "
                    "validator returns early on every document and will keep "
                    "doing so if the block name is a typo. Fix the name, delete "
                    "the mount, or declare it in FORWARD_PROVISIONED with the "
                    "reason it is armed early."
                ),
            )
        )
    return findings
