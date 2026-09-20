"""``metadata.audit_waivers``: acknowledging a warning an arm's physics makes unavoidable.

``spectramr audit`` is ``--strict`` by default and a warning exits 2, which the
SLURM dispatcher reads as "skip train" (non-negotiable 4). That is the right
posture for a warning somebody can act on. It is the wrong posture for a warning
whose fix does not exist: the 2026-09-16 run never launched
``experiment_11_kspace_cold_diffusion_radial`` or ``_spiral`` because
``schedule.nesting_leakfree`` fires on every non-Cartesian cascade, and the
witness's own fix hint -- ``undersampling.enforce_nested: true`` -- RAISES on
those families (``sampling.py`` refuses a cascade collapsed below
``nested_tolerance``: radial keeps 3.5 % of k-space at t=1 where its own draw
kept 50 %). An unsatisfiable gate is not a gate; it teaches everyone to reach
for ``--allow-warnings``, which is the flag that then hides the warnings that
DID have a fix.

So the arm may acknowledge the finding by name, in writing, in its own YAML::

    metadata:
      audit_waivers:
      - check: schedule.nesting_leakfree
        justification: >
          Radial redraws its spoke set per timestep, so the cascade cannot nest
          (#1573) and enforce_nested raises here. The leak IS this arm's subject.

The key already existed in the corpus and nothing read it -- an advertised knob
with no reader (pitfall #15) is indistinguishable from one that works, and every
arm that set it believed it had taken effect.

What a waiver can and cannot do
-------------------------------
* It clears a **warning** only. A failing ERROR keeps failing and the waiver is
  reported as a misuse: "warnings are not OK" stays true for everything an arm
  could fix.
* It must carry a ``justification``. The acknowledgement IS the text; a bare
  name is the same silence the key had before.
* A waiver that matched nothing -- a typo, a retired check, or a finding that has
  since been fixed -- is itself a WARNING, so a stale acknowledgement surfaces
  the day it stops being true instead of quietly outliving its reason.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from spectramr.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Tier,
    WitnessVerdict,
)

#: Where an arm declares them, and the name the bookkeeping verdicts carry.
WAIVER_YAML_KEY = "metadata.audit_waivers"
WAIVER_WITNESS_NAME = "audit.waivers"
_CATEGORY = "audit_waivers"

__all__ = ["WAIVER_WITNESS_NAME", "WAIVER_YAML_KEY", "apply_audit_waivers", "declared_waivers"]


@dataclass(frozen=True)
class Waiver:
    """One acknowledged finding: the check's name and why it cannot be fixed."""

    check: str
    justification: str


def declared_waivers(raw_config: dict[str, Any]) -> list[Any]:
    """The raw ``metadata.audit_waivers`` list, or ``[]`` when none is declared."""
    metadata = raw_config.get("metadata")
    if not isinstance(metadata, dict):
        return []
    declared = metadata.get("audit_waivers")
    if declared is None:
        return []
    return list(declared) if isinstance(declared, (list, tuple)) else [declared]


def _misuse(message: str, fix_hint: str) -> WitnessVerdict:
    return WitnessVerdict(
        witness_name=WAIVER_WITNESS_NAME,
        passed=False,
        message=message,
        severity=Severity.ERROR,
        category=_CATEGORY,
        stage=Stage.DECLARE,
        tier=Tier.T1,
        yaml_keys=(WAIVER_YAML_KEY,),
        fix_hint=fix_hint,
    )


def _stale(message: str) -> WitnessVerdict:
    return WitnessVerdict(
        witness_name=WAIVER_WITNESS_NAME,
        passed=False,
        message=message,
        severity=Severity.WARNING,
        category=_CATEGORY,
        stage=Stage.DECLARE,
        tier=Tier.T1,
        yaml_keys=(WAIVER_YAML_KEY,),
        fix_hint="Delete the waiver: the finding it acknowledges is no longer reported.",
    )


def _parse(entry: Any) -> tuple[Waiver | None, WitnessVerdict | None]:
    """One declared entry as a :class:`Waiver`, or the verdict that rejects it."""
    if not isinstance(entry, dict):
        return None, _misuse(
            f"a waiver must be a mapping with 'check' and 'justification', got {entry!r}",
            "audit_waivers:\n- check: <witness name>\n  justification: <why>",
        )
    check = str(entry.get("check") or "").strip()
    justification = str(entry.get("justification") or "").strip()
    if not check:
        return None, _misuse(
            f"a waiver names no check: {entry!r}",
            "Name the check exactly as the audit prints it, e.g. schedule.nesting_leakfree.",
        )
    if not justification:
        return None, _misuse(
            f"waiver for {check!r} carries no justification",
            "State why the finding cannot be fixed on this arm; the text IS the waiver.",
        )
    return Waiver(check=check, justification=justification), None


def _matches(verdict_name: str, check: str) -> bool:
    """A waiver may name the verdict, or the bare check behind a ``health:`` prefix."""
    return verdict_name == check or verdict_name == f"health:{check}"


def _waived(verdict: WitnessVerdict, waiver: Waiver) -> WitnessVerdict:
    """The same finding, still printed in full, no longer blocking."""
    justification = " ".join(waiver.justification.split())
    return WitnessVerdict(
        witness_name=verdict.witness_name,
        passed=True,
        message=f"WAIVED ({WAIVER_YAML_KEY}): {verdict.message} — {justification}",
        severity=verdict.severity,
        category=verdict.category,
        class_ids=verdict.class_ids,
        stage=verdict.stage,
        tier=verdict.tier,
        yaml_keys=verdict.yaml_keys,
        fix_hint=verdict.fix_hint,
    )


def apply_audit_waivers(
    verdicts: Sequence[WitnessVerdict], raw_config: dict[str, Any]
) -> list[WitnessVerdict]:
    """Clear the warnings this arm acknowledged, and report every waiver that did not.

    Args:
        verdicts: What the witness ladder returned, in report order.
        raw_config: The resolved config as a dict (``WitnessSubject.raw_config``).

    Returns:
        The verdicts with acknowledged warnings marked ``WAIVED`` and passing,
        followed by one verdict per waiver that was malformed, aimed at an error,
        or matched nothing.
    """
    declared = declared_waivers(raw_config)
    if not declared:
        return list(verdicts)

    waivers: list[Waiver] = []
    bookkeeping: list[WitnessVerdict] = []
    for entry in declared:
        waiver, rejection = _parse(entry)
        if rejection is not None:
            bookkeeping.append(rejection)
        elif waiver is not None:
            waivers.append(waiver)

    unused = {w.check for w in waivers}
    adjusted: list[WitnessVerdict] = []
    for verdict in verdicts:
        waiver = next((w for w in waivers if _matches(verdict.witness_name, w.check)), None)
        if waiver is None or verdict.passed:
            adjusted.append(verdict)
            continue
        if verdict.severity is not Severity.WARNING:
            adjusted.append(verdict)
            bookkeeping.append(
                _misuse(
                    f"waiver for {waiver.check!r} names a {verdict.severity} finding; "
                    "a waiver acknowledges a warning nothing can fix, it does not "
                    "clear an error",
                    "Fix the error, or remove the waiver.",
                )
            )
            unused.discard(waiver.check)
            continue
        adjusted.append(_waived(verdict, waiver))
        unused.discard(waiver.check)

    bookkeeping.extend(
        _stale(
            f"waiver for {check!r} matched no reported finding: either the check name "
            "is wrong, or the finding is fixed and the acknowledgement outlived it"
        )
        for check in sorted(unused)
    )
    return [*adjusted, *bookkeeping]


def summarize(verdicts: Iterable[WitnessVerdict]) -> tuple[int, int]:
    """``(waived, flagged)`` counts, for a caller that wants to say so in one line."""
    waived = sum(1 for v in verdicts if v.passed and v.message.startswith("WAIVED ("))
    flagged = sum(1 for v in verdicts if v.witness_name == WAIVER_WITNESS_NAME and not v.passed)
    return waived, flagged
