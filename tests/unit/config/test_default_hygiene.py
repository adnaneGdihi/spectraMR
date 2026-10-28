"""Default hygiene: a gated gate.

The rule this file enforces:

    In a schema class whose own gate is ``enabled: bool = False``, no
    ``enable_*`` / ``use_*`` sub-field may default to ``True``.

Why it matters is a *presentation* problem with a behavioural tail. While the
parent is off the sub-flag does nothing, but the resolved-config artifact stamps
every default (deliberately -- pitfall #15c), so the run record reads "motion
corruption: on" for a twin that never ran. And the moment somebody sets
``enabled: true`` they silently buy every sub-flag nobody asked for.

**The gate is the deliverable, not the sweep.** Measuring the five current
violations against the corpus found that fixing them is not a hygiene edit at
all -- see ``KNOWN_TRUE_UNDER_DISABLED_PARENT``, where every entry carries the
number of arms that would change behaviour. A convention nobody can violate
accidentally outlives a one-time rename; this test is that convention.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import ClassVar

import pytest
from pydantic import BaseModel


def _schema_classes() -> dict[str, type[BaseModel]]:
    """Every Pydantic model reachable from ``spectramr.config.schemas``.

    Walked live rather than grepped: an inventory built by grep cannot prove
    completeness, which is the lesson from the 2,536-key schema audit (#539).
    """
    import spectramr.config.schemas as schemas

    found: dict[str, type[BaseModel]] = {}
    for mod_info in pkgutil.walk_packages(schemas.__path__, schemas.__name__ + "."):
        try:
            mod = importlib.import_module(mod_info.name)
        except Exception:  # pragma: no cover - optional extras
            continue
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if issubclass(cls, BaseModel) and cls is not BaseModel:
                found[cls.__name__] = cls
    return found


def _violations() -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for name, cls in _schema_classes().items():
        gate = cls.model_fields.get("enabled")
        if gate is None or gate.default is not False:
            continue
        for field, info in cls.model_fields.items():
            if field == "enabled":
                continue
            if (
                field.startswith("enable_") or field.startswith("use_")
            ) and info.default is True:
                out.add((name, field))
    return out


class TestNoTrueSubFlagUnderADisabledParent:
    #: Sub-flags that violate the rule and are NOT yet resolved. Measured
    #: 2026-08-03 over the loadable corpus, counting only arms where the PARENT
    #: is explicitly ``enabled: true``.
    #:
    #: This set may shrink, never grow. It was five; three were flipped and one
    #: reclassified on 2026-08-03, and the last, the unread
    #: `DataConsistencyConfig.enable_acs_replacement`, is refused at load since #563.
    KNOWN_TRUE_UNDER_DISABLED_PARENT: ClassVar[frozenset[tuple[str, str]]] = frozenset()

    #: Sub-flags that trip the rule and SHOULD keep defaulting True. Separated
    #: from the set above because the two mean opposite things: that one is
    #: "not fixed yet", this one is "fixing it would be the bug". Kept as data
    #: with its own anti-rot test rather than special-cased inside `_violations`,
    #: so the exemption stays visible and reviewable.
    DELIBERATE_TRUE: ClassVar[frozenset[tuple[str, str]]] = frozenset(
        {
            # `use_orig_params=True` is the CORRECT FSDP setting -- it is what
            # makes optimizer param groups and torch.compile work under
            # sharding. Flipping it is FREE today (the single FSDP arm sets it
            # explicitly, so 0 arms rely on the default) and WRONG tomorrow:
            # every future FSDP arm that omits it would silently get the
            # setting PyTorch itself recommends against. The rule's premise --
            # "enabling the parent buys the sub-feature unasked" -- is exactly
            # what you want here.
            ("FSDPConfigSchema", "use_orig_params"),
        }
    )

    def test_no_new_violations(self) -> None:
        new = (
            _violations()
            - self.KNOWN_TRUE_UNDER_DISABLED_PARENT
            - self.DELIBERATE_TRUE
        )
        assert not new, (
            f"new enable_*/use_* sub-field(s) defaulting True under a parent "
            f"whose own `enabled` defaults False: {sorted(new)}. The resolved "
            "config will report the feature as on for every arm that leaves the "
            "parent off, and enabling the parent buys it unasked. Default it to "
            "False, or add it here with the corpus measurement that justifies it."
        )

    @pytest.mark.parametrize(
        "entry", sorted(KNOWN_TRUE_UNDER_DISABLED_PARENT | DELIBERATE_TRUE)
    )
    def test_the_known_entries_still_exist(self, entry: tuple[str, str]) -> None:
        """Anti-rot. A stale exception is an exception that stops guarding, and
        the ratchet would never notice it had gone slack."""
        assert entry in _violations(), (
            f"{entry[0]}.{entry[1]} no longer violates the rule -- remove it "
            "from KNOWN_TRUE_UNDER_DISABLED_PARENT / DELIBERATE_TRUE so the "
            "sets keep shrinking"
        )

    def test_the_rule_finds_something(self) -> None:
        """Anti-vacuity: if the walker silently imported nothing, every
        assertion above would pass over an empty set."""
        assert len(_schema_classes()) > 200
        assert _violations(), "the detector found zero violations -- verify it works"

    def test_the_two_sets_are_disjoint(self) -> None:
        """"Not fixed yet" and "fixing it would be the bug" are opposite claims.

        An entry in both would let a genuine violation hide behind an exemption.
        """
        overlap = self.KNOWN_TRUE_UNDER_DISABLED_PARENT & self.DELIBERATE_TRUE
        assert not overlap, f"entry claimed as both unresolved and deliberate: {overlap}"

    def test_the_deliberate_set_is_not_a_dumping_ground(self) -> None:
        """Anti-rot for the escape hatch itself.

        `DELIBERATE_TRUE` exists so a false positive of the rule does not sit in
        a "to fix" list forever. That is only safe while it stays tiny — the
        moment it is easier to add an entry here than to fix the default, the
        ratchet has been inverted.
        """
        assert len(self.DELIBERATE_TRUE) <= 2, (
            "DELIBERATE_TRUE has grown; each entry must be a case where "
            "defaulting True is CORRECT when the parent is on, not merely "
            "inconvenient to change"
        )

    def test_the_digital_twin_flags_were_actually_flipped(self) -> None:
        """Pins the 2026-08-03 flip so it cannot regress silently.

        These three defaulted True under a parent defaulting False. They were
        flipped, and the 9 arms relying on the old default now declare the value
        explicitly — so every twin-enabled arm resolves exactly as before. A
        revert would restore the hygiene violation AND silently re-enable
        corruption for any arm that omits the flag.
        """
        from spectramr.config.schemas.physics import DigitalTwinConfig

        for field in ("enable_motion", "enable_b0", "enable_b1"):
            assert DigitalTwinConfig.model_fields[field].default is False, (
                f"DigitalTwinConfig.{field} defaults True again; enabling the "
                "twin would buy this corruption unasked"
            )
