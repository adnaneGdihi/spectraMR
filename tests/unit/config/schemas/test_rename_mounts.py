"""The mount audit, planted against before it is trusted (non-negotiable 15).

Every assertion about the REAL tree here is worth exactly as much as the
detector behind it, and a detector that has only ever been run on a green tree
has never been shown to go red. So each shape the audit claims to catch is
planted as a synthetic schema tree plus a synthetic record table, and asserted
to produce the finding. The real-tree tests come last, after the detector has
earned them.

The plants are call-site plants: they run the same :func:`audit_mounts` the
real-tree tests run, on inputs it accepts by parameter. Nothing is stubbed.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, model_validator

from spectramr.config.schemas.rename_mounts import (
    FORWARD_PROVISIONED,
    MountFinding,
    audit_mounts,
    discover_mounts,
)
from spectramr.config.schemas.renames import (
    RenameRecord,
    fold_renamed_keys,
    reject_renamed_keys,
)
from spectramr.config.settings import TrainingSettings


def _record(legacy: str, canonical: str, posture: str = "raise") -> RenameRecord:
    return RenameRecord(
        legacy=legacy,
        canonical=canonical,
        since="2026-08-22",
        reason="Planted by the mount audit's own tests.",
        posture=posture,  # type: ignore[arg-type]
    )


def _table(*records: RenameRecord) -> dict[str, RenameRecord]:
    return {record.legacy: record for record in records}


def _kinds(findings: list[MountFinding]) -> set[tuple[str, str, str]]:
    return {(f.kind, f.block, f.posture) for f in findings}


class TestPlantedViolationsTurnTheAuditRed:
    """One plant per shape the audit claims to catch."""

    def test_a_block_with_records_and_no_mount_is_found(self) -> None:
        """The `reporting` shape: a record whose message can never be shown.

        This is the defect that motivated the module. Before the mount landed,
        `ReportingSettings` looked exactly like `Widget` here -- a real block,
        a real record, and nothing joining them.
        """

        class Widget(BaseModel):
            model_config = ConfigDict(extra="forbid")
            keep: bool = True

        class Root(BaseModel):
            widget: Widget = Widget()

        table = _table(_record("widget.old", "widget.keep"))
        findings = audit_mounts(Root, table, allowed_inert={})

        assert _kinds(findings) == {("missing_mount", "widget", "raise")}
        assert "widget.old" in findings[0].detail, "name the orphaned record"
        assert "reject_renamed_keys" in findings[0].detail, "name what to mount"

    def test_a_removal_with_no_mount_is_found(self) -> None:
        """A removed key on an unmounted `extra="ignore"` block would vanish silently."""
        from spectramr.config.schemas.removals import RemovalRecord

        class Widget(BaseModel):
            model_config = ConfigDict(extra="ignore")
            keep: bool = True

        class Root(BaseModel):
            widget: Widget = Widget()

        removals = {"widget.old": RemovalRecord("widget.old", "2026-10-04", "Gone.")}
        findings = audit_mounts(Root, {}, allowed_inert={}, removals=removals)

        assert _kinds(findings) == {("missing_mount", "widget", "raise")}
        assert "widget.old" in findings[0].detail

    def test_the_production_removals_are_all_mounted(self) -> None:
        from spectramr.config.settings import TrainingSettings

        assert not [f for f in audit_mounts(TrainingSettings) if f.kind == "missing_mount"]

    def test_a_mount_of_the_wrong_posture_does_not_cover_the_other(self) -> None:
        """A reject mount is not a fold mount.

        The dangerous direction: a block that already mounts `reject` looks
        covered to a reader, so its first `fold` record can be added and
        silently never fire. Posture is part of the key precisely so this is
        not a judgement call.
        """
        table = _table(
            _record("widget.retired", "widget.keep"),
            _record("widget.staged", "widget.keep", posture="fold"),
        )

        class Widget(BaseModel):
            model_config = ConfigDict(extra="forbid")
            keep: bool = True
            _reject = model_validator(mode="before")(
                classmethod(reject_renamed_keys("widget", table))
            )

        class Root(BaseModel):
            widget: Widget = Widget()

        findings = audit_mounts(Root, table, allowed_inert={})
        assert _kinds(findings) == {("missing_mount", "widget", "fold")}
        assert "fold_renamed_keys" in findings[0].detail

    def test_a_misspelled_block_name_is_found_as_an_inert_mount(self) -> None:
        """The failure the early return hides.

        `reject_renamed_keys('wdiget')` resolves to an empty table and its
        validator returns on the first line of every document. Nothing warns,
        no test fails, and the block it was meant to protect is bare.
        """
        table = _table(_record("widget.old", "widget.keep"))

        class Widget(BaseModel):
            model_config = ConfigDict(extra="forbid")
            keep: bool = True
            _reject = model_validator(mode="before")(
                classmethod(reject_renamed_keys("wdiget", table))
            )

        class Root(BaseModel):
            widget: Widget = Widget()

        findings = audit_mounts(Root, table, allowed_inert={})
        assert _kinds(findings) == {
            ("missing_mount", "widget", "raise"),
            ("inert_mount", "wdiget", "raise"),
        }

    def test_the_forward_provisioned_allowlist_is_load_bearing(self) -> None:
        """An inert mount is red by default and green only when declared.

        Asserted in BOTH directions: an allowlist that silenced everything, or
        one that silenced nothing, would each pass a one-sided test.
        """
        table = _table(_record("widget.old", "widget.keep"))

        class Widget(BaseModel):
            model_config = ConfigDict(extra="forbid")
            keep: bool = True
            _reject = model_validator(mode="before")(
                classmethod(reject_renamed_keys("widget", table))
            )
            _fold = model_validator(mode="before")(classmethod(fold_renamed_keys("widget", table)))

        class Root(BaseModel):
            widget: Widget = Widget()

        assert _kinds(audit_mounts(Root, table, allowed_inert={})) == {
            ("inert_mount", "widget", "fold")
        }
        assert audit_mounts(Root, table, allowed_inert={("widget", "fold"): "planted"}) == []

    def test_a_mount_on_an_unreachable_class_does_not_count(self) -> None:
        """A validator no document builds protects nothing.

        `Orphan` mounts the right block, but no field reaches it, so the block
        must still read as unmounted rather than as covered-by-something.
        """
        table = _table(_record("widget.old", "widget.keep"))

        class Orphan(BaseModel):
            _reject = model_validator(mode="before")(
                classmethod(reject_renamed_keys("widget", table))
            )

        class Root(BaseModel):
            keep: bool = True

        assert _kinds(audit_mounts(Root, table, allowed_inert={})) == {
            ("missing_mount", "widget", "raise")
        }


class TestDiscoveryFindsMountsGrepCannot:
    """Why the audit reads pydantic's registry rather than the source text."""

    def test_an_inherited_mount_is_found_on_every_subclass(self) -> None:
        """`training/base.py` mounts one validator seven `*Params` inherit.

        The mount is found on the SUBCLASS, which is the class a document
        actually builds -- the base is not itself a field annotation, so it is
        not reachable and does not appear. A source-text census sees one call
        site and would have to know about inheritance to expand it; reading
        pydantic's registry gets the expansion for free.
        """
        table = _table(_record("widget.old", "widget.keep"))

        class Base(BaseModel):
            _reject = model_validator(mode="before")(
                classmethod(reject_renamed_keys("widget", table))
            )

        class Child(Base):
            keep: bool = True

        class Root(BaseModel):
            widget: Child = Child()

        mounts = discover_mounts(Root)
        assert mounts[("widget", "raise")] == frozenset({"Child"})
        assert audit_mounts(Root, table, allowed_inert={}) == []

    def test_the_attribute_name_a_mount_is_bound_to_is_irrelevant(self) -> None:
        """Nothing requires the name `_reject_renamed`; the audit must not either."""
        table = _table(_record("widget.old", "widget.keep"))

        class Widget(BaseModel):
            keep: bool = True
            _named_anything_at_all = model_validator(mode="before")(
                classmethod(reject_renamed_keys("widget", table))
            )

        class Root(BaseModel):
            widget: Widget = Widget()

        assert audit_mounts(Root, table, allowed_inert={}) == []


class TestTheRealTree:
    """Only meaningful because of the plants above."""

    def test_every_block_in_renames_is_mounted(self) -> None:
        findings = audit_mounts(TrainingSettings)
        assert findings == [], "\n".join(str(f) for f in findings)

    def test_the_audit_is_not_vacuous_on_the_real_tree(self) -> None:
        """A discovery walk that returned nothing would pass the test above.

        `data`/`raise` is the largest mount in the table (36 records) and the
        least likely to be removed; if it stops being found, discovery broke.
        """
        mounts = discover_mounts(TrainingSettings)
        assert ("data", "raise") in mounts
        assert "DataConfigSchema" in mounts[("data", "raise")]
        assert len(mounts) > 10, f"only {len(mounts)} mounts discovered"

    def test_reporting_is_mounted(self) -> None:
        """The fix this module was written to make permanent."""
        assert ("reporting", "raise") in discover_mounts(TrainingSettings)

    @pytest.mark.parametrize("entry", sorted(FORWARD_PROVISIONED))
    def test_each_forward_provisioned_mount_still_exists_and_is_still_inert(
        self, entry: tuple[str, str]
    ) -> None:
        """The allowlist must not outlive what it exempts.

        Two rots: the mount is deleted (the entry is then a claim about nothing)
        or the block gains its first record (the entry is then hiding a mount
        that has started doing work, and the exemption should go).
        """
        from spectramr.config.schemas.renames import RENAMES, renames_for_block

        block, posture = entry
        assert entry in discover_mounts(TrainingSettings), (
            f"{entry} is allowlisted as forward-provisioned but nothing mounts "
            "it any more; drop the FORWARD_PROVISIONED entry."
        )
        assert not renames_for_block(block, RENAMES, posture=posture), (
            f"{entry} now serves records, so it is no longer forward-provisioned; "
            "drop the FORWARD_PROVISIONED entry and let the audit cover it."
        )


class TestProductionSurfaceIsFullyMounted:
    """The audit must be clean against the REAL table, not only synthetic ones.

    Every other test in this file injects a synthetic ``table``, which proves the
    mechanism works but says nothing about whether the shipped records are
    actually mounted. That gap is not hypothetical: the eight
    ``losses.reconstruction.*`` records added for #421 were committed with no
    mount, and every one of them loaded silently -- the retired key was accepted
    and discarded, exactly the ``missing_mount`` shape this module exists to
    name. This test is what makes that unshippable.
    """

    def test_no_findings_against_the_real_table(self):
        findings = audit_mounts(TrainingSettings)
        assert findings == [], "\n".join(str(f) for f in findings)
