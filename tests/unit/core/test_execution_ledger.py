"""Tests for the runtime substitution ledger.

Every case here is a **sensitivity pair**: one config where the knob survives
and one where it is dropped, asserting the ledger stays quiet on the first and
fires on the second. That polarity is the point. Issue #550 shipped a blocking
gate that reported ``defects: none`` for all 47 arms it was written to protect,
and it passed review because its tests only ever asserted the clean case. A
detector tested only against healthy input cannot be distinguished from a
detector that never fires.

This mirrors the ``WIRED``/``FACADE`` control groups in
``tests/unit/config/test_knob_behaviour.py``, for the same stated reason: a
suite where every case is expected to fire cannot tell "the defect is real"
from "the harness is stuck on".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel, Field

from spectramr.core.execution_ledger import (
    NO_VALUE,
    ExecutionLedger,
    SubstitutionClass,
    diff_declared_vs_resolved,
    unconsumed_keys,
)


@pytest.fixture(autouse=True)
def _disarm():
    """Every test starts unarmed and leaves the context clean."""
    ExecutionLedger.reset()
    yield
    ExecutionLedger.reset()


# ----------------------------------------------------------------------
# Fixtures modelling the three strictness regimes that actually exist
# ----------------------------------------------------------------------


class Ignoring(BaseModel):
    """The issue #550 shape: undeclared keys are silently discarded."""

    model_config = {"extra": "ignore"}
    center_fraction: float = 0.08
    min_center_fraction: float | None = None


class Allowing(BaseModel):
    """The 14-paradigm-block shape: undeclared keys survive unvalidated."""

    model_config = {"extra": "allow"}
    name: str = "unset"


class Nested(BaseModel):
    model_config = {"extra": "ignore"}
    acceleration: Ignoring = Field(default_factory=Ignoring)
    model_kwargs: dict = Field(default_factory=dict)


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


def test_recording_is_false_until_armed():
    assert ExecutionLedger.recording() is False
    ExecutionLedger.begin_run(source="test")
    assert ExecutionLedger.recording() is True


def test_begin_run_gives_each_logical_run_a_fresh_ledger():
    """ablation.py and hpo.py drive many variants in one process."""
    first = ExecutionLedger.begin_run(source="variant-a")
    first.record(
        class_id=SubstitutionClass.DROPPED_UNCONSUMED_KWARG,
        site="s",
        stage="model_build",
        path="a",
    )
    second = ExecutionLedger.begin_run(source="variant-b")
    assert second is not first
    assert second.substitutions == [], "variant A's records leaked into variant B"


def test_current_or_begin_flags_incomplete_coverage(caplog):
    """An empty ledger and an unwatched run must not look identical.

    Both channels matter: the log warns the operator, and the note travels into
    the artifact so a later reader of ``_ledger`` knows the absence of records
    is not evidence of absence of substitutions.
    """
    with caplog.at_level("WARNING", logger="spectramr.core.execution_ledger"):
        ledger = ExecutionLedger.current_or_begin(source="programmatic")

    assert ledger.notes, "a late-armed ledger must say its records are incomplete"
    assert any("missing" in n for n in ledger.notes)
    assert "incomplete, not empty" in caplog.text
    assert "missing" in ledger.to_dict()["notes"][0]


def test_record_serialises_eagerly_so_the_seam_is_blamed():
    """An unserialisable payload must not be deferred to flush time."""
    ledger = ExecutionLedger.begin_run()
    sub = ledger.record(
        class_id=SubstitutionClass.DROPPED_UNCONSUMED_KWARG,
        site="s",
        stage="model_build",
        path="p",
        requested=object(),  # not JSON-serialisable
    )
    # Coerced at record time, so the artifact write cannot fail later.
    json.dumps(sub.to_dict())


def test_no_value_is_distinct_from_none_in_the_artifact():
    """`resolved: None` means the run used None; NO_VALUE means it never arrived."""
    ledger = ExecutionLedger.begin_run()
    dropped = ledger.record(
        class_id=SubstitutionClass.EXTRA_IGNORE_DROPPED,
        site="s",
        stage="config_finalize",
        path="dropped",
        resolved=NO_VALUE,
    )
    explicit = ledger.record(
        class_id=SubstitutionClass.VALUE_CHANGED_ON_FINALIZE,
        site="s",
        stage="config_finalize",
        path="nulled",
        resolved=None,
    )
    assert dropped.to_dict()["resolved"] == "__unset__"
    assert explicit.to_dict()["resolved"] is None


# ----------------------------------------------------------------------
# The #550 class: extra="ignore" silently discards a declared knob
# ----------------------------------------------------------------------


def test_declared_key_that_survives_records_no_drop():
    """CONTROL: min_center_fraction is declared on the schema, so it arrives."""
    ledger = ExecutionLedger.begin_run()
    raw = {"center_fraction": 0.08, "min_center_fraction": 0.02}
    diff_declared_vs_resolved(raw, Ignoring(**raw), ledger=ledger)
    assert [
        s
        for s in ledger.substitutions
        if s.class_id is SubstitutionClass.EXTRA_IGNORE_DROPPED
    ] == []


def test_declared_key_the_schema_ignores_is_recorded_as_dropped():
    """DEFECT: the exact #550 mechanism, in miniature.

    The YAML still shows ``min_centre_fraction``; the run never sees it. Before
    this ledger nothing anywhere recorded that.
    """
    ledger = ExecutionLedger.begin_run()
    raw = {"center_fraction": 0.08, "min_centre_fraction": 0.02}  # British spelling
    diff_declared_vs_resolved(raw, Ignoring(**raw), ledger=ledger)

    dropped = [
        s
        for s in ledger.substitutions
        if s.class_id is SubstitutionClass.EXTRA_IGNORE_DROPPED
    ]
    assert len(dropped) == 1
    assert dropped[0].path == "min_centre_fraction"
    assert dropped[0].requested == 0.02
    assert dropped[0].resolved is NO_VALUE
    assert dropped[0].severity == "error"


def test_extra_allow_block_is_recorded_as_untyped_not_dropped():
    """The two are different diagnoses: present-but-unvalidated vs gone."""
    ledger = ExecutionLedger.begin_run()
    raw = {"name": "twin_dps", "undeclared_knob": 7}
    diff_declared_vs_resolved(raw, Allowing(**raw), ledger=ledger)

    kinds = {s.class_id for s in ledger.substitutions}
    assert SubstitutionClass.EXTRA_ALLOW_UNTYPED in kinds
    assert SubstitutionClass.EXTRA_IGNORE_DROPPED not in kinds


def test_nested_block_is_classified_by_its_own_strictness():
    """Strictness is per-block, so the walk must recurse into the field's model."""
    ledger = ExecutionLedger.begin_run()
    raw = {"acceleration": {"center_fraction": 0.08, "bogus_key": 1}}
    diff_declared_vs_resolved(raw, Nested(**raw), ledger=ledger)

    dropped = [
        s
        for s in ledger.substitutions
        if s.class_id is SubstitutionClass.EXTRA_IGNORE_DROPPED
    ]
    assert [s.path for s in dropped] == [
        "acceleration.bogus_key"
    ], "the drop must be reported at its full dotted path, not the block root"


def test_dict_field_subkeys_are_flagged_as_unvalidated():
    """model_kwargs-shaped fields bypass pydantic entirely."""
    ledger = ExecutionLedger.begin_run()
    raw = {"model_kwargs": {"attention_type": "swin"}}
    diff_declared_vs_resolved(raw, Nested(**raw), ledger=ledger)

    assert any(
        s.class_id is SubstitutionClass.RAW_DICT_UNVALIDATED
        and s.path == "model_kwargs"
        for s in ledger.substitutions
    )


def test_defaults_are_counted_not_itemised():
    """2500+ keys sit at their default; itemising them buries the real findings."""
    ledger = ExecutionLedger.begin_run()
    diff_declared_vs_resolved({}, Ignoring(), ledger=ledger)
    assert ledger.substitutions == []
    assert ledger.defaults_injected == 2


def test_value_rewritten_during_finalisation_is_recorded():
    """The coil-processing legacy bridges rewrite declared values in place."""
    ledger = ExecutionLedger.begin_run()
    instance = Ignoring(center_fraction=0.08)
    diff_declared_vs_resolved({"center_fraction": 0.99}, instance, ledger=ledger)
    changed = [
        s
        for s in ledger.substitutions
        if s.class_id is SubstitutionClass.VALUE_CHANGED_ON_FINALIZE
    ]
    assert len(changed) == 1
    assert changed[0].requested == 0.99
    assert changed[0].resolved == 0.08


def test_list_and_tuple_are_not_reported_as_a_change():
    """YAML lists arrive as tuples on some fields; that is not a rewrite."""

    class WithSeq(BaseModel):
        model_config = {"extra": "ignore"}
        rungs: tuple[int, ...] = ()

    ledger = ExecutionLedger.begin_run()
    diff_declared_vs_resolved(
        {"rungs": [2, 4, 8]}, WithSeq(rungs=(2, 4, 8)), ledger=ledger
    )
    assert ledger.substitutions == [], "tuple/list normalisation regressed"


# ----------------------------------------------------------------------
# Construction-seam helper
# ----------------------------------------------------------------------


def test_unconsumed_keys_records_each_dropped_kwarg():
    """DEFECT: the sampler_steps / signature-filter class (#480)."""
    ledger = ExecutionLedger.begin_run()
    unknown = unconsumed_keys(
        {"sampler_steps": 25, "in_channels": 2},
        accepted={"in_channels"},
        site="model_factory:498",
        stage="model_build",
        consumer="SomeUNet.__init__",
    )
    assert unknown == ["sampler_steps"]
    assert len(ledger.substitutions) == 1
    assert ledger.substitutions[0].consumer == "SomeUNet.__init__"
    assert ledger.substitutions[0].requested == 25


def test_unconsumed_keys_is_quiet_when_everything_is_accepted():
    """CONTROL: proves the helper is not simply always firing."""
    ledger = ExecutionLedger.begin_run()
    assert (
        unconsumed_keys(
            {"in_channels": 2},
            accepted={"in_channels"},
            site="s",
            stage="model_build",
            consumer="SomeUNet.__init__",
        )
        == []
    )
    assert ledger.substitutions == []


def test_unconsumed_keys_still_reports_when_unarmed():
    """Instrumentation must never change the caller's control flow."""
    assert ExecutionLedger.recording() is False
    assert unconsumed_keys(
        {"bad": 1},
        accepted=set(),
        site="s",
        stage="model_build",
        consumer="C",
    ) == ["bad"]


# ----------------------------------------------------------------------
# Artifact shape
# ----------------------------------------------------------------------


def test_to_dict_is_json_serialisable_and_declares_write_status():
    ledger = ExecutionLedger.begin_run(source="cfg.yaml")
    ledger.record(
        class_id=SubstitutionClass.EXTRA_IGNORE_DROPPED,
        site="s",
        stage="config_finalize",
        path="acceleration.min_center_fraction",
        requested=0.02,
        severity="error",
    )
    payload = ledger.to_dict(run_id="run-1")
    json.dumps(payload)  # must not raise
    assert payload["write_status"] == "ok"
    assert payload["counts"]["error"] == 1
    assert payload["substitutions"][0]["class_id"] == "extra_ignore_dropped"


def test_attach_health_report_captures_the_otherwise_discarded_findings():
    """validate_config_health runs on every run and its report is thrown away."""

    class FakeResult:
        pass

    class FakeReport:
        passed = False
        errors: ClassVar[list] = [FakeResult()]
        warnings: ClassVar[list] = [FakeResult(), FakeResult()]

        def to_dict(self):
            return {"results": [{"check_name": "silent_fallback_x", "passed": False}]}

    ledger = ExecutionLedger.begin_run()
    ledger.attach_health_report(FakeReport())
    assert ledger.health_report is not None
    assert ledger.health_report["n_errors"] == 1
    assert ledger.health_report["n_warnings"] == 2
    assert ledger.to_dict()["health_report"]["results"][0]["check_name"] == (
        "silent_fallback_x"
    )


# ----------------------------------------------------------------------
# DEFAULT_COINCIDENCE: the invisible case
# ----------------------------------------------------------------------


def test_default_coincidence_is_recorded_when_declared_equals_the_default():
    """DEFECT SHAPE: `eta_min: 1e-6` equalled torch's own default (#533).

    The value in force was correct, so no artifact could distinguish "honoured"
    from "dropped and defaulted" — which is why the oscillating LR looked right in
    every log. Only a record made AT the seam carries that distinction.
    """
    from spectramr.core.execution_ledger import record_default_coincidence

    ledger = ExecutionLedger.begin_run()
    made = record_default_coincidence(
        path="optimization.eta_min",
        declared=1e-6,
        library_default=1e-6,
        site="scheduler_resolution",
        stage="scheduler_build",
        consumer="CosineAnnealingLR",
    )
    assert made is True
    sub = ledger.substitutions[0]
    assert sub.class_id is SubstitutionClass.DEFAULT_COINCIDENCE
    assert sub.library_default == 1e-6
    assert sub.severity == "info", "the value is correct; only the evidence is weak"


def test_no_coincidence_when_the_declared_value_differs():
    """CONTROL: proves it is not simply always recording."""
    from spectramr.core.execution_ledger import record_default_coincidence

    ledger = ExecutionLedger.begin_run()
    assert (
        record_default_coincidence(
            path="optimization.eta_min",
            declared=5e-5,
            library_default=1e-6,
            site="s",
            stage="scheduler_build",
            consumer="C",
        )
        is False
    )
    assert ledger.substitutions == []


def test_every_substitution_class_has_an_emitter():
    """Guards this module against the pitfall it exists to detect.

    Four members were once declared with no emitter and no wiring plan, which told
    a reader the ledger covered ground it did not. This asserts the enum stays
    honest: every class must be constructible by some code path in src/.
    """
    import subprocess

    from spectramr.core import execution_ledger as mod

    # parents[1] is the spectramr package itself: .../src/spectramr/core/x.py -> spectramr
    package = Path(mod.__file__).resolve().parents[1]
    emitted = set()
    for member in SubstitutionClass:
        hits = subprocess.run(
            ["grep", "-rl", f"SubstitutionClass.{member.name}", str(package)],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if hits:
            emitted.add(member.name)
    missing = {m.name for m in SubstitutionClass} - emitted
    assert not missing, f"declared with no emitter anywhere in src/: {sorted(missing)}"


class TestFoldedInputKeysAreNotDrops:
    """A staged rename's legacy spelling is an input alias, not a discarded key.

    A ``fold`` record (see ``config/schemas/renames.py``) leaves the legacy name
    in the raw YAML while the value lands on a sub-block. To the key scan that
    looks identical to the issue #550 mechanism -- present in the document,
    absent from ``model_fields`` -- so without the ``__folded_input_keys__``
    exemption the ledger reports it as ``EXTRA_IGNORE_DROPPED`` at severity
    "error", claiming "the run never sees it" about a value the run does see.

    At 35 fold records and ~826 unmigrated arms that is tens of thousands of
    false alarms, and a ledger nobody believes is worse than no ledger.
    """

    @staticmethod
    def _model():
        """Mirrors the real shape: a block that FOLDS a legacy key into a
        sub-block at parse time and publishes the name it accepts."""
        from pydantic import model_validator

        class Inner(BaseModel):
            model_config = {"extra": "forbid"}
            new_name: int = 0

        class Outer(BaseModel):
            model_config = {"extra": "forbid"}
            inner: Inner = Field(default_factory=Inner)
            __folded_input_keys__ = frozenset({"old_name"})

            @model_validator(mode="before")
            @classmethod
            def _fold(cls, data):
                if isinstance(data, dict) and "old_name" in data:
                    data = dict(data)
                    inner = dict(data.get("inner") or {})
                    inner["new_name"] = data.pop("old_name")
                    data["inner"] = inner
                return data

        return Outer

    def _drops(self, raw: dict, model) -> list[str]:
        from spectramr.core.execution_ledger import (
            ExecutionLedger,
            SubstitutionClass,
            diff_declared_vs_resolved,
        )

        ledger = ExecutionLedger.begin_run(source="test")
        diff_declared_vs_resolved(raw, model.model_validate(raw), ledger=ledger)
        return [
            s.path
            for s in ledger.substitutions
            if s.class_id is SubstitutionClass.EXTRA_IGNORE_DROPPED
        ]

    def test_a_folded_key_is_not_reported_as_dropped(self) -> None:
        model = self._model()
        assert self._drops({"inner": {"new_name": 1}, "old_name": 1}, model) == []

    def test_an_unknown_key_is_still_reported(self) -> None:
        """Both directions: the exemption must not blanket-silence the scan."""
        model = self._model()

        class Loose(model):  # type: ignore[misc,valid-type]
            model_config = {"extra": "ignore"}

        assert self._drops({"typo_name": 1}, Loose) == ["typo_name"]

    def test_a_block_with_no_folds_is_unaffected(self) -> None:
        """Anti-vacuity: the exemption reads an attribute most classes lack, so
        a typo in its name would silently do nothing and every test above would
        still pass. This pins that the default is an empty set, not a skip."""
        from spectramr.core.execution_ledger import diff_declared_vs_resolved

        class Plain(BaseModel):
            model_config = {"extra": "ignore"}
            kept: int = 0

        assert not hasattr(Plain, "__folded_input_keys__")
        assert diff_declared_vs_resolved is not None
        assert self._drops({"kept": 1, "stray": 2}, Plain) == ["stray"]

    def test_the_exemption_does_not_inflate_the_defaults_count(self) -> None:
        """``defaults_injected`` counts FIELDS left at their default. A folded
        legacy name is not a field, so counting it would overstate the number
        by one per record on every arm."""
        from spectramr.core.execution_ledger import (
            ExecutionLedger,
            diff_declared_vs_resolved,
        )

        model = self._model()
        ledger = ExecutionLedger.begin_run(source="test")
        raw = {"old_name": 1}
        diff_declared_vs_resolved(raw, model.model_validate(raw), ledger=ledger)
        # Exactly one real field (`inner`) went undeclared. Not two.
        assert ledger.defaults_injected == 1


class TestDefaultsArePerKeyNotACount:
    """``defaults_injected`` was a bare ``int``; it is now ``len(defaults_paths)``.

    Two reasons, one of which is a defect the count was hiding.

    The stated reason is the plan's: a count cannot answer *which* knobs a run
    takes on trust, and ``diff_declared_vs_resolved`` already walks past every
    one of them.

    The unstated one is that the number is a property of the config's SPELLING,
    not of the run. The walker descends only into ``raw_keys & resolved_keys``,
    so a sub-block the YAML never mentions costs 1 and its own fields are never
    reached; folding a leaf into a new sub-block puts that block in ``raw``, the
    walker descends, and its remaining fields become countable. Measured on one
    real arm across the 2026-08-02 canonical-key drain -- whose resolved
    document is byte-identical either way -- the count moved 563 -> 625.
    """

    @staticmethod
    def _walk(raw: dict, model_cls):
        from spectramr.core.execution_ledger import (
            ExecutionLedger,
            diff_declared_vs_resolved,
        )

        ledger = ExecutionLedger.begin_run(source="test")
        diff_declared_vs_resolved(raw, model_cls.model_validate(raw), ledger=ledger)
        return ledger

    @staticmethod
    def _nested():
        class Inner(BaseModel):
            model_config = {"extra": "ignore"}
            a: int = 1
            b: int = 2
            c: int = 3

        class Outer(BaseModel):
            model_config = {"extra": "ignore"}
            inner: Inner = Inner()
            top: int = 0

        return Outer

    def test_paths_are_recorded_not_just_counted(self) -> None:
        ledger = self._walk({"top": 5}, self._nested())
        assert ledger.defaults_paths == ["inner"]
        assert ledger.defaults_injected == 1

    def test_the_count_is_derived_from_the_paths(self) -> None:
        """One derivation, never two.

        A separate ``+=`` counter alongside the list is how the two disagree
        later -- the failure mode this module exists to record.
        """
        ledger = self._walk({"top": 5}, self._nested())
        ledger.defaults_paths.append("synthetic.path")
        assert ledger.defaults_injected == len(ledger.defaults_paths) == 2

    def test_naming_a_block_makes_its_own_defaults_countable(self) -> None:
        """The spelling-dependence, pinned on a minimal case.

        Same resolved model both ways -- only the raw dict's depth differs. The
        second form is what a fold produces, and it is why two configs that run
        identically report different totals.
        """
        outer = self._nested()
        silent = self._walk({"top": 5}, outer)
        named = self._walk({"top": 5, "inner": {"a": 1}}, outer)

        assert outer.model_validate({"top": 5}) == outer.model_validate(
            {"top": 5, "inner": {"a": 1}}
        ), "the two spellings must resolve identically, else this proves nothing"

        assert silent.defaults_paths == ["inner"]
        assert named.defaults_paths == ["inner.b", "inner.c"]
        assert named.defaults_injected > silent.defaults_injected

    def test_to_dict_emits_both_from_one_source(self) -> None:
        ledger = self._walk({"top": 5}, self._nested())
        payload = ledger.to_dict()
        assert payload["defaults"] == ledger.defaults_paths
        assert payload["defaults_injected"] == len(payload["defaults"])
        assert payload["schema_version"] == 2, "adding `defaults` bumped the version"

    def test_defaults_never_enter_the_substitutions_list(self) -> None:
        """A typical arm defaults ~610 knobs against ~14 substitutions.

        Recording them as ``Substitution`` objects would bury every record that
        matters, which is why the original chose a count over per-key records.
        Paths keep the itemisation without paying that cost.
        """
        ledger = self._walk({"top": 5}, self._nested())
        assert ledger.defaults_paths
        assert ledger.substitutions == []


# ----------------------------------------------------------------------
# A FOLDED block is moved, not dropped -- and must still be descended into
# ----------------------------------------------------------------------


class _Folded(BaseModel):
    """Stands in for a canonical block a legacy name folds into."""

    model_config = {"extra": "ignore"}

    center_fraction: float = 0.08


class _WithFold(BaseModel):
    """A schema that accepts a legacy block name and folds it into place.

    Mirrors the real pair: `acceleration:` is accepted as input and folded to
    `undersampling:`, and the class publishes both the accepted set and the
    canonical chain so `core/` need not import `config/`.
    """

    model_config = {"extra": "ignore"}

    __folded_input_keys__: ClassVar[frozenset[str]] = frozenset({"acceleration"})
    __folded_input_paths__: ClassVar[dict[str, tuple[str, ...]]] = {
        "acceleration": ("undersampling",)
    }

    undersampling: _Folded = Field(default_factory=_Folded)


def test_a_folded_block_is_not_reported_as_dropped():
    """CONTROL: the whole point of the accepted-set is that a MOVE is not a drop."""
    ledger = ExecutionLedger.begin_run()
    raw = {"acceleration": {"center_fraction": 0.08}}
    diff_declared_vs_resolved(raw, _WithFold(undersampling=_Folded()), ledger=ledger)
    assert [
        s
        for s in ledger.substitutions
        if s.class_id is SubstitutionClass.EXTRA_IGNORE_DROPPED
    ] == []


def test_a_key_dropped_inside_a_folded_block_is_still_recorded():
    """DEFECT: accepting the key stopped the ledger descending into it.

    Being in the accepted set kept `acceleration` out of the drop loop -- right --
    but it is also absent from `resolved_keys`, so the recursion skipped it too
    and everything beneath became invisible. `acceleration:` is a whole top-level
    block, so that hid every `extra="ignore"` drop under it. Four independent
    call sites asserted the ledger records `acceleration.min_centre_fraction`
    and all four saw an empty ledger.
    """
    ledger = ExecutionLedger.begin_run()
    raw = {"acceleration": {"center_fraction": 0.08, "min_centre_fraction": 0.02}}
    diff_declared_vs_resolved(raw, _WithFold(undersampling=_Folded()), ledger=ledger)

    dropped = [
        s
        for s in ledger.substitutions
        if s.class_id is SubstitutionClass.EXTRA_IGNORE_DROPPED
    ]
    assert len(dropped) == 1, f"expected the British spelling to be caught, got {dropped}"
    # Reported under the LEGACY prefix the author actually wrote, not the
    # canonical one they would have to translate back from.
    assert dropped[0].path == "acceleration.min_centre_fraction"


def test_a_folded_scalar_has_no_sub_keys_and_is_skipped_quietly():
    """Only a MAPPING can hide a dropped sub-key; a scalar must not be walked."""
    ledger = ExecutionLedger.begin_run()
    diff_declared_vs_resolved({"acceleration": 4.0}, _WithFold(), ledger=ledger)
    assert ledger.substitutions == []

def test_unconsumed_keys_defaults_to_the_drop_class():
    """Every original caller meant "dropped"; the default must not move."""
    from spectramr.core.execution_ledger import (
        ExecutionLedger,
        SubstitutionClass,
        unconsumed_keys,
    )

    ledger = ExecutionLedger.begin_run(source="test")
    assert unconsumed_keys(
        {"a": 1, "b": 2}, {"a"}, site="s", stage="st", consumer="C.__init__"
    ) == ["b"]
    assert [s.class_id for s in ledger.substitutions] == [
        SubstitutionClass.DROPPED_UNCONSUMED_KWARG
    ]


def test_unconsumed_keys_var_kwargs_class_keeps_the_value_and_says_why():
    """A key swallowed by **kwargs is NOT dropped, so it must not resolve to
    NO_VALUE -- it is present but has no verified reader (#878)."""
    from spectramr.core.execution_ledger import (
        NO_VALUE,
        ExecutionLedger,
        SubstitutionClass,
        unconsumed_keys,
    )

    ledger = ExecutionLedger.begin_run(source="test")
    unconsumed_keys(
        {"b": 2},
        {"a"},
        site="s",
        stage="st",
        consumer="C.__init__(**kwargs)",
        class_id=SubstitutionClass.EXTRA_ALLOW_UNTYPED,
    )
    (sub,) = ledger.substitutions
    assert sub.class_id is SubstitutionClass.EXTRA_ALLOW_UNTYPED
    assert sub.resolved == 2 and sub.resolved is not NO_VALUE
    assert "only via **kwargs" in sub.reason


# --------------------------------------------------------------------------
# A dict -> model coercion is not a rewrite
# --------------------------------------------------------------------------
#
# `_normalise` canonicalises containers and enums but has no case for a pydantic
# model, and the caller's model-recursion branch only fires when the FIELD is a
# model -- a `list[LossComponentConfig]` is a *list*, so it fell through to raw
# equality and compared a list of dicts against a list of model objects. Never
# equal, so every arm declaring a loss list was reported as
# `value_changed_on_finalize` at severity "warning" over a coercion in which no
# value moved. `audit` is --strict by default and warnings exit 2 (CLAUDE.md #4),
# so this false positive gated real arms: two of
# experiment_11_attention_none's eight warnings were exactly this.


class _Component(BaseModel):
    name: str
    weight: float = 1.0
    enabled: bool = True


class _CoercingHolder(BaseModel):
    """Mirrors `LossConfigSchema.image_losses`: a list field of sub-models."""

    components: list[_Component] = Field(default_factory=list)
    scale: float = 1.0


def test_a_list_of_dicts_coerced_to_models_is_not_a_rewrite():
    """Sensitivity pair, half 1: pydantic parsing must stay quiet."""
    ledger = ExecutionLedger.begin_run(source="test")
    raw = {"components": [{"name": "hfen", "weight": 0.3, "enabled": True}]}
    diff_declared_vs_resolved(raw, _CoercingHolder(**raw), ledger=ledger, stage="st")
    assert [s.path for s in ledger.substitutions] == [], (
        "a dict -> model coercion in which no declared value moved was reported "
        f"as a rewrite: {[(s.path, s.class_id) for s in ledger.substitutions]}"
    )


def test_a_partial_declaration_does_not_diff_against_unwritten_defaults():
    """The inverse false positive a full `model_dump` would have introduced.

    An arm declaring only `{name}` must not be diffed against three dumped fields
    and reported as rewritten on the two defaults it never wrote.
    """
    ledger = ExecutionLedger.begin_run(source="test")
    raw = {"components": [{"name": "hfen"}]}
    diff_declared_vs_resolved(raw, _CoercingHolder(**raw), ledger=ledger, stage="st")
    assert [s.path for s in ledger.substitutions] == []


def test_a_real_rewrite_inside_a_coerced_list_still_fires():
    """Sensitivity pair, half 2 -- the half that proves the detector still works.

    A validator that clamps a DECLARED value changes a declared key, so it must
    still be reported. Without this case, silencing the false positive above
    would be indistinguishable from disabling the check.
    """

    class _Clamping(BaseModel):
        name: str
        weight: float = 1.0

        def __init__(self, **data):
            if "weight" in data:
                data["weight"] = min(float(data["weight"]), 0.5)
            super().__init__(**data)

    class _ClampingHolder(BaseModel):
        components: list[_Clamping] = Field(default_factory=list)

    ledger = ExecutionLedger.begin_run(source="test")
    raw = {"components": [{"name": "hfen", "weight": 0.9}]}
    diff_declared_vs_resolved(raw, _ClampingHolder(**raw), ledger=ledger, stage="st")
    assert [s.path for s in ledger.substitutions] == ["components"], (
        "a clamped declared weight was not reported; recorded "
        f"{[s.path for s in ledger.substitutions]}"
    )
    (sub,) = ledger.substitutions
    assert sub.class_id is SubstitutionClass.VALUE_CHANGED_ON_FINALIZE
    # The record must carry BOTH halves, or "rewritten" is unactionable.
    assert sub.requested[0]["weight"] == 0.9
    assert sub.resolved[0].weight == 0.5


def test_a_length_change_in_a_coerced_list_still_fires():
    """Dropping or adding an entry is a rewrite by any reading."""

    class _Truncating(BaseModel):
        components: list[_Component] = Field(default_factory=list)

        def __init__(self, **data):
            if "components" in data:
                data["components"] = data["components"][:1]
            super().__init__(**data)

    ledger = ExecutionLedger.begin_run(source="test")
    raw = {"components": [{"name": "a"}, {"name": "b"}]}
    diff_declared_vs_resolved(raw, _Truncating(**raw), ledger=ledger, stage="st")
    assert [s.path for s in ledger.substitutions] != []


# --------------------------------------------------------------------------
# Three quantities spelled "version"; the block used to name only one
# --------------------------------------------------------------------------


def test_the_ledger_says_what_its_schema_version_versions():
    """`schema_version: 2` beside a `metadata.version` of "6.0" read as a
    contradiction. It is not one -- and the artifact, not just a docstring, has to
    say so, because the audit's --json report keeps `_ledger` and discards the
    resolved config around it."""
    ledger = ExecutionLedger.begin_run(source="test")
    block = ledger.to_dict(config_version="1.0")
    assert block["schema_version"] == 2, "the ledger file-format version is unchanged"
    assert block["schema_version_of"] == "execution_ledger"
    assert block["config_version"] == "1.0"


def test_an_unknown_config_version_is_none_not_invented():
    """A tier the caller could not determine must read as "not stated"."""
    ledger = ExecutionLedger.begin_run(source="test")
    assert ledger.to_dict()["config_version"] is None


# ----------------------------------------------------------------------
# validation_alias -- the input-only spelling
# ----------------------------------------------------------------------
#
# `alias` and `validation_alias` are different attributes, and the ledger read
# only the first. The schema's one aliased field at the time, the since-retired
# `checkpoint.checkpoint_dir` (`validation_alias="save_dir"`, #2842), was
# therefore recorded as EXTRA_IGNORE_DROPPED at severity "error" on 293 corpus
# arms while its value landed intact. The stand-in below keeps that shape.
#
# Sensitivity pair, per this module's opening docstring: the alias must go
# quiet AND a genuinely-unknown key on the same model must still fire. Without
# the second half, `resolved_keys |= everything` would score green.


class AliasedByValidationAlias(BaseModel):
    """`validation_alias` only -- the shape the ledger was blind to."""

    model_config: ClassVar[dict] = {"extra": "ignore", "populate_by_name": True}
    checkpoint_dir: str = Field(default="./ckpt", validation_alias="save_dir")


def _dropped(ledger):
    return [
        s.path
        for s in ledger.substitutions
        if s.class_id is SubstitutionClass.EXTRA_IGNORE_DROPPED
    ]


def test_a_validation_alias_is_not_reported_as_dropped():
    """CLEAN: the declared spelling is an accepted input alias."""
    ledger = ExecutionLedger.begin_run(source="t")
    raw = {"save_dir": "/tmp/x"}
    instance = AliasedByValidationAlias(**raw)
    # The premise the assertion rests on: the value really does land.
    assert instance.checkpoint_dir == "/tmp/x"
    diff_declared_vs_resolved(raw, instance, ledger=ledger)
    assert _dropped(ledger) == []


def test_an_unknown_key_on_the_same_model_still_fires():
    """DEFECT: the alias fix must not blanket-accept every extra key."""
    ledger = ExecutionLedger.begin_run(source="t")
    raw = {"save_dir": "/tmp/x", "no_such_knob": 1}
    diff_declared_vs_resolved(raw, AliasedByValidationAlias(**raw), ledger=ledger)
    assert _dropped(ledger) == ["no_such_knob"]


def test_the_canonical_field_name_is_still_accepted():
    """`populate_by_name` means both spellings are legal input."""
    ledger = ExecutionLedger.begin_run(source="t")
    raw = {"checkpoint_dir": "/tmp/x"}
    diff_declared_vs_resolved(raw, AliasedByValidationAlias(**raw), ledger=ledger)
    assert _dropped(ledger) == []


def test_an_alias_choices_container_contributes_every_spelling():
    """`validation_alias` is not always a str -- AliasChoices must not be str()'d."""
    from pydantic import AliasChoices

    class MultiAlias(BaseModel):
        model_config: ClassVar[dict] = {"extra": "ignore", "populate_by_name": True}
        out_dir: str = Field(
            default="./o", validation_alias=AliasChoices("save_dir", "output_dir")
        )

    for spelling in ("save_dir", "output_dir"):
        ledger = ExecutionLedger.begin_run(source="t")
        raw = {spelling: "/tmp/x"}
        diff_declared_vs_resolved(raw, MultiAlias(**raw), ledger=ledger)
        assert _dropped(ledger) == [], f"{spelling} reported dropped"
    ledger = ExecutionLedger.begin_run(source="t")
    diff_declared_vs_resolved({"nope": 1}, MultiAlias(), ledger=ledger)
    assert _dropped(ledger) == ["nope"]
