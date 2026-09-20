"""``metadata.audit_waivers``: what an arm may acknowledge, and what it may not.

The key sat in the corpus with no reader until the 2026-09-16 run made the cost
visible: ``experiment_11_kspace_cold_diffusion_radial`` and ``_spiral`` never
launched, because ``schedule.nesting_leakfree`` warns on every non-Cartesian
cascade and the witness's own fix hint (``enforce_nested: true``) RAISES on those
families. ``audit --strict`` exits 2 on a warning and the dispatcher skips train,
so the arm was unrunnable without ``--allow-warnings``, which CLAUDE.md forbids.

Each test here plants one shape the mechanism must handle, including the three
misuses it must refuse.
"""

from __future__ import annotations

from spectramr.infrastructure.validation.audit_waivers import (
    WAIVER_WITNESS_NAME,
    apply_audit_waivers,
    declared_waivers,
)
from spectramr.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Tier,
    WitnessVerdict,
)

NESTING = "schedule.nesting_leakfree"


def _warning(name: str = NESTING) -> WitnessVerdict:
    return WitnessVerdict(
        witness_name=name,
        passed=False,
        message="28 of 29 levels re-introduce removed k-space bins",
        severity=Severity.WARNING,
        category="schedule_certification",
        stage=Stage.CONSTRUCT,
        tier=Tier.T1,
    )


def _error(name: str = "channels.in_out_match") -> WitnessVerdict:
    return WitnessVerdict(
        witness_name=name,
        passed=False,
        message="in_channels != out_channels",
        severity=Severity.ERROR,
    )


def _config(*waivers: dict) -> dict:
    return {"metadata": {"audit_waivers": list(waivers)}}


def _waiver(check: str = NESTING, justification: str = "radial cannot nest (#1573)") -> dict:
    return {"check": check, "justification": justification}


def _flags(verdicts) -> list[WitnessVerdict]:
    return [v for v in verdicts if v.witness_name == WAIVER_WITNESS_NAME]


# --- reading the declaration ------------------------------------------------


def test_no_declaration_is_not_a_finding():
    verdicts = [_warning()]
    assert apply_audit_waivers(verdicts, {}) == verdicts
    assert apply_audit_waivers(verdicts, {"metadata": {}}) == verdicts
    assert declared_waivers({"metadata": {"audit_waivers": None}}) == []


def test_a_lone_mapping_is_read_as_one_waiver():
    """YAML lets an author write the mapping without the list dash."""
    assert len(declared_waivers({"metadata": {"audit_waivers": _waiver()}})) == 1


# --- the case the mechanism exists for --------------------------------------


def test_an_acknowledged_warning_stops_blocking_but_stays_printed():
    out = apply_audit_waivers([_warning()], _config(_waiver()))
    (waived,) = [v for v in out if v.witness_name == NESTING]
    assert waived.passed
    assert "28 of 29 levels" in waived.message, "the finding must still be readable"
    assert "radial cannot nest (#1573)" in waived.message
    assert not _flags(out)


def test_a_health_check_may_be_named_without_its_prefix():
    """The audit prints ``[health:x]``; an arm naming ``x`` means that check."""
    out = apply_audit_waivers(
        [_warning("health:held_out_test_declared")], _config(_waiver("held_out_test_declared"))
    )
    assert out[0].passed
    assert not _flags(out)


def test_unwaived_findings_are_untouched():
    out = apply_audit_waivers([_warning(), _warning("schedule.no_inert_steps")], _config(_waiver()))
    assert out[0].passed
    assert not out[1].passed


# --- the three misuses ------------------------------------------------------


def test_a_waiver_cannot_clear_an_error():
    out = apply_audit_waivers([_error()], _config(_waiver("channels.in_out_match")))
    assert not out[0].passed, "the error itself must keep failing"
    (flag,) = _flags(out)
    assert flag.severity is Severity.ERROR
    assert "does not clear an error" in flag.message


def test_a_waiver_without_a_justification_is_refused():
    (flag,) = _flags(apply_audit_waivers([_warning()], _config({"check": NESTING})))
    assert flag.severity is Severity.ERROR
    assert "no justification" in flag.message


def test_a_malformed_entry_is_refused():
    (flag,) = _flags(apply_audit_waivers([_warning()], _config("schedule.nesting_leakfree")))
    assert flag.severity is Severity.ERROR
    assert "must be a mapping" in flag.message


# --- the ratchet: a waiver must not outlive its reason ----------------------


def test_a_waiver_for_a_finding_that_no_longer_fires_is_reported_stale():
    passing = WitnessVerdict(
        witness_name=NESTING, passed=True, message="cascade is leak-free", severity=Severity.WARNING
    )
    (flag,) = _flags(apply_audit_waivers([passing], _config(_waiver())))
    assert flag.severity is Severity.WARNING
    assert "matched no reported finding" in flag.message


def test_a_waiver_naming_a_check_that_did_not_run_is_reported_stale():
    (flag,) = _flags(apply_audit_waivers([_warning()], _config(_waiver("schedule.typo"))))
    assert not flag.passed
    assert "schedule.typo" in flag.message
