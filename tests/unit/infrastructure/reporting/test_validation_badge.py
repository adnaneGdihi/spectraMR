r"""Tests for the validation-badge document (ULF-PR-26 / V.3).

Targets ``spectramr.infrastructure.reporting.validation_badge``.

Categories:

- ``GateResult`` dataclass round-trips through dict
- ``ValidationBadge.from_gates`` flags missing gates explicitly
- All-passed badge is ``eligible = True``
- Any failed gate disqualifies the badge (audit-gate semantic)
- Custom ``required_gates`` list is respected (ablation experiments
  may opt out of specific gates)
- ``to_json`` produces a valid JSON document
- Non-``GateResult`` value rejected at construction
- ``DEFAULT_REQUIRED_GATES`` matches the plan's six gates
"""

from __future__ import annotations

import json

import pytest

from spectramr.infrastructure.reporting.validation_badge import (
    BadgeAuditResult,
    DEFAULT_REQUIRED_GATES,
    GateResult,
    ValidationBadge,
    validation_badge_gates_complete,
)


# ---------------------------------------------------------------------------
# DEFAULT_REQUIRED_GATES contract (matches the plan)
# ---------------------------------------------------------------------------


def test_default_required_gates_match_plan() -> None:
    """The default required-gate list matches the six the plan codifies."""
    expected = (
        "iqm",
        "downstream",
        "hallucination",
        "conformal_coverage",
        "pathology_recall",
        "pac_bayes",
    )
    assert DEFAULT_REQUIRED_GATES == expected


# ---------------------------------------------------------------------------
# GateResult
# ---------------------------------------------------------------------------


def test_gate_result_to_dict_round_trip() -> None:
    """``GateResult.to_dict`` carries all fields."""
    g = GateResult(
        passed=True, details="bound = 0.05 < target = 0.10",
        threshold=0.10, observed=0.05,
    )
    d = g.to_dict()
    assert d == {
        "passed": True,
        "details": "bound = 0.05 < target = 0.10",
        "threshold": 0.10,
        "observed": 0.05,
    }


def test_gate_result_optional_threshold_observed() -> None:
    """Threshold / observed default to ``None``."""
    g = GateResult(passed=True, details="binary check")
    assert g.threshold is None
    assert g.observed is None


# ---------------------------------------------------------------------------
# ValidationBadge — eligibility logic
# ---------------------------------------------------------------------------


def _all_pass_gates() -> dict[str, GateResult]:
    """Helper: gates dict with every required gate passing."""
    return {
        name: GateResult(passed=True, details="ok")
        for name in DEFAULT_REQUIRED_GATES
    }


def test_all_pass_eligible() -> None:
    """All required gates passing → ``eligible = True``."""
    badge = ValidationBadge.from_gates(_all_pass_gates())
    assert badge.eligible is True
    assert badge.missing_gates == ()
    assert badge.failed_gates == ()


def test_missing_required_gate_disqualifies() -> None:
    """Missing a required gate disqualifies the badge."""
    gates = _all_pass_gates()
    del gates["pac_bayes"]
    badge = ValidationBadge.from_gates(gates)
    assert badge.eligible is False
    assert "pac_bayes" in badge.missing_gates


def test_failed_required_gate_disqualifies() -> None:
    """A failed required gate disqualifies the badge."""
    gates = _all_pass_gates()
    gates["hallucination"] = GateResult(passed=False, details="FPR exceeded β")
    badge = ValidationBadge.from_gates(gates)
    assert badge.eligible is False
    assert badge.failed_gates == ("hallucination",)


def test_extra_gates_do_not_disqualify() -> None:
    """Extra gates beyond the required set do not affect eligibility."""
    gates = _all_pass_gates()
    gates["custom_iqm_extension"] = GateResult(passed=False, details="ignored")
    badge = ValidationBadge.from_gates(gates)
    assert badge.eligible is True


def test_custom_required_gates_respected() -> None:
    """A custom required-gates tuple overrides the default set."""
    gates = {"iqm": GateResult(passed=True, details="ok")}
    badge = ValidationBadge.from_gates(
        gates, required_gates=("iqm",)
    )
    assert badge.eligible is True


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_non_dict_gates_rejected() -> None:
    """``gates`` must be a dict."""
    with pytest.raises(TypeError, match="gates"):
        ValidationBadge.from_gates(gates=[("iqm", True)])  # type: ignore[arg-type]


def test_non_gate_result_value_rejected() -> None:
    """Each gate value must be a ``GateResult``."""
    with pytest.raises(TypeError, match="GateResult"):
        ValidationBadge.from_gates(gates={"iqm": True})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_to_dict_includes_required_meta() -> None:
    """Top-level dict has ``eligible``, ``missing_gates``, ``failed_gates``."""
    badge = ValidationBadge.from_gates(_all_pass_gates())
    d = badge.to_dict()
    assert "eligible" in d
    assert "missing_gates" in d
    assert "failed_gates" in d
    assert "gates" in d


def test_to_json_round_trips() -> None:
    """``to_json`` produces a valid JSON document with the eligibility flag."""
    badge = ValidationBadge.from_gates(_all_pass_gates())
    text = badge.to_json()
    parsed = json.loads(text)
    assert parsed["eligible"] is True
    assert "iqm" in parsed["gates"]


def test_to_json_failure_case() -> None:
    """Failure case round-trips correctly through JSON."""
    gates = _all_pass_gates()
    gates["pac_bayes"] = GateResult(passed=False, details="vacuous bound")
    badge = ValidationBadge.from_gates(gates)
    parsed = json.loads(badge.to_json())
    assert parsed["eligible"] is False
    assert parsed["failed_gates"] == ["pac_bayes"]
    assert parsed["gates"]["pac_bayes"]["passed"] is False


# ---------------------------------------------------------------------------
# `.write(path)` disk emitter + `validation_badge_gates_complete` audit hook
# ---------------------------------------------------------------------------


def _all_passing_gates() -> dict[str, GateResult]:
    return {
        name: GateResult(passed=True, details=f"{name} OK")
        for name in DEFAULT_REQUIRED_GATES
    }


def test_write_creates_json_file_with_full_payload(tmp_path) -> None:
    """``ValidationBadge.write(path)`` emits a JSON file equal to ``to_dict()``."""
    badge = ValidationBadge.from_gates(_all_passing_gates())
    out = tmp_path / "subdir" / "badge.json"
    written = badge.write(out)
    assert written == out
    assert out.exists()
    assert json.loads(out.read_text()) == badge.to_dict()


def test_audit_hook_passes_when_all_required_gates_green() -> None:
    """All required gates pass → audit hook returns ``passed=True``."""
    badge = ValidationBadge.from_gates(_all_passing_gates())
    result = validation_badge_gates_complete(badge)
    assert isinstance(result, BadgeAuditResult)
    assert result.passed is True
    assert result.missing_gates == ()
    assert result.failed_gates == ()
    assert "clinical-pilot eligible" in result.message


def test_audit_hook_flags_missing_gate() -> None:
    """Missing required gates surface in the audit result."""
    gates = _all_passing_gates()
    del gates["pac_bayes"]  # required gate omitted
    badge = ValidationBadge.from_gates(gates)
    result = validation_badge_gates_complete(badge)
    assert result.passed is False
    assert "pac_bayes" in result.missing_gates
    assert "missing" in result.message
    assert "pac_bayes" in result.message


def test_audit_hook_flags_failed_gate() -> None:
    """Failed required gates surface in the audit result."""
    gates = _all_passing_gates()
    gates["hallucination"] = GateResult(passed=False, details="FPR > bound")
    badge = ValidationBadge.from_gates(gates)
    result = validation_badge_gates_complete(badge)
    assert result.passed is False
    assert result.failed_gates == ("hallucination",)
    assert "failed" in result.message
    assert "hallucination" in result.message


def test_audit_hook_reports_both_missing_and_failed_gates() -> None:
    """If both missing and failed gates exist, both surface in the message."""
    gates = _all_passing_gates()
    gates["hallucination"] = GateResult(passed=False, details="FPR > bound")
    del gates["iqm"]
    badge = ValidationBadge.from_gates(gates)
    result = validation_badge_gates_complete(badge)
    assert result.passed is False
    assert "iqm" in result.missing_gates
    assert "hallucination" in result.failed_gates
    assert "missing" in result.message
    assert "failed" in result.message


# ── from_certificate_paths: upstream JSON -> gates, failing closed (#3041) ─────


def _write(tmp_path, name, payload) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_missing_artefacts_fail_every_required_gate(tmp_path) -> None:
    badge = ValidationBadge.from_certificate_paths({}, required_gates=DEFAULT_REQUIRED_GATES)
    assert badge.eligible is False
    assert set(badge.failed_gates) == set(DEFAULT_REQUIRED_GATES)
    assert badge.missing_gates == ()


def test_unreadable_and_absent_artefacts_are_failed_not_skipped(tmp_path) -> None:
    bad = tmp_path / "r1.json"
    bad.write_text("{not json", encoding="utf-8")
    badge = ValidationBadge.from_certificate_paths(
        {"r1_cdfc": str(bad), "r5_pac_bayes": str(tmp_path / "gone.json")},
        required_gates=("conformal_coverage", "pac_bayes"),
    )
    assert set(badge.failed_gates) == {"conformal_coverage", "pac_bayes"}
    assert badge.gates["conformal_coverage"].details.startswith("r1_cdfc: unreadable")


def test_each_certificate_maps_to_its_gate(tmp_path) -> None:
    upstream = {
        "r1_cdfc": _write(
            tmp_path, "r1.json", {"alpha": 0.1, "empirical_coverage": 0.93}
        ),
        "r2_chd": _write(
            tmp_path,
            "r2.json",
            {"beta": 0.05, "empirical_fpr": 0.02, "bound_is_tight": True},
        ),
        "r4_prc": _write(tmp_path, "r4.json", {"is_certified": True}),
        "r5_pac_bayes": _write(tmp_path, "r5.json", {"is_non_vacuous": False, "bound": 1.4}),
    }
    badge = ValidationBadge.from_certificate_paths(
        upstream,
        required_gates=("conformal_coverage", "hallucination", "pathology_recall", "pac_bayes"),
    )
    assert badge.gates["conformal_coverage"].passed is True
    assert badge.gates["hallucination"].passed is True
    assert badge.gates["pathology_recall"].passed is True
    assert badge.gates["pac_bayes"].passed is False
    assert badge.failed_gates == ("pac_bayes",)


def test_chd_with_a_loose_dkw_bound_fails_the_hallucination_gate(tmp_path) -> None:
    upstream = {
        "r2_chd": _write(
            tmp_path, "r2.json", {"beta": 0.05, "empirical_fpr": 0.01, "bound_is_tight": False}
        )
    }
    badge = ValidationBadge.from_certificate_paths(upstream, required_gates=("hallucination",))
    assert badge.gates["hallucination"].passed is False


def test_unknown_certificate_letter_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown certificate letter 'r9_x'"):
        ValidationBadge.from_certificate_paths({"r9_x": "x.json"})


def test_malformed_artefact_fails_its_gate_instead_of_raising(tmp_path) -> None:
    upstream = {"r1_cdfc": _write(tmp_path, "r1.json", {"empirical_coverage": 0.95})}
    badge = ValidationBadge.from_certificate_paths(
        upstream, required_gates=("conformal_coverage",)
    )
    assert badge.gates["conformal_coverage"].passed is False
    assert "malformed artefact" in badge.gates["conformal_coverage"].details
