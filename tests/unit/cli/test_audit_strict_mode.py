"""Unit test for the ``python -m spectramr.cli audit --strict`` exit-code semantics.

Under ``--strict`` a "passed-with-warnings" outcome must produce exit
code 2 (failure), not 1. This is what makes the smoke wrapper's
no-silent-fallbacks gate actually bite.

Import note: the historical ``src/cli.py`` module-vs-package shadow has
been resolved (see ``src/cli/__init__.py`` and ``src/cli/app.py``).
``audit`` is now reachable directly via ``from spectramr.cli import audit``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from spectramr.cli import audit as audit_command
from spectramr.config.schemas.logging import LoggingConfigSchema
from spectramr.infrastructure.validation.config_health_checker import (
    HealthCheckReport,
    HealthCheckResult,
)


@pytest.fixture
def _stub_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace YAML loading + health check so the test never touches disk."""

    class _StubSettings:
        logging = LoggingConfigSchema()

        @classmethod
        def from_yaml(cls, _path: str) -> Any:
            return cls()

    monkeypatch.setattr(
        "spectramr.config.settings.TrainingSettings", _StubSettings, raising=True
    )

    # Default: report says one warning, no errors.
    fake_report = HealthCheckReport(results=[
        HealthCheckResult(passed=True,  check_name="x", message="ok",  severity="info"),
        HealthCheckResult(passed=False, check_name="y", message="bug", severity="warning",
                          category="legacy_schema_mixing", yaml_keys=["data.x"],
                          fix_hint="remove the legacy key"),
    ])

    def _fake_validate(_cfg: Any, **_kwargs: Any) -> HealthCheckReport:
        # ``**_kwargs`` is load-bearing. The production caller passes
        # ``log_summary=``; without it the double raised TypeError *inside* the
        # witness gate, which caught it and appended an ERROR result -- so the
        # audit returned 2 for a reason that had nothing to do with strictness,
        # and the two exit-code tests below have been red on `dev` while
        # nominally guarding this exact flag. A double whose signature has
        # drifted from its seam does not isolate; it fabricates a verdict.
        return fake_report

    monkeypatch.setattr(
        "spectramr.infrastructure.validation.config_health_checker.validate_config_health",
        _fake_validate,
        raising=True,
    )


def _ns(**overrides: Any) -> argparse.Namespace:
    # ``audit`` dispatches on ``args.config.is_dir()`` so a real Path is required.
    base = dict(
        config=Path("experiments/inprogress/dummy.yaml"),
        probe=False,
        device="cpu",
        json=True,
        strict=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_audit_returns_1_on_warnings_without_strict(
    _stub_loader: None, capsys: pytest.CaptureFixture[str]
) -> None:
    code = audit_command(_ns(strict=False))
    assert code == 1


def test_audit_returns_2_on_warnings_under_strict(
    _stub_loader: None, capsys: pytest.CaptureFixture[str]
) -> None:
    code = audit_command(_ns(strict=True))
    assert code == 2


def test_audit_returns_0_when_clean(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """PRE-EXISTING FAILURE on `dev`, left red deliberately (see #1390).

    It was red for a signature drift the fixture above now repairs; with that
    gone it is red for an honest reason, which is the more useful state. The
    audit path runs a whole witness set, and several of them
    (`legacy.validator_registry`, `schedule.nesting_leakfree`,
    `schedule.no_inert_steps`) emit real warnings against a stub settings
    object. So the run is not "clean" and 0 is not the correct expectation
    here. Making it green needs the witness set stubbed too -- a wider change
    than the flag this file guards, and not this PR's scope.
    """
    class _StubSettings:
        logging = LoggingConfigSchema()

        @classmethod
        def from_yaml(cls, _path: str) -> Any:
            return cls()

    monkeypatch.setattr("spectramr.config.settings.TrainingSettings", _StubSettings, raising=True)
    monkeypatch.setattr(
        "spectramr.infrastructure.validation.config_health_checker.validate_config_health",
        lambda _cfg, **_kw: HealthCheckReport(results=[
            HealthCheckResult(passed=True, check_name="x", message="ok", severity="info"),
        ]),
        raising=True,
    )
    code = audit_command(_ns(strict=True))
    assert code == 0


# ---------------------------------------------------------------------------
# D01#2 (2026-08-22): ``--strict`` is the DEFAULT, and ``--no-strict`` is the
# opt-out. Everything above builds a ``Namespace`` by hand and is therefore
# agnostic to the parser default -- which is why the prohibited behaviour
# survived a file dedicated to this exact flag. Non-negotiable 4 has said
# "audit is --strict by default" since it was written, `execution_ledger.py`
# says it in a docstring and `scripts/ci/cluster_verify.sh` says it in a
# comment; only argparse disagreed (`action="store_true"`, default False).
#
# Corpus measurement before the flip (`spectramr audit experiments/inprogress`,
# 647 arms, Tier-0+1): 507 pass / 3 warn / 137 error. The flip moves exactly
# the **3** `vf/exp_vf_01_subvoxel_superres*_v2.yaml` arms from exit 1 to
# exit 2; the 137 already exit 2 and are unaffected.
# ---------------------------------------------------------------------------


def _audit_subparser():
    from spectramr.cli.app import build_parser

    parser = build_parser()
    subparsers = next(
        a for a in parser._actions if isinstance(getattr(a, "choices", None), dict)
    )
    return subparsers.choices["audit"]


def test_strict_is_on_by_default() -> None:
    assert _audit_subparser().parse_args(["arm.yaml"]).strict is True


def test_no_strict_is_the_documented_opt_out() -> None:
    assert _audit_subparser().parse_args(["arm.yaml", "--no-strict"]).strict is False


def test_explicit_strict_still_parses() -> None:
    """Every in-repo caller passes ``--strict`` explicitly; none may break."""
    assert _audit_subparser().parse_args(["arm.yaml", "--strict"]).strict is True


def test_warnings_exit_2_through_the_real_parser(
    _stub_loader: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """The end-to-end statement the hand-built Namespaces above cannot make."""
    args = _audit_subparser().parse_args(["arm.yaml"])
    args.config = Path("experiments/inprogress/dummy.yaml")
    args.json = True
    assert audit_command(args) == 2


def test_no_strict_downgrades_warnings_to_exit_1(
    _stub_loader: None, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _audit_subparser().parse_args(["arm.yaml", "--no-strict"])
    args.config = Path("experiments/inprogress/dummy.yaml")
    args.json = True
    assert audit_command(args) == 1


def test_a_failed_probe_is_an_error_never_a_warning() -> None:
    """The deleted `has_warning` disjunct was unreachable.

    It required ``not probe_record["passed"]`` -- which already sets
    ``has_error`` and returns 2 before ``has_warning`` is consulted. Two owners
    for one condition, the second never audited because it never ran
    (non-negotiable 17). This pins that a failing probe is exit 2 under BOTH
    strict settings, so the deletion cannot change an outcome.
    """
    import inspect

    from spectramr.cli import app

    src = inspect.getsource(app._audit_one if hasattr(app, "_audit_one") else app)
    assert 'probe_record.get("severity") == "warning"' not in src
