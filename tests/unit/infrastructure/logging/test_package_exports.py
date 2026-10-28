"""Logging package exports — a single ``__all__`` (2026-06-19 audit #6).

``spectramr.infrastructure.logging.__init__`` used to declare ``__all__`` TWICE.
Module bodies execute top-to-bottom, so the second assignment silently
*replaced* the first — dropping the eagerly re-exported names (``banner``,
``phase``, ``smart_progress``, the provenance helpers) from
``from ... import *``. This pins the merged single list so the regression
cannot reappear.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import spectramr.infrastructure.logging as pkg

_INIT = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "spectramr"
    / "infrastructure"
    / "logging"
    / "__init__.py"
)


def test_exactly_one_dunder_all_assignment() -> None:
    tree = ast.parse(_INIT.read_text())
    assigns = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)
    ]
    assert len(assigns) == 1, (
        f"expected exactly one __all__ assignment, found {len(assigns)} — a "
        "second one clobbers the first (#6 of the 2026-06-19 audit)."
    )


def test_all_contains_eager_and_lazy_names() -> None:
    eager = {
        "banner",
        "phase",
        "make_progress",
        "collect_run_provenance",
        "count_parameters",
        "format_provenance_lines",
        "git_provenance",
        "log_provenance",
        "runtime_environment",
        "torch_runtime",
    }
    lazy = {
        "LoggingService",
        "LoggingServiceFactory",
        "MetricsTracker",
        "ColoredConsoleFormatter",
        "bootstrap_console_logging",
    }
    exported = set(pkg.__all__)
    missing = (eager | lazy) - exported
    assert not missing, f"__all__ is missing exported names: {sorted(missing)}"


def test_every_all_name_is_resolvable() -> None:
    """Eager names are real attributes; lazy names resolve via ``__getattr__``."""
    for name in pkg.__all__:
        assert getattr(pkg, name) is not None, (
            f"{name!r} is in __all__ but cannot be resolved on the package."
        )


def test_bootstrap_import_failure_names_the_real_cause(monkeypatch) -> None:
    """A broken transitive import must not read as a missing ``bootstrap_console_logging``."""
    import builtins
    import sys

    monkeypatch.delitem(
        sys.modules, "spectramr.infrastructure.services.logging_service", raising=False
    )
    real_import = builtins.__import__

    def failing(name, *args, **kwargs):
        if name == "spectramr.infrastructure.services.logging_service":
            raise ImportError("torch 2.7.1 is older than required")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing)
    with pytest.raises(ImportError, match=r"torch 2\.7\.1 is older than required"):
        pkg.__getattr__("bootstrap_console_logging")
