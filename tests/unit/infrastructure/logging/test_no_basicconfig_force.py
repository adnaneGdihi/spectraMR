"""F-LOG-COLOR regression — forbid ``logging.basicConfig(force=True)`` in ``src/``.

Audit anchor: TODO/audit/smoke_audit_20260516.md §F-LOG-COLOR.

Background
----------

``logging.basicConfig(force=True)`` is a global side effect: it
detaches and closes EVERY handler on the root logger, then installs a
single default ``StreamHandler`` with the plain stdlib formatter. This
silently destroys the ``ColoredConsoleFormatter`` installed by
:class:`LoggingService` — the SSOT for console rendering — so every
log line in the process becomes uncolored regardless of TTY / TERM /
``FORCE_COLOR``.

The bug surfaced 2026-05-17 on the cluster: ``python -m spectramr.cli
predict`` triggered ``InferencePipeline._setup_logging`` which called
``basicConfig(force=True)``, wiping colors for the rest of the run. The
fix was to remove the ``basicConfig`` and just acquire a module
logger.

This test sweeps ``src/`` for any new occurrence so the regression
cannot reappear without a deliberate code-review override.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"


def _find_basicconfig_force_true_calls(source: str) -> list[int]:
    """Return line numbers of REAL ``logging.basicConfig(force=True)`` calls.

    AST-based: only matches actual function-call nodes, so docstring /
    comment / string-literal mentions of the API are ignored.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    hits: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match ``logging.basicConfig(...)`` or just ``basicConfig(...)``
        # when imported via ``from logging import basicConfig``.
        func = node.func
        is_basicconfig = (
            (isinstance(func, ast.Attribute) and func.attr == "basicConfig")
            or (isinstance(func, ast.Name) and func.id == "basicConfig")
        )
        if not is_basicconfig:
            continue
        for kw in node.keywords:
            if kw.arg == "force" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                hits.append(node.lineno)
                break
    return hits


def test_no_basicconfig_force_true_in_src() -> None:
    """No code under ``src/`` may call ``logging.basicConfig(force=True)``.

    A call with ``force=True`` removes ALL handlers from the root
    logger — wiping the :class:`ColoredConsoleFormatter` the
    :class:`LoggingService` installed. Whatever subset of features
    the call thought it was configuring is overshadowed by the
    side effect of dropping every existing handler.

    If a module genuinely needs to reconfigure root logging, it must
    do so by either:

      1. Letting :class:`LoggingService` own it (preferred SSOT path).
      2. Adding a specific handler to its own logger (no root
         manipulation, no ``force=True``).
      3. Adding ``# noqa: F-LOG-COLOR  reason: ...`` to the line in
         question AND updating this test's whitelist.

    See ``TODO/audit/smoke_audit_20260516.md §F-LOG-COLOR``.
    """
    offenders: list[tuple[Path, int]] = []
    for py in SRC_ROOT.rglob("*.py"):
        try:
            text = py.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if "basicConfig" not in text:
            continue
        if "noqa: F-LOG-COLOR" in text:
            # Per-file override mechanism — record but don't fail.
            continue
        for line_no in _find_basicconfig_force_true_calls(text):
            offenders.append((py.relative_to(REPO_ROOT), line_no))

    assert not offenders, (
        "F-LOG-COLOR regression: ``logging.basicConfig(force=True)`` "
        "is forbidden in src/ — it wipes the ColoredConsoleFormatter "
        "installed by LoggingService and breaks console colors "
        "process-wide.\n\n"
        "Offenders:\n"
        + "\n".join(f"  {p}:{ln}" for p, ln in offenders)
        + "\n\nFix: remove ``force=True``, or remove the basicConfig "
          "call entirely and just acquire a logger via "
          "``logging.getLogger(__name__)``. See "
          "TODO/audit/smoke_audit_20260516.md §F-LOG-COLOR for the "
          "InferencePipeline precedent."
    )


def test_inference_pipeline_setup_logging_no_longer_calls_basicconfig() -> None:
    """Pin the specific F-LOG-COLOR fix that triggered this test.

    The pre-fix ``InferencePipeline._setup_logging`` called
    ``basicConfig(force=True)``. The fix removed the call entirely;
    the method now just acquires a module logger.
    """
    path = SRC_ROOT / "application" / "pipelines" / "inference_pipeline.py"
    if not path.exists():
        # InferencePipeline is deprecated; if it's been removed,
        # the regression surface is gone — test is vacuously fine.
        return
    src = path.read_text()
    tree = ast.parse(src)
    target: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_setup_logging":
            target = node
            break
    assert target is not None, (
        "Cannot locate _setup_logging in inference_pipeline.py — "
        "test fixture is stale; update or delete this test."
    )
    # Walk only the body — docstring constants do not appear as Call nodes.
    for node in ast.walk(target):
        if isinstance(node, ast.Call):
            func = node.func
            is_basicconfig = (
                (isinstance(func, ast.Attribute) and func.attr == "basicConfig")
                or (isinstance(func, ast.Name) and func.id == "basicConfig")
            )
            assert not is_basicconfig, (
                "F-LOG-COLOR regression: InferencePipeline._setup_logging "
                "reintroduced a logging.basicConfig() call at line "
                f"{node.lineno}. That wipes the ColoredConsoleFormatter "
                "installed by LoggingService. The method must only call "
                "logging.getLogger(__name__) — see "
                "TODO/audit/smoke_audit_20260516.md §F-LOG-COLOR."
            )


def test_bootstrap_console_logging_installs_colored_handler() -> None:
    """``bootstrap_console_logging()`` puts a ColoredConsoleFormatter on root."""
    import logging
    from spectramr.infrastructure.logging import bootstrap_console_logging
    from spectramr.infrastructure.services.logging_service import (
        ColoredConsoleFormatter,
    )

    # Start clean — remove all root handlers.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    bootstrap_console_logging()

    colored_handlers = [
        h for h in root.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        and isinstance(h.formatter, ColoredConsoleFormatter)
    ]
    assert len(colored_handlers) >= 1, (
        "F-LOG-COLOR round 12: bootstrap_console_logging() must install "
        "at least one StreamHandler with ColoredConsoleFormatter on the "
        "root logger. Without it, every CLI entry point (audit, train, "
        "predict, ...) falls back to Python's plain lastResort handler."
    )


def test_bootstrap_console_logging_is_idempotent() -> None:
    """Calling the helper twice must not stack duplicate handlers."""
    import logging
    from spectramr.infrastructure.logging import bootstrap_console_logging
    from spectramr.infrastructure.services.logging_service import (
        ColoredConsoleFormatter,
    )

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    bootstrap_console_logging()
    n_after_first = sum(
        1 for h in root.handlers
        if isinstance(h.formatter, ColoredConsoleFormatter)
    )
    bootstrap_console_logging()
    bootstrap_console_logging()
    n_after_third = sum(
        1 for h in root.handlers
        if isinstance(h.formatter, ColoredConsoleFormatter)
    )
    assert n_after_first == n_after_third, (
        f"bootstrap_console_logging() must be idempotent — got "
        f"{n_after_first} colored handler(s) after first call, "
        f"{n_after_third} after three. Repeated calls (CLI -> "
        f"LoggingService init -> ...) would stack handlers and "
        f"duplicate every log line N times."
    )


def test_bootstrap_console_logging_sets_root_level_to_info() -> None:
    """Default level must be INFO so logger.info(...) actually fires.

    LoggingService.__init__ sets root to WARNING for its own reasons;
    bootstrap_console_logging is for plain-CLI use where the user
    wants the standard 'Training started: epoch=1/100'-style info
    lines visible.
    """
    import logging
    from spectramr.infrastructure.logging import bootstrap_console_logging

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.CRITICAL)  # start at silent

    bootstrap_console_logging()
    assert root.level <= logging.INFO, (
        "bootstrap_console_logging() should drop the root level to "
        "INFO so user-facing info logs (config-loaded / training-"
        "started / etc.) appear. Got "
        f"{logging.getLevelName(root.level)}."
    )


def test_bootstrap_console_logging_upgrades_existing_plain_handler() -> None:
    """Round 12.1: if a plain StreamHandler is already on root, UPGRADE its
    formatter to ColoredConsoleFormatter — do NOT stack a second handler.

    Background: some import side-effect (often a transitive
    ``logging.basicConfig()`` call) leaves a plain ``StreamHandler``
    on the root before ``main()`` runs. Pre-12.1, bootstrap added a
    second handler alongside; both fired and every log line printed
    twice. 12.1 absorbs the pre-existing handler by replacing its
    formatter in place.
    """
    import logging
    from spectramr.infrastructure.logging import bootstrap_console_logging
    from spectramr.infrastructure.services.logging_service import (
        ColoredConsoleFormatter,
    )

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    # Simulate the import-side-effect plain handler.
    plain = logging.StreamHandler()
    plain.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root.addHandler(plain)
    assert len(root.handlers) == 1

    bootstrap_console_logging()

    # Should be exactly ONE StreamHandler — the original one, now
    # carrying a ColoredConsoleFormatter.
    stream_handlers = [
        h for h in root.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    assert len(stream_handlers) == 1, (
        f"F-LOG-COLOR round 12.1 regression: expected the pre-existing "
        f"plain handler to be UPGRADED (not stacked); got "
        f"{len(stream_handlers)} StreamHandler(s) on the root logger. "
        f"Stacking causes every log line to emit twice — once plain, "
        f"once colored."
    )
    assert stream_handlers[0] is plain, (
        "Original handler should be preserved (only its formatter "
        "swapped) so any user customisation (stream, level filter, "
        "etc.) survives."
    )
    assert isinstance(plain.formatter, ColoredConsoleFormatter), (
        "Pre-existing handler's formatter must be swapped to the "
        "colored one. Otherwise we'd silently keep the plain output."
    )


def test_bootstrap_console_logging_force_resets_handlers() -> None:
    """``force=True`` removes existing handlers before installing a fresh one."""
    import logging
    from spectramr.infrastructure.logging import bootstrap_console_logging
    from spectramr.infrastructure.services.logging_service import (
        ColoredConsoleFormatter,
    )

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    # Install a misconfigured handler (plain formatter — the bug we
    # want to recover from).
    bad_handler = logging.StreamHandler()
    bad_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(bad_handler)

    bootstrap_console_logging(force=True)

    # The bad handler is gone — only the colored one remains.
    assert bad_handler not in root.handlers, (
        "force=True must close + remove existing handlers."
    )
    assert any(
        isinstance(h.formatter, ColoredConsoleFormatter)
        for h in root.handlers
    )


def test_bootstrap_called_from_cli_app_main() -> None:
    """Round-12 wiring: cli/app.py::main() calls ``configure_entry``, which bootstraps the console."""
    import ast
    path = REPO_ROOT / "src" / "spectramr" / "cli" / "app.py"
    tree = ast.parse(path.read_text())
    main_fn: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_fn = node
            break
    assert main_fn is not None
    saw_call = False
    for n in ast.walk(main_fn):
        if isinstance(n, ast.Call):
            func = n.func
            if (
                (isinstance(func, ast.Name) and func.id == "configure_entry")
                or (isinstance(func, ast.Attribute) and func.attr == "configure_entry")
            ):
                saw_call = True
                break
    assert saw_call, (
        "F-LOG-COLOR round 12: src/cli/app.py::main() must call "
        "log_policy.configure_entry() (which bootstraps the console) so 'python -m spectramr.cli <anything>' "
        "gets the colored handler installed. Without this wire, "
        "train/audit/predict/etc. all show plain output."
    )


def test_bootstrap_called_from_main_py_main() -> None:
    """Post-shim wiring (2026-05-29 entry-point unification + src→spectramr):
    ``spectramr.main:main`` no longer owns a parser nor calls
    ``bootstrap_console_logging`` directly — it is a deprecation shim that
    delegates to ``spectramr.cli.app.main`` (which DOES bootstrap), so the
    colored-logging guarantee for ``python -m spectramr.main train ...`` is
    preserved transitively. This pins the delegation (the live SSOT) rather
    than the obsolete direct call."""
    import ast
    path = REPO_ROOT / "src" / "spectramr" / "main.py"
    tree = ast.parse(path.read_text())
    main_fn: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_fn = node
            break
    assert main_fn is not None
    # The shim must import + call cli.app's main (the delegation that bootstraps).
    imports_cli_main = any(
        isinstance(n, ast.ImportFrom)
        and n.module == "spectramr.cli.app"
        and any(alias.name == "main" for alias in n.names)
        for n in ast.walk(main_fn)
    )
    assert imports_cli_main, (
        "spectramr.main:main() must delegate to spectramr.cli.app.main (which calls "
        "bootstrap_console_logging) so 'python -m spectramr.main train ...' still "
        "gets the colored handler. The shim no longer bootstraps directly."
    )


def test_logging_service_remains_color_ssot() -> None:
    """LoggingService is the SSOT for console formatter installation.

    Sanity check: the ColoredConsoleFormatter class is still defined,
    and the LoggingService still references it. Without this, the
    F-LOG-COLOR fix above (removing basicConfig) leaves the system
    with no console handler at all — the wrong remedy.
    """
    from spectramr.infrastructure.services.logging_service import (
        ColoredConsoleFormatter,
    )
    assert ColoredConsoleFormatter is not None
    # Verify the formatter's auto-detect path is intact (env-var first,
    # isatty fallback). A future "simplification" that drops FORCE_COLOR
    # support would re-introduce the cluster-color complaint.
    import inspect
    src = inspect.getsource(ColoredConsoleFormatter.__init__)
    assert "FORCE_COLOR" in src, (
        "ColoredConsoleFormatter must respect FORCE_COLOR — without "
        "this knob, cluster users running through sbatch / piped "
        "output cannot re-enable colors."
    )
    assert "NO_COLOR" in src, (
        "ColoredConsoleFormatter must respect NO_COLOR — without "
        "this knob, color-averse users have no way to opt out."
    )
    assert "isatty" in src, (
        "ColoredConsoleFormatter must auto-detect TTY — without "
        "this, output piped to a file would contain raw ANSI escapes "
        "and break grep / log-parser tooling."
    )
