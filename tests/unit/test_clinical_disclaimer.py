"""Tests for the package-level clinical disclaimer (:func:`spectramr._emit_clinical_disclaimer`).

Regression (2026-06-30): the ``NOT FOR CLINICAL USE`` ``UserWarning`` lived at
module top level, so every ``spawn``-ed DataLoader worker re-imported the package
and re-emitted it — N+1 copies for N workers cluttering batch logs ("disclaimer
should only be one").

Regression (2026-08-15): the child gate that fix installed never fired. It tested
``multiprocessing.parent_process() is not None``, and at *import* time in a
spawned child that is still ``None`` — the re-import happens while UNPICKLING the
target, strictly before ``BaseProcess._bootstrap`` assigns
``multiprocessing._parent_process``. Measured on 3.12, both ``spawn`` and
``forkserver``::

    IMPORT parent_process_is_None=True name='ForkServerProcess-2' _inheriting=True
    IMPORT parent_process_is_None=True name='SpawnProcess-3'      _inheriting=True

So the burst came back the moment DataLoader workers started. ``current_process()``
IS populated in that window (it is unpickled from the parent), which is what
:func:`spectramr._in_child_process` uses instead. A third gate was added at the same
time: N ranks of a torchrun launch are N interpreters, so they emitted N copies of
one legal notice.

These tests pin all three gates, and — because the failed gate was a *plausible*
API used in the wrong window — the child gate is also exercised against a real
spawned child rather than only a monkeypatched predicate.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import warnings

import pytest

import spectramr


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("SPECTRAMR_SUPPRESS_CLINICAL_WARNING", raising=False)
    # The rank gate must not read a torchrun environment leaking in from a
    # parent job (these tests assert single-process behaviour by default).
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)


def test_emits_in_main_process_by_default():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emitted = spectramr._emit_clinical_disclaimer()
    assert emitted is True
    assert any("NOT FOR CLINICAL USE" in str(w.message) for w in caught)


def test_env_flag_suppresses(monkeypatch):
    monkeypatch.setenv("SPECTRAMR_SUPPRESS_CLINICAL_WARNING", "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emitted = spectramr._emit_clinical_disclaimer()
    assert emitted is False
    assert caught == []


# --------------------------------------------------------------------------- #
# Gate 2: spawned children (DataLoader workers)
# --------------------------------------------------------------------------- #
class _FakeProcess:
    def __init__(self, name, inheriting=False):
        self.name = name
        if inheriting:
            self._inheriting = True


@pytest.mark.parametrize(
    "proc",
    [
        _FakeProcess("SpawnProcess-3", inheriting=True),
        _FakeProcess("ForkServerProcess-2", inheriting=True),
        # A worker whose private ``_inheriting`` flag is gone (it is CPython
        # internal); the process name still discriminates.
        _FakeProcess("SpawnPoolWorker-1"),
    ],
    ids=["spawn", "forkserver", "name-only"],
)
def test_silent_in_child_process(proc, monkeypatch):
    monkeypatch.setattr(spectramr.multiprocessing, "current_process", lambda: proc)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emitted = spectramr._emit_clinical_disclaimer()
    assert emitted is False
    assert caught == []


def test_parent_process_is_not_what_the_gate_reads():
    """The 2026-08-15 root cause, pinned as an executable statement.

    ``parent_process()`` returning ``None`` must NOT be read as "main process":
    it is ``None`` in a spawned child too, during the import window this gate
    runs in. If someone reintroduces that test, this fails.
    """
    monkey = _FakeProcess("SpawnProcess-1", inheriting=True)
    assert spectramr._in_child_process.__doc__ is not None
    import unittest.mock as m

    with (
        m.patch.object(spectramr.multiprocessing, "current_process", lambda: monkey),
        m.patch.object(spectramr.multiprocessing, "parent_process", lambda: None),
    ):
        # parent_process() lies here; the gate must still say "child".
        assert spectramr._in_child_process() is True


@pytest.mark.parametrize("method", ["spawn", "forkserver"])
def test_real_child_process_does_not_re_emit(method, tmp_path):
    """End-to-end: import spectramr in a genuinely spawned child.

    The monkeypatched cases above pin the predicate; this one pins that the
    predicate is consulted in the window that actually matters. A child that
    re-imports the package must record ``emitted=False``.
    """
    if method not in mp.get_all_start_methods():  # pragma: no cover - platform
        pytest.skip(f"{method} start method unavailable")
    out = tmp_path / "child.txt"
    ctx = mp.get_context(method)
    proc = ctx.Process(target=_child_reports_emission, args=(str(out),))
    proc.start()
    proc.join(timeout=180)
    assert proc.exitcode == 0, f"child failed (exitcode={proc.exitcode})"
    assert out.read_text().strip() == "False", (
        "a spawned child re-emitted the clinical disclaimer — the import-time "
        "child gate is inoperative again"
    )


def _child_reports_emission(path: str) -> None:
    """Runs in the child. Module-level, so it is picklable by spawn."""
    # Re-run the guard rather than trusting the import-time call: the child has
    # already executed the module body by the time this function is unpickled.
    import spectramr as child_pkg

    with open(path, "w") as fh:
        fh.write(str(child_pkg._emit_clinical_disclaimer()))


# --------------------------------------------------------------------------- #
# Gate 3: non-zero ranks of a distributed launch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rank", ["1", "2", "3"])
def test_silent_on_secondary_rank(rank, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", rank)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert spectramr._emit_clinical_disclaimer() is False
    assert caught == []


def test_still_emits_on_rank_zero(monkeypatch):
    """One notice per job, not zero — this is a legal disclaimer."""
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert spectramr._emit_clinical_disclaimer() is True
    assert any("NOT FOR CLINICAL USE" in str(w.message) for w in caught)


def test_rank_env_names_match_the_env_ssot():
    """``spectramr/__init__.py`` reads RANK/WORLD_SIZE literally, not via the SSOT.

    The original reason was cost: ``from spectramr.core import env`` executed an
    eager ``spectramr.core.__init__`` that pulled torch on every ``import
    spectramr``. **That reason no longer holds.** ``core/__init__.py`` became
    lazy in #1130, and the same import was measured on this machine at
    0.013 s / 102 modules / 12 MB with torch absent, against
    3.3 s / 4344 modules / 976 MB before. The literals are therefore no longer
    *required* — only still sufficient, and left alone here because rewriting a
    legal-disclaimer gate is not this change's business.

    The test keeps its full value either way, because it never rested on the
    cost argument: it pins the literals to the names ``core/env.py`` declares,
    so a rename there cannot silently strand the gate.
    """
    import inspect

    from spectramr.core import env

    src = inspect.getsource(spectramr._emit_clinical_disclaimer)
    assert f'"{env.WORLD_SIZE}"' in src
    assert f'"{env.RANK}"' in src
    assert f'"{env.SPECTRAMR_SUPPRESS_CLINICAL_WARNING}"' in src


def test_package_import_is_torch_free():
    """The gate must not have dragged torch into ``import spectramr``."""
    assert "spectramr" in sys.modules, "precondition: package already imported"
    code = "import spectramr, sys; print('torch' in sys.modules)"
    import subprocess

    env = dict(os.environ, SPECTRAMR_SUPPRESS_CLINICAL_WARNING="1")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "False", (
        "importing spectramr now pulls torch — the disclaimer gate must read "
        "os.environ directly, not through spectramr.core.env"
    )


# ---------------------------------------------------------------------------
# Third-party warning filters registered at package import
# ---------------------------------------------------------------------------
#
# `spectramr/__init__.py` registers these at the package-init seam rather than in
# `main.py`, because `import spectramr.<anything>` runs first and a filter
# installed later is a filter that never sees the import-time warning.

#: The literal text torch emits, copied from a cluster job log (8590891, torch
#: 2.14.0+cu126) rather than retyped — a regex that matches a paraphrase and not
#: the real string is the whole failure mode under test.
_TORCH_JIT_MESSAGES = (
    "`torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.",
    "`torch.jit.interface` is deprecated. Please use `torch.compile` instead.",
)

_OLD_FILTER_CATEGORY = DeprecationWarning
_CURRENT_FILTER = {
    "message": r"`?torch\.jit\.(script|script_method|interface)`? is deprecated",
    "category": Warning,
}


def _swallowed(filter_kwargs: dict, message: str, category: type[Warning]) -> bool:
    """True when a filter built from *filter_kwargs* suppresses this warning.

    Each call gets its own `catch_warnings` context with a cleared filter list,
    so the filters the real package import already installed cannot make an
    assertion pass for the wrong reason.
    """
    with warnings.catch_warnings(record=True) as seen:
        warnings.resetwarnings()
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", **filter_kwargs)
        warnings.warn(message, category, stacklevel=1)
    return not seen


@pytest.mark.parametrize("message", _TORCH_JIT_MESSAGES)
@pytest.mark.parametrize("category", [DeprecationWarning, FutureWarning])
def test_torch_jit_notice_is_swallowed_under_either_category(
    message: str, category: type[Warning]
) -> None:
    """torch has raised this same text under both categories.

    2.13 uses DeprecationWarning (`torch/jit/_script.py:1490`), 2.14 uses
    FutureWarning (`:1491`). Keying the filter on the base `Warning` class covers
    both and survives the next re-categorisation; the message is what identifies
    the notice.
    """
    assert _swallowed(_CURRENT_FILTER, message, category)


@pytest.mark.parametrize("message", _TORCH_JIT_MESSAGES)
def test_the_previous_category_keyed_filter_let_futurewarning_through(
    message: str,
) -> None:
    """The planted violation (non-negotiable 15).

    Without this, the test above would have been green against the filter it
    replaces: `category=DeprecationWarning` matched fine on torch 2.13 and
    stopped matching at the 2.14 bump, with nothing to report the change. Every
    job log since carried two lines this filter claimed to remove.
    """
    old = {"message": _CURRENT_FILTER["message"], "category": _OLD_FILTER_CATEGORY}
    assert _swallowed(old, message, DeprecationWarning), "old filter never worked at all"
    assert not _swallowed(old, message, FutureWarning), (
        "the old category-keyed filter is being credited with a suppression it "
        "did not perform — this test no longer demonstrates the regression"
    )


def test_the_filter_is_registered_by_importing_the_package() -> None:
    """Registering it is half the job; the package must actually install it.

    Asserted against the live `warnings.filters` rather than by re-reading the
    source, so deleting the `filterwarnings` call fails here (pitfall 16).

    In a SUBPROCESS, because pytest replaces the process filter list from
    `pyproject.toml`'s `filterwarnings` for the duration of each test — an
    in-process assertion here reports pytest's configuration, not the package's,
    and would have failed against correct code.
    """
    import subprocess

    probe = (
        "import warnings, spectramr\n"
        # 'torch' and 'jit' as separate substrings, NOT 'torch.jit': the
        # compiled pattern escapes the dots, so the literal spelling never
        # matches and the probe would report the filter missing when it is there.
        "hits = [f for f in warnings.filters "
        "if f[0] == 'ignore' and f[1] is not None "
        "and 'torch' in f[1].pattern and 'jit' in f[1].pattern]\n"
        "print(len(hits), hits[0][2].__name__ if hits else '')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env={**os.environ}
    )
    assert result.returncode == 0, result.stderr
    count, category = result.stdout.split()
    assert count == "1", f"expected exactly one torch.jit filter, got {count}"
    assert category == "Warning", (
        f"the torch.jit filter is keyed on {category}, not the base Warning class; "
        "torch has already moved this notice between DeprecationWarning and "
        "FutureWarning once"
    )


def test_third_party_invalid_escape_is_swallowed() -> None:
    """POT's `ot.datasets` docstring emits this while being COMPILED.

    So it fires on a cold `__pycache__` — and on every task of a job that
    disables the bytecode cache, which is how it reached the array logs.
    """
    assert _swallowed(
        {"message": r"invalid escape sequence", "category": SyntaxWarning},
        "invalid escape sequence '\\d'",
        SyntaxWarning,
    )


def test_our_own_invalid_escapes_are_still_caught_by_lint(tmp_path) -> None:
    """Anti-vacuity for the filter above.

    That filter silences `invalid escape sequence` repo-wide, which is only
    acceptable because something else still owns OURS. Ruff's W605 does, on
    every added line (non-negotiable 24). If W605 is ever switched off, the
    runtime filter turns into a way to hide a real defect.
    """
    import subprocess

    offender = tmp_path / "offender.py"
    offender.write_text('x = "\\d"\n')
    result = subprocess.run(
        ["ruff", "check", str(offender), "--output-format=concise"],
        capture_output=True,
        text=True,
    )
    assert "W605" in result.stdout, (
        "ruff no longer flags our own invalid escape sequences, so the runtime "
        f"SyntaxWarning filter is now hiding them: {result.stdout!r}"
    )
