"""Regression tests for the centralised env-var module."""

from __future__ import annotations

from pathlib import Path

import pytest

from spectramr.core import env


def test_names_lists_every_module_constant() -> None:
    """``env.names()`` must enumerate every public name advertised via __all__."""
    names = env.names()
    # Must contain the load-bearing SPECTRAMR_ variables.
    must = {
        "SPECTRAMR_DATA_ROOT",
        "PROJECT_ROOT",
        "SPECTRAMR_SUPPRESS_CLINICAL_WARNING",
        "CUBLAS_WORKSPACE_CONFIG",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
    }
    missing = must - set(names)
    assert not missing, f"env.names() missing required entries: {missing}"


def test_data_root_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without SPECTRAMR_DATA_ROOT, data_root() falls back to ./databases."""
    monkeypatch.delenv(env.SPECTRAMR_DATA_ROOT, raising=False)
    assert env.data_root() == Path("./databases")


def test_data_root_honours_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.SPECTRAMR_DATA_ROOT, "/tmp/spectramr_data")
    assert env.data_root() == Path("/tmp/spectramr_data")


def test_project_root_prefers_project_root_over_data_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env.PROJECT_ROOT, "/a")
    monkeypatch.setenv(env.SPECTRAMR_DATA_ROOT, "/b")
    assert env.project_root() == Path("/a")


def test_project_root_falls_back_to_data_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(env.PROJECT_ROOT, raising=False)
    monkeypatch.setenv(env.SPECTRAMR_DATA_ROOT, "/b")
    assert env.project_root() == Path("/b")


def test_project_root_none_when_neither_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(env.PROJECT_ROOT, raising=False)
    monkeypatch.delenv(env.SPECTRAMR_DATA_ROOT, raising=False)
    assert env.project_root() is None


def test_cache_root_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.SPECTRAMR_CACHE_ROOT, "/tmp/cache")
    assert env.cache_root() == Path("/tmp/cache")


def test_cache_root_xdg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(env.SPECTRAMR_CACHE_ROOT, raising=False)
    monkeypatch.setenv(env.XDG_CACHE_HOME, "/xdg")
    assert env.cache_root() == Path("/xdg/spectramr")


def test_suppress_clinical_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(env.SPECTRAMR_SUPPRESS_CLINICAL_WARNING, raising=False)
    assert env.suppress_clinical_warning() is False
    monkeypatch.setenv(env.SPECTRAMR_SUPPRESS_CLINICAL_WARNING, "1")
    assert env.suppress_clinical_warning() is True
    monkeypatch.setenv(env.SPECTRAMR_SUPPRESS_CLINICAL_WARNING, "true")
    assert env.suppress_clinical_warning() is True
    monkeypatch.setenv(env.SPECTRAMR_SUPPRESS_CLINICAL_WARNING, "0")
    assert env.suppress_clinical_warning() is False


def test_force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(env.FORCE_CPU, raising=False)
    assert env.force_cpu() is False
    monkeypatch.setenv(env.FORCE_CPU, "yes")
    assert env.force_cpu() is True


def test_is_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(env.WORLD_SIZE, raising=False)
    assert env.is_distributed() is False
    monkeypatch.setenv(env.WORLD_SIZE, "1")
    assert env.is_distributed() is False  # 1 GPU is not distributed
    monkeypatch.setenv(env.WORLD_SIZE, "4")
    assert env.is_distributed() is True


def test_distributed_rank_requires_world_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """``RANK`` alone is not proof of a torchrun launch.

    torchrun always exports RANK and WORLD_SIZE together. Requiring both keeps
    an unrelated environment that happens to export ``RANK`` from silencing a
    single-process run's console output.
    """
    monkeypatch.delenv(env.WORLD_SIZE, raising=False)
    monkeypatch.setenv(env.RANK, "3")
    assert env.distributed_rank() is None
    assert env.is_secondary_rank() is False


def test_distributed_rank_reads_torchrun_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.WORLD_SIZE, "4")
    monkeypatch.setenv(env.RANK, "2")
    assert env.distributed_rank() == 2
    assert env.is_secondary_rank() is True


def test_local_rank_alone_is_not_a_global_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """LOCAL_RANK numbers the processes on one node; as a global rank it made every node's
    first process rank 0. The rank rule is ``core.topology.resolve_rank`` (#2701)."""
    monkeypatch.setenv(env.WORLD_SIZE, "2")
    monkeypatch.delenv(env.RANK, raising=False)
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.setenv(env.LOCAL_RANK, "1")
    assert env.distributed_rank() is None
    assert env.is_secondary_rank() is False


def test_distributed_rank_reads_slurm_procid_beside_a_world_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env.WORLD_SIZE, "4")
    monkeypatch.delenv(env.RANK, raising=False)
    monkeypatch.setenv("SLURM_PROCID", "3")
    assert env.distributed_rank() == 3
    assert env.is_secondary_rank() is True


def test_rank_zero_is_not_secondary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rank 0 keeps its console output — otherwise a run narrates nothing."""
    monkeypatch.setenv(env.WORLD_SIZE, "4")
    monkeypatch.setenv(env.RANK, "0")
    assert env.distributed_rank() == 0
    assert env.is_secondary_rank() is False


def test_single_process_run_is_not_secondary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(env.WORLD_SIZE, raising=False)
    monkeypatch.delenv(env.RANK, raising=False)
    monkeypatch.delenv(env.LOCAL_RANK, raising=False)
    assert env.distributed_rank() is None
    assert env.is_secondary_rank() is False


def test_as_bool_recognises_canonical_truthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for v in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("TEST_VAR", v)
        assert env.as_bool("TEST_VAR") is True
    for v in ("0", "false", "off", "", "garbage"):
        monkeypatch.setenv("TEST_VAR", v)
        assert env.as_bool("TEST_VAR") is False


def test_as_int_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_INT", "42")
    assert env.as_int("TEST_INT", default=0) == 42
    monkeypatch.setenv("TEST_INT", "")
    assert env.as_int("TEST_INT", default=7) == 7
    monkeypatch.setenv("TEST_INT", "garbage")
    assert env.as_int("TEST_INT", default=9) == 9
    monkeypatch.delenv("TEST_INT", raising=False)
    assert env.as_int("TEST_INT", default=11) == 11


def test_env_example_lists_every_constant_name() -> None:
    """The .env.example file at the repo root must mention every constant.

    This catches drift between the Python module and the documented
    canonical list — if you add a new constant without updating .env.example,
    this test fails.
    """
    repo_root = Path(__file__).resolve().parents[3]
    example = repo_root / ".env.example"
    assert example.exists(), f"{example} missing"
    text = example.read_text()
    # Skip variables that are documented as auto-set by torchrun (RANK,
    # LOCAL_RANK, etc.) and that may appear only commented out for
    # reference. We require every name to APPEAR (commented or live).
    missing = [n for n in env.names() if n not in text]
    assert not missing, (
        f".env.example is missing entries for: {missing}. "
        "Add a stanza (commented if optional) for each."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Outward coverage: what the framework SETS must be declared here
# ─────────────────────────────────────────────────────────────────────────────
#
# ``test_names_lists_every_module_constant`` guards the inward direction -- a
# constant defined in ``env.py`` reaches ``names()``. Nothing guarded the
# outward one, and that asymmetry is why three variables that ``main.py``
# actively sets (TORCH_METRICS_CACHE, TORCH_CUDA_EAGER_CACHE_MANAGER, and
# TRITON_CACHE_DIR when it was added) were absent from the SSOT for as long as
# they were: adding an ``os.environ[...] = ...`` line tripped nothing.
#
# The cost is not tidiness. ``names()`` exists so smoke wrappers and SLURM
# templates can echo the resolved environment for diagnostics, so an
# unregistered variable is invisible precisely where an operator would look to
# find out where a cache went -- which is the question TRITON_CACHE_DIR exists
# to answer.
#
# The `.env.example` half of the docstring's "three lines per variable" is
# already held by ``test_env_example_lists_every_constant_name`` above; it is
# deliberately not restated here.


def _env_vars_set_by_main() -> set[str]:
    """Every ``os.environ["X"] = ...`` / ``os.environ.setdefault("X", ...)`` in main.

    Parsed from source rather than by importing: the assignments are
    unconditional module-level side effects, so importing to observe them would
    mutate this process's environment.
    """
    import re
    from pathlib import Path

    main_py = Path(__file__).resolve().parents[3] / "src" / "spectramr" / "main.py"
    source = main_py.read_text()
    return set(re.findall(r'os\.environ(?:\.setdefault\(\s*|\[)"([A-Z_][A-Z0-9_]*)"', source))


def test_main_sets_env_vars_at_all() -> None:
    """Anti-vacuity: an empty parse would make the coverage test below pass
    without checking anything, which is the exact failure it exists to catch."""
    found = _env_vars_set_by_main()
    assert len(found) >= 5, f"parsed only {found} from main.py; the regex is wrong"


def test_every_env_var_main_sets_is_declared_here() -> None:
    """``core/env.py`` is the SSOT for env-var names (canonical homes, #6)."""
    undeclared = sorted(_env_vars_set_by_main() - set(env.names()))
    assert not undeclared, (
        f"main.py sets {undeclared} but core/env.py does not declare them. Add a "
        "constant (with a comment saying what reads it), list it in __all__, and "
        "add a .env.example entry -- otherwise names() cannot report it and the "
        "smoke/SLURM environment echo silently omits it."
    )


# ─────────────────────────────────────────────────────────────────────────────
# The two gaps the tests above did not close
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. ``names()`` is derived from ``__all__``, so ``__all__`` is doing double duty
#    as export list AND registry. A constant declared in the module but omitted
#    from ``__all__`` is invisible to ``names()`` -- and therefore invisible to
#    ``test_env_example_lists_every_constant_name``, which iterates ``names()``.
#    ``SPECTRAMR_GPU_MEMORY_FRACTION`` sat in exactly that hole: constant present,
#    validating resolver present, its own docstring claiming it was "registered
#    in core/env.py", and no advertisement anywhere.
#
# 2. The outward test above covers only what ``main.py`` SETS. Nothing covered
#    what the package READS, which is the larger surface and the one an operator
#    moving to a new cluster actually needs -- twelve further names were read in
#    ``src/`` while absent from the SSOT. Three of those twelve were found only
#    after the census moved from a call-site regex to an AST walk, because they
#    reach ``os.environ`` through a named constant or an f-string prefix.


def test_all_constants_are_exported() -> None:
    """Every ``NAME = "NAME"`` constant must be listed in ``__all__``.

    Guards the omission that ``names()``-derived tests structurally cannot see.
    """
    declared = {
        name
        for name, value in vars(env).items()
        if isinstance(value, str) and name == value and not name.startswith("_")
    }
    unexported = sorted(declared - set(env.__all__))
    assert not unexported, (
        f"{unexported} are declared in core/env.py but missing from __all__. "
        "names() is derived from __all__, so an unexported constant is invisible "
        "to every consumer and to the .env.example coverage test."
    )


def _spectramr_vars_read_in_src() -> dict[str, str]:
    """Every ``SPECTRAMR_*`` string literal in the package, in any position.

    Maps name -> first file containing it, so a failure says where to look.
    Scoped to ``SPECTRAMR_*``: third-party and standard names (``TMPDIR``,
    ``CUDA_VISIBLE_DEVICES``, …) are read all over and are not ours to own.
    ``core/env.py`` itself is skipped -- it is the declaration site.

    AST, not a regex over the ``os.environ.get(...)`` call site, and that is
    not incidental. The first version of this census matched only a literal
    argument at the call site, and it missed three names for two distinct
    reasons: ``SPECTRAMR_PLUGINS`` and ``SPECTRAMR_LEDGER_STRICT`` are read through
    a named module constant (``PLUGIN_ENV_VAR``, ``STRICT_ENV``), and the
    ``SPECTRAMR_LAUNCH_*`` family is composed with an f-string from a prefix.
    A call-site regex is therefore blindest exactly where the code is tidiest.
    Do not "simplify" this back to a regex.

    The cost of matching any literal is that a prefix constant looks like a
    variable, so the caller drops trailing-underscore names -- see
    ``test_every_spectramr_var_read_in_src_is_declared_here``.
    """
    import ast
    import re

    name_re = re.compile(r"SPECTRAMR_[A-Z0-9_]*")
    src = Path(__file__).resolve().parents[3] / "src" / "spectramr"
    found: dict[str, str] = {}
    for py in sorted(src.rglob("*.py")):
        if py.name == "env.py" and py.parent.name == "core":
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and name_re.fullmatch(node.value)
            ):
                found.setdefault(node.value, str(py.relative_to(src)))
    return found


def test_src_reads_spectramr_vars_at_all() -> None:
    """Anti-vacuity twin: a broken census would make the check below vacuous."""
    found = _spectramr_vars_read_in_src()
    assert len(found) >= 18, (
        f"parsed only {len(found)} names {sorted(found)} from src/; the census "
        "is under-collecting. 21 SPECTRAMR_* literals were present as of "
        "2026-08-16 (20 variables + the SPECTRAMR_LAUNCH_ prefix)."
    )


def test_every_spectramr_var_read_in_src_is_declared_here() -> None:
    """Reading a ``SPECTRAMR_*`` name anywhere in the package must register it.

    The inverse does not hold: a declared name need not be read in ``src/``
    (``SPECTRAMR_NO_GPU_PROBE`` is read by a shell entrypoint), so this checks one
    direction only.

    One carve-out: a literal ending in ``_`` is a prefix, not a variable
    (``SPECTRAMR_LAUNCH_``, from which ``backends.py`` composes the launch->child
    provenance handoff). Those are framework-internal and must NOT be set by an
    operator, so registering them would advertise the opposite of the intent.
    The exclusion is structural rather than a name allowlist, so a future prefix
    is covered without editing this test -- and any real variable, which by
    construction does not end in ``_``, still fails the check.
    """
    read = _spectramr_vars_read_in_src()
    undeclared = sorted(n for n in set(read) - set(env.names()) if not n.endswith("_"))
    assert not undeclared, (
        "these SPECTRAMR_* variables are read in src/ but not declared in "
        f"core/env.py: {[(n, read[n]) for n in undeclared]}. Add a constant, "
        "list it in __all__, and add a .env.example stanza -- an operator moving "
        "to another machine cannot set a knob that nothing advertises."
    )
