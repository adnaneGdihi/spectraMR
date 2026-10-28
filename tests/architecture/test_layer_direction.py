"""
TASK III.2 – Layer-direction fitness function.

Enforces CLAUDE.md "Architecture (load-bearing)" inward-only dep rule:

    cli/ → pipelines/ → application/ → infrastructure/ → models/, domain/ → core/, config/

Lower layers must never import from higher ones.  Also: nothing under
src/spectramr/ may import from a repo-root tree that is not in the wheel
(``scripts``, ``tools``, ``scratch``, ``runners``, ``tests``) -- see
``NONPACKAGE_ROOTS``.

Gate test (fast, always runs):
    Collect violations, subtract the pre-existing allowlist from
    ``_known_violations.json``, fail if any NEW violations exist.

Cleanup tracker (slow, opt-in with -m slow):
    Fails as long as known violations remain (debt tracker).
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
SRC_ROOT = REPO_ROOT / "src" / "spectramr"
VIOLATIONS_FILE = Path(__file__).parent / "_known_violations.json"

# Repo-root trees that are NOT part of the installed distribution
# (``pyproject.toml`` ships ``packages = ["src/spectramr"]`` and nothing else), so
# an import of one from ``src/`` is broken for every wheel install.  A literal
# set, deliberately, rather than one derived from ``ls REPO_ROOT``: the
# ``public`` branch excludes several of these directories, and a disk-derived
# rule would enforce a different policy per checkout.  Each name is justified:
#
#   scripts   non-negotiable 5, verbatim: "src/ never imports from scripts/"
#   scratch   CLAUDE.md repository map: "not production surface"
#   runners   CLAUDE.md repository map: "not production surface"
#   tools     repo-root dev tree, absent from ``packages``
#   tests     test code is never a production dependency
#
# Matching is on the FIRST dotted segment only, so the in-package
# ``spectramr.tools`` package is untouched -- there is a negative control for
# exactly that in tests/unit/architecture/test_layer_direction_detector.py.
NONPACKAGE_ROOTS: frozenset[str] = frozenset({"scripts", "scratch", "runners", "tools", "tests"})

# ---------------------------------------------------------------------------
# Layer rules
# ---------------------------------------------------------------------------

# Maps a module-prefix to the set of prefixes it is FORBIDDEN from importing.
# Ordered by specificity — more-specific layers first.
LAYER_FORBIDDEN: dict[str, list[str]] = {
    "spectramr.core": [
        "spectramr.infrastructure",
        "spectramr.application",
        "spectramr.pipelines",
        "spectramr.cli",
    ],
    "spectramr.config": [
        "spectramr.infrastructure",
        "spectramr.application",
        "spectramr.pipelines",
        "spectramr.cli",
    ],
    "spectramr.domain": [
        "spectramr.infrastructure",
        "spectramr.application",
        "spectramr.pipelines",
        "spectramr.cli",
    ],
    # models/ is allowed to reach infrastructure but NOT application/pipelines/cli
    "spectramr.models": [
        "spectramr.application",
        "spectramr.pipelines",
        "spectramr.cli",
    ],
    # data/ sits logically between domain and infrastructure; it may not reach
    # up into infrastructure, except the physics SSOT (see LAYER_EXEMPT below).
    "spectramr.data": [
        "spectramr.application",
        "spectramr.pipelines",
        "spectramr.cli",
        "spectramr.infrastructure",
    ],
    # --- outer layers -------------------------------------------------------
    # Elected sole owner of import direction (#1398).  These three keys mirror
    # the ``emit`` blocks deleted from scripts/ci/check_layering.sh: every grep
    # there was anchored at ``^``, so a function-local import was structurally
    # invisible to it.  Both violations this table records are at col 8.
    "spectramr.infrastructure": [
        # infrastructure -> application closes a hole NEITHER checker had:
        # the shell only ever tested infrastructure -> pipelines|cli.  Measured
        # 0 violations across src/spectramr/ on 2026-08-22, so it lands as a pure
        # ratchet with no baselined debt.
        "spectramr.application",
        "spectramr.pipelines",
        "spectramr.cli",
    ],
    "spectramr.application": [
        "spectramr.pipelines",
        "spectramr.cli",
    ],
    "spectramr.pipelines": [
        "spectramr.cli",
    ],
}

#: Carve-outs from a forbidden prefix, by source layer. ``infrastructure.physics``
#: is the physics SSOT (``.claude/rules/physics.md``); a data transform that
#: simulates noise, estimates coil maps or applies a forward operator needs it,
#: and no port for it exists. The rest of ``infrastructure`` stays forbidden.
#: ``core``/``domain`` are left as they are pending #1398.
LAYER_EXEMPT: dict[str, list[str]] = {
    "spectramr.data": ["spectramr.infrastructure.physics"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_layer(py_file: Path, src_root: Path = SRC_ROOT) -> str | None:
    """Return the layer prefix for a .py file, or None if not in a tracked layer.

    ``src_root`` is a parameter so the detector can be pointed at a planted tree
    (non-negotiable 15): a gate nobody has watched fail is not a gate, and this
    one cannot be watched fail while its scan root is a module constant.
    """
    rel = str(py_file.relative_to(src_root.parent))
    mod = rel.replace("/", ".").removesuffix(".py")
    for layer in LAYER_FORBIDDEN:
        if mod == layer or mod.startswith(layer + "."):
            return layer
    return None


def _collect_imports(tree: ast.AST, *, absolute_only: bool = False) -> list[str]:
    """Return all module names imported in an AST.

    ``ast.ImportFrom.module`` drops the leading dots, so ``from .tools import x``
    yields the bare name ``"tools"`` -- indistinguishable from the repo-root tree
    of the same name.  ``absolute_only`` skips ``level > 0`` nodes for callers
    that match on a first segment; a relative import cannot resolve outside the
    package by construction, so this cannot hide a real violation.
    """
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module and not (absolute_only and node.level):
                names.append(node.module)
    return names


def _scan_violations(src_root: Path = SRC_ROOT) -> list[dict[str, str]]:
    """Walk ``src_root`` and return all layer-direction violations found."""
    violations: list[dict[str, str]] = []
    for py_file in sorted(src_root.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        layer = _file_layer(py_file, src_root)
        if layer is None:
            continue
        forbidden = LAYER_FORBIDDEN[layer]
        exempt = LAYER_EXEMPT.get(layer, [])
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for imp in _collect_imports(tree):
            if any(imp == e or imp.startswith(e + ".") for e in exempt):
                continue
            for f in forbidden:
                if imp == f or imp.startswith(f + "."):
                    violations.append(
                        {
                            "file": str(py_file.relative_to(src_root.parent)),
                            "layer": layer,
                            "import": imp,
                            "forbidden_layer": f,
                        }
                    )
    return violations


def _scan_nonpackage_imports(src_root: Path = SRC_ROOT) -> list[dict[str, str]]:
    """Return files under ``src_root`` importing a repo tree outside the wheel.

    ``src_root`` is a parameter for the same reason ``_file_layer`` takes one
    (non-negotiable 15): a detector whose scan root is a module constant cannot
    be pointed at a planted tree, and so can never be watched fail.  This one
    had recorded zero violations since it was written while two real ones sat
    at ``cli/app.py`` col 8 -- it only ever looked for the name ``scripts``.
    """
    violations: list[dict[str, str]] = []
    pattern = re.compile(r"\b(" + "|".join(sorted(NONPACKAGE_ROOTS)) + r")\b")
    for py_file in sorted(src_root.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
        except OSError:
            continue
        if not pattern.search(source):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for imp in _collect_imports(tree, absolute_only=True):
            root = imp.split(".")[0]
            if root in NONPACKAGE_ROOTS:
                violations.append(
                    {
                        "file": str(py_file.relative_to(src_root.parent)),
                        "import": imp,
                        "root": root,
                    }
                )
    return violations


# Every top-level key the allowlist file is allowed to carry.  ``.get(key, [])``
# turns a typo into an empty allowlist, which reddens the gate rather than
# silencing it -- loud, but it names the wrong thing.  Reject the unknown key
# instead, so a half-applied rename says what actually happened.
_ALLOWLIST_KEYS = frozenset(
    {
        "_comment",
        "layer_direction",
        "nonpackage_imports",
        "data_io",
        "training_loop",
        "stray_prints",
        "raw_torch_fft",
    }
)


def _load_allowlist() -> tuple[list[dict], list[dict]]:
    """Return (layer_known, nonpackage_known) from _known_violations.json."""
    if not VIOLATIONS_FILE.exists():
        return [], []
    data = json.loads(VIOLATIONS_FILE.read_text())
    unknown = sorted(set(data) - _ALLOWLIST_KEYS)
    if unknown:
        raise AssertionError(
            f"{VIOLATIONS_FILE.name} carries unrecognised top-level key(s) "
            f"{unknown}; add them to _ALLOWLIST_KEYS or fix the spelling. "
            "An unread key is an allowlist entry that exempts nothing."
        )
    return data.get("layer_direction", []), data.get("nonpackage_imports", [])


def _violation_key(v: dict) -> tuple[str, ...]:
    return (v["file"], v.get("layer", ""), v["import"])


def _new_only(found: list[dict], known: list[dict]) -> list[dict]:
    """Return violations that are NOT in the known set."""
    known_keys = {_violation_key(k) for k in known}
    return [v for v in found if _violation_key(v) not in known_keys]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.architecture
def test_no_new_layer_direction_violations() -> None:
    """Gate: fail on ANY violation not recorded in the known-violations allowlist.

    This test is fast (pure AST, no imports executed) and runs on every
    CI push.  To add a new violation to the allowlist, add it to
    tests/architecture/_known_violations.json and include a comment
    explaining why it can't be fixed immediately.
    """
    all_layer = _scan_violations()
    all_nonpackage = _scan_nonpackage_imports()
    layer_known, nonpackage_known = _load_allowlist()

    new_layer = _new_only(all_layer, layer_known)
    new_nonpackage = _new_only(
        [{"file": v["file"], "import": v["import"], "layer": ""} for v in all_nonpackage],
        [{"file": v["file"], "import": v["import"], "layer": ""} for v in nonpackage_known],
    )

    messages: list[str] = []
    if new_layer:
        messages.append("NEW layer-direction violations (not in _known_violations.json):")
        for v in new_layer:
            messages.append(f"  {v['file']}  [{v['layer']}] imports {v['import']}")
    if new_nonpackage:
        messages.append(
            "NEW imports of a non-wheel repo tree from src/spectramr/ "
            "(not in _known_violations.json). These break every wheel install:"
        )
        for v in new_nonpackage:
            messages.append(f"  {v['file']} imports {v['import']}")

    assert not messages, "\n".join(messages)


@pytest.mark.architecture
def test_layer_allowlist_has_no_stale_entries() -> None:
    """Hard gate: every recorded exemption must still describe a real import.

    A stale entry (deleted file, removed import) makes the allowlist lie about
    the codebase and silently re-exempts the path if the import comes back.
    Actionable today, so unlike the debt report below this stays a hard failure
    (#629).
    """
    found_layer = {_violation_key(v) for v in _scan_violations()}
    found_nonpackage = {(v["file"], v["import"]) for v in _scan_nonpackage_imports()}
    layer_known, nonpackage_known = _load_allowlist()

    stale_layer = [v for v in layer_known if _violation_key(v) not in found_layer]
    stale_nonpackage = [
        v for v in nonpackage_known if (v["file"], v["import"]) not in found_nonpackage
    ]

    messages: list[str] = []
    if stale_layer:
        messages.append(
            f"{len(stale_layer)} stale layer_direction entries in "
            "_known_violations.json (import no longer present — remove them):"
        )
        messages += [f"  {v['file']} imports {v['import']}" for v in stale_layer]
    if stale_nonpackage:
        messages.append(
            f"{len(stale_nonpackage)} stale nonpackage_imports entries in "
            "_known_violations.json (import no longer present — remove them):"
        )
        messages += [f"  {v['file']} imports {v['import']}" for v in stale_nonpackage]

    assert not messages, "\n".join(messages)


@pytest.mark.architecture
@pytest.mark.slow
@pytest.mark.debt_tracker
@pytest.mark.xfail(
    strict=False,
    reason="Debt report: red until every recorded layer-direction violation is "
    "fixed. XPASS means the allowlist can be cleared — delete this marker.",
)
def test_no_known_layer_violations_remain() -> None:
    """Debt report: how many recorded layer-direction violations are left.

    Run ``pytest -m debt_tracker -rx`` to read them. An XPASS means the
    _known_violations.json allowlist can be cleared.
    """
    all_layer = _scan_violations()
    all_nonpackage = _scan_nonpackage_imports()
    layer_known, nonpackage_known = _load_allowlist()

    still_present = [
        v for v in layer_known if any(_violation_key(v) == _violation_key(f) for f in all_layer)
    ]
    nonpackage_still = [
        v
        for v in nonpackage_known
        if any(v["file"] == f["file"] and v["import"] == f["import"] for f in all_nonpackage)
    ]

    messages: list[str] = []
    if still_present:
        messages.append(
            f"{len(still_present)} known layer violations still present "
            "(remove from _known_violations.json once fixed):"
        )
        for v in still_present:
            messages.append(f"  {v['file']}  [{v['layer']}] imports {v['import']}")
    if nonpackage_still:
        messages.append(f"{len(nonpackage_still)} known non-wheel-tree imports still present:")
        for v in nonpackage_still:
            messages.append(f"  {v['file']} imports {v['import']}")

    assert not messages, "\n".join(messages)
