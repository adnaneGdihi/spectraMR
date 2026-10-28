"""Planted violations for the layer-direction fitness function (NN15, #1398).

``tests/architecture/test_layer_direction.py`` was elected sole owner of import
direction, and the five ``emit`` blocks that enforced it in
``scripts/ci/check_layering.sh`` were deleted in the same change.  An election is
only safe if the winner is watched catching what the loser caught, so every rule
this table carries is planted here -- **in both shapes**, because shape is the
whole reason the election went this way:

* **top-of-file, column 0** -- the only shape the deleted greps could ever see
  (each was anchored at ``^``).
* **function-local, indented** -- structurally invisible to those greps.  The two
  real violations the election surfaced are exactly this shape, both at col 8 in
  ``application/use_cases/hpo_use_case.py`` (#1398).

The tests below therefore come in pairs: ``test_detects_*`` asserts the plant is
caught, and ``test_removing_*`` mutates the rule table and asserts the same plant
goes UNCAUGHT.  A plant no mutation kills is not a demonstration.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DETECTOR = _REPO_ROOT / "tests" / "architecture" / "test_layer_direction.py"


def _load_detector():
    """Import the detector by path, under a name pytest will not re-collect."""
    spec = importlib.util.spec_from_file_location("_layer_direction_detector", _DETECTOR)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load the layer-direction detector at {_DETECTOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


det = _load_detector()


def _plant(tmp_path: Path, files: dict[str, str]) -> Path:
    """Write a synthetic ``src/spectramr/`` tree and return its scan root."""
    src_root = tmp_path / "src" / "spectramr"
    for rel, body in files.items():
        target = src_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    return src_root


def _scan(src_root: Path) -> set[tuple[str, str]]:
    return {(v["file"], v["import"]) for v in det._scan_violations(src_root)}


def _top_level(module: str) -> str:
    return f"from {module} import thing\n"


def _function_local(module: str) -> str:
    """The col-8 shape: an import inside a method body, indented."""
    return f"class C:\n    def m(self):\n        from {module} import thing\n        return thing\n"


# (source layer, forbidden module, why the rule exists)
FORBIDDEN = [
    ("infrastructure", "spectramr.pipelines.train"),
    ("infrastructure", "spectramr.cli.app"),
    # infrastructure -> application: a rule NEITHER checker had before #1398.
    ("infrastructure", "spectramr.application.use_cases.train_use_case"),
    ("application", "spectramr.pipelines.hpo"),
    ("application", "spectramr.cli.app"),
    ("pipelines", "spectramr.cli.app"),
]

# Imports that are legal in the same layers -- a detector that fires on these is
# worse than none, because every real finding then reads as noise.
ALLOWED = [
    ("infrastructure", "spectramr.models.registry"),
    ("infrastructure", "spectramr.core.compute_device"),
    ("infrastructure", "spectramr.domain.services.services"),
    ("application", "spectramr.infrastructure.di.di_container"),
    ("application", "spectramr.models.registry"),
    ("pipelines", "spectramr.application.use_cases.train_use_case"),
    ("pipelines", "spectramr.infrastructure.di.di_container"),
]

SHAPES = {"top_level": _top_level, "function_local": _function_local}


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("layer,module", FORBIDDEN, ids=lambda v: v.replace(".", "_"))
def test_detects_forbidden_import(tmp_path, layer: str, module: str, shape: str) -> None:
    """Every forbidden edge is caught, in both the col-0 and the indented shape."""
    rel = f"{layer}/planted.py"
    found = _scan(_plant(tmp_path, {rel: SHAPES[shape](module)}))
    assert (f"spectramr/{rel}", module) in found, (
        f"{shape} import of {module} from {layer}/ was NOT detected; found={found}"
    )


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("layer,module", ALLOWED, ids=lambda v: v.replace(".", "_"))
def test_allows_inward_import(tmp_path, layer: str, module: str, shape: str) -> None:
    """Control: legal inward imports must not fire, in either shape."""
    rel = f"{layer}/planted.py"
    assert _scan(_plant(tmp_path, {rel: SHAPES[shape](module)})) == set()


@pytest.mark.parametrize("layer,module", FORBIDDEN, ids=lambda v: v.replace(".", "_"))
def test_detects_plain_import_statement(tmp_path, layer: str, module: str) -> None:
    """``import x.y`` has no symbol list; it must be caught like ``from x import y``."""
    rel = f"{layer}/planted.py"
    found = _scan(_plant(tmp_path, {rel: f"import {module}\n"}))
    assert (f"spectramr/{rel}", module) in found


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("layer,module", FORBIDDEN, ids=lambda v: v.replace(".", "_"))
def test_removing_the_rule_lets_the_plant_through(
    tmp_path, monkeypatch, layer: str, module: str, shape: str
) -> None:
    """Mutation: drop the source layer's key and the same plant must go UNCAUGHT.

    This is what makes the plant above a demonstration rather than an assertion.
    If the plant still fails with its rule deleted, something else is catching it
    and the rule is not the thing under test.
    """
    table = {k: v for k, v in det.LAYER_FORBIDDEN.items() if k != f"spectramr.{layer}"}
    monkeypatch.setattr(det, "LAYER_FORBIDDEN", table)
    rel = f"{layer}/planted.py"
    assert _scan(_plant(tmp_path, {rel: SHAPES[shape](module)})) == set()


@pytest.mark.parametrize("layer,module", FORBIDDEN, ids=lambda v: v.replace(".", "_"))
def test_narrowing_the_rule_lets_the_plant_through(
    tmp_path, monkeypatch, layer: str, module: str
) -> None:
    """Mutation: keep the key, drop only THIS forbidden target.

    Distinguishes "the layer is audited" from "this specific edge is audited" --
    without it, one over-broad entry would score every edge in the layer green.
    """
    key = f"spectramr.{layer}"
    target = ".".join(module.split(".")[:2])
    table = dict(det.LAYER_FORBIDDEN)
    table[key] = [f for f in table[key] if f != target]
    monkeypatch.setattr(det, "LAYER_FORBIDDEN", table)
    rel = f"{layer}/planted.py"
    assert _scan(_plant(tmp_path, {rel: _top_level(module)})) == set()


def test_outer_layers_are_actually_in_the_table() -> None:
    """The election is void if the winner never gained the loser's source layers."""
    missing = [
        layer
        for layer in ("spectramr.infrastructure", "spectramr.application", "spectramr.pipelines")
        if layer not in det.LAYER_FORBIDDEN
    ]
    assert not missing, (
        f"{missing} absent from LAYER_FORBIDDEN, but the matching emit blocks were "
        "deleted from scripts/ci/check_layering.sh -- those rules are now unenforced."
    )


def test_the_real_hpo_violation_is_the_function_local_shape() -> None:
    """Pin the fact that motivated the election: the live hits are indented.

    If these ever move to column 0 the deleted greps would have caught them, and
    this test should be re-read rather than re-baselined.
    """
    src = _REPO_ROOT / "src" / "spectramr" / "application" / "use_cases" / "hpo_use_case.py"
    if not src.exists():  # pragma: no cover - file renamed/removed
        pytest.skip(f"{src} no longer exists")
    offenders = [
        line
        for line in src.read_text(encoding="utf-8").splitlines()
        if "spectramr.pipelines" in line and ("import " in line)
    ]
    assert offenders, "the recorded #1398 imports vanished -- un-baseline them"
    assert all(line.startswith(" ") for line in offenders), (
        f"a column-0 spectramr.pipelines import appeared in {src.name}: {offenders}"
    )


# ---------------------------------------------------------------------------
# Non-package repo trees (D01#16)
#
# `_scan_nonpackage_imports` is the SECOND rule this file owns, and until this
# section it had never been watched fail: it took no scan root, so it could not
# be pointed at a planted tree, and it looked only for the literal name
# `scripts`.  It recorded zero violations for its whole life while two real ones
# sat at `cli/app.py:474,487` -- function-local, importing `tools`.
# ---------------------------------------------------------------------------


def _scan_np(src_root: Path) -> set[tuple[str, str]]:
    return {(v["file"], v["import"]) for v in det._scan_nonpackage_imports(src_root)}


# One entry per name in NONPACKAGE_ROOTS. Parametrised off the frozenset itself,
# so adding a name without a plant is impossible.
NONPACKAGE_SHAPES = {
    "top_level": lambda m: _top_level(m),
    "function_local": lambda m: _function_local(m),
    "plain_import": lambda m: f"import {m}\n",
    "plain_import_local": lambda m: f"def f():\n    import {m}\n    return {m}\n",
}


@pytest.mark.parametrize("shape", list(NONPACKAGE_SHAPES))
@pytest.mark.parametrize("root", sorted(det.NONPACKAGE_ROOTS))
def test_detects_nonpackage_import(tmp_path, root: str, shape: str) -> None:
    """Every non-wheel tree is caught, in all four import shapes."""
    module = f"{root}.some_module"
    found = _scan_np(_plant(tmp_path, {"cli/planted.py": NONPACKAGE_SHAPES[shape](module)}))
    assert ("spectramr/cli/planted.py", module) in found, (
        f"{shape} import of {module} was NOT detected; found={found}"
    )


@pytest.mark.parametrize("root", sorted(det.NONPACKAGE_ROOTS))
def test_the_bare_root_name_is_caught_not_only_a_submodule(tmp_path, root: str) -> None:
    """``import tools`` with no dot must fire; the match is on the first segment."""
    found = _scan_np(_plant(tmp_path, {"cli/planted.py": f"import {root}\n"}))
    assert ("spectramr/cli/planted.py", root) in found


@pytest.mark.parametrize("shape", list(NONPACKAGE_SHAPES))
@pytest.mark.parametrize("root", sorted(det.NONPACKAGE_ROOTS))
def test_the_in_package_namesake_is_not_a_violation(tmp_path, root: str, shape: str) -> None:
    """Negative control, and the reason the match is first-segment-only.

    ``src/spectramr/tools/`` genuinely exists, and ``spectramr.tools`` is a perfectly
    legal import. A substring or ``endswith`` match would conflate the two and
    make every in-package import of it read as a wheel-breaking violation.
    """
    module = f"spectramr.{root}.some_module"
    body = NONPACKAGE_SHAPES[shape](module)
    assert _scan_np(_plant(tmp_path, {"cli/planted.py": body})) == set()


@pytest.mark.parametrize("root", sorted(det.NONPACKAGE_ROOTS))
def test_removing_the_root_lets_the_plant_through(tmp_path, monkeypatch, root: str) -> None:
    """Mutation: drop the name from the set and the same plant must go UNCAUGHT.

    Without this, one over-broad name would score every other root green.
    """
    monkeypatch.setattr(det, "NONPACKAGE_ROOTS", det.NONPACKAGE_ROOTS - {root})
    body = _function_local(f"{root}.some_module")
    assert _scan_np(_plant(tmp_path, {"cli/planted.py": body})) == set()


RELATIVE_SHAPES = {
    "relative_top_level": lambda r: f"from .{r} import thing\n",
    "relative_function_local": lambda r: (
        f"def f():\n    from .{r} import thing\n    return thing\n"
    ),
    "relative_parent": lambda r: f"from ..{r} import thing\n",
}


@pytest.mark.parametrize("shape", list(RELATIVE_SHAPES))
@pytest.mark.parametrize("root", sorted(det.NONPACKAGE_ROOTS))
def test_a_relative_import_of_the_namesake_is_not_a_violation(
    tmp_path, root: str, shape: str
) -> None:
    """Negative control: ``ast`` drops the dots, so ``.tools`` arrives as ``tools``.

    ``ImportFrom.module`` is the bare name for a relative import, which is
    exactly the string the first-segment match looks for -- so the naive scanner
    condemned ``from .tools import x``, a legal in-package import of the real
    ``src/spectramr/tools/``. A relative import cannot reach a repo-root tree by
    construction, so ``absolute_only`` skips them.
    """
    body = RELATIVE_SHAPES[shape](root)
    assert _scan_np(_plant(tmp_path, {"cli/pkg/planted.py": body})) == set()
    # Discrimination: the plant is not vacuously green. The same file written
    # with the absolute spelling of the same name IS caught, so the scanner does
    # reach this path and the clean result is the ``level`` skip, not a no-op.
    absolute = body.replace(f"from ..{root} ", f"from {root} ").replace(
        f"from .{root} ", f"from {root} "
    )
    assert _scan_np(_plant(tmp_path, {"cli/pkg/planted.py": absolute})) == {
        ("spectramr/cli/pkg/planted.py", root)
    }


def test_dropping_the_level_skip_makes_the_relative_plant_fire(tmp_path, monkeypatch) -> None:
    """Mutation on the fix itself: without ``absolute_only`` the control goes red.

    ``_collect_imports`` defaults to keeping relative nodes because
    ``_scan_violations`` needs them harmless; patching the call site back to the
    default is the one-line regression this guards.
    """
    real = det._collect_imports
    monkeypatch.setattr(det, "_collect_imports", lambda tree, **kw: real(tree))
    body = "from .tools import thing\n"
    assert _scan_np(_plant(tmp_path, {"cli/pkg/planted.py": body})) == {
        ("spectramr/cli/pkg/planted.py", "tools")
    }


def test_a_name_that_merely_looks_like_a_root_is_not_caught(tmp_path) -> None:
    """``toolsuite`` shares a prefix with ``tools`` and is a third-party name."""
    lookalikes = "\n".join(f"import {r}suite" for r in sorted(det.NONPACKAGE_ROOTS))
    assert _scan_np(_plant(tmp_path, {"cli/planted.py": lookalikes + "\n"})) == set()


def test_the_scan_reaches_every_layer_not_only_cli(tmp_path) -> None:
    """The rule is tree-wide: it keys on no layer, unlike LAYER_FORBIDDEN.

    A file in a directory absent from LAYER_FORBIDDEN (``spectramr/core``'s
    siblings, plugin trees) must still be scanned, or the rule silently applies
    to a subset of the package.
    """
    planted = {f"{d}/planted.py": _top_level("scripts.ci.thing") for d in ("core", "unlisted")}
    found = _scan_np(_plant(tmp_path, planted))
    assert found == {
        ("spectramr/core/planted.py", "scripts.ci.thing"),
        ("spectramr/unlisted/planted.py", "scripts.ci.thing"),
    }


def test_the_recorded_app_py_sites_are_the_function_local_shape() -> None:
    """Pin what makes this row a detector row and not a code row.

    Both live hits are indented, so no ``^``-anchored grep in
    ``check_layering.sh`` could ever have seen them -- and the AST table could
    not either, because it is keyed entirely on ``spectramr.*`` prefixes and
    ``tools`` is not one.
    """
    src = _REPO_ROOT / "src" / "spectramr" / "cli" / "app.py"
    if not src.exists():  # pragma: no cover - file renamed/removed
        pytest.skip(f"{src} no longer exists")
    offenders = [
        line
        for line in src.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith(("from tools.", "import tools"))
    ]
    assert offenders, "the recorded tools imports vanished -- un-baseline them"
    assert all(line.startswith(" ") for line in offenders), (
        f"a column-0 tools import appeared in {src.name}: {offenders}"
    )


def test_every_recorded_nonpackage_entry_names_a_root_in_the_set() -> None:
    """An allowlist entry for a root the scanner no longer knows exempts nothing.

    The stale-entry gate in the detector catches a vanished *import*; this
    catches a vanished *rule*, which would leave the entry looking like coverage.
    """
    _, nonpackage_known = det._load_allowlist()
    orphans = [e for e in nonpackage_known if e["import"].split(".")[0] not in det.NONPACKAGE_ROOTS]
    assert not orphans, (
        f"{orphans} are recorded as exemptions but their root is not in "
        "NONPACKAGE_ROOTS, so nothing would flag them if the exemption were removed."
    )


def test_an_unrecognised_allowlist_key_is_named_not_silently_dropped(tmp_path, monkeypatch):
    """A half-applied key rename must say so.

    ``data.get(key, [])`` turns a typo into an empty allowlist. That reddens the
    gate -- loud -- but it reports "N NEW violations" and names the wrong cause,
    sending the reader to the code instead of to the one-word typo.
    """
    import json

    bad = tmp_path / "_known_violations.json"
    bad.write_text(json.dumps({"layer_direction": [], "scripts_imports": []}))
    monkeypatch.setattr(det, "VIOLATIONS_FILE", bad)
    with pytest.raises(AssertionError, match="unrecognised top-level key"):
        det._load_allowlist()


# ---------------------------------------------------------------------------
# #2340: data/ may import the physics SSOT, and nothing else from infrastructure.
# Each carve-out plant comes with its mutation: with LAYER_EXEMPT emptied, the
# same physics import must be caught again.
# ---------------------------------------------------------------------------

DATA_PHYSICS = [
    "spectramr.infrastructure.physics.fft_ops",
    "spectramr.infrastructure.physics.coil_sensitivity",
]
DATA_STILL_FORBIDDEN = [
    "spectramr.infrastructure.errors",
    "spectramr.infrastructure.io.adapters",
    "spectramr.infrastructure.training.utils.kspace_masks",
    # A sibling whose name merely STARTS with "physics" is not the SSOT.
    "spectramr.infrastructure.physics_extras",
]


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("module", DATA_PHYSICS)
def test_data_may_import_the_physics_ssot(tmp_path, module: str, shape: str) -> None:
    assert _scan(_plant(tmp_path, {"data/planted.py": SHAPES[shape](module)})) == set()


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("module", DATA_STILL_FORBIDDEN)
def test_the_carve_out_does_not_widen_past_physics(tmp_path, module: str, shape: str) -> None:
    found = _scan(_plant(tmp_path, {"data/planted.py": SHAPES[shape](module)}))
    assert ("spectramr/data/planted.py", module) in found


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("module", DATA_PHYSICS)
def test_removing_the_carve_out_catches_the_physics_import_again(
    tmp_path, monkeypatch, module: str, shape: str
) -> None:
    monkeypatch.setattr(det, "LAYER_EXEMPT", {})
    found = _scan(_plant(tmp_path, {"data/planted.py": SHAPES[shape](module)}))
    assert ("spectramr/data/planted.py", module) in found


@pytest.mark.parametrize("layer", ["core", "domain"])
def test_the_carve_out_is_scoped_to_data(tmp_path, layer: str) -> None:
    """core/ and domain/ keep forbidding all of infrastructure pending #1398."""
    module = "spectramr.infrastructure.physics.fft_ops"
    found = _scan(_plant(tmp_path, {f"{layer}/planted.py": _top_level(module)}))
    assert (f"spectramr/{layer}/planted.py", module) in found


def test_the_key_guard_accepts_the_real_file() -> None:
    """Discrimination: the guard above must not fire on the committed allowlist."""
    layer_known, nonpackage_known = det._load_allowlist()
    # Assert on ``layer_direction`` only. ``nonpackage_imports`` is one entry of
    # live debt (#1492); pinning it non-empty would turn paying that debt off
    # into a red test whose message names the wrong thing -- a gate that
    # punishes the fix is the facade shape non-negotiable 16 is about.
    assert layer_known, "the real allowlist read back empty"
    assert isinstance(nonpackage_known, list)
