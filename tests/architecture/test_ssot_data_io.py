"""
TASK III.2 – Data-IO SSOT fitness function.

Enforces CLAUDE.md pitfall #11:

    No ``torch.load``, ``h5py.File``, ``nib.load`` / ``nibabel.load``,
    ``pickle.load`` or safetensors ``load_file`` / ``safe_open`` outside
    ``spectramr/data/`` and the checkpoint codecs
    (``spectramr/infrastructure/checkpointing/codecs/``).

All file-to-tensor loading must go through the DataPipelineDirector and live
under src/spectramr/data/.  Higher layers consume datasets via the director;
they never open files directly.

Gate test (fast, always runs):
    Fail on any violation not in _known_violations.json["data_io"].

Cleanup tracker (slow, opt-in with -m slow):
    Fails until all known violations are fixed.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
SRC_ROOT = REPO_ROOT / "src" / "spectramr"
VIOLATIONS_FILE = Path(__file__).parent / "_known_violations.json"

DATA_SSOT_PREFIX = "spectramr/data/"

#: The one sanctioned home of checkpoint IO (gate option (b)): every checkpoint file is
#: read and written by the codecs there, so this prefix is the whole exemption.
CHECKPOINT_IO_PREFIX = "spectramr/infrastructure/checkpointing/codecs/"

#: ``(module, attribute) -> reported call name``. The key is the name **bound in
#: the scanned module**, so both ``import nibabel`` and ``import nibabel as nib``
#: resolve; the value is the stable string recorded in the baseline.
DOTTED_CALLS: dict[tuple[str, str], str] = {
    ("torch", "load"): "torch.load",
    ("h5py", "File"): "h5py.File",
    ("nib", "load"): "nib.load",
    ("nibabel", "load"): "nib.load",
    ("pickle", "load"): "pickle.load",
    ("safetensors", "safe_open"): "safetensors.safe_open",
    ("safetensors.torch", "load_file"): "safetensors.load_file",
    ("safetensors.numpy", "load_file"): "safetensors.load_file",
}

#: ``from <module> import <name>`` bindings that mean the same call. The regex
#: oracle this replaced could not see these at all.
FROM_IMPORTS: dict[str, dict[str, str]] = {
    "torch": {"load": "torch.load"},
    "h5py": {"File": "h5py.File"},
    "nib": {"load": "nib.load"},
    "nibabel": {"load": "nib.load"},
    "pickle": {"load": "pickle.load"},
    "safetensors.torch": {"load_file": "safetensors.load_file"},
    "safetensors.numpy": {"load_file": "safetensors.load_file"},
    "safetensors": {"safe_open": "safetensors.safe_open"},
}

#: Modules whose ``import <mod> as <alias>`` binding must be followed. Keyed on
#: the *real* module name; ``nib`` is listed because ``DOTTED_CALLS`` also
#: enumerates it as a head, so an ``import nib as x`` would otherwise escape.
ALIASABLE_MODULES = frozenset(
    {
        "torch", "h5py", "nib", "nibabel", "pickle",
        "safetensors", "safetensors.torch", "safetensors.numpy",
    }
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_bindings(tree: ast.AST) -> dict[str, str]:
    """Every name an ``import`` statement binds -> the module it really names.

    Records *all* imports, not only the guarded ones, because the map has two
    jobs: resolving ``import torch as th`` to ``torch``, and **shadowing** —
    ``import json as torch`` rebinds the head, so ``torch.load(p)`` there is
    ``json.load`` and must not be reported. Recording only guarded modules made
    the shadowed case fall through to the literal head and fire; that was caught
    by the negative plant below, not by review.
    """
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    out[alias.asname] = alias.name
                else:
                    # ``import torch.utils.checkpoint`` binds the TOP package
                    # ``torch``, not the dotted name. Recording the dotted name
                    # made ``torch`` look shadowed by an unguarded module and
                    # silently dropped the real ``torch.load`` at
                    # ``training/builders/model_builder.py:141``. The corpus
                    # scan caught that, not the plants — hence the plant below.
                    top = alias.name.split(".")[0]
                    out[top] = top
        elif isinstance(node, ast.ImportFrom) and node.module:
            # ``from safetensors import torch as st`` binds a guarded submodule.
            for alias in node.names:
                full = f"{node.module}.{alias.name}"
                if full in ALIASABLE_MODULES:
                    out[alias.asname or alias.name] = full
    return out


def _dotted_path(node: ast.AST) -> list[str] | None:
    """``torch.load`` -> ``["torch", "load"]``; ``None`` if not a plain dotted name."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return list(reversed(parts))


def _aliased_names(tree: ast.AST) -> dict[str, str]:
    """Local names bound by ``from <forbidden> import <name>``."""
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in FROM_IMPORTS:
            table = FROM_IMPORTS[node.module]
            for alias in node.names:
                if alias.name in table:
                    bound[alias.asname or alias.name] = table[alias.name]
    return bound


def find_io_calls(source: str) -> set[str]:
    r"""Every forbidden raw-IO **call** in ``source``, as reported call names.

    AST, not text. The predecessor was ``re.compile(r"\btorch\.load\b")`` over
    the raw file, which cannot tell a call from prose — and it did not, twice:

    * ``pipelines/infer.py:539`` carries a docstring reading *"pipeline layer
      never calls ``h5py.File`` / ``nib.load`` / ``np.load``"*. The oracle read a
      sentence **asserting compliance** and recorded the file as breaking the
      rule, so two of that file's three baseline entries described nothing.
    * ``core/module_utils.py:199`` says *"payload: Whatever ``torch.load``
      returned."* — unbaselined, so it **failed the gate outright**.

    Five of the 44 text matches across ``src/spectramr/`` were of this kind.

    Module aliases are followed too (``import torch as th; th.load(p)``). That
    hole was free to close: the only nonstandard aliases of the four guarded
    modules anywhere in ``src/spectramr/`` are two ``import torch as _torch``
    (``meta_evaluation/figures.py:895``, ``mixins/metrics_mixin.py:101``), and
    neither reaches ``.load`` — so it adds 0 baseline entries today and shuts
    the door before the first one does.

    **Not detected, deliberately:** binding the function object and calling it
    later (``fn = torch.load; fn(p)``). That needs dataflow, and the one
    non-call reference in the tree today is a version probe
    (``getattr(torch.load, "__code__", None)`` in ``shared/utils/safe_io.py``)
    whose file is baselined for the real call beneath it. Flagging a symbol
    reference as file IO would teach the wrong rule; this gap is recorded here
    rather than left to be inferred from a green run.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    aliased = _aliased_names(tree)
    bindings = _module_bindings(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted_path(node.func)
        if dotted is not None and len(dotted) >= 2:
            head, *middle, attr = dotted
            if head in bindings:
                real = bindings[head]
                if real not in ALIASABLE_MODULES:
                    continue  # the head is shadowed by an unguarded module
            else:
                # Never imported in this module (re-export, injected name):
                # fall back to the literal head rather than miss the call.
                real = head
            # ``safetensors.torch.load_file`` names its module in the middle segments.
            module = ".".join([real, *middle])
            if (module, attr) in DOTTED_CALLS:
                found.add(DOTTED_CALLS[(module, attr)])
        elif isinstance(node.func, ast.Name) and node.func.id in aliased:
            found.add(aliased[node.func.id])
    return found


def _scan_violations() -> list[dict[str, str]]:
    """Return files outside spectramr/data/ that make raw IO calls."""
    violations: list[dict[str, str]] = []
    for py_file in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        rel = str(py_file.relative_to(REPO_ROOT / "src"))
        if rel.startswith((DATA_SSOT_PREFIX, CHECKPOINT_IO_PREFIX)):
            continue  # data/ and the checkpoint codecs are the canonical IO homes
        try:
            source = py_file.read_text(encoding="utf-8")
        except OSError:
            continue
        for call_name in sorted(find_io_calls(source)):
            violations.append({"file": rel, "call": call_name})
    return violations


def _load_known() -> list[dict[str, str]]:
    if not VIOLATIONS_FILE.exists():
        return []
    data = json.loads(VIOLATIONS_FILE.read_text())
    return data.get("data_io", [])


def _vkey(v: dict) -> tuple[str, str]:
    return (v["file"], v["call"])


def _new_only(found: list[dict], known: list[dict]) -> list[dict]:
    known_keys = {_vkey(k) for k in known}
    return [v for v in found if _vkey(v) not in known_keys]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.architecture
def test_no_new_data_io_outside_data_layer() -> None:
    """Gate: fail if any NEW raw IO call appears outside spectramr/data/.

    Pre-existing violations are tracked in
    ``tests/architecture/_known_violations.json["data_io"]``.  New code
    that needs to open a file must either:
      (a) add a Dataset class under ``spectramr/data/datasets/`` and route
          through DataPipelineDirector, or
      (b) if it is truly infrastructure-level IO (checkpoints, adapters),
          add it to the allowlist with a justification comment.
    """
    found = _scan_violations()
    known = _load_known()
    new = _new_only(found, known)
    if new:
        msg = (
            "NEW raw IO calls outside spectramr/data/ "
            '(not in _known_violations.json["data_io"]):\n'
            + "\n".join(f"  {v['file']}  uses  {v['call']}" for v in new)
            + "\n\nRoute file loading through DataPipelineDirector "
            "(spectramr.infrastructure.builders.directors.data_pipeline_director)."
        )
        pytest.fail(msg)


@pytest.mark.architecture
def test_data_io_allowlist_has_no_stale_entries() -> None:
    """Hard gate: every recorded exemption must still describe a real IO call.

    A stale entry (deleted file, IO routed through the director since) makes the
    allowlist lie about the codebase and silently re-exempts the path if raw IO
    reappears there. Actionable today, so unlike the debt report below this stays
    a hard failure (#629).
    """
    found = {_vkey(v) for v in _scan_violations()}
    stale = [v for v in _load_known() if _vkey(v) not in found]
    assert not stale, (
        f"{len(stale)} stale data_io entries in _known_violations.json "
        "(call no longer present — remove them):\n"
        + "\n".join(f"  {v['file']}  {v['call']}" for v in stale)
    )


@pytest.mark.architecture
@pytest.mark.slow
@pytest.mark.debt_tracker
@pytest.mark.xfail(
    strict=False,
    reason="Debt report: red until every recorded data-IO violation is routed "
    "through DataPipelineDirector. XPASS means the debt is paid — delete this marker.",
)
def test_no_known_data_io_violations_remain() -> None:
    """Debt report: how many recorded raw-IO violations are left.

    Run ``pytest -m debt_tracker -rx`` to read them.
    """
    found = _scan_violations()
    known = _load_known()
    still_present = [v for v in known if any(_vkey(v) == _vkey(f) for f in found)]
    if still_present:
        msg = (
            f"{len(still_present)} known data-IO violations still present "
            "(remove from _known_violations.json once fixed):\n"
            + "\n".join(f"  {v['file']}  {v['call']}" for v in still_present)
        )
        pytest.fail(msg)


# ---------------------------------------------------------------------------
# Planted violations — the detector is only a gate for shapes it has failed on
# ---------------------------------------------------------------------------


@pytest.mark.architecture
class TestDetectorIsNotVacuous:
    """One plant per shape ``find_io_calls`` claims, per non-negotiable 15.

    The oracle this replaced was a single ``\\b``-anchored regex per call name, so
    it scored green on the two shapes below that matter most: it fired on prose
    (5 of its 44 findings were docstrings) and it could not see a call reached
    through ``from torch import load`` at all. Neither gap was visible from a
    green run — which is the whole reason these plants are committed rather than
    exercised once by hand.
    """

    def test_plant_dotted_call(self) -> None:
        src = "import torch\n\ndef f(p):\n    return torch.load(p)\n"
        assert find_io_calls(src) == {"torch.load"}

    def test_plant_aliased_module_call(self) -> None:
        """``import nibabel as nib`` — the alias is the *reported* name."""
        src = "import nibabel as nib\n\ndef f(p):\n    return nib.load(p)\n"
        assert find_io_calls(src) == {"nib.load"}

    def test_plant_unaliased_module_call(self) -> None:
        """``nibabel.load`` maps to the same reported name as ``nib.load``."""
        src = "import nibabel\n\ndef f(p):\n    return nibabel.load(p)\n"
        assert find_io_calls(src) == {"nib.load"}

    def test_plant_bare_name_call_via_from_import(self) -> None:
        """The shape the regex oracle was structurally blind to."""
        src = "from torch import load\n\ndef f(p):\n    return load(p)\n"
        assert find_io_calls(src) == {"torch.load"}

    def test_plant_renamed_bare_name_call(self) -> None:
        src = "from h5py import File as _F\n\ndef f(p):\n    return _F(p, 'r')\n"
        assert find_io_calls(src) == {"h5py.File"}

    def test_plant_function_local_import(self) -> None:
        """Function-local, the shape ``^``-anchored greps miss (#1183)."""
        src = "def f(p):\n    import pickle\n\n    return pickle.load(p)\n"
        assert find_io_calls(src) == {"pickle.load"}

    def test_plant_every_call_name_fires(self) -> None:
        """No entry of the table is dead."""
        for stmt, expected in [
            ("import torch\ntorch.load(p)\n", "torch.load"),
            ("import h5py\nh5py.File(p)\n", "h5py.File"),
            ("import nibabel as nib\nnib.load(p)\n", "nib.load"),
            ("import pickle\npickle.load(p)\n", "pickle.load"),
        ]:
            assert expected in find_io_calls(stmt), f"{expected} never fires"

    def test_plant_safetensors_load_file(self) -> None:
        src = "from safetensors.torch import load_file\n\ndef f(p):\n    return load_file(p)\n"
        assert find_io_calls(src) == {"safetensors.load_file"}

    def test_plant_safetensors_safe_open(self) -> None:
        src = "from safetensors import safe_open\n\ndef f(p):\n    return safe_open(p, 'pt')\n"
        assert find_io_calls(src) == {"safetensors.safe_open"}

    @pytest.mark.parametrize(
        "src, expected",
        [
            ("import safetensors\nsafetensors.safe_open(p, 'pt')\n", "safetensors.safe_open"),
            ("import safetensors.torch as st\nst.load_file(p)\n", "safetensors.load_file"),
            ("import safetensors.torch\nsafetensors.torch.load_file(p)\n", "safetensors.load_file"),
            ("from safetensors import torch as st\nst.load_file(p)\n", "safetensors.load_file"),
            ("from safetensors.numpy import load_file\nload_file(p)\n", "safetensors.load_file"),
        ],
        ids=["module-safe_open", "aliased-submodule", "three-part-dotted", "from-submodule", "numpy-flavour"],
    )
    def test_plant_every_safetensors_spelling(self, src: str, expected: str) -> None:
        assert find_io_calls(src) == {expected}

    def test_safetensors_save_file_is_not_a_load(self) -> None:
        src = "from safetensors.torch import save_file\n\ndef f(t, p):\n    save_file(t, p)\n"
        assert find_io_calls(src) == set()

    # -- negative plants: these must NOT turn the gate red -------------------

    def test_prose_mentioning_a_call_does_not_fire(self) -> None:
        """The exact defect: ``pipelines/infer.py`` documenting its compliance.

        Its docstring says the pipeline layer never calls ``h5py.File`` or
        ``nib.load``; the regex recorded both as violations of the rule the
        sentence asserts. Two of that file's three baseline entries described
        nothing at all.
        """
        src = (
            'def f():\n'
            '    """This layer never calls ``h5py.File`` or ``nib.load``.\n\n'
            '    Returns whatever ``torch.load`` returned.\n'
            '    """\n'
            '    return None\n'
        )
        assert find_io_calls(src) == set()

    def test_attribute_reference_without_a_call_does_not_fire(self) -> None:
        """``getattr(torch.load, "__code__")`` is a version probe, not file IO."""
        src = "import torch\n\nHAS = getattr(torch.load, '__code__', None) is not None\n"
        assert find_io_calls(src) == set()

    def test_unrelated_same_named_call_does_not_fire(self) -> None:
        """``self.load`` / ``json.load`` are not the guarded symbols."""
        src = "import json\n\nclass C:\n    def f(self, p):\n        self.load(p)\n        json.load(p)\n"
        assert find_io_calls(src) == set()

    def test_syntax_error_yields_no_findings_not_a_crash(self) -> None:
        assert find_io_calls("def f(:\n") == set()


@pytest.mark.architecture
class TestScanWalksAndExcludesDataLayer:
    """The walk itself: a plant outside ``data/`` is seen, one inside is not."""

    @staticmethod
    def _plant(tmp_path, monkeypatch) -> list[dict[str, str]]:
        src = tmp_path / "src" / "spectramr"
        (src / "pipelines").mkdir(parents=True)
        (src / "data" / "datasets").mkdir(parents=True)
        (src / "pipelines" / "planted.py").write_text("import torch\ntorch.load('x')\n")
        (src / "data" / "datasets" / "planted.py").write_text("import torch\ntorch.load('x')\n")
        monkeypatch.setattr("tests.architecture.test_ssot_data_io.REPO_ROOT", tmp_path)
        monkeypatch.setattr("tests.architecture.test_ssot_data_io.SRC_ROOT", src)
        return _scan_violations()

    def test_plant_outside_data_layer_is_reported(self, tmp_path, monkeypatch) -> None:
        found = self._plant(tmp_path, monkeypatch)
        assert {"file": "spectramr/pipelines/planted.py", "call": "torch.load"} in found

    def test_plant_inside_data_layer_is_exempt(self, tmp_path, monkeypatch) -> None:
        found = self._plant(tmp_path, monkeypatch)
        assert not [v for v in found if v["file"].startswith(DATA_SSOT_PREFIX)]

    def test_plant_inside_codecs_is_exempt_and_beside_it_is_not(self, tmp_path, monkeypatch) -> None:
        src = tmp_path / "src" / "spectramr"
        codecs_dir = src / "infrastructure" / "checkpointing" / "codecs"
        codecs_dir.mkdir(parents=True)
        (codecs_dir / "planted.py").write_text("import torch\ntorch.load('x')\n")
        (codecs_dir.parent / "planted.py").write_text("import torch\ntorch.load('x')\n")
        monkeypatch.setattr("tests.architecture.test_ssot_data_io.REPO_ROOT", tmp_path)
        monkeypatch.setattr("tests.architecture.test_ssot_data_io.SRC_ROOT", src)
        found = {v["file"] for v in _scan_violations()}
        assert "spectramr/infrastructure/checkpointing/planted.py" in found
        assert not [f for f in found if f.startswith(CHECKPOINT_IO_PREFIX)]


@pytest.mark.architecture
class TestModuleAliasResolution:
    """``import <guarded> as <alias>`` — measured free to close (0 findings)."""

    def test_plant_module_alias_call(self) -> None:
        src = "import torch as th\n\ndef f(p):\n    return th.load(p)\n"
        assert find_io_calls(src) == {"torch.load"}

    def test_plant_module_alias_function_local(self) -> None:
        src = "def f(p):\n    import pickle as _pk\n\n    return _pk.load(p)\n"
        assert find_io_calls(src) == {"pickle.load"}

    def test_real_torch_alias_in_tree_is_not_a_load(self) -> None:
        """The two ``import torch as _torch`` sites use it for dtype probes."""
        src = "import torch as _torch\n\ndef f(t):\n    return _torch.is_complex(t)\n"
        assert find_io_calls(src) == set()

    def test_alias_of_an_unguarded_module_is_not_followed(self) -> None:
        """``import json as torch`` must not make ``torch.load`` a finding."""
        src = "import json as torch\n\ndef f(p):\n    return torch.load(p)\n"
        assert find_io_calls(src) == set()

    def test_plant_submodule_import_does_not_shadow_the_package(self) -> None:
        """The real shape at ``training/builders/model_builder.py:9-11,141``."""
        src = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.utils.checkpoint\n\n"
            "def f(p):\n"
            "    return torch.load(p)\n"
        )
        assert find_io_calls(src) == {"torch.load"}

    def test_plant_submodule_alias_is_not_the_package(self) -> None:
        """``nn`` is bound to ``torch.nn``; ``nn.load`` is not ``torch.load``."""
        src = "import torch.nn as nn\n\ndef f(p):\n    return nn.load(p)\n"
        assert find_io_calls(src) == set()
