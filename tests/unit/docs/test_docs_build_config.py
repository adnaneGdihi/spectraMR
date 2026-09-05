"""Pin the two files that decide whether the published documentation builds.

Nothing in this repository read ``.readthedocs.yaml`` or ``docs/conf.py`` before
this module existed, and both carried a defect that only the live site could show:

* ``.readthedocs.yaml`` ran its ``pre_install`` torch step under the ``pip==23.1``
  that ``virtualenv`` seeds -- ``build.jobs.pre_install`` precedes Read the Docs'
  own pip upgrade -- and that pip discards a wheel whose metadata ``Name:`` is not
  byte-equal to the index's project name. It therefore rejected
  ``typing_extensions-4.16.0-py3-none-any.whl``, fell back to the sdist, and could
  not build it because ``--index-url`` replaces PyPI and the PEP 517 build
  dependency ``flit_core`` lives there. Every version failed at ~22 s.
* ``docs/conf.py`` stated ``release = "0.1.0"`` as a literal. The version has one
  writer (``scripts/release/bump_version.py``) and one comparator
  (``scripts/release/build_dist.py``), and neither reads this file, so it drifted:
  ``v0.1.1`` shipped while every published page still rendered ``0.1.0``.

Both are configuration rather than code, which is why lint, the layering gates and
the source-test pairing rule all saw nothing. The failure mode they share is that
the only oracle is a remote build -- so these tests are the local one.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
RTD_CONFIG = REPO / ".readthedocs.yaml"
CONF_PY = REPO / "docs" / "conf.py"
INIT_PY = REPO / "src" / "spectramr" / "__init__.py"
INDEX_RST = REPO / "docs" / "index.rst"


def _pre_install() -> list[str]:
    config = yaml.safe_load(RTD_CONFIG.read_text(encoding="utf-8"))
    return config["build"]["jobs"]["pre_install"]


def _exec_conf() -> dict:
    """Execute ``docs/conf.py`` the way Sphinx does, and return its namespace.

    Sphinx ``exec()``s the file with ``__file__`` bound, and ``conf.py`` resolves
    the package location relative to that. Reading the file as text instead would
    only re-assert its source, which is the fidelity trap non-negotiable 15 names.
    """
    namespace: dict = {"__file__": str(CONF_PY)}
    exec(compile(CONF_PY.read_text(encoding="utf-8"), str(CONF_PY), "exec"), namespace)
    return namespace


def test_pre_install_upgrades_pip_before_anything_else() -> None:
    """The pip upgrade must be FIRST, not merely present.

    Ordering is the whole fix: a pip upgrade placed after the torch line runs too
    late to matter, and the config still parses and still reads plausibly.
    """
    commands = _pre_install()
    assert commands, ".readthedocs.yaml declares no pre_install commands"
    first = commands[0]
    assert re.search(r"pip\s+install\s+(-U|--upgrade)\s+pip", first), (
        "the first build.jobs.pre_install command must upgrade pip -- it runs "
        f"under the pip==23.1 virtualenv seeds otherwise. Got: {first!r}"
    )


def test_torch_step_follows_the_pip_upgrade_and_pins_the_cpu_index() -> None:
    """Torch installs from the CPU index, and only after pip can read a wheel."""
    commands = _pre_install()
    torch_steps = [i for i, c in enumerate(commands) if re.search(r"\btorch\b", c)]
    assert torch_steps, "no pre_install command installs torch"
    for i in torch_steps:
        assert i > 0, "the torch install must not be the first pre_install command"
        assert "download.pytorch.org/whl/cpu" in commands[i], (
            "torch must resolve from the CPU index, or pip install .[docs] pulls "
            f"the nvidia-*-cu13 stack from PyPI. Got: {commands[i]!r}"
        )


def test_conf_release_matches_the_package_version() -> None:
    """``docs/conf.py`` must report the version its single writer wrote."""
    declared = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']', INIT_PY.read_text(encoding="utf-8"), re.M
    )
    assert declared is not None, f"no __version__ in {INIT_PY}"
    assert _exec_conf()["release"] == declared.group(1)


def test_conf_short_version_is_derived_not_restated() -> None:
    """``version`` is the X.Y prefix of ``release``, not an independent literal."""
    namespace = _exec_conf()
    release, short = namespace["release"], namespace["version"]
    assert short == ".".join(release.split(".")[:2])


def test_conf_states_no_version_literal() -> None:
    """A literal assignment here would be a fifth declaration nothing reconciles."""
    for line in CONF_PY.read_text(encoding="utf-8").splitlines():
        assert not re.match(r'^\s*(release|version)\s*=\s*["\']\d', line), (
            f"docs/conf.py assigns a hard-coded version: {line.strip()!r}. Read it "
            "from src/spectramr/__init__.py instead -- see bump_version.py."
        )


def test_public_index_page_carries_no_hard_coded_release() -> None:
    """The landing page names the version through ``|release|``.

    Scoped to ``index.rst`` on purpose: ``docs/versioning.rst`` uses concrete
    numbers as worked examples of the scheme, and those are content, not claims
    about what this tree currently is.
    """
    text = INDEX_RST.read_text(encoding="utf-8")
    hits = re.findall(r"\bv?\d+\.\d+\.\d+\b", text)
    assert not hits, (
        f"docs/index.rst hard-codes {hits} -- use the |release| substitution so the "
        "landing page cannot outlive the release it names."
    )


def test_rtd_config_is_published() -> None:
    """The fix only reaches Read the Docs if the exporter ships the file."""
    allowlist = (REPO / "scripts" / "release" / "public_allowlist.txt").read_text(encoding="utf-8")
    assert any(line.strip() == ".readthedocs.yaml" for line in allowlist.splitlines()), (
        "public_allowlist.txt no longer ships .readthedocs.yaml"
    )


def test_sphinx_mock_raises_for_pep604_unions_and_nothing_else() -> None:
    """Pin the mechanism ``docs/conf.py`` and ``.readthedocs.yaml`` both describe.

    Those two files justify installing the real (CPU) torch by naming exactly what
    a mocked one costs, and the naming is easy to get wrong in the expensive
    direction: the claim that stood here until 2026-09-05 was that a mocked torch
    breaks every ``torch.Tensor | None`` annotation in the tree. It does not.
    ``from __future__ import annotations`` makes a signature annotation a string
    that is never evaluated, so the union is only reached where something evaluates
    it -- a module-level alias, a default, a pydantic field.

    What *is* true is that the union operator is the ONLY shape that raises. Sphinx
    could make ``_MockObject.__or__`` return another mock in any release, at which
    point the paragraphs in both files become quietly wrong and nothing would say
    so. This test is what says so.
    """
    from sphinx.ext.autodoc.mock import _MockModule

    torch = _MockModule("torch")

    # The union raises with the mock on either side -- this is the whole cost.
    for label, operand in (
        ("mock | None", lambda: torch.Tensor | None),
        ("None | mock", lambda: None | torch.Tensor),
        ("type | mock", lambda: str | torch.device),
    ):
        try:
            operand()
        except TypeError as exc:
            assert "unsupported operand type(s) for |" in str(exc), label
        else:  # pragma: no cover - only reached when sphinx changes behaviour
            raise AssertionError(
                f"{label} no longer raises: a mocked torch now costs less than "
                "docs/conf.py and .readthedocs.yaml claim -- re-measure both."
            )

    # Everything else survives as a mock. If any of these starts raising, a mocked
    # torch costs MORE than those files claim, which is the dangerous direction.
    assert torch.no_grad()(lambda: None) is not None, "decorator shape now raises"
    assert torch.Tensor is not None, "bare attribute annotation now raises"
    assert torch.Tensor["N K"] is not None, "subscript (jaxtyping) shape now raises"


def _documented_suffixes(repo_root: Path) -> tuple[str, ...]:
    """Read ``source_suffix`` out of ``docs/conf.py`` without importing it.

    ``conf.py`` owns which extensions sphinx reads (non-negotiable 17); importing it
    would drag in the autodoc mock machinery, so the dict literal is read by AST.
    Hardcoding ``.rst`` here is what made the first version of the test below blind
    to the 38 MyST pages this tree already ships.
    """
    tree = ast.parse((repo_root / "docs" / "conf.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "source_suffix" for t in node.targets
        ):
            assert isinstance(node.value, ast.Dict), "source_suffix is no longer a dict literal"
            return tuple(ast.literal_eval(k) for k in node.value.keys)
    raise AssertionError("docs/conf.py no longer assigns source_suffix at module level")


def test_two_public_pages_use_no_index_so_anchor_counting_undercounts() -> None:
    """Pin why ``.readthedocs.yaml`` says to count signatures, not index anchors.

    ``:no-index:`` suppresses the ``id="spectramr.<...>"`` attribute while still
    rendering the signature, so grepping built HTML for those anchors misses every
    object on these two pages -- 16 of 105, which is how the comment in
    ``.readthedocs.yaml`` first shipped the torch-less loss as "61 of 89" instead
    of 77 of 105. Note the two are not in step: one ``:no-index:`` on an
    ``automodule`` hides however many members that module documents (here 1 option
    hides 2 signatures), so the option count is a tripwire, not the correction.
    """
    repo_root = Path(__file__).resolve().parents[3]

    # Measured 2026-09-05 against the public tree at b83ecb18a. A change in either
    # number moves the signature counts quoted in .readthedocs.yaml.
    for page, options in (("losses_reference", 7), ("strategies_reference", 1)):
        text = (repo_root / "docs" / f"{page}.rst").read_text(encoding="utf-8")
        assert text.count(":no-index:") == options, (
            f"docs/{page}.rst carries {text.count(':no-index:')} `:no-index:` options, "
            f"not the {options} measured when .readthedocs.yaml quoted 77 of 105 "
            "signatures lost without torch -- re-measure that comment by counting "
            '`class="sig sig-object py"` in the built HTML, never `id="spectramr.`.'
        )

    # The whole hazard is that these pages exist at all. If no public page opted out
    # of the index, anchor counting would be safe and the warning above would be dead.
    # Sweep EVERY suffix sphinx reads: a MyST `{autoclass}` fence spells `:no-index:`
    # identically, and docs/ ships 38 markdown pages beside the reST ones.
    suffixes = _documented_suffixes(repo_root)
    assert ".md" in suffixes, "markdown left source_suffix -- re-check this sweep's reach"
    opted_out = sorted(
        path.relative_to(repo_root).as_posix()
        for suffix in suffixes
        for path in (repo_root / "docs").rglob(f"*{suffix}")
        # docs/api/ is generated, private-only and never exported; it opts out wholesale.
        if "/api/" not in path.as_posix() and ":no-index:" in path.read_text(encoding="utf-8")
    )
    assert opted_out == [
        "docs/losses_reference.rst",
        "docs/strategies_reference.rst",
    ], f"the set of public pages using `:no-index:` changed: {opted_out}"
