"""
TASK III.2 – Physics SSOT fitness function.

Enforces CLAUDE.md pitfall #2:

    No ``torch.fft.*`` calls outside
    ``spectramr/infrastructure/physics/fft_ops.py``, EXCEPT:

    1. All files under ``spectramr/infrastructure/physics/**``  (blanket exempt).
    2. Files in ``_fft_allowlist.txt`` — spectral-operator blocks
       (FNO/Hyena/S4D/spectral/Toeplitz) under ``spectramr/models/blocks/``
       that operate on *real-valued feature maps*.  These are LEGITIMATE
       uses that will never be flagged.
    3. Files in ``_known_violations.json["raw_torch_fft"]`` — pre-existing
       genuine violations recorded as technical debt.

Gate test (fast, always runs):
    Fail if torch.fft.* appears in a file that is neither blanket-exempt,
    nor in the legitimate allowlist, nor in the pre-existing debt list.
    This catches NEW violations from the moment they appear.

Cleanup tracker (slow, opt-in with -m slow):
    Fails until ALL known_violations["raw_torch_fft"] entries are fixed,
    AND until _fft_allowlist.txt shrinks to zero (spectral blocks migrated).
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
ALLOWLIST_FILE = Path(__file__).parent / "_fft_allowlist.txt"
VIOLATIONS_FILE = Path(__file__).parent / "_known_violations.json"

# Blanket-exempt subtrees (all files always allowed)
BLANKET_EXEMPT_PREFIXES: list[str] = [
    "spectramr/infrastructure/physics/",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_legitimate_allowlist() -> set[str]:
    """Files with intentional spectral-operator usage (real feature maps)."""
    if not ALLOWLIST_FILE.exists():
        return set()
    lines = ALLOWLIST_FILE.read_text().splitlines()
    return {
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    }


def _load_known_violations() -> set[str]:
    """Files that are pre-existing debt (flagged, but gated, not new)."""
    if not VIOLATIONS_FILE.exists():
        return set()
    data = json.loads(VIOLATIONS_FILE.read_text())
    return {v["file"] for v in data.get("raw_torch_fft", [])}


def _is_blanket_exempt(rel: str) -> bool:
    return any(rel.startswith(p) for p in BLANKET_EXEMPT_PREFIXES)


def uses_raw_torch_fft(source: str) -> bool:
    """Whether ``source`` reaches ``torch.fft`` in code: attribute, import or alias.

    Parsed, not grepped: a ``"torch.fft."`` string, comment or docstring is not a
    call, while ``import torch as T; T.fft.fft2`` and ``from torch import fft`` are.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    torch_names = {"torch"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch.fft" or alias.name.startswith("torch.fft."):
                    return True
                if alias.name == "torch" and alias.asname:
                    torch_names.add(alias.asname)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module == "torch.fft" or (node.module or "").startswith("torch.fft."):
                return True
            if node.module == "torch" and any(a.name == "fft" for a in node.names):
                return True
    return any(
        isinstance(node, ast.Attribute)
        and node.attr == "fft"
        and isinstance(node.value, ast.Name)
        and node.value.id in torch_names
        for node in ast.walk(tree)
    )


def _scan_all_raw_fft_files() -> list[str]:
    """All files outside blanket-exempt that use torch.fft.*."""
    results: list[str] = []
    for py_file in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        rel = str(py_file.relative_to(REPO_ROOT / "src"))
        if _is_blanket_exempt(rel):
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
        except OSError:
            continue
        if uses_raw_torch_fft(source):
            results.append(rel)
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.architecture
def test_no_new_raw_torch_fft_outside_physics_ssot() -> None:
    """Gate: fail on any NEW raw torch.fft.* usage outside allowed locations.

    Allowed locations (not flagged):
      1. ``spectramr/infrastructure/physics/**``  — the SSOT (blanket exempt).
      2. Files in ``_fft_allowlist.txt`` — legitimate spectral-operator blocks
         operating on real-valued feature maps.
      3. Files in ``_known_violations.json["raw_torch_fft"]`` — pre-existing
         debt (gated, not new violations).

    To add a genuinely new spectral-operator block:
      - Append to ``_fft_allowlist.txt`` with a comment.

    For pre-existing violations discovered during development:
      - Add to ``_known_violations.json["raw_torch_fft"]`` temporarily.

    For new MRI k-space round-trips:
      - Fix: use ``spectramr.infrastructure.physics.fft_ops.fft2c / ifft2c``.
    """
    all_files = _scan_all_raw_fft_files()
    legitimate = _load_legitimate_allowlist()
    known_debt = _load_known_violations()

    new_violations = [
        f for f in all_files if f not in legitimate and f not in known_debt
    ]

    if new_violations:
        msg = (
            "NEW raw torch.fft.* usage outside the physics SSOT "
            "(not in _fft_allowlist.txt or _known_violations.json):\n"
            + "\n".join(f"  {v}" for v in new_violations)
            + "\n\nFix: route complex MRI k-space FFTs through "
            "spectramr.infrastructure.physics.fft_ops (fft2c/ifft2c). "
            "If this is a spectral neural-operator block on real feature maps, "
            "add the file to tests/architecture/_fft_allowlist.txt."
        )
        pytest.fail(msg)


@pytest.mark.architecture
def test_fft_allowlists_have_no_stale_entries() -> None:
    """Hard gate: every recorded exemption must still describe a real file.

    A stale entry makes the allowlist lie about the codebase — and, worse,
    silently re-exempts the path if a future edit reintroduces ``torch.fft``
    there. This half of the old cleanup tracker is actionable *today*, so it
    stays a hard failure rather than riding along with the xfail debt report
    below (#629).
    """
    all_files_set = set(_scan_all_raw_fft_files())
    stale_debt = sorted(_load_known_violations() - all_files_set)
    stale_allowlist = sorted(_load_legitimate_allowlist() - all_files_set)

    messages: list[str] = []
    if stale_debt:
        messages.append(
            "Stale entries in _known_violations.json[raw_torch_fft] "
            "(file no longer uses torch.fft.* — remove them):\n"
            + "\n".join(f"  {f}" for f in stale_debt)
        )
    if stale_allowlist:
        messages.append(
            "Stale entries in _fft_allowlist.txt "
            "(file no longer uses torch.fft.* — remove them):\n"
            + "\n".join(f"  {f}" for f in stale_allowlist)
        )
    assert not messages, "\n\n".join(messages)


@pytest.mark.architecture
@pytest.mark.slow
@pytest.mark.debt_tracker
@pytest.mark.xfail(
    strict=False,
    reason="Debt report: red until raw_torch_fft debt and the spectral allowlist "
    "are both empty. XPASS means the debt is paid — delete this marker.",
)
def test_known_fft_violations_are_being_paid_down() -> None:
    """Debt report: how much recorded raw-``torch.fft`` debt is left.

    Run ``pytest -m debt_tracker -rx`` to read the remaining items. Passes (as
    XPASS) only when ``_known_violations.json["raw_torch_fft"]`` is empty AND
    ``_fft_allowlist.txt`` is empty (all spectral blocks migrated).
    """
    all_files_set = set(_scan_all_raw_fft_files())
    still_in_debt = sorted(_load_known_violations() & all_files_set)
    still_in_allowlist = sorted(_load_legitimate_allowlist() & all_files_set)

    messages: list[str] = []
    if still_in_debt:
        messages.append(
            f"{len(still_in_debt)} known raw_torch_fft violations still present "
            "(fix and remove from _known_violations.json):\n"
            + "\n".join(f"  {f}" for f in still_in_debt)
        )
    if still_in_allowlist:
        messages.append(
            f"{len(still_in_allowlist)} spectral-operator blocks still use raw torch.fft "
            "(consider migrating to fft_ops utilities):\n"
            + "\n".join(f"  {f}" for f in still_in_allowlist)
        )

    assert not still_in_debt, "\n\n".join(messages)


@pytest.mark.architecture
@pytest.mark.parametrize(
    "source",
    [
        "import torch\nx = torch.fft.fft2(y)",
        "import torch\nf = torch.fft",
        "import torch as T\nx = T.fft.ifft2(y)",
        "import torch.fft\n",
        "import torch.fft as F\nx = F.fft2(y)",
        "from torch.fft import fft2\n",
        "from torch import fft\n",
        "from torch import nn, fft as tfft\n",
    ],
)
def test_scanner_flags_every_spelling_of_torch_fft(source: str) -> None:
    """Planted violations: each spelling that reaches torch.fft turns the gate red."""
    assert uses_raw_torch_fft(source)


@pytest.mark.architecture
@pytest.mark.parametrize(
    "source",
    [
        'PREFIXES = ("np.fft.", "torch.fft.")',
        "# torch.fft.fft2 is not called here\nx = 1",
        '"""Docstring naming torch.fft.ifft2."""',
        "import numpy as np\nx = np.fft.fft2(y)",
        "from spectramr.infrastructure.physics.fft_ops import fft2c",
        "x = model.fft(y)",
    ],
)
def test_scanner_ignores_text_that_is_not_a_torch_fft_reference(source: str) -> None:
    """A string, comment or docstring naming torch.fft is not a call (#3119)."""
    assert not uses_raw_torch_fft(source)
