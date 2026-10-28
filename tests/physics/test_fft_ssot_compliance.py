"""Regression test: SSOT compliance for FFT operations.

CLAUDE.md pitfall #2 forbids raw ``torch.fft.fft2`` / ``torch.fft.ifft2`` (and
their N-dim / 1-D variants) on complex MRI k-space. Use
``spectramr.infrastructure.physics.fft_ops.fft2c`` / ``ifft2c`` instead — they
handle centering, ``norm="ortho"``, and AMP-safe FP32 context together.

This test AST-walks every ``src/`` module and partitions raw ``torch.fft.*``
call sites into three buckets:

1. ``INTENTIONAL_EXEMPT`` — files where the raw call is architecturally
   justified (SSOT itself, spectral PDE solvers, 1D temporal filters).
2. ``LEGACY_EXEMPT_BUDGET`` — pre-SSOT-migration tech debt, locked at the
   current per-file call count. New violations in these files (count
   grows) and new files outside both lists both fail the test.
3. Everything else — a violation if any forbidden call is present.

``rfft`` / ``irfft`` / ``rfft2`` / ``irfft2`` / ``rfftn`` / ``irfftn`` and
the spectrum-shape helpers ``fftshift`` / ``ifftshift`` / ``fftfreq`` are
NOT in the forbidden set: they're used by spectral neural operators (FNO,
S4D, Hyena, Hermitian-equivariant blocks) on real-valued feature maps,
which are explicitly exempt per CLAUDE.md.

To clear a legacy entry: rewrite the file to use ``fft2c`` / ``ifft2c`` and
delete the entry from ``LEGACY_EXEMPT_BUDGET``. To add a NEW intentional
exemption: add to ``INTENTIONAL_EXEMPT`` with a written reason explaining
why ``fft2c`` is not applicable.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

import pytest


def _find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` to the directory that owns ``src/spectramr``.

    This deliberately replaces the depth-coded ``Path(__file__).parents[3]``
    this module used while it lived under ``tests/unit/physics/``. That form
    is what kept the guard *out* of the blocking lane for as long as it did:
    moving the file to ``tests/physics/`` (one level shallower) silently
    re-rooted ``SRC_DIR`` at a directory with no ``src/``, and the primary
    scan below -- the one that covers the whole corpus -- then iterated an
    empty ``rglob`` and **passed**. Measured on the naive move: 4 of the 6
    tests went green in 0.17 s (against 2.86 s in place), and the only two
    reds complained about "missing files", whose obvious repair is to delete
    the budget entries, which would have made the guard permanently blind.

    Anchoring on ``src/spectramr`` rather than on ``pyproject.toml`` or ``.git``
    is deliberate too: ``.git`` is a *file* inside a git worktree, and a bare
    ``src`` marker would happily bind to some *other* checkout's ``src/`` at
    the wrong depth -- a wrong tree scanned in silence, which is strictly
    worse than no tree at all.

    Raises rather than falling back to a default root (non-negotiable 3): an
    unlocatable corpus is a state to report, never one to infer.
    """
    for candidate in (start, *start.parents):
        if (candidate / "src" / "spectramr").is_dir():
            return candidate
    raise RuntimeError(
        f"Cannot locate the repository root above {start}: no ancestor owns "
        "src/spectramr. This guard scans the real corpus and must never fall "
        "back to a default root -- a guard that scans nothing passes."
    )


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
SRC_DIR = REPO_ROOT / "src"

# A scan that sees "no violations" because it saw *no files* is the failure
# mode this guard shipped with. `test_the_scan_is_not_vacuous` pins a loose
# floor rather than the exact count (2,045 on `5b982f331`): an exact figure
# becomes a ratchet that fires on unrelated deletions, while a loose floor
# still catches the only thing it exists to catch.
MIN_SCANNED_FILES = 500

FORBIDDEN_FNS: frozenset[str] = frozenset({"fft", "ifft", "fft2", "ifft2", "fftn", "ifftn"})

# Architecturally justified — should not grow.
INTENTIONAL_EXEMPT: dict[str, str] = {
    "src/spectramr/infrastructure/physics/fft_ops.py": (
        "SSOT implementation — defines fft2c / ifft2c / volume helpers."
    ),
    "src/spectramr/infrastructure/physics/conformal_geometry.py": (
        "Beltrami PDE solver — spectral derivatives on unit-square grid, "
        "paired with torch.fft.fftfreq multipliers in the unshifted "
        "(corner-DC) layout. Not an MRI k-space round-trip; fft2c would "
        "silently break the multiplier algebra."
    ),
    "src/spectramr/infrastructure/sensors/adapters/respiratory.py": (
        "1D temporal respiratory-waveform low/band-pass filter, not MRI "
        "k-space. Operates on sensor time-series."
    ),
    "src/spectramr/core/metrics/no_reference_extended.py": (
        "Power-spectrum NR feature on a real grayscale image (forward FFT only, "
        "no k-space round-trip / reconstruction). The file's own docstring notes "
        "this; fft2c centering is irrelevant to a |spectrum|^2 feature."
    ),
    "src/spectramr/core/metrics/nr_texture.py": (
        "Log-Gabor phase-congruency filter bank: fft2 -> multiply by a log-Gabor "
        "filter defined in the corner-DC fftfreq layout -> ifft2. fft2c (centered) "
        "would break the filter's DC handling (log_gabor[0, 0]) and radius algebra, "
        "like the conformal_geometry exemption. Real-image feature, not k-space."
    ),
    "src/spectramr/infrastructure/physics/signal_models/spectroscopy.py": (
        "1-D FID -> spectrum along the time axis. Only the OUTPUT is shifted: an "
        "FID's origin really is index 0, so the input ifftshift that fft2c applies "
        "to an image (origin at N // 2) would mis-centre it. norm='ortho' and the "
        "disabled autocast mirror fft_ops. Not a 2-D k-space round-trip."
    ),
    "src/spectramr/infrastructure/physics/dct_ops.py": (
        "DCT-II/III construction via FFT (even-mirror reorder + twiddle factors). A "
        "real-valued cosine transform, not a complex MRI k-space round-trip; fft2c "
        "is inapplicable (it would destroy the DCT's reorder/twiddle algebra)."
    ),
    "src/spectramr/infrastructure/physics/galois_unwrap.py": (
        "Spectral Poisson solver for least-squares phase unwrapping (cosine-"
        "eigenvalue discrete Laplacian on an even-extended grid). A PDE solve, not a "
        "k-space round-trip — fft2c's centering would break the eigenvalue "
        "multiplier algebra (cf. conformal_geometry)."
    ),
    "src/spectramr/infrastructure/physics/helmholtz_hodge.py": (
        "Spectral Poisson solve Delta u = rhs on a periodic 3-D grid (sin^2-"
        "eigenvalue Laplacian via fftfreq), for the Helmholtz-Hodge decomposition. A "
        "PDE solver like conformal_geometry; fft2c would break the eigenvalue algebra."
    ),
    "src/spectramr/models/losses/phase_stego_score.py": (
        "Phase-only forward map FFT(exp(i*arg(x))) for a steganography-detection "
        "statistic — a forward transform of the phase unit vector with no "
        "reconstruction / k-space round-trip. Centering is irrelevant to the "
        "spectral-energy statistic the score consumes."
    ),
    "src/spectramr/core/metrics/srf_bound.py": (
        "Radial power-spectrum band energies of a REAL magnitude image (forward "
        "FFT only, no reconstruction). It does its own fftshift and builds the "
        "radius from the matching fftshift(fftfreq) grid, so the layout is "
        "self-consistent; ortho-norm would only rescale a log1p band feature. "
        "Same class as the no_reference_extended / nr_texture exemptions."
    ),
    "src/spectramr/infrastructure/physics/acquisition_codesign.py": (
        "1-D LOUPE-style sampling co-design on synthetic [N, W] rows: "
        "ifft(mask*gain * fft(x)) along a single axis. The mask is LEARNED over "
        "raw column indices, so no absolute k-space position is assumed and the "
        "unshifted layout is self-consistent; fft/ifft without norm are exact "
        "inverses. Not a 2-D MRI k-space round-trip — fft2c does not apply."
    ),
}

# Pre-SSOT-migration tech debt. The number is the current per-file count
# of forbidden torch.fft.* call sites. The test fails if any count grows
# OR if a new file appears with violations. As each file is migrated to
# fft_ops.fft2c / ifft2c, delete its entry — the test then enforces that
# it stays at zero.
LEGACY_EXEMPT_BUDGET: dict[str, int] = {
    "src/spectramr/infrastructure/physics/coil_sensitivity.py": 1,
    "src/spectramr/infrastructure/physics/dc_navigator.py": 2,
    # the degradations package fully migrated to fft_ops.fft2c / ifft2c (0 raw
    # torch.fft.* calls) — stale budget entry removed per the "drop the entry"
    # contract of test_legacy_budget_entries_still_violate.
    "src/spectramr/infrastructure/physics/dipole.py": 2,
    "src/spectramr/infrastructure/physics/forward_operator.py": 8,
    "src/spectramr/infrastructure/physics/implementations/fft_operator.py": 8,
    "src/spectramr/infrastructure/physics/motion_correction.py": 3,
    "src/spectramr/infrastructure/physics/motion_simulation.py": 1,
    "src/spectramr/infrastructure/physics/qsm.py": 6,
    # ulf_forward_operator.py fully migrated to fft_ops (0 raw torch.fft.* calls)
    # — stale budget entry removed per test_legacy_budget_entries_still_violate.
    "src/spectramr/infrastructure/physics/vf_corrections.py": 7,
    "src/spectramr/infrastructure/physics/vf_field_extraction.py": 5,
    "src/spectramr/infrastructure/physics/vf_operators_extended.py": 1,
    "src/spectramr/models/generators/aftnet_generator.py": 3,
    "src/spectramr/models/generators/vf_field_generators.py": 3,
}


def _find_violations(path: Path) -> list[tuple[int, str]]:
    """Return ``(line_no, call_name)`` for every forbidden ``torch.fft.*`` call."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Attribute):
            continue
        if not isinstance(node.value.value, ast.Name):
            continue
        if node.value.value.id != "torch":
            continue
        if node.value.attr != "fft":
            continue
        if node.attr in FORBIDDEN_FNS:
            hits.append((node.lineno, f"torch.fft.{node.attr}"))
    return hits


def scan_for_unbudgeted(
    src_dir: Path,
    root: Path,
    intentional: Mapping[str, str],
    budget: Mapping[str, int],
) -> list[str]:
    """Return ``path:line  call`` for every forbidden call in an unexcused file.

    Parameterized on the tree rather than closing over ``SRC_DIR`` so the
    planted-violation tests at the foot of this module can drive the *real*
    matcher over a synthetic tree. A guard whose only exercise is the corpus
    it already passes on has never been observed to fail (non-negotiable 15),
    and a violation cannot be planted in ``src/`` and committed.
    """
    violations: list[str] = []
    for py_file in sorted(src_dir.rglob("*.py")):
        rel = py_file.relative_to(root).as_posix()
        if rel in intentional or rel in budget:
            continue
        for line_no, fn_name in _find_violations(py_file):
            violations.append(f"{rel}:{line_no}  {fn_name}")
    return violations


def scan_for_budget_growth(root: Path, budget: Mapping[str, int]) -> list[str]:
    """Return a report line per legacy file whose violation count grew."""
    regressed: list[str] = []
    for rel, allowed in budget.items():
        path = root / rel
        if not path.is_file():
            # `test_legacy_budget_entries_exist` owns the stale-entry report;
            # counting a file that is not there would double-report it.
            continue
        actual = len(_find_violations(path))
        if actual > allowed:
            regressed.append(f"  - {rel}: budget={allowed}, now {actual} (+{actual - allowed} new)")
    return regressed


def scanned_file_count(src_dir: Path) -> int:
    """How many ``.py`` files the corpus scan actually visits."""
    return sum(1 for _ in src_dir.rglob("*.py"))


def test_no_new_raw_torch_fft_outside_ssot() -> None:
    """No new file may introduce a raw ``torch.fft.{fft,ifft,fft2,ifft2,fftn,ifftn}`` call.

    Use ``spectramr.infrastructure.physics.fft_ops.fft2c`` / ``ifft2c`` for MRI
    k-space round-trips. If a new exemption is genuinely needed (new
    spectral PDE solver), add it to ``INTENTIONAL_EXEMPT`` with a written
    reason.
    """
    violations = scan_for_unbudgeted(SRC_DIR, REPO_ROOT, INTENTIONAL_EXEMPT, LEGACY_EXEMPT_BUDGET)
    assert not violations, (
        "Raw torch.fft.* calls found in a file that is not in the SSOT, "
        "an intentional exemption, or the legacy budget. Use "
        "spectramr.infrastructure.physics.fft_ops.fft2c / ifft2c for MRI k-space "
        "round-trips, or add a justified entry to INTENTIONAL_EXEMPT.\n\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_legacy_budget_does_not_grow() -> None:
    """Per-file legacy violation counts must not exceed the baseline.

    The legacy budget locks the current state of pre-SSOT-migration code.
    Any addition of a new forbidden torch.fft.* call in a legacy file
    fails the test. Removals are encouraged — delete the entry once the
    file is fully migrated to fft2c / ifft2c.
    """
    regressed = scan_for_budget_growth(REPO_ROOT, LEGACY_EXEMPT_BUDGET)
    assert not regressed, (
        "Legacy FFT-SSOT violation budget exceeded. Either migrate the "
        "new call sites to fft2c / ifft2c, or — if architecturally "
        "necessary — raise the budget AND document why.\n\n" + "\n".join(regressed)
    )


def test_intentional_exempt_files_actually_exist() -> None:
    """Every entry in ``INTENTIONAL_EXEMPT`` points to a real file.

    Catches drift when an exempt file is renamed or deleted — a stale
    exemption silently becoming dead config is the same anti-pattern
    we're guarding against in the source.
    """
    missing = [rel for rel in INTENTIONAL_EXEMPT if not (REPO_ROOT / rel).is_file()]
    assert not missing, "INTENTIONAL_EXEMPT references files that no longer exist:\n" + "\n".join(
        f"  - {m}" for m in missing
    )


def test_intentional_exempt_files_contain_forbidden_calls() -> None:
    """Every intentional exemption is actually load-bearing.

    If an entry no longer contains a forbidden call, drop it — keeping
    stale exemptions hides the day a new violation lands inside an
    exempt file.
    """
    stale = []
    for rel in INTENTIONAL_EXEMPT:
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        if not _find_violations(path):
            stale.append(rel)
    assert not stale, (
        "Files in INTENTIONAL_EXEMPT no longer contain any forbidden "
        "torch.fft.* call — drop the exemption:\n" + "\n".join(f"  - {s}" for s in stale)
    )


def test_legacy_budget_entries_exist() -> None:
    """Every legacy-budget entry points to a real file.

    Catches drift when a legacy file is deleted or renamed — the budget
    entry must be removed in the same commit.
    """
    missing = [rel for rel in LEGACY_EXEMPT_BUDGET if not (REPO_ROOT / rel).is_file()]
    assert not missing, (
        "LEGACY_EXEMPT_BUDGET references files that no longer exist — "
        "remove the budget entry:\n" + "\n".join(f"  - {m}" for m in missing)
    )


def test_legacy_budget_entries_still_violate() -> None:
    """Every legacy-budget entry actually still contains a forbidden call.

    If a file in LEGACY_EXEMPT_BUDGET no longer has any raw torch.fft.*
    call (because it was migrated to fft_ops.fft2c / ifft2c), drop the
    entry rather than carrying a 0-budget exemption forward.
    """
    clean = []
    for rel in LEGACY_EXEMPT_BUDGET:
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        if not _find_violations(path):
            clean.append(rel)
    assert not clean, (
        "Files in LEGACY_EXEMPT_BUDGET no longer contain any forbidden "
        "torch.fft.* call — congratulations on the migration; drop the "
        "entry from the budget:\n" + "\n".join(f"  - {c}" for c in clean)
    )


# --------------------------------------------------------------------------- #
# The guard guards itself
#
# Everything above scans the real ``src/`` tree, which this suite has always
# passed on -- so none of it has ever been *observed* to go red (non-negotiable
# 15). A violation cannot be planted in ``src/`` and committed, so the plants
# below drive the same matcher (`scan_for_unbudgeted`, `scan_for_budget_growth`,
# `_find_violations`) over synthetic trees under ``tmp_path``. One per rule the
# guard claims to enforce, and one per shape that rule can take.
# --------------------------------------------------------------------------- #


def _tree(root: Path, files: Mapping[str, str]) -> Path:
    """Materialise ``{relative_path: source}`` under ``root/src`` and return root."""
    for rel, source in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


def test_the_scan_is_not_vacuous() -> None:
    """The corpus scan must actually visit the corpus.

    This is the plant for the defect this file shipped with: relocated one
    directory shallower, ``SRC_DIR`` pointed at a tree with no ``src/``,
    ``rglob`` yielded nothing, and the whole-corpus check passed in 0.17 s
    while reporting zero violations. "Found nothing" and "looked at nothing"
    are the same green, so the floor has to be asserted separately.
    """
    assert SRC_DIR.is_dir(), f"SRC_DIR does not exist: {SRC_DIR}"
    scanned = scanned_file_count(SRC_DIR)
    assert scanned >= MIN_SCANNED_FILES, (
        f"The FFT-SSOT scan visited only {scanned} file(s) under {SRC_DIR} "
        f"(floor: {MIN_SCANNED_FILES}). A guard that scans nothing passes "
        "silently -- check that REPO_ROOT still resolves to the repository."
    )


def test_an_empty_tree_fails_the_vacuity_floor(tmp_path: Path) -> None:
    """The floor is load-bearing: an empty tree must not clear it."""
    empty = _tree(tmp_path, {})
    (empty / "src").mkdir()
    assert scanned_file_count(empty / "src") < MIN_SCANNED_FILES


def test_a_planted_raw_fft_in_an_unexcused_file_is_found(tmp_path: Path) -> None:
    """The primary rule: a new raw ``torch.fft.*`` call outside both lists."""
    root = _tree(
        tmp_path,
        {
            "src/spectramr/models/newcomer.py": (
                "import torch\n\n\ndef go(x):\n    return torch.fft.fft2(x)\n"
            )
        },
    )
    found = scan_for_unbudgeted(root / "src", root, {}, {})
    assert found == ["src/spectramr/models/newcomer.py:5  torch.fft.fft2"], found


def test_an_excused_file_is_not_reported_under_either_list(tmp_path: Path) -> None:
    """Both excuse lists must actually excuse -- otherwise the guard is noise."""
    rel = "src/spectramr/models/newcomer.py"
    root = _tree(
        tmp_path,
        {rel: "import torch\n\n\ndef go(x):\n    return torch.fft.fft2(x)\n"},
    )
    assert scan_for_unbudgeted(root / "src", root, {rel: "reason"}, {}) == []
    assert scan_for_unbudgeted(root / "src", root, {}, {rel: 1}) == []


def test_the_spectral_carve_out_is_not_flagged(tmp_path: Path) -> None:
    """``rfft*``/``fftshift``/``fftfreq`` are exempt per CLAUDE.md non-negotiable 2.

    Spectral neural operators (FNO, S4D, Hyena) run these on *real* feature
    maps, where centering is meaningless. A guard that flagged them would be
    turned off, so the carve-out is as load-bearing as the rule.
    """
    root = _tree(
        tmp_path,
        {
            "src/spectramr/models/fno.py": (
                "import torch\n\n\ndef go(x):\n"
                "    y = torch.fft.rfft2(x)\n"
                "    y = torch.fft.fftshift(y)\n"
                "    return torch.fft.irfftn(y)\n"
            )
        },
    )
    assert scan_for_unbudgeted(root / "src", root, {}, {}) == []


def test_a_budgeted_file_that_grew_is_found(tmp_path: Path) -> None:
    """The ratchet direction: a legacy file may shrink, never grow."""
    rel = "src/spectramr/legacy.py"
    root = _tree(
        tmp_path,
        {
            rel: (
                "import torch\n\n\ndef go(x):\n"
                "    a = torch.fft.fft2(x)\n"
                "    return torch.fft.ifft2(a)\n"
            )
        },
    )
    assert scan_for_budget_growth(root, {rel: 2}) == []
    grown = scan_for_budget_growth(root, {rel: 1})
    assert len(grown) == 1 and "budget=1, now 2" in grown[0], grown


def test_the_aliased_call_shape_is_out_of_scope_by_construction(
    tmp_path: Path,
) -> None:
    """The matcher keys on the literal name ``torch``, and that is a real limit.

    ``np.fft.fft2`` and ``self.fft.fft2`` are invisible to it. On
    ``5b982f331`` the corpus holds 20 such sites across 6 files: 16 NumPy
    calls on real image-domain features (the same class as the
    ``no_reference_extended`` / ``nr_texture`` exemptions) and 4 routed
    through the SSOT's own ``FFTTransformer``. None is a violation of the
    rule as written.

    The four routed through ``FFTTransformer`` USED to reach its *uncentered*
    family (``fft2`` -> ``fft_volume_spatial``, no shifts) rather than ``fft2c``
    -- dossier D13 finding V1. #1350 resolved it: those call sites now use
    ``fft2c`` / ``fftnc``, and the uncentered family was renamed
    ``*_uncentered`` so no caller can reach it without naming the convention.
    The remaining ``self.fft.*`` receivers this matcher cannot see are therefore
    centred; the blindness is unchanged, what it hides is not.

    This test does not endorse the gap; it pins it, so the day someone widens
    the matcher to aliased receivers the change is deliberate and this is the
    test they edit.
    """
    root = _tree(
        tmp_path,
        {
            "src/spectramr/aliased.py": (
                "import numpy as np\n\n\ndef go(x):\n    return np.fft.fft2(x)\n"
            )
        },
    )
    assert scan_for_unbudgeted(root / "src", root, {}, {}) == []


def test_the_repo_root_is_found_by_anchor_not_by_depth(tmp_path: Path) -> None:
    """Relocating this file must not silently re-root the scan."""
    root = _tree(tmp_path, {"src/spectramr/__init__.py": ""})
    deep = root / "tests" / "a" / "b" / "c" / "test_x.py"
    deep.parent.mkdir(parents=True, exist_ok=True)
    deep.write_text("")
    shallow = root / "tests" / "test_x.py"
    shallow.write_text("")
    assert _find_repo_root(deep) == root
    assert _find_repo_root(shallow) == root


def test_a_bare_src_directory_does_not_capture_the_root(tmp_path: Path) -> None:
    """Anchoring on ``src/spectramr`` and not on ``src`` alone.

    A nearer directory that merely owns *a* ``src/`` would bind first and the
    guard would scan the wrong tree in silence -- strictly worse than scanning
    none, because the vacuity floor above would not catch it either.
    """
    root = _tree(tmp_path, {"src/spectramr/__init__.py": ""})
    decoy = root / "vendor"
    (decoy / "src").mkdir(parents=True)
    probe = decoy / "tests" / "test_x.py"
    probe.parent.mkdir(parents=True)
    probe.write_text("")
    assert _find_repo_root(probe) == root


def test_an_unlocatable_root_raises_instead_of_defaulting(tmp_path: Path) -> None:
    """Non-negotiable 3: absent is a state to report, never one to infer."""
    orphan = tmp_path / "nowhere" / "test_x.py"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("")
    with pytest.raises(RuntimeError, match="Cannot locate the repository root"):
        _find_repo_root(orphan)
