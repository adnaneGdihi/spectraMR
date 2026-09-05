"""Regression tests for the smoke wrapper's ``--exclude-vf`` flag.

``scripts/ci/smoke_test_vf_configs.sh --exclude-vf`` (alias ``--no-vf``) prunes
the Virtual-Fiducial cohort (``experiments/inprogress/vf/``) from the run so the
rest of the smoke matrix can be exercised without the heavy digital-twin arms.

Two contracts are pinned:
1. The flag, its alias, and the directory-scoped filter exist in the wrapper.
2. The filter expression drops ``/inprogress/vf/`` paths while keeping siblings
   — including ``pillars/exp_pillar_07_vf_fourier_shift`` which is NOT part of
   the VF directory cohort and must survive.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests.utils.repo_scripts import require_repo_file

_WRAPPER_REL = "scripts/ci/smoke_test_vf_configs.sh"


def _wrapper() -> Path:
    """The wrapper, or an explained skip: ``scripts/ci/`` ships only in part.

    Resolved per-test, not as a module constant: the filter-expression test below
    exercises the grep itself and is independent of the wrapper file, so it must
    keep running in the export.
    """
    return require_repo_file(_WRAPPER_REL)


def test_wrapper_is_valid_bash() -> None:
    assert subprocess.run(["bash", "-n", str(_wrapper())]).returncode == 0


def test_exclude_vf_flag_and_alias_present() -> None:
    text = _wrapper().read_text()
    assert "--exclude-vf|--no-vf)" in text, "flag + alias must be parsed"
    assert "EXCLUDE_VF=true" in text
    assert 'grep -v "/inprogress/vf/"' in text, (
        "exclusion must be scoped to the vf/ directory, not a bare 'vf' substring"
    )
    # Usage line documents the flag.
    assert "--exclude-vf" in text and "Virtual-Fiducial" in text


def test_filter_expression_drops_only_vf_dir() -> None:
    """The exact grep used by the wrapper keeps non-vf paths and the
    pillars/*vf* arm, dropping only experiments/inprogress/vf/ entries."""
    paths = "\n".join(
        [
            "/repo/experiments/inprogress/vf/exp_vf_04_b0_drift.yaml",
            "/repo/experiments/inprogress/diffusion/exp_06.yaml",
            "/repo/experiments/inprogress/vf/exp_vf_33_phase_nav.yaml",
            "/repo/experiments/inprogress/pillars/exp_pillar_07_vf_fourier_shift.yaml",
        ]
    )
    result = subprocess.run(
        ["grep", "-v", "/inprogress/vf/"],
        input=paths,
        capture_output=True,
        text=True,
    )
    kept = result.stdout.split()
    assert any("diffusion/exp_06" in p for p in kept)
    assert any("pillars/exp_pillar_07_vf_fourier_shift" in p for p in kept)
    assert not any("/inprogress/vf/" in p for p in kept)
    assert len(kept) == 2
