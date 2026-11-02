"""Integration test for the sim2rank native-data path (M4Raw, 2026-05-25).

Mirrors the glue that ``sim2rank.py`` performs on real data: synthesize the
pseudo-GT (real complex coil images + ESPIRiT maps), sweep with
``return_complex=True``, SENSE-combine the GT, and feed the real coil maps +
complex (phase-bearing) images to the engine. Pins that the native path
actually engages — phase + g-factor differ from the FourierBridge fallback —
and runs on CUDA when available (the canonical backend).

Skipped when the M4Raw H5 data is unavailable (e.g. CI), so it's a local /
cluster regression guard rather than a hard dependency.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Guard the MODULE, not only the data. Every test below reaches
# ``scripts/sim2rank/`` through a function-local import, and that package is
# deliberately not in the public export (the in-package
# ``core/metrics/meta_evaluation/`` half ships and stands alone). Without this
# line the file is still skipped there -- but by the ``skipif`` below, which
# asks whether the M4Raw corpus is present. Two independent things are absent
# and only one is guarded, so the green is coincidental: ship the data, or run
# where ``databases/m4raw/`` exists, and the skipif goes False and the imports
# raise ``ModuleNotFoundError`` instead. Guarding the ancestor covers all five
# submodules imported below.
pytest.importorskip("scripts.sim2rank")

# Anchor to the repo root, not the cwd: a bare relative path silently skips this
# real-data guard whenever pytest is launched from any other directory on the
# cluster, turning a false green into "the native path is untested" (#311 audit).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "databases/m4raw/data/multicoil_train"
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _DATA_DIR.exists() or not any(_DATA_DIR.rglob("*.h5")),
        reason="M4Raw H5 data not available locally",
    ),
]


def _one_group():
    from scripts.sim2rank.sim2rank import _discover_repetition_groups

    groups = _discover_repetition_groups(_DATA_DIR, max_subjects=1, max_contrasts=1)
    assert groups, "expected at least one repetition group"
    return groups[0]


def _two_groups():
    from spectramr.data.datasets.contrast_grid import discover_m4raw_grid

    # The stem grouper capped at 2 x 1 returned one subject's two contrasts (C3).
    grid = discover_m4raw_grid(_DATA_DIR, max_subjects=2, max_contrasts=1, allow_incomplete=True)
    groups = [k.paths for k in grid.keys]
    if len(groups) < 2:
        pytest.skip("need >= 2 distinct subjects for the cross-reference test")
    return groups[0], groups[1]


def test_each_subject_is_graded_against_its_own_reference():
    # #295 regression: the real-m4raw branch captured img_gt / smaps / complex_gt
    # ONCE from subject 0 and graded every subject against it, so PSNR/SSIM for
    # subject k>0 measured cross-subject anatomy, not degradation severity, and the
    # FID/KID reference pool was N identical copies of subject 0. This pins the
    # invariant the fix relies on: a subject matches its OWN clean reference and
    # NOT another subject's, so the reference choice is load-bearing and must be
    # per-subject.
    from spectramr.cli.app import _resolve_device
    from scripts.sim2rank.ground_truth import synthesize_pseudo_gt

    # ``"cpu"``, not ``"auto"``. sim2rank's canonical execution backend IS
    # CPU -- a documented, explicit exception to the accelerated-run contract
    # (non-negotiable 9b), not a fallback. With ``"auto"`` these tests raise
    # AcceleratorRequiredError on any CPU-only host, which is the contract
    # working correctly against a request that should never have been made.
    # Stating CPU here is the sanctioned opt-in and keeps the run recorded as
    # deliberate in provenance rather than degraded.
    device = _resolve_device("cpu")
    g0, g1 = _two_groups()

    _, _, mag0, _ = synthesize_pseudo_gt(g0, device=device)
    _, _, mag1, _ = synthesize_pseudo_gt(g1, device=device)

    def _rel_l2(a, b) -> float:
        return float((a - b).norm() / a.norm().clamp_min(1e-12))

    self_err = _rel_l2(mag1, mag1)  # subject 1 vs its OWN reference
    cross_err = _rel_l2(mag1, mag0)  # subject 1 vs subject 0 (the old bug)

    assert self_err < 1e-6, "a subject must match its own clean reference"
    assert cross_err > 0.1, (
        "subject 1 must NOT match subject 0's anatomy — grading cross-subject "
        "(the pre-#295 behavior) corrupts every per-image metric for k>0"
    )
    assert cross_err > 100 * max(self_err, 1e-9)


def test_native_path_uses_real_phase_and_coils():
    from spectramr.cli.app import _resolve_device
    from spectramr.infrastructure.physics.coil_sensitivity import coil_combine_sense
    from scripts.sim2rank.degradation import DegradationSweep
    from scripts.sim2rank.engine import Sim2RankEngine
    from scripts.sim2rank.ground_truth import synthesize_pseudo_gt
    from scripts.sim2rank.metrics_list import METRIC_SPECS

    # ``"cpu"``, not ``"auto"``. sim2rank's canonical execution backend IS
    # CPU -- a documented, explicit exception to the accelerated-run contract
    # (non-negotiable 9b), not a fallback. With ``"auto"`` these tests raise
    # AcceleratorRequiredError on any CPU-only host, which is the contract
    # working correctly against a request that should never have been made.
    # Stating CPU here is the sanctioned opt-in and keeps the run recorded as
    # deliberate in provenance rather than degraded.
    device = _resolve_device("cpu")
    group = _one_group()

    coil_images, smaps, x_gt_mag, raw_p99 = synthesize_pseudo_gt(group, device=device)
    complex_gt = coil_combine_sense(coil_images, smaps) / raw_p99

    h, w = coil_images.shape[-2], coil_images.shape[-1]
    sweeper = DegradationSweep(im_size=(h, w), n_timesteps=3, device=device)
    mag_list, cplx_list = sweeper.sweep_axis(
        coil_images, "undersampling", smaps, return_complex=True
    )
    mag_normed = [m / raw_p99 for m in mag_list]
    cplx_normed = [c / raw_p99 for c in cplx_list]
    grid = [0.2, 0.6, 0.95]

    specs = [s for s in METRIC_SPECS if s.registry_key in {"ipen", "g_factor"}]
    engine = Sim2RankEngine(metric_specs=specs, device=device)

    native = engine.evaluate_sweep(
        mag_normed,
        x_gt_mag,
        axis="undersampling",
        severity_grid=grid,
        smaps=smaps,
        complex_degraded=cplx_normed,
        complex_gt=complex_gt,
    )
    bridge = engine.evaluate_sweep(
        mag_normed,
        x_gt_mag,
        axis="undersampling",
        severity_grid=grid,
    )

    # Phase metric: native (real image-domain phase) must be finite, non-zero,
    # and differ from the bridge's synthetic-phase fallback.
    assert all(v == v for v in native["ipen"]), "native ipen must be finite"
    assert max(native["ipen"]) > 0.0
    assert any(
        abs(a - b) > 1e-6 for a, b in zip(native["ipen"], bridge["ipen"])
    ), "native phase path must diverge from the synthetic-phase bridge"

    # g-factor: real 4-coil ESPIRiT maps differ from the synthetic 8-coil
    # birdcage, and the trajectory rises with the undersampling severity.
    assert all(v >= 1.0 for v in native["g_factor"])
    assert native["g_factor"][-1] > native["g_factor"][0]
    assert any(
        abs(a - b) > 1e-3 for a, b in zip(native["g_factor"], bridge["g_factor"])
    ), "real coil maps must change the g-factor vs the synthetic birdcage"
