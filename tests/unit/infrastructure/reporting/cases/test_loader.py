import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import numpy as np

from spectramr.infrastructure.reporting.cases.loader import (
    load_image_pair_cases,
    load_report_cases,
)
from spectramr.infrastructure.reporting.cases.recorder import ReportCaseRecorder


def _make_cases(tmp_path):
    rec = ReportCaseRecorder(n_cases=3, selection="best_median_worst",
                             primary_metric="ssim", higher_is_better=True)
    for i in range(6):
        rec.observe(
            case_id=f"s{i}",
            arrays={
                "input": np.zeros((8, 8), dtype=np.float32),
                "prediction": np.full((8, 8), float(i) / 6, dtype=np.float32),
                "target": np.ones((8, 8), dtype=np.float32),
            },
            metrics={"ssim": float(i) / 6, "psnr": 20.0 + i},
            domain={"acceleration": 4},
        )
    rec.write(tmp_path, subdir="report_cases")


def test_loader_routes_reconstruction(tmp_path):
    _make_cases(tmp_path)
    payload = load_report_cases(tmp_path, task="reconstruction")
    assert "cases_kspace" in payload
    assert len(payload["cases_kspace"]) == 3
    case = payload["cases_kspace"][0]
    assert set(case) >= {"prediction", "target", "metrics", "rank"}
    assert "predictions_df" in payload


def test_loader_missing_dir_returns_empty(tmp_path):
    assert load_report_cases(tmp_path, task="reconstruction") == {}


def test_loader_preserves_3d_volume_arrays(tmp_path):
    """A recorded ``*_volume`` (3-D) array must survive the round-trip so the
    interactive 3-D viewer can consume it."""
    rec = ReportCaseRecorder(n_cases=1, selection="first", primary_metric="psnr",
                             higher_is_better=True, record_volumes=True)
    rec.observe(case_id="s0",
                arrays={"input": np.zeros((8, 8), np.float32),
                        "prediction": np.ones((8, 8), np.float32),
                        "target": np.ones((8, 8), np.float32),
                        "prediction_volume": np.ones((5, 8, 8), np.float32),
                        "target_volume": np.zeros((5, 8, 8), np.float32)},
                metrics={"psnr": 30.0}, domain={})
    rec.write(tmp_path, subdir="report_cases")
    payload = load_report_cases(tmp_path, task="reconstruction")
    case = payload["cases_kspace"][0]
    assert case["prediction_volume"].shape == (5, 8, 8)
    assert case["target_volume"].shape == (5, 8, 8)
    assert case["prediction"].ndim == 2  # the 2-D representative is untouched


def test_loader_sr_route_supplies_lr_hr_aliases(tmp_path):
    _make_cases(tmp_path)
    payload = load_report_cases(tmp_path, task="super_resolution")
    assert "cases_sr" in payload
    case = payload["cases_sr"][0]
    # SR triptych plotter reads lr/pred/hr by direct subscript
    assert {"lr", "pred", "hr"} <= set(case)


def test_loader_synthesis_route_has_input_pred_target(tmp_path):
    _make_cases(tmp_path)
    payload = load_report_cases(tmp_path, task="synthesis")
    assert "cases_synth" in payload
    case = payload["cases_synth"][0]
    # synthesis plotter reads input/pred/target by direct subscript
    assert {"input", "pred", "target"} <= set(case)


def _write_png_pairs(tmp_path, epoch=2, step=100, n=3):
    real_dir = tmp_path / "metrics" / "real_images"
    fake_dir = tmp_path / "metrics" / "fake_images"
    real_dir.mkdir(parents=True, exist_ok=True); fake_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        stem = f"validation_epoch{epoch:03d}_step{step:06d}_20260101_000000"
        mpimg.imsave(real_dir / f"{stem}_real_{i:03d}.png",
                     rng.random((8, 8)), cmap="gray")
        mpimg.imsave(fake_dir / f"{stem}_fake_{i:03d}.png",
                     rng.random((8, 8)), cmap="gray")


def test_png_pair_fallback_recovers_cases(tmp_path):
    _write_png_pairs(tmp_path)
    cases = load_image_pair_cases(tmp_path)
    assert len(cases) == 3
    c = cases[0]
    assert {"prediction", "target"} <= set(c)
    assert c["prediction"].ndim == 2


def test_png_fallback_invents_no_input_and_no_baseline(tmp_path):
    """No input PNG is saved. Copying the target into ``input`` showed the answer
    under the label "input", and aliasing ``baseline`` onto it titled the
    zero-filled image "CG-SENSE" for a baseline that never ran."""
    _write_png_pairs(tmp_path)
    c = load_image_pair_cases(tmp_path)[0]
    assert "input" not in c and "zero_filled" not in c and "baseline" not in c


def test_png_fallback_reads_the_requested_pass(tmp_path):
    _write_png_pairs(tmp_path, epoch=1, step=50, n=2)
    _write_png_pairs(tmp_path, epoch=3, step=300, n=4)
    cases = load_image_pair_cases(tmp_path, step=120)
    assert len(cases) == 2 and cases[0]["case_id"].startswith("epoch001_step000050")


def test_unranked_png_cases_are_not_failures(tmp_path):
    _write_png_pairs(tmp_path)
    assert load_report_cases(tmp_path, task="reconstruction")["cases_failure"] == []


def test_loader_falls_back_to_png_pairs_when_no_npz(tmp_path):
    # no report_cases/ dir → loader must recover from PNG pairs
    _write_png_pairs(tmp_path)
    payload = load_report_cases(tmp_path, task="reconstruction")
    assert "cases_kspace" in payload
    assert len(payload["cases_kspace"]) == 3


def test_png_fallback_picks_latest_group(tmp_path):
    _write_png_pairs(tmp_path, epoch=1, step=50, n=2)
    _write_png_pairs(tmp_path, epoch=3, step=300, n=4)
    cases = load_image_pair_cases(tmp_path)
    # latest (epoch3/step300) group has 4 pairs
    assert len(cases) == 4
    assert cases[0]["case_id"].startswith("epoch003_step000300")


def test_png_fallback_empty_when_no_images(tmp_path):
    assert load_image_pair_cases(tmp_path) == []


def _write_pair(base, step=100):
    """Write one real/fake PNG pair under ``base`` the way MetricsTracker does."""
    real, fake = base / "real_images", base / "fake_images"
    real.mkdir(parents=True)
    fake.mkdir(parents=True)
    img = np.zeros((8, 8), dtype=np.float32)
    mpimg.imsave(str(real / f"epoch001_step{step:06d}_real_000.png"), img, cmap="gray")
    mpimg.imsave(str(fake / f"epoch001_step{step:06d}_fake_000.png"), img, cmap="gray")
    return real, fake


def test_a_profiling_runs_pngs_are_not_adopted_as_the_arms_cases(tmp_path):
    """`spectramr profile` writes a throwaway run under `profiles/<id>/run/`.

    Its PNGs are a *measurement* of the arm, at a truncated iteration budget and
    often a capped validation loop. Reported as the arm's validation cases they
    would be indistinguishable from real results. The recursive fallback in
    `_find_image_dirs` reaches them, and it fires precisely when the arm has no
    images of its own -- i.e. exactly when the throwaway would be adopted.
    """
    from spectramr.infrastructure.reporting.cases.loader import _find_image_dirs

    arm = tmp_path / "experiments" / "results" / "exp_11"
    _write_pair(arm / "profiles" / "train-full-all-20260824" / "run" / "metrics")

    assert _find_image_dirs(arm) == (None, None)
    assert load_image_pair_cases(arm) == []


def test_the_arms_own_nested_pngs_are_still_recovered(tmp_path):
    """Discrimination: the filter must not disable the fallback it guards.

    A test that only asserts the profiling case is dropped stays green if
    `_find_image_dirs` were made to return `(None, None)` unconditionally.
    """
    from spectramr.infrastructure.reporting.cases.loader import _find_image_dirs

    arm = tmp_path / "experiments" / "results" / "exp_11"
    real, fake = _write_pair(arm / "some" / "nested" / "download")

    assert _find_image_dirs(arm) == (real, fake)
    assert len(load_image_pair_cases(arm)) == 1


def test_a_real_run_beside_a_profiling_run_still_reports_its_own_images(tmp_path):
    """The mixed tree: both present. The arm's own must win, not merely survive."""
    from spectramr.infrastructure.reporting.cases.loader import _find_image_dirs

    arm = tmp_path / "experiments" / "results" / "exp_11"
    _write_pair(arm / "profiles" / "run-abc" / "run" / "metrics", step=999)
    real, fake = _write_pair(arm / "metrics")

    assert _find_image_dirs(arm) == (real, fake)

def test_png_fallback_reads_nothing_when_every_pass_is_later(tmp_path):
    _write_png_pairs(tmp_path, epoch=1, step=500, n=2)
    assert load_image_pair_cases(tmp_path, step=100) == []


def test_recorded_cases_are_read_at_one_pass():
    """The recorder pools cases over every pass; a panel must show one model (#2503)."""
    from spectramr.infrastructure.reporting.cases.loader import _cases_at_pass

    cases = [{"case_id": cid} for cid in (
        "val_step70000_R2x", "val_step70000_R8x", "val_step35000_R32x", "val_step17500_R2x"
    )]
    assert [c["case_id"] for c in _cases_at_pass(cases, 70000)] == [
        "val_step70000_R2x", "val_step70000_R8x"
    ]
    assert [c["case_id"] for c in _cases_at_pass(cases, 40000)] == ["val_step35000_R32x"]
    assert _cases_at_pass(cases, 100) == []
    assert _cases_at_pass([{"case_id": "c0"}], 100) == [{"case_id": "c0"}]

def test_without_a_step_the_latest_recorded_pass_is_read():
    from spectramr.infrastructure.reporting.cases.loader import _cases_at_pass

    cases = [{"case_id": c} for c in ("val_step100_R2x", "val_step300_R2x", "val_step300_R8x")]
    assert [c["case_id"] for c in _cases_at_pass(cases, 10**12)] == [
        "val_step300_R2x", "val_step300_R8x"
    ]
