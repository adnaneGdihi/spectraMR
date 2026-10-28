"""Synthetic images, arms and certificate payloads for the image and certificate figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def make_arms() -> Any:
    """Two in-memory arms for the cross-run plotters: per-case rows and aligned cases."""
    from spectramr.infrastructure.reporting.comparison.arms import Arms
    from spectramr.infrastructure.reporting.metric_names import annotate_frame

    rng = np.random.default_rng(11)
    per_case = pd.DataFrame(
        [
            {
                "run_id": arm,
                "method": arm,
                "case_id": f"c{b}",
                "pair_key": f"R{r}#{b}",
                "acceleration": float(r),
                "metric": "val_psnr",
                "base_metric": "psnr",
                "value": 30 - r + shift + rng.normal(0, 0.2),
                "step": 10,
            }
            for arm, shift in (("a", 0.0), ("b", 1.0))
            for r in (2, 4)
            for b in range(6)
        ]
    )
    df = annotate_frame(
        pd.DataFrame(
            [
                {
                    "run_id": f"{arm}_run{i}",
                    "method": arm,
                    "split": "val",
                    "step": 10,
                    "metric": "val_psnr",
                    "value": 30.0 + i,
                }
                for i, arm in enumerate("ab")
            ]
        )
    )
    cases = make_image_cases(n=2)
    shared = [{"target": c["target"], "prediction": c["pred"], "acceleration": 2.0} for c in cases]
    return Arms(
        ["a", "b"],
        {},
        df,
        {"a": 10, "b": 10},
        per_case,
        {"a": shared, "b": [dict(c, prediction=c["target"] * 0.9) for c in shared]},
    )


#: (centre_x, centre_y, semi_x, semi_y, angle_deg, intensity): a head-like
#: ellipse phantom in the Shepp-Logan family, so example images read as anatomy.
_ELLIPSES = (
    (0.0, 0.0, 0.69, 0.92, 0.0, 1.0),
    (0.0, -0.018, 0.62, 0.87, 0.0, -0.75),
    (0.22, 0.0, 0.11, 0.31, -18.0, -0.2),
    (-0.22, 0.0, 0.16, 0.41, 18.0, -0.2),
    (0.0, 0.35, 0.21, 0.25, 0.0, 0.15),
    (0.0, -0.6, 0.05, 0.05, 0.0, 0.25),
)


def make_phantom(side: int = 64, seed: int = 0) -> np.ndarray:
    """Ellipse phantom in ``[0, 1]``; ``seed`` adds two small lesions per slice."""
    y, x = np.mgrid[-1 : 1 : side * 1j, -1 : 1 : side * 1j]
    img = np.zeros((side, side))
    rng = np.random.default_rng(seed)
    extra = [
        (rng.uniform(-0.3, 0.3), rng.uniform(-0.4, 0.4), 0.06, 0.06, 0.0, 0.2) for _ in range(2)
    ]
    for cx, cy, ax, ay, ang, val in (*_ELLIPSES, *extra):
        t = np.deg2rad(ang)
        xr, yr = (
            (x - cx) * np.cos(t) + (y - cy) * np.sin(t),
            -(x - cx) * np.sin(t) + (y - cy) * np.cos(t),
        )
        img[(xr / ax) ** 2 + (yr / ay) ** 2 <= 1.0] += val
    return np.clip(img / img.max(), 0.0, 1.0).astype(np.float32)


def undersample(image: np.ndarray, acceleration: int = 4, centre: int = 6) -> np.ndarray:
    """Zero-filled magnitude: keep every ``acceleration``-th phase-encode line plus the centre."""
    import torch

    from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

    k = fft2c(torch.from_numpy(image.astype(np.complex64)))
    mask = torch.zeros(image.shape[0], 1)
    mask[::acceleration] = 1
    mid = image.shape[0] // 2
    mask[mid - centre // 2 : mid + centre // 2] = 1
    return ifft2c(k * mask).abs().numpy().astype(np.float32)


def make_image_cases(n: int = 3, h: int = 64, w: int = 64) -> list[dict[str, Any]]:
    """Image cases for the qualitative figures: phantom target, undersampled input."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(3)
    cases: list[dict[str, Any]] = []
    for i in range(n):
        target = make_phantom(h, seed=i)
        zero_filled = undersample(target, acceleration=4)
        pred = np.clip(gaussian_filter(target, 0.6) + rng.normal(0, 0.02, target.shape), 0, 1)
        pred = pred.astype(np.float32)
        baseline = (0.5 * zero_filled + 0.5 * pred).astype(np.float32)
        cases.append(
            {
                "name": f"case_{i:02d}",
                "case_id": f"example_{i:02d}",
                "image": target,
                "input": zero_filled,
                "pred": pred,
                "predicted": pred,
                "prediction": pred,
                "target": target,
                "residual": float(np.abs(pred - target).mean()),
                "metrics": {},
                # SR keys: a 2x box-downsampled input
                "lr": target.reshape(h // 2, 2, w // 2, 2)
                .mean(axis=(1, 3))
                .repeat(2, 0)
                .repeat(2, 1),
                "hr": target,
                # k-space reconstruction keys
                "zero_filled": zero_filled,
                "baseline": baseline,
                "fs_ref": target,
            }
        )
    return cases


def make_cohort_dict() -> dict[str, Any]:
    return {
        "train_n": 80,
        "val_n": 10,
        "test_n": 10,
        "age_mean": 45.0,
        "age_std": 12.0,
        "sex_split": "55/45 F/M",
        "scanners": "3T Siemens",
        "pathology": "knee MRI",
        "split_rule": "subject-level (no slice leakage)",
    }


def make_qmap_slices() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(12)
    return {
        "euclidean": np.abs(rng.normal(0, 0.2, (48, 48))).astype(np.float32),
        "riemannian": np.abs(rng.normal(0, 0.11, (48, 48))).astype(np.float32),
    }


def make_beltrami_mu(n: int = 64) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n] / n
    mag = 0.6 * np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.05)
    return (mag * np.exp(1j * 2 * np.pi * (xx + yy))).astype(np.complex64)


def make_fingerprint_embedding(n_per: int = 60, k: int = 4):
    rng = np.random.default_rng(13)
    embs, labels = [], []
    for c in range(k):
        centre = rng.normal(0, 3, 5)
        embs.append(centre + rng.normal(0, 0.6, (n_per, 5)))
        labels.append(np.full(n_per, c))
    return np.vstack(embs).astype(np.float32), np.concatenate(labels)


def make_ksd_result():
    """Minimal object with the fields ``render_ksd_certificate`` reads."""
    import types

    return types.SimpleNamespace(
        passed=True,
        ksd_squared=1.2e-3,
        p_value=0.31,
        threshold=0.05,
        n_samples=256,
        bandwidth=1.0,
        message="synthetic KSD result",
    )


def make_certificate_paths():
    """A partial certificate (R1 payload) so the summary figure renders."""
    import json
    import tempfile

    # one reused directory, not a new temp dir per call
    d = Path(tempfile.gettempdir()) / "spectramr_catalog_examples"
    d.mkdir(exist_ok=True)
    (d / "r1.json").write_text(json.dumps({"empirical_coverage": 0.91, "alpha": 0.1}))
    return {"r1_path": d / "r1.json"}


def make_acquisition_arms():
    from spectramr.infrastructure.reporting.plotters.learnable_acquisition_pareto import (
        ArmResult,
    )

    return [
        ArmResult("full", 0.5, 33.0, 0.92),
        ArmResult("mid", 0.25, 31.0, 0.88),
        ArmResult("low", 0.125, 28.0, 0.82),
    ]


def make_slice_df(n_arms: int = 3, n_subjects: int = 6, n_slices: int = 8) -> Any:
    """Long per-slice rows (``Arms.per_case`` schema) for the distribution figures."""
    import pandas as pd

    rng = np.random.default_rng(5)
    rows = []
    for a in range(n_arms):
        for s in range(n_subjects):
            for k in range(n_slices):
                case = f"sub{s}|T{1 + k % 2}|{k}|0"
                base = rng.normal(24 + a, 2.0)
                values = {
                    "psnr": base,
                    "ssim": float(np.clip(0.3 + 0.02 * base + rng.normal(0, 0.03), 0, 1)),
                    "nmse": float(np.exp(-base / 8) + abs(rng.normal(0, 0.01))),
                    "hfen": float(abs(rng.normal(0.6, 0.1))),
                    "mae": float(abs(rng.normal(0.05, 0.01))),
                    "bias_squared": float(np.exp(rng.normal(-7, 0.6))),
                    "rep_variance": float(np.exp(rng.normal(-7, 0.6))),
                }
                rows += [
                    {
                        "method": f"arm{a}",
                        "case_id": case,
                        "base_metric": m,
                        "value": v,
                        "acceleration": float("nan"),
                        "pair_key": f"slice#{case}",
                    }
                    for m, v in values.items()
                ]
    return pd.DataFrame(rows)


def make_checkpoint_runs(n_arms: int = 2, steps: tuple[int, ...] = (50, 2000, 4000)) -> dict[str, Any]:
    """Run directories holding tiny safetensors checkpoints (``best`` plus periodic steps)."""
    import tempfile
    from pathlib import Path

    import torch
    from safetensors.torch import save_file

    root = Path(tempfile.mkdtemp(prefix="report_example_ckpt_"))
    runs: dict[str, Any] = {}
    for a in range(n_arms):
        folder = root / f"arm{a}" / "checkpoints"
        folder.mkdir(parents=True)
        gen = torch.Generator().manual_seed(a)
        theta = {f"{g}.{i}.weight": torch.randn(8, 8, generator=gen) for g in ("encoder", "decoder") for i in range(3)}
        for k, step in enumerate((*steps, None)):
            moved = {
                f"{n}:models.generator.{name}": w + 0.05 * (k + 1) * torch.randn(w.shape, generator=gen)
                for n, (name, w) in enumerate(theta.items())
            }
            moved[f"{len(moved)}:optim.generator.state[0].step"] = torch.tensor(1.0)
            save_file(moved, str(folder / ("best.safetensors" if step is None else f"step_{step:08d}.safetensors")))
        runs[f"arm{a}"] = root / f"arm{a}"
    return runs


def make_png_run(n_passes: int = 4, size: int = 48) -> Any:
    """A run directory with the validation PNG pairs every pass writes."""
    import tempfile
    from pathlib import Path

    import matplotlib.image as mpimg

    run = Path(tempfile.mkdtemp(prefix="report_example_png_"))
    real, fake = run / "logs" / "metrics" / "real_images", run / "logs" / "metrics" / "fake_images"
    real.mkdir(parents=True)
    fake.mkdir(parents=True)
    rng = np.random.default_rng(2)
    target = make_phantom(size, seed=0)
    for k in range(n_passes):
        stem = f"validation_epoch{k:03d}_step{(k + 1) * 2000:06d}_20260101_000000"
        mpimg.imsave(str(real / f"{stem}_real_000.png"), target, cmap="gray", vmin=0, vmax=1)
        noisy = np.clip(target + rng.normal(0, 0.2 / (k + 1), target.shape), 0, 1)
        mpimg.imsave(str(fake / f"{stem}_fake_000.png"), noisy, cmap="gray", vmin=0, vmax=1)
    return run


def make_phase_table() -> pd.DataFrame:
    """Two arms, three metrics, across training, validation and inference (spread on two phases)."""
    rows = [
        {
            "arm": arm,
            "phase": phase,
            "metric": metric,
            "mean": base + shift + 0.5 * i,
            "std": 0.0 if phase == "val" else 0.15,
            "n": 1 if phase == "val" else 20,
            "unit": "example",
            "step": 100,
            "direction": "lower" if metric == "cjv" else "higher",
        }
        for arm, shift in (("a", 0.0), ("b", 1.0))
        for i, (metric, base) in enumerate((("psnr", 30.0), ("ssim", 0.8), ("cjv", 0.4)))
        for phase in ("train", "val", "test")
    ]
    frame = pd.DataFrame(rows)
    frame.loc[frame["phase"] == "val", "std"] = float("nan")
    return frame


def make_paired_subjects() -> dict:
    """``{(phase, metric, variant): PairedSubjects}`` for arms b and c against baseline a."""
    from spectramr.infrastructure.reporting.per_subject_stats import PairedSubjects

    rng = np.random.default_rng(5)
    out = {}
    for phase in ("val", "test"):
        for metric, base in (("psnr", 30.0), ("nmse", 0.05)):
            for variant, effect in (("b", 0.8), ("c", -0.2)):
                subjects = [f"s{i}" for i in range(8)]
                b = base + rng.normal(0, 1.0 if metric == "psnr" else 0.01, 8)
                a = (
                    b
                    + effect * (1 if metric == "psnr" else -0.01)
                    + rng.normal(0, 0.2 if metric == "psnr" else 0.002, 8)
                )
                table = pd.DataFrame({"a": a, "b": b}, index=pd.Index(subjects, name="subject_id"))
                out[(phase, metric, variant)] = PairedSubjects(
                    table=table,
                    n_slices=8,
                    nonfinite_dropped={"a": 0, "b": 0},
                    unpaired_slices={"a": 0, "b": 0},
                    scale="metric",
                )
    return out


def make_paired_rows() -> pd.DataFrame:
    """The ``paired_stats.csv`` rows of :func:`make_paired_subjects`, tested with the real code."""
    from spectramr.infrastructure.reporting.comparison.contrast_stats import (
        PAIRED_COLUMNS,
        _row,
        apply_holm,
    )
    from spectramr.infrastructure.reporting.per_subject_stats import compare_subjects

    rows = []
    for (phase, metric, variant), paired in make_paired_subjects().items():
        rows.append(
            _row(
                "example",
                metric,
                variant,
                "a",
                paired,
                compare_subjects(paired, n_bootstrap=200),
                "family",
                phase,
                (None, None),
            )
        )
    apply_holm(rows)
    return pd.DataFrame(rows, columns=list(PAIRED_COLUMNS))
