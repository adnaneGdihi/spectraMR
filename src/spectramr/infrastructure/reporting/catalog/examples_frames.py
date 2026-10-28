"""Synthetic long-format frames: one well-formed input per figure family."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def make_predictions_df() -> pd.DataFrame:
    """``predicted`` vs ``target`` long table for regression-style plotters."""
    rng = np.random.default_rng(1)
    n = 200
    target = rng.uniform(0.0, 1.0, size=n)
    predicted = target + rng.normal(0.0, 0.05, size=n)
    return pd.DataFrame({"predicted": predicted, "target": target})


def make_stratified_df() -> pd.DataFrame:
    """Per-subgroup scalar metric table for stratified-performance plotter."""
    rng = np.random.default_rng(2)
    rows: list[dict[str, Any]] = []
    for method in ("full", "ablated_a"):
        for group in ("pediatric", "adult", "geriatric"):
            for _ in range(6):
                rows.append(
                    {
                        "method": method,
                        "metric": "psnr",
                        "subgroup": group,
                        "value": float(rng.uniform(28, 36)),
                    }
                )
    return pd.DataFrame(rows)


def make_metric_frame() -> pd.DataFrame:
    """Per-subject scalar metric table (distribution / significance / forest)."""
    rng = np.random.default_rng(4)
    rows: list[dict[str, Any]] = []
    for method in ("full", "ablated_a", "ablated_b"):
        for subj in range(8):
            for metric, lo, hi in (("psnr", 28, 36), ("ssim", 0.7, 0.95)):
                rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "value": float(rng.uniform(lo, hi)),
                        "subject_id": f"s{subj:03d}",
                        "split": "test",
                    }
                )
    return pd.DataFrame(rows)


def make_accel_frame() -> pd.DataFrame:
    """Metric vs acceleration factor (acceleration_sweep)."""
    rng = np.random.default_rng(5)
    rows: list[dict[str, Any]] = []
    for method in ("full", "ablated_a"):
        for r in (2, 4, 8, 16):
            for _ in range(4):
                rows.append(
                    {
                        "method": method,
                        "metric": "psnr",
                        "acceleration": r,
                        "value": float(36 - 1.4 * np.log2(r) + rng.normal(0, 0.3)),
                    }
                )
    return pd.DataFrame(rows)


def make_calibration_frame() -> pd.DataFrame:
    """Nominal vs empirical coverage (calibration_coverage)."""
    rng = np.random.default_rng(6)
    nominal = np.linspace(0.05, 0.95, 10)
    rows: list[dict[str, Any]] = []
    for method in ("full", "ablated_a"):
        bias = 0.0 if method == "full" else 0.06
        for nm in nominal:
            rows.append(
                {
                    "method": method,
                    "nominal": float(nm),
                    "empirical": float(np.clip(nm - bias + rng.normal(0, 0.01), 0, 1)),
                }
            )
    return pd.DataFrame(rows)


def make_heatmap_frame() -> pd.DataFrame:
    """Two-axis ablation grid (ablation_heatmap)."""
    rng = np.random.default_rng(7)
    rows: list[dict[str, Any]] = []
    for x in (16, 32, 64):
        for y in ("l1", "l1+ssim", "l1+adv"):
            rows.append({"axis_x": x, "axis_y": y, "value": float(rng.uniform(28, 35))})
    return pd.DataFrame(rows)


def make_bland_frame() -> pd.DataFrame:
    """Reference vs measurement pairs (bland_altman)."""
    rng = np.random.default_rng(8)
    ref = rng.uniform(0.3, 0.9, 80)
    return pd.DataFrame(
        {"metric": "ssim", "reference": ref, "value": ref + rng.normal(0, 0.03, 80)}
    )


def make_acquisition_frame() -> pd.DataFrame:
    """policy / step / psnr_db for competing acquisition policies (fig_2_15)."""
    rng = np.random.default_rng(9)
    rows: list[dict[str, Any]] = []
    for policy, gain in (("IB-active", 1.0), ("BALD", 0.85), ("random", 0.6)):
        for step in range(1, 17):
            rows.append(
                {
                    "policy": policy,
                    "step": step,
                    "psnr_db": float(22 + gain * 6 * np.log1p(step) + rng.normal(0, 0.2)),
                }
            )
    return pd.DataFrame(rows)


def make_bloch_frame() -> pd.DataFrame:
    """tissue / bloch_eq_residual / step (fig_b3)."""
    rng = np.random.default_rng(10)
    rows: list[dict[str, Any]] = []
    for tissue, b0 in {"WM": 0.08, "GM": 0.10, "CSF": 0.25, "fat": 0.18}.items():
        for step in range(0, 1000, 100):
            for _ in range(4):
                rows.append(
                    {
                        "tissue": tissue,
                        "step": step,
                        "bloch_eq_residual": float(
                            b0 * np.exp(-step / 600) * (1 + rng.uniform(-0.2, 0.2))
                        ),
                    }
                )
    return pd.DataFrame(rows)


def make_qmap_frame() -> pd.DataFrame:
    """method / per-pixel error (fig_b7)."""
    rng = np.random.default_rng(11)
    rows: list[dict[str, Any]] = []
    for method, scale in (("euclidean", 0.20), ("riemannian", 0.11)):
        for _ in range(400):
            rows.append({"method": method, "error": float(abs(rng.normal(0, scale)) + 1e-3)})
    return pd.DataFrame(rows)


def make_task_ablation_frame() -> pd.DataFrame:
    """Tidy ``[family, role, arm, metric, value]`` frame for ``cohort_task_ablation_bars``.

    Two families, method + one/two ablation control(s) each, on the three
    faceted validation metrics — matches the plotter's documented contract.
    """
    rng = np.random.default_rng(14)
    rows: list[dict[str, Any]] = []
    families = {"b1": ["b1_full", "b1_ablate_x"], "b2": ["b2_full", "b2_ablate_y", "b2_ablate_z"]}
    for fam, arms in families.items():
        for arm in arms:
            role = "ablation" if "_ablate_" in arm else "method"
            for metric, (lo, hi) in {
                "ssim": (0.6, 0.9),
                "psnr": (24.0, 34.0),
                "lpips": (0.1, 0.4),
            }.items():
                rows.append(
                    {
                        "family": fam,
                        "role": role,
                        "arm": f"mrixfields_{arm}",
                        "metric": metric,
                        "value": float(rng.uniform(lo, hi)),
                    }
                )
    return pd.DataFrame(rows)
