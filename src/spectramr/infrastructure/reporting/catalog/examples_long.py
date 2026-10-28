"""The long-format metric frame ``aggregator.aggregate`` produces, synthesised."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def make_long_metrics_df() -> pd.DataFrame:
    """Long-format DataFrame matching ``aggregator.aggregate*`` output.

    Contains the columns the Phase-1 plotters look at:
    ``method``, ``metric``, ``step``, ``value``, ``split``, ``subject_id``.
    Includes the metric names referenced by every Phase-1 / generative
    plotter so each can find at least one matching row.
    """
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []

    methods = ["full", "ablated_a", "ablated_b"]
    steps = list(range(0, 100, 10))

    # Per-step training-curve metrics (loss family + components).
    curve_metrics = [
        "loss",
        "val_loss",
        "reconstruction_loss",
        "kl_loss",
        "g_loss",
        "d_loss",
        "d_acc",
        "gp_norm",
    ]
    for method in methods:
        for step in steps:
            for m in curve_metrics:
                base = 1.0 - step / 200.0
                noise = rng.normal(0, 0.02)
                value = max(base + noise, 1e-3)
                rows.append(
                    {
                        "method": method,
                        "metric": m,
                        "step": float(step),
                        "value": float(value),
                        "split": "train" if not m.startswith("val_") else "val",
                        "subject_id": None,
                    }
                )
        # Per-step quality metrics on BOTH splits so the train/val-gap plotter
        # can match base names and the metric-correlation heatmap has >=2 varying
        # metrics over >=3 val steps.
        for step in steps:
            frac = step / max(steps)
            for base, (lo, hi) in {
                "psnr": (24.0, 34.0),
                "ssim": (0.6, 0.92),
                "nmse": (0.05, 0.008),
            }.items():
                train_v = lo + (hi - lo) * frac + rng.normal(0, 0.02 * abs(hi - lo))
                val_v = train_v - 0.06 * abs(hi - lo) + rng.normal(0, 0.02 * abs(hi - lo))
                rows.append(
                    {
                        "method": method,
                        "metric": base,
                        "step": float(step),
                        "value": float(train_v),
                        "split": "train",
                        "subject_id": None,
                    }
                )
                rows.append(
                    {
                        "method": method,
                        "metric": f"val_{base}",
                        "step": float(step),
                        "value": float(val_v),
                        "split": "val",
                        "subject_id": None,
                    }
                )
        # Run-summary / best / final scalar rows — mirror the aggregator's folding
        # of run_summary.json (split="run") + final_metrics.json (best/final) so
        # the run-summary card fires exactly as on a real run.
        for name, v in {
            "params_m": rng.uniform(5, 80),
            "iterations_per_sec": rng.uniform(2, 12),
            "duration_min": rng.uniform(30, 300),
            "effective_batch": float(rng.choice([8, 16, 32])),
        }.items():
            rows.append(
                {
                    "method": method,
                    "metric": name,
                    "step": -1.0,
                    "value": float(v),
                    "split": "run",
                    "subject_id": None,
                }
            )
        for name, (lo, hi) in {"psnr": (30.0, 36.0), "ssim": (0.85, 0.95)}.items():
            rows.append(
                {
                    "method": method,
                    "metric": name,
                    "step": -1.0,
                    "value": float(rng.uniform(lo, hi)),
                    "split": "best",
                    "subject_id": None,
                }
            )
        for name, v in {
            "final_loss": rng.uniform(0.05, 0.2),
            "early_stopping_best_value": rng.uniform(0.8, 0.92),
        }.items():
            rows.append(
                {
                    "method": method,
                    "metric": name,
                    "step": -1.0,
                    "value": float(v),
                    "split": "final",
                    "subject_id": None,
                }
            )

    # Per-step diffusion / generative binned metrics.
    for binned in ("loss_per_timestep", "psnr_vs_nfe", "psnr_vs_cfg"):
        for step in steps:
            rows.append(
                {
                    "method": "full",
                    "metric": binned,
                    "step": float(step),
                    "value": float(rng.uniform(20, 35)),
                    "split": "train",
                    "subject_id": None,
                }
            )

    # Per-latent KL for VAE collapse check.
    for latent_idx in range(4):
        rows.append(
            {
                "method": "full",
                "metric": f"kl_latent_{latent_idx}",
                "step": float(max(steps)),
                "value": float(rng.uniform(0.0, 1.0)),
                "split": "train",
                "subject_id": None,
            }
        )

    # Scalar evaluation metrics (per-subject, no step). The Pareto and
    # ablation plotters consume these.
    for method in methods:
        for subj in range(8):
            rows.append(
                {
                    "method": method,
                    "metric": "psnr",
                    "step": float("nan"),
                    "value": float(rng.uniform(28, 36)),
                    "split": "test",
                    "subject_id": f"s{subj:03d}",
                }
            )
            rows.append(
                {
                    "method": method,
                    "metric": "ssim",
                    "step": float("nan"),
                    "value": float(rng.uniform(0.7, 0.95)),
                    "split": "test",
                    "subject_id": f"s{subj:03d}",
                }
            )
        # Cost axis used by the headline Pareto.
        rows.append(
            {
                "method": method,
                "metric": "params_m",
                "step": float("nan"),
                "value": float(rng.uniform(5, 80)),
                "split": "meta",
                "subject_id": None,
            }
        )

    # Computational-profile metrics.
    for step in steps:
        for m in ("data_load_s", "forward_s", "backward_s", "optim_s", "log_s"):
            rows.append(
                {
                    "method": "full",
                    "metric": m,
                    "step": float(step),
                    "value": float(rng.uniform(0.005, 0.05)),
                    "split": "train",
                    "subject_id": None,
                }
            )
        for m in ("gpu_mem_gb", "cpu_mem_gb"):
            rows.append(
                {
                    "method": "full",
                    "metric": m,
                    "step": float(step),
                    "value": float(rng.uniform(1.0, 12.0)),
                    "split": "train",
                    "subject_id": None,
                }
            )

    return pd.DataFrame(rows)
