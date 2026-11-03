"""Step-wise training rows for the training-dynamics catalog examples (synthetic)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_dynamics_df(n_arms: int = 3, steps: int = 40) -> pd.DataFrame:
    """Tidy ``split, step, metric, value, method`` rows: bands, input scores, lr, loss, NaN fixes."""
    rng = np.random.default_rng(11)
    rows = []

    def add(method, split, step, metric, value):
        rows.append(
            {
                "method": method,
                "split": split,
                "step": float(step),
                "metric": metric,
                "value": float(value),
            }
        )

    for a in range(n_arms):
        arm = f"arm{a}"
        for i in range(steps):
            step, t = 500 * i, i / steps
            lr = 1e-4 * min(1.0, (i + 1) / 5) * 0.5 * (1 + np.cos(np.pi * t))
            loss = (
                0.02 * np.exp(-3 * t)
                + 0.004
                + abs(rng.normal(0, 4e-4))
                + (0.03 if i in (11, 27) and a == 1 else 0)
            )
            add(arm, "train", step, "lr_opt_g", lr)
            add(arm, "train", step, "loss", loss)
            add(
                arm,
                "train",
                step,
                "loss_nonfinite_substitutions",
                float(a == 2 and i in (8, 9, 30)),
            )
            if i % 4:
                continue
            for band, floor, rate in (
                ("low", 2e-4, 6.0),
                ("mid", 9e-4, 3.0 - a),
                ("high", 3e-3, 0.8 + 0.6 * a),
            ):
                add(
                    arm,
                    "val",
                    step,
                    f"val_band_corrected_mse_{band}",
                    floor * (1 + 4 * np.exp(-rate * t)),
                )
            add(
                arm,
                "val",
                step,
                "val_psnr",
                24 + (2 + a) * (1 - np.exp(-4 * t)) + rng.normal(0, 0.1),
            )
            add(arm, "val", step, "val_zf_psnr", 24 + rng.normal(0, 0.05))
            add(arm, "val", step, "val_ssim", 0.70 + (0.08 + 0.02 * a) * (1 - np.exp(-4 * t)))
            add(arm, "val", step, "val_zf_ssim", 0.71)
    return pd.DataFrame(rows)
