"""Synthetic run directories in the layout a finished training run writes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

STEPS = (100, 200, 300)
ACCELS = (2, 8)
N_CALLS = 6
N_SLICES = 2


def _slice(i: int) -> np.ndarray:
    yy, xx = np.mgrid[0:32, 0:32]
    return np.clip(np.sin(xx / (3 + i)) * np.cos(yy / 4) * 0.5 + 0.5, 0, 1)


def make_run(
    root: Path,
    name: str,
    *,
    shift: float = 0.0,
    selected: int | None = 200,
    per_call: bool = True,
    pngs: bool = True,
    noise: float = 0.05,
    reference_seed: int = 0,
) -> Path:
    run = root / name
    (run / "logs").mkdir(parents=True)
    pd.DataFrame({"iteration": list(STEPS), "loss": [0.3, 0.2, 0.15]}).to_csv(
        run / "logs" / "training_metrics.csv", index=False
    )
    pd.DataFrame(
        {
            "iteration": list(STEPS),
            "val_psnr": [25 + shift + s / 100 for s in STEPS],
            "val_ssim": [0.7 + shift / 100 + s / 10_000 for s in STEPS],
            "val_zf_psnr": [24.0] * len(STEPS),
        }
    ).to_csv(run / "logs" / "validation_metrics.csv", index=False)
    payload: dict = {"schema_version": "1", "best": {"val_psnr_best": 99.0}}
    if selected is not None:
        payload["early_stopping_best_iteration"] = selected
    (run / "final_metrics.json").write_text(json.dumps(payload))
    if per_call:
        rng = np.random.default_rng(len(name))
        rows = [
            {
                "case_id": f"val_step{s}_R{r}x",
                "split": "val",
                "step": s,
                "acceleration_level": float(r),
                "batch_index": b,
                "heldout": False,
                "val_psnr": 40 - 2 * r + shift + 0.3 * b + rng.normal(0, 0.05),
                "val_zf_psnr": 30 - 2 * r + 0.3 * b,
            }
            for s in STEPS
            for r in ACCELS
            for b in range(N_CALLS)
        ]
        pd.DataFrame(rows).to_csv(run / "per_call_metrics.csv", index=False)
    if pngs:
        real, fake = run / "metrics" / "real_images", run / "metrics" / "fake_images"
        real.mkdir(parents=True)
        fake.mkdir(parents=True)
        rng = np.random.default_rng(reference_seed + len(name))
        for s in STEPS:
            for r in ACCELS:
                stem = f"validation_R{r}x_epoch000_step{s:06d}_20260101_000000"
                for i in range(N_SLICES):
                    ref = _slice(i + reference_seed)
                    pred = np.clip(ref + rng.normal(0, noise * r / 2, ref.shape), 0, 1)
                    Image.fromarray((ref * 255).astype(np.uint8)).save(
                        real / f"{stem}_real_{i:03d}.png"
                    )
                    Image.fromarray((pred * 255).astype(np.uint8)).save(
                        fake / f"{stem}_fake_{i:03d}.png"
                    )
    return run


SUBJECTS = 6
SLICE_ROWS = 4
ROW_HEADER = [
    "iteration",
    "epoch",
    "batch_index",
    "n_samples",
    "subject_id",
    "contrast",
    "slice_index",
    "repetition_group",
    "file_id",
    "val_psnr",
    "val_reference_noise_corrected_mse",
]


def add_slice_rows(run: Path, *, offset: float, seed: int) -> Path:
    """Per-slice validation rows: a shared subject-and-slice signal plus ``offset``.

    The selected pass (iteration 200) carries the signal; iteration 300 carries a
    value ten units away, so a reader that takes any other pass is caught.
    """
    import csv

    rng = np.random.default_rng(seed)
    path = run / "logs" / "validation_batches.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(ROW_HEADER)
        for iteration, shift in ((200, 0.0), (300, 10.0)):
            keys = [(s, z) for s in range(SUBJECTS) for z in range(SLICE_ROWS)]
            for batch_index, j in enumerate(rng.permutation(len(keys))):
                s, z = keys[j]
                value = 25.0 + 2.0 * s + 1.5 * z + offset + shift + rng.normal(0.0, 0.05)
                mse = 0.01 * (1 + s) - 0.001 * offset + rng.normal(0.0, 1e-5)
                writer.writerow(
                    [iteration, 0, batch_index, 1, f"s{s}", "T1", z, "rep0", "f", value, mse]
                )
    return path


def add_training_metrics(run: Path, *, level: float, repeats_first_rows: bool = False) -> Path:
    """``train_psnr`` and ``train_nmse`` per iteration 1..300, rewriting the training log.

    ``level`` shifts ``train_psnr`` so arms differ. With ``repeats_first_rows`` the
    file also holds an earlier appended run: iterations 1..50 again with a far value,
    as a resumed run that restarted its step counter would leave.
    """
    steps = list(range(1, 301))
    frame = pd.DataFrame(
        {
            "iteration": steps,
            "loss": [0.3] * len(steps),
            "train_psnr": [level + 0.01 * s for s in steps],
            "train_nmse": [0.1 - 0.0001 * s for s in steps],
        }
    )
    if repeats_first_rows:
        old = frame.iloc[:50].copy()
        old["train_psnr"] = -1000.0
        frame = pd.concat([old, frame], ignore_index=True)
    path = run / "logs" / "training_metrics.csv"
    frame.to_csv(path, index=False)
    return path


def add_inference_results(run: Path, *, offset: float, cases: int = 8, skipped: bool = True) -> Path:
    """``final_eval.json`` per case for ``psnr`` and ``nmse``, and the skip record beside it.

    Case ``k`` of every arm scores ``30 + k + offset``, so arms pair on case id.
    """
    import json

    payload = {
        "psnr": {f"case{k}": 30.0 + k + offset for k in range(cases)},
        "nmse": {f"case{k}": 0.05 + 0.001 * k - 0.001 * offset for k in range(cases)},
    }
    (run / "final_eval.json").write_text(json.dumps(payload))
    if skipped:
        pd.DataFrame(
            [
                {
                    "metric": "ssim",
                    "subject_id": "",
                    "value": "",
                    "status": "skipped",
                    "reason": "no_reference_available_on_this_path",
                }
            ]
        ).to_csv(run / "inference_metrics.csv", index=False)
    return run / "final_eval.json"


def add_validation_columns(run: Path, names: list[str], *, offset: float = 0.0) -> None:
    """Extra ``val_<name>`` columns in the validation log, one value per pass."""
    path = run / "logs" / "validation_metrics.csv"
    frame = pd.read_csv(path)
    for i, name in enumerate(names):
        frame[f"val_{name}"] = [0.1 * (i + 1) + offset + 0.001 * s for s in range(len(frame))]
    frame.to_csv(path, index=False)
