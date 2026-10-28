"""Synthetic n2n run directories for the paper-asset script tests.

Four arms in the downloaded layout (``resolved_config.json``, ``run_summary.json``,
``final_metrics.json``, ``logs/*_metrics.csv``, ``logs/metrics/{fake,real}_images``),
small enough to build in milliseconds: 32x32 tiles, two or three validation passes.
Arm A validates once more than the others so the common budget differs from its
final iteration, and arm B peaks at its first validation so the early-stopping
selection differs from its final iteration.

``protocol="v1"`` is the shape of the first download: every arm early-stops on
``val_psnr`` and arms A and B log no corrected MSE. ``protocol="v2"`` records
``val_reference_noise_corrected_mse`` (mode ``min``) as the selector and every arm
logs it. Arm G's PSNR peaks at its first validation but its corrected MSE
(``0.02 / N_eq``) at its second, so the two protocols select different rows.
``record_selector=False`` records no selector at all.

:func:`add_slice_rows` adds the per-slice rows a paired table needs, each arm's
slices written in a different shuffled order.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

RICH = (
    "val_nex_equivalent",
    "val_reference_noise_corrected_mse",
    "val_bias_squared",
    "val_rep_variance",
    "val_null_space_residual",
    "val_null_space_residual_fraction",
)

#: name -> (target_mode, split angle, [(iteration, psnr, ssim, nex_equivalent | None)])
ARMS: dict[str, tuple[str, float, list[tuple[int, float, float, float | None]]]] = {
    "n2n_a_r2r_single_ex_m4raw": (
        "r2r",
        45.0,
        [(1000, 22.0, 0.40, None), (2000, 22.5, 0.41, None), (3000, 22.4, 0.42, None)],
    ),
    "n2n_b_rep_pair_m4raw": (
        "rep_pair",
        45.0,
        [(1000, 22.7, 0.44, None), (2000, 22.6, 0.40, None)],
    ),
    "n2n_g_nullspace_r2r_m4raw": (
        "nullspace_r2r",
        45.0,
        [(1000, 22.2, 0.41, 5.0), (2000, 22.1, 0.39, 6.0)],
    ),
    "n2n_g_theta20_nullspace_r2r_m4raw": (
        "nullspace_r2r",
        20.0,
        [(1000, 22.3, 0.42, 5.4), (2000, 22.2, 0.40, 6.5)],
    ),
}
IDENTITY_PSNR = 23.8
#: Under v2 the arms without the Rep-BV columns still log the selector, one per pass.
CORRECTED_MSE: dict[str, tuple[float, ...]] = {
    "n2n_a_r2r_single_ex_m4raw": (0.0040, 0.0030, 0.0035),
    "n2n_b_rep_pair_m4raw": (0.0030, 0.0035),
}
SIZE = 32
SLICES = 2


def reference_tile(index: int) -> np.ndarray:
    """A bright 8x8 block on a dim floor; the floor level marks the reference's noise."""
    tile = np.full((SIZE, SIZE), 20 + index, dtype=np.uint8)
    tile[12:20, 12:20] = 200
    return tile


def prediction_tile(index: int, iteration: int) -> np.ndarray:
    """The block a little brighter, the floor pushed to zero, and a step-dependent pixel."""
    tile = np.full((SIZE, SIZE), 2, dtype=np.uint8)
    tile[12:20, 12:20] = 210
    tile[0, 0] = min(255, iteration // 100 + index)
    return tile


def _png(path: Path, tile: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tile, mode="L").save(path)


PROTOCOL_SELECTOR = {
    "v1": {"metric": "val_psnr", "mode": "max"},
    "v2": {"metric": "val_reference_noise_corrected_mse", "mode": "min"},
}


def build_runs(root: Path, protocol: str = "v1", record_selector: bool = True) -> Path:
    v2 = protocol == "v2"
    for name, (target_mode, angle, rows) in ARMS.items():
        run = root / name
        (run / "logs").mkdir(parents=True, exist_ok=True)
        rich = rows[0][3] is not None
        compute = ["psnr", "ssim"] + ([m.removeprefix("val_") for m in RICH] if rich else [])
        resolved: dict = {
            "data": {"target_mode": target_mode, "r2r_split_angle_deg": angle},
            "validation": {"scoring": {"compute": compute}},
        }
        if record_selector:
            resolved["early_stopping"] = {"enabled": True, **PROTOCOL_SELECTOR[protocol]}
        (run / "resolved_config.json").write_text(json.dumps(resolved))
        (run / "run_summary.json").write_text(
            json.dumps(
                {
                    "git": {
                        "sha_short": "abc123def456",
                        "dirty": name.endswith("theta20_nullspace_r2r_m4raw"),
                    },
                    "host": "compute-1-1",
                    "hardware": {"gpu_types": {"Tesla V100": 1}},
                    "duration": "01:00:00",
                    "iterations_per_sec": 0.5,
                    "slurm": {"SLURM_JOB_ID": "1"},
                    "model_params": 12,
                    "started_at": "2026-01-01T00:00:00",
                }
            )
        )
        (run / "final_metrics.json").write_text(
            json.dumps({"best": {"val_psnr_best": max(r[1] for r in rows)}})
        )
        header = ["iteration", "epoch", "val_loss", "val_psnr", "val_ssim", "val_zf_psnr"]
        header += list(RICH) if rich else (["val_reference_noise_corrected_mse"] if v2 else [])
        with (run / "logs" / "validation_metrics.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for i, (iteration, psnr, ssim, nex) in enumerate(rows):
                cells = [iteration, iteration // 1000, 0.025, psnr, ssim, IDENTITY_PSNR]
                if rich:
                    tau = math.tan(math.radians(angle))
                    cells += [nex, 0.02 / nex, 0.001 * tau, 0.001 / tau, 0.0003, 0.0016]
                elif v2:
                    cells.append(CORRECTED_MSE[name][i])
                writer.writerow(cells)
        with (run / "logs" / "training_metrics.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["iteration", "loss"])
            for iteration in range(100, rows[-1][0] + 700, 100):
                writer.writerow([iteration, 1.0 / iteration])
        for iteration, *_ in rows:
            stamp = f"validation_epoch{iteration // 1000:03d}_step{iteration:06d}_20260101_{iteration // 10:06d}"
            for index in range(SLICES):
                _png(
                    run / "logs/metrics/fake_images" / f"{stamp}_fake_{index:03d}.png",
                    prediction_tile(index, iteration),
                )
                _png(
                    run / "logs/metrics/real_images" / f"{stamp}_real_{index:03d}.png",
                    reference_tile(index),
                )
    return root


#: Per-arm N_eq offset of the per-slice rows; G - A is the known paired effect.
SLICE_NEQ = {"A": 2.0, "B": 2.0, "G": 3.0, "G20": 3.5}
SUBJECTS = 4
SLICES_PER_SUBJECT = 3
SLICE_HEADER = [
    "iteration",
    "epoch",
    "batch_index",
    "n_samples",
    "subject_id",
    "contrast",
    "slice_index",
    "repetition_group",
    "file_id",
    "val_nex_equivalent",
    "val_reference_noise_corrected_mse",
]


def add_slice_rows(root: Path, seed: int = 0) -> None:
    """``logs/validation_batches.csv`` for every arm and pass, one row per slice.

    Each arm writes its slices in its own random order, so ``batch_index`` names
    a different slice in each arm, as it does under shuffling loaders.
    """
    rng = np.random.default_rng(seed)
    keys = {"n2n_a": "A", "n2n_b": "B", "n2n_g_theta20": "G20", "n2n_g_nullspace": "G"}
    for name, (_, _, rows) in ARMS.items():
        key = next(v for k, v in keys.items() if name.startswith(k))
        slices = [(s, z) for s in range(SUBJECTS) for z in range(SLICES_PER_SUBJECT)]
        noise = {sz: rng.normal(0.0, 0.05) for sz in slices}
        with (root / name / "logs" / "validation_batches.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(SLICE_HEADER)
            for iteration, *_ in rows:
                order = rng.permutation(len(slices))
                for batch_index, j in enumerate(order):
                    subject, z = slices[j]
                    neq = SLICE_NEQ[key] + 0.2 * subject + noise[(subject, z)]
                    sid, cmse = f"subj{subject}", 0.02 / neq
                    epoch = iteration // 1000
                    writer.writerow([iteration, epoch, batch_index, 1, sid, "T1", z, "", sid, neq, cmse])
