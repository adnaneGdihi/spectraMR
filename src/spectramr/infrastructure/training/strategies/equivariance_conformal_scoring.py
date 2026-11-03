r"""Scoring and exchangeability test for the equivariance-defect conformal strategy.

:func:`run_equivariance_calibration` scores a calibration loader with the
equivariance-defect non-conformity score
(:mod:`spectramr.core.metrics.equivariance_defect`), fits the split-conformal
quantile :math:`q_\alpha` on it, and checks exchangeability in one of two declared
modes (``exchangeability_split``):

- ``within_sample``: KS test between the two halves of the calibration scores. It is
  a self-consistency proxy and cannot detect calibration-vs-deployment shift.
- ``held_out``: a second, disjoint loader is scored separately; the KS test compares
  calibration against held-out scores, and coverage is measured on the held-out set
  against the calibration quantile.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from spectramr.core.metrics.equivariance_defect import (
    conformal_quantile,
    equivariance_defect_score,
)

#: The advertised ``exchangeability_split`` values (the schema Literal mirrors it).
EXCHANGEABILITY_SPLITS: tuple[str, ...] = ("within_sample", "held_out")


def _extract_measurement(batch: Any) -> torch.Tensor:
    """Pull the measurement tensor ``y`` from a batch; raises when none matches."""
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, (tuple, list)) and batch:
        return batch[0]
    if isinstance(batch, dict):
        for key in ("input", "kspace", "image", "y"):
            val = batch.get(key)
            if torch.is_tensor(val):
                return val
    raise ValueError(
        "Could not extract a measurement tensor from the batch. Expected a "
        "tensor, an (input, target) sequence, or a dict with an 'input'/"
        "'kspace'/'image' key."
    )


def score_loader(
    recon_fn: Callable[..., torch.Tensor],
    dataloader: Any,
    *,
    num_rotations: int,
    group: str,
    seed: int,
    max_shift_px: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Equivariance-defect score of every sample in ``dataloader`` (float64, CPU).

    Sample ``i`` draws its rotations from ``seed + i``: a different but reproducible
    set per sample, from the same distribution, which keeps the scores exchangeable.
    """
    chunks: list[torch.Tensor] = []
    running = 0
    with torch.no_grad():
        for batch in dataloader:
            y = _extract_measurement(batch).to(device)
            if y.ndim == 3:  # [C, H, W] -> [1, C, H, W]
                y = y.unsqueeze(0)
            per_sample = equivariance_defect_score(
                recon_fn,
                y,
                num_rotations=num_rotations,
                group=group,
                seed=seed + running,
                max_shift_px=max_shift_px,
            )
            chunks.append(per_sample.detach().flatten())
            running += y.shape[0]
    if not chunks:
        return torch.empty(0, dtype=torch.float64)
    # One device-to-host copy for the whole loader.
    return torch.cat(chunks).to(device="cpu", dtype=torch.float64)


def two_sample_ks(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    r"""Two-sample Kolmogorov-Smirnov statistic and asymptotic p-value (torch only).

    The p-value uses :math:`Q_{KS}(\lambda)=2\sum_{j\ge 1}(-1)^{j-1}e^{-2j^2\lambda^2}`.
    """
    a, _ = torch.sort(a.flatten())
    b, _ = torch.sort(b.flatten())
    n, m = a.numel(), b.numel()
    grid = torch.cat([a, b]).unique()
    cdf_a = torch.searchsorted(a, grid, right=True).double() / n
    cdf_b = torch.searchsorted(b, grid, right=True).double() / m
    d = float((cdf_a - cdf_b).abs().max().item())
    en = (n * m / (n + m)) ** 0.5
    lam = (en + 0.12 + 0.11 / en) * d
    # The series does not converge as lam -> 0, where the exact survival function is 1.
    if lam < 1e-3:
        return d, 1.0
    j = torch.arange(1, 101, dtype=torch.float64)
    terms = torch.where(j % 2 == 1, 1.0, -1.0) * torch.exp(-2.0 * j * j * lam * lam)
    p_value = float(max(0.0, min(1.0, 2.0 * terms.sum().item())))
    return d, p_value


def _ks_block(a: torch.Tensor, b: torch.Tensor, threshold: float) -> dict[str, Any]:
    ks_stat, p_value = two_sample_ks(a, b)
    return {
        "test": "ks_test",
        "ks_statistic": ks_stat,
        "p_value": p_value,
        "threshold": threshold,
        "flagged": p_value < threshold,
    }


def run_equivariance_calibration(
    recon_fn: Callable[..., torch.Tensor],
    dataloader: Any,
    *,
    alpha: float = 0.1,
    num_rotations: int = 8,
    group: str = "SO2",
    seed: int = 0,
    max_shift_px: int = 4,
    device: torch.device | str = "cpu",
    exchangeability_test: str = "ks_test",
    exchangeability_ks_pvalue_threshold: float = 0.01,
    exchangeability_split: str = "within_sample",
    test_dataloader: Any = None,
) -> dict[str, Any]:
    r"""Score the calibration loader, fit :math:`q_\alpha`, and test exchangeability.

    Raises:
        ValueError: unknown ``exchangeability_split`` or ``exchangeability_test``; an
            empty loader; ``held_out`` without a ``test_dataloader``.
    """
    if exchangeability_split not in EXCHANGEABILITY_SPLITS:
        raise ValueError(
            f"exchangeability_split={exchangeability_split!r} is not one of "
            f"{list(EXCHANGEABILITY_SPLITS)}"
        )
    if exchangeability_test not in ("ks_test", "none"):
        raise ValueError(f"exchangeability_test={exchangeability_test!r} is not 'ks_test'/'none'")
    if exchangeability_split == "held_out" and test_dataloader is None:
        raise ValueError(
            "exchangeability_split='held_out' needs a held-out test loader "
            "(declare data.source.test_index_path, disjoint from the calibration manifest)."
        )
    common: dict[str, Any] = {
        "num_rotations": num_rotations,
        "group": group,
        "max_shift_px": max_shift_px,
        "device": device,
    }
    cal = score_loader(recon_fn, dataloader, seed=seed, **common)
    if cal.numel() == 0:
        raise ValueError("Calibration loader yielded no samples.")
    q = conformal_quantile(cal, alpha)
    report: dict[str, Any] = {
        "quantile": q,
        "n_calibration": int(cal.numel()),
        "alpha": alpha,
        "num_rotations_K": num_rotations,
        "group": group,
        "mean_score": float(cal.mean().item()),
        "score_fn": "equivariance_defect",
        "exchangeability_split": exchangeability_split,
    }

    if exchangeability_split == "held_out":
        # Seeds continue past the calibration indices so no rotation set is reused.
        test = score_loader(recon_fn, test_dataloader, seed=seed + cal.numel(), **common)
        if test.numel() == 0:
            raise ValueError("Held-out test loader yielded no samples.")
        coverage = float((test <= q).double().mean().item())
        report["n_test"] = int(test.numel())
        report["mean_test_score"] = float(test.mean().item())
        report["coverage_at_alpha"] = coverage
        report["empirical_coverage"] = coverage
        if exchangeability_test == "ks_test":
            report["exchangeability"] = _ks_block(cal, test, exchangeability_ks_pvalue_threshold)
        return report

    # within_sample: coverage from a seeded random half-split of the calibration scores.
    n = int(cal.numel())
    if n >= 2:
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed)))
        shuffled = cal[perm]
        half = n // 2
        q_split = conformal_quantile(shuffled[:half], alpha)
        coverage = float((shuffled[half:] <= q_split).double().mean().item())
        report["coverage_at_alpha"] = coverage
        report["empirical_coverage"] = coverage
    else:
        report["coverage_at_alpha"] = None
        report["empirical_coverage"] = None
    if exchangeability_test == "ks_test" and n >= 4:
        half = n // 2
        report["exchangeability"] = _ks_block(
            cal[:half], cal[half:], exchangeability_ks_pvalue_threshold
        )
    return report


__all__ = [
    "EXCHANGEABILITY_SPLITS",
    "run_equivariance_calibration",
    "score_loader",
    "two_sample_ks",
]
