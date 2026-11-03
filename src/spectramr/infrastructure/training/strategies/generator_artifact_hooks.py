"""Strategy side of a generator's auxiliary losses and its corrupted-input robustness pass.

A generator opts in by defining the members read here; every other generator
gets empty results. ``artifact_robust_generator`` is the one that does today:
it stamps the detector BCE as an auxiliary loss, and names the artifacts the
validation step re-scores the batch under (``val_<artifact>_<metric>`` beside
the clean ``val_<metric>``, plus ``val_<artifact>_psnr_drop``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from spectramr.core.module_utils import unwrap_model


def auxiliary_loss_keys(generator: nn.Module | None) -> frozenset[str]:
    """The loss keys the generator stamps while training, for the CSV header."""
    keys = getattr(unwrap_model(generator), "auxiliary_loss_keys", None)
    return frozenset(keys) if keys else frozenset()


def fold_auxiliary_losses(
    generator: nn.Module | None, total: torch.Tensor, losses: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Add the generator's weighted auxiliary terms to ``total`` and record each."""
    collect = getattr(unwrap_model(generator), "auxiliary_losses", None)
    if not callable(collect):
        return total
    for key, term in collect().items():
        losses[key] = term.detach()
        total = total + term
    return total


def corrupted_input_metrics(
    generator: nn.Module | None,
    clean: Mapping[str, Any],
    run_pass: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Re-run the validation pass once per declared artifact, with that artifact forced.

    ``run_pass`` is the strategy's own metric pass, so the corrupted scores are
    graded exactly like the clean ones. The zero-filled baseline columns grade
    the clean input and are dropped from the corrupted passes.
    """
    inner = unwrap_model(generator)
    artifacts = getattr(inner, "robustness_eval_artifacts", None)
    if not callable(artifacts):
        return {}
    out: dict[str, Any] = {}
    for name in artifacts():
        with inner.forced_artifact(name):
            scored = run_pass()
        for key, value in scored.items():
            if key.startswith("val_zf_") or not key.startswith("val_"):
                continue
            out[f"val_{name}_{key[4:]}"] = value
        if "val_psnr" in clean and "val_psnr" in scored:
            out[f"val_{name}_psnr_drop"] = clean["val_psnr"] - scored["val_psnr"]
    return out


__all__ = ["auxiliary_loss_keys", "corrupted_input_metrics", "fold_auxiliary_losses"]
