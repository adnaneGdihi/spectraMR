"""Behavioural tests for ``spectramr.models.losses.fmri_mrf_losses``."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.fmri_mrf_losses import HRFLikelihoodLoss


def _causal_convolution(design: torch.Tensor, hrf: torch.Tensor) -> torch.Tensor:
    """Reference ``sim[b, t] = sum_j design[b, t - j] * hrf[b, j]`` by explicit loops."""
    n_batch, n_time = design.shape
    out = torch.zeros(n_batch, n_time, dtype=design.dtype)
    for b in range(n_batch):
        for t in range(n_time):
            for j in range(hrf.shape[-1]):
                if t - j >= 0:
                    out[b, t] += design[b, t - j] * hrf[b, j]
    return out


def test_hrf_likelihood_is_zero_at_the_exact_glm_prediction_for_a_batch() -> None:
    torch.manual_seed(0)
    design = torch.rand(3, 20, dtype=torch.float64)
    hrf = torch.rand(3, 5, dtype=torch.float64)
    target = _causal_convolution(design, hrf)
    out = HRFLikelihoodLoss()(hrf, target, design_matrix=design)
    assert float(out["hrf_likelihood"]) == pytest.approx(0.0, abs=1e-20)


def test_hrf_likelihood_does_not_mix_samples() -> None:
    torch.manual_seed(1)
    design = torch.rand(2, 16, dtype=torch.float64)
    hrf = torch.rand(2, 4, dtype=torch.float64)
    target = _causal_convolution(design, hrf)
    batched = HRFLikelihoodLoss()(hrf, target, design_matrix=design)["hrf_likelihood"]
    per_sample = [
        HRFLikelihoodLoss()(hrf[b : b + 1], target[b : b + 1], design_matrix=design[b : b + 1])[
            "hrf_likelihood"
        ]
        for b in range(2)
    ]
    torch.testing.assert_close(batched, torch.stack(per_sample).mean())


def test_hrf_likelihood_broadcasts_a_single_shared_hrf() -> None:
    torch.manual_seed(2)
    design = torch.rand(3, 12, dtype=torch.float64)
    hrf = torch.rand(1, 4, dtype=torch.float64)
    target = _causal_convolution(design, hrf.expand(3, -1))
    out = HRFLikelihoodLoss()(hrf, target, design_matrix=design)
    assert float(out["hrf_likelihood"]) == pytest.approx(0.0, abs=1e-20)


def test_hrf_likelihood_rejects_a_batch_mismatch() -> None:
    design = torch.rand(3, 12)
    with pytest.raises(ValueError, match="does not match design_matrix batch"):
        HRFLikelihoodLoss()(torch.rand(2, 4), torch.rand(3, 12), design_matrix=design)
