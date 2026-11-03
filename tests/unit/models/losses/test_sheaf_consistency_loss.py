"""The sheaf loss evaluates the identity-restriction quadratic form in O(n) memory.

The reference below is the dense form the loss used to build: an ``(E, n, n)``
identity restriction per edge, which is 17 GB of ``torch.eye`` at 256x256. The
stalk-1 decomposition must reproduce it, value and gradient.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.cellular_sheaf import (
    mayer_vietoris_obstruction_norm,
    sheaf_laplacian_quadratic_form,
)
from spectramr.models.losses.sheaf_consistency_loss import SheafConsistencyLoss


def _dense_reference(
    prediction: torch.Tensor,
    target: torch.Tensor,
    laplacian_weight: float,
    obstruction_weight: float,
    reduction: str,
) -> torch.Tensor:
    batch, cells = prediction.shape[0], prediction.shape[1]
    pred_flat = prediction.reshape(batch, cells, -1)
    tgt_flat = target.reshape(batch, cells, -1)
    stalk = pred_flat.shape[-1]
    u = torch.arange(max(cells - 1, 0))
    edges = torch.stack([u, u + 1], dim=1) if cells > 1 else torch.empty(0, 2, dtype=torch.long)
    eye = torch.eye(stalk, dtype=prediction.dtype).unsqueeze(0).expand(edges.shape[0], stalk, stalk)
    per_sample = []
    for b in range(batch):
        value = prediction.new_zeros(())
        if laplacian_weight > 0 and edges.shape[0] > 0:
            lap = sheaf_laplacian_quadratic_form(pred_flat[b], edges, eye, eye)
            value = value + laplacian_weight * lap / (edges.shape[0] * stalk)
        if obstruction_weight > 0:
            p, t = pred_flat[b].reshape(-1), tgt_flat[b].reshape(-1)
            value = (
                value + obstruction_weight * mayer_vietoris_obstruction_norm(p, t, p, t) / p.numel()
            )
        per_sample.append(value)
    stacked = torch.stack(per_sample)
    return stacked.mean() if reduction == "mean" else stacked.sum()


_WEIGHTS_AND_CELLS = [
    (lap_w, obs_w, cells)
    for lap_w, obs_w in [(1.0, 1.0), (0.7, 0.0), (0.0, 2.0)]
    for cells in [1, 2, 4]
    if cells > 1 or lap_w == 0  # one cell with a Laplacian term raises (test below)
]


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize(("lap_w", "obs_w", "cells"), _WEIGHTS_AND_CELLS)
def test_matches_the_dense_identity_restriction_form(
    reduction: str, lap_w: float, obs_w: float, cells: int
) -> None:
    torch.manual_seed(cells)
    pred = torch.randn(3, cells, 5, 6, dtype=torch.float64, requires_grad=True)
    target = torch.randn(3, cells, 5, 6, dtype=torch.float64)
    loss_fn = SheafConsistencyLoss(
        laplacian_weight=lap_w, obstruction_weight=obs_w, reduction=reduction
    )
    ours = loss_fn(pred, target)
    reference = _dense_reference(pred, target, lap_w, obs_w, reduction)
    torch.testing.assert_close(ours, reference)
    if ours.requires_grad:
        (g_ours,) = torch.autograd.grad(ours, pred)
        (g_ref,) = torch.autograd.grad(reference, pred)
        torch.testing.assert_close(g_ours, g_ref)


def test_a_realistic_image_size_needs_no_dense_restriction_maps() -> None:
    """128x128 needed a 1 GB identity; the stalk-1 form allocates O(B*C*H*W)."""
    pred = torch.rand(2, 2, 128, 128, requires_grad=True)
    loss = SheafConsistencyLoss()(pred, torch.rand(2, 2, 128, 128))
    loss.backward()
    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_a_single_cell_with_a_laplacian_term_raises() -> None:
    """One channel makes the Laplacian identically zero; that must not train silently."""
    x = torch.rand(2, 1, 8, 8)
    with pytest.raises(ValueError, match="cover cells"):
        SheafConsistencyLoss(laplacian_weight=1.0)(x, x)
    assert SheafConsistencyLoss(laplacian_weight=0.0)(x, x) == 0


def test_the_multi_contrast_arm_trains_a_two_cell_sheaf_term() -> None:
    """Reachability: the arm's YAML -> LossBuilder -> the reconstruction loss computer the
    strategy calls each step; the sheaf term fires on the two contrast channels."""
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.training.builders.loss_builder import LossBuilder
    from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
        UnifiedReconstructionLossComputer,
    )

    arm = "experiments/inprogress/novel_2026/breakthrough_sheaf_multi_contrast.yaml"
    cfg = TrainingSettings.from_yaml(arm)
    cpu = torch.device("cpu")
    losses = LossBuilder(cfg, cpu).build_reconstruction_losses().build()
    computer = UnifiedReconstructionLossComputer(config=cfg, device=cpu)
    cells = cfg.model.out_channels
    assert cells == cfg.data.domain.target_channels == 2

    def sheaf_term(x: torch.Tensor) -> torch.Tensor:
        out = computer.compute(pred=x, target=x, epoch=0, iteration=0, losses_dict=losses)
        return out.components["sheaf_consistency"]

    # Equal prediction and target: l1 and the obstruction are zero, so this is the Laplacian.
    x = torch.rand(2, cells, 16, 16)
    assert sheaf_term(x) > 0
    assert sheaf_term(x[:, :1].expand(-1, cells, -1, -1).contiguous()) == 0


@pytest.mark.parametrize(
    "arm",
    [
        "experiments/inprogress/novel_2026/breakthrough_sheaf_multi_contrast.yaml",
        "experiments/inprogress/novel_2026/control_sheaf_multi_contrast_hermitian_fno.yaml",
    ],
)
def test_the_multi_contrast_arms_score_each_contrast_unmixed(arm: str) -> None:
    """The two channels are contrasts, not Re/Im: the metric view must not RSS them together."""
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin

    cfg = TrainingSettings.from_yaml(arm)
    mixin = MetricsMixin.__new__(MetricsMixin)
    mixin.config = cfg
    pred, target = torch.rand(2, 2, 8, 8), torch.rand(2, 2, 8, 8)
    views = mixin._metric_transform_views(pred, target, cfg.validation)
    assert torch.equal(views[0], pred)
    assert torch.equal(views[1], target)
