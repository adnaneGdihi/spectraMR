"""Tests for ConfluenceStrategy (B-1.1 cross-source Fréchet-mean consensus)."""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.infrastructure.training.strategies.confluence_strategy import (
    ConfluenceStrategy,
    compute_confluence_losses,
)
from spectramr.models.generators.cross_field_renderer import AnatomyFieldRenderer


def _net() -> AnatomyFieldRenderer:
    return AnatomyFieldRenderer(in_channels=1, out_channels=1, latent_channels=16)


def _batch(n: int = 4) -> dict:
    return {
        "sources": torch.rand(2, n, 1, 16, 16),
        "source_field_strengths": torch.tensor([0.1, 1.5, 3.0, 5.0][:n]).repeat(2, 1),
        "field_strength": torch.tensor([0.1, 0.1]),
        "input": torch.rand(2, 1, 16, 16),
        "target": torch.rand(2, 1, 16, 16),
        "field_strength_target": torch.tensor([7.0, 7.0]),
        "contrast_id": torch.tensor([0, 1]),
    }


def test_loss_keys_and_finite() -> None:
    out = compute_confluence_losses(_net(), _batch(), lambda_consensus=1.0)
    assert {"loss_total", "loss_fidelity", "loss_consensus"} <= set(out)
    assert torch.isfinite(out["loss_total"])


def test_builder_image_losses_folded_onto_consensus_via_seam() -> None:
    """Declarative image losses (hfen/ms_ssim) fold onto the consensus deliverable via the
    loss-SSOT seam; the inline per-source l1 fidelity is skipped (no double-count)."""
    from spectramr.models.losses.charbonnier_loss import CharbonnierLoss
    from spectramr.models.losses.hfen_loss import HFENLoss

    strat = object.__new__(ConfluenceStrategy)
    strat.env = types.SimpleNamespace(
        generator=_net(), losses={"l1": CharbonnierLoss(), "hfen": HFENLoss()})
    strat.config = types.SimpleNamespace(losses=types.SimpleNamespace(
        image_losses=[{"name": "l1", "weight": 1.0}, {"name": "hfen", "weight": 0.2}],
        kspace_losses=[], complex_losses=[]))

    out = strat._compute_losses_impl(input_batch=_batch(), target_batch=None, epoch=0)
    assert "loss_builder_aux" in out and "loss_hfen" in out
    assert torch.isfinite(out["loss_total"])
    assert "prediction" not in out and "target_image" not in out


def test_consensus_is_positive_when_predictions_differ() -> None:
    out = compute_confluence_losses(_net(), _batch(), lambda_consensus=1.0)
    assert float(out["loss_consensus"].detach()) > 0.0


def test_lambda_zero_drops_consensus_cleanly() -> None:
    # The one-knob control: lambda=0 -> total == per-source fidelity (still a valid
    # multi-source translator, NOT degenerate). Isolates the consensus contribution (#17).
    out = compute_confluence_losses(_net(), _batch(), lambda_consensus=0.0)
    assert torch.allclose(out["loss_total"], out["loss_fidelity"])


def test_all_tuple_sources_are_consumed() -> None:
    # ANTI-FACADE (#16): perturbing ANY source in the tuple must change the loss — proves the
    # full multi-source tuple is used, not just the first source.
    net = _net()
    batch = _batch()
    base = float(compute_confluence_losses(net, batch, lambda_consensus=1.0)["loss_total"].detach())
    for i in range(batch["sources"].shape[1]):
        b = dict(batch)
        b["sources"] = batch["sources"].clone()
        b["sources"][:, i] += 0.5
        pert = float(compute_confluence_losses(net, b, lambda_consensus=1.0)["loss_total"].detach())
        assert abs(pert - base) > 1e-5, f"source {i} not consumed"


def test_loss_reduces() -> None:
    torch.manual_seed(0)
    net = _net()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    batch = _batch()
    first = None
    out = None
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        out = compute_confluence_losses(net, batch, lambda_consensus=1.0)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_validation_renders_consensus_mean() -> None:
    # Validation grades the variance-reduced CONSENSUS MEAN over the N source predictions
    # (the actual deliverable), so the metric reflects the claimed variance reduction (#18).
    net = _net().eval()
    strat = object.__new__(ConfluenceStrategy)
    strat.env = types.SimpleNamespace(generator=net)
    strat._cf_contrast = True
    batch = _batch()
    with torch.no_grad():
        pred = strat._validation_forward(
            batch["input"],
            {
                "sources": batch["sources"],
                "source_field_strengths": batch["source_field_strengths"],
                "field_strength_target": batch["field_strength_target"],
                "contrast_id": batch["contrast_id"],
            },
        )
        # the consensus mean equals the average of the per-source renders
        gen = net
        b_s = batch["source_field_strengths"]
        preds = [gen.render(gen.encode(batch["sources"][:, i], b_s[:, i]), torch.tensor([7.0, 7.0]), batch["contrast_id"]) for i in range(4)]
        expect = torch.stack(preds, 0).mean(0).clamp(0.0, 1.0)
    assert pred.shape == (2, 1, 16, 16)
    assert torch.allclose(pred, expect, atol=1e-5)
    assert float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0


def test_validation_exposes_collapse_monitor_and_caches_visual() -> None:
    # #4: a consensus collapse (encoder driven source-invariant, rho->1) zeroes the consensus
    # loss WITHOUT genuine variance reduction, indistinguishable from a real run unless the
    # inter-source dispersion is monitored. _validation_forward must expose it.
    # #9: it must also cache the graded consensus so the image-capture path saves it.
    net = _net().eval()
    strat = object.__new__(ConfluenceStrategy)
    strat.env = types.SimpleNamespace(generator=net)
    strat._cf_contrast = True
    batch = _batch()
    with torch.no_grad():
        strat._validation_forward(
            batch["input"],
            {
                "sources": batch["sources"],
                "source_field_strengths": batch["source_field_strengths"],
                "field_strength_target": batch["field_strength_target"],
                "contrast_id": batch["contrast_id"],
            },
        )
    assert hasattr(strat, "_last_intersource_std")
    assert strat._last_intersource_std > 0.0  # distinct sources -> non-collapsed dispersion
    assert hasattr(strat, "_last_visual_pred")
    assert strat._last_visual_pred.shape == (2, 1, 16, 16)


def test_validation_falls_back_to_single_input_without_tuple() -> None:
    net = _net().eval()
    strat = object.__new__(ConfluenceStrategy)
    strat.env = types.SimpleNamespace(generator=net)
    strat._cf_contrast = True
    pred = strat._validation_forward(
        torch.rand(2, 1, 16, 16),
        {
            "field_strength": torch.tensor([0.1, 3.0]),
            "field_strength_target": torch.tensor([7.0, 7.0]),
            "contrast_id": torch.tensor([0, 1]),
        },
    )
    assert pred.shape == (2, 1, 16, 16)


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    from spectramr.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(ConfluenceStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._cf_lambda_consensus = 1.0
    strat._cf_contrast = True
    out = strat._compute_losses_impl(input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb)
    assert torch.isfinite(out["loss_total"])


def test_compute_losses_rejects_tensor_batch() -> None:
    import pytest

    strat = object.__new__(ConfluenceStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._cf_lambda_consensus = 1.0
    strat._cf_contrast = True
    with pytest.raises(ValueError, match="mapping batch"):
        strat._compute_losses_impl(
            input_batch=torch.rand(2, 1, 16, 16), target_batch=torch.rand(2, 1, 16, 16),
            epoch=0, batch=torch.rand(2, 1, 16, 16),
        )


def test_compute_losses_requires_sources() -> None:
    import pytest

    strat = object.__new__(ConfluenceStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._cf_lambda_consensus = 1.0
    strat._cf_contrast = True
    with pytest.raises(ValueError, match="multi-source tuple 'sources'"):
        strat._compute_losses_impl(
            input_batch=None, target_batch=None, epoch=0,
            batch={"input": torch.rand(2, 1, 16, 16), "target": torch.rand(2, 1, 16, 16)},
        )


def test_strategy_registered_and_config_mounted() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "confluence" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "confluence" in TrainingStrategyConfigSchema.model_fields


def test_each_source_is_encoded_at_its_own_field() -> None:
    # #197: the target is pinned at 7 T, so the per-source fields are the only field signal.
    seen: list[torch.Tensor] = []

    class _Rec:
        def encode(self, x, b_s):
            seen.append(b_s)
            return x

        def render(self, q, b_t, cid):
            return q

    compute_confluence_losses(_Rec(), _batch(), lambda_consensus=1.0)
    assert [float(b[0]) for b in seen] == pytest.approx([0.1, 1.5, 3.0, 5.0])


def test_missing_source_fields_raise() -> None:
    batch = _batch()
    del batch["source_field_strengths"]
    with pytest.raises(ValueError, match="source_field_strengths"):
        compute_confluence_losses(_net(), batch, lambda_consensus=1.0)
