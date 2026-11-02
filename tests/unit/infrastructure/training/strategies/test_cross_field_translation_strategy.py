"""Tests for CrossFieldTranslationStrategy loss math + factory registration."""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.infrastructure.training.strategies.cross_field_translation_strategy import (
    CrossFieldTranslationStrategy,
    compute_cross_field_losses,
)
from spectramr.models.generators.cross_field_renderer import AnatomyFieldRenderer


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 16, 16),
        "target": torch.rand(2, 1, 16, 16),
        "field_strength": torch.tensor([0.1, 3.0]),
        "field_strength_target": torch.tensor([7.0, 7.0]),
        "contrast_id": torch.tensor([0, 1]),
    }


class _RecordingRenderer:
    """Renderer stub that records the contrast id it was rendered with, to assert
    the contrast_conditioning knob actually gates the FiLM contrast input."""

    def __init__(self) -> None:
        self.last_cid: object = "unset"

    def encode(self, x, b_s):
        self.last_bs = b_s
        return x

    def render(self, q, b_t, cid):
        self.last_cid = cid
        return q


def test_compute_losses_keys_and_finite() -> None:
    gen = AnatomyFieldRenderer(in_channels=1, out_channels=1, latent_channels=16)
    out = compute_cross_field_losses(gen, _batch(), lambda_latent_cycle=0.1)
    assert {"loss_total", "loss_recon", "loss_latent_cycle"} <= set(out)
    assert torch.isfinite(out["loss_total"])
    assert float(out["loss_latent_cycle"].detach()) >= 0.0


def test_builder_image_losses_folded_via_seam() -> None:
    """Declarative image losses (hfen/ms_ssim) fold onto the rendered prediction via the
    loss-SSOT seam; the inline l1 placeholder is skipped (no double-count). This is the
    shared strategy behind b17/b19 (add sharpness) and b38 (bare l1 -> folds nothing)."""
    from spectramr.models.losses.charbonnier_loss import CharbonnierLoss
    from spectramr.models.losses.hfen_loss import HFENLoss

    strat = object.__new__(CrossFieldTranslationStrategy)
    strat.env = types.SimpleNamespace(
        generator=AnatomyFieldRenderer(in_channels=1, out_channels=1, latent_channels=16),
        losses={"l1": CharbonnierLoss(), "hfen": HFENLoss()})
    strat.config = types.SimpleNamespace(losses=types.SimpleNamespace(
        image_losses=[{"name": "l1", "weight": 1.0}, {"name": "hfen", "weight": 0.2}],
        kspace_losses=[], complex_losses=[]))

    out = strat._compute_losses_impl(input_batch=_batch(), target_batch=None, epoch=0)
    assert "loss_builder_aux" in out and "loss_hfen" in out
    assert torch.isfinite(out["loss_total"])
    assert "prediction" not in out and "target_image" not in out


def test_bare_l1_placeholder_folds_nothing_via_seam() -> None:
    """The b38 case: an arm that keeps only the bare l1 placeholder folds NOTHING (skip-l1),
    so loss_total is byte-identical to the inline-only objective (no double-count, no drift)."""
    from spectramr.models.losses.charbonnier_loss import CharbonnierLoss

    torch.manual_seed(0)
    gen = AnatomyFieldRenderer(in_channels=1, out_channels=1, latent_channels=16)
    batch = _batch()

    strat = object.__new__(CrossFieldTranslationStrategy)
    strat.env = types.SimpleNamespace(generator=gen, losses={"l1": CharbonnierLoss()})
    strat.config = types.SimpleNamespace(losses=types.SimpleNamespace(
        image_losses=[{"name": "l1", "weight": 1.0}], kspace_losses=[], complex_losses=[]))
    seam_out = strat._compute_losses_impl(input_batch=batch, target_batch=None, epoch=0)

    inline = compute_cross_field_losses(gen, batch, lambda_latent_cycle=0.1)
    assert "loss_builder_aux" not in seam_out  # l1 skipped -> nothing folded
    assert torch.allclose(seam_out["loss_total"], inline["loss_total"])


def test_recon_decreases_toward_target() -> None:
    torch.manual_seed(0)
    gen = AnatomyFieldRenderer(in_channels=1, out_channels=1, latent_channels=16)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-2)
    x = torch.rand(2, 1, 16, 16)
    y = torch.rand(2, 1, 16, 16)
    batch = {
        "input": x,
        "target": y,
        "field_strength": torch.tensor([0.1, 3.0]),
        "field_strength_target": torch.tensor([7.0, 7.0]),
        "contrast_id": torch.tensor([0, 0]),
    }
    first = None
    out = None
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        out = compute_cross_field_losses(gen, batch, lambda_latent_cycle=0.0)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_recon"].detach())
    assert out is not None and first is not None
    assert float(out["loss_recon"].detach()) < first


def test_latent_cycle_zero_when_disabled() -> None:
    gen = AnatomyFieldRenderer(in_channels=1, out_channels=1, latent_channels=16)
    out = compute_cross_field_losses(gen, _batch(), lambda_latent_cycle=0.0)
    assert float(out["loss_latent_cycle"]) == 0.0


def test_contrast_conditioning_false_drops_cid_in_training() -> None:
    # The cross_field.contrast_conditioning knob must actually gate the FiLM
    # contrast input: with it False, the real batch contrast_id is NOT passed.
    gen = _RecordingRenderer()
    compute_cross_field_losses(
        gen, _batch(), lambda_latent_cycle=0.0, contrast_conditioning=False
    )
    assert gen.last_cid is None


def test_contrast_conditioning_true_passes_cid_in_training() -> None:
    gen = _RecordingRenderer()
    compute_cross_field_losses(
        gen, _batch(), lambda_latent_cycle=0.0, contrast_conditioning=True
    )
    assert gen.last_cid is not None


def test_validation_contrast_conditioning_gates_cid() -> None:
    # _validation_forward must honor the same knob: False -> cid None at validation
    # (so train and val use the SAME conditioning); True -> the forwarded id flows.
    import types

    from spectramr.infrastructure.training.strategies.cross_field_translation_strategy import (
        CrossFieldTranslationStrategy,
    )

    for flag, expect_none in ((False, True), (True, False)):
        gen = _RecordingRenderer()
        strat = object.__new__(CrossFieldTranslationStrategy)
        strat.env = types.SimpleNamespace(generator=gen)
        strat._contrast_conditioning = flag
        strat._validation_forward(
            torch.rand(2, 1, 16, 16),
            {"use_dc": False},
            field_strength=torch.tensor([0.1, 3.0]),
            field_strength_target=torch.tensor([7.0, 7.0]),
            contrast_id=torch.tensor([0, 1]),
        )
        assert (gen.last_cid is None) is expect_none


def test_strategy_registered_in_factory() -> None:
    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    assert "cross_field_translation" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS


def test_validation_forward_renders_at_target_field() -> None:
    # The inherited reconstruction _validation_forward calls generator(x) WITHOUT
    # field_strength, which AnatomyFieldRenderer rejects -> crash. The override must
    # render at the validation sample's target field, reading it from kwargs (as the
    # pipeline forwards it) rather than a bare batch_context bracket access.
    import types

    from spectramr.infrastructure.training.strategies.cross_field_translation_strategy import (
        CrossFieldTranslationStrategy,
    )

    gen = AnatomyFieldRenderer(in_channels=1, out_channels=1, latent_channels=16).eval()
    strat = object.__new__(CrossFieldTranslationStrategy)
    strat.env = types.SimpleNamespace(generator=gen)
    x0 = torch.rand(2, 1, 16, 16)
    pred = strat._validation_forward(
        x0,
        {"use_dc": False, "measured_kspace": None},
        field_strength=torch.tensor([0.1, 3.0]),
        field_strength_target=torch.tensor([7.0, 7.0]),
        contrast_id=torch.tensor([0, 1]),
    )
    assert pred.shape == (2, 1, 16, 16)


def test_validation_forward_raises_when_target_field_absent() -> None:
    import types

    from spectramr.infrastructure.training.strategies.cross_field_translation_strategy import (
        CrossFieldTranslationStrategy,
    )

    gen = AnatomyFieldRenderer(in_channels=1, out_channels=1, latent_channels=16)
    strat = object.__new__(CrossFieldTranslationStrategy)
    strat.env = types.SimpleNamespace(generator=gen)
    with pytest.raises(ValueError, match="field_strength_target"):
        strat._validation_forward(torch.rand(1, 1, 16, 16), {"use_dc": False})


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    # REGRESSION (cohort guard, 2026-06-16): the canonical training pipeline converts the
    # loader dict to a TrainingBatch (BatchAdapter.from_dict) and forwards it as
    # batch=<TrainingBatch>, which is NOT a dict. The original `isinstance(batch, dict)`
    # guard rejected it and crashed every MICCAI arm at step 0; the dict-only helper tests
    # never exercised this path. The guard must accept any mapping exposing .get.
    import types

    from spectramr.data.batch_types import BatchAdapter
    from spectramr.infrastructure.training.strategies.cross_field_translation_strategy import (
        CrossFieldTranslationStrategy,
    )
    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(CrossFieldTranslationStrategy)
    strat.env = types.SimpleNamespace(generator=AnatomyFieldRenderer(in_channels=1, out_channels=1, latent_channels=16))

    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    assert torch.isfinite(out["loss_total"])


def test_training_encodes_at_the_source_field() -> None:
    # #197: the Task-1 target field is a constant 7 T; the per-sample SOURCE field must reach
    # the encoder, or the "field-invariant" latent never sees which field it must factor out.
    gen = _RecordingRenderer()
    compute_cross_field_losses(gen, _batch(), lambda_latent_cycle=0.0)
    torch.testing.assert_close(gen.last_bs, torch.tensor([0.1, 3.0]))
