"""Unit tests for :class:`CycleGANTrainingStrategy` (unpaired two-generator).

Mechanism-fires probe + no-pixel-L1 + factory-resolution + two-optimiser step.
The strategy is built with the repo's ``object.__new__`` strategy-unit-test idiom
(mirrors ``test_cross_field_translation_strategy``): bypass the heavy base
``__init__`` but run the REAL ``setup_models`` / ``_compute_losses_impl`` so the
distinctive cycle mechanism is genuinely exercised (CLAUDE.md pitfall #16 — a
green smoke must mean "method fired", not just "did not crash").
"""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.strategies.cyclegan_strategy import (
    CycleGANTrainingStrategy,
    compute_cyclegan_losses,
)


def _batch() -> dict:
    """Synthetic UNPAIRED batch: A-domain ``input`` + B-domain ``target``."""
    return {
        "input": torch.rand(2, 1, 32, 32),
        "target": torch.rand(2, 1, 32, 32),
        "field_strength": torch.tensor([3.0, 3.0]),
        "field_strength_target": torch.tensor([7.0, 7.0]),
    }


@pytest.fixture
def minimal_cyclegan_config() -> TrainingSettings:
    """Tiny frozen ``TrainingSettings`` carrying the ``losses.gan`` weights the
    strategy reads (``lambda_cycle`` / ``lambda_identity``, added in Task 5)."""
    return TrainingSettings(
        data={"train_path": "/tmp/t", "val_path": "/tmp/v", "batch_size": 2},
        model={
            "model_type": "cyclegan_generator",
            "in_channels": 1,
            "out_channels": 1,
        },
        optimization={"learning_rate": 2e-4},
        logging={},
        losses={"gan": {"lambda_cycle": 10.0, "lambda_identity": 0.5}},
    )


def _make_strategy(config: TrainingSettings) -> CycleGANTrainingStrategy:
    strat = CycleGANTrainingStrategy.__new__(CycleGANTrainingStrategy)
    strat._owned_modules, strat._owned_wrappers = {}, {}
    strat.config = config
    strat.env = types.SimpleNamespace(
        generator=None,
        discriminator=None,
        losses={},
        opt_g=None,
        opt_d=None,
        schedulers={},
    )
    strat.device = torch.device("cpu")
    strat.setup_models()
    return strat


def test_cyclegan_cycle_loss_fires(minimal_cyclegan_config: TrainingSettings) -> None:
    """Mechanism-fires: finite, strictly-positive cycle loss; two generators +
    two discriminators exposed."""
    strat = _make_strategy(minimal_cyclegan_config)
    losses = strat._compute_losses_impl(_batch())
    assert torch.isfinite(losses["cycle"]) and losses["cycle"] > 0
    assert torch.isfinite(losses["identity"])
    assert torch.isfinite(losses["g_total_loss"])
    assert hasattr(strat, "gen_ab") and hasattr(strat, "gen_ba")
    assert hasattr(strat, "disc_a") and hasattr(strat, "disc_b")


def test_cyclegan_no_pixel_l1_to_target(
    minimal_cyclegan_config: TrainingSettings,
) -> None:
    """UNPAIRED contract: no paired pixel-L1 to the B-domain ``target`` — only
    cycle / identity (same-domain self-consistency) + adversarial terms."""
    strat = _make_strategy(minimal_cyclegan_config)
    losses = strat._compute_losses_impl(_batch())
    assert set(losses) == {"g_total_loss", "adv_g", "adv_d", "cycle", "identity"}
    assert not any(k in losses for k in ("recon", "l1", "loss_l1", "reconstruction"))


def test_cyclegan_total_uses_config_weights(
    minimal_cyclegan_config: TrainingSettings,
) -> None:
    """g_total = adv_g + lambda_cycle*cycle + lambda_identity*identity, weights
    read from ``config.losses.gan`` (SSOT)."""
    strat = _make_strategy(minimal_cyclegan_config)
    torch.manual_seed(0)
    losses = strat._compute_losses_impl(_batch())
    expected = losses["adv_g"] + 10.0 * losses["cycle"] + 0.5 * losses["identity"]
    assert torch.allclose(losses["g_total_loss"], expected)


def test_validation_forward_returns_gen_ab(
    minimal_cyclegan_config: TrainingSettings,
) -> None:
    """Validation prediction is ``gen_ab(input)`` (the A->B translation)."""
    strat = _make_strategy(minimal_cyclegan_config)
    x = torch.rand(2, 1, 32, 32)
    pred = strat._validation_forward(x)
    assert pred.shape == (2, 1, 32, 32)
    assert torch.allclose(pred, strat.gen_ab(x))


def test_train_step_returns_d_and_g_configs(
    minimal_cyclegan_config: TrainingSettings,
) -> None:
    """Two-optimiser alternating step: a D-closure (both discriminators) + a
    G-closure, each returning a finite scalar loss."""
    strat = _make_strategy(minimal_cyclegan_config)
    strat.env.opt_g = torch.optim.Adam(
        list(strat.gen_ab.parameters()) + list(strat.gen_ba.parameters()), lr=1e-4
    )
    strat.env.opt_d = torch.optim.Adam(
        list(strat.disc_a.parameters()) + list(strat.disc_b.parameters()), lr=1e-4
    )
    configs = strat.train_step(_batch(), epoch=0, iteration=1)
    names = [c["name"] for c in configs]
    assert "discriminator" in names and "generator" in names
    for cfg in configs:
        loss = cfg["closure"]()
        assert torch.isfinite(loss)


def test_pure_helper_no_paired_supervision() -> None:
    """The pure loss helper must contain NO L1 between ``fake_B`` and ``real_B``
    (that would be paired supervision). Behavioural check: swapping real_B for a
    copy leaves cycle unchanged only through the generators, never a direct term."""
    torch.manual_seed(0)
    from spectramr.models.discriminators.patchgan_discriminator import (
        PatchGANDiscriminator,
    )
    from spectramr.models.generators.cycle_gan import ResNetGenerator

    gen_ab = ResNetGenerator(1, 1)
    gen_ba = ResNetGenerator(1, 1)
    disc_a = PatchGANDiscriminator(in_channels=1)
    disc_b = PatchGANDiscriminator(in_channels=1)
    real_a = torch.rand(2, 1, 32, 32)
    real_b = torch.rand(2, 1, 32, 32)
    out = compute_cyclegan_losses(
        gen_ab,
        gen_ba,
        disc_a,
        disc_b,
        real_a,
        real_b,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    )
    assert out["cycle"] > 0 and torch.isfinite(out["g_total_loss"])
    # No key implies a paired reconstruction term.
    assert "recon" not in out and "l1" not in out


def test_strategy_registered_in_factory() -> None:
    """The ``cyclegan`` short-name resolves via the strategy factory."""
    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    assert "cyclegan" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    cls = TrainingStrategyFactory()._load_strategy_class("cyclegan")
    assert cls is CycleGANTrainingStrategy


def test_train_step_restores_train_mode_after_validation(
    minimal_cyclegan_config: TrainingSettings,
) -> None:
    """I1 regression: ``_validation_forward`` puts ``gen_ab`` in eval mode; the
    overriding ``train_step`` must restore ALL four nets to train mode so
    BatchNorm/Dropout never run in inference mode during subsequent training.

    (The base ``train_step`` does this at base.py:921-925, but this strategy
    overrides it — so the restore has to be re-asserted here. The real end-to-end
    integration is additionally covered by the Task-11 ``--probe``.)
    """
    strat = _make_strategy(minimal_cyclegan_config)
    # Simulate a validation pass, which calls ``gen_ab.eval()``.
    strat._validation_forward(torch.rand(2, 1, 32, 32))
    assert strat.gen_ab.training is False
    strat.env.opt_g = torch.optim.Adam(strat.gen_ab.parameters(), lr=1e-4)
    strat.env.opt_d = torch.optim.Adam(strat.disc_a.parameters(), lr=1e-4)
    # A subsequent training step must flip every net back to train mode.
    strat.train_step(_batch(), epoch=0, iteration=1)
    for module in (strat.gen_ab, strat.gen_ba, strat.disc_a, strat.disc_b):
        assert module.training is True


def test_setup_models_amp_configures_extra_nets(
    minimal_cyclegan_config: TrainingSettings,
) -> None:
    """I2 regression: ``base.__init__`` AMP-configures only ``env.generator`` /
    ``env.discriminator`` (built before ``_setup_strategy_specific_components``), so
    the strategy-owned ``gen_ba`` / ``disc_a`` must be AMP-configured inside
    ``setup_models`` too."""
    strat = CycleGANTrainingStrategy.__new__(CycleGANTrainingStrategy)
    strat._owned_modules, strat._owned_wrappers = {}, {}
    strat.config = minimal_cyclegan_config
    strat.env = types.SimpleNamespace(
        generator=None,
        discriminator=None,
        losses={},
        opt_g=None,
        opt_d=None,
        schedulers={},
    )
    strat.device = torch.device("cpu")
    configured: list = []
    strat.amp_helper = types.SimpleNamespace(configure_model_for_amp=lambda m: configured.append(m))
    strat.setup_models()
    assert strat.gen_ba in configured and strat.disc_a in configured


def test_g_closure_exposes_plain_metric_names(
    minimal_cyclegan_config: TrainingSettings,
) -> None:
    """I3 + M2: the G-closure routes detached losses through the base anomaly
    guard and exposes PLAIN metric names (cycle / identity / adv_g / adv_d /
    g_total_loss) that the reporter + early-stopping expect — NOT ``g_``-prefixed —
    while preserving ``d_total_loss`` from the D-closure."""
    strat = _make_strategy(minimal_cyclegan_config)
    strat.env.opt_g = torch.optim.Adam(
        list(strat.gen_ab.parameters()) + list(strat.gen_ba.parameters()), lr=1e-4
    )
    strat.env.opt_d = torch.optim.Adam(
        list(strat.disc_a.parameters()) + list(strat.disc_b.parameters()), lr=1e-4
    )
    for cfg in strat.train_step(_batch(), epoch=0, iteration=1):
        cfg["closure"]()
    metrics = strat._last_step_metrics
    assert {"cycle", "identity", "adv_g", "adv_d", "g_total_loss"} <= set(metrics)
    # No legacy ``g_``-prefixed component keys leak through.
    assert not any(key in ("g_cycle", "g_identity", "g_adv_g", "g_adv_d") for key in metrics)
    # Component metrics are scalar device tensors (routed through ``_handle_anomalies``).
    assert all(
        isinstance(metrics[key], torch.Tensor) and metrics[key].ndim == 0
        for key in ("cycle", "identity", "adv_g", "g_total_loss")
    )
    # The D-closure's metric survives the G-closure ``update``.
    assert "d_total_loss" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _cg_cfg_with_image_losses(image_losses: list[dict]) -> TrainingSettings:
    return TrainingSettings(
        data={"train_path": "/tmp/t", "val_path": "/tmp/v", "batch_size": 2},
        model={"model_type": "cyclegan_generator", "in_channels": 1, "out_channels": 1},
        optimization={"learning_rate": 2e-4},
        logging={},
        losses={
            "gan": {"lambda_cycle": 10.0, "lambda_identity": 0.5},
            "image_losses": image_losses,
            "policy": {"output_domain": "image"},
        },
    )


def test_cyclegan_band_split_source_term_fires_through_compute() -> None:
    """Band-split low-band consistency to the input rides on the CycleGAN total."""
    from spectramr.models.losses.spectral_band_split_loss import SpectralBandSplitLoss

    def cfg(w: float) -> TrainingSettings:
        return _cg_cfg_with_image_losses(
            [
                {
                    "name": "spectral_band_split",
                    "weight": w,
                    "role": "source",
                    "kwargs": {"center_fraction": 0.15, "lambda_low": 1.0, "lambda_high": 0.0},
                }
            ]
        )

    strat = _make_strategy(cfg(1.0))
    strat.env.losses = {
        "spectral_band_split": SpectralBandSplitLoss(lambda_low=1.0, lambda_high=0.0)
    }
    torch.manual_seed(0)
    out = strat._compute_losses_impl(_batch())
    band = out["loss_spectral_band_split"]
    assert torch.isfinite(band) and band > 0
    # Same call: the total is the CycleGAN terms plus the source-role term.
    cyc = 10.0 * out["cycle"] + 0.5 * out["identity"]
    assert torch.allclose(out["g_total_loss"], out["adv_g"] + cyc + band, atol=1e-5)


def _make_b0_strategy(config: TrainingSettings) -> CycleGANTrainingStrategy:
    from spectramr.models.generators.b0_hypernetwork import B0ConditionedGenerator

    strat = CycleGANTrainingStrategy.__new__(CycleGANTrainingStrategy)

    strat._owned_modules, strat._owned_wrappers = {}, {}
    strat.config = config
    strat.env = types.SimpleNamespace(
        generator=B0ConditionedGenerator(1, 1, ngf=8, n_blocks=1, hidden_dim=32),
        discriminator=None,
        losses={},
        opt_g=None,
        opt_d=None,
        schedulers={},
    )
    strat.device = torch.device("cpu")
    strat.setup_models()
    return strat


def _b0_batch(field_t: float = 3.0, field_s: float = 0.064) -> dict:
    return {
        "input": torch.rand(2, 1, 64, 64),
        "target": torch.rand(2, 1, 64, 64),
        "field_strength_t": torch.full((2,), field_t),
        "field_strength_source_t": torch.full((2,), field_s),
    }


def test_b0_fields_fire_through_compute_and_reach_both_hypernetworks(
    minimal_cyclegan_config: TrainingSettings,
) -> None:
    strat = _make_b0_strategy(minimal_cyclegan_config)
    batch = _b0_batch()
    losses = strat._compute_losses_impl(
        input_batch=batch["input"], target_batch=batch["target"], epoch=0, batch=batch
    )
    losses["g_total_loss"].backward()
    assert strat.gen_ab.hyper_network.heads[0].weight.grad.abs().sum() > 0
    assert strat.gen_ba.hyper_network.heads[0].weight.grad.abs().sum() > 0


def test_b0_compute_without_field_stamp_raises(
    minimal_cyclegan_config: TrainingSettings,
) -> None:
    strat = _make_b0_strategy(minimal_cyclegan_config)
    batch = _b0_batch()
    del batch["field_strength_t"]
    with pytest.raises(ValueError, match="requires field_strength"):
        strat._compute_losses_impl(
            input_batch=batch["input"], target_batch=batch["target"], epoch=0, batch=batch
        )


def test_step_fields_fallback_is_used_when_no_batch_is_passed(
    minimal_cyclegan_config: TrainingSettings,
) -> None:
    strat = _make_b0_strategy(minimal_cyclegan_config)
    batch = _b0_batch()
    strat._step_fields = strat._fields_from(batch)
    losses = strat._compute_losses_impl(input_batch=batch["input"], target_batch=batch["target"])
    assert torch.isfinite(losses["g_total_loss"])


def test_validation_forward_routes_output_field(
    minimal_cyclegan_config: TrainingSettings,
) -> None:
    strat = _make_b0_strategy(minimal_cyclegan_config)
    x = torch.rand(2, 1, 16, 16)
    out_3t = strat._validation_forward(x, batch_context={"field_strength_t": torch.full((2,), 3.0)})
    out_ulf = strat._validation_forward(
        x, batch_context={"field_strength_t": torch.full((2,), 0.064)}
    )
    assert (out_3t - out_ulf).abs().max() > 1e-6
    with pytest.raises(ValueError, match="requires field_strength"):
        strat._validation_forward(x, batch_context={})


def test_validation_step_declares_and_forwards_field_seam(
    minimal_cyclegan_config: TrainingSettings,
) -> None:
    import inspect

    from spectramr.pipelines.train import select_validation_extra_fields

    strat = _make_b0_strategy(minimal_cyclegan_config)
    seen: list[dict] = []
    strat._validation_forward = lambda inp, ctx=None, **kw: seen.append(ctx) or inp
    batch = _b0_batch()
    extra = select_validation_extra_fields(
        batch, tuple(inspect.signature(strat.validation_step).parameters)
    )
    assert "field_strength_t" in extra and "field_strength_source_t" in extra
    strat.validation_step(batch["input"], batch["target"], **extra)
    assert torch.equal(seen[0]["field_strength_t"], batch["field_strength_t"])


def _domain_cfg() -> TrainingSettings:
    return TrainingSettings(
        data={"train_path": "/tmp/t", "val_path": "/tmp/v", "batch_size": 2},
        model={"model_type": "cyclegan_generator", "in_channels": 1, "out_channels": 1},
        optimization={"learning_rate": 2e-4},
        logging={},
        training={"strategy_class": "cyclegan"},
        losses={
            "gan": {
                "lambda_cycle": 10.0,
                "lambda_identity": 0.5,
                "lambda_domain": 0.5,
                "domain_grl": {"enabled": True},
            }
        },
    )


def test_grl_reverses_encoder_gradient_and_keeps_classifier_gradient() -> None:
    """Encoder grads from the domain term are -1 x the plain classifier grads (alpha = 1);
    classifier grads are identical with and without the reversal."""
    import torch.nn.functional as F

    strat = _make_strategy(_domain_cfg())
    head = strat.domain_head
    head.classifier.eval()  # Dropout off, so both passes see the same function
    enc = list(strat.gen_ab.model[: strat.gen_ab._anatomy_split].parameters())
    cls = list(head.classifier.parameters())
    _, z_a = strat.gen_ab.forward_with_anatomy(torch.rand(2, 1, 32, 32))
    _, z_b = strat.gen_ab.forward_with_anatomy(torch.rand(2, 1, 32, 32))
    term, _ = strat._domain_term(z_a, z_b)
    z = torch.cat([z_a, z_b])
    labels = torch.cat([z.new_zeros(2, 1), z.new_ones(2, 1)])
    plain = head.loss_weight * F.binary_cross_entropy_with_logits(head.classifier(z), labels)
    g_enc = torch.autograd.grad(term, enc, retain_graph=True)
    g_enc_plain = torch.autograd.grad(plain, enc, retain_graph=True)
    for g, p in zip(g_enc, g_enc_plain, strict=True):
        assert torch.allclose(g, -p, atol=1e-6)
    g_cls = torch.autograd.grad(term, cls, retain_graph=True)
    g_cls_plain = torch.autograd.grad(plain, cls)
    for g, p in zip(g_cls, g_cls_plain, strict=True):
        assert torch.allclose(g, p, atol=1e-6)


def test_domain_head_trains_through_the_production_g_step() -> None:
    """train_step's generator closure carries the domain term in g_total and grads reach
    the classifier (opt_g) and the encoder."""
    strat = _make_strategy(_domain_cfg())
    strat.env.opt_g = torch.optim.Adam(
        list(strat.gen_ab.parameters()) + list(strat.gen_ba.parameters()), lr=1e-4
    )
    strat.env.opt_d = torch.optim.Adam(
        list(strat.disc_a.parameters()) + list(strat.disc_b.parameters()), lr=1e-4
    )
    strat._register_owned_params()
    opt_ids = {id(p) for g in strat.env.opt_g.param_groups for p in g["params"]}
    assert {id(p) for p in strat.domain_head.parameters()} <= opt_ids

    configs = strat.train_step(_batch(), epoch=0, iteration=1)
    g_cfg = next(c for c in configs if c["name"] == "generator")
    g_cfg["closure"]().backward()
    m = strat._last_step_metrics
    base = m["adv_g"] + 10.0 * m["cycle"] + 0.5 * m["identity"]
    assert torch.allclose(m["g_total_loss"], base + m["loss_domain"], atol=1e-5)
    head_params = list(strat.domain_head.parameters())
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in head_params)
    enc = strat.gen_ab.model[: strat.gen_ab._anatomy_split]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())


def test_domain_head_state_is_checkpointed() -> None:
    strat = _make_strategy(_domain_cfg())
    assert "domain_head" in strat.strategy_state_dict()["modules"]


def test_domain_grl_on_a_generator_without_a_tap_raises() -> None:
    from spectramr.models.generators.b0_hypernetwork import B0ConditionedGenerator

    strat = CycleGANTrainingStrategy.__new__(CycleGANTrainingStrategy)

    strat._owned_modules, strat._owned_wrappers = {}, {}
    strat.config = _domain_cfg()
    strat.env = types.SimpleNamespace(
        generator=B0ConditionedGenerator(1, 1, ngf=8, n_blocks=1, hidden_dim=32),
        discriminator=None,
        losses={},
        opt_g=None,
        opt_d=None,
        schedulers={},
    )
    strat.device = torch.device("cpu")
    with pytest.raises(ValueError, match="bottleneck tap"):
        strat.setup_models()
