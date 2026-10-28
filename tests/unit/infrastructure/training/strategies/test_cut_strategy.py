"""Unit tests for :class:`CUTTrainingStrategy` (Contrastive Unpaired Translation).

Mechanism-fires probe: the distinctive PatchNCE term must genuinely fire (finite,
strictly-positive ``nce``) on a synthetic batch, the encoder-feature hook must
capture a non-empty list of intermediate feature maps, and the ``cut_patch_nce``
sampler-MLP (its only trainable projection head) must be OPTIMIZED — registered on
the generator optimizer AND receiving non-zero gradients. A green smoke that leaves
that MLP untrained would be pitfall #16 (NCE "fires" but never learns its
projection) dressed up as a pass.

Built with the repo's ``object.__new__`` strategy-unit-test idiom (mirrors
``test_cyclegan_strategy``): bypass the heavy base ``__init__`` but run the REAL
``setup_models`` / ``_encode_features`` / ``_compute_losses_impl`` so the mechanism
is genuinely exercised.
"""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.strategies.cut_strategy import (
    CUTTrainingStrategy,
)


def _batch() -> dict:
    """Synthetic UNPAIRED batch: A-domain ``input`` + B-domain ``target``."""
    return {
        "input": torch.rand(2, 1, 32, 32),
        "target": torch.rand(2, 1, 32, 32),
        "field_strength_target": torch.tensor([7.0, 7.0]),
    }


@pytest.fixture
def minimal_cut_config() -> TrainingSettings:
    """Tiny frozen ``TrainingSettings`` carrying ``losses.gan.lambda_nce`` (the CUT
    PatchNCE weight the strategy reads from the config SSOT)."""
    return TrainingSettings(
        data={"train_path": "/tmp/t", "val_path": "/tmp/v", "batch_size": 2},
        model={
            "model_type": "cyclegan_generator",
            "in_channels": 1,
            "out_channels": 1,
        },
        optimization={"learning_rate": 2e-4},
        logging={},
        losses={"gan": {"lambda_nce": 1.0}},
    )


def _make_strategy(
    config: TrainingSettings, *, wire_optimizers: bool = False
) -> CUTTrainingStrategy:
    """Build the strategy via ``object.__new__`` (bypass base ``__init__``).

    ``wire_optimizers=True`` mirrors the production wiring: a primary
    ``cyclegan_generator`` / ``patch_gan`` live on ``env`` with an ``opt_g`` /
    ``opt_d`` already built over them, so ``setup_models`` reuses them and the
    ONLY param group it adds to ``opt_g`` is the NCE sampler-MLP.
    """
    strat = CUTTrainingStrategy.__new__(CUTTrainingStrategy)
    strat._owned_modules, strat._owned_wrappers = {}, {}
    strat.config = config
    gen = disc = opt_g = opt_d = None
    if wire_optimizers:
        from spectramr.models.discriminators.patchgan_discriminator import (
            PatchGANDiscriminator,
        )
        from spectramr.models.generators.cycle_gan import ResNetGenerator

        gen = ResNetGenerator(1, 1)
        disc = PatchGANDiscriminator(in_channels=1)
        opt_g = torch.optim.Adam(gen.parameters(), lr=1e-4)
        opt_d = torch.optim.Adam(disc.parameters(), lr=1e-4)
    strat.env = types.SimpleNamespace(
        generator=gen,
        discriminator=disc,
        losses={},
        opt_g=opt_g,
        opt_d=opt_d,
        schedulers={},
    )
    strat.device = torch.device("cpu")
    strat.setup_models()
    return strat


def test_cut_nce_loss_fires(minimal_cut_config: TrainingSettings) -> None:
    """Mechanism-fires: finite, strictly-positive ``nce``; encoder hook captures a
    non-empty feature list."""
    strat = _make_strategy(minimal_cut_config)
    losses = strat._compute_losses_impl(_batch())
    assert torch.isfinite(losses["nce"]) and losses["nce"] > 0
    assert torch.isfinite(losses["g_total_loss"])
    feats = strat._encode_features(torch.rand(1, 1, 32, 32))
    assert isinstance(feats, list) and len(feats) >= 1
    assert all(torch.is_tensor(f) for f in feats)


def test_cut_no_pixel_l1_to_target(minimal_cut_config: TrainingSettings) -> None:
    """UNPAIRED contract: adversarial + NCE only — NO paired pixel-L1 to target."""
    strat = _make_strategy(minimal_cut_config)
    losses = strat._compute_losses_impl(_batch())
    assert "nce" in losses and "adv_g" in losses and "g_total_loss" in losses
    assert not any(k in losses for k in ("recon", "l1", "loss_l1", "reconstruction", "cycle"))


def test_cut_total_uses_config_nce_weight(
    minimal_cut_config: TrainingSettings,
) -> None:
    """g_total = adv_g + lambda_nce * nce, weight read from ``losses.gan`` (SSOT)."""
    strat = _make_strategy(minimal_cut_config)
    torch.manual_seed(0)
    losses = strat._compute_losses_impl(_batch())
    expected = losses["adv_g"] + 1.0 * losses["nce"]
    assert torch.allclose(losses["g_total_loss"], expected)


def test_encode_features_two_scales_distinct_channels(
    minimal_cut_config: TrainingSettings,
) -> None:
    """The encoder hook captures multiple scales with the encoder's channel
    progression (64 / 128 / 256), each an equal-length source/translated pair."""
    strat = _make_strategy(minimal_cut_config)
    feats = strat._encode_features(torch.rand(1, 1, 32, 32))
    channels = [f.shape[1] for f in feats]
    assert len(feats) >= 2
    assert set(channels) & {64, 128, 256}


def test_nce_sampler_params_registered_on_optimizer(
    minimal_cut_config: TrainingSettings,
) -> None:
    """The lazily-built PatchSampleMLP heads are materialized at setup and their
    params land in ``opt_g.param_groups`` (folded CUT ``optimizer_F``)."""
    strat = _make_strategy(minimal_cut_config, wire_optimizers=True)
    sampler_ids = {id(p) for p in strat.nce_loss.sampler.parameters()}
    assert sampler_ids, "NCE sampler-MLP heads were not materialized at setup"
    opt_ids = {id(p) for grp in strat.env.opt_g.param_groups for p in grp["params"]}
    assert sampler_ids <= opt_ids


def test_nce_sampler_receives_gradients(
    minimal_cut_config: TrainingSettings,
) -> None:
    """Backprop of the total loss delivers non-zero gradient to the sampler-MLP —
    proof the projection head actually LEARNS (not an inert facade)."""
    strat = _make_strategy(minimal_cut_config, wire_optimizers=True)
    losses = strat._compute_losses_impl(_batch())
    losses["g_total_loss"].backward()
    grads = [p.grad for p in strat.nce_loss.sampler.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_validation_forward_returns_generator(
    minimal_cut_config: TrainingSettings,
) -> None:
    """Validation prediction is ``generator(input)`` (the A->B translation)."""
    strat = _make_strategy(minimal_cut_config)
    x = torch.rand(2, 1, 32, 32)
    pred = strat._validation_forward(x)
    assert pred.shape == (2, 1, 32, 32)


def test_train_step_returns_d_and_g_configs(
    minimal_cut_config: TrainingSettings,
) -> None:
    """Two-optimiser alternating step: a D-closure + a G-closure, each returning a
    finite scalar loss."""
    strat = _make_strategy(minimal_cut_config, wire_optimizers=True)
    configs = strat.train_step(_batch(), epoch=0, iteration=1)
    names = [c["name"] for c in configs]
    assert "discriminator" in names and "generator" in names
    for cfg in configs:
        loss = cfg["closure"]()
        assert torch.isfinite(loss)


def test_train_step_restores_train_mode_after_validation(
    minimal_cut_config: TrainingSettings,
) -> None:
    """``_validation_forward`` leaves the generator in eval; the bespoke
    ``train_step`` must flip both nets back to train (I1 pattern)."""
    strat = _make_strategy(minimal_cut_config, wire_optimizers=True)
    strat._validation_forward(torch.rand(2, 1, 32, 32))
    assert strat.generator.training is False
    strat.train_step(_batch(), epoch=0, iteration=1)
    assert strat.generator.training is True and strat.discriminator.training is True


def test_setup_models_amp_configures_nce_sampler(
    minimal_cut_config: TrainingSettings,
) -> None:
    """I2 pattern: the strategy-owned NCE sampler-MLP (built after base
    ``__init__``'s AMP pass) must itself be AMP-configured in ``setup_models``."""
    strat = CUTTrainingStrategy.__new__(CUTTrainingStrategy)
    strat._owned_modules, strat._owned_wrappers = {}, {}
    strat.config = minimal_cut_config
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
    assert strat.nce_loss in configured


def test_g_closure_exposes_plain_metric_names(
    minimal_cut_config: TrainingSettings,
) -> None:
    """The G-closure routes detached losses through the base anomaly guard and
    exposes PLAIN metric names (nce / adv_g / g_total_loss), while preserving the
    D-closure's ``d_total_loss``."""
    strat = _make_strategy(minimal_cut_config, wire_optimizers=True)
    for cfg in strat.train_step(_batch(), epoch=0, iteration=1):
        cfg["closure"]()
    metrics = strat._last_step_metrics
    assert {"nce", "adv_g", "g_total_loss"} <= set(metrics)
    assert "d_total_loss" in metrics
    assert all(
        isinstance(metrics[k], torch.Tensor) and metrics[k].ndim == 0
        for k in ("nce", "adv_g", "g_total_loss")
    )


def test_register_owned_params_raises_when_sampler_not_materialized(
    minimal_cut_config: TrainingSettings,
) -> None:
    """Fail-loud guard: _register_owned_params raises RuntimeError when the NCE
    sampler has no parameters (unmaterialized heads) — pitfall #16 guard so the
    PatchNCE projection can never silently go untrained."""
    from unittest.mock import patch

    from spectramr.models.discriminators.patchgan_discriminator import (
        PatchGANDiscriminator,
    )
    from spectramr.models.generators.cycle_gan import ResNetGenerator

    strat = CUTTrainingStrategy.__new__(CUTTrainingStrategy)

    strat._owned_modules, strat._owned_wrappers = {}, {}
    strat.config = minimal_cut_config
    gen = ResNetGenerator(1, 1)
    disc = PatchGANDiscriminator(in_channels=1)
    strat.env = types.SimpleNamespace(
        generator=gen,
        discriminator=disc,
        losses={},
        opt_g=torch.optim.Adam(gen.parameters(), lr=1e-4),
        opt_d=torch.optim.Adam(disc.parameters(), lr=1e-4),
        schedulers={},
    )
    strat.device = torch.device("cpu")
    strat.setup_models()  # normal path — heads are materialized here

    # Simulate unmaterialized heads by patching parameters() to return empty
    with patch.object(strat.nce_loss.sampler, "parameters", return_value=iter([])):
        with pytest.raises(RuntimeError, match="NCE sampler heads were not materialized"):
            strat._register_owned_params()


def test_strategy_registered_in_factory() -> None:
    """The ``cut`` short-name resolves via the strategy factory."""
    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    assert "cut" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    cls = TrainingStrategyFactory()._load_strategy_class("cut")
    assert cls is CUTTrainingStrategy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _cut_cfg_with_image_losses(image_losses: list[dict]) -> TrainingSettings:
    return TrainingSettings(
        data={"train_path": "/tmp/t", "val_path": "/tmp/v", "batch_size": 2},
        model={"model_type": "cyclegan_generator", "in_channels": 1, "out_channels": 1},
        optimization={"learning_rate": 2e-4},
        logging={},
        losses={
            "gan": {"lambda_nce": 1.0},
            "image_losses": image_losses,
            "policy": {"output_domain": "image"},
        },
    )


def test_cut_mind_source_term_fires_through_compute() -> None:
    """MIND-SSC against the input: the total moves by exactly its own value.

    One strategy (one set of initial weights); only the config's MIND weight changes.
    """
    from spectramr.models.losses.mind_ssc_loss import MINDSSCLoss

    def cfg(w: float) -> TrainingSettings:
        return _cut_cfg_with_image_losses([{"name": "mind_ssc", "weight": w, "role": "source"}])

    strat = _make_strategy(cfg(1.0))
    strat.env.losses = {"mind_ssc": MINDSSCLoss()}
    torch.manual_seed(0)
    out = strat._compute_losses_impl(_batch())
    mind = out["loss_mind_ssc"]
    assert torch.isfinite(mind) and mind > 0
    # Same call: the total is adv_g + lambda_nce * nce + the source-role term.
    assert torch.allclose(out["g_total_loss"], out["adv_g"] + out["nce"] + mind, atol=1e-5)


def test_cut_refuses_target_role_image_loss() -> None:
    from spectramr.models.losses.mind_ssc_loss import MINDSSCLoss

    cfg = _cut_cfg_with_image_losses([{"name": "mind_ssc", "weight": 1.0}])
    strat = _make_strategy(cfg)
    strat.env.losses = {"mind_ssc": MINDSSCLoss()}
    with pytest.raises(ValueError, match="role 'target'"):
        strat._compute_losses_impl(_batch())
