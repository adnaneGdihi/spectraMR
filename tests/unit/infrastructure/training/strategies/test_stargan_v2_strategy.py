"""Unit tests for :class:`StarGANv2TrainingStrategy` (multi-domain FIELD translation).

StarGAN v2 (Choi et al., CVPR 2020) trains FOUR networks jointly — generator,
mapping network, style encoder (generator side) and a multi-domain discriminator.
The load-bearing anti-facade requirement (CLAUDE.md pitfall #16): the mapping
network and the style encoder are *distinctive* StarGAN-v2 machinery, so a green
smoke must prove they are BOTH (a) optimized (their params live in an optimizer)
and (b) actually learning (they receive non-zero gradients through the total
loss). An unoptimized mapping net = a StarGAN-v2 that silently collapses to a
plain AdaIN denoiser.

Built with the repo's ``object.__new__`` strategy-unit-test idiom (mirrors
``test_cut_strategy`` / ``test_cyclegan_strategy``): bypass the heavy base
``__init__`` but run the REAL ``setup_models`` / ``_field_to_domain`` /
``_compute_losses_impl`` so the four distinctive terms are genuinely exercised.
"""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.strategies.stargan_v2_strategy import (
    StarGANv2TrainingStrategy,
)

pytestmark = pytest.mark.unit


def _batch() -> dict:
    """Synthetic MRIxFields batch: source ``input`` (0.1 T ULF), target-field
    ``target``, source + target Tesla, and a contrast id."""
    return {
        "input": torch.rand(2, 1, 32, 32),
        "target": torch.rand(2, 1, 32, 32),
        "field_strength": torch.tensor([0.1, 0.1]),
        "field_strength_target": torch.tensor([7.0, 3.0]),
        "contrast_id": torch.tensor([0, 0]),
    }


@pytest.fixture
def minimal_stargan_config() -> TrainingSettings:
    """Tiny frozen ``TrainingSettings`` carrying the four ``losses.gan`` weights the
    StarGAN-v2 strategy reads from the config SSOT (Task 5)."""
    return TrainingSettings(
        data={"train_path": "/tmp/t", "val_path": "/tmp/v", "batch_size": 2},
        model={
            "model_type": "stargan_v2_generator",
            "in_channels": 1,
            "out_channels": 1,
        },
        optimization={"learning_rate": 2e-4},
        logging={},
        losses={
            "gan": {
                "lambda_style": 1.0,
                "lambda_diversity": 1.0,
                "lambda_cycle": 1.0,
                "lambda_r1": 1.0,
                "enable_r1": True,
            }
        },
    )


def _make_strategy(
    config: TrainingSettings, *, wire_optimizers: bool = False
) -> StarGANv2TrainingStrategy:
    """Build the strategy via ``object.__new__`` (bypass base ``__init__``).

    ``wire_optimizers=True`` mirrors the production wiring: a primary
    ``stargan_v2_generator`` / ``stargan_v2_discriminator`` live on ``env`` with an
    ``opt_g`` / ``opt_d`` already built over them, so ``setup_models`` reuses them
    and folds the strategy-owned mapping-network + style-encoder into ``opt_g``.
    """
    strat = StarGANv2TrainingStrategy.__new__(StarGANv2TrainingStrategy)
    strat._owned_modules, strat._owned_wrappers = {}, {}
    strat.config = config
    gen = disc = opt_g = opt_d = None
    if wire_optimizers:
        from spectramr.models.discriminators.stargan_v2_discriminator import (
            StarGANv2Discriminator,
        )
        from spectramr.models.generators.stargan_v2 import StarGANv2Generator

        gen = StarGANv2Generator(img_channels=1, style_dim=64)
        disc = StarGANv2Discriminator(img_channels=1, num_domains=5)
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


# --------------------------------------------------------------------------- #
# _field_to_domain
# --------------------------------------------------------------------------- #
def test_field_to_domain_exact(minimal_stargan_config: TrainingSettings) -> None:
    """Exact field levels snap to their own domain index 0..4."""
    strat = _make_strategy(minimal_stargan_config)
    dom = strat._field_to_domain(torch.tensor([0.1, 1.5, 3.0, 5.0, 7.0]))
    assert dom.tolist() == [0, 1, 2, 3, 4]
    assert dom.dtype == torch.long


def test_field_to_domain_nearest(minimal_stargan_config: TrainingSettings) -> None:
    """Continuous Tesla snaps to the NEAREST discrete level (2.9→3 T=idx 2,
    6.5→7 T=idx 4, 0.4→0.1 T=idx 0)."""
    strat = _make_strategy(minimal_stargan_config)
    dom = strat._field_to_domain(torch.tensor([2.9, 6.5, 0.4]))
    assert dom.tolist() == [2, 4, 0]


# --------------------------------------------------------------------------- #
# Mechanism-fires: the four distinctive terms genuinely fire
# --------------------------------------------------------------------------- #
def test_stargan_style_diversity_cycle_fire(
    minimal_stargan_config: TrainingSettings,
) -> None:
    """The distinctive style / diversity / cycle terms are finite and diversity is
    non-zero (proof the two-latent branch actually produced two DIFFERENT fakes)."""
    strat = _make_strategy(minimal_stargan_config)
    torch.manual_seed(0)
    losses = strat._compute_losses_impl(_batch())
    for key in ("style", "diversity", "cycle", "adv_g", "g_total_loss"):
        assert key in losses, f"missing loss component {key}"
        assert torch.isfinite(losses[key]), f"{key} is not finite"
    # Diversity is MAXIMIZED (negative sign) and must be non-zero: two distinct
    # latents produced two distinct fakes.
    assert losses["diversity"] != 0
    assert losses["diversity"] < 0  # maximize diversity => negative contribution


def test_stargan_no_pixel_l1_to_target(
    minimal_stargan_config: TrainingSettings,
) -> None:
    """StarGAN v2 has NO paired pixel-L1 between the fake and the real target — the
    only reconstruction term is the CYCLE back to the source (pitfall #16)."""
    strat = _make_strategy(minimal_stargan_config)
    losses = strat._compute_losses_impl(_batch())
    assert not any(
        k in losses for k in ("recon", "l1", "loss_l1", "reconstruction", "identity")
    )


def test_stargan_total_is_sum_of_components(
    minimal_stargan_config: TrainingSettings,
) -> None:
    """g_total = adv_g + style + diversity + cycle (each already lambda-weighted)."""
    strat = _make_strategy(minimal_stargan_config)
    torch.manual_seed(0)
    losses = strat._compute_losses_impl(_batch())
    expected = losses["adv_g"] + losses["style"] + losses["diversity"] + losses["cycle"]
    assert torch.allclose(losses["g_total_loss"], expected)


# --------------------------------------------------------------------------- #
# Anti-facade: mapping + style_encoder are OPTIMIZED and TRAINED
# --------------------------------------------------------------------------- #
def test_mapping_and_style_encoder_registered_on_optimizer(
    minimal_stargan_config: TrainingSettings,
) -> None:
    """The mapping-network + style-encoder params (strategy-owned, built after the
    base AMP pass) land in ``opt_g.param_groups`` — folded into the generator
    optimizer so all of G/mapping/style_encoder train together."""
    strat = _make_strategy(minimal_stargan_config, wire_optimizers=True)
    owned = {id(p) for p in strat.mapping.parameters()}
    owned |= {id(p) for p in strat.style_encoder.parameters()}
    assert owned, "mapping / style_encoder have no parameters?"
    opt_ids = {id(p) for grp in strat.env.opt_g.param_groups for p in grp["params"]}
    assert owned <= opt_ids, "mapping/style_encoder params NOT in opt_g (facade!)"


def test_mapping_and_style_encoder_receive_gradients(
    minimal_stargan_config: TrainingSettings,
) -> None:
    """Backprop of the total loss delivers non-zero gradient to BOTH the mapping
    network and the style encoder — proof neither is an inert facade."""
    strat = _make_strategy(minimal_stargan_config, wire_optimizers=True)
    torch.manual_seed(0)
    losses = strat._compute_losses_impl(_batch())
    losses["g_total_loss"].backward()

    map_grads = [p.grad for p in strat.mapping.parameters()]
    sty_grads = [p.grad for p in strat.style_encoder.parameters()]
    assert any(
        g is not None and torch.any(g != 0) for g in map_grads
    ), "mapping network received no gradient (facade)"
    assert any(
        g is not None and torch.any(g != 0) for g in sty_grads
    ), "style encoder received no gradient (facade)"


# --------------------------------------------------------------------------- #
# Two-optimiser step + I1/I2/I3 patterns
# --------------------------------------------------------------------------- #
def test_train_step_returns_d_and_g_configs(
    minimal_stargan_config: TrainingSettings,
) -> None:
    """Two-optimiser alternating step: a D-closure (adversarial + R1) + a
    G-closure, each returning a finite scalar loss."""
    strat = _make_strategy(minimal_stargan_config, wire_optimizers=True)
    configs = strat.train_step(_batch(), epoch=0, iteration=1)
    names = [c["name"] for c in configs]
    assert "discriminator" in names and "generator" in names
    for cfg in configs:
        loss = cfg["closure"]()
        assert torch.isfinite(loss)


def test_train_step_restores_train_mode_after_validation(
    minimal_stargan_config: TrainingSettings,
) -> None:
    """I1: ``_validation_forward`` leaves the generator in eval; the bespoke
    ``train_step`` must flip every net back to train mode."""
    strat = _make_strategy(minimal_stargan_config, wire_optimizers=True)
    strat._validation_forward(torch.rand(2, 1, 32, 32))
    assert strat.generator.training is False
    strat.train_step(_batch(), epoch=0, iteration=1)
    for module in (
        strat.generator,
        strat.mapping,
        strat.style_encoder,
        strat.discriminator,
    ):
        assert module.training is True


def test_setup_models_amp_configures_owned_nets(
    minimal_stargan_config: TrainingSettings,
) -> None:
    """I2: the strategy-owned mapping-network + style-encoder (built after the base
    ``__init__`` AMP pass) must themselves be AMP-configured in ``setup_models``."""
    strat = StarGANv2TrainingStrategy.__new__(StarGANv2TrainingStrategy)
    strat._owned_modules, strat._owned_wrappers = {}, {}
    strat.config = minimal_stargan_config
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
    strat.amp_helper = types.SimpleNamespace(
        configure_model_for_amp=lambda m: configured.append(m)
    )
    strat.setup_models()
    assert strat.mapping in configured and strat.style_encoder in configured


def test_g_closure_exposes_plain_metric_names(
    minimal_stargan_config: TrainingSettings,
) -> None:
    """I3: the G-closure routes detached losses through the base anomaly guard and
    exposes PLAIN metric names (style / diversity / cycle / adv_g / g_total_loss),
    while preserving the D-closure's ``d_total_loss``."""
    strat = _make_strategy(minimal_stargan_config, wire_optimizers=True)
    for cfg in strat.train_step(_batch(), epoch=0, iteration=1):
        cfg["closure"]()
    metrics = strat._last_step_metrics
    assert {"style", "diversity", "cycle", "adv_g", "g_total_loss"} <= set(metrics)
    assert "d_total_loss" in metrics
    assert all(
        isinstance(metrics[k], torch.Tensor) and metrics[k].ndim == 0
        for k in ("style", "diversity", "cycle", "g_total_loss")
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_validation_forward_returns_translated_image(
    minimal_stargan_config: TrainingSettings,
) -> None:
    """Validation prediction is a translated image of the input shape (rendered at
    the default target field via the mapping network when no reference is given)."""
    strat = _make_strategy(minimal_stargan_config)
    x = torch.rand(2, 1, 32, 32)
    pred = strat._validation_forward(x)
    assert pred.shape == (2, 1, 32, 32)
    assert torch.isfinite(pred).all()


def test_validation_forward_reference_guided(
    minimal_stargan_config: TrainingSettings,
) -> None:
    """A reference image + batch field yields a reference-guided style translation
    of the input shape."""
    strat = _make_strategy(minimal_stargan_config)
    x = torch.rand(2, 1, 32, 32)
    ref = torch.rand(2, 1, 32, 32)
    pred = strat._validation_forward(x, reference=ref, batch=_batch())
    assert pred.shape == (2, 1, 32, 32)


# --------------------------------------------------------------------------- #
# R1 gradient penalty is non-zero (anti-facade: R1 fires, not silently zeroed)
# --------------------------------------------------------------------------- #
def test_r1_penalty_is_finite_and_positive(
    minimal_stargan_config: TrainingSettings,
) -> None:
    """R1RegularizationLoss must produce a strictly positive, finite value on a
    real synthetic batch — guards against the ``allow_unused=True`` / NaN-guard
    silent-zero paths inside R1RegularizationLoss (a #16-adjacent facade).

    The D-closure records ``r1`` in ``_last_step_metrics``; we drive both closures
    and read that value. On random data with lambda_r1=1 and a discriminator that
    produces non-constant logits, the gradient penalty is strictly > 0.
    """
    strat = _make_strategy(minimal_stargan_config, wire_optimizers=True)
    torch.manual_seed(42)
    configs = strat.train_step(_batch(), epoch=0, iteration=1)
    # Execute the D-closure first (populates _last_step_metrics["r1"]).
    for cfg in configs:
        cfg["closure"]()
    r1_val = strat._last_step_metrics["r1"]
    # R1 must be finite.
    assert torch.isfinite(r1_val), f"R1 penalty is not finite: {r1_val}"
    # R1 must be strictly positive: a real gradient penalty on random inputs is > 0.
    # A zero here means allow_unused returned None grads, or NaN/Inf guard fired —
    # both indicate the discriminator's gradient path is broken (facade, pitfall #16).
    assert r1_val > 0, (
        f"R1 penalty is zero (silently dead): {r1_val}. "
        "The discriminator gradient path through _DomainBoundDiscriminator is broken."
    )


# --------------------------------------------------------------------------- #
# Fix B — enable_r1 knob is wired (pitfall #15)
# --------------------------------------------------------------------------- #
def test_r1_disabled_when_enable_r1_false() -> None:
    """With ``enable_r1: false`` the D-step's ``r1`` metric must be exactly 0.0,
    even when ``lambda_r1`` is non-zero — the knob must be read, not silently
    ignored (CLAUDE.md pitfall #15)."""
    config = TrainingSettings(
        data={"train_path": "/tmp/t", "val_path": "/tmp/v", "batch_size": 2},
        model={"model_type": "stargan_v2_generator", "in_channels": 1, "out_channels": 1},
        optimization={"learning_rate": 2e-4},
        logging={},
        losses={
            "gan": {
                "lambda_style": 1.0,
                "lambda_diversity": 1.0,
                "lambda_cycle": 1.0,
                "lambda_r1": 1.0,   # non-zero weight — yet R1 must be suppressed
                "enable_r1": False,
            }
        },
    )
    strat = _make_strategy(config, wire_optimizers=True)
    torch.manual_seed(42)
    for cfg in strat.train_step(_batch(), epoch=0, iteration=1):
        cfg["closure"]()
    r1_val = strat._last_step_metrics.get("r1", None)
    # The metric must be present and zero (R1 penalty was skipped).
    assert r1_val is not None, "r1 key absent from _last_step_metrics"
    assert float(r1_val) == 0.0, (
        f"R1 was applied despite enable_r1=False: {r1_val}. "
        "The enable_r1 knob is not being read (pitfall #15)."
    )


def test_r1_enabled_when_enable_r1_true() -> None:
    """With ``enable_r1: true`` and a positive ``lambda_r1``, the D-step's ``r1``
    metric must be strictly > 0 — the gradient penalty actually fires."""
    config = TrainingSettings(
        data={"train_path": "/tmp/t", "val_path": "/tmp/v", "batch_size": 2},
        model={"model_type": "stargan_v2_generator", "in_channels": 1, "out_channels": 1},
        optimization={"learning_rate": 2e-4},
        logging={},
        losses={
            "gan": {
                "lambda_style": 1.0,
                "lambda_diversity": 1.0,
                "lambda_cycle": 1.0,
                "lambda_r1": 1.0,
                "enable_r1": True,
            }
        },
    )
    strat = _make_strategy(config, wire_optimizers=True)
    torch.manual_seed(42)
    for cfg in strat.train_step(_batch(), epoch=0, iteration=1):
        cfg["closure"]()
    r1_val = strat._last_step_metrics.get("r1", None)
    assert r1_val is not None, "r1 key absent from _last_step_metrics"
    assert torch.isfinite(r1_val), f"R1 penalty is not finite: {r1_val}"
    assert float(r1_val) > 0, (
        f"R1 is zero despite enable_r1=True and lambda_r1=1.0: {r1_val}. "
        "Discriminator gradient path may be broken (facade, pitfall #16)."
    )


# --------------------------------------------------------------------------- #
# Factory registration
# --------------------------------------------------------------------------- #
def test_strategy_registered_in_factory() -> None:
    """The ``stargan_v2`` short-name resolves via the strategy factory."""
    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    assert "stargan_v2" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    cls = TrainingStrategyFactory()._load_strategy_class("stargan_v2")
    assert cls is StarGANv2TrainingStrategy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
