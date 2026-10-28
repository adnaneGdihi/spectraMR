"""Unit tests for CycleBlochStrategy."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from spectramr.infrastructure.training.strategies.cycle_bloch_strategy import (
    CycleBlochStrategy,
    ParameterEstimator,
)


@pytest.fixture
def mock_env():
    """Create a mock training environment (NEW API - env only)."""
    env = MagicMock()
    env.device = torch.device("cpu")

    # Mock generator/discriminator
    gen = MagicMock()
    gen.return_value = torch.randn(2, 1, 64, 64)  # fake_hf
    gen.training = True
    gen.train = MagicMock()
    env.generator = gen
    env.models = {"generator": gen}

    disc = MagicMock()
    disc.return_value = torch.randn(2, 1, 64, 64)  # pred_fake
    disc.training = True
    disc.train = MagicMock()
    env.discriminator = disc
    env.models["discriminator"] = disc

    # Mock optimizers
    env.opt_g = MagicMock()
    env.opt_d = MagicMock()
    env.optimizers = {"opt_g": env.opt_g, "opt_d": env.opt_d}

    # Config — must give concrete values for the SSOT loss-weight read so the
    # strategy doesn't materialise lambda_cycle as a MagicMock attribute.
    env.config = MagicMock()
    env.config.training.loss_weights = {"cycle_bloch": 5.0, "adv": 2.0}
    env.config.optimization.optimizer.learning_rate = 1e-4
    # schema defaults; bare MagicMock fails StandardOptimizerStepper's raise-on-unknown
    env.config.optimization.gradient.clip.enabled = False
    # mixed_precision.py validates amp_dtype; a bare MagicMock fails it
    env.config.optimization.precision.dtype = "float32"
    env.config.optimization.gradient.clip.method = "norm"
    env.config.optimization.gradient.clip.value = 1.0
    env.config.model.model_type = "cycle_bloch"
    env.config.losses.gan.lambda_cycle_bloch = 5.0
    env.config.losses.gan.lambda_cycle_adv = 2.0
    env.config.losses.gan.gan_loss_type = "lsgan"
    env.config.losses.gan.enable_gradient_penalty = False
    env.config.losses.gan.lambda_gp = 10.0
    env.config.losses.gan.label_smoothing = 0.0
    env.config.training.cycle_bloch = None
    env.config.physics.field_strength = None
    # A bare MagicMock would read as an unknown backend when the estimator is registered.
    env.config.parallel = None
    env.model_type = "cycle_bloch"

    return env


@pytest.fixture
def strategy(mock_env):
    """Instantiate CycleBlochStrategy (NEW API - env only)."""
    with patch("spectramr.infrastructure.di.di_container.resolve_service") as mock_resolve:
        mock_resolve.return_value = MagicMock()
        return CycleBlochStrategy(env=mock_env)


def test_initialization(strategy):
    """Test strategy initialization.

    The ParameterEstimator is created lazily on first forward pass to
    match the actual generator output-channel count (which a
    ``ChannelAdapter`` may rewrite). At construction time
    ``strategy.estimator`` is therefore ``None`` — it materialises
    after the first call to ``_get_estimator``.
    """
    assert strategy is not None
    assert strategy.estimator is None
    assert strategy.lambda_cycle == 5.0
    assert strategy.lambda_adv == 2.0
    # Lazy materialisation works.
    dummy = torch.zeros(1, 1, 4, 4)
    est = strategy._get_estimator(dummy)
    assert isinstance(est, ParameterEstimator)


def test_train_step(strategy, mock_env):
    """Test train_step execution and loss computation."""
    input_batch = torch.randn(2, 1, 64, 64, requires_grad=True)  # ULF
    target_batch = torch.randn(2, 1, 64, 64, requires_grad=True)  # HF

    # Estimator is lazy-init; materialise it once so we can patch its forward.
    strategy._get_estimator(target_batch)

    # Mock estimator output (PD, T1, T2)
    with patch.object(
        strategy.estimator,
        "forward",
        return_value=torch.randn(2, 3, 64, 64, requires_grad=True),
    ) as mock_est:
        # Mock BlochSimulator
        with patch(
            "spectramr.infrastructure.training.strategies.cycle_bloch_physics.BlochSimulator.spin_echo"
        ) as mock_bloch:
            mock_bloch.return_value = torch.randn(2, 1, 64, 64, requires_grad=True)  # rec_ulf

            # Mock criteria to return scalar tensors with gradients
            strategy.criterion_gan = MagicMock(return_value=torch.tensor(1.0, requires_grad=True))
            strategy.criterion_cycle = MagicMock(return_value=torch.tensor(1.0, requires_grad=True))

            # Mock discriminator
            mock_env.discriminator.return_value = torch.randn(2, 1, requires_grad=True)

            steps = strategy.train_step(
                batch=None,
                epoch=0,
                iteration=0,
                input_batch=input_batch,
                target_batch=target_batch,
            )

            # Execute closures to get metrics
            for step in steps:
                step["closure"]()

            metrics = strategy.get_last_metrics()

            assert isinstance(metrics, dict)
            assert "loss_g" in metrics
            assert "loss_cycle" in metrics
            assert "loss_d" in metrics

            # Verify flow
            mock_est.assert_called()
            mock_bloch.assert_called()


def test_train_step_missing_discriminator(strategy, mock_env):
    """Test error when discriminator is missing."""
    mock_env.discriminator = None
    input_batch = torch.randn(2, 1, 64, 64)
    target_batch = torch.randn(2, 1, 64, 64)

    with pytest.raises(ValueError, match="CycleBlochStrategy requires a discriminator"):
        strategy.train_step(None, 0, input_batch=input_batch, target_batch=input_batch)


# ── training.cycle_bloch.backward_model ────────────────────────────────────────


def _ulf_forward_cfg(**overrides):
    from spectramr.config.schemas.training.cycle_bloch import CycleBlochConfig

    kwargs = {
        "backward_model": "ulf_forward",
        "ulf_tr_ms": 500.0,
        "ulf_te_ms": 14.0,
        "ulf_flip_deg": 90.0,
        "tissue_field_tesla": 3.0,
    }
    kwargs.update(overrides)
    return CycleBlochConfig(**kwargs)


def _build(mock_env, cycle_cfg, field_strength):
    mock_env.config.training.cycle_bloch = cycle_cfg
    mock_env.config.physics.field_strength = field_strength
    with patch("spectramr.infrastructure.di.di_container.resolve_service") as mock_resolve:
        mock_resolve.return_value = MagicMock()
        return CycleBlochStrategy(env=mock_env)


def _tissue(batch=2, size=8):
    g = torch.Generator().manual_seed(0)
    pd = 0.5 + torch.rand(batch, 1, size, size, generator=g)
    t1 = 300.0 + 1500.0 * torch.rand(batch, 1, size, size, generator=g)
    t2 = 40.0 + 100.0 * torch.rand(batch, 1, size, size, generator=g)
    return pd, t1, t2


def test_default_backward_model_is_bloch_fixed(strategy):
    assert {
        "backward_model": "bloch_fixed",
        "tr_ms": 500.0,
        "te_ms": 14.0,
    }.items() <= strategy.cycle_bloch_provenance.items()


def test_ulf_forward_at_equal_fields_reproduces_spin_echo(mock_env):
    """No dispersion and a 90 degree flip: the Ernst signal is the spin-echo equation."""
    from spectramr.infrastructure.physics.biophysics.bloch import BlochSimulator

    strat = _build(mock_env, _ulf_forward_cfg(tissue_field_tesla=0.064), 0.064)
    pd, t1, t2 = _tissue()
    expected = BlochSimulator.spin_echo(m0=pd, t1=t1, t2=t2, te=14.0, tr=500.0)
    torch.testing.assert_close(strat._render_ulf(pd, t1, t2, None), expected, rtol=1e-4, atol=1e-5)


def test_ulf_forward_disperses_relaxation_to_the_declared_field(mock_env):
    fixed = _build(mock_env, None, 0.064)
    ulf = _build(mock_env, _ulf_forward_cfg(), 0.064)
    pd, t1, t2 = _tissue()
    a, b = fixed._render_ulf(pd, t1, t2, None), ulf._render_ulf(pd, t1, t2, None)
    assert (a - b).abs().mean() > 1e-3
    assert {
        "backward_model": "ulf_forward",
        "tr_ms": 500.0,
        "te_ms": 14.0,
        "flip_deg": 90.0,
        "b0_target_tesla": 0.064,
        "tissue_field_tesla": 3.0,
    }.items() <= ulf.cycle_bloch_provenance.items()


def _generator_cycle_loss(strat, mock_env, tissue):
    """Run the production generator closure with a fixed estimator output."""
    real_ulf = torch.rand(2, 1, 8, 8)
    mock_env.generator.return_value = torch.rand(2, 1, 8, 8)
    mock_env.discriminator.return_value = torch.zeros(2, 1, 2, 2)
    strat._get_estimator(mock_env.generator.return_value)
    with patch.object(strat.estimator, "forward", return_value=torch.cat(tissue, dim=1)):
        steps = strat.train_step(
            batch=None, epoch=0, iteration=0, input_batch=real_ulf, target_batch=real_ulf
        )
        steps[0]["closure"]()
    return strat.get_last_metrics()["loss_cycle"]


def test_backward_model_changes_the_cycle_loss_on_the_train_path(mock_env):
    torch.manual_seed(0)
    tissue = _tissue()
    fixed = _generator_cycle_loss(_build(mock_env, None, 0.064), mock_env, tissue)
    torch.manual_seed(0)
    ulf = _generator_cycle_loss(_build(mock_env, _ulf_forward_cfg(), 0.064), mock_env, tissue)
    assert not torch.isclose(fixed, ulf, rtol=1e-3)


def test_ulf_forward_without_field_strength_raises(mock_env):
    with pytest.raises(ValueError, match=r"physics\.field_strength"):
        _build(mock_env, _ulf_forward_cfg(), None)


def test_unknown_backward_model_raises(mock_env):
    cfg = MagicMock()
    cfg.backward_model = "epg_magic"
    with pytest.raises(ValueError, match="backward_model='epg_magic'"):
        _build(mock_env, cfg, 0.064)


def test_first_steps_snapshot_stamps_the_backward_model(mock_env):
    strat = _build(mock_env, _ulf_forward_cfg(), 0.064)
    assert strat._first_steps_extra(0)["cycle_bloch"]["backward_model"] == "ulf_forward"


# ── losses.gan.gan_loss_type: the cycle_bloch critic ──────────────────────────


def _critic_strategy(mock_env, critic, gp):
    mock_env.config.losses.gan.gan_loss_type = critic
    mock_env.config.losses.gan.enable_gradient_penalty = gp
    torch.manual_seed(0)
    mock_env.discriminator = torch.nn.Conv2d(1, 1, 3, padding=1)
    return _build(mock_env, None, 0.064)


def _closures(strat, mock_env, target):
    real_ulf = torch.rand(2, 1, 8, 8)
    mock_env.generator.return_value = torch.rand(
        2, 1, 8, 8, generator=torch.Generator().manual_seed(1)
    )
    strat._get_estimator(mock_env.generator.return_value)
    tissue = torch.cat(_tissue(), dim=1)
    steps = strat.train_step(
        batch=None, epoch=0, iteration=0, input_batch=real_ulf, target_batch=target
    )
    with patch.object(strat.estimator, "forward", return_value=tissue):
        g_loss = steps[0]["closure"]()
    return g_loss, steps[1]["closure"]


def test_lsgan_critic_reproduces_the_historical_mse_losses(mock_env):
    strat = _critic_strategy(mock_env, "lsgan", False)
    target = torch.rand(2, 1, 8, 8)
    _closures(strat, mock_env, target)
    disc, fake = mock_env.discriminator, mock_env.generator.return_value
    mse = torch.nn.functional.mse_loss
    with torch.no_grad():
        expected_g = mse(disc(fake), torch.ones_like(disc(fake)))
        expected_d = 0.5 * (
            mse(disc(target), torch.ones_like(disc(target)))
            + mse(disc(fake), torch.zeros_like(disc(fake)))
        )
    _, d_closure = _closures(strat, mock_env, target)
    torch.testing.assert_close(strat.critic.compute_generator_loss(disc(fake)), expected_g)
    torch.testing.assert_close(d_closure(), expected_d)


def test_wgan_gp_critic_adds_the_gradient_penalty_on_the_train_path(mock_env):
    from spectramr.models.losses.gan_loss_library import gradient_penalty_loss

    strat = _critic_strategy(mock_env, "wgan-gp", True)
    target = torch.rand(2, 1, 8, 8)
    _, d_closure = _closures(strat, mock_env, target)
    disc, fake = mock_env.discriminator, mock_env.generator.return_value
    torch.manual_seed(3)
    loss_d = d_closure()
    torch.manual_seed(3)
    gp = gradient_penalty_loss(disc, target, fake)
    expected = -disc(target).mean() + disc(fake).mean() + 10.0 * gp
    torch.testing.assert_close(loss_d, expected)
    assert strat.cycle_bloch_provenance["critic"] == "wgan-gp"
    assert strat.cycle_bloch_provenance["lambda_gp"] == 10.0


def test_critic_changes_the_discriminator_loss(mock_env):
    target = torch.rand(2, 1, 8, 8)
    ls = _closures(_critic_strategy(mock_env, "lsgan", False), mock_env, target)[1]()
    wg = _closures(_critic_strategy(mock_env, "wgan-gp", True), mock_env, target)[1]()
    assert not torch.isclose(ls.detach(), wg.detach(), rtol=1e-3)


@pytest.mark.parametrize(
    ("critic", "gp", "match"),
    [
        ("hinge", False, "not implemented"),
        ("wgan-gp", False, "enable_gradient_penalty=True"),
        ("lsgan", True, "enable_gradient_penalty=False"),
    ],
)
def test_critic_misconfiguration_raises(mock_env, critic, gp, match):
    with pytest.raises(ValueError, match=match):
        _critic_strategy(mock_env, critic, gp)


def test_wgan_gp_without_a_real_target_raises(mock_env):
    strat = _critic_strategy(mock_env, "wgan-gp", True)
    real_ulf = torch.rand(2, 1, 8, 8)
    mock_env.generator.return_value = torch.rand(2, 1, 8, 8)
    d_closure = strat._train_discriminator_step(real_ulf, None, mock_env.discriminator, 0, 0, {})
    with pytest.raises(ValueError, match="real HF target"):
        d_closure()


# ── backward_model: epg_tse, and the estimated B1+ map ─────────────────────────

_EPG = {
    "backward_model": "epg_tse",
    "ulf_tr_ms": 500.0,
    "ulf_te_ms": 14.0,
    "echo_spacing_ms": 7.0,
    "echo_train_length": 4,
    "refocus_deg": 150.0,
}


def _cb(**kwargs):
    from spectramr.config.schemas.training.cycle_bloch import CycleBlochConfig

    return CycleBlochConfig(**kwargs)


def test_epg_tse_with_ideal_single_echo_reproduces_spin_echo(mock_env):
    from spectramr.infrastructure.physics.biophysics.bloch import BlochSimulator

    cfg = {**_EPG, "echo_spacing_ms": 14.0, "echo_train_length": 1, "refocus_deg": 180.0}
    strat = _build(mock_env, _cb(**cfg), None)
    pd, t1, t2 = _tissue()
    expected = BlochSimulator.spin_echo(m0=pd, t1=t1, t2=t2, te=14.0, tr=500.0)
    torch.testing.assert_close(strat._render_ulf(pd, t1, t2, None), expected, rtol=1e-3, atol=1e-5)
    assert strat.cycle_bloch_provenance["echo_train_length"] == 1


def test_epg_tse_changes_the_cycle_loss_on_the_train_path(mock_env):
    tissue = _tissue()
    fixed = _generator_cycle_loss(_build(mock_env, None, None), mock_env, tissue)
    epg = _generator_cycle_loss(_build(mock_env, _cb(**_EPG), None), mock_env, tissue)
    assert not torch.isclose(fixed, epg, rtol=1e-3)


def test_epg_tse_rejects_a_te_outside_the_echo_train(mock_env):
    with pytest.raises(ValueError, match="outside"):
        _build(mock_env, _cb(**{**_EPG, "ulf_te_ms": 70.0}), None)


def test_estimated_b1_scales_the_render_and_adds_its_tv_penalty(mock_env):
    strat = _build(mock_env, _cb(**_EPG, b1_plus="estimated", b1_tv_weight=0.5), None)
    assert strat.cycle_bloch_provenance["b1_plus"] == "estimated"
    pd, t1, t2 = _tissue()
    nominal = strat._render_ulf(pd, t1, t2, None)
    assert not torch.allclose(strat._render_ulf(pd, t1, t2, torch.full_like(pd, 0.6)), nominal)

    real_ulf = torch.rand(2, 1, 8, 8)
    mock_env.generator.return_value = torch.rand(2, 1, 8, 8)
    mock_env.discriminator.return_value = torch.zeros(2, 1, 2, 2)
    est = strat._get_estimator(mock_env.generator.return_value)
    assert est.net[-1].out_channels == 4
    b1 = torch.linspace(0.5, 1.5, 64).reshape(1, 1, 8, 8).expand(2, 1, 8, 8)
    maps = torch.cat([*_tissue(), b1], dim=1)
    with patch.object(strat.estimator, "forward", return_value=maps):
        steps = strat.train_step(
            batch=None, epoch=0, iteration=0, input_batch=real_ulf, target_batch=real_ulf
        )
        steps[0]["closure"]()
    m = strat.get_last_metrics()
    assert m["loss_b1_tv"] > 0
    # The critic outputs 0, so the LSGAN generator term is (0 - 1)^2 = 1.
    expected = strat.lambda_adv + strat.lambda_cycle * m["loss_cycle"] + 0.5 * m["loss_b1_tv"]
    torch.testing.assert_close(m["loss_g"], expected)


def test_estimator_b1_channel_is_nominal_at_zero_logit():
    est = ParameterEstimator(in_channels=1, with_b1=True)
    torch.nn.init.zeros_(est.net[-1].weight)
    torch.nn.init.zeros_(est.net[-1].bias)
    out = est(torch.rand(1, 1, 4, 4))
    assert out.shape[1] == 4
    torch.testing.assert_close(out[:, 3], torch.ones_like(out[:, 3]))


@pytest.mark.parametrize("backend", ["bloch_fixed", "ulf_forward", "epg_tse_b1"])
def test_generator_closure_backpropagates_into_the_tissue_maps(mock_env, backend):
    cfg = {
        "bloch_fixed": None,
        "ulf_forward": _ulf_forward_cfg(),
        "epg_tse_b1": _cb(**_EPG, b1_plus="estimated", b1_tv_weight=0.5),
    }[backend]
    strat = _build(mock_env, cfg, 0.064)
    real_ulf = torch.rand(2, 1, 8, 8)
    mock_env.generator.return_value = torch.rand(2, 1, 8, 8)
    mock_env.discriminator.return_value = torch.zeros(2, 1, 2, 2)
    strat._get_estimator(mock_env.generator.return_value)
    channels = list(_tissue())
    if backend == "epg_tse_b1":
        channels.append(torch.linspace(0.5, 1.5, 64).reshape(1, 1, 8, 8).expand(2, 1, 8, 8))
    maps = torch.cat(channels, dim=1).requires_grad_()
    with patch.object(strat.estimator, "forward", return_value=maps):
        steps = strat.train_step(
            batch=None, epoch=0, iteration=0, input_batch=real_ulf, target_batch=real_ulf
        )
        steps[0]["closure"]().backward()
    assert torch.isfinite(maps.grad).all() and maps.grad.abs().sum() > 0
    if backend == "epg_tse_b1":
        assert maps.grad[:, 3].abs().sum() > 0


def _table_strategy(mock_env):
    from spectramr.config.schemas.training.cycle_bloch import CycleBlochConfig

    seqs = {
        "T1w": {"tr_ms": 880.0, "te_ms": 5.03, "ti_ms": 340.0, "flip_deg": 90.0},
        "T2w": {"tr_ms": 2000.0, "te_ms": 194.8, "flip_deg": 90.0},
    }
    mock_env.config.data.multi_contrast.enabled = True
    mock_env.config.data.multi_contrast.contrast_map = {"T1w": 0, "T2w": 1}
    cfg = CycleBlochConfig(backward_model="ulf_forward", tissue_field_tesla=3.0, sequences=seqs)
    return _build(mock_env, cfg, 0.064)


def _cycle_loss_for(strat, mock_env, tissue, contrast):
    """The production generator closure on a loop-shaped TrainingBatch carrying contrast_idx."""
    from spectramr.data.batch_types import BatchAdapter

    real_ulf = torch.rand(2, 1, 8, 8, generator=torch.Generator().manual_seed(1))
    batch = BatchAdapter.from_dict(
        {"input": real_ulf, "target": real_ulf, "contrast_idx": torch.tensor(contrast)}
    )
    mock_env.generator.return_value = torch.rand(2, 1, 8, 8)
    mock_env.discriminator.return_value = torch.zeros(2, 1, 2, 2)
    strat._get_estimator(mock_env.generator.return_value)
    with patch.object(strat.estimator, "forward", return_value=torch.cat(tissue, dim=1)):
        steps = strat.train_step(
            batch=batch, epoch=0, iteration=0, input_batch=real_ulf, target_batch=real_ulf
        )
        steps[0]["closure"]()
    return strat.get_last_metrics()["loss_cycle"]


def test_the_batch_contrast_selects_the_sequence_on_the_train_path(mock_env):
    strat = _table_strategy(mock_env)
    tissue = _tissue()
    t1w = _cycle_loss_for(strat, mock_env, tissue, [0, 0])
    t2w = _cycle_loss_for(strat, mock_env, tissue, [1, 1])
    assert torch.isfinite(t1w) and torch.isfinite(t2w)
    assert not torch.isclose(t1w, t2w, rtol=1e-3)
    assert set(strat.cycle_bloch_provenance["sequences"]) == {"T1w", "T2w"}
    assert CycleBlochStrategy.consumes_contrast_id(mock_env.config)


def test_a_table_without_a_contrast_map_raises(mock_env):
    from spectramr.config.schemas.training.cycle_bloch import CycleBlochConfig

    mock_env.config.data.multi_contrast.enabled = False
    cfg = CycleBlochConfig(
        backward_model="ulf_forward",
        tissue_field_tesla=3.0,
        sequences={"T2w": {"tr_ms": 2000.0, "te_ms": 194.8, "flip_deg": 90.0}},
    )
    with pytest.raises(ValueError, match="contrast_map"):
        _build(mock_env, cfg, 0.064)


def test_estimator_is_owned_and_a_resume_rebuilds_it_before_restoring(strategy, mock_env):
    """#2629: the estimator is registered for DDP under its own key, and a fresh strategy
    rebuilds it from the saved weights (``estimator`` is ``None`` until the first step,
    so a resume used to raise on the unplaced ``modules.estimator``)."""
    mock_env.opt_g = None
    strategy._get_estimator(torch.rand(2, 3, 8, 8))
    assert strategy.owned_modules()["estimator"] is strategy.estimator
    state = strategy.strategy_state_dict()
    assert sorted(state["modules"]) == ["estimator"] and state["params"] == {}

    with patch("spectramr.infrastructure.di.di_container.resolve_service") as mock_resolve:
        mock_resolve.return_value = MagicMock()
        fresh = CycleBlochStrategy(env=mock_env)
    assert fresh.estimator is None
    fresh.load_strategy_state_dict(state)
    assert fresh.owned_modules()["estimator"] is fresh.estimator
    pairs = zip(fresh.estimator.parameters(), strategy.estimator.parameters(), strict=True)
    for got, want in pairs:
        assert torch.equal(got, want)
