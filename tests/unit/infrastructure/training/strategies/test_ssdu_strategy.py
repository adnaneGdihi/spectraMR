"""Tests for the SSDU reference-free reconstruction strategy."""

import torch

from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from spectramr.infrastructure.training.strategies.ssdu_strategy import (
    SSDUReconstructionStrategy,
    inject_noisier_kspace,
    split_acquired_mask,
)
from spectramr.models.losses.ssdu_loss import SSDULoss


def test_split_is_disjoint_and_covers_acquired():
    # acquired mask: 1 where k-space was sampled
    acq = torch.zeros(1, 1, 8, 8)
    acq[..., ::2] = 1.0  # half the columns acquired
    lam, theta = split_acquired_mask(
        acq, theta_fraction=0.4, generator=torch.Generator().manual_seed(0)
    )
    # Lambda and Theta are disjoint
    assert torch.all((lam * theta) == 0)
    # Lambda union Theta == acquired (no leakage outside acquired support)
    assert torch.allclose(lam + theta, acq)
    # Theta holds ~40% of acquired points
    n_acq = int(acq.sum().item())
    assert abs(int(theta.sum().item()) - round(0.4 * n_acq)) <= 1


def test_split_rejects_bad_fraction():
    acq = torch.ones(1, 1, 4, 4)
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            split_acquired_mask(acq, theta_fraction=bad)
        except ValueError:
            continue
        raise AssertionError(f"theta_fraction={bad} should have raised")


def test_split_handles_empty_mask():
    acq = torch.zeros(1, 1, 4, 4)
    lam, theta = split_acquired_mask(acq, theta_fraction=0.4)
    assert int(theta.sum().item()) == 0
    assert torch.allclose(lam, acq)


def test_ssdu_loss_consumes_strategy_context():
    """The strategy's context dict must satisfy SSDULoss's contract."""
    b, c, h, w = 1, 1, 8, 8
    pred_image = torch.randn(b, c, h, w, requires_grad=True)
    target_kspace = torch.randn(b, c, h, w, dtype=torch.complex64)
    theta = torch.zeros(b, 1, h, w)
    theta[..., ::3] = 1.0
    loss = SSDULoss()(
        prediction=pred_image,
        target=torch.zeros_like(pred_image),
        context={"target_kspace": target_kspace, "theta_mask": theta},
    )
    assert loss.requires_grad and loss.ndim == 0


def test_strategy_class_extends_reconstruction():
    assert issubclass(SSDUReconstructionStrategy, ReconstructionTrainingStrategy)


# --- Robust SSDU (Noisier2Noise) -------------------------------------------
def test_noisier_injection_zero_std_is_identity():
    """Corner case: σ=0 → no extra noise → vanilla SSDU."""
    y = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    acq = torch.zeros(1, 1, 8, 8)
    acq[..., ::2] = 1.0
    z = inject_noisier_kspace(y, acq, noise_std=0.0)
    assert torch.equal(z, y)


def test_noisier_injection_adds_complex_noise_on_acquired_only():
    y = torch.zeros(1, 1, 16, 16, dtype=torch.complex64)
    acq = torch.zeros(1, 1, 16, 16)
    acq[..., ::2] = 1.0
    z = inject_noisier_kspace(y, acq, noise_std=0.5, generator=torch.Generator().manual_seed(0))
    diff = z - y
    assert z.is_complex()
    # noise only on acquired support; unacquired entries stay exactly zero
    assert torch.all(diff[acq == 0] == 0)
    assert diff.abs().sum() > 0
    # injected magnitude std is in the right ballpark of noise_std
    acq_diff = diff[acq > 0]
    assert 0.2 < acq_diff.abs().std().item() < 0.9


def test_robust_ssdu_strategy_wires_noisier2noise():
    """The strategy must consume the noisier2noise knob + the robust_ssdu mode."""
    import inspect

    src = inspect.getsource(SSDUReconstructionStrategy)
    assert "inject_noisier_kspace" in src
    assert "noisier2noise" in src.lower()
    assert "robust_ssdu" in src


def test_it_declares_its_own_loss_ownership() -> None:
    """Issue #1918: it iterates env.losses but its loop body skips every name without 'ssdu'.

    Read off ``__dict__``, never the inherited value: this class sits under
    ``ReconstructionTrainingStrategy``, whose ``folds_image_losses = True`` is truthful for ITSELF and
    becomes a lie the moment a subclass replaces ``_compute_losses_impl``. An
    inherited True makes the audit's ``image_losses_reach_the_objective`` witness
    PASS every declared ``losses.image_losses`` entry on this strategy's arms
    while the training step discards them.
    """
    assert SSDUReconstructionStrategy.__dict__["folds_image_losses"] is False
    assert SSDUReconstructionStrategy.__dict__["inline_losses"] == frozenset()



def test_it_declares_the_ssdu_loss_columns_the_step_stamps() -> None:
    strategy = object.__new__(SSDUReconstructionStrategy)
    strategy._ssdu_weights_cache = {"ssdu_normalized": 1.0, "l1": 0.5}
    assert strategy.declared_metric_keys() == {"loss_ssdu_normalized"}

# --- #2327: the blockers, through the arm's own YAML and loader chain -------

import inspect  # noqa: E402

import pytest  # noqa: E402


def test_injection_pairs_interleaved_channels_and_keeps_the_layout() -> None:
    """Each of 8 real channels used to receive its own imaginary noise, which made
    a complex (B, 8, H, W) tensor the reshape then read as 8 slices."""
    y = torch.zeros(2, 8, 16, 16)
    acq = torch.ones(2, 1, 16, 16)
    z = inject_noisier_kspace(y, acq, 0.5, generator=torch.Generator().manual_seed(0))
    assert z.shape == y.shape and not torch.is_complex(z)
    # The planted shape: the old per-channel draw returned complex data.
    assert not torch.is_complex(z)


def test_injection_accepts_a_device_sigma() -> None:
    """The self-calibrated sigma is a tensor and must never be pulled to the host."""
    y = torch.zeros(1, 2, 8, 8, dtype=torch.complex64)
    z = inject_noisier_kspace(y, torch.ones(1, 1, 8, 8), torch.tensor(0.3))
    assert z.abs().sum() > 0


def test_the_split_is_exact_per_sample_without_a_host_sync() -> None:
    acq = torch.zeros(3, 1, 8, 8)
    acq[0, ..., ::2] = 1.0
    acq[1, ..., ::4] = 1.0
    _, theta = split_acquired_mask(acq, 0.4, generator=torch.Generator().manual_seed(1))
    for b in range(3):
        n = int(acq[b].sum())
        assert int(theta[b].sum()) == round(0.4 * n)
    assert ".item()" not in inspect.getsource(split_acquired_mask)


@pytest.fixture(scope="module", params=["robust_ssdu_m4raw_r4", "robust_ssdu_selfcal_m4raw_r4"])
def _ssdu_run(request):
    from tests.fixtures.self_supervised_batches import build_strategy, load_arm, real_batches

    config = load_arm(request.param)
    train, val = real_batches(config, coils=4)
    return request.param, build_strategy(config), train, val


def test_both_robust_arms_take_a_training_step(_ssdu_run) -> None:
    _, strategy, train, _ = _ssdu_run
    losses = strategy._compute_losses_impl(train.input, train.target, 0, batch=train)
    losses["g_total_loss"].backward()
    assert torch.isfinite(losses["g_total_loss"]) and float(losses["loss_ssdu"].detach()) > 0
    assert any(p.grad is not None for p in strategy.generator_model.parameters())


def test_both_robust_arms_take_a_validation_step(_ssdu_run) -> None:
    _, strategy, _, val = _ssdu_run
    metrics = strategy.validation_step(val.input, val.target, batch_data=val)
    # The zero-filled baseline on the columns the arm is read on, not only RSS.
    zf_context = ("val_zf_sense_psnr", "val_zf_reference_noise_corrected_mse")
    for key in ("val_psnr", "val_ssim", "val_zf_psnr", *zf_context):
        assert key in metrics and metrics[key] == metrics[key], f"{key} missing or NaN"
    # #2593: the second-repetition forward runs on input_alt.
    assert metrics["val_rep_variance"] > 0


def test_val_loss_is_the_ssdu_objective_on_a_fixed_split(_ssdu_run) -> None:
    """val_loss was an L1 of |image| against k-space channel 0, 30x the training loss."""
    _, strategy, train, val = _ssdu_run
    with torch.no_grad():
        train_loss = float(strategy._compute_losses_impl(train.input, train.target, 0, batch=train)["loss"])
        first = strategy.validation_step(val.input, val.target, batch_data=val)["val_loss"]
        second = strategy.validation_step(val.input, val.target, batch_data=val)["val_loss"]
    assert 0.1 < float(first) / train_loss < 10.0
    assert float(first) == float(second), "the validation split must not move between passes"


def test_both_robust_arms_render_on_the_capture_context(_ssdu_run) -> None:
    """#2431: the loop's image capture carried no mask or measurement, so
    ``_measurement`` raised and neither arm ever saved a validation image."""
    from spectramr.pipelines.train import visual_capture_inputs

    _, strategy, _, val = _ssdu_run
    x, _, context = visual_capture_inputs(strategy, val.input, val.target, val)
    assert x is val.input, "the capture must render on the tensors validation_step graded"
    with torch.no_grad():
        image = strategy._validation_forward(x, context)
    assert image.shape[0] == val.input.shape[0] and torch.isfinite(image).all()


def test_the_validation_image_is_zero_where_no_coil_measures(_ssdu_run) -> None:
    from spectramr.infrastructure.physics.coil_sensitivity import coil_support

    _, strategy, _, val = _ssdu_run
    context = strategy.validation_batch_context(val.input, val.target, val)
    maps = context["coil_sensitivities"].clone()
    maps[..., :8] = 0  # planted: a band no coil covers
    context = {**context, "coil_sensitivities": maps}
    with torch.no_grad():
        image = strategy._validation_forward(val.input, context)
    outside = ~coil_support(maps).expand_as(image)
    assert outside.any() and torch.all(image[outside] == 0)


def test_validation_step_builds_its_context_through_the_shared_hook(
    _ssdu_run, monkeypatch
) -> None:
    """The metric pass and the image capture must not drift into two builders."""
    _, strategy, _, val = _ssdu_run
    seen: list = []
    hook = strategy.validation_batch_context

    def spy(*args):
        seen.append(args[2])
        return hook(*args)

    monkeypatch.setattr(strategy, "validation_batch_context", spy)
    strategy.validation_step(val.input, val.target, batch_data=val)
    assert seen == [val]


def test_the_network_is_handed_an_image_and_dc_gets_the_coils(_ssdu_run) -> None:
    """Blocker 4: the first convolution used to see Lambda-masked k-space."""
    _, strategy, train, _ = _ssdu_run
    seen = {}

    def hook(_module, args, kwargs):
        seen["x"], seen["kwargs"] = args[0], kwargs

    handle = strategy.env.generator.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        strategy._compute_losses_impl(train.input, train.target, 0, batch=train)
    finally:
        handle.remove()
    assert seen["x"].shape[1] == 2, "the network must receive the 2-channel A^H y"
    assert torch.is_complex(seen["kwargs"]["kspace_measured"])
    assert seen["kwargs"]["kspace_measured"].shape[1] == 4
    assert seen["kwargs"]["sensitivity_maps"].shape[1] == 4


def test_the_correction_fires_at_validation(_ssdu_run, monkeypatch) -> None:
    """Eq. 36: validation returns the corrected image, not f(A^H y)."""
    from spectramr.infrastructure.training.strategies import ssdu_strategy

    _, strategy, _, val = _ssdu_run
    calls = []
    real = ssdu_strategy.noisier2noise_correct

    def spy(*args, **kwargs):
        calls.append(args[3])
        return real(*args, **kwargs)

    monkeypatch.setattr(ssdu_strategy, "noisier2noise_correct", spy)
    strategy.validation_step(val.input, val.target, batch_data=val)
    assert calls and all(a == strategy.alpha for a in calls)


def test_robust_ssdu_scores_every_acquired_line(_ssdu_run, monkeypatch) -> None:
    """Eq. 26: the loss mask is Omega, not Theta, and it carries Eq. 35's weights."""
    _, strategy, train, _ = _ssdu_run
    captured = {}
    loss_fn = strategy.env.losses["ssdu"]
    original = loss_fn.forward

    def spy(prediction, target, context=None, **kwargs):
        captured.update(context)
        return original(prediction, target, context=context, **kwargs)

    monkeypatch.setattr(loss_fn, "forward", spy)
    strategy._compute_losses_impl(train.input, train.target, 0, batch=train)
    assert torch.equal(captured["theta_mask"], train.mask)
    assert captured["loss_weights"] is not None
    assert captured["coil_sensitivities"] is not None


def test_the_self_calibrated_sigma_is_fitted_on_the_acs_band(_ssdu_run, monkeypatch) -> None:
    """The measurement is masked: the fit must run under the ACS policy, and the
    declared-sigma control must not fit at all."""
    from spectramr.infrastructure.training.strategies import ssdu_strategy

    arm, strategy, train, _ = _ssdu_run
    calls = []
    real = ssdu_strategy.sigma_from_kspace_batch

    def spy(*args, **kwargs):
        calls.append(kwargs.get("support_policy"))
        return real(*args, **kwargs)

    monkeypatch.setattr(ssdu_strategy, "sigma_from_kspace_batch", spy)
    strategy._compute_losses_impl(train.input, train.target, 0, batch=train)
    if arm.endswith("selfcal_m4raw_r4"):
        assert calls and all(policy == "acs" for policy in calls), calls
    else:
        assert not calls, "the declared-sigma control fitted sigma"


def test_the_self_calibrated_sigma_reads_the_declared_acquisition(monkeypatch) -> None:
    """A fit that infers the support from zeros in y tripped on an acquired sample that
    happened to be zero (robust_ssdu_selfcal, iteration 905); the strategy holds the mask."""
    from types import SimpleNamespace

    from spectramr.infrastructure.training.strategies import ssdu_strategy

    strategy = _ssdu_on_rank(monkeypatch, None)
    strategy.noise_std_source = "self_calibrated"
    seen: dict = {}

    def fake_fit(y, maps, **kwargs):
        seen.update(kwargs)
        return torch.ones(y.shape[0])

    monkeypatch.setattr(ssdu_strategy, "sigma_from_kspace_batch", fake_fit)
    y = torch.zeros(2, 4, 8, 8, dtype=torch.complex64)
    acquired = torch.ones(2, 1, 1, 8)
    strategy._sigma(y, SimpleNamespace(), acquired)
    assert seen["support_policy"] == "acs"
    assert seen["acquired"] is acquired


# ---------------------------------------------------------------------------
# #2762: the split-and-noise stream is per rank and survives a resume
# ---------------------------------------------------------------------------


def _ssdu_on_rank(monkeypatch, rank: int | None) -> SSDUReconstructionStrategy:
    """The real ``__init__`` over the arm's config, without building the model."""
    from types import SimpleNamespace

    from spectramr.infrastructure.training.strategies import rng_streams
    from tests.fixtures.self_supervised_batches import load_arm

    config = load_arm("ssdu_m4raw_r4")
    monkeypatch.setattr(rng_streams, "distributed_rank", lambda: rank)

    def bare_init(self, env=None, device=None, **kwargs):
        self.config = config
        self.env = SimpleNamespace(generator=None, discriminator=None, losses={})

    monkeypatch.setattr(ReconstructionTrainingStrategy, "__init__", bare_init)
    return SSDUReconstructionStrategy()


def _split(strategy: SSDUReconstructionStrategy) -> torch.Tensor:
    acquired = torch.zeros(1, 1, 16, 16)
    acquired[..., ::2] = 1.0
    return split_acquired_mask(acquired, theta_fraction=0.4, generator=strategy._rng)[1]


def test_two_ranks_draw_different_splits(monkeypatch) -> None:
    rank0, rank1 = _ssdu_on_rank(monkeypatch, 0), _ssdu_on_rank(monkeypatch, 1)
    assert not torch.equal(_split(rank0), _split(rank1))


def test_a_resume_continues_the_split_stream(monkeypatch) -> None:
    saved = _ssdu_on_rank(monkeypatch, None)
    _split(saved)
    state = saved.strategy_state_dict()
    expected = _split(saved)

    resumed = _ssdu_on_rank(monkeypatch, None)
    resumed.load_strategy_state_dict(state)
    assert torch.equal(_split(resumed), expected), "the resumed run replayed the first split"


# ---------------------------------------------------------------------------
# #2509: Rep-BV grades F S x_hat; #2508: the snapshot names the real model input.
# Stub instances only -- no generator is built.
# ---------------------------------------------------------------------------


def _complex(*shape: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=gen), torch.randn(*shape, generator=gen))


def test_the_context_metrics_grade_the_ssdu_image_as_coil_kspace() -> None:
    from spectramr.core.metrics.coil_manifold import as_interleaved
    from spectramr.infrastructure.physics.fft_ops import sense_forward

    h, w = 12, 20  # non-square
    maps = _complex(1, 4, h, w, seed=0)
    image = _complex(1, 1, h, w, seed=1)
    st = object.__new__(SSDUReconstructionStrategy)
    st.n_coils = 4
    context = {
        "mask": torch.ones(1, 1, h, w),
        "measured_kspace": as_interleaved(_complex(1, 4, h, w, seed=2)),
        "coil_sensitivities": maps,
    }
    graded, domain = st._context_metric_prediction(as_interleaved(image), context)
    assert domain == "kspace" and graded.shape == (1, 8, h, w)
    torch.testing.assert_close(graded, as_interleaved(sense_forward(image, maps, None)))


def test_the_step_declares_the_adjoint_image_it_feeds_the_network() -> None:
    """#2508: ``input_prepared`` is the coil k-space; the network gets ``A^H y~``."""
    from spectramr.core.metrics.coil_manifold import as_interleaved

    assert SSDUReconstructionStrategy.snapshot_prepared_is_model_input is False
    assert SSDUReconstructionStrategy.snapshot_model_input_tag == "ssdu_model_input"
    h, w = 12, 20
    acquired = torch.zeros(1, 1, h, w)
    acquired[..., ::2] = 1.0
    y = as_interleaved(_complex(1, 1, h, w, seed=3)) * acquired
    st = object.__new__(SSDUReconstructionStrategy)
    st.theta_fraction, st.num_masks, st.noisier2noise, st.n_coils = 0.4, 1, False, 1
    st._rng = torch.Generator().manual_seed(0)
    st.env = type("Env", (), {"losses": {"ssdu": lambda p, t, context: p.abs().mean()}})()
    st._ssdu_weights_cache = {"ssdu": 1.0}
    st._prepare_batch_context = lambda *a, **k: {"mask": acquired, "measured_kspace": y}
    fed = torch.randn(1, 2, h, w)
    st._network = lambda ctx, kspace, mask, maps, **_: (fed, {}, ctx)
    st._generate_predictions = lambda lr, fk, bc: (0.5 * lr, {})
    declared: list = []
    st._declare_model_input = lambda tensors, **kw: declared.append((tensors, kw))
    st._compute_losses_impl(y, y, 0)
    assert len(declared) == 1
    tensors, kw = declared[0]
    assert tensors["model_input_image"] is fed and kw["in_kspace_keys"] == set()
    # The wrapper's contract, not a stub: the declaration must name its model input.
    from spectramr.infrastructure.training.model_input_contract import require_model_input_key

    tag = SSDUReconstructionStrategy.snapshot_model_input_tag
    key = require_model_input_key(
        strategy_name="SSDUReconstructionStrategy",
        tag=tag,
        tensors=tensors,
        model_input_key=kw["extra"]["model_input_key"],
    )
    assert key == "model_input_image"
    with pytest.raises(ValueError, match="without naming"):  # the sabine crash, planted
        require_model_input_key(
            strategy_name="SSDUReconstructionStrategy", tag=tag, tensors=tensors, model_input_key=None
        )



def _weighted_step(weight: float) -> float:
    """One SSDU step with a constant loss, at the declared ``weight``."""
    from spectramr.core.metrics.coil_manifold import as_interleaved

    h, w = 12, 20
    acquired = torch.zeros(1, 1, h, w)
    acquired[..., ::2] = 1.0
    y = as_interleaved(_complex(1, 1, h, w, seed=3)) * acquired
    st = object.__new__(SSDUReconstructionStrategy)
    st.theta_fraction, st.num_masks, st.noisier2noise, st.n_coils = 0.4, 1, False, 1
    st._rng = torch.Generator().manual_seed(0)
    st.env = type("Env", (), {"losses": {"ssdu": lambda p, t, context: torch.tensor(2.0)}})()
    st._ssdu_weights_cache = {"ssdu": weight}
    st._prepare_batch_context = lambda *a, **k: {"mask": acquired, "measured_kspace": y}
    st._network = lambda ctx, kspace, mask, maps, **_: (torch.randn(1, 2, h, w), {}, ctx)
    st._generate_predictions = lambda lr, fk, bc: (0.5 * lr, {})
    st._declare_model_input = lambda tensors, **kw: None
    return float(st._compute_losses_impl(y, y, 0)["g_total_loss"])


def test_the_ssdu_loss_carries_its_declared_weight() -> None:
    """Non-negotiable 8: a YAML weight other than 1 used to be ignored."""
    assert _weighted_step(1.0) == pytest.approx(2.0)
    assert _weighted_step(0.5) == pytest.approx(1.0)  # planted: the ignored weight read 2.0


def test_the_vanilla_arm_scores_the_structural_zeros_with_the_normalised_loss(monkeypatch) -> None:
    """ssdu_m4raw_r4: Theta gains the columns off the acquired band, where y = 0."""
    import spectramr.infrastructure.training.strategies.ssdu_strategy as ss
    from tests.fixtures.self_supervised_batches import build_strategy, load_arm, real_batches

    config = load_arm("ssdu_m4raw_r4")
    assert config.training.ssdu.score_structural_zeros
    train, _ = real_batches(config, coils=4, pe_zero_fill=4)
    strategy = build_strategy(config)
    seen = []
    real = ss.batch_support
    monkeypatch.setattr(ss, "batch_support", lambda b, like: seen.append(real(b, like)) or seen[-1])
    losses = strategy._compute_losses_impl(train.input, train.target, 0, batch=train)
    losses["g_total_loss"].backward()
    assert "loss_ssdu_normalized" in losses and torch.isfinite(losses["g_total_loss"])
    assert seen and (~seen[0]).any(), "no structural zeros reached Theta"


def test_num_masks_above_one_raises_instead_of_scoring_input_lines(monkeypatch) -> None:
    """Planted: a second split's Theta overlaps the single forward's Lambda."""
    from tests.fixtures.self_supervised_batches import load_arm

    config = load_arm("ssdu_m4raw_r4")
    object.__setattr__(config.training.ssdu, "num_masks", 4)
    monkeypatch.setattr(
        ReconstructionTrainingStrategy,
        "__init__",
        lambda self, env=None, device=None, **kw: setattr(self, "config", config),
    )
    with pytest.raises(ValueError, match="num_masks=4"):
        SSDUReconstructionStrategy()
