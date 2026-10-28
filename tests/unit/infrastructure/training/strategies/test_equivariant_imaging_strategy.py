r"""Tests for the Equivariant Imaging (EI) reconstruction strategy (Phase A step 3).

EI (Chen et al., ICCV 2021) supervises a reconstruction network *without ground
truth* by enforcing :math:`f(A T_g f(A^* y)) = T_g f(A^* y)`. The registered
``EquivariantSSLReconLoss`` was an **inert facade** — nothing emitted
``context["transformed_recon"]`` (the reference branch :math:`T_g f(A^* y)`), so
the term could never fire. These tests pin the *mechanism-fires* contract: the
pure core ``equivariant_imaging_branches`` produces both branches, and feeding
them to the registered loss yields a non-zero, grad-carrying objective that
actually pushes gradient into the reconstruction network.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.group_actions import DihedralGroup
from spectramr.infrastructure.training.strategies.equivariant_imaging_strategy import (
    EquivariantImagingStrategy,
    equivariant_imaging_branches,
)
from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from spectramr.models.losses.equivariant_recon_loss import EquivariantSSLReconLoss


def _partial_mask(h: int, w: int):
    mask = torch.zeros(1, 1, h, w)
    mask[..., :, ::4] = 1.0
    mask[..., :, w // 2 - 2 : w // 2 + 2] = 1.0
    return mask


def _ei_closures(h: int = 16, w: int = 16):
    """Build (reconstruct, forward_op, measured_kspace, net) for a tiny EI loop."""
    torch.manual_seed(0)
    mask_c = _partial_mask(h, w).to(torch.complex64)
    net = nn.Conv2d(1, 1, 3, padding=1)

    def forward_op(x):
        x_c = x if x.is_complex() else torch.complex(x, torch.zeros_like(x))
        return mask_c * fft2c(x_c)

    def reconstruct(kspace):
        zero_filled = ifft2c(mask_c * kspace).real  # A^* y (real)
        return net(zero_filled)

    phantom = torch.randn(1, 1, h, w)
    y = forward_op(phantom)
    return reconstruct, forward_op, y, net


# ---------------------------------------------------------------------------
# Pure core — branches + mechanism-fires
# ---------------------------------------------------------------------------
def test_branches_shapes_and_grad():
    reconstruct, forward_op, y, _net = _ei_closures()
    group = DihedralGroup()
    x_hat, pred, transformed = equivariant_imaging_branches(
        reconstruct, forward_op, group, y, g_index=2
    )
    assert x_hat.shape == pred.shape == transformed.shape
    assert pred.requires_grad and transformed.requires_grad


def test_identity_element_leaves_reference_equal_to_recon():
    """g == 0 (identity) => transformed_recon == x_hat exactly (clean invariant)."""
    reconstruct, forward_op, y, _net = _ei_closures()
    group = DihedralGroup()
    x_hat, _pred, transformed = equivariant_imaging_branches(
        reconstruct, forward_op, group, y, g_index=0
    )
    assert torch.equal(transformed, x_hat)


def test_mechanism_fires_equivariance_loss_trains_the_network():
    """The whole point of the un-trap: the registered EI loss must produce a
    non-zero, grad-carrying objective whose gradient reaches the recon net."""
    reconstruct, forward_op, y, net = _ei_closures()
    group = DihedralGroup()
    _x_hat, pred, transformed = equivariant_imaging_branches(
        reconstruct, forward_op, group, y, g_index=3
    )
    loss = EquivariantSSLReconLoss()(
        prediction=pred,
        target=torch.zeros_like(pred),
        context={"transformed_recon": transformed},
    )
    assert loss.ndim == 0 and loss.requires_grad
    assert float(loss.detach()) > 0.0, "equivariance loss is zero — mechanism did not fire"
    net.zero_grad(set_to_none=True)
    loss.backward()
    grad_sq = sum(float(p.grad.pow(2).sum()) for p in net.parameters() if p.grad is not None)
    assert grad_sq > 0.0, "EI loss produced no gradient on the recon network"


def test_branches_call_reconstruct_twice():
    """EI requires a second forward on the re-measured transformed recon."""
    reconstruct, forward_op, y, _net = _ei_closures()
    calls = {"n": 0}

    def counting_reconstruct(kspace):
        calls["n"] += 1
        return reconstruct(kspace)

    equivariant_imaging_branches(counting_reconstruct, forward_op, DihedralGroup(), y, g_index=1)
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Strategy class wiring
# ---------------------------------------------------------------------------
def test_strategy_extends_reconstruction():
    assert issubclass(EquivariantImagingStrategy, ReconstructionTrainingStrategy)


def test_strategy_routes_equivariance_through_registered_loss():
    """The strategy must consume the registered EquivariantSSLReconLoss (the
    facade target), not a private re-implementation — asserted from source."""
    import inspect

    src = inspect.getsource(EquivariantImagingStrategy)
    assert "transformed_recon" in src
    assert "EquivariantSSLReconLoss" in src
    # measurement-consistency anchor is present (prevents the trivial f≡const).
    assert "consistency" in src.lower()


def test_it_declares_its_own_loss_ownership() -> None:
    """Issue #1918: the objective is self-supervised: no ground truth is used at all.

    Read off ``__dict__``, never the inherited value: this class sits under
    ``ReconstructionTrainingStrategy``, whose ``folds_image_losses = True`` is truthful for ITSELF and
    becomes a lie the moment a subclass replaces ``_compute_losses_impl``. An
    inherited True makes the audit's ``image_losses_reach_the_objective`` witness
    PASS every declared ``losses.image_losses`` entry on this strategy's arms
    while the training step discards them.
    """
    assert EquivariantImagingStrategy.__dict__["folds_image_losses"] is False
    assert EquivariantImagingStrategy.__dict__["inline_losses"] == frozenset()


def test_only_the_multicoil_operator_arm_builds_its_own_model_input() -> None:
    """#2240: with ``multicoil_operator`` the strategy SENSE-combines the coils, so the
    loaded batch width is not what the network sees and the base guard must defer."""
    st = object.__new__(EquivariantImagingStrategy)
    st.multicoil_operator = True
    assert st.builds_own_model_input() is True
    st.multicoil_operator = False
    assert st.builds_own_model_input() is False


# ---------------------------------------------------------------------------
# End to end, through the arm's own YAML, loader chain, collate and strategy.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _multicoil_run():
    from tests.fixtures.self_supervised_batches import build_strategy, load_arm, real_batches

    config = load_arm("ei_multicoil_m4raw_r4")
    train, val = real_batches(config, coils=4)
    return build_strategy(config), train, val


def test_the_multicoil_arm_takes_a_training_step(_multicoil_run) -> None:
    """#2326: the Re/Im output used to reach ``sense_forward`` as a 2-coil image."""
    strategy, train, _ = _multicoil_run
    losses = strategy._compute_losses_impl(train.input, train.target, 0, batch=train)
    losses["g_total_loss"].backward()
    grads = [p.grad for p in strategy.generator_model.parameters() if p.grad is not None]
    assert grads and torch.isfinite(losses["g_total_loss"])
    assert float(losses["loss_equivariance"]) > 0.0


def test_every_stamped_term_has_a_csv_column(_multicoil_run) -> None:
    """The loop's header takes these from declared_metric_keys, or drops them on write."""
    strategy, train, _ = _multicoil_run
    losses = strategy._compute_losses_impl(train.input, train.target, 0, batch=train)
    stamped = set(losses) - {"loss", "g_total_loss"}
    assert stamped <= strategy.declared_metric_keys()


def test_the_multicoil_arm_takes_a_validation_step(_multicoil_run) -> None:
    """#2326: validation got a bare context, found no coil maps and raised every batch."""
    strategy, _, val = _multicoil_run
    metrics = strategy.validation_step(val.input, val.target, batch_data=val)
    zf_context = ("val_zf_sense_psnr", "val_zf_reference_noise_corrected_mse")
    for key in ("val_psnr", "val_ssim", "val_zf_psnr", *zf_context):
        assert key in metrics and metrics[key] == metrics[key], f"{key} missing or NaN"
    # #2593: the second-repetition forward runs on input_alt, so the variance is
    # the estimator's own and not the 0 a context-reading forward used to give.
    assert metrics["val_rep_variance"] > 0
    assert metrics["val_bias_squared"] == metrics["val_bias_squared"]


def test_the_validation_image_is_zero_where_no_coil_measures(_multicoil_run) -> None:
    """The bound's RSS PSNR fell to 0 dB on a background A = M F S never sees."""
    from spectramr.infrastructure.physics.coil_sensitivity import coil_support

    strategy, _, val = _multicoil_run
    context = strategy.validation_batch_context(val.input, val.target, val)
    maps = context["coil_sensitivities"].clone()
    maps[..., :8] = 0  # planted: a band no coil covers
    context = {**context, "coil_sensitivities": maps}
    with torch.no_grad():
        image = strategy._validation_forward(val.input, context)
    outside = ~coil_support(maps).expand_as(image)
    assert outside.any() and torch.all(image[outside] == 0)


def test_validation_without_the_batch_is_still_refused(_multicoil_run) -> None:
    """The operator must not revert to A = M F when the maps cannot be reached."""
    strategy, _, val = _multicoil_run
    with pytest.raises(ValueError, match="requires complex coil"):
        strategy._validation_forward(val.input, {"use_dc": False, "measured_kspace": None})


def test_the_validation_image_renders_on_the_capture_context(_multicoil_run) -> None:
    """#2431: the loop's image capture handed the forward a bare context, so the
    operator raised on every pass and no validation image was ever saved."""
    from spectramr.pipelines.train import visual_capture_inputs

    strategy, _, val = _multicoil_run
    x, _, context = visual_capture_inputs(strategy, val.input, val.target, val)
    assert x is val.input, "the capture must render on the tensors validation_step graded"
    with torch.no_grad():
        image = strategy._validation_forward(x, context)
    assert image.shape[0] == val.input.shape[0] and torch.isfinite(image).all()


def test_validation_step_builds_its_context_through_the_shared_hook(
    _multicoil_run, monkeypatch
) -> None:
    """The metric pass and the image capture must not drift into two builders."""
    strategy, _, val = _multicoil_run
    seen: list = []
    hook = strategy.validation_batch_context

    def spy(*args):
        seen.append(args[2])
        return hook(*args)

    monkeypatch.setattr(strategy, "validation_batch_context", spy)
    strategy.validation_step(val.input, val.target, batch_data=val)
    assert seen == [val]


# ---------------------------------------------------------------------------
# #2325: validation scores an image against an image.
# ---------------------------------------------------------------------------


def _scored_target(transform: str | None):
    from tests.fixtures.self_supervised_batches import build_strategy, load_arm, real_batches

    config = load_arm("ei_unet_m4raw_r4")
    scoring = config.validation.scoring.model_copy(update={"output_transform": transform})
    validation = config.validation.model_copy(update={"scoring": scoring})
    config = config.model_copy(update={"validation": validation})
    strategy = build_strategy(config)
    _, val = real_batches(config)
    prediction = torch.randn(val.input.shape[0], 2, *val.input.shape[-2:])
    _, target = strategy._apply_metric_transforms(prediction, val.target, config.validation)
    return target


def _centre_error(target: torch.Tensor) -> str | None:
    """Why the scored target is not an image, or ``None``."""
    h, w = target.shape[-2:]
    flat = int(target[0, 0].reshape(-1).argmax())
    if (flat // w, flat % w) == (h // 2, w // 2):
        return "the scored target peaks at the k-space centre: it is |k|, the DC sample"
    return None


def test_the_ei_target_is_scored_as_an_image() -> None:
    assert _centre_error(_scored_target("ifft_target_magnitude")) is None


def test_the_undeclared_transform_scores_kspace_magnitude() -> None:
    """The planted shape #2325 reported: auto-magnitude on a k-space target."""
    assert _centre_error(_scored_target(None)) is not None


def test_the_arms_declare_the_target_transform() -> None:
    from tests.fixtures.self_supervised_batches import load_arm

    for arm in ("ei_unet_m4raw_r4", "ei_control_mc_only_m4raw_r4", "ei_multicoil_m4raw_r4"):
        assert load_arm(arm).validation.scoring.output_transform == "ifft_target_magnitude"


def test_an_image_network_owns_its_zero_filled_baseline(_multicoil_run) -> None:
    """The mixin grades the INPUT with the arm's transform; for this arm that is |k_masked|."""
    strategy, _, _ = _multicoil_run
    assert strategy._owns_zero_filled_baseline is True


# ---------------------------------------------------------------------------
# Rep-EI (n2n PAPER section 4.5): a k-space network, the multi-coil operator,
# and the anchor on the second repetition.
# ---------------------------------------------------------------------------


def _rep_ei(alpha: float | None = None):
    from tests.fixtures.self_supervised_batches import build_strategy, load_arm, real_batches

    config = load_arm("rep_ei_m4raw_r2")
    if alpha is not None:
        ei = config.training.equivariant_imaging.model_copy(update={"alpha_equivariance": alpha})
        config = config.model_copy(
            update={"training": config.training.model_copy(update={"equivariant_imaging": ei})}
        )
    train, val = real_batches(config, coils=4, route="m4raw")
    return config, build_strategy(config), train, val


@pytest.fixture(scope="module")
def _rep_ei_run():
    return _rep_ei()


def test_rep_ei_takes_a_training_and_a_validation_step(_rep_ei_run) -> None:
    _, strategy, train, val = _rep_ei_run
    losses = strategy._compute_losses_impl(train.input, train.target, 0, batch=train)
    losses["g_total_loss"].backward()
    assert float(losses["loss_equivariance"].detach()) > 0.0
    metrics = strategy.validation_step(val.input, val.target, batch_data=val)
    assert metrics["val_psnr"] == metrics["val_psnr"]
    # A k-space network is scored by the symmetric ifft_magnitude, so the mixin's
    # own zero-filled baseline is the right one here.
    assert strategy._owns_zero_filled_baseline is False


def test_rep_ei_reconstructs_from_the_masked_first_repetition(_rep_ei_run) -> None:
    """On the M4Raw route `kspace` is repetition 2, whole (#2324's other route)."""
    _, strategy, train, _ = _rep_ei_run
    context = strategy._prepare_batch_context(train.input, train.target, batch=train)
    assert torch.equal(context["measured_kspace"], train.input)
    assert not torch.equal(context["measured_kspace"], train.target)


def test_at_alpha_zero_rep_ei_is_the_same_mask_arm_exactly() -> None:
    """The one knob is the EI term: without it the objective is the baseline's."""
    from tests.fixtures.self_supervised_batches import build_strategy, load_arm

    _, rep_ei, train, _ = _rep_ei(alpha=0.0)
    baseline = build_strategy(load_arm("conex_r2_samemask_m4raw"))
    baseline.env.generator.load_state_dict(rep_ei.env.generator.state_dict())

    torch.manual_seed(0)
    ei_total = rep_ei._compute_losses_impl(train.input, train.target, 0, batch=train)["loss"]
    torch.manual_seed(0)
    base_total = baseline._compute_losses_impl(train.input, train.target, 0, batch=train)["loss"]
    assert torch.allclose(ei_total.detach(), base_total.detach(), rtol=1e-5, atol=1e-7)


def test_the_alternate_repetition_anchor_needs_its_mask_and_its_loss(_rep_ei_run) -> None:
    from spectramr.infrastructure.training.strategies.equivariant_imaging_ops import (
        alternate_repetition_anchor,
    )

    _, strategy, train, _ = _rep_ei_run
    out = torch.zeros_like(train.target)
    with pytest.raises(ValueError, match="loss_mask"):
        alternate_repetition_anchor(out, train.target, {}, None, True, strategy.env)
    with pytest.raises(ValueError, match="masked_kspace_l2"):
        alternate_repetition_anchor(out, train.target, {"loss_mask": train.mask}, None, True, None)


def test_a_multicoil_kspace_network_needs_the_multicoil_operator() -> None:
    """Without maps the group would act on coil images and rotate the sensitivities."""
    from spectramr.infrastructure.training.strategies.equivariant_imaging_ops import (
        network_image,
    )

    with pytest.raises(ValueError, match="multicoil_operator"):
        network_image(torch.randn(1, 8, 8, 8), None, True)


# ---------------------------------------------------------------------------
# #2762: per-rank group stream, carried through a resume; alpha = 0 skips the branch
# ---------------------------------------------------------------------------


def _ei_on_rank(monkeypatch, rank: int | None) -> EquivariantImagingStrategy:
    """The real ``__init__`` over the arm's config, without building the model."""
    from types import SimpleNamespace

    from spectramr.infrastructure.training.strategies import rng_streams
    from tests.fixtures.self_supervised_batches import load_arm

    config = load_arm("ei_multicoil_m4raw_r4")
    monkeypatch.setattr(rng_streams, "distributed_rank", lambda: rank)

    def bare_init(self, env=None, device=None, **kwargs):
        self.config = config
        self.env = SimpleNamespace(generator=None, discriminator=None, losses={})

    monkeypatch.setattr(ReconstructionTrainingStrategy, "__init__", bare_init)
    return EquivariantImagingStrategy()


def _group_draws(strategy: EquivariantImagingStrategy, n: int = 16) -> list[int]:
    return [strategy._group.sample(strategy._rng) for _ in range(n)]


def test_two_ranks_draw_different_group_elements(monkeypatch) -> None:
    rank0, rank1 = _ei_on_rank(monkeypatch, 0), _ei_on_rank(monkeypatch, 1)
    assert _group_draws(rank0) != _group_draws(rank1)


def test_a_resume_continues_the_group_stream(monkeypatch) -> None:
    saved = _ei_on_rank(monkeypatch, None)
    _group_draws(saved)
    state = saved.strategy_state_dict()
    expected = _group_draws(saved)

    resumed = _ei_on_rank(monkeypatch, None)
    resumed.load_strategy_state_dict(state)
    assert _group_draws(resumed) == expected, "the resumed run replayed the group sequence"


def _stub_step(alpha: float) -> tuple[EquivariantImagingStrategy, list[int], dict]:
    """One step over a stub network: the strategy, its forward calls, its losses."""
    from spectramr.core.metrics.coil_manifold import as_interleaved
    from spectramr.infrastructure.physics.multicoil_ei_operator import interleaved_to_complex

    n = 16
    mask = _partial_mask(n, n)
    y = as_interleaved(mask * fft2c(_complex(1, 1, n, n, seed=4)))
    st = object.__new__(EquivariantImagingStrategy)
    st.multicoil_operator, st._consumes_kspace, st._emits_kspace = False, False, False
    st._group, st._rng, st.n_group_samples = DihedralGroup(), torch.Generator().manual_seed(0), 2
    st.measurement_target, st.alpha, st._gsure = "measured", alpha, None
    st._prepare_batch_context = lambda *a, **k: {"measured_kspace": y, "mask": mask}

    def prepare(bc, kspace, **_):
        coils = kspace if kspace.is_complex() else interleaved_to_complex(kspace)
        return as_interleaved(ifft2c(coils)), {}

    calls: list[int] = []
    weight = torch.ones((), requires_grad=True)

    def generate(lr, fk, bc):
        calls.append(1)
        return weight * lr, {}

    st._prepare_generator_inputs = prepare
    st._generate_predictions = generate
    st._resolve_equivariant_loss = EquivariantSSLReconLoss
    st._declare_model_input = lambda tensors, **kw: None
    losses = st._compute_losses_impl(y, torch.zeros_like(y), 0)
    return st, calls, losses


def test_alpha_zero_runs_one_forward_and_keeps_the_stream_paired() -> None:
    control, control_calls, control_losses = _stub_step(alpha=0.0)
    paired, paired_calls, _ = _stub_step(alpha=1.0)
    assert len(paired_calls) == 4, "x_hat and the re-measured branch, two group samples"
    assert len(control_calls) == 1, "alpha = 0 still ran the equivariance branch"
    assert torch.equal(control._rng.get_state(), paired._rng.get_state())
    assert float(control_losses["loss_equivariance"]) == 0.0
    assert control_losses["loss"].requires_grad


def test_alpha_zero_keeps_a_nan_branch_out_of_the_total(monkeypatch) -> None:
    """0 * NaN is NaN: a non-finite equivariance term used to poison the control's loss."""
    import spectramr.infrastructure.training.strategies.equivariant_imaging_strategy as ei_module

    def nan_branches(reconstruct, forward_op, group, y, g):
        x_hat = reconstruct(y)
        return x_hat, x_hat * float("nan"), x_hat

    monkeypatch.setattr(ei_module, "equivariant_imaging_branches", nan_branches)
    _, _, losses = _stub_step(alpha=0.0)
    assert torch.isfinite(losses["loss"])


# ---------------------------------------------------------------------------
# #2509: Rep-BV grades F S x_hat; #2508: the snapshot names the real model input.
# Stub instances only -- no generator is built.
# ---------------------------------------------------------------------------


def _complex(*shape: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=gen), torch.randn(*shape, generator=gen))


@pytest.mark.parametrize("multicoil", [True, False])
def test_the_context_metrics_grade_the_network_image_as_coil_kspace(multicoil) -> None:
    from spectramr.core.metrics.coil_manifold import as_interleaved
    from spectramr.infrastructure.physics.fft_ops import sense_forward

    h, w = 12, 20  # non-square
    maps = _complex(1, 4, h, w, seed=0)
    maps = maps / maps.abs().pow(2).sum(1, keepdim=True).sqrt()
    image = _complex(1, 1, h, w, seed=1)
    st = object.__new__(EquivariantImagingStrategy)
    st.multicoil_operator, st._emits_kspace = multicoil, False
    context = {"coil_sensitivities": maps, "mask": torch.ones(1, 1, h, w)}
    graded, domain = st._context_metric_prediction(as_interleaved(image), context)
    expected = sense_forward(image, maps, None) if multicoil else fft2c(image)
    assert domain == "kspace"
    torch.testing.assert_close(graded, as_interleaved(expected))


def test_a_kspace_network_is_graded_on_its_own_output() -> None:
    st = object.__new__(EquivariantImagingStrategy)
    st.multicoil_operator, st._emits_kspace = True, True
    raw = torch.randn(1, 8, 12, 20)
    graded, domain = st._context_metric_prediction(raw, {})
    assert graded is raw and domain == "kspace"


def test_the_step_declares_the_tensor_the_network_is_fed_once() -> None:
    """#2508: ``input_prepared`` is k-space; the network gets ``ifft2c`` of it."""
    from spectramr.core.metrics.coil_manifold import as_interleaved
    from spectramr.infrastructure.physics.multicoil_ei_operator import interleaved_to_complex

    assert EquivariantImagingStrategy.snapshot_prepared_is_model_input is False
    assert EquivariantImagingStrategy.snapshot_model_input_tag == "ei_model_input"
    n = 16
    mask = _partial_mask(n, n)
    y = as_interleaved(mask * fft2c(_complex(1, 1, n, n, seed=2)))
    st = object.__new__(EquivariantImagingStrategy)
    st.multicoil_operator, st._consumes_kspace, st._emits_kspace = False, False, False
    st._group, st._rng, st.n_group_samples = DihedralGroup(), torch.Generator().manual_seed(0), 2
    st.measurement_target, st.alpha, st._gsure = "measured", 1.0, None
    st._prepare_batch_context = lambda *a, **k: {"measured_kspace": y, "mask": mask}
    fed: list[torch.Tensor] = []

    def prepare(bc, kspace, **_):
        coils = kspace if kspace.is_complex() else interleaved_to_complex(kspace)
        image = as_interleaved(ifft2c(coils))
        fed.append(image)
        return image, {}

    st._prepare_generator_inputs = prepare
    st._generate_predictions = lambda lr, fk, bc: (0.5 * lr, {})
    st._resolve_equivariant_loss = EquivariantSSLReconLoss
    declared: list[dict] = []
    st._declare_model_input = lambda tensors, **kw: declared.append((tensors, kw))
    st._compute_losses_impl(y, torch.zeros_like(y), 0)
    assert len(fed) == 4, "x_hat and the re-measured branch, two group samples"
    assert len(declared) == 1
    tensors, kw = declared[0]
    assert tensors["model_input"] is fed[0] and kw["in_kspace_keys"] == set()
    # An image network's A^H y is not k-space; without this the canonical union IFFTs it (#3043).
    assert kw["domain_exact"] is True
    # The wrapper's contract, not a stub: the declaration must name its model input.
    from spectramr.infrastructure.training.model_input_contract import require_model_input_key

    tag = EquivariantImagingStrategy.snapshot_model_input_tag
    key = require_model_input_key(
        strategy_name="EquivariantImagingStrategy",
        tag=tag,
        tensors=tensors,
        model_input_key=kw["extra"]["model_input_key"],
    )
    assert key == "model_input"
    with pytest.raises(ValueError, match="without naming"):  # the sabine crash, planted
        require_model_input_key(
            strategy_name="EquivariantImagingStrategy", tag=tag, tensors=tensors, model_input_key=None
        )


class TestTheAnchorCarriesItsDeclaredWeight:
    """Non-negotiable 8: the alternate-repetition anchor ignored the YAML weight."""

    @staticmethod
    def _strategy(config):
        st = object.__new__(EquivariantImagingStrategy)
        st.config = config
        return st

    def test_rep_ei_reads_weight_one(self) -> None:
        from tests.fixtures.self_supervised_batches import load_arm

        assert self._strategy(load_arm("rep_ei_m4raw_r2"))._anchor_weight() == pytest.approx(1.0)

    def test_a_declared_weight_reaches_the_anchor(self, monkeypatch) -> None:
        import spectramr.infrastructure.training.strategies.equivariant_imaging_strategy as mod

        monkeypatch.setattr(mod, "declared_loss_weights", lambda config: {"masked_kspace_l2": 0.25})
        assert self._strategy(object())._anchor_weight() == pytest.approx(0.25)

    def test_planted_a_missing_declaration_raises(self, monkeypatch) -> None:
        import spectramr.infrastructure.training.strategies.equivariant_imaging_strategy as mod

        monkeypatch.setattr(mod, "declared_loss_weights", lambda config: {"l1": 0.0})
        with pytest.raises(ValueError, match="no entry for it"):
            self._strategy(object())._anchor_weight()


# ---------------------------------------------------------------------------
# The acquired band: structural zeros in the anchor, a band-limited EI target.
# ---------------------------------------------------------------------------


def _band_arm(**knobs):
    from tests.fixtures.self_supervised_batches import build_strategy, load_arm, real_batches

    config = load_arm("ei_multicoil_m4raw_r4")
    ei = config.training.equivariant_imaging.model_copy(update=knobs)
    config = config.model_copy(
        update={"training": config.training.model_copy(update={"equivariant_imaging": ei})}
    )
    train, _ = real_batches(config, coils=4, pe_zero_fill=4)
    return build_strategy(config), train


def test_the_band_knobs_take_a_training_step_and_project_the_target(monkeypatch) -> None:
    import spectramr.infrastructure.training.strategies.equivariant_imaging_strategy as eis

    calls = []
    real = eis.band_project
    monkeypatch.setattr(eis, "band_project", lambda x, s: calls.append(s) or real(x, s))
    strategy, train = _band_arm(
        anchor_mask="with_structural_zeros",
        anchor_normalization="selected",
        equivariance_target="band_limited",
    )
    losses = strategy._compute_losses_impl(train.input, train.target, 0, batch=train)
    losses["g_total_loss"].backward()
    assert torch.isfinite(losses["g_total_loss"])
    assert calls, "the band-limited target never reached the equivariance term"
    assert (~calls[0]).any(), "the simulated partial FOV produced no structural zeros"


def test_the_structural_zeros_change_the_anchor() -> None:
    base, train = _band_arm(anchor_mask="input", anchor_normalization="selected")
    zeros, _ = _band_arm(anchor_mask="with_structural_zeros", anchor_normalization="selected")
    zeros.generator_model.load_state_dict(base.generator_model.state_dict())
    a = base._compute_losses_impl(train.input, train.target, 0, batch=train)["loss_consistency"]
    b = zeros._compute_losses_impl(train.input, train.target, 0, batch=train)["loss_consistency"]
    assert float(a) != float(b)


def test_a_band_knob_without_the_support_is_refused() -> None:
    import copy

    strategy, train = _band_arm(equivariance_target="band_limited")
    batch = copy.copy(train)
    batch.metadata = {k: v for k, v in train.metadata.items() if k != "support_mask"}
    with pytest.raises(ValueError, match="support_mask"):
        strategy._compute_losses_impl(train.input, train.target, 0, batch=batch)


def test_val_loss_is_the_training_anchor_not_an_image_vs_kspace_l1(_multicoil_run) -> None:
    """The generic fallback compared |image| with k-space channel 0 (~0.2 against a 0.004 train loss)."""
    strategy, train, val = _multicoil_run
    train_loss = strategy._compute_losses_impl(train.input, train.target, 0, batch=train)
    metrics = strategy.validation_step(val.input, val.target, batch_data=val)
    ratio = float(metrics["val_loss"]) / float(train_loss["loss_consistency"])
    assert 0.1 < ratio < 10.0, f"val_loss {metrics['val_loss']} is off the anchor's scale"
