"""Tests for the I-JEPA masking helper.

Review finding (2026-07-01): ``multi_block_mask`` sampled mask geometry on the
input's ``device`` (CUDA in real training) and ``.item()``-synced every scalar
draw — ~16 blocking host/device syncs per training step (performance.md). Mask
geometry is scalar bookkeeping; it now runs on CPU (where ``.item()`` is a free
host read) and the boolean masks are moved to ``device`` exactly once.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.training.strategies.jepa_strategy import (  # noqa: E402
    multi_block_mask,
)
from spectramr.infrastructure.training.strategies.jepa_targets import GEOMETRY_DIM  # noqa: E402


def test_multi_block_mask_contract() -> None:
    ctx, targets = multi_block_mask(32, 24, n_targets=4)
    assert ctx.shape == (32, 24)
    assert ctx.dtype == torch.bool
    assert len(targets) == 4
    assert all(t.shape == (32, 24) and t.dtype == torch.bool for t in targets)
    # Context is the complement of the union of targets (up to the dilation
    # fallback), so at least some pixels are masked out of context.
    assert ctx.sum() < ctx.numel()


def test_multi_block_mask_returns_on_requested_device() -> None:
    """The masks must come back on the requested device (moved once at the end)."""
    dev = torch.device("cpu")
    ctx, targets = multi_block_mask(16, 16, n_targets=3, device=dev)
    assert ctx.device == dev
    assert all(t.device == dev for t in targets)


def test_multi_block_mask_target_count_scales() -> None:
    for n in (1, 2, 6):
        _, targets = multi_block_mask(20, 20, n_targets=n)
        assert len(targets) == n


# ── the target encoder and predictor survive a resume (#2820) ───────────────


def _jepa(seed: int):
    from types import SimpleNamespace

    from spectramr.infrastructure.training.strategies.jepa_strategy import JEPAStrategy

    torch.manual_seed(seed)
    gen = torch.nn.Linear(5, 5)
    s = object.__new__(JEPAStrategy)
    s.env = SimpleNamespace(generator=gen, opt_g=torch.optim.Adam(gen.parameters()))
    s._target_encoder = None
    s._predictor = None
    # What BaseTrainingStrategy.__init__ sets up for owned modules (#2629).
    s.config, s.device = SimpleNamespace(parallel=None), torch.device("cpu")
    s._owned_modules, s._owned_wrappers = {}, {}
    return s


def _trained_jepa():
    s = _jepa(0)
    s._ensure_modules(embed_dim=5)
    query = torch.cat([s.env.generator(torch.randn(3, 5)), torch.zeros(3, GEOMETRY_DIM)], dim=1)
    loss = s._predictor(query).pow(2).sum()
    loss.backward()
    s.env.opt_g.step()
    with torch.no_grad():
        for p in s._target_encoder.parameters():
            p.add_(0.5)
    return s


def test_a_resume_restores_the_predictor_and_target_encoder() -> None:
    saved = _trained_jepa()
    resumed = _jepa(1)

    resumed.load_strategy_state_dict(saved.strategy_state_dict())
    resumed.env.opt_g.load_state_dict(saved.env.opt_g.state_dict())

    for name in ("_predictor", "_target_encoder"):
        live = getattr(resumed, name).state_dict()
        for key, value in getattr(saved, name).state_dict().items():
            assert torch.equal(live[key], value), f"{name}.{key}"
    assert resumed.env.opt_g.param_groups[-1]["params"] == list(resumed._predictor.parameters())
    assert not any(p.requires_grad for p in resumed._target_encoder.parameters())


def test_a_resume_without_a_generator_refuses_the_saved_modules() -> None:
    state = _trained_jepa().strategy_state_dict()
    resumed = _jepa(1)
    resumed.env.generator = None
    with pytest.raises(RuntimeError, match="_predictor"):
        resumed.load_strategy_state_dict(state)


# ── block-embedding objective and frozen-encoder linear probe, reached from training_mode ──


def _latent_unet():
    from spectramr.models.interfaces.latent_access import LatentAccessMixin

    class _Net(LatentAccessMixin, torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc = torch.nn.Conv2d(1, 6, 3, stride=2, padding=1)
            self.dec = torch.nn.Conv2d(6, 1, 1)

        def forward(self, x):
            z = torch.relu(self.enc(x))
            self._set_latent_state(embedding=z.mean((-2, -1)), bottleneck=z)
            return torch.nn.functional.interpolate(self.dec(z), size=x.shape[-2:])

    return _Net()


def _resolved_jepa():
    from types import SimpleNamespace

    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    cls = TrainingStrategyFactory().get_strategy_class(
        SimpleNamespace(training=SimpleNamespace(strategy_class=None, training_mode="jepa"))
    )
    torch.manual_seed(0)
    s = object.__new__(cls)
    gen = _latent_unet()
    s.env = SimpleNamespace(generator=gen, opt_g=torch.optim.Adam(gen.parameters()))
    s._target_encoder = s._predictor = s._probe = None
    s.adapter_chains = {}
    s.config, s.device = SimpleNamespace(parallel=None), torch.device("cpu")
    s._owned_modules, s._owned_wrappers = {}, {}
    s._validation_computer = SimpleNamespace(
        get_metric_names=lambda: [], reset=lambda: None, finalize=lambda: {}
    )
    return s


def test_training_mode_jepa_predicts_bottleneck_block_embeddings() -> None:
    s = _resolved_jepa()
    out = s._compute_losses_impl({"input": torch.rand(2, 1, 16, 16)})
    out["loss_total"].backward()
    # The predictor takes the 6-channel bottleneck embedding plus the block geometry,
    # not the pooled one-channel output.
    assert s._predictor[0].in_features == 6 + GEOMETRY_DIM
    assert s.env.generator.enc.weight.grad.abs().sum() > 0


def test_validation_writes_val_loss_and_the_contrast_probe() -> None:
    s = _resolved_jepa()
    s._compute_losses_impl({"input": torch.rand(2, 1, 16, 16)})
    s.reset_validation_metrics()
    for subject in range(4):
        for contrast in range(3):
            image = torch.rand(1, 1, 3, 16, 16) + 2.0 * contrast  # one volume of 3 slices
            batch = {"contrast_idx": torch.tensor([contrast]), "subject_id": [f"s{subject}"]}
            metrics = s.validation_step(image, image, batch_data=batch)
            assert torch.isfinite(torch.tensor(metrics["val_loss"]))
    summary = s.finalize_validation()
    assert summary["val_probe_split_by_subject"] == 1.0
    assert summary["val_probe_contrast_acc"] > summary["val_probe_contrast_chance"]
    assert s.env.generator.training  # validation restored the train mode


def test_validation_before_training_or_without_contrast_labels_raises() -> None:
    s = _resolved_jepa()
    with pytest.raises(RuntimeError, match="before the first training step"):
        s.validation_step(torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16), batch_data={})
    s._compute_losses_impl({"input": torch.rand(2, 1, 16, 16)})
    s.reset_validation_metrics()
    with pytest.raises(KeyError, match="contrast_idx"):
        s.validation_step(torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16), batch_data={})


def test_pre_model_chain_feeds_training_and_validation() -> None:
    """A 2-channel batch reaches the 1-channel encoder only through the pre_model chain."""
    s = _resolved_jepa()
    calls = []

    def rss(x):
        calls.append(tuple(x.shape))
        return x.pow(2).sum(1, keepdim=True).sqrt()

    s.adapter_chains = {"pre_model": [rss]}
    two_ch = torch.rand(2, 2, 16, 16)
    s._compute_losses_impl({"input": two_ch})  # direct call: the strategy applies the chain
    assert calls == [(2, 2, 16, 16)]
    # train_step hands the already-adapted tensor; the chain is not applied twice.
    s._compute_losses_impl(input_batch=rss(two_ch), batch={"input": two_ch})
    assert len(calls) == 2
    s.reset_validation_metrics()
    volume = torch.rand(1, 2, 3, 16, 16)
    batch = {"contrast_idx": torch.tensor([0]), "subject_id": ["s0"]}
    metrics = s.validation_step(volume, volume, batch_data=batch)
    assert (1, 2, 3, 16, 16) in calls[2:]
    assert torch.isfinite(torch.tensor(metrics["val_loss"]))
