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
    loss = s._predictor(s.env.generator(torch.randn(3, 5))).pow(2).sum()
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
