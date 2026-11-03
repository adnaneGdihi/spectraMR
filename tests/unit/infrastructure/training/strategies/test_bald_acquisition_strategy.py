"""BALD: real MC-dropout disagreement over k-space lines, applied to what the backbone sees."""

from __future__ import annotations

import types
import warnings

import pytest
import torch
from torch import nn

from spectramr.infrastructure.training.strategies import reconstruction
from spectramr.infrastructure.training.strategies.bald_acquisition_strategy import (
    BALDAcquisitionStrategy,
    posterior_samples,
)


class _Tiny(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 1) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(in_channels, 8, 3, padding=1), nn.BatchNorm2d(8))
        self.mid = nn.Conv2d(8, 8, 3, padding=1)
        self.head = nn.Conv2d(8, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.mid(torch.relu(self.enc(x)))))


@pytest.fixture
def wrapper(monkeypatch):
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    monkeypatch.setitem(
        MODEL_REGISTRY, "_tiny", {**MODEL_REGISTRY["configurable_unet"], "class": _Tiny}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from spectramr.models.factories.model_factory import ModelFactory

    def build(p: float):
        torch.manual_seed(0)
        return ModelFactory().create_generator(
            model_type="bald_active_recon",
            in_channels=2,
            out_channels=1,
            _declared_keys=frozenset({"backbone", "mc_dropout_p", "n_posterior_samples"}),
            backbone="_tiny",
            mc_dropout_p=p,
            n_posterior_samples=4,
        )

    return build


def _strategy(model, enabled=True, query=2):
    s = object.__new__(BALDAcquisitionStrategy)
    s.device = torch.device("cpu")
    s.env = types.SimpleNamespace(generator=model)
    s._active = types.SimpleNamespace(
        enabled=enabled,
        retrain_every=1,
        target_rate=0.5,
        seed_mask_lf_fraction=0.125,
        query_batch=query,
    )
    s._acquired_mask, s._n_acquired, s._step_count, s._n_samples = None, 0, 0, 4
    return s


def test_dropout_reaches_every_stage_but_the_head_and_only_while_sampling(wrapper) -> None:
    model = wrapper(0.5).eval()
    assert model.mc_dropout_stages == ["enc", "mid"]
    x = torch.randn(1, 2, 8, 16)
    torch.testing.assert_close(model(x), model(x))
    model.mc_active = True
    assert not torch.allclose(model(x), model(x))


def test_scoring_keeps_bn_statistics_and_restores_the_mode(wrapper) -> None:
    model = wrapper(0.5).train()
    before = model.backbone.enc[1].running_mean.clone()
    s = _strategy(model)
    scores = s._bald_score(torch.randn(2, 2, 8, 16))
    assert scores.shape == (16,) and bool((scores > 0).all())
    torch.testing.assert_close(model.backbone.enc[1].running_mean, before)
    assert model.training and not model.mc_active


def test_no_mc_dropout_is_refused(wrapper) -> None:
    s = _strategy(wrapper(0.0))
    with pytest.raises(RuntimeError, match="mc_dropout_p"):
        s._bald_score(torch.randn(1, 2, 8, 16))


def test_acquisition_masks_what_the_backbone_sees(wrapper, monkeypatch) -> None:
    s = _strategy(wrapper(0.5))
    seen = {}
    monkeypatch.setattr(
        reconstruction.ReconstructionTrainingStrategy,
        "_compute_losses_impl",
        lambda self, input_batch=None, **kw: seen.update(x=input_batch) or {},
    )
    x = torch.ones(1, 2, 8, 16)
    s._compute_losses_impl(input_batch=x, target_batch=x, epoch=0)
    assert s._n_acquired == int(s._acquired_mask.sum()) == 3 + 2  # seed of 3 lines + one query of 2
    torch.testing.assert_close(seen["x"][0, 0, 0], s._acquired_mask.float())
    monkeypatch.setattr(
        reconstruction.ReconstructionTrainingStrategy,
        "validation_step",
        lambda self, i, t, **kw: {"seen": i},
    )
    out = s.validation_step(x, x)
    torch.testing.assert_close(out["seen"][0, 0, 0], s._acquired_mask.float())


def test_a_disabled_loop_leaves_the_input_untouched(wrapper) -> None:
    s = _strategy(wrapper(0.5), enabled=False)
    x = torch.ones(1, 2, 8, 16)
    assert s._masked(x) is x


def test_posterior_sample_count_is_read_and_checked() -> None:
    cfg = types.SimpleNamespace(
        model=types.SimpleNamespace(model_kwargs={"n_posterior_samples": 16})
    )
    assert posterior_samples(cfg) == 16
    cfg.model.model_kwargs = {"n_posterior_samples": 1}
    with pytest.raises(ValueError, match="at least 2"):
        posterior_samples(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sync-debug mode needs a GPU")
def test_acquisition_does_not_sync_the_gpu() -> None:
    class _W(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net, self.mc_dropout_p, self.mc_active = nn.Conv2d(2, 1, 1), 0.3, False

        def forward(self, x):
            return nn.functional.dropout(self.net(x), 0.3, self.training or self.mc_active)

    s = _strategy(_W().cuda())
    s.device = torch.device("cuda")
    x = torch.randn(2, 2, 8, 16, device="cuda")
    s._maybe_acquire_lines(x)  # warm-up; seeds the mask
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        s._maybe_acquire_lines(x)
        s._masked(x)
    finally:
        torch.cuda.set_sync_debug_mode("default")
