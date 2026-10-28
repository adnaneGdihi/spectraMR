"""Strategies probe the generator through DistributedDataParallel (issue #2636).

DDP and DataParallel do not forward attribute reads to the module they wrap, so a
``hasattr(generator, ...)`` probe answers "absent" for a capability the inner module
has. Each test below runs the production method under a real two-rank gloo DDP and
checks the capability is seen (or, when the config requires it and it cannot be
reached through the wrapper's forward, that the run raises on every rank).
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

WORLD = 2


class _PosteriorCapturedError(Exception):
    """Stops the VAE step once the posterior has been handed to the loss computer."""


class _DeepSupervised(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv(x)

    def forward_with_intermediates(self, x):
        y = self.conv(x)
        return [0.5 * y, y]


class _Plain(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv(x)


class _BareTensorVAE(torch.nn.Module):
    """Returns a bare tensor and caches its posterior, as ``slat_vae_slab_to_volume`` does."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self._aux: dict | None = None

    def forward(self, x):
        y = self.conv(x)
        self._aux = {
            "mu": torch.full((x.shape[0], 2), 0.5),
            "logvar": torch.full((x.shape[0], 2), -0.25),
        }
        return y

    def last_aux(self):
        return self._aux


class _MarkerConditioned(torch.nn.Module):
    """Exposes a ``cross_attention`` attribute and records which kwargs reached forward."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 1)
        self.cross_attention = torch.nn.Identity()
        self.received: set[str] = set()

    def forward(self, x, **kwargs):
        self.received = set(kwargs)
        return self.conv(x)


def _ddp(module):
    from torch.nn.parallel import DistributedDataParallel

    return DistributedDataParallel(module)


def _reconstruction_strategy(generator, ds_weight: float):
    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    s = object.__new__(ReconstructionTrainingStrategy)
    s.env = SimpleNamespace(generator=generator)
    s.config = SimpleNamespace(
        training=SimpleNamespace(enforce_output_range=False),
        losses=SimpleNamespace(lambda_deep_supervision=ds_weight),
        model=SimpleNamespace(model_type="test"),
    )
    s.logging_service = None
    return s


def _check_deep_supervision_required(rank: int) -> None:
    """Required deep supervision under DDP must raise on every rank, naming the knob."""
    s = _reconstruction_strategy(_ddp(_DeepSupervised()), ds_weight=0.1)
    with pytest.raises(RuntimeError, match="lambda_deep_supervision"):
        s._generate_predictions(torch.randn(1, 1, 4, 4), {}, {})


def _check_deep_supervision_not_required(rank: int) -> None:
    """Without a deep-supervision loss the wrapper's forward is the whole answer."""
    x = torch.randn(1, 1, 4, 4)
    s = _reconstruction_strategy(_ddp(_DeepSupervised()), ds_weight=0.0)
    out, intermediates = s._generate_predictions(x, {}, {})
    assert intermediates is None
    assert out.shape == x.shape


def _check_deep_supervision_unwrapped(rank: int) -> None:
    """Control: an unwrapped module still yields its intermediates."""
    x = torch.randn(1, 1, 4, 4)
    s = _reconstruction_strategy(_DeepSupervised(), ds_weight=0.1)
    _, intermediates = s._generate_predictions(x, {}, {})
    assert isinstance(intermediates, list) and len(intermediates) == 2


def _check_forward_kwargs_follow_inner_signature(rank: int) -> None:
    """``timesteps`` must not be injected for a model whose forward has no such argument."""
    s = _reconstruction_strategy(_ddp(_Plain()), ds_weight=0.0)
    s.config.data = SimpleNamespace(multislice_enabled=False)
    s.config.model.in_channels = 1
    s.config.validation = None
    s._needs_kspace_adjoint = lambda batch_context: False
    s._is_implicit_model = lambda: False
    _, kwargs = s._prepare_generator_inputs(
        {"use_dc": False}, torch.randn(1, 1, 4, 4), validation=True
    )
    assert "timesteps" not in kwargs, kwargs


def _check_im_size_from_generator(rank: int) -> None:
    """The generator's declared im_size is read through the wrapper, not replaced by 256."""
    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    inner = _Plain()
    inner.im_size = (64, 64)
    s = _reconstruction_strategy(_ddp(inner), ds_weight=0.0)
    assert ReconstructionTrainingStrategy._declared_im_size(s, {}) == (64, 64)


def _check_vae_posterior_reaches_kl(rank: int) -> None:
    """A bare-tensor VAE under DDP must hand its real posterior to the KL term."""
    from spectramr.infrastructure.training.strategies.vae import VAETrainingStrategy

    class _Spy:
        def __init__(self) -> None:
            self.posterior = None

        def compute(self, **kwargs):
            self.posterior = kwargs["posterior"]
            raise _PosteriorCapturedError

    spy = _Spy()
    s = object.__new__(VAETrainingStrategy)
    s.env = SimpleNamespace(generator=_ddp(_BareTensorVAE()), losses={})
    s.precision_manager = SimpleNamespace(
        prepare_latent_for_kl_computation=lambda mu, logvar: (mu, logvar)
    )
    s.loss_computer = spy
    x = torch.randn(1, 1, 4, 4)
    with pytest.raises(_PosteriorCapturedError):
        s._compute_losses_impl(x, x.clone(), epoch=0, iteration=0)
    mu, _ = spy.posterior
    assert torch.allclose(mu, torch.full_like(mu, 0.5)), "KL saw a zero posterior"


def _check_marker_conditioning_reaches_wrapped_model(rank: int) -> None:
    """The marker branch is chosen from the inner model and the call still goes through DDP."""
    from spectramr.infrastructure.training.strategies.virtual_fiducial_strategy import (
        ConcreteVirtualFiducialStrategy,
    )

    inner = _MarkerConditioned()
    s = object.__new__(ConcreteVirtualFiducialStrategy)
    s.env = SimpleNamespace(generator=_ddp(inner))
    s._ensure_conditioner = lambda channels: None
    s._conditioning_context = None
    s.simulator = None
    corrupted = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
    marker = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
    s._forward_model(corrupted, marker)
    assert "marker_residual" in inner.received, inner.received


SCENARIOS = {
    "ds_required": _check_deep_supervision_required,
    "ds_not_required": _check_deep_supervision_not_required,
    "ds_unwrapped": _check_deep_supervision_unwrapped,
    "forward_kwargs": _check_forward_kwargs_follow_inner_signature,
    "im_size": _check_im_size_from_generator,
    "vae_posterior": _check_vae_posterior_reaches_kl,
    "marker": _check_marker_conditioning_reaches_wrapped_model,
}


def _worker(rank: int, store: str, scenario: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        SCENARIOS[scenario](rank)
    finally:
        dist.destroy_process_group()


def _spawn(scenario: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(os.path.join(tmp, "store"), scenario), nprocs=WORLD, join=True)


def test_required_deep_supervision_raises_on_every_rank_under_ddp() -> None:
    _spawn("ds_required")


def test_unrequired_deep_supervision_is_not_forced_under_ddp() -> None:
    _spawn("ds_not_required")


def test_unwrapped_deep_supervision_still_returns_intermediates() -> None:
    _spawn("ds_unwrapped")


def test_forward_kwargs_are_gated_by_the_inner_signature_under_ddp() -> None:
    _spawn("forward_kwargs")


def test_generator_im_size_is_read_through_ddp() -> None:
    _spawn("im_size")


def test_vae_posterior_reaches_kl_under_ddp() -> None:
    _spawn("vae_posterior")


def test_marker_conditioning_reaches_wrapped_model_under_ddp() -> None:
    _spawn("marker")
