"""Hidden-attribute probes and non-forward calls under DDP (follow-ups to #2636).

Each test runs the production method under a real two-rank gloo DDP. Before the
fix each probe read "absent" on the wrapper and took a degraded path silently;
after it, the capability is seen, or a config-required non-forward call raises.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

WORLD = 2


class _Log:
    def log_info(self, *args, **kwargs):
        pass

    def log_warning(self, *args, **kwargs):
        pass


class _DCModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.dc_layer = torch.nn.Identity()
        self.kspace_process = SimpleNamespace(
            mask_generator=SimpleNamespace(default_pattern="uniform", _accelerator_kwargs={})
        )

    def forward(self, x):
        return self.conv(x)


class _FieldModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(1))  # DDP refuses parameter-free modules

    def forward(self, x, field_strength=1.0):
        return x * field_strength


class _TimeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(1))  # DDP refuses parameter-free modules

    def forward(self, x, t):
        return x


class _NoTimeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(1))  # DDP refuses parameter-free modules

    def forward(self, x):
        return x


class _ScoreNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x_t):
        return self.conv(x_t)


class _QMapNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv(x)


class _CompiledLike(torch.nn.Module):
    """Nests the model as ``torch.compile`` does (``_orig_mod``), so DDP wraps a non-module."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self._orig_mod = inner

    def forward(self, x):
        return self._orig_mod(x)


class _GraphInner(torch.nn.Module):
    """Graph generator that owns a ``model`` whose forward takes no timesteps."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _ScoreNet()

    def forward(self, x_t, timesteps=None):
        return self.model(x_t)


def _ddp(module):
    from torch.nn.parallel import DistributedDataParallel

    return DistributedDataParallel(module)


def _check_dc_layer_probe(rank: int) -> None:
    """The model-owned DC layer is adopted by the strategy even when the model is wrapped."""
    from spectramr.infrastructure.training.strategies.mixins.kspace import KspaceMixin

    class _Host(KspaceMixin):
        pass

    inner = _DCModel()
    s = object.__new__(_Host)
    s.config = SimpleNamespace(model=SimpleNamespace(model_type="kspace_cold_diffusion"))
    s.env = SimpleNamespace(generator=_ddp(inner))
    s.setup_kspace_components()
    assert s.dc_layer is inner.dc_layer, s.dc_layer


def _check_strategy_dc_reused(rank: int) -> None:
    """use_dc follows a compiled-then-DDP model's DC layer, not a silent False."""
    from spectramr.infrastructure.training.strategy_helpers import (
        StrategyInitializationHelper,
    )

    inner = _DCModel()
    s = SimpleNamespace(generator_model=_ddp(_CompiledLike(inner)), logging_service=_Log())
    StrategyInitializationHelper.initialize_data_consistency(s, use_dc=True)
    assert s.use_dc is True
    assert s.dc_layer is inner.dc_layer


def _check_conformal_field_strength(rank: int) -> None:
    """A field-conditioned generator receives the batch's field under DDP."""
    from spectramr.infrastructure.training.strategies.conformal_calibration_strategy import (
        ConformalCalibrationStrategy,
    )

    x = torch.ones(1, 1, 2, 2)
    batch = {"input": x, "target": x, "field_strength_target": 3.0}
    pred, _ = ConformalCalibrationStrategy._unpack_calibration_pair(batch, _ddp(_FieldModel()))
    assert torch.allclose(pred, 3.0 * x), pred


def _check_time_arg_detected_through_ddp(rank: int) -> None:
    """A time-conditioned generator under DDP is detected; a 1-arg one is not."""
    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )

    assert DiffusionTrainingStrategy._generator_accepts_time(_ddp(_TimeModel())) is True
    assert DiffusionTrainingStrategy._generator_accepts_time(_ddp(_NoTimeModel())) is False


def _check_score_timesteps_gated_by_inner_forward(rank: int) -> None:
    """A score net without a timesteps argument is called without one, through DDP."""
    from spectramr.infrastructure.training.strategies.bloch_manifold_dps_strategy import (
        BlochManifoldDPSStrategy,
    )

    net = _ScoreNet()
    x = torch.randn(1, 1, 4, 4)
    out = BlochManifoldDPSStrategy._forward_score(_ddp(net), x, torch.zeros(1))
    assert torch.allclose(out, net.conv(x))


def _check_qmap_render_refuses_wrapped_grad_path(rank: int) -> None:
    """Non-forward renders on a wrapped generator raise while gradients are on."""
    from spectramr.domain.exceptions import ConfigurationError
    from spectramr.infrastructure.training.strategies.calibrated_qmap_strategy import (
        CalibratedQMapStrategy,
    )

    s = object.__new__(CalibratedQMapStrategy)
    s.config = SimpleNamespace(parallel=SimpleNamespace(strategy="ddp"))
    s.env = SimpleNamespace(generator=_ddp(_QMapNet()))
    with pytest.raises(ConfigurationError, match="outside the wrapped forward"):
        s._render(torch.randn(1, 1, 4, 4), None)


def _check_graph_cold_timesteps_gated_by_inner_model(rank: int) -> None:
    """The timesteps probe reads the wrapped generator's ``model``, not the DDP forward."""
    from spectramr.infrastructure.training.strategies.graph_cold_diffusion_strategy import (
        GraphColdDiffusionStrategy,
    )

    s = object.__new__(GraphColdDiffusionStrategy)
    s.env = SimpleNamespace(generator=_ddp(_GraphInner()))
    kwargs = s._build_forward_kwargs(
        torch.randn(1, 1, 4, 4), torch.zeros(1), None, mask=torch.ones(1, 1, 4, 4)
    )
    assert "timesteps" not in kwargs and "time" not in kwargs, kwargs


SCENARIOS = {
    "dc_layer_probe": _check_dc_layer_probe,
    "strategy_dc_reused": _check_strategy_dc_reused,
    "conformal_field_strength": _check_conformal_field_strength,
    "time_arg": _check_time_arg_detected_through_ddp,
    "score_timesteps": _check_score_timesteps_gated_by_inner_forward,
    "qmap_render": _check_qmap_render_refuses_wrapped_grad_path,
    "graph_cold_timesteps": _check_graph_cold_timesteps_gated_by_inner_model,
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


def test_kspace_dc_layer_is_found_through_ddp() -> None:
    _spawn("dc_layer_probe")


def test_strategy_dc_is_reused_through_ddp() -> None:
    _spawn("strategy_dc_reused")


def test_conformal_field_strength_reaches_wrapped_generator() -> None:
    _spawn("conformal_field_strength")


def test_time_argument_is_detected_through_ddp() -> None:
    _spawn("time_arg")


def test_score_timesteps_are_gated_by_inner_forward_under_ddp() -> None:
    _spawn("score_timesteps")


def test_qmap_non_forward_render_refuses_ddp_with_grad() -> None:
    _spawn("qmap_render")


def test_graph_cold_timesteps_are_gated_by_inner_model_under_ddp() -> None:
    _spawn("graph_cold_timesteps")
