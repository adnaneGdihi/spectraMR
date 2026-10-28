"""Non-forward calls and hidden probes under DDP: raise paths and config-required reads.

Each scenario runs the production method under a real two-rank gloo DDP. Red before the fix,
green after: a config-required non-forward call on a wrapped generator raises on every rank, and a
probe reads the inner module."""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

WORLD = 2
# The backend the gloo run is under; the refusal is decided from it, as the audit decides.
_DDP_CONFIG = SimpleNamespace(parallel=SimpleNamespace(strategy="ddp"))


def _ddp(module):
    from torch.nn.parallel import DistributedDataParallel

    return DistributedDataParallel(module)


class _Linear(torch.nn.Module):
    """Parameter-bearing base: DDP refuses a module without trainable parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x, *args, **kwargs):
        return self.conv(x)


class _PadNet(_Linear):
    def predict_q_maps(self, x):
        return torch.zeros(x.shape[0], 3, *x.shape[-2:])


class _DiffNet(_Linear):
    n_steps = 10

    def bloch(self, x):
        return x


class _VAENet(_Linear):
    def encode_content(self, x):
        return x

    def decode(self, z, phys):
        return z

    def forward(self, x, phys):
        y = self.conv(x)
        return y, torch.zeros(x.shape[0], 2), torch.zeros(x.shape[0], 2)


class _CoordNet(_Linear):
    use_coordinate_sampling = True

    def __init__(self) -> None:
        super().__init__()
        self.received_query_coords = False

    def forward(self, x, query_coords=None, **kwargs):
        self.received_query_coords = query_coords is not None
        return self.conv(x)


class _Backbone(_Linear):
    receptive_half_width = 2


class _CompiledLike(torch.nn.Module):
    """Nests the model as torch.compile does (``_orig_mod``), so one ``.module`` hop is not enough."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self._orig_mod = inner

    def forward(self, x):
        return self._orig_mod(x)


class _KNet(_Linear):
    def forward(self, k_t, t):
        return self.conv(k_t)


def _check_padnet_predict_q_maps_refuses_wrapper(rank: int) -> None:
    from spectramr.domain.exceptions import ConfigurationError
    from spectramr.infrastructure.training.strategies.padnet_strategy import (
        PaDNetTrainingStrategy,
    )

    s = object.__new__(PaDNetTrainingStrategy)
    s.config = _DDP_CONFIG
    s.env = SimpleNamespace(generator=_ddp(_PadNet()))
    x = torch.randn(1, 1, 4, 4)
    with pytest.raises(ConfigurationError, match="outside the wrapped forward"):
        s._compute_losses_impl(x, x, 0, batch={"tr": 1.0, "te": 0.01})


def _check_disentangled_bloch_refuses_wrapper(rank: int) -> None:
    import torch.nn.functional as F

    from spectramr.domain.exceptions import ConfigurationError
    from spectramr.infrastructure.training.strategies.disentangled_diffusion_strategy import (
        DisentangledDiffusionStrategy,
    )

    s = object.__new__(DisentangledDiffusionStrategy)
    s.config = _DDP_CONFIG
    s.device = torch.device("cpu")
    s.env = SimpleNamespace(generator=_ddp(_DiffNet()), losses={})
    s._declare_model_input = lambda *a, **k: None
    s._noise_criterion = F.mse_loss
    s._lambda_noise = 1.0
    s._lambda_bloch_dc = 1.0
    s._bloch_dc_interval = 1
    s.loss_computer = SimpleNamespace(compute=lambda **k: SimpleNamespace(total=torch.tensor(0.5)))
    x = torch.rand(1, 1, 4, 4)
    with pytest.raises(ConfigurationError, match="outside the wrapped forward"):
        s.compute_generator_loss({"input": x}, iteration=0)


def _check_disentangled_vae_refuses_wrapper(rank: int) -> None:
    from spectramr.domain.exceptions import ConfigurationError
    from spectramr.infrastructure.training.strategies.disentangled_vae_strategy import (
        DisentangledVAETrainingStrategy,
    )

    s = object.__new__(DisentangledVAETrainingStrategy)
    s.config = _DDP_CONFIG
    s.env = SimpleNamespace(generator=_ddp(_VAENet()), losses={})
    x = torch.rand(1, 1, 4, 4)
    with pytest.raises(ConfigurationError, match="outside the wrapped forward"):
        s._compute_losses_impl(x, x, epoch=0)


def _check_physics_coordinate_sampling_reaches_wrapper(rank: int) -> None:
    from spectramr.infrastructure.training.strategies.physics_driven_strategy import (
        PhysicsDrivenTrainingStrategy,
    )

    net = _CoordNet()
    s = object.__new__(PhysicsDrivenTrainingStrategy)
    s.device = torch.device("cpu")
    s._b0_simulator = None
    s.env = SimpleNamespace(generator=_ddp(net))
    s._generate_predictions(torch.rand(1, 1, 4, 4), {}, {})
    assert net.received_query_coords, "use_coordinate_sampling was not seen through DDP"


def _check_dtn2s_backbone_declaration_seen_through_ddp(rank: int) -> None:
    from spectramr.infrastructure.training.strategies.dtn2s_strategy import DTN2SStrategy

    s = object.__new__(DTN2SStrategy)
    s.config = SimpleNamespace(training=SimpleNamespace(dtn2s=SimpleNamespace(receptive_window=8)))
    s._require_j_invariant_backbone(_ddp(_CompiledLike(_Backbone())))


def _check_cross_contrast_contrast_idx_gated_by_inner(rank: int) -> None:
    from spectramr.infrastructure.training.strategies.cross_contrast_kspace_diffusion_strategy import (
        CrossContrastKspaceDiffusionStrategy,
    )

    s = object.__new__(CrossContrastKspaceDiffusionStrategy)
    s.env = SimpleNamespace(generator=_ddp(_KNet()))
    s.forward_corrupt = lambda k_src, k_dst, t: k_src
    s.alpha_schedule = lambda t: torch.sigmoid(t)
    s.lambda_destination = 1.0
    s.lambda_residual = 1.0
    s._declare_model_input = lambda *a, **k: None
    k = torch.rand(1, 1, 4, 4)
    out = s._compute_losses_impl(None, None, 0, batch={"kspace_source": k, "kspace_target": k})
    assert "loss_total" in out


SCENARIOS = {
    "padnet_predict_q_maps": _check_padnet_predict_q_maps_refuses_wrapper,
    "disentangled_bloch": _check_disentangled_bloch_refuses_wrapper,
    "disentangled_vae": _check_disentangled_vae_refuses_wrapper,
    "physics_coordinates": _check_physics_coordinate_sampling_reaches_wrapper,
    "dtn2s_declaration": _check_dtn2s_backbone_declaration_seen_through_ddp,
    "cross_contrast_idx": _check_cross_contrast_contrast_idx_gated_by_inner,
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


def test_padnet_predict_q_maps_refuses_ddp() -> None:
    _spawn("padnet_predict_q_maps")


def test_disentangled_bloch_refuses_ddp() -> None:
    _spawn("disentangled_bloch")


def test_disentangled_vae_refuses_ddp() -> None:
    _spawn("disentangled_vae")


def test_physics_coordinate_sampling_reaches_ddp_wrapper() -> None:
    _spawn("physics_coordinates")


def test_dtn2s_backbone_declaration_is_seen_through_ddp() -> None:
    _spawn("dtn2s_declaration")


def test_cross_contrast_contrast_idx_is_gated_by_inner_under_ddp() -> None:
    _spawn("cross_contrast_idx")
