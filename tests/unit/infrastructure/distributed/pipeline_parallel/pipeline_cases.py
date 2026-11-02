"""Models, batches and the production-path harness for the pipeline gloo tests.

``tiny`` is a test net with a stage-0 parameter whose output rides in the payload
(as DiffVarNet's embedding does), complex k-space context, an unbatched context entry
and a real data-consistency step through ``fft2c``/``ifft2c``. ``nc_pdnet`` and
``unrolled`` are the registered models, small: absent payload entries, a shared
trajectory, NUFFT buffers, BatchNorm buffers and the shared forward operator.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import os
import tempfile
from types import SimpleNamespace

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.models.blocks.pipeline_stages import (
    CascadeStage,
    build_cascade_stages,
    cascade_at,
)

B, H = 4, 8
MAX_NORM = 0.05


class _DC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lam = nn.Parameter(torch.tensor([0.7]))  # FSDP2 refuses 0-d parameters

    def forward(self, est, kspace, mask):
        image = torch.complex(est[:, 0], est[:, 1]).unsqueeze(1)
        k = fft2c(image)
        image = ifft2c(k + self.lam * mask * (kspace - k)).squeeze(1)
        return torch.stack([image.real, image.imag], dim=1)


class _Scale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return self.scale * x


def _enter(owner, x, kspace, mask, weights):
    return (owner.entry(x), owner.gain(torch.ones(x.shape[0], 1, 1, 1)), kspace, mask, weights)


def _cascade(owner, cascades, payload):
    est, gain, kspace, mask, weights = payload
    for i in cascades:
        est = est + gain * cascade_at(owner.blocks, i)(est) * weights
        est = cascade_at(owner.dcs, i)(est, kspace, mask)
    return (est, *payload[1:])


class _Stage(CascadeStage):
    def _enter(self, x, measured_kspace, mask, weights):
        return _enter(self, x, measured_kspace, mask, weights)

    def _cascade(self, payload):
        return _cascade(self, self.cascades, payload)


class TinyUnrolled(nn.Module):
    def __init__(self, cascades: int = 4) -> None:
        super().__init__()
        self.entry = nn.Conv2d(2, 2, 1)
        self.gain = _Scale()
        self.blocks = nn.ModuleList(nn.Conv2d(2, 2, 3, padding=1) for _ in range(cascades))
        self.dcs = nn.ModuleList(_DC() for _ in range(cascades))

    def forward(self, x, measured_kspace, mask, weights):
        return _cascade(
            self, range(len(self.blocks)), _enter(self, x, measured_kspace, mask, weights)
        )[0]

    def pipeline_stages(self, n):
        def paths(cascades, first, _last):
            owned = [f"{g}.{i}" for i in cascades for g in ("blocks", "dcs")]
            return owned + (["entry", "gain"] if first else [])

        return build_cascade_stages(self, n, len(self.blocks), _Stage, paths)


def _tiny_batch(g, size):
    mask = (torch.rand(size, 1, H, H, generator=g) > 0.5).float()
    kwargs = {
        "measured_kspace": torch.randn(size, 1, H, H, generator=g, dtype=torch.complex64) * mask,
        "mask": mask,
        "weights": torch.linspace(0.5, 1.5, H),  # unbatched context
    }
    return torch.randn(size, 2, H, H, generator=g), kwargs, torch.randn(size, 2, H, H, generator=g)


def _nc_pdnet():
    from spectramr.models.generators.nc_pdnet import NCPDNet

    model = NCPDNet(in_channels=2, out_channels=2, image_size=16, num_cascades=4, width=4, depth=2)
    for reg in model.regularisers:  # the zero-init exit would make each cascade pure DC
        nn.init.normal_(reg.body[-1].weight, std=0.1)
    # The DC step scale is a power iteration from a global-RNG start, cached on first use:
    # fill the cache from a fixed seed so every copy of the model (and rank) agrees.
    x, kwargs, _ = _nc_pdnet_batch(torch.Generator().manual_seed(0), 1)
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(1)
        model(x, **kwargs)
    return model


def _nc_pdnet_batch(g, size):
    from spectramr.infrastructure.physics.nufft_ops import NUFFTForwardModel
    from spectramr.infrastructure.physics.trajectories import get_trajectory

    traj, dcf = get_trajectory("spiral", im_size=(16, 16))
    truth = torch.randn(size, 1, 16, 16, generator=g, dtype=torch.complex64)
    samples = NUFFTForwardModel(im_size=(16, 16)).forward_project(truth, traj)
    kwargs = {"measured_kspace": samples, "trajectory": traj, "dcf": dcf}
    return (
        torch.randn(size, 2, 16, 16, generator=g),
        kwargs,
        torch.randn(size, 2, 16, 16, generator=g),
    )


def _unrolled():
    from spectramr.models.generators.unrolled_reconstruction_generator import (
        UnrolledReconstructionGenerator,
    )

    return UnrolledReconstructionGenerator(
        in_channels=2, out_channels=2, img_size=(H, H), num_unrolls=4, features=4, num_coils=2
    )


def _unrolled_batch(g, size):
    maps = torch.randn(size, 2, H, H, generator=g, dtype=torch.complex64)
    maps = maps / maps.abs().pow(2).sum(1, keepdim=True).sqrt()
    mask = torch.zeros(size, 1, H, H)
    mask[..., ::2, :] = 1.0
    kspace = torch.randn(size, 2, H, H, generator=g, dtype=torch.complex64) * mask
    kwargs = {"kspace_measured": kspace, "mask": mask, "sensitivity_maps": maps}
    return torch.randn(size, 2, H, H, generator=g), kwargs, torch.rand(size, 2, H, H, generator=g)


CASES = {
    "tiny": (TinyUnrolled, _tiny_batch),
    "nc_pdnet": (_nc_pdnet, _nc_pdnet_batch),
    "unrolled": (_unrolled, _unrolled_batch),
}


def batch(case: str, step: int, shard: int | None = None, dp: int = 1, size: int = B):
    """The step's full batch, or data-parallel shard ``shard`` of it."""
    x, kwargs, target = CASES[case][1](torch.Generator().manual_seed(100 + step), size)
    if shard is None:
        return x, kwargs, target
    per = size // dp
    cut = lambda t: t[shard * per : (shard + 1) * per] if t.shape[0] == size else t  # noqa: E731
    return cut(x), {k: cut(v) for k, v in kwargs.items()}, cut(target)


def context(strategy: str, world: int, microbatches: int = 2, **fsdp):
    from spectramr.config.schemas.base import ParallelismConfigSchema
    from spectramr.infrastructure.distributed.mesh import build_parallel_mesh
    from spectramr.infrastructure.distributed.strategy_registry import ParallelContext

    extra = {"fsdp": {"enabled": True, **fsdp}} if strategy == "fsdp" else {}
    mesh = {"pipeline": 2, "pipeline_microbatches": microbatches}
    parallel = ParallelismConfigSchema(strategy=strategy, mesh=mesh, **extra)
    config = SimpleNamespace(
        optimization=SimpleNamespace(
            precision=SimpleNamespace(enabled=False, dtype=None),
            compile=SimpleNamespace(enabled=False),
            gradient=SimpleNamespace(accumulation_steps=1),
        ),
        ema=None,
    )
    built = build_parallel_mesh(parallel, device_type="cpu", local_world_size=world)
    return ParallelContext(config=config, device=torch.device("cpu"), parallel=parallel, mesh=built)


def build(ctx, case: str):
    """``(pipelined generator, unsplit reference, executor, checkpoint adapter, result)``."""
    from spectramr.infrastructure.distributed.strategy_registry import resolve_parallel_strategy
    from spectramr.infrastructure.training.step_executor import StepExecutor

    torch.manual_seed(0)
    model = CASES[case][0]()
    reference = copy.deepcopy(model)
    plugin = resolve_parallel_strategy(ctx.parallel.strategy)
    models = plugin.prepare_models({"generator": model}, ctx)
    result = plugin.adopt(models, {}, {}, ctx)
    executor = StepExecutor(
        SimpleNamespace(scaler=None, get_autocast_context=contextlib.nullcontext), None
    )
    executor.adopt_step_policy(result.step_policy(max_grad_norm=MAX_NORM))
    return models["generator"], reference, executor, plugin.checkpoint_adapter(ctx), result


def optimizers(params):
    """Two SGD then two Adam steps. A conv bias feeding BatchNorm has an exactly zero
    gradient, so its float noise differs between layouts; Adam's default eps would turn
    that noise into lr-sized steps, hence the larger eps."""
    params = list(params)
    sgd, adam = torch.optim.SGD(params, lr=0.5), torch.optim.Adam(params, lr=0.01, eps=1e-3)
    return [sgd, sgd, adam, adam]


def train(gen, executor, case, shard, dp, steps=range(4), opts=None):
    losses, opts = [], opts or optimizers(gen.parameters())
    for step in steps:
        x, kwargs, target = batch(case, step, shard, dp)

        def closure(x=x, kwargs=kwargs, target=target):
            return (gen(x, **kwargs) - target).abs().mean()

        record = executor.execute_step(
            {"optimizer": opts[step % len(opts)], "closure": closure, "model": gen, "name": "g"},
            epoch=0,
            global_step=step,
        )
        losses.append(float(record["g_loss"]))
    return losses, opts


def reference_train(reference, case, dp, steps=range(4), opts=None):
    """One process; the loss is the mean of the dp shards' means, as DDP averages."""
    norms, losses, opts = [], [], opts or optimizers(reference.parameters())
    per = B // dp
    for step in steps:
        x, kwargs, target = batch(case, step)
        out = reference(x, **kwargs)
        shard_losses = [
            (out[i * per : (i + 1) * per] - target[i * per : (i + 1) * per]).abs().mean()
            for i in range(dp)
        ]
        torch.stack(shard_losses).mean().backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(reference.parameters(), MAX_NORM)))
        opts[step % len(opts)].step()
        opts[step % len(opts)].zero_grad(set_to_none=True)
        losses.append([float(v) for v in shard_losses])
    return norms, losses, opts


def close(a, b) -> bool:
    return all(torch.allclose(a[k], b[k], atol=1e-5, rtol=1e-4) for k in b)


@contextlib.contextmanager
def patched(patches):
    saved = [(owner, name, owner.__dict__[name]) for owner, name, _ in patches]
    try:
        for owner, name, value in patches:
            setattr(owner, name, value)
        yield
    finally:
        for owner, name, value in saved:
            setattr(owner, name, value)


def _worker(rank: int, world: int, store: str, target: str, args: tuple) -> None:
    import torch.distributed as dist

    os.environ.setdefault("SPECTRAMR_SUPPRESS_CLINICAL_WARNING", "1")
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    try:
        torch.set_num_threads(1)
        module, name = target.rsplit(":", 1)
        getattr(importlib.import_module(module), name)(rank, *args)
    finally:
        from spectramr.core.parallel_groups import reset_dp_layout

        reset_dp_layout()
        dist.destroy_process_group()


def spawn(world: int, target: str, *args) -> None:
    """Run ``module:function(rank, *args)`` on ``world`` gloo ranks."""
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        store = os.path.join(tmp, "store")
        mp.spawn(_worker, args=(world, store, target, args), nprocs=world, join=True)
