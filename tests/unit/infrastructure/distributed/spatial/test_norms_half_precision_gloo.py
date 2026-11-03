"""The depth-split norms keep their statistics in float32 for a half-precision slab (gloo).

An fp16 sum over a volume overflows at 65504: these inputs (mean 100, 8x16x16 voxels per
slab) push every slab sum past it. Each norm must still match the native layer run in
float32 on the whole volume, return the slab's own dtype, and keep float32 running stats.
"""

from __future__ import annotations

import copy
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

pytestmark = [pytest.mark.unit]
WORLD = 2


def _norms():
    return [
        nn.GroupNorm(2, 4),
        nn.InstanceNorm3d(4, affine=True),
        nn.BatchNorm3d(4),
        nn.AdaptiveAvgPool3d(1),
        nn.GroupNorm(2, 4).half(),  # FSDP fp16 parameters, no autocast
    ]


def _scenario(rank: int) -> None:
    import torch.distributed as dist

    from spectramr.infrastructure.distributed.spatial.convert import convert_to_spatial
    from spectramr.infrastructure.distributed.spatial.halo import spatial_scope

    group = dist.group.WORLD
    full = 100 + torch.randn(1, 4, 8 * WORLD, 16, 16, generator=torch.Generator().manual_seed(0))
    full = full.half().float()  # the reference sees the same fp16-rounded volume
    slab = full[:, :, rank * 8 : (rank + 1) * 8]
    assert float(slab[:, :2].sum()) > 65504, "the slab sum must overflow fp16"
    for native in _norms():
        native.train()
        split = copy.deepcopy(native)
        convert_to_spatial(nn.Sequential(split), group)
        with spatial_scope():
            got = split(slab.half())
        expected = copy.deepcopy(native).float()(full)
        if expected.shape[2] == full.shape[2]:
            expected = expected[:, :, rank * 8 : (rank + 1) * 8]
        name = type(split).__name__
        assert torch.isfinite(got).all(), f"{name} overflowed in fp16"
        assert got.dtype == torch.float16, f"{name} returned {got.dtype}"
        torch.testing.assert_close(got.float(), expected, atol=2e-2, rtol=1e-2, msg=name)
        if isinstance(split, nn.BatchNorm3d):
            assert split.running_mean.dtype == torch.float32
            reference = copy.deepcopy(native)
            reference(full)
            torch.testing.assert_close(split.running_mean, reference.running_mean)
            torch.testing.assert_close(split.running_var, reference.running_var, rtol=1e-3, atol=0)


def _worker(rank: int, store: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        torch.set_num_threads(1)
        _scenario(rank)
    finally:
        dist.destroy_process_group()


def test_spatial_norms_take_half_slabs_in_float32() -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(os.path.join(tmp, "store"),), nprocs=WORLD, join=True)
