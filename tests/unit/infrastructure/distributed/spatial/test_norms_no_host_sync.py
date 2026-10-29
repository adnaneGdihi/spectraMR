"""The depth-split norms' forward makes no host sync (GPU only: CPU tensors never sync)."""

from __future__ import annotations

import os
import tempfile
import warnings

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and torch.distributed.is_nccl_available()),
        reason="sync-debug mode needs CUDA and NCCL",
    ),
]


def _worker(rank: int, store: str) -> None:
    import torch.distributed as dist

    from spectramr.infrastructure.distributed.spatial.convert import convert_to_spatial
    from spectramr.infrastructure.distributed.spatial.halo import spatial_scope

    device = torch.device("cuda:0")
    dist.init_process_group(
        "nccl", init_method=f"file://{store}", rank=0, world_size=1, device_id=device
    )
    try:
        net = nn.Sequential(
            nn.Conv3d(4, 4, 3, padding=1),
            nn.GroupNorm(2, 4),
            nn.InstanceNorm3d(4, affine=True),
            nn.BatchNorm3d(4),
            nn.AdaptiveAvgPool3d(1),
        ).to(device)
        convert_to_spatial(net, dist.group.WORLD)
        x = torch.randn(2, 4, 8, 8, 8, device=device)
        with spatial_scope():
            net(x)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("warn")
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    net(x)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        syncs = [str(w.message) for w in caught if "synchroniz" in str(w.message).lower()]
        syncs = [m for m in syncs if "prototype" not in m]
        assert not syncs, syncs
    finally:
        dist.destroy_process_group()


def test_spatial_norm_forward_makes_no_host_sync() -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(os.path.join(tmp, "store"),), nprocs=1, join=True)
