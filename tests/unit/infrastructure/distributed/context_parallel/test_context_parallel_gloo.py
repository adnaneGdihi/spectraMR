"""Context-parallel attention, the equivalence gate and its planted violations (2 gloo ranks).

The backbones are the real ``DiTBackbone`` / ``UViTBackbone`` at toy sizes, so the
plan lookup by backbone class is exercised. Each planted defect must make the gate
raise on every rank; a gate that stays green under them would prove nothing.
"""

from __future__ import annotations

import copy
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

pytestmark = [pytest.mark.unit]


def tiny_dit():
    from spectramr.models.generators.dit_backbone import DiTBackbone

    return DiTBackbone(
        in_channels=2, out_channels=2, image_size=8, dim=16, depth=2, heads=4, patch_size=2
    )


def tiny_uvit():
    from spectramr.models.generators.uvit_backbone import UViTBackbone

    # A 3x3 grid: 9 patch tokens plus the time token divide by cp=2.
    return UViTBackbone(
        in_channels=2, out_channels=2, image_size=6, dim=16, depth=3, heads=4, patch_size=2
    )


class Generator(nn.Module):
    """Stands in for the cold-diffusion generator that holds the backbone."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x, t):
        return self.backbone(x, t)


def seeded(make=tiny_dit, seed: int = 0) -> Generator:
    """Identical on every rank, with the zero-initialised gates randomised."""
    torch.manual_seed(seed)
    model = Generator(make())
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if not p.any():
                p.copy_(0.2 * torch.randn(p.shape, generator=gen))
    return model


def _scenario_attention_mask(rank: int) -> None:
    from spectramr.infrastructure.distributed.context_parallel.attention import (
        ContextParallelAttention,
        local_mask_rows,
    )
    from spectramr.models.generators.diffusion_transformer_stem import TokenAttention

    group = torch.distributed.group.WORLD
    torch.manual_seed(0)
    reference = TokenAttention(16, 4)
    split = ContextParallelAttention.adopt(copy.deepcopy(reference), group)
    x = torch.randn(2, 8, 16)
    causal = torch.ones(8, 8, dtype=torch.bool).tril()
    rows = slice(rank * 4, rank * 4 + 4)
    torch.testing.assert_close(local_mask_rows(causal, 4, group), causal[rows])
    torch.testing.assert_close(split(x[:, rows], attn_mask=causal), reference(x, causal)[:, rows])
    assert local_mask_rows(causal[:1], 4, group).shape[0] == 1  # broadcast rows pass through
    with pytest.raises(ValueError, match="query rows"):
        local_mask_rows(causal[:6], 4, group)
    assert list(split.state_dict()) == list(reference.state_dict())


def _scenario_gate_and_checkpoint(rank: int) -> None:
    from spectramr.infrastructure.distributed.context_parallel import parallelize_context

    group = torch.distributed.group.WORLD
    for make, attentions in ((tiny_dit, 2), (tiny_uvit, 3)):
        model = seeded(make)
        reference = copy.deepcopy(model)
        shapes = {k: v.shape for k, v in model.state_dict().items()}
        plan = parallelize_context(model, group)
        assert (plan.size, plan.backbones, plan.attentions) == (2, ("backbone",), attentions)
        assert plan.worst_relative_error < 1e-4
        assert {k: v.shape for k, v in model.state_dict().items()} == shapes
        side = 8 if make is tiny_dit else 6
        x, t = torch.randn(2, 2, side, side), torch.rand(2)
        torch.testing.assert_close(model(x, t), reference(x, t), atol=1e-5, rtol=1e-4)
        unsplit = make()
        unsplit.load_state_dict(model.backbone.state_dict(), strict=True)
        torch.testing.assert_close(unsplit(x, t), reference(x, t), atol=1e-5, rtol=1e-4)


def _expect_gate_failure(match: str, model=None) -> None:
    from spectramr.infrastructure.distributed.context_parallel import (
        ContextParallelError,
        parallelize_context,
    )

    with pytest.raises(ContextParallelError, match=match):
        parallelize_context(model if model is not None else seeded(), torch.distributed.group.WORLD)


def _scenario_planted(rank: int) -> None:
    from spectramr.infrastructure.distributed.context_parallel import attention, comms

    # 1. Keys/values gathered with no reduce-scatter: each rank keeps only its own slice.
    original_kv = comms._GatherKeysValues.backward
    comms._GatherKeysValues.backward = staticmethod(
        lambda ctx, grad: (comms._local_rows(grad, ctx.group), None)
    )
    try:
        _expect_gate_failure("attention input gradient")
    finally:
        comms._GatherKeysValues.backward = original_kv

    # 2. The trunk's output gather without the x cp: trunk gradients come out 1/cp.
    original_seq = comms._GatherSequence.backward
    comms._GatherSequence.backward = staticmethod(
        lambda ctx, grad: (comms._local_rows(grad, ctx.group), None)
    )
    try:
        _expect_gate_failure(r"input gradient \(cp mean\)")
    finally:
        comms._GatherSequence.backward = original_seq

    # 3. A mask whose rows are not offset by this rank's position.
    original_rows = attention.local_mask_rows
    attention.local_mask_rows = lambda mask, local, group: (
        None if mask is None else mask.narrow(-2, 0, local)
    )
    try:
        # Rank 0's rows happen to be right; its keys still get rank 1's wrong gradients.
        _expect_gate_failure("masked attention output" if rank else "attention input gradient")
    finally:
        attention.local_mask_rows = original_rows

    # 4. A trunk that mixes tokens outside attention (a token-mean inside a block).
    model = seeded()
    model.backbone.blocks[0].register_forward_hook(
        lambda _m, _a, out: out - out.mean(dim=1, keepdim=True)
    )
    _expect_gate_failure("output", model)


def _scenario_refusals(rank: int) -> None:
    from spectramr.infrastructure.distributed.context_parallel import (
        ContextParallelError,
        parallelize_context,
    )
    from spectramr.models.generators.hat_backbone import HATBackbone

    group = torch.distributed.group.WORLD
    with pytest.raises(ContextParallelError, match="no context-parallel plan"):
        parallelize_context(
            Generator(HATBackbone(in_channels=2, out_channels=2, dim=8, depth=1, heads=2)), group
        )
    _expect_gate_failure("weights differ", seeded(seed=rank))  # rank-dependent init

    model = seeded()
    parallelize_context(model, group)
    with pytest.raises(ValueError, match="divides by"):
        model(torch.randn(1, 2, 6, 6), torch.rand(1))  # 9 tokens over 2 ranks
    model.backbone.set_grad_checkpointing(True)
    with pytest.raises(ContextParallelError, match="gradient checkpointing"):
        model(torch.randn(1, 2, 8, 8), torch.rand(1))
    model.backbone.set_grad_checkpointing(False)

    fresh = seeded()
    parallelize_context(fresh, group)
    torch.manual_seed(rank)  # planted: per-rank noise, as a global-rank stream would draw
    with pytest.raises(ContextParallelError, match="different inputs"):
        fresh(torch.randn(1, 2, 8, 8), torch.rand(1))


def _worker(rank: int, world: int, store: str, scenario: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    try:
        torch.set_num_threads(1)
        globals()[f"_scenario_{scenario}"](rank)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "scenario", ["attention_mask", "gate_and_checkpoint", "planted", "refusals"]
)
def test_context_parallel_two_ranks(scenario: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(2, os.path.join(tmp, "store"), scenario), nprocs=2, join=True)
