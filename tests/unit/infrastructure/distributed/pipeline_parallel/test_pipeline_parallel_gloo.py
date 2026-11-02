"""Pipeline parallelism on real gloo ranks (CPU) against one process.

Every case goes through the production path: ``build_parallel_mesh`` -> the ddp or
fsdp plugin's ``prepare_models``/``adopt`` -> the step policy it hands back, adopted
by a real ``StepExecutor`` -> ``clip_grad_norm_global_`` -> the plugin's checkpoint
adapter. The clip threshold is below the reference norm so clipping binds; the
planted variants (loss back-propagated on the wrong stage, no gradient sync over dp,
a per-stage clip norm, peers on different batches) must each diverge or raise.
Models and harness: ``pipeline_cases``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tests.unit.infrastructure.distributed.pipeline_parallel import (  # noqa: E402
    pipeline_cases as pc,
)

pytestmark = [pytest.mark.unit]
_HERE = __name__


def _scenario_parity(rank, strategy, world, case, microbatches, fsdp) -> None:
    from spectramr.core.parallel_groups import dp_rank, dp_world_size

    ctx = pc.context(strategy, world, microbatches, **fsdp)
    gen, reference, executor, adapter, result = pc.build(ctx, case)
    dp = dp_world_size()
    assert result.provenance["pipeline_parallel"] == {
        "size": 2,
        "microbatches": microbatches,
        "schedule": "gpipe_fill_drain",
        "last_stage_backward": "full_batch",
        "short_batch": "gcd(batch, microbatches) microbatches",
        "cascades_per_stage": [2, 2],
        "grad_sync": "fsdp2" if strategy == "fsdp" else "allreduce",
    }
    assert torch.distributed.get_backend(gen.meta_group) == "gloo"
    losses, opts = pc.train(gen, executor, case, dp_rank(), dp)
    norms, ref_losses, ref_opts = pc.reference_train(reference, case, dp)
    assert min(norms) > pc.MAX_NORM, "clipping must bind or the global norm is untested"
    # Every stage reports the loss of its own dp shard, so rank 0 (stage 0) can log it.
    torch.testing.assert_close(
        torch.tensor(losses), torch.tensor([row[dp_rank()] for row in ref_losses])
    )

    io = adapter.state_io
    state = io.model_state(gen)
    if rank == 0:
        assert sorted(state) == sorted(reference.state_dict()), "keys must be the unsplit ones"
        assert pc.close(state, reference.state_dict()), "pipelined weights differ from one process"
        unsplit = pc.CASES[case][0]()
        unsplit.load_state_dict(state, strict=True)  # what `spectramr infer` does
        x, kwargs, _ = pc.batch(case, 9)
        unsplit.eval()
        reference.eval()  # a train-mode forward would move BatchNorm statistics on rank 0 only
        torch.testing.assert_close(unsplit(x, **kwargs), reference(x, **kwargs))
        reference.train()

    # Validation: a no-grad forward on every rank returns the whole output, also for a
    # batch the microbatch count does not divide.
    gen.eval()
    reference.eval()
    with torch.no_grad():
        for size in (pc.B, 3):
            x, kwargs, _ = pc.batch(case, 7, size=size)
            torch.testing.assert_close(gen(x, **kwargs), reference(x, **kwargs))
    gen.train()
    reference.train()
    _check_metric_counts(dp)

    # Resume: weights and optimizer state, as every rank reads them from the file.
    files = [state, io.optimizer_state(gen, opts[-1])]
    torch.distributed.broadcast_object_list(files, src=0)
    fresh, _, fresh_executor, _, _ = pc.build(ctx, case)
    fresh_opts = pc.optimizers(fresh.parameters())
    io.load_model_state(fresh, files[0])
    io.load_optimizer_state(fresh, fresh_opts[-1], files[1])
    pc.train(fresh, fresh_executor, case, dp_rank(), dp, steps=[3], opts=fresh_opts)
    pc.reference_train(reference, case, dp, steps=[3], opts=ref_opts)
    resumed = io.model_state(fresh)
    if rank == 0:
        assert pc.close(resumed, reference.state_dict()), "resume diverged from one process"


def _check_metric_counts(dp: int) -> None:
    """Validation sums reduce over dp only: pipeline peers must not count a sample twice."""
    from spectramr.pipelines.train import _all_reduce_val_metrics

    per = float(pc.B // dp)
    one = {"psnr": per}
    _, _, _, weight = _all_reduce_val_metrics(one, one, one, per, torch.device("cpu"))
    assert int(weight) == pc.B, f"{weight} samples counted for a batch of {pc.B}"


def _planted_state(ctx, rank: int) -> None:
    from spectramr.core.parallel_groups import dp_rank, dp_world_size

    gen, reference, executor, adapter, _ = pc.build(ctx, "tiny")
    pc.train(gen, executor, "tiny", dp_rank(), dp_world_size())
    pc.reference_train(reference, "tiny", dp_world_size())
    state = adapter.state_io.model_state(gen)
    if rank == 0:
        assert not pc.close(state, reference.state_dict()), "the planted defect went unnoticed"


def _scenario_planted(rank: int, world: int) -> None:
    from spectramr.infrastructure.distributed import grad_norm
    from spectramr.infrastructure.distributed.pipeline_parallel import step_policy
    from spectramr.infrastructure.training.optimizers.amp_policy import AMPPolicy

    ctx = pc.context("ddp", world)
    wrong_stage = [
        # 1. The loss is back-propagated on the first stage, whose output is a detached copy.
        (
            step_policy.PipelineStepPolicy,
            "_holds_the_loss_graph",
            staticmethod(lambda p: p.is_first),
        ),
        (AMPPolicy, "_ensure_gradients_arrived", lambda *a, **k: None),
    ]
    with pc.patched(wrong_stage):
        _planted_state(ctx, rank)
    # 2. The dp replicas never average their gradients.
    with pc.patched([(step_policy, "average_replica_gradients", lambda _p: None)]):
        _planted_state(ctx, rank)
    # 3. Each stage clips by its own norm instead of the norm over every stage.
    per_stage = lambda root, max_norm, _group: torch.nn.utils.clip_grad_norm_(  # noqa: E731
        [p for p in root.parameters() if p.grad is not None], max_norm
    )
    with pc.patched([(grad_norm, "_clip_over_stages", per_stage)]):
        _planted_state(ctx, rank)
    # 4. Pipeline peers that drew different batches are refused before training.
    gen = pc.build(ctx, "tiny")[0]
    x, kwargs, _ = pc.batch("tiny", rank)  # a different batch on every rank
    with pytest.raises(RuntimeError, match="different batches"):
        gen(x, **kwargs)


def test_pp2_matches_one_process() -> None:
    pc.spawn(2, f"{_HERE}:_scenario_parity", "ddp", 2, "tiny", 2, {})


@pytest.mark.parametrize(
    "fsdp",
    [{}, {"auto_wrap_policy": "size_based", "min_num_params": 1}],
    ids=["root_unit", "every_module_a_unit"],
)
def test_dp2_x_pp2_fsdp_matches_one_process(fsdp) -> None:
    pc.spawn(4, f"{_HERE}:_scenario_parity", "fsdp", 4, "tiny", 2, fsdp)


def test_dp2_x_pp2_ddp_matches_one_process() -> None:
    pc.spawn(4, f"{_HERE}:_scenario_parity", "ddp", 4, "tiny", 2, {})


@pytest.mark.parametrize(("case", "microbatches"), [("nc_pdnet", 2), ("unrolled", 1)])
def test_registered_models_pp2_match_one_process(case: str, microbatches: int) -> None:
    pc.spawn(2, f"{_HERE}:_scenario_parity", "ddp", 2, case, microbatches, {})


def test_planted_pipeline_defects_diverge() -> None:
    pc.spawn(4, f"{_HERE}:_scenario_planted", 4)


_SLEEP = 0.05


def _cascade_windows(gen, x, kwargs) -> list[tuple[float, float]]:
    """``(start, end)`` of each cascade this stage runs during one no-grad forward."""
    import time

    stage, windows = gen.stage, []
    forward = type(stage).forward

    def timed(phase, *args, **kw):
        if phase != "cascade":
            return forward(stage, phase, *args, **kw)
        start = time.time()
        time.sleep(_SLEEP)
        out = forward(stage, phase, *args, **kw)
        windows.append((start, time.time()))
        return out

    stage.forward = timed
    try:
        with torch.no_grad():
            gen(x, **kwargs)
    finally:
        del stage.forward
    return windows


def _post_after_every_cascade(self, microbatches, batched, count):
    """The defect this file plants: every cascade first, then one burst of sends."""
    from spectramr.infrastructure.distributed.pipeline_parallel import transport

    outs = [self.stage("cascade", *mb) for mb in microbatches]
    if not self.is_last:
        dst, device = self.pp_ranks[self.pp_index + 1], self._device()
        posted = [transport.Posted(transport.encode_header(outs[0], batched, count, device),
                                   dst, self.meta_group)]  # fmt: skip
        for out in outs:
            posted += transport.send_payload(out, dst, self.group, self.meta_group, device)
        for work in posted:
            work.wait()
    return outs


def _scenario_overlap(rank: int, world: int) -> None:
    """Stage 1 starts microbatch 0 before stage 0 finishes its last microbatch."""
    from spectramr.infrastructure.distributed.pipeline_parallel.module import PipelinedGenerator

    ctx = pc.context("ddp", world, microbatches=4)
    gen = pc.build(ctx, "tiny")[0]
    gen.eval()
    x, kwargs, _ = pc.batch("tiny", 0)

    def overlap() -> float:
        windows = [None] * world
        torch.distributed.all_gather_object(windows, _cascade_windows(gen, x, kwargs))
        assert [len(w) for w in windows] == [4, 4]
        return windows[0][-1][1] - windows[1][0][0]  # > 0: the stages ran concurrently

    assert overlap() > _SLEEP, "stage 1 waited for stage 0's whole batch"
    with pc.patched([(PipelinedGenerator, "_cascade_and_post", _post_after_every_cascade)]):
        assert overlap() < 0, "the planted send-after-everything schedule went unnoticed"


def _scenario_replica_average(rank: int, world: int) -> None:
    """Complex and real gradients average over replicas without losing an imaginary part."""
    from types import SimpleNamespace

    from spectramr.infrastructure.distributed.pipeline_parallel.step_policy import (
        average_replica_gradients,
    )

    stage = torch.nn.Module()
    stage.real = torch.nn.Parameter(torch.zeros(3))
    stage.cplx = torch.nn.Parameter(torch.zeros(3, dtype=torch.complex64))
    stage.real.grad = torch.full((3,), float(rank))
    stage.cplx.grad = torch.full((3,), complex(rank, 2 * rank), dtype=torch.complex64)
    average_replica_gradients(SimpleNamespace(replica_group=torch.distributed.group.WORLD,
                                              stage=stage))  # fmt: skip
    mean = sum(range(world)) / world
    torch.testing.assert_close(stage.real.grad, torch.full((3,), mean))
    torch.testing.assert_close(
        stage.cplx.grad, torch.full((3,), complex(mean, 2 * mean), dtype=torch.complex64)
    )


def _scenario_host_entries(rank: int, world: int) -> None:
    """A host entry (the unbatched ``weights``) crosses the gloo channel and stays on the
    host, as DiffVarNet's ``restore`` does beside a GPU stage; training matches one process."""
    from spectramr.infrastructure.distributed.pipeline_parallel import transport

    routed = []

    def host_weights(entry, _device):
        host = entry.ndim == 1 and not entry.requires_grad
        routed.append(host)
        return host

    with pc.patched([(transport, "is_host", host_weights)]):
        gen, reference, executor, _, _ = pc.build(pc.context("ddp", world), "tiny")
        pc.train(gen, executor, "tiny", None, 1)
        pc.reference_train(reference, "tiny", 1)
        mine, ref = gen.stage.state_dict(), reference.state_dict()
        state = pc.close(mine, {k: ref[k] for k in mine})
    if gen.is_first:  # the receiver reads the routing from the header, not is_host
        assert any(routed), "no entry took the host channel"
    assert state, "a host entry changed the pipelined result"


def test_host_entries_cross_the_gloo_channel() -> None:
    pc.spawn(2, f"{_HERE}:_scenario_host_entries", 2)


def test_stages_overlap_across_microbatches() -> None:
    pc.spawn(2, f"{_HERE}:_scenario_overlap", 2)


def test_replica_average_keeps_complex_gradients() -> None:
    pc.spawn(2, f"{_HERE}:_scenario_replica_average", 2)
