"""Where ``torch.compile`` may be applied, per parallel strategy.

One placement served every strategy, and it was wrong for most of them:
``ModelBuilder.compile()`` ran at director step 1, before every wrap. Nothing in
the corpus caught it because no arm combines compilation with a non-``none``
strategy — 5 compile-enabled arms, all single-device.

Measured on torch 2.14.0+cu126 / deepspeed 0.19.6:

============================================  =====================================
placement                                     outcome
============================================  =====================================
``DeepSpeedEngine(compile(m))``, ZeRO-1/2      works
``DeepSpeedEngine(compile(m))``, ZeRO-3        ``AttributeError: 'dict' object has
                                               no attribute '_in_forward'``
``engine.compile()`` after ``initialize``      ``InternalTorchDynamoError``
``DDP(compile(m))`` — what shipped             runs, and forfeits DDPOptimizer
============================================  =====================================

That last row is the expensive one. ``torch/_dynamo/backends/distributed.py``
states DDPOptimizer "applies when dynamo compiles models wrapped in
DistributedDataParallel", so overlapping allreduce with backward requires
``torch.compile(DDP(m))`` — the reverse of what shipped. HuggingFace Accelerate
reaches the same conclusion from the other direction: ``prepare_model`` applies
compilation last, after all distributed wrapping.

**This module holds the fact; the director holds the sequencing.** A hook the
base strategy invoked would be skipped by the four plugins that override
``adopt`` without calling ``super()`` — advertised and inert, which is the
failure mode this framework's audit found most often. A table the director reads
cannot be bypassed that way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "CompilePlacement",
    "CompileStage",
    "resolve_compile_placement",
]

#: The two points the director can compile at, named for the step they follow.
#: There is deliberately no "after_models": the plan carried one, and every row
#: that would have used it moved to ``after_stage_b`` so the optimizer keeps
#: seeing a bare module (#2174). A stage no table row returns is a branch in the
#: director that nothing can reach.
CompileStage = Literal["after_stage_a", "after_stage_b"]


@dataclass(frozen=True)
class CompilePlacement:
    """Where this strategy tolerates compilation, or why it does not."""

    stage: CompileStage | None
    refusal: str | None = None
    advisory: str | None = None
    #: True when the optimizer is built *after* compilation, so it introspects a
    #: wrapped module. Unavoidable for the strategies that must wrap before the
    #: optimizer exists; see the module note on #2174.
    optimizer_sees_wrapper: bool = False

    @property
    def refused(self) -> bool:
        return self.refusal is not None


_ZERO3 = 3

# ``deepspeed.initialize`` consumes a module and an optimizer, so anything
# compiled for it must already be compiled when Stage B runs. Stage A is a
# deliberate no-op for DeepSpeed, which makes ``after_stage_a`` the last point
# before the engine exists.
_DEEPSPEED_PRE_WRAP = "after_stage_a"


#: ZeRO stages DeepCompile has a pass for. ``z3`` rewrites the
#: parameter-partitioning collectives (stage 3); ``z1`` the optimizer-state
#: ones, which exist at stages 1 AND 2 -- see
#: ``DeepSpeedConfigSchema._deepcompile_passes_match_the_stage``. Stage 0 is
#: ZeRO off, so there are no collectives to rewrite and no pass applies.
_DEEPCOMPILE_STAGES = frozenset({1, 2, 3})


def _deepspeed(zero_stage: int | None) -> CompilePlacement:
    """DeepSpeed compiles through DeepCompile, not ``torch.compile``.

    ``torch.compile`` runs on the bare module before ``deepspeed.initialize``,
    so the ZeRO collectives are opaque calls the compiler never traces across --
    it optimises the arithmetic and leaves the communication alone, which on a
    sharded run is the part that costs. DeepCompile traces the engine graph
    *including* the allgather/reduce-scatter, which is the whole reason it
    exists. Wherever it has a pass, it is the route.

    **The two refusals are not the same claim, and the messages say so.** At
    ZeRO-3 ``torch.compile`` is a measured crash. At stages 1 and 2 it runs
    perfectly well -- it is simply the weaker of two available options, and
    silently taking the weaker one on 83 of 85 DeepSpeed arms is the kind of
    default nobody revisits.
    """
    if zero_stage == _ZERO3:
        return CompilePlacement(
            stage=None,
            refusal=(
                "optimization.compile.enabled=true with parallel.strategy='deepspeed' at "
                "zero_stage=3. Measured: the engine raises \"AttributeError: 'dict' object "
                "has no attribute '_in_forward'\" when initialize() receives an "
                "already-compiled module, and engine.compile() after initialize() raises "
                "inside dynamo on this stack. Use DeepCompile instead -- set "
                "parallel.deepspeed.compile.enabled: true with passes: [z3]."
            ),
        )
    if zero_stage in _DEEPCOMPILE_STAGES:
        return CompilePlacement(
            stage=None,
            refusal=(
                "optimization.compile.enabled=true with parallel.strategy='deepspeed' at "
                f"zero_stage={zero_stage}. This does run, unlike ZeRO-3 -- but it compiles "
                "the bare module before deepspeed.initialize, so the ZeRO collectives stay "
                "opaque to the compiler and the communication this arm shards for is left "
                "untouched. DeepCompile traces the engine graph including those collectives: "
                "set parallel.deepspeed.compile.enabled: true with passes: [z1], and "
                "optimization.compile.enabled: false. The two are alternatives, not layers."
            ),
        )
    # Stage 0 is ZeRO off: no collectives, so DeepCompile has no pass to apply
    # and torch.compile before initialize() is the only route. Measured working.
    return CompilePlacement(
        stage=_DEEPSPEED_PRE_WRAP,
        optimizer_sees_wrapper=True,
        advisory=(
            f"parallel.strategy='deepspeed' at zero_stage={zero_stage}: ZeRO is off, so "
            "DeepCompile has no pass to apply and torch.compile before initialize() is the "
            "route. Compiling before the wrap means the optimizer introspects a compiled "
            "module (#2174)."
        ),
    )


def _fsdp(_zero_stage: int | None) -> CompilePlacement:
    """``torch.compile(FSDP(m))`` — the order PyTorch recommends.

    FSDP must wrap in Stage A (it re-points parameter storage, so an optimizer
    built first would hold shard-shaped gradients against full-shape moments),
    which puts compilation after the wrap and before the optimizer.
    """
    return CompilePlacement(stage="after_stage_a", optimizer_sees_wrapper=True)


def _data_parallel(_zero_stage: int | None) -> CompilePlacement:
    """Allowed, with a warning: ``DataParallel`` re-replicates every forward.

    Not refused. It is not formally deprecated upstream and no arm uses it, so
    there is nothing to measure against -- stating the expectation is honest,
    refusing on an unmeasured expectation is not.
    """
    return CompilePlacement(
        stage="after_stage_b",
        advisory=(
            "parallel.strategy='dp' replicates the module on every forward, so compiled "
            "guards are re-evaluated against fresh replica objects. Compilation is unlikely "
            "to pay here; ddp is the supported multi-GPU path."
        ),
    )


def _after_wrap(_zero_stage: int | None) -> CompilePlacement:
    """Compile last, after every wrapper — the Accelerate order.

    Used by ``ddp`` (so dynamo sees the wrapper and DDPOptimizer engages) and by
    the single-device path, where it additionally keeps the optimizer looking at
    a bare module (#2174).
    """
    return CompilePlacement(stage="after_stage_b")


#: strategy -> placement resolver. A table rather than an ``if/elif`` chain, so a
#: new strategy that forgets to declare a placement fails a lookup instead of
#: silently inheriting one (non-negotiable 6).
_PLACEMENTS = {
    "none": _after_wrap,
    "ddp": _after_wrap,
    "dp": _data_parallel,
    "fsdp": _fsdp,
    "deepspeed": _deepspeed,
}


def resolve_compile_placement(
    strategy: str | None,
    *,
    zero_stage: int | None = None,
) -> CompilePlacement:
    """Where compilation goes for *strategy*.

    Args:
        strategy: ``parallel.strategy``; ``None`` means no ``parallel:`` block,
            which is the single-device path.
        zero_stage: ``parallel.deepspeed.zero_stage``; read only for DeepSpeed.

    Raises:
        ValueError: on a strategy with no declared placement. Silently defaulting
            would put a new backend on whatever placement happened to be first.
    """
    name = strategy or "none"
    resolver = _PLACEMENTS.get(name)
    if resolver is None:
        raise ValueError(
            f"No compile placement is declared for parallel.strategy={name!r}. "
            f"Known: {sorted(_PLACEMENTS)}. Add it to _PLACEMENTS with the "
            "measurement that justifies it."
        )
    return resolver(zero_stage)
