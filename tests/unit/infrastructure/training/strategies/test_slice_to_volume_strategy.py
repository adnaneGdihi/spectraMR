"""``SliceToVolumeStrategy``: the loss-computer iteration seam.

Paired with ``infrastructure/training/strategies/slice_to_volume_strategy.py``
(non-negotiable 10).

``_compute_losses_impl`` threads an ``iteration`` into
``UnifiedReconstructionLossComputer.compute``, which is what advances every
iteration-dependent schedule the computer owns -- the warm-up gate above all, where
``resolve_loss_weight`` returns ``0.0`` while ``iteration < warmup_iterations``. It
read ``int(kwargs.get("step", 0))``, but the training loop passes ``iteration=`` and
never ``step=``, so the value was a constant ``0``: the gate stayed shut for the whole
run and every warm-up-gated loss was **absent from ``components``**, not scaled to
zero (pitfall #16, #1937).

No ``experiments/inprogress/`` arm resolves to this strategy today, so the defect was
latent here -- it is live on ``vf_admm_strategy.py``, which carried the identical read.
That is precisely why it needs a test: a latent defect has no run to reveal it.

The spy below **observes the value arrive** at ``compute`` (non-negotiable 16) and
probes with an iteration that is neither ``0`` nor a schema default.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from spectramr.infrastructure.training.loop_state import LoopState
from spectramr.infrastructure.training.strategies.slice_to_volume_strategy import (
    SliceToVolumeStrategy,
)

#: Past the schema-default ``warmup_iterations`` of 1000, and not 0.
_LIVE_ITERATION = 1337


class _SpyComputer:
    """Records the kwargs ``_compute_losses_impl`` hands the real computer."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def compute(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            total=torch.zeros((), requires_grad=True),
            components={"l1": torch.zeros(())},
        )


def _s2v(iteration: int) -> tuple[Any, _SpyComputer]:
    """A strategy stitched with only what ``_compute_losses_impl`` touches."""
    s = SliceToVolumeStrategy.__new__(SliceToVolumeStrategy)
    spy = _SpyComputer()
    s.loss_computer = spy
    s.loop_state = LoopState(iteration=iteration, epoch=3)
    s.env = SimpleNamespace(generator=lambda x: x, losses={"l1": {"weight": 1.0}})
    s.lambda_through_plane = 0.1
    s.lambda_ortho = 0.05
    return s, spy


#: (B, C, X, Y, Z) -- both helpers this hook calls reject anything but 5-D.
def _vol() -> torch.Tensor:
    return torch.rand(1, 1, 4, 8, 8)


#: A replicated stack needs the slice_profile stamp (ratio 2 along TorchIO z).
_STAMP = {"slice_profile_ratio": [2], "slice_profile_z_axis": [3]}


@pytest.mark.unit
def test_loss_computer_receives_the_live_iteration_not_a_frozen_zero() -> None:
    strategy, spy = _s2v(_LIVE_ITERATION)
    strategy._compute_losses_impl(input_batch=_vol(), target_batch=_vol(), epoch=3, batch=_STAMP)
    assert len(spy.calls) == 1
    assert spy.calls[0]["iteration"] == _LIVE_ITERATION, (
        "the loss computer must receive the live loop iteration; frozen at 0 the "
        "warm-up gate never opens and a gated loss never enters components "
        "(pitfall #16, #1937)"
    )


@pytest.mark.unit
def test_a_stale_step_kwarg_does_not_win_over_the_loop_state() -> None:
    """The loop passes ``iteration=``; a caller passing ``step=`` must not steer it.

    This is the shape the fix retires: had the read stayed on ``kwargs``, a caller
    supplying ``step=0`` would still freeze the schedule.
    """
    strategy, spy = _s2v(_LIVE_ITERATION)
    strategy._compute_losses_impl(
        input_batch=_vol(), target_batch=_vol(), epoch=3, step=0, batch=_STAMP
    )
    assert spy.calls[0]["iteration"] == _LIVE_ITERATION


@pytest.mark.unit
def test_the_iteration_advances_with_the_loop() -> None:
    """Two steps, two values -- a constant would satisfy a single-point assertion."""
    strategy, spy = _s2v(0)
    strategy._compute_losses_impl(input_batch=_vol(), target_batch=_vol(), epoch=0, batch=_STAMP)
    strategy.loop_state.iteration = _LIVE_ITERATION
    strategy._compute_losses_impl(input_batch=_vol(), target_batch=_vol(), epoch=0, batch=_STAMP)
    assert [c["iteration"] for c in spy.calls] == [0, _LIVE_ITERATION]


@pytest.mark.unit
def test_the_hook_still_returns_its_own_anchor_terms() -> None:
    """Guard the edit's blast radius: the two bespoke terms survive the change."""
    strategy, _ = _s2v(_LIVE_ITERATION)
    out = strategy._compute_losses_impl(
        input_batch=_vol(), target_batch=_vol(), epoch=3, batch=_STAMP
    )
    assert {"g_total_loss", "g_through_plane_consistency", "g_isotropy_consistency"} <= set(out)


# --- the thick-slice anchor, driven through the production path ------------------

_ARM = "experiments/inprogress/spec_tasks/task_2_7_slice_to_volume_oasis.yaml"


def _arm_patches() -> tuple[Any, list[Any]]:
    """Arm config -> the real train chain -> torchio patches at two different z phases."""
    from pathlib import Path

    import torchio as tio

    from spectramr.config.settings import TrainingSettings
    from spectramr.data.builders.torchio_transform_builder import (
        TorchIOTransformBuilder,
        TorchIOTransformConfig,
    )

    settings = TrainingSettings.from_yaml(str(Path(__file__).resolve().parents[5] / _ARM))
    chain = TorchIOTransformBuilder.build_train_transforms(
        TorchIOTransformConfig.from_settings(settings)
    )
    # Full patch_size in-plane, so _EnsureMinimumSpatialSize pads nothing.
    vol = 100 * torch.rand(1, 128, 128, 32, generator=torch.Generator().manual_seed(0))
    subject = chain(
        tio.Subject(input=tio.ScalarImage(tensor=vol), target=tio.ScalarImage(tensor=vol.clone()))
    )
    torch.manual_seed(0)
    by_phase: dict[int, Any] = {}
    for patch in tio.UniformSampler((16, 16, 16))(subject, num_patches=64):
        by_phase.setdefault(int(patch["location"][2]) % 4, patch)
    return settings, [by_phase[p] for p in sorted(by_phase)[:2]]


def _collate(settings: Any, patches: list[Any]) -> Any:
    """The collate strategy the train queue selects for this arm, then the loop's adapter."""
    from spectramr.data.batch_types import BatchAdapter
    from spectramr.data.collation.strategies import CollateStrategyFactory
    from spectramr.data.collation.strategy_selector import CollationStrategySelector

    name, kwargs = CollationStrategySelector.select_strategy(
        config=settings.data.collation,
        dataset_type=settings.data.dataset_type,
        enable_slab_mode=False,
        patch_size=tuple(settings.data.sampling.patch_size),
    )
    collated = CollateStrategyFactory.create(name, **kwargs).collate(patches)
    return BatchAdapter.from_dict(collated)


@pytest.mark.unit
def test_the_arm_anchor_pools_each_patch_back_onto_its_own_thick_slices() -> None:
    """Identity on a replicated stack matches every thick slice exactly -- at any phase.

    A pool that ignored the patch z-start (or pooled along X, dim 2) would see the
    staircase straddle its windows and report a non-zero anchor.
    """
    from spectramr.infrastructure.training.strategies.slice_to_volume_strategy import (
        read_slice_profile_stamp,
        replicated_stack_anchor,
    )
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    settings, patches = _arm_patches()
    assert TrainingStrategyFactory().get_strategy_class(settings) is SliceToVolumeStrategy
    batch = _collate(settings, patches)
    assert batch.metadata["slice_profile_ratio"] == [4, 4]  # host-side ints, no sync
    assert len({int(k) % 4 for k in batch.metadata["location"][:, 2]}) == 2  # two phases
    assert not torch.equal(batch.input, batch.target)  # only the input was degraded

    strategy, _ = _s2v(_LIVE_ITERATION)
    out = strategy._compute_losses_impl(
        input_batch=batch.input, target_batch=batch.target, epoch=0, batch=batch
    )
    assert out["g_through_plane_consistency"].item() == pytest.approx(0.0, abs=1e-12)

    stamp = read_slice_profile_stamp(batch)
    assert (stamp.ratio, stamp.batch_dim) == (4, 4)
    # Control: the same pool with the patch z-start ignored straddles the blocks.
    unaligned = type(stamp)(ratio=4, batch_dim=4, z_start=torch.zeros_like(stamp.z_start))
    assert replicated_stack_anchor(batch.input, batch.input, unaligned).item() > 1e-4
    toward_target = strategy._through_plane_consistency(batch.target, batch.input, stamp)
    assert toward_target.item() > 1e-4  # the isotropic volume is not its own thick slices


@pytest.mark.unit
def test_a_ratio_of_one_is_refused() -> None:
    from spectramr.infrastructure.training.strategies.slice_to_volume_strategy import (
        SliceProfileStamp,
        replicated_stack_anchor,
    )

    x = torch.rand(1, 1, 4, 4, 8)
    with pytest.raises(ValueError, match="identity MSE"):
        replicated_stack_anchor(x, x, SliceProfileStamp(ratio=1, batch_dim=4, z_start=None))


@pytest.mark.unit
def test_the_isotropy_prior_follows_the_stamped_axis() -> None:
    """Blur along Z (dim 4) is penalised only when the prior looks along Z."""
    x = torch.rand(1, 1, 8, 8, 8)
    blurred = x.mean(dim=-1, keepdim=True).expand_as(x).contiguous()
    strategy, _ = _s2v(0)
    along_z = strategy._isotropy_consistency(blurred, tp_dim=4)
    along_x = strategy._isotropy_consistency(blurred, tp_dim=2)
    assert along_z.item() > along_x.item()
