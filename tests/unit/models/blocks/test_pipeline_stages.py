"""Pipeline stages of the unrolled cascades: the composed stages ARE the model.

Each model is split into 1..cascades stages; the stages run back to back on one
process must reproduce the unsplit forward exactly, in values and in gradients, and
their state dicts must partition the unsplit keys. The planted stage that drops the
mask is the shape the context check exists to catch.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.pipeline_stages import (
    CascadeStage,
    balanced_partition,
    run_pipeline_stages,
)
from spectramr.models.generators.diff_varnet import DiffVarNet
from spectramr.models.generators.nc_pdnet import NCPDNet
from spectramr.models.generators.unrolled_reconstruction_generator import (
    UnrolledReconstructionGenerator,
)

pytestmark = [pytest.mark.unit]


def _unrolled():
    torch.manual_seed(0)
    model = UnrolledReconstructionGenerator(
        in_channels=2,
        out_channels=2,
        img_size=(8, 8),
        num_unrolls=4,
        features=4,
        num_coils=2,
        output_activation="tanh",
    ).eval()  # BatchNorm in train mode would make the forward batch-dependent; eval is exact.
    maps = torch.randn(2, 2, 8, 8, dtype=torch.complex64)
    maps = maps / maps.abs().pow(2).sum(1, keepdim=True).sqrt()
    mask = torch.zeros(2, 1, 8, 8)
    mask[..., ::2, :] = 1.0
    kspace = torch.randn(2, 2, 8, 8, dtype=torch.complex64) * mask
    kwargs = {"kspace_measured": kspace, "mask": mask, "sensitivity_maps": maps}
    return model, torch.randn(2, 2, 8, 8), kwargs


def _diff_varnet(volume: bool = False):
    torch.manual_seed(0)
    model = DiffVarNet(
        in_channels=2,
        out_channels=2,
        image_size=8,
        num_unrolls=5,
        base_channels=4,
        physics_emb_dim=16,
    )
    shape = (2, 2, 3, 8, 8) if volume else (2, 2, 8, 8)
    mask = (torch.rand(*shape[:1], 1, *shape[2:]) > 0.5).float()
    kwargs = {"mask": mask, "measured_kspace": torch.randn(shape) * mask, "max_timesteps": 10.0}
    return model, torch.randn(shape), {"timesteps": torch.tensor([2, 7]), **kwargs}


def _nc_pdnet():
    from spectramr.infrastructure.physics.nufft_ops import NUFFTForwardModel
    from spectramr.infrastructure.physics.trajectories import get_trajectory

    traj, dcf = get_trajectory("spiral", im_size=(16, 16))
    torch.manual_seed(0)
    samples = NUFFTForwardModel(im_size=(16, 16)).forward_project(
        torch.randn(2, 1, 16, 16, dtype=torch.complex64), traj
    )
    model = NCPDNet(in_channels=2, out_channels=2, image_size=16, num_cascades=3, width=4, depth=2)
    for reg in model.regularisers:  # the zero-init exit would make every cascade a pure DC step
        torch.nn.init.normal_(reg.body[-1].weight, std=0.1)
    kwargs = {"measured_kspace": samples, "trajectory": traj, "dcf": dcf}
    return model, torch.randn(2, 2, 16, 16), kwargs


CASES = {"unrolled": _unrolled, "diff_varnet": _diff_varnet, "nc_pdnet": _nc_pdnet}


def _cascades(model) -> int:
    for name in ("blocks", "regularisers", "regularization_blocks"):
        if hasattr(model, name):
            return len(getattr(model, name))
    raise AssertionError(type(model).__name__)


@pytest.mark.parametrize("case", sorted(CASES))
def test_composed_stages_equal_the_unsplit_forward(case: str) -> None:
    model, x, kwargs = CASES[case]()
    reference = model(x, **kwargs)
    for n in range(1, _cascades(model) + 1):
        stages = model.pipeline_stages(n)
        assert len(stages) == n
        torch.testing.assert_close(
            run_pipeline_stages(stages, x, **kwargs), reference, rtol=0, atol=0
        )
        keys = [k for stage in stages for k in stage.state_dict()]
        assert sorted(keys) == sorted(model.state_dict()), "stages must partition the unsplit keys"


@pytest.mark.parametrize("case", sorted(CASES))
def test_gradients_through_the_stages_equal_the_unsplit_ones(case: str) -> None:
    model, x, kwargs = CASES[case]()
    model(x, **kwargs).square().sum().backward()
    expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    stages = model.pipeline_stages(2)
    run_pipeline_stages(stages, x, **kwargs).square().sum().backward()
    got = {n: p.grad for s in stages for n, p in s.named_parameters() if p.grad is not None}
    assert got.keys() == expected.keys()
    for name, grad in expected.items():
        torch.testing.assert_close(got[name], grad, rtol=0, atol=0)


def test_a_folded_volume_is_restored_by_the_last_stage() -> None:
    model, x, kwargs = _diff_varnet(volume=True)
    torch.testing.assert_close(
        run_pipeline_stages(model.pipeline_stages(3), x, **kwargs), model(x, **kwargs)
    )


def test_balanced_partition_is_contiguous_and_refuses_more_stages_than_cascades() -> None:
    assert [len(r) for r in balanced_partition(7, 3)] == [3, 2, 2]
    assert [list(r) for r in balanced_partition(4, 2)] == [[0, 1], [2, 3]]
    model, _, _ = _nc_pdnet()
    with pytest.raises(ValueError, match="between 1 and 3"):
        model.pipeline_stages(4)


def test_only_the_first_stage_enters_and_only_the_last_exits() -> None:
    model, x, kwargs = _unrolled()
    first, last = model.pipeline_stages(2)
    with pytest.raises(RuntimeError, match="only the first stage"):
        last("enter", x, **kwargs)
    with pytest.raises(RuntimeError, match="only the last stage"):
        first("exit", x)
    with pytest.raises(ValueError, match="unknown pipeline phase"):
        first("backward", x)


def test_planted_a_stage_that_drops_the_mask_is_refused(monkeypatch) -> None:
    model, x, kwargs = _unrolled()
    stages = model.pipeline_stages(2)
    original = type(stages[0])._cascade

    def drops_mask(self, payload):
        estimate, kspace, _mask, maps = original(self, payload)
        return (estimate, kspace, None, maps)

    monkeypatch.setattr(type(stages[0]), "_cascade", drops_mask)
    with pytest.raises(RuntimeError, match="changed the pipeline context"):
        run_pipeline_stages(stages, x, **kwargs)


def test_planted_a_partition_that_loses_a_cascade_is_refused() -> None:
    from spectramr.models.blocks.pipeline_stages import build_cascade_stages

    model, _, _ = _unrolled()

    class _Stage(CascadeStage):
        pass

    with pytest.raises(RuntimeError, match="does not partition the model"):
        build_cascade_stages(
            model, 2, 4, _Stage, lambda c, *_: [f"regularization_blocks.{i}" for i in c]
        )
