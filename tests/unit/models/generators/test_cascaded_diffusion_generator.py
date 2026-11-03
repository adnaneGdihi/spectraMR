"""CascadedDiffusionGenerator: stage blocks, channel layout, timestep threading."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.cascaded_diffusion_generator import CascadedDiffusionGenerator

STAGE_1 = {"kspace_filter": "low_pass_circle", "cutoff_frequency": 3}
STAGE_2 = {"input_conditioning": "stage_1_output", "prediction_target": "high_pass_residual"}


def _gen(**kw: object) -> CascadedDiffusionGenerator:
    torch.manual_seed(0)
    args: dict = {"features": (4, 8), "time_dim": 8, "stage_1": STAGE_1, "stage_2": STAGE_2}
    args.update(kw)
    return CascadedDiffusionGenerator(**args)


def test_two_stages_have_the_conditioning_widths() -> None:
    gen = _gen()
    assert gen.has_stage_2 and gen.cutoff_frequency == 3
    assert gen.stage_1_net.config.in_channels == 2  # [y, x_t]
    assert gen.stage_2_net.config.in_channels == 3  # [y, stage-1 output, r_t]


def test_timesteps_reach_both_stages() -> None:
    gen = _gen().eval()
    y, z, s1 = (torch.rand(1, 1, 16, 16) for _ in range(3))
    t0, t9 = torch.tensor([0]), torch.tensor([900])
    with torch.no_grad():
        assert not torch.allclose(gen.denoise_stage_1(y, z, t0), gen.denoise_stage_1(y, z, t9))
        assert not torch.allclose(
            gen.denoise_stage_2(y, s1, z, t0), gen.denoise_stage_2(y, s1, z, t9)
        )
        # forward routes to stage 2 only when handed the stage-1 output.
        x = torch.cat([y, z], dim=1)
        assert torch.equal(gen(x, t9), gen.denoise_stage_1(y, z, t9))
        assert torch.equal(gen(x, t9, stage_1_output=s1), gen.denoise_stage_2(y, s1, z, t9))


def test_all_pass_control_has_one_stage() -> None:
    gen = _gen(stage_1={"kspace_filter": "all_pass"}, stage_2=None)
    assert not gen.has_stage_2 and gen.stage_2_net is None


@pytest.mark.parametrize(
    ("stage_1", "stage_2", "match"),
    [
        (None, STAGE_2, "stage_1 is required"),
        ({"kspace_filter": "gaussian", "cutoff_frequency": 3}, STAGE_2, "kspace_filter"),
        ({"kspace_filter": "low_pass_circle", "cutoff_frequency": 0}, STAGE_2, "cutoff"),
        ({**STAGE_1, "resolution": "all"}, STAGE_2, "unknown keys"),
        (STAGE_1, None, "needs a stage_2"),
        (STAGE_1, {**STAGE_2, "input_conditioning": "ground_truth"}, "input_conditioning"),
        (STAGE_1, {**STAGE_2, "prediction_target": "full_image"}, "prediction_target"),
        ({"kspace_filter": "all_pass"}, STAGE_2, "all_pass leaves no residual"),
        ({"kspace_filter": "all_pass", "cutoff_frequency": 3}, None, "cutoff_frequency"),
    ],
)
def test_stage_blocks_raise_on_anything_unadvertised(stage_1, stage_2, match) -> None:
    with pytest.raises(ValueError, match=match):
        _gen(stage_1=stage_1, stage_2=stage_2)
