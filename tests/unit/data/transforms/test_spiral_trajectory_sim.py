"""MRF rotated-spiral simulator: schedules, the transform, and its production reach."""

from __future__ import annotations

import math

import pytest
import torch

tio = pytest.importorskip("torchio")
pytest.importorskip("torchkbnufft")

from spectramr.data.transforms.registry import build_transform, get_transform  # noqa: E402
from spectramr.data.transforms.spiral_trajectory_sim import (  # noqa: E402
    MRFSpiralTrajectorySimulation,
    rotate_arm,
    rotation_angles,
)
from spectramr.infrastructure.physics.trajectories import SPIRAL_GOLDEN_ANGLE  # noqa: E402

ARM_LINEAR = "experiments/inprogress/mrf_2026/ablations/idea_1_linear_rotation_schedule.yaml"
ARM_GOLDEN = "experiments/inprogress/mrf_2026/idea_1_spatiotemporal_mrf_recon.yaml"
ARM_IDEA2 = "experiments/inprogress/mrf_2026/ablations/idea_2_extra_wide_diffusion.yaml"


def _subject(size: int = 32) -> tio.Subject:
    img = torch.zeros(1, size, size, 1)
    img[:, size // 4 : 3 * size // 4, size // 3 : 2 * size // 3, :] = 1.0
    return tio.Subject(input=tio.ScalarImage(tensor=img))


def _sim(schedule: str, frames: int = 6, **kw) -> MRFSpiralTrajectorySimulation:
    return build_transform(
        "mrf_spiral_trajectory_sim",
        schedule=schedule,
        n_timepoints=100,
        seed=7,
        frames_per_input=frames,
        n_interleaves=8,
        **kw,
    )


def test_schedules_follow_their_definitions():
    golden = rotation_angles("golden_angle", 5, 48, 0)
    linear = rotation_angles("linear", 5, 48, 0)
    assert torch.allclose(golden[1].double(), torch.tensor(SPIRAL_GOLDEN_ANGLE).double(), atol=1e-6)
    assert torch.allclose(linear, torch.arange(5) * (2 * math.pi / 48), atol=1e-6)
    r1, r2 = rotation_angles("random", 50, 48, 3), rotation_angles("random", 50, 48, 3)
    assert torch.equal(r1, r2)
    assert not torch.equal(r1, rotation_angles("random", 50, 48, 4))


def test_unknown_schedule_raises():
    with pytest.raises(ValueError, match="schedule"):
        rotation_angles("spiral_out", 4, 48, 0)
    with pytest.raises(ValueError, match="schedule"):
        _sim("spiral_out")


def test_frames_beyond_schedule_raise():
    with pytest.raises(ValueError, match="frames_per_input"):
        _sim("linear", frames=101)


def test_rotate_arm_concatenates_rotated_copies():
    parent = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    out = rotate_arm(parent, torch.tensor([0.0, math.pi / 2]))
    assert torch.allclose(
        out, torch.tensor([[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]]), atol=1e-6
    )


def test_schedule_changes_the_simulated_input():
    outs = {s: _sim(s)(_subject())["input"].data for s in ("golden_angle", "linear")}
    assert outs["golden_angle"].shape == (1, 32, 32, 1)
    assert not torch.allclose(outs["golden_angle"], outs["linear"], atol=1e-3)


def test_target_keeps_the_clean_image_and_provenance_is_stamped():
    subject = _subject()
    clean = subject["input"].data.clone()
    sim = _sim("linear")
    out = sim(subject)
    assert torch.equal(out["target"].data, clean)
    rec = sim.applied_record()
    assert rec["schedule"] == "linear" and rec["window"] == [0, 6]
    assert rec["first_angles_deg"][:2] == [0.0, 45.0]
    assert "32x32" in rec["num_turns_by_size"]


def test_registry_entry_declares_its_effect():
    entry = get_transform("mrf_spiral_trajectory_sim")
    assert entry.modifies == ("input",)
    assert set(entry.cls.RUN_BLOCK_KWARGS) == {"schedule", "n_timepoints", "seed"}


def test_production_path_builds_the_simulator_from_the_mrf_block():
    """Arm YAML -> TrainingSettings -> transform config -> both chains carry the simulator."""
    from spectramr.config.settings import TrainingSettings
    from spectramr.data.builders.torchio_transform_builder import (
        TorchIOTransformBuilder,
        TorchIOTransformConfig,
    )

    schedules = {}
    for arm in (ARM_LINEAR, ARM_GOLDEN):
        cfg = TorchIOTransformConfig.from_settings(TrainingSettings.from_yaml(arm))
        for build in (
            TorchIOTransformBuilder.build_train_transforms,
            TorchIOTransformBuilder.build_val_transforms,
        ):
            sims = [
                t for t in build(cfg).transforms if isinstance(t, MRFSpiralTrajectorySimulation)
            ]
            assert len(sims) == 1
        schedules[arm] = sims[0].schedule
        out = sims[0](_subject())
        assert out["input"].data.shape == (1, 32, 32, 1)
    assert schedules == {ARM_LINEAR: "linear", ARM_GOLDEN: "golden_angle"}

    idea2 = TorchIOTransformConfig.from_settings(TrainingSettings.from_yaml(ARM_IDEA2))
    assert not any(name == "mrf_spiral_trajectory_sim" for name, _ in idea2.extra_transforms)


def test_toeplitz_operator_matches_forward_adjoint_nufft():
    """The cached Toeplitz kernel must equal A^H D A computed with explicit NUFFTs."""
    import torchkbnufft as tkbn

    from spectramr.infrastructure.physics.trajectories import TrajectoryFactory

    sim = _sim("golden_angle")
    size = (24, 20)
    plan = sim._plan(size)
    parent, _ = TrajectoryFactory.get_spiral_trajectory(
        size, num_arms=1, num_turns=plan["num_turns"]
    )
    traj = rotate_arm(parent, sim.window_angles)
    dcf = tkbn.calc_density_compensation_function(
        ktraj=traj, im_size=size, num_iterations=sim.dcf_iterations
    ).reshape(-1)
    x = torch.rand(2, 1, *size).to(torch.complex64)
    ref = tkbn.KbNufftAdjoint(im_size=size)(tkbn.KbNufft(im_size=size)(x, traj) * dcf, traj)
    out = sim._apply(plan, x) / plan["scale"]
    assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3 * float(ref.abs().max()))
