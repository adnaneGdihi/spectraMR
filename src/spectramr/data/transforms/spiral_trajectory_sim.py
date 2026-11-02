r"""Rotated-spiral MRF acquisition simulator (MRF idea 1 rotation schedule).

An MRF sequence acquires one spiral interleaf per timepoint :math:`t`, the parent arm
rotated by :math:`\theta_t` [1]. This transform replaces ``input`` with the
density-compensated adjoint NUFFT of the interleaves acquired in a window of
``frames_per_input`` consecutive timepoints starting at :math:`t = 0`:

.. math::

    \hat{x} = A_W^{H}\, D_W\, A_W\, x, \qquad
    W = \{\theta_t\}_{t=0}^{F-1},

so the rotation schedule decides which part of k-space the model sees. The schedules are

* ``golden_angle``: :math:`\theta_t = t\,\pi(3-\sqrt{5})`, the full-circle golden angle
  for spiral interleaves [2];
* ``linear``: :math:`\theta_t = 2\pi t / N_i`, the FISP-MRF ramp with :math:`N_i`
  interleaves per full rotation (7.5 degrees for :math:`N_i = 48`) [1];
* ``random``: :math:`\theta_t \sim U(0, 2\pi)`, drawn once from ``run.seed`` because the
  schedule is part of the acquisition, not an augmentation, and train and validation share it.

``schedule``, ``n_timepoints`` and ``seed`` are owned by the ``mrf:`` and ``run:`` blocks;
the transform builder injects them (:data:`RUN_BLOCK_KWARGS`) and refuses a YAML entry that
restates them. The image is static, so every timepoint sees the same contrast: this
simulates the trajectory, not the Bloch fingerprint.

References:
    [1] Y. Jiang et al., "MR fingerprinting using FISP with spiral readout," MRM 74(6), 2015.
    [2] S. Winkelmann et al., "An optimal radial profile order based on the golden ratio
        for time-resolved MRI," IEEE TMI 26(1), 2007.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Final

import torch
import torchio as tio
from torchio.transforms import Transform

from spectramr.data.transforms.registry import register_transform
from spectramr.infrastructure.physics.trajectories import SPIRAL_GOLDEN_ANGLE, TrajectoryFactory

logger = logging.getLogger(__name__)

try:
    import torchkbnufft as tkbn

    HAVE_TORCHKBNUFFT = True
except ImportError:
    tkbn = None
    HAVE_TORCHKBNUFFT = False

#: Advertised schedules; mirrors ``MRFConfigSchema.spiral_rotation_schedule``.
ROTATION_SCHEDULES: Final[tuple[str, ...]] = ("golden_angle", "linear", "random")


def rotation_angles(
    schedule: str, n_timepoints: int, n_interleaves: int, seed: int
) -> torch.Tensor:
    """The rotation angle of every timepoint's interleaf, in radians, shape ``(n_timepoints,)``."""
    t = torch.arange(n_timepoints, dtype=torch.float64)
    if schedule == "golden_angle":
        angles = t * SPIRAL_GOLDEN_ANGLE
    elif schedule == "linear":
        angles = t * (2.0 * math.pi / n_interleaves)
    elif schedule == "random":
        gen = torch.Generator().manual_seed(int(seed))
        angles = torch.rand(n_timepoints, generator=gen, dtype=torch.float64) * (2.0 * math.pi)
    else:
        raise ValueError(
            f"Unknown spiral rotation schedule {schedule!r}; expected one of {ROTATION_SCHEDULES}."
        )
    return torch.remainder(angles, 2.0 * math.pi).to(torch.float32)


def rotate_arm(parent: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate a ``(2, S)`` parent arm by each angle; returns the ``(2, F*S)`` concatenation."""
    cos, sin = torch.cos(angles), torch.sin(angles)
    rot = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)  # (F,2,2)
    return torch.einsum("fij,js->ifs", rot, parent).reshape(2, -1)


@register_transform("mrf_spiral_trajectory_sim", requires=("input",), modifies=("input",))
class MRFSpiralTrajectorySimulation(Transform):
    """Replace ``input`` with the windowed rotated-spiral MRF reconstruction of it.

    Args:
        schedule: Rotation schedule, injected from ``mrf.spiral_rotation_schedule``.
        n_timepoints: Schedule length, injected from ``mrf.n_timepoints``.
        seed: Seed of the ``random`` schedule, injected from ``run.seed``.
        frames_per_input: Consecutive timepoints whose interleaves form one input.
        n_interleaves: Interleaves per full rotation at Nyquist. It sets the parent arm's
            turns (``max(H, W) / (2 n_interleaves)``) for every schedule and the ``linear`` step.
        dcf_iterations: Pipe iterations for the window's density compensation.
    """

    #: Constructor kwargs owned by run-level blocks: ``kwarg -> (block, field)``.
    RUN_BLOCK_KWARGS: Final[dict[str, tuple[str, str]]] = {
        "schedule": ("mrf", "spiral_rotation_schedule"),
        "n_timepoints": ("mrf", "n_timepoints"),
        "seed": ("run", "seed"),
    }

    def __init__(
        self,
        *,
        schedule: str,
        n_timepoints: int,
        seed: int,
        frames_per_input: int,
        n_interleaves: int = 48,
        dcf_iterations: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not HAVE_TORCHKBNUFFT:
            raise ImportError(
                "mrf_spiral_trajectory_sim needs torchkbnufft (the `.[mri]` extra); it is not "
                "installed, and the spiral acquisition cannot be simulated without it."
            )
        if schedule not in ROTATION_SCHEDULES:
            raise ValueError(f"schedule={schedule!r} is not one of {ROTATION_SCHEDULES}.")
        if n_timepoints is None or int(n_timepoints) < 1:
            raise ValueError(f"mrf.n_timepoints must be a positive int, got {n_timepoints!r}.")
        if seed is None:
            raise ValueError("run.seed is required to fix the spiral rotation schedule.")
        if not 1 <= int(frames_per_input) <= int(n_timepoints):
            raise ValueError(
                f"frames_per_input={frames_per_input} must lie in [1, n_timepoints={n_timepoints}]."
            )
        if int(n_interleaves) < 1 or int(dcf_iterations) < 1:
            raise ValueError("n_interleaves and dcf_iterations must be positive.")
        self.schedule = schedule
        self.n_timepoints = int(n_timepoints)
        self.seed = int(seed)
        self.frames_per_input = int(frames_per_input)
        self.n_interleaves = int(n_interleaves)
        self.dcf_iterations = int(dcf_iterations)
        self.args_names = (
            "schedule",
            "n_timepoints",
            "seed",
            "frames_per_input",
            "n_interleaves",
            "dcf_iterations",
        )
        all_angles = rotation_angles(schedule, self.n_timepoints, self.n_interleaves, self.seed)
        self.window_angles = all_angles[: self.frames_per_input]
        # Operators depend only on the knobs and the in-plane size: built once per size.
        self._plans: dict[tuple[int, int], dict[str, Any]] = {}
        logger.info("[TRANSFORM] mrf_spiral_trajectory_sim: %s", self.applied_record())

    def _plan(self, im_size: tuple[int, int]) -> dict[str, Any]:
        plan = self._plans.get(im_size)
        if plan is not None:
            return plan
        num_turns = max(im_size) / (2.0 * self.n_interleaves)
        parent, _ = TrajectoryFactory.get_spiral_trajectory(
            im_size, num_arms=1, num_turns=num_turns
        )
        traj = rotate_arm(parent, self.window_angles)
        dcf = tkbn.calc_density_compensation_function(
            ktraj=traj, im_size=im_size, num_iterations=self.dcf_iterations
        ).reshape(-1)
        plan = {
            # A^H D A as one Toeplitz convolution: one FFT pair per slice instead of a
            # forward + adjoint NUFFT (same result to ~1e-5, about 3x faster on CPU).
            "kernel": tkbn.calc_toeplitz_kernel(traj, list(im_size), weights=dcf.reshape(1, -1)),
            "toep": tkbn.ToepNufft(),
            "num_turns": num_turns,
            "scale": 1.0,
        }
        # A flat image passes the band-limited operator near-unchanged at its centre, so this
        # one constant restores the input's intensity scale.
        flat = torch.ones(1, 1, *im_size, dtype=torch.complex64)
        centre = self._apply(plan, flat)[0, 0, im_size[0] // 2, im_size[1] // 2].abs()
        plan["scale"] = float(1.0 / centre.clamp_min(1e-12))
        self._plans[im_size] = plan
        return plan

    @staticmethod
    def _apply(plan: dict[str, Any], batch: torch.Tensor) -> torch.Tensor:
        return plan["toep"](batch, plan["kernel"]) * plan["scale"]

    def applied_record(self) -> dict[str, Any]:
        """What the simulation does, for the snapshot provenance block."""
        n = self.frames_per_input
        return {
            "schedule": self.schedule,
            "window": [0, n],
            "first_angles_deg": [round(math.degrees(float(a)), 4) for a in self.window_angles[:8]],
            "nominal_acceleration": self.n_interleaves / n,
            "num_turns_by_size": {f"{h}x{w}": p["num_turns"] for (h, w), p in self._plans.items()},
        }

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Simulate the spiral window on ``input``; ``target`` keeps the clean image."""
        image = subject["input"].data
        if image.ndim != 4:
            raise ValueError(
                f"mrf_spiral_trajectory_sim expects (C, H, W, D), got {tuple(image.shape)}."
            )
        if "target" not in subject:
            subject["target"] = type(subject["input"])(
                tensor=image.clone(), affine=subject["input"].affine
            )
        plan = self._plan((int(image.shape[1]), int(image.shape[2])))
        mag = image.abs().float() if image.is_complex() else image.float()
        batch = mag.permute(3, 0, 1, 2).to(torch.complex64)  # (D, C, H, W): slices as the batch
        recon = self._apply(plan, batch).abs()
        subject["input"].set_data(recon.permute(1, 2, 3, 0).float())
        return subject


__all__ = [
    "HAVE_TORCHKBNUFFT",
    "ROTATION_SCHEDULES",
    "MRFSpiralTrajectorySimulation",
    "rotate_arm",
    "rotation_angles",
]
