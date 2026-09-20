"""Cold-diffusion forward process for a golden-angle radial acquisition.

The grid process degrades by deleting Cartesian bins; this one degrades by
deleting whole spokes from a fixed off-grid trajectory. The two share the rung
ladder -- rung ``t`` keeps ``round(S_full / R_t)`` spokes for the same ``R_t``
the Cartesian arms use -- so per-timestep curves stay comparable across the
cohort without re-indexing.

Golden-angle ordering is what makes that free: consecutive spokes are separated
by the golden angle, so any prefix is both near-uniform in angle and a subset of
every longer prefix. The nesting cold diffusion requires (``M_{t+1}`` subset of
``M_t``) therefore holds by construction rather than by coercion.

``torchkbnufft`` owns the off-grid transform here. That is the documented
exemption to non-negotiable 2: ``fft2c``/``ifft2c`` have no meaning for samples
that do not lie on the grid, and this module still routes every on-grid hop
through them.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.models.diffusion.kspace_process import KSpaceUndersamplingProcess
from spectramr.models.diffusion.sample_measurement import SampleMeasurement

__all__ = ["NonCartesianSpokeProcess"]


def _inert_rungs(budgets: list[int]) -> list[int]:
    """Rungs whose spoke budget equals the previous rung's.

    Module level because construction needs it before the buffer exists and
    ``inert_step_report`` needs it after: one owner for the comparison rather
    than the same test written twice (non-negotiable 17).
    """
    return [t for t in range(1, len(budgets)) if budgets[t] == budgets[t - 1]]


class NonCartesianSpokeProcess(KSpaceUndersamplingProcess):
    """Degrade by dropping golden-angle spokes instead of Cartesian bins.

    The diffusion state stays ``[B, C, H, W]`` k-space, the same tensor the grid
    process emits, so the reverse loop, the six cohort losses and the snapshot
    contract are untouched. What changes is the path between ``x_0`` and
    ``x_t``: project onto the trajectory, keep a spoke prefix, grid back.
    """

    #: Hard data consistency pins ACQUIRED bins. Under a non-Cartesian
    #: acquisition no Cartesian bin is a measurement, so a grid-domain DC layer
    #: has nothing correct to pin and must not be handed a mask it would treat
    #: as one (non-negotiable 3 -- fail loud rather than degrade).
    supports_grid_data_consistency: bool = False

    def __init__(
        self,
        *,
        num_spokes: int = 256,
        samples_per_spoke: int = 256,
        im_size: tuple[int, int] = (256, 256),
        density_compensation: str = "radial",
        schedule_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(schedule_kwargs=schedule_kwargs, **kwargs)
        self.schedule_kwargs = dict(schedule_kwargs or {})
        if num_spokes < 1 or samples_per_spoke < 1:
            raise ValueError(
                f"num_spokes and samples_per_spoke must be >= 1, got "
                f"{num_spokes} and {samples_per_spoke}"
            )
        declared = (self.schedule_kwargs or {}).get("acceleration_range")
        if not declared:
            raise ValueError(
                "NonCartesianSpokeProcess needs an explicit "
                "undersampling.acceleration_range: the spoke budget is "
                "num_spokes / R at each rung, and the fallback schedule reports "
                "an R the arm did not declare (#2114)."
            )
        self.acceleration_range = [float(r) for r in declared]
        if len(self.acceleration_range) != self.num_timesteps:
            raise ValueError(
                f"acceleration_range has {len(self.acceleration_range)} rungs but "
                f"num_timesteps is {self.num_timesteps}; one rung per timestep."
            )
        # Knobs the parent reads that this acquisition cannot honour. Leaving
        # them declared-but-unread is the shape non-negotiable 8 forbids.
        if getattr(self, "enable_dynamic_mask", False):
            raise ValueError(
                "enable_dynamic_mask randomises the Cartesian mask per sample; "
                "a golden-angle prefix IS the acquisition and has no per-sample "
                "randomisation. Set it false on a non-Cartesian arm."
            )
        if getattr(self, "prior_channel_range", None) is not None:
            raise ValueError(
                "prior_channel_range keeps a channel range fully sampled on the "
                "grid; this acquisition has no grid bins to keep."
            )
        self.num_spokes = int(num_spokes)
        self.samples_per_spoke = int(samples_per_spoke)
        self.im_size = (int(im_size[0]), int(im_size[1]))
        self.density_compensation = str(density_compensation)

        from spectramr.infrastructure.physics.nufft_ops import (
            GoldenAngleTrajectory,
            NUFFTForwardModel,
        )

        traj = (
            GoldenAngleTrajectory(self.im_size, self.num_spokes, self.samples_per_spoke)
            .generate(num_frames=1)
            .squeeze(0)
        )
        self.register_buffer("trajectory", traj, persistent=False)
        self.nufft = NUFFTForwardModel(im_size=self.im_size)
        self.register_buffer("dcf", self._build_dcf(traj), persistent=False)

        # One spoke budget and one unit-gain constant per rung, resolved at
        # construction. RadialDCF is unnormalised and its gain moves with the
        # spoke count, so without this the ladder would read as a brightness
        # ramp rather than an aliasing one; measured 1.94e-04 at 64 spokes
        # against 4.41e-03 at 2. Computing either inside the step would put 29
        # NUFFT calls in the training loop (non-negotiable 9).
        spokes = self._spoke_budget_per_rung()
        self.register_buffer("spokes_per_rung", spokes, persistent=False)
        self.register_buffer("sample_masks", self._build_sample_masks(spokes), persistent=False)
        self.register_buffer("rung_gain", self._calibrate_gain(), persistent=False)
        self.register_buffer("grid_coverage", self._build_grid_coverage(), persistent=False)
        self._last_sample_measurement: SampleMeasurement | None = None

    def _build_dcf(self, traj: Tensor) -> Tensor:
        """Density weights for the full trajectory, before any spoke is dropped."""
        if self.density_compensation == "radial":
            from spectramr.infrastructure.physics.gridding import RadialDCF

            return RadialDCF(self.num_spokes, self.samples_per_spoke).forward(traj.transpose(0, 1))
        if self.density_compensation == "iterative":
            raise ValueError(
                "density_compensation='iterative' is not usable here yet. This "
                "process builds ONE density map for the full trajectory and "
                "corrects each rung with a scalar gain, which is exact for "
                "'radial' -- uniformly thinning the angular spokes of a ramp "
                "filter rescales it -- and wrong for the iterative solution, "
                "which would have to be re-solved per rung against that rung's "
                "own spoke set. Use 'radial', or extend this to hold one "
                "IterativeDCF per rung."
            )
        raise ValueError(
            f"Unknown density_compensation {self.density_compensation!r}; "
            f"expected 'radial' or 'iterative'."
        )

    def _spoke_budget_per_rung(self) -> Tensor:
        """Spokes retained at each rung, from the ladder's own R schedule.

        Raises when two rungs round to the same budget. Such a pair emits an
        identical mask, which trains the time embedding to separate states that
        are pixel-identical and leaves the reverse step with nothing to reveal --
        the inert-rung defect #1155 removed from this cohort's Cartesian ladder.
        The budget is ``num_spokes / R``, so the fix is always more spokes.
        """
        # The declared ladder, read directly. NOT
        # ``get_acceleration_schedule()``: that method interpolates linearly
        # between 1 and ``max_accel`` and reads no ``acceleration_range``, so on
        # this cohort's ladder it reports 16.5 where the arm declares 8.0
        # (#2114), and a budget taken from it collapses six of 29 rungs.
        accel = torch.tensor(self.acceleration_range, dtype=torch.float64)
        budget = torch.round(self.num_spokes / accel.clamp_min(1.0))
        budget = budget.clamp(1, self.num_spokes).to(torch.long)
        inert = _inert_rungs(budget.tolist())
        if inert:
            collided = sorted({int(budget[t]) for t in inert})
            smallest = float(accel.max())
            needed = math.ceil(smallest * len(budget))
            raise ValueError(
                f"num_spokes={self.num_spokes} makes the ladder inert: spoke "
                f"budgets {budget.tolist()} repeat at {collided}, so those rungs "
                f"emit the same mask. Raise num_spokes (roughly {needed} or more "
                f"for {len(budget)} rungs up to R={smallest:g}) or shorten the "
                f"acceleration_range."
            )
        return budget

    def _build_sample_masks(self, spokes: Tensor) -> Tensor:
        """``[T, N]`` spoke-prefix masks; a prefix is nested in every longer one."""
        idx = torch.arange(self.num_spokes).view(1, -1, 1)
        keep = (idx < spokes.view(-1, 1, 1)).to(torch.float32)
        return keep.expand(-1, -1, self.samples_per_spoke).reshape(spokes.numel(), -1)

    @torch.no_grad()
    def _calibrate_gain(self) -> Tensor:
        """Per-rung scale making ``A^H(dcf*m*A(x))`` unit-gain on a smooth probe.

        A disc with a smooth phase ramp stands in for the object: the gain of a
        density-compensated adjoint depends on the sampling density, which is
        what changes between rungs, not on the probe's detail.
        """
        h, w = self.im_size
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h), torch.linspace(-1.0, 1.0, w), indexing="ij"
        )
        radius = (xx.pow(2) + yy.pow(2)).sqrt()
        magnitude = torch.nn.functional.avg_pool2d(
            ((radius < 0.7).float() * (1.0 - 0.5 * radius))[None, None], 5, 1, 2
        )
        probe = (magnitude[0, 0] * torch.exp(1j * (1.2 * xx + 0.8 * yy)))[None, None]
        y_probe = self.nufft.forward_project(probe, self.trajectory)

        gains = []
        for mask in self.sample_masks:
            gridded = self.nufft.adjoint_project(
                y_probe * (self.dcf * mask).to(y_probe.dtype), self.trajectory
            )
            # <g, g> is real by construction but carries a complex dtype, which
            # clamp rejects; take the real part before guarding the divide.
            denom = (gridded.conj() * gridded).sum().real.clamp_min(1e-12)
            gains.append((gridded.conj() * probe).sum() / denom)
        return torch.stack(gains)

    def _build_grid_coverage(self) -> Tensor:
        """``[T, H, W]``: which Cartesian bins the retained spokes inform.

        The reverse step is ``x_{t-1} = x_t - D(x0,t) + D(x0,t-1)``, realised as
        ``mask_{t-1} - mask_t`` over the bins that become clean. A mask that is
        empty at every rung makes that difference zero and the whole reverse
        loop an identity -- so the returned mask has to carry which bins gained
        information, not which bins were measured. **No Cartesian bin is ever
        measured here**; these are interpolated, which is why grid data
        consistency is refused outright.

        Nested for free: coverage is monotone in the spoke set, and the spoke
        sets are prefixes.
        """
        height, width = self.im_size
        kx, ky = self.trajectory[0], self.trajectory[1]
        two_pi = 2.0 * math.pi
        ix = ((kx / two_pi + 0.5) * width).round().long().clamp(0, width - 1)
        iy = ((ky / two_pi + 0.5) * height).round().long().clamp(0, height - 1)
        flat = iy * width + ix
        coverage = torch.zeros(self.sample_masks.shape[0], height * width)
        for rung, mask in enumerate(self.sample_masks):
            hit = flat[mask > 0]
            coverage[rung].scatter_(0, hit, 1.0)
        return coverage.reshape(-1, height, width)

    def describe_ladder(self, image_shape: tuple[int, int]) -> list[tuple[int, float, float, int]]:
        """``(t, R_nominal, R_effective, spokes_kept)`` for the SPOKE ladder.

        The inherited version reports the parent's Cartesian mask generator,
        which this acquisition never consults -- it would certify a bin ladder
        this arm does not run, and report 32.0 where the spoke budget realises
        30.92. That is the reporter-disagrees-with-realiser defect #2114, and
        overriding is what keeps it out of this class.

        ``R_effective = num_spokes / spokes_kept``: acceleration here is one over
        the retained fraction of the trajectory, the quantity the degradation
        actually applies.
        """
        del image_shape  # the trajectory fixes the matrix size at construction
        return [
            (t, float(self.acceleration_range[t]), self.num_spokes / int(kept), int(kept))
            for t, kept in enumerate(self.spokes_per_rung)
        ]

    def inert_step_report(self, image_shape: tuple[int, int] | None = None) -> list[int]:
        """Rungs whose spoke budget equals the previous rung's.

        One owner for "is a rung inert" (non-negotiable 17): construction's
        raise calls this rather than repeating the comparison.
        """
        del image_shape
        return _inert_rungs(self.spokes_per_rung.tolist())

    def nesting_leak_report(
        self, image_shape: tuple[int, int] | None = None, *, raw: bool = False
    ) -> list[dict]:
        """Spokes re-acquired after being dropped -- empty by construction.

        Prefixes of an ordered trajectory are nested, so this reports the fact
        rather than assuming it: a future ordering that is not prefix-nested
        would surface here instead of silently breaking the cocycle.
        """
        del image_shape, raw
        masks = self.sample_masks
        leaks = []
        for t in range(1, masks.shape[0]):
            readded = int(((masks[t] > 0) & ~(masks[t - 1] > 0)).sum())
            if readded:
                leaks.append({"timestep": t, "reintroduced_samples": readded})
        return leaks

    def coverage_at(self, t: int) -> Tensor:
        """Fraction of Cartesian bins the rung's spokes inform, for reporting."""
        return self.grid_coverage[int(t)]

    def sample_mask_at(self, t: int) -> Tensor:
        """The ``[N]`` spoke mask at rung ``t`` -- the acquisition's real support."""
        return self.sample_masks[int(t)]

    def q_sample(
        self, x_start: Tensor, t: Tensor, noise: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Degrade by spoke removal, returning grid-shaped k-space.

        Args:
            x_start: Clean k-space ``[B, C, H, W]``.
            t: Rung indices ``[B]``.
            noise: Unused; the degradation is deterministic, as for the grid
                process this subclasses.

        Returns:
            ``(x_t, coverage)`` where ``x_t`` is ``[B, C, H, W]`` k-space and
            ``coverage`` is ``[B, 1, H, W]`` marking the bins the retained
            spokes INFORM. Those bins are interpolated, never measured, which
            is why ``apply_data_consistency`` refuses to treat them as data.
        """
        if x_start.dim() != 4:
            raise ValueError(
                f"NonCartesianSpokeProcess expects [B, C, H, W] k-space, got "
                f"shape {tuple(x_start.shape)}."
            )
        batch, channels, height, width = x_start.shape
        if (height, width) != self.im_size:
            raise ValueError(
                f"x_start is {height}x{width} but the trajectory was built for "
                f"{self.im_size[0]}x{self.im_size[1]}; rebuild the process."
            )
        device = x_start.device
        traj = self.trajectory.to(device)
        dcf = self.dcf.to(device)

        image = ifft2c(x_start)
        y_full = self.nufft.forward_project(image, traj)
        masks = self.sample_masks.to(device)[t.clamp(0, self.sample_masks.shape[0] - 1)]
        gains = self.rung_gain.to(device)[t.clamp(0, self.rung_gain.shape[0] - 1)]

        weights = (dcf.unsqueeze(0) * masks).unsqueeze(1).to(y_full.dtype)
        gridded = self.nufft.adjoint_project(y_full * weights, traj)
        gridded = gridded * gains.view(-1, *([1] * (gridded.dim() - 1)))

        # Publish what this rung measured. The samples are detached because
        # they are data: a gradient path back through them would let the network
        # move the measurement it is being scored against.
        self._last_sample_measurement = SampleMeasurement(
            samples=y_full.detach(),
            mask=masks.detach(),
            trajectory=traj,
            projector=self.nufft,
        )

        coverage = self.grid_coverage.to(device)[
            t.clamp(0, self.grid_coverage.shape[0] - 1)
        ].unsqueeze(1)
        return fft2c(gridded.reshape(batch, channels, height, width)), coverage

    @property
    def last_sample_measurement(self) -> SampleMeasurement | None:
        """The measurement the most recent :meth:`q_sample` produced.

        ``q_sample`` grids the samples away, so the sample-domain fidelity term
        cannot recover them from its output. Reusing this rather than
        re-projecting costs one NUFFT forward per step instead of two.
        """
        return self._last_sample_measurement

    def apply_data_consistency(self, *args, **kwargs):
        """Refuse grid-domain DC: no Cartesian bin is a measurement here."""
        raise NotImplementedError(
            "NonCartesianSpokeProcess acquires off-grid samples, so no Cartesian "
            "bin is a measurement and grid-domain data consistency would pin "
            "interpolated values as if they were measured. Declare a "
            "sample-domain consistency term on the arm instead."
        )
