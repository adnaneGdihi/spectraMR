"""PILOT — Physics-Informed Learned Optimal Trajectories (PR-11 / M3).

Continuous-trajectory variant of LOUPE: we learn (k_x, k_y) coordinates
under hardware-feasibility constraints (slew rate, max gradient).

Reference: Weiss et al. "PILOT: Physics-Informed Learned Optimal
Trajectories for Accelerated MRI" (MRM 2021).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch
from torch import nn

from spectramr.config.schemas.enums import Regime, Task
from spectramr.infrastructure.distributed.owned_modules import OWNED_MODULE, OwnedParameter
from spectramr.infrastructure.physics.implementations.nufft_operator import (
    TORCHKBNUFFT_AVAILABLE,
    NUFFTOperator,
)
from spectramr.infrastructure.physics.nufft_omega_grad import (
    nufft_adjoint_omega,
    nufft_forward_omega,
)
from spectramr.models.capabilities import StrategyCapabilities

from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)

#: Proton gyromagnetic ratio, Hz/T.
GAMMA_HZ_PER_T = 42.577e6


class PILOTStrategy(ReconstructionTrainingStrategy):
    """Joint trajectory + reconstruction strategy with hardware constraints."""

    # parallel_compat: The trajectory is an owned module, DDP-wrapped and read via call_owned.
    parallel_compat = frozenset({"replicated"})

    #: Declared rather than inherited. Inheriting ``ReconstructionTrainingStrategy``
    #: also inherited its ``{RECONSTRUCTION}`` tag, so the one strategy in the
    #: repo that genuinely *designs an acquisition* never said so — which is why
    #: ``ACQUISITION_DESIGN`` read as a ``Task`` no profile supported. It is
    #: BOTH: PILOT learns the (k_x, k_y) trajectory *and* reconstructs from it,
    #: and dropping RECONSTRUCTION here would un-tag the half that already works.
    capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities(
        workflows=frozenset({Regime.STRUCTURAL}),
        tasks=frozenset({Task.RECONSTRUCTION, Task.ACQUISITION_DESIGN}),
    )

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        return (*super().owned_module_declarations(config), ("_trajectory", OWNED_MODULE))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._codesign = self.config.acquisition.codesign
        if not self._codesign.enabled:
            logger.warning(
                "PILOTStrategy instantiated but acquisition.codesign.enabled is False — "
                "trajectory will stay frozen at the YAML-declared init."
            )

        self._trajectory: nn.Parameter | None = None
        # ``model.model_kwargs`` is a ``dict[str, Any]`` — getattr() never finds
        # these as attributes, so the historical getattr() form silently ignored
        # any YAML override and always used the defaults (pitfall #15: an
        # exposed knob that is never read). Use dict access, validate, and stamp.
        model_kwargs = self.config.model.model_kwargs or {}
        n_shots = int(model_kwargs.get("n_shots", 16))
        samples_per_shot = int(model_kwargs.get("samples_per_shot", 256))
        if n_shots <= 0:
            raise ValueError(
                f"PILOTStrategy: model.model_kwargs.n_shots must be > 0, got {n_shots}."
            )
        if samples_per_shot <= 0:
            raise ValueError(
                "PILOTStrategy: model.model_kwargs.samples_per_shot must be > 0, "
                f"got {samples_per_shot}."
            )
        # Stamp the resolved knob values into run provenance (pitfall #15).
        self._n_shots = n_shots
        self._samples_per_shot = samples_per_shot
        logger.info(
            "PILOTStrategy trajectory init: n_shots=%d, samples_per_shot=%d (total points=%d)",
            n_shots,
            samples_per_shot,
            n_shots * samples_per_shot,
        )
        # Initial spiral trajectory in (k_x, k_y) space, normalised to [-0.5, 0.5].
        t = torch.linspace(0, 1, samples_per_shot)
        init_traj = []
        for i in range(n_shots):
            phi = 2 * torch.pi * i / n_shots
            r = 0.5 * t
            kx = r * torch.cos(2 * torch.pi * t * 8 + phi)
            ky = r * torch.sin(2 * torch.pi * t * 8 + phi)
            init_traj.append(torch.stack([kx, ky], dim=-1))
        self._trajectory = nn.Parameter(torch.cat(init_traj, dim=0).to(self.device))
        self.register_owned_module("_trajectory", OwnedParameter(self._trajectory))

        opt = getattr(self.env, "opt_g", None)
        if opt is not None:
            lr = self.config.undersampling.adaptive.mask_learning_rate
            opt.add_param_group({"params": [self._trajectory], "lr": lr})

        # The sampling is a NUFFT on the learned trajectory; without torchkbnufft
        # there is no non-Cartesian sampling, so the arm refuses rather than falling back.
        if not TORCHKBNUFFT_AVAILABLE:
            raise ImportError(
                "PILOTStrategy samples on the learned trajectory and needs torchkbnufft; "
                "it does not fall back to Cartesian sampling."
            )
        if self._codesign.enabled and self._codesign.field_of_view_m is None:
            raise ValueError(
                "PILOTStrategy needs acquisition.codesign.field_of_view_m to express the "
                "trajectory in T/m for the hardware-feasibility penalty."
            )
        self._fov_m = self._codesign.field_of_view_m
        self._im_size = tuple(int(v) for v in self.config.data.sampling.patch_size[:2])
        self._nufft = NUFFTOperator(traj=self._omega(), im_size=self._im_size)
        self._nufft.nufft_obj.to(self.device)
        self._nufft.adj_nufft_obj.to(self.device)

    def _omega(self, traj: torch.Tensor | None = None) -> torch.Tensor:
        """The trajectory in torchkbnufft's convention: radians per sample, row coordinate first.

        The trajectory is stored as ``(kx, ky)`` in normalised units ``[-0.5, 0.5]``.
        ``traj`` is the training step's ``call_owned`` output; it defaults to the parameter.
        """
        assert self._trajectory is not None
        return (2 * torch.pi * (self._trajectory if traj is None else traj)).flip(-1).T

    def _dcf(self) -> torch.Tensor:
        """Ramp density compensation of the current trajectory, shape ``(1, 1, N)``."""
        assert self._trajectory is not None
        radius = self._trajectory.detach().norm(dim=-1).clamp_min(1.0 / (2 * self._n_shots))
        return radius.reshape(1, 1, -1)

    def _sampled_input(
        self, input_batch: torch.Tensor, traj: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Image-domain input the UNet sees: the image sampled on the learned trajectory.

        ``input_batch`` is the pre_model image, real-interleaved ``[B, 2C, H, W]``.
        The NUFFT forward and the DCF-weighted adjoint are scaled by ``sqrt(H*W)`` so
        that a Cartesian trajectory with unit DCF returns the image (tkbn is unnormalised).
        """
        if input_batch.ndim != 4 or input_batch.shape[1] % 2:
            raise ValueError(
                f"PILOT expects a real-interleaved image [B, 2C, H, W], got {tuple(input_batch.shape)}."
            )
        b, c2, h, w = input_batch.shape
        image = torch.complex(input_batch[:, 0::2], input_batch[:, 1::2])
        omega = self._omega(traj)
        scale = (h * w) ** 0.5
        samples = nufft_forward_omega(self._nufft, image, omega, 1.0 / scale)
        adjoint = nufft_adjoint_omega(
            self._nufft, samples * self._dcf(), omega, 1.0 / scale, (h, w)
        )
        return torch.stack([adjoint.real, adjoint.imag], dim=2).reshape(b, c2, h, w)

    def _feasibility_penalty(self, traj: torch.Tensor | None = None) -> torch.Tensor:
        """Slew-rate + gradient-amplitude + acoustic-comfort penalty, in physical units.

        The trajectory is converted to cycles per metre (``k_norm * matrix / FOV``),
        differentiated into gradient (T/m) and slew (T/m/s), and each overshoot is
        taken relative to its limit so the weight does not depend on the hardware.
        Weiss 2021 Eq. 8, with the acoustic-comfort term (Eq. 12) on the slew spectrum.
        """
        traj = self._trajectory if traj is None else traj
        if traj is None or self._samples_per_shot < 3:
            return torch.tensor(0.0, device=self.device)

        # Differences run within each shot only: a flat diff would join one shot's
        # last sample to the next shot's first and invent a boundary spike.
        traj = traj.reshape(self._n_shots, self._samples_per_shot, 2)
        cycles_per_m = traj * (self._im_size[0] / self._fov_m)

        constraints = self._codesign.constraints
        dt = constraints.gradient_dwell_us * 1e-6
        grad = torch.diff(cycles_per_m, dim=1) / (GAMMA_HZ_PER_T * dt)
        slew = torch.diff(grad, dim=1) / dt

        gmax = constraints.gradient_max_mT_per_m / 1000.0
        smax = constraints.slew_rate_max_T_per_m_per_s
        g_overshoot = torch.relu(grad.abs() / gmax - 1.0)
        s_overshoot = torch.relu(slew.abs() / smax - 1.0)
        feas = g_overshoot.pow(2).sum() + s_overshoot.pow(2).sum()

        # rfft on the real slew waveform along the samples axis; a spectral op, exempt from fft2c (physics.md).
        if slew.shape[1] >= 8:
            slew_fft = torch.fft.rfft(slew / smax, dim=1)
            cutoff = slew_fft.shape[1] // 3
            feas = feas + 0.1 * slew_fft[:, cutoff:].abs().pow(2).mean()
        return feas * self._codesign.feasibility_lambda

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # F6 (round 6 2026-05-17): orchestrator-kwargs canonical signature.
        # See TODO/audit/smoke_audit_20260516.md §F6.
        #
        # The UNet input is the image sampled on the learned trajectory, so the data
        # term reaches the trajectory through the NUFFT (nufft_omega_grad) as well as
        # the feasibility penalty. Both read one call_owned output, one DDP forward.
        traj = self.call_owned("_trajectory")
        base_losses = super()._compute_losses_impl(
            input_batch=self._sampled_input(input_batch, traj),
            target_batch=target_batch,
            epoch=epoch,
            **kwargs,
        )
        if not self._codesign.enabled:
            return base_losses

        feas = self._feasibility_penalty(traj)
        base_losses["loss_gradient_feasibility"] = feas
        for total_key in ("loss_total", "g_total_loss"):
            if total_key in base_losses:
                base_losses[total_key] = base_losses[total_key] + feas
                break
        return base_losses

    def _validation_forward(
        self,
        input_batch: torch.Tensor,
        batch_context: dict[str, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Grade the same sampled input the training loss sees, not the fully sampled one."""
        return super()._validation_forward(
            self._sampled_input(input_batch), batch_context, **kwargs
        )
