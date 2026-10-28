r"""Hamiltonian-NODE acquisition-learning training strategy (B-4 / Bundle P7).

Jointly optimises a learnable k-space trajectory (the potential :math:`V_\theta`
inside :class:`HamiltonianTrajectoryGenerator`) against a reconstruction-fidelity
objective, subject to two physics penalties:

* :class:`HamiltonianEnergyConservationLoss` — keeps the symplectic integration
  energy-faithful;
* :class:`GradientHardwareComplianceLoss` — soft-penalises gradient-amplitude /
  slew-rate violations.

Reconstruction-fidelity surrogate (documented simplification)
-------------------------------------------------------------
The full P7 arm trains a *second* reconstruction network on data acquired along
the generated trajectory with a NUFFT (the plan's §"New strategy file"). To keep
this component self-contained, runnable, and free of a second registered model
or a NUFFT dependency, the strategy uses a **differentiable Cartesian-coverage
surrogate**:

1. the continuous trajectory ``k(t) ∈ [-0.5, 0.5]^d`` is soft-rasterised onto
   the target's Cartesian k-space grid with a Gaussian kernel, yielding a
   differentiable sampling density ``W in [0, 1]^{HxW}``;
2. the target k-space is masked by ``W`` (``y_acq = W ⊙ F x``), inverse-FFT'd,
   and its magnitude compared to the target image with the configured image
   losses.

The gradient of this fidelity term flows through ``W`` back to the trajectory
and hence to :math:`V_\theta`, so the potential learns to place samples where
the target has energy — a faithful, if reduced, stand-in for the full
joint-recon objective. Swapping in a learned recon net + NUFFT is a drop-in
replacement at step 7 below and does not change the loss bookkeeping.

All trajectory parameters live in ``self.generator_model`` (the registered
``hamiltonian_trajectory_generator``), so the base optimizer already updates
them — no extra ``param_group`` wiring is needed.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from spectramr.config.schemas.training.hamiltonian_acquisition import (
    HamiltonianAcquisitionConfig,
)
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.mixins.kspace import KspaceMixin
from spectramr.infrastructure.training.strategies.mixins.reconstruction import (
    ReconstructionMixin,
)

# Importing the acquisition losses here triggers their ``@register_loss``
# decorators so ``create_loss("hamiltonian_energy_conservation")`` /
# ``create_loss("gradient_hardware_compliance")`` resolve whenever this strategy
# is importable (the strategy is the consumer, so it owns the registration
# wiring). The model is likewise imported so its ``@register_model`` fires.
from spectramr.models.acquisition.hamiltonian_trajectory_generator import (  # noqa: F401
    HamiltonianTrajectoryGenerator,
)
from spectramr.models.losses.acquisition_losses import (  # noqa: F401
    GradientHardwareComplianceLoss,
    HamiltonianEnergyConservationLoss,
)
from spectramr.models.losses.registry import create_loss

logger = logging.getLogger(__name__)


class HamiltonianAcquisitionStrategy(BaseTrainingStrategy, KspaceMixin, ReconstructionMixin):
    """Joint trajectory-potential + reconstruction-surrogate training strategy."""

    def __init__(
        self,
        env: Any,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, device=device, **kwargs)
        # No second ``self._setup_strategy_specific_components()``: the base __init__
        # already runs it, and it is not idempotent once it adds optimizer param groups.

    # ------------------------------------------------------------------
    def _coerce_config(self) -> HamiltonianAcquisitionConfig:
        """Coerce the raw ``training.hamiltonian_acquisition`` block to a typed model.

        Mirrors ``TTOTrainingStrategy.setup()``: the base schema tolerates the
        block as an extra field, and the strategy gives it a typed, validated
        face here so attribute access downstream is safe.
        """
        raw = getattr(self.config.training, "hamiltonian_acquisition", None)
        if raw is None:
            raise ValueError(
                "[HamiltonianAcquisitionStrategy] Missing "
                "training.hamiltonian_acquisition block — required to configure "
                "the potential, integrator, hardware bounds and loss weights."
            )
        if isinstance(raw, HamiltonianAcquisitionConfig):
            return raw
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        elif not isinstance(raw, dict):
            raw = dict(raw)
        return HamiltonianAcquisitionConfig.model_validate(raw)

    def _assert_model_kwargs_consistent(self) -> None:
        """Refuse a divergent-truth config (pitfall #15).

        The trajectory model is built from ``model.model_kwargs`` (its SSOT),
        while the strategy reads physics / loss-weight knobs from
        ``training.hamiltonian_acquisition``. Several model-architecture fields
        are duplicated across the two blocks (potential arch/dims/layers,
        integrator, dt, T_total, shots, samples) — editing one and not the other
        would silently desync the *built model* from the *configured
        acquisition*. Require the overlapping fields to match so the duplication
        can never diverge unnoticed (this is why a blind schema trim is unsafe:
        the training block legitimately carries these for human readability —
        they must simply agree with model_kwargs).
        """
        mk = getattr(self.config.model, "model_kwargs", None) or {}
        shared = (
            "potential_arch",
            "potential_hidden_dim",
            "potential_num_layers",
            "integrator",
            "dt_seconds",
            "T_total_seconds",
            "num_shots",
            "samples_per_shot",
        )
        mismatches = [
            f"{f}: training={getattr(self.ham_cfg, f, None)!r} vs model_kwargs={mk[f]!r}"
            for f in shared
            if f in mk and mk[f] != getattr(self.ham_cfg, f, None)
        ]
        if mismatches:
            raise ValueError(
                "[HamiltonianAcquisitionStrategy] training.hamiltonian_acquisition "
                "disagrees with model.model_kwargs on model-architecture fields "
                "(the model is built from model_kwargs; divergent copies would "
                "silently desync the model from the configured acquisition): "
                + "; ".join(mismatches)
            )

    def _setup_strategy_specific_components(self) -> None:
        self.ham_cfg = self._coerce_config()
        self._assert_model_kwargs_consistent()

        # Physics losses (registered domain="physics").
        self.loss_energy = create_loss("hamiltonian_energy_conservation").to(self.device)
        self.loss_hardware = create_loss(
            "gradient_hardware_compliance",
            g_max_mT_per_m=self.ham_cfg.G_max_mT_per_m,
            s_max_T_per_m_per_s=self.ham_cfg.S_max_T_per_m_per_s,
            dt_seconds=self.ham_cfg.dt_seconds,
            gamma_hz_per_t=self.ham_cfg.gamma_hz_per_t,
            fov_m=self.ham_cfg.fov_m,
        ).to(self.device)

        # Image-domain reconstruction losses from the declarative loss schema.
        # We route every configured image/complex loss to the magnitude image of
        # the coverage-reconstructed target (the recon surrogate is real-valued).
        self._recon_loss_specs: list[tuple[str, torch.nn.Module, float]] = []
        for domain in ("image", "complex"):
            for comp in getattr(self.config.losses, f"{domain}_losses", []) or []:
                if getattr(comp, "enabled", True) and comp.weight > 0:
                    self._recon_loss_specs.append(
                        (comp.name, create_loss(comp.name).to(self.device), comp.weight)
                    )
        if not self._recon_loss_specs:
            # No silent L2 fallback (CLAUDE.md #9).
            raise ValueError(
                "[HamiltonianAcquisitionStrategy] No enabled reconstruction losses "
                "found under losses.{image,complex}_losses. Declare at least one "
                "(e.g. l1) — refusing an implicit L2 fallback."
            )

        logger.info(
            "[HamiltonianAcquisitionStrategy] potential=%s/%s integrator=%s "
            "G_max=%.1f mT/m S_max=%.1f T/m/s λ_energy=%.3g λ_hw=%.3g "
            "recon_losses=%s",
            self.ham_cfg.potential_arch,
            self.ham_cfg.potential_num_layers,
            self.ham_cfg.integrator,
            self.ham_cfg.G_max_mT_per_m,
            self.ham_cfg.S_max_T_per_m_per_s,
            self.ham_cfg.energy_conservation_weight,
            self.ham_cfg.hardware_penalty_weight,
            {n: w for n, _, w in self._recon_loss_specs},
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _to_image_magnitude(batch: torch.Tensor) -> torch.Tensor:
        """Reduce an arbitrary input batch to a single-channel magnitude image.

        Handles 5D TorchIO volumes (middle slice), complex tensors, and
        real-stacked / multi-coil channel layouts → ``[B, 1, H, W]`` real.
        """
        if batch.ndim == 5:
            batch = batch[..., batch.shape[-1] // 2]
        if torch.is_complex(batch):
            mag = batch.abs()
        else:
            c = batch.shape[1]
            if c == 1:
                mag = batch.abs()
            elif c % 2 == 0:
                re = batch[:, 0::2]
                im = batch[:, 1::2]
                mag = torch.sqrt((re**2 + im**2).clamp_min(1e-12).sum(dim=1, keepdim=True))
            else:
                mag = torch.sqrt((batch**2).sum(dim=1, keepdim=True).clamp_min(1e-12))
        if mag.ndim == 3:
            mag = mag.unsqueeze(1)
        # Collapse any residual channel dim to 1 (mean) for the surrogate.
        if mag.shape[1] != 1:
            mag = mag.mean(dim=1, keepdim=True)
        return mag

    def _coverage_weight(self, k_traj: torch.Tensor, h: int, w: int) -> torch.Tensor:
        r"""Soft-rasterise a trajectory to a differentiable Cartesian density.

        Each trajectory sample ``k ∈ [-0.5, 0.5]^2`` contributes a Gaussian bump
        on the ``HxW`` grid; the per-pixel max over samples gives a coverage
        weight in ``[0, 1]`` that is differentiable w.r.t. the sample positions
        (hence w.r.t. :math:`V_\theta`).

        Args:
            k_traj: Trajectory positions ``[S, T+1, d]`` (``d >= 2``; only the
                first two axes are used for the 2-D grid).
            h: Grid height. w: Grid width.

        Returns:
            Coverage weight ``[1, 1, H, W]``.
        """
        device = k_traj.device
        # Flatten all shots/steps into a single sample set.
        pts = k_traj[..., :2].reshape(-1, 2)  # [N, 2]
        # Grid coordinates in normalised k-space [-0.5, 0.5].
        gy = torch.linspace(-0.5, 0.5, h, device=device, dtype=k_traj.dtype)
        gx = torch.linspace(-0.5, 0.5, w, device=device, dtype=k_traj.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")  # [H, W]
        grid = torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2)  # [H*W, 2]

        # Kernel width ~ one grid cell so coverage is local but smooth.
        sigma = 1.0 / float(max(h, w))
        # Squared distances [H*W, N] — chunk over samples to bound memory.
        # bump = exp(-||grid - pt||^2 / (2 sigma^2)); take max over samples.
        inv = 1.0 / (2.0 * sigma * sigma)
        cov = torch.zeros(grid.shape[0], device=device, dtype=k_traj.dtype)
        chunk = 4096
        for start in range(0, pts.shape[0], chunk):
            p = pts[start : start + chunk]  # [n, 2]
            d2 = (grid[:, None, :] - p[None, :, :]).pow(2).sum(-1)  # [H*W, n]
            cov = torch.maximum(cov, torch.exp(-d2 * inv).amax(dim=1))
        return cov.reshape(1, 1, h, w)

    # ------------------------------------------------------------------
    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Integrate trajectory, apply physics penalties + coverage-recon term."""
        # 1. Integrate the trajectory from the generator (learnable V_theta).
        traj = self.generator_model(return_energy=True)
        k_traj = traj["k"]  # [S, T+1, d]

        # 2. Physics penalties.
        loss_energy = self.loss_energy(traj)
        loss_hardware = self.loss_hardware(traj)

        # 3. Reconstruction-fidelity surrogate (documented simplification).
        target_img = self._to_image_magnitude(self._to_device(target_batch))  # [B,1,H,W]
        h, w = target_img.shape[-2], target_img.shape[-1]
        weight = self._coverage_weight(k_traj, h, w)  # [1,1,H,W]

        target_k = fft2c(target_img.to(torch.complex64))
        recon = ifft2c(target_k * weight.to(target_k.dtype)).abs()  # [B,1,H,W]

        # Normalise both to a common peak for stable image losses.
        peak = target_img.amax().clamp_min(1e-8)
        recon_n = recon / peak
        target_n = target_img / peak

        g_total = torch.zeros((), device=self.device)
        loss_dict: dict[str, torch.Tensor] = {}
        for name, fn, wgt in self._recon_loss_specs:
            val = fn(recon_n, target_n)
            g_total = g_total + wgt * val
            loss_dict[f"loss_{name}"] = val.detach()

        g_total = (
            g_total
            + self.ham_cfg.energy_conservation_weight * loss_energy
            + self.ham_cfg.hardware_penalty_weight * loss_hardware
        )

        loss_dict["loss_hamiltonian_energy_conservation"] = loss_energy.detach()
        loss_dict["loss_gradient_hardware_compliance"] = loss_hardware.detach()
        loss_dict["g_total_loss"] = g_total
        loss_dict["loss"] = g_total
        return loss_dict

    # ------------------------------------------------------------------
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validate: recompute the surrogate recon and report PSNR + diagnostics."""
        with torch.no_grad():
            # #1190: calling ``_compute_losses_impl`` directly bypasses the
            # ``_compute_losses`` wrapper that emits the ``model_output``
            # snapshot, so arm it here. ``snapshot_source("val")`` is the OUTER
            # manager so the phase is still "val" when the emit runs on exit --
            # without it the record would claim the training data chain.
            #
            # Note what ``model_output`` means for THIS paradigm: the generator
            # is a trajectory model invoked as ``generator_model(return_energy=
            # True)``, so the captured tensor is the k-space trajectory, not an
            # image. ``_first_tensor`` takes ``traj["k"]`` (first tensor value in
            # the returned dict). That is the honest record of what this model
            # emits; the ``input``/``target`` rows stay the image batch.
            with (
                self.snapshot_source("val"),
                self._capture_model_output(
                    module=self.generator_model,
                    input_batch=target_batch,
                    target_batch=target_batch,
                    step=int(kwargs.get("batch_idx", 0) or 0),
                ),
            ):
                losses = self._compute_losses_impl(target_batch, target_batch, epoch=0)

            traj = self.generator_model(return_energy=True)
            k_traj = traj["k"]
            target_img = self._to_image_magnitude(self._to_device(target_batch))
            h, w = target_img.shape[-2], target_img.shape[-1]
            weight = self._coverage_weight(k_traj, h, w)
            target_k = fft2c(target_img.to(torch.complex64))
            recon = ifft2c(target_k * weight.to(target_k.dtype)).abs()

            peak = target_img.amax().clamp_min(1e-8)
            recon_n = recon / peak
            target_n = target_img / peak
            mse = torch.nn.functional.mse_loss(recon_n, target_n)
            data_range = (target_n.amax() - target_n.amin()).clamp_min(1e-8)
            psnr = 10.0 * torch.log10(data_range**2 / (mse + 1e-10))

            # k-space coverage fraction as a trajectory-efficiency proxy.
            coverage = (weight > 0.5).float().mean()

            self._last_visual_pred = recon_n.detach().cpu()
            self._last_visual_target = target_n.detach().cpu()

        result = {k: float(v) for k, v in losses.items() if isinstance(v, torch.Tensor)}
        result["val_psnr"] = float(psnr)
        result["val_kspace_coverage"] = float(coverage)

        computer = getattr(self, "validation_metrics_computer", None)
        if computer is not None:
            try:
                computed = computer.compute(recon_n, target_n)
                result.update({k: float(v) for k, v in computed.items()})
            except Exception as exc:  # log, don't mask (CLAUDE.md #9)
                logger.warning(
                    "[HamiltonianAcquisitionStrategy] validation metrics failed: %s",
                    exc,
                )
        return result


__all__ = ["HamiltonianAcquisitionStrategy"]
