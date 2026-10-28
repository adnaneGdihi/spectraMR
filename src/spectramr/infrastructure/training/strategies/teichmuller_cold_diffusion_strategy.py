"""Teichmüller-scheduled cold diffusion (§6 of the SFC plan).

Wires the standalone :class:`TeichmullerScheduleHead` into a cold-
diffusion-style training loop. The schedule head produces a per-step
radial parameter ``r(t)`` along the Teichmüller geodesic between two
endpoint Beltrami coefficients ``(μ_0, μ_T)``; the resulting per-step
``μ_t`` parameterises a quasi-conformal degradation of k-space that the
cold-diffusion restorer is trained against. The geodesic regulariser
anchors the schedule to a constant-speed (constant log-dilatation)
curve, which is the natural OT-style choice between the two SFC
endpoints.

Cold-diffusion forward op (the §6 paradigm math, no longer deferred to
the generator):

1. Sample the schedule ``r(t)`` for ``t ∈ linspace(0, 1, n_steps)``.
2. Pick one step ``t*`` per minibatch and form ``μ_{t*}`` via the
   closed-form Teichmüller geodesic (``conformal_geometry`` SSOT).
3. Degrade ``target`` by a ``μ_{t*}``-parameterised k-space
   attenuation (round-trip via ``fft2c`` / ``ifft2c`` — never raw
   ``torch.fft`` on complex k-space): stronger ``|μ_{t*}|`` ⇒ more
   high-frequency suppression, the SFC analogue of the cold-diffusion
   forward degradation.
4. Restore through the generator and supervise ``L1(restored, target)``.

Because the restoration loss depends on BOTH the data and the schedule
(through ``μ_t`` → degradation strength), the schedule head receives a
batch-dependent gradient — the previous ``0.0 * μ`` no-op gave it none.

References:
    [1]  Ahlfors, *Lectures on Quasiconformal Mappings*, AMS, 2006.
    [9]  Gardiner & Lakic, *Quasiconformal Teichmüller Theory*, AMS, 2000.
    [12] Bansal et al., "Cold diffusion: Inverting arbitrary image
         transforms without noise", *NeurIPS*, 36, 2023.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from spectramr.infrastructure.distributed.owned_modules import OWNED_MODULE
from spectramr.infrastructure.physics.conformal_geometry import (
    enforce_beltrami_constraint,
)
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.models.diffusion.teichmuller_schedule import (
    TeichmullerScheduleHead,
    teichmuller_mu_schedule,
)


class TeichmullerColdDiffusionStrategy(BaseTrainingStrategy):
    """Cold-diffusion-style training with a Teichmüller-geodesic schedule.

    The strategy holds a :class:`TeichmullerScheduleHead` and two fixed
    endpoint Beltrami coefficients ``μ_0`` (trivial zero — i.e. identity
    SFC) and ``μ_T`` (a spatially-varying, maximally-warped SFC field
    constrained to the unit ball). For each training step the strategy
    samples a schedule index ``t ∈ [0, 1]``, computes the corresponding
    ``μ_t`` along the closed-form geodesic, degrades the target via a
    ``μ_t``-parameterised k-space attenuation, restores it through the
    generator, and supervises the restoration. The geodesic
    constant-speed regulariser anchors the schedule.

    All required hyperparameters live in the typed ``training.teichmuller``
    sub-block; a missing block RAISES at construction time (no silent
    fallback — CLAUDE.md pitfall #9), matching the Tier-1 audit validator
    ``validate_teichmuller_cold_diffusion``.
    """

    # parallel_compat: The schedule head is an owned module, DDP-wrapped and run via call_owned.
    parallel_compat = frozenset({"replicated"})

    @classmethod
    def owned_module_declarations(cls, config: object) -> tuple[tuple[str, str], ...]:
        return (*super().owned_module_declarations(config), ("schedule_head", OWNED_MODULE),)

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        # Typed, frozen, nested sub-block — REQUIRED. A missing block is a
        # configuration error: raise rather than silently defaulting (#9/#15).
        training = getattr(self.config, "training", None)
        cfg = getattr(training, "teichmuller", None) if training is not None else None
        if cfg is None:
            raise ValueError(
                "TeichmullerColdDiffusionStrategy requires the "
                "'training.teichmuller' sub-block (reads r_max, spatial_size, "
                "n_steps, lambda_teichmuller, mu_t_max, lambda_degrade). "
                "Add it to the experiment YAML — the strategy will not "
                "silently fall back to hard-coded defaults."
            )

        r_max = float(cfg.r_max)
        spatial = int(cfg.spatial_size)
        self.n_steps = int(cfg.n_steps)
        self.lambda_teich = float(cfg.lambda_teichmuller)
        self.lambda_degrade = float(cfg.lambda_degrade)
        mu_t_max = float(cfg.mu_t_max)

        self.schedule_head = TeichmullerScheduleHead(hidden=16, r_max=r_max)
        if self.device is not None:
            self.schedule_head = self.schedule_head.to(self.device)
        # The schedule head is learnable: register the module for DDP, then put its
        # parameters on opt_g (else it never trains). env.opt_g is READ-ONLY.
        self.register_owned_module("schedule_head", self.schedule_head)
        opt = getattr(self.env, "opt_g", None)
        if opt is not None and hasattr(opt, "param_groups"):
            existing = {p for g in opt.param_groups for p in g["params"]}
            new_params = [p for p in self.schedule_head.parameters() if p not in existing]
            if new_params:
                opt.add_param_group({"params": new_params})

        # Fixed Beltrami endpoints. This strategy is NOT an nn.Module (base +
        # mixins are plain classes), so register_buffer would AttributeError at
        # construct time — store the endpoints as plain device-resident attrs.
        #   μ_0 = 0 (identity SFC, Hilbert curve).
        self.mu0 = torch.zeros(spatial, spatial, dtype=torch.complex64, device=self.device)
        #   μ_T = a spatially-VARYING radial Beltrami field, constrained to
        #   |μ_T| ≤ mu_t_max < 1. A spatially-constant 0.5 endpoint (the old
        #   code) is a degenerate geodesic endpoint carrying no structure.
        self.muT = self._build_radial_beltrami_endpoint(spatial, mu_t_max)

        # Stamp the resolved knobs into run provenance (#15) so a downstream
        # run-summary / config dump records exactly what was used.
        self.resolved_teichmuller_config: dict[str, float | int] = {
            "r_max": r_max,
            "spatial_size": spatial,
            "n_steps": self.n_steps,
            "lambda_teichmuller": self.lambda_teich,
            "lambda_degrade": self.lambda_degrade,
            "mu_t_max": mu_t_max,
        }

    def _build_radial_beltrami_endpoint(self, spatial: int, mu_t_max: float) -> torch.Tensor:
        """Spatially-varying μ_T field with ``|μ_T| ≤ mu_t_max < 1``.

        Magnitude grows radially from centre to edge and the phase winds
        with the polar angle, so the endpoint carries genuine spatial /
        phase structure (unlike a uniform real constant). The smooth
        ``tanh`` projection in :func:`enforce_beltrami_constraint`
        guarantees the unit-ball constraint.
        """
        ys = torch.linspace(-1.0, 1.0, spatial, device=self.device)
        xs = torch.linspace(-1.0, 1.0, spatial, device=self.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        radius = torch.sqrt(xx**2 + yy**2)
        angle = torch.atan2(yy, xx)
        # Unconstrained complex field: radial magnitude, angular phase.
        raw = radius.to(torch.complex64) * torch.exp(1j * angle.to(torch.complex64))
        return enforce_beltrami_constraint(raw, k_max=mu_t_max)

    def _sample_schedule(self, device: torch.device) -> torch.Tensor:
        """Return r(t) for t ∈ linspace(0, 1, n_steps)."""
        t = torch.linspace(0.0, 1.0, self.n_steps, device=device)
        return self.call_owned("schedule_head", t)  # [n_steps]

    def _sfc_degrade(self, image: torch.Tensor, mu_t: torch.Tensor) -> torch.Tensor:
        """Apply a ``μ_t``-parameterised k-space attenuation to ``image``.

        The SFC degradation severity scales with the local Beltrami
        magnitude ``|μ_t|``: k-space is multiplied by ``1 - |μ_t|`` after
        resampling ``|μ_t|`` to the image grid, so larger dilatation ⇒
        stronger high-frequency suppression. Round-trips go through the
        centred ``fft2c`` / ``ifft2c`` SSOT (never raw ``torch.fft`` on
        complex k-space).

        Args:
            image: ``[B, C, H, W]`` real (or interleaved-complex) tensor.
            mu_t: complex Beltrami map ``[H_mu, W_mu]`` for this step.
        """
        b, c, h, w = image.shape
        # |μ_t| in [0, 1) resampled to the image grid → a per-pixel
        # attenuation gate. Carries the schedule gradient (via μ_t) AND
        # multiplies the data (via k-space), so the loss is batch-dependent.
        mu_mag = mu_t.abs().to(image.dtype).view(1, 1, *mu_t.shape)
        gate = F.interpolate(mu_mag, size=(h, w), mode="bilinear", align_corners=False)
        attenuation = (1.0 - gate).clamp_min(0.0)  # [1, 1, H, W]

        # Flatten channels into the batch so each real channel is an odd
        # (=1) channel count: fft_ops then does a genuine real→complex
        # round-trip per channel, channel-shape-stable for any C parity
        # (the even-C interleaved-complex interpretation would otherwise
        # halve the channel count).
        flat = image.reshape(b * c, 1, h, w)
        kspace = fft2c(flat)  # complex [B*C, 1, H, W]
        kspace = kspace * attenuation
        degraded = ifft2c(kspace)
        return degraded.real.to(image.dtype).reshape(b, c, h, w)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch
        if target_batch is None:
            raise ValueError(
                "TeichmullerColdDiffusionStrategy requires a target_batch; "
                "the cold-diffusion restoration is supervised against the "
                "clean target."
            )
        device = target_batch.device

        # Schedule for this minibatch: shape [n_steps].
        r = self._sample_schedule(device)
        # Per-step μ_t schedule: [n_steps, H, W].
        mu_schedule = teichmuller_mu_schedule(self.mu0, self.muT, r)

        # Cold-diffusion forward op: pick one schedule step, degrade the
        # target with the corresponding μ_t, and restore through the
        # generator. The step is selected via a TENSOR index (no ``.item()``
        # GPU sync in the training loop — CLAUDE.md performance rule) and
        # gathered out of the schedule, so μ_t keeps its graph connection to
        # the schedule head.
        step_idx = torch.randint(0, self.n_steps, (1,), device=device)
        mu_t = mu_schedule.index_select(0, step_idx).squeeze(0)
        degraded = self._sfc_degrade(target_batch, mu_t)
        restored = self.generator_model(degraded)
        degrade_loss = F.l1_loss(restored, target_batch)

        # Geodesic-smoothness penalty on log-dilatation second differences.
        log_k = ((1.0 + r) / (1.0 - r).clamp_min(1e-6)).log()
        teich_smooth = (log_k.diff().diff()).pow(2).mean()
        # Diagnostic: maximal dilatation along the schedule.
        dilatation_max = ((1.0 + r) / (1.0 - r).clamp_min(1e-6)).max()

        total = self.lambda_degrade * degrade_loss + self.lambda_teich * teich_smooth
        return {
            "g_total_loss": total,
            "teich_cold_recon": degrade_loss.detach(),
            "teich_cold_geodesic_smoothness": teich_smooth.detach(),
            "teich_cold_dilatation_max": dilatation_max.detach(),
        }


__all__ = ["TeichmullerColdDiffusionStrategy"]
