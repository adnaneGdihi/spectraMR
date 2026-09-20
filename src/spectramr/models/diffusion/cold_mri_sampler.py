"""Cold-MRI sampler — registered wrapper around PhysicsInformedColdDiffusion.

Audit finding B3: every cold-diffusion YAML under
``experiments/inprogress/{kspace_filling,diffusion}/`` declares
``training.diffusion.sampler: cold_mri`` and the schema accepts it
(``spectramr.config.schemas.enums.DiffusionSampler.COLD_MRI``), but no
``@register_sampler(name="cold_mri", ...)`` decoration existed anywhere
in ``src/models/diffusion/``. This forced
:meth:`KSpaceColdDiffusionGenerator.sample` to bypass the registry and
construct :class:`PhysicsInformedColdDiffusion` directly, silently
ignoring ``guidance_scale`` / ``cond_drop_prob`` from YAML at inference.

This module closes the gap by registering a thin ``ColdMRISampler``
under canonical name ``"cold_mri"`` (alias ``"ColdMRI"``) that wraps
:class:`~spectramr.models.diffusion.kspace_process.PhysicsInformedColdDiffusion`
and forwards ``sample`` calls to it. The pattern mirrors
:class:`~spectramr.models.diffusion.pula_sampler.DPSMRISampler`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.models.diffusion.kspace_process import PhysicsInformedColdDiffusion
from spectramr.models.diffusion.samplers import register_sampler


@register_sampler(name="cold_mri", aliases=["ColdMRI"])
class ColdMRISampler(nn.Module):
    """Registry-routed cold-diffusion sampler for MRI k-space inpainting.

    Wraps :class:`PhysicsInformedColdDiffusion` so the sampler can be
    resolved through ``get_sampler("cold_mri", ...)`` like every other
    diffusion sampler in the repo. The wrapped class implements the
    physics-informed reverse process (k-space data consistency +
    cold-diffusion restoration formula); this layer only routes
    construction through the registry and delegates ``sample``.

    Args:
        model: The denoiser / score network. Typically the
            ``KSpaceColdDiffusionGenerator`` itself.
        num_timesteps: Total diffusion timesteps used at training.
        max_acceleration: Final acceleration factor at ``t = T``.
        center_fraction: Fraction of low-frequency lines always kept
            (matches the dataset's ACS region).
        dc_method: ``"hard"`` replaces predicted lines with measurement
            lines under the mask; ``"soft"`` applies a residual
            correction scaled by ``dc_weight``.
        dc_weight: Weight on the soft-DC residual term.
        sampling_steps: Number of reverse steps at inference; defaults
            to ``num_timesteps`` if ``None``.
        reverse_mode: ``"additive"`` (legacy Bansal-Alg-2 accumulate loop)
            or ``"replace_freeze"`` (corrected bounded monotone infill).
            Forwarded to :class:`PhysicsInformedColdDiffusion`.
        reverse_clip_ratio: per-sample magnitude ceiling ratio used by
            ``replace_freeze``.
        clip_reference: what that ratio multiplies -- ``global_max`` (the whole
            tensor's max, i.e. the k-space DC peak) or ``band_local`` (the
            coefficient's own radial band). See issue #536.
        sampler_sigma: C6 reverse-step noise scale; 0 keeps the sampler
            deterministic. Forwarded to :class:`PhysicsInformedColdDiffusion`.
        sampler_seed: seed of the dedicated noise generator (``None`` draws
            from entropy). Forwarded likewise.
        selection_rule: which lines each reverse step reveals; only ``fixed``
            exists. Validated by the wrapped class at construction.
        kspace_log_scaled: whether the k-space tensor arrives in ``log1p`` units
            (``data.processing.enable_log_scaling``). **Required and
            keyword-only** -- ``reverse_clip_ratio`` is a *physical* multiplier,
            so applying it in ``log1p`` units yields a physical bound of
            ``expm1(ratio*m)/expm1(m)``, exponential in the dynamic range. There
            is no safe default: guessing either way silently mis-scales the
            ceiling, so an unwired caller must fail loud (pitfall #9).

    .. note::

       Only :class:`~spectramr.models.generators.kspace_cold_diffusion_generator.KSpaceColdDiffusionGenerator`
       currently supplies ``kspace_log_scaled`` (``ModelBuilder`` injects it from
       the ``data.processing`` SSOT through the ``_get_contract`` signature
       seam). ``inference_sampler: cold_mri`` is *also* accepted by the schema
       for score-based arms, whose generator forwards a fixed kwarg allowlist
       that does not include it -- that pairing raises ``TypeError`` at
       construction. No arm in ``experiments/`` uses it today. See issue #1288.
    """

    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        max_acceleration: float = 8.0,
        center_fraction: float = 0.0325,
        dc_method: str = "hard",
        dc_weight: float = 1.0,
        sampling_steps: int | None = None,
        reverse_mode: str = "additive",
        reverse_clip_ratio: float = 4.0,
        clip_reference: str = "global_max",
        sampler_sigma: float = 0.0,
        sampler_seed: int | None = None,
        selection_rule: str = "fixed",
        *,
        kspace_log_scaled: bool,
    ) -> None:
        """Thin registry wrapper over :class:`PhysicsInformedColdDiffusion`.

        ``kspace_log_scaled`` is required and keyword-only, mirroring the core
        sampler: ``reverse_clip_ratio`` is a PHYSICAL multiplier and cannot be
        applied correctly without knowing whether the k-space handed in is
        ``log1p``-compressed (issue #1281).

        Note this wrapper declares NO ``**kwargs``, so its parameter list is a
        hard filter on what ``get_sampler`` can forward. Until 2026-09 that
        filter dropped ``sampler_sigma`` / ``sampler_seed`` / ``selection_rule``
        (issue #1286): a YAML that set them got the deterministic sampler
        whatever it declared. They are now named parameters, so the C6 knobs a
        generator forwards reach the core sampler on this path too.
        """
        super().__init__()
        self._diffusion = PhysicsInformedColdDiffusion(
            model=model,
            num_timesteps=num_timesteps,
            max_acceleration=max_acceleration,
            center_fraction=center_fraction,
            dc_method=dc_method,
            dc_weight=dc_weight,
            sampling_steps=sampling_steps,
            reverse_mode=reverse_mode,
            reverse_clip_ratio=reverse_clip_ratio,
            clip_reference=clip_reference,
            sampler_sigma=sampler_sigma,
            sampler_seed=sampler_seed,
            selection_rule=selection_rule,
            kspace_log_scaled=kspace_log_scaled,
        )

    @torch.no_grad()
    def sample(
        self,
        measurement: torch.Tensor,
        mask: torch.Tensor,
        return_trajectory: bool = False,
        start_timestep: int | None = None,
        seed_offset: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Reverse-diffuse from undersampled k-space to a fully-sampled estimate.

        Args:
            measurement: Acquired (undersampled) k-space, shape
                ``[B, C, H, W]``.
            mask: Sampling mask, shape ``[B, 1, H, W]`` (broadcastable).
            return_trajectory: If ``True``, also return all intermediate
                states.
            start_timestep: Timestep the measurement is degraded AT -- the head
                of the reverse trajectory. ``None`` starts fully degraded at
                ``num_timesteps - 1``, which is only correct when the input sits
                at MAX acceleration.
                :meth:`KSpaceColdDiffusionGenerator.sample` forwards this only
                to samplers whose signature DECLARES it, so omitting it here
                silently discarded the cascading-validation head
                (``start_timestep=t_used``) that implements the #535/#1388 fix.
                It must stay a NAMED parameter -- absorbing it into ``**kwargs``
                would satisfy the generator's gate while still dropping the
                value, turning a warned defect into a silent one (issue #1422).
            seed_offset: Ensemble member index, seeding this call's noise
                stream with ``sampler_seed + seed_offset`` (see
                :meth:`PhysicsInformedColdDiffusion.sample`). NAMED for the same
                reason as ``start_timestep``: the generator forwards it only when
                the signature declares it, and RAISES for a non-zero offset it
                cannot forward, because an ensemble whose members share one
                noise stream is a facade.

        Returns:
            The reconstructed k-space tensor, optionally with a list of
            intermediate states when ``return_trajectory=True``.
        """
        return self._diffusion.sample(
            measurement=measurement,
            mask=mask,
            return_trajectory=return_trajectory,
            start_timestep=start_timestep,
            seed_offset=seed_offset,
        )

    @property
    def reverse_stats(self) -> dict[str, Any]:
        """What the wrapped reverse loop ran on the last :meth:`sample` call.

        Delegated rather than recomputed: this wrapper owns no schedule, and a
        second derivation of "how many steps ran" would be a second owner of the
        answer (non-negotiable 17).
        """
        return self._diffusion.reverse_stats


__all__ = ["ColdMRISampler"]
