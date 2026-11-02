"""Continuous-field guided diffusion (MICCAI MRIxFields2026, B-3.5).

Source-conditioned Gaussian diffusion that generates the target-field image, with
**classifier-free guidance on the continuous field** (field-CFG). The score net
``ε_θ(x_t, t, b | src)`` (:class:`FieldGuidedScoreUNet`) is conditioned on the source
image ``src`` (always-on translation condition) and the continuous target field ``b``
(``= 10**(log10 B0)``). During training the field is dropped with probability
``guidance_prob`` (the net's learned null-field FiLM takes over); at sampling the
field-conditional and null-field scores are combined,
``eps = eps_null + guidance_scale*(eps_field - eps_null)``. ``guidance_scale = 1`` recovers plain
conditional sampling (no guidance), so the knob directly tests field-CFG against it.

Self-contained on the standard DDPM math (cosine/linear ``alphas_cumprod`` + DDIM
deterministic reverse), reusing the registered ``FieldGuidedScoreUNet`` rather than the
shared k-space diffusion machinery. Pure helpers hold the testable math.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, ClassVar

import torch

from .reconstruction import ReconstructionTrainingStrategy


def make_alphas_cumprod(
    timesteps: int,
    kind: str = "cosine",
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ``alphas_cumprod`` of length ``timesteps`` (monotone decreasing in t)."""
    if kind == "cosine":
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=device, dtype=dtype)
        acp = torch.cos(((x / timesteps + s) / (1.0 + s)) * math.pi * 0.5) ** 2
        acp = acp / acp[0].clamp_min(1e-12)
        return acp[:timesteps].clamp(1e-5, 1.0)
    if kind == "linear":
        # Rescale the canonical 1000-step linear betas (1e-4..2e-2) by 1000/timesteps
        # so the integrated noise is approximately T-invariant (IDDPM rescaling) —
        # otherwise a small `timesteps` under-noises x_T and the DDIM init (pure noise)
        # mismatches the model's t=T marginal. Clamp to keep 1-beta in (0, 1].
        scale = 1000.0 / float(timesteps)
        betas = torch.linspace(1e-4, 2e-2, timesteps, device=device, dtype=dtype) * scale
        betas = betas.clamp(1e-4, 0.999)
        return torch.cumprod(1.0 - betas, dim=0).clamp(1e-5, 1.0)
    raise ValueError(f"Unknown beta schedule {kind!r}; expected 'cosine' or 'linear'.")


def q_sample(
    x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alphas_cumprod: torch.Tensor
) -> torch.Tensor:
    """Forward diffusion: ``x_t = sqrt(acp_t)·x0 + sqrt(1-acp_t)·noise``."""
    a = alphas_cumprod[t].reshape(-1, 1, 1, 1)
    return a.sqrt() * x0 + (1.0 - a).sqrt() * noise


def compute_field_guided_diffusion_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    alphas_cumprod: torch.Tensor,
    guidance_prob: float = 0.1,
    fold_image_losses: Callable[
        [torch.Tensor, torch.Tensor, dict[str, torch.Tensor]], torch.Tensor | None
    ]
    | None = None,
) -> dict[str, torch.Tensor]:
    """ε-matching loss with per-sample field dropout (the CFG training rule).

    ``fold_image_losses(x0_hat, x0, components)`` adds declared image losses on the x0 estimate.
    """
    x0 = batch["target"]
    src = batch["input"]
    b_t = batch["field_strength_target"]
    bsz = x0.shape[0]
    timesteps = alphas_cumprod.shape[0]
    t = torch.randint(0, timesteps, (bsz,), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    # Field dropout: keep the field with prob (1 - guidance_prob); else use the null.
    field_cond = (torch.rand(bsz, device=x0.device) >= guidance_prob).float()
    eps = model(
        x_t,
        timesteps=t,
        field_strength=b_t,
        cond_image=src,
        field_cond=field_cond,
        contrast_id=batch.get("contrast_id"),
    )
    loss = torch.mean((eps - noise) ** 2)
    components = {"loss_eps": loss}
    total = loss
    if fold_image_losses is not None:
        # Same x0 estimate (and clamp) as the sampler's per-step update.
        a_t = alphas_cumprod[t].reshape(-1, 1, 1, 1)
        x0_hat = ((x_t - (1.0 - a_t).sqrt() * eps) / a_t.sqrt().clamp_min(1e-6)).clamp(-1.0, 2.0)
        aux = fold_image_losses(x0_hat, x0, components)
        if aux is not None:
            total = total + aux
    return {"loss_total": total, **components}


@torch.no_grad()
def field_guided_sample(
    model: Any,
    src: torch.Tensor,
    b_t: torch.Tensor,
    *,
    alphas_cumprod: torch.Tensor,
    n_steps: int = 50,
    guidance_scale: float = 3.0,
    contrast_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """DDIM (deterministic) reverse sampling with field-CFG, from noise → target.

    At each step the field-conditional score is computed; when ``guidance_scale != 1``
    the null-field score is also computed and the guided score
    ``eps_null + guidance_scale*(eps_field - eps_null)`` is used. The source image conditions every
    step (translation); the field is the guided variable. ``guidance_scale = 1`` is
    plain conditional sampling.
    """
    timesteps = alphas_cumprod.shape[0]
    n = max(1, min(int(n_steps), timesteps))
    idx = torch.linspace(timesteps - 1, 0, n, device=src.device).round().long()
    bsz = src.shape[0]
    ones = torch.ones(bsz, device=src.device)
    zeros = torch.zeros(bsz, device=src.device)
    x = torch.randn_like(src)
    for i in range(n):
        t = int(idx[i])
        t_batch = torch.full((bsz,), t, device=src.device, dtype=torch.long)
        eps = model(
            x,
            timesteps=t_batch,
            field_strength=b_t,
            cond_image=src,
            field_cond=ones,
            contrast_id=contrast_id,
        )
        if guidance_scale != 1.0:
            eps_u = model(
                x,
                timesteps=t_batch,
                field_strength=b_t,
                cond_image=src,
                field_cond=zeros,
                contrast_id=contrast_id,
            )
            eps = eps_u + guidance_scale * (eps - eps_u)
        a_t = alphas_cumprod[t]
        # Data-range guard. CFG EXTRAPOLATES (guidance_scale=3.0 in b35_field_cfg), so
        # eps can leave the region the model was trained on, and x0_hat is fed straight
        # back into the DDIM update below -- the error compounds across every step.
        # Unclamped, the b35 run reached PSNR -20.8 dB with NEGATIVE SSIM. Images are
        # [0,1]; the loose [-1,2] bound only kills divergence. The sibling samplers
        # already do this (ulf_dps:100, generative_refiner:148, doob_bridge:142) -- its
        # absence here was a divergent-sibling omission, not a deliberate difference.
        x0_hat = ((x - (1.0 - a_t).sqrt() * eps) / a_t.sqrt().clamp_min(1e-6)).clamp(-1.0, 2.0)
        if i < n - 1:
            a_next = alphas_cumprod[int(idx[i + 1])]
            x = a_next.sqrt() * x0_hat + (1.0 - a_next).sqrt() * eps
        else:
            x = x0_hat
    # Final clamp to the [0,1] data range. The per-step x0_hat clamp above only bounds the
    # DDIM divergence; the returned sample must land in [0,1] or the image metrics grade it
    # off-scale (b35_field_cfg reached PSNR -3.22 dB / SSIM 0.004 with the intermediate
    # clamp alone). The sibling samplers already clamp on return (doob_bridge:160).
    return x.clamp(0.0, 1.0)


class FieldGuidedDiffusionStrategy(ReconstructionTrainingStrategy):
    """Train a source-conditioned, continuous-field-guided diffusion model (B-3.5)."""

    #: Loss ownership (mrixfields review 2026-09-03): the eps-matching term is computed inline; declared
    #: ``losses.image_losses`` are folded onto the x0 estimate the sampler also uses (against the target).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = True
    empty_loss_stack_is_design: ClassVar[bool] = True

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("field_guided_diffusion", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "field_guided_diffusion", None)
        self._fgd_timesteps = int(getattr(cfg, "timesteps", 1000)) if cfg else 1000
        self._fgd_schedule = str(getattr(cfg, "beta_schedule", "cosine")) if cfg else "cosine"
        self._fgd_guidance_prob = float(getattr(cfg, "guidance_prob", 0.1)) if cfg else 0.1
        self._fgd_guidance_scale = float(getattr(cfg, "guidance_scale", 3.0)) if cfg else 3.0
        self._fgd_sampling_steps = int(getattr(cfg, "sampling_steps", 50)) if cfg else 50
        self._acp_cache: dict[tuple, torch.Tensor] = {}

    def _alphas_cumprod(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (self._fgd_timesteps, self._fgd_schedule, str(device), str(dtype))
        acp = self._acp_cache.get(key)
        if acp is None:
            acp = make_alphas_cumprod(
                self._fgd_timesteps, self._fgd_schedule, device=device, dtype=dtype
            )
            self._acp_cache[key] = acp
        return acp

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if batch is None or not hasattr(batch, "get"):  # dict OR TrainingBatch (canonical pipeline)
            raise ValueError(
                "FieldGuidedDiffusionStrategy requires a mapping batch (dict/TrainingBatch) from the "
                f"mrixfields dataset; got {type(batch)!r}."
            )
        acp = self._alphas_cumprod(batch["target"].device, batch["target"].dtype)
        return compute_field_guided_diffusion_loss(
            self.env.generator,
            batch,
            alphas_cumprod=acp,
            guidance_prob=getattr(self, "_fgd_guidance_prob", 0.1),
            fold_image_losses=lambda pred, target, comps: self._apply_builder_image_losses(
                pred, target, comps
            ),
        )

    def validation_step(
        self,
        input_batch: Any,
        target_batch: Any,
        field_strength_target: Any = None,
        contrast_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Fold the per-sample target field + contrast id into batch_context (the gated
        seam only forwards fields the signature declares)."""
        bc = kwargs.pop("batch_context", None) or {}
        if field_strength_target is not None:
            bc["field_strength_target"] = field_strength_target
        if contrast_id is not None:
            bc["contrast_id"] = contrast_id
        return super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation = field-CFG sampling from the source to the target field."""
        src = batch_context.get("input", input_batch)
        b_t = batch_context.get("field_strength_target", kwargs.get("field_strength_target"))
        if b_t is None:
            raise ValueError(
                "FieldGuidedDiffusionStrategy validation requires per-sample "
                "'field_strength_target' (set data.expose_field_strength_target on the "
                "mrixfields dataset). Got field_strength_target=None."
            )
        acp = self._alphas_cumprod(src.device, src.dtype)
        gen = self.env.generator
        n_steps = getattr(self, "_fgd_sampling_steps", 50)
        gscale = getattr(self, "_fgd_guidance_scale", 3.0)
        # val_chunk_size: the reverse-sampling loop (up to 2 forward passes/step for
        # CFG) over a full val batch can OOM (the same failure cold-diffusion hit). When
        # set, sample in micro-batches so the knob is load-bearing, not an inert no-op.
        val_cfg = getattr(self.config, "validation", None)
        chunk = int((val_cfg.loader.chunk_size if val_cfg else 0) or 0)
        b_t = b_t if torch.is_tensor(b_t) else torch.as_tensor(b_t, device=src.device)
        cid = batch_context.get("contrast_id", kwargs.get("contrast_id"))
        if chunk > 0 and src.shape[0] > chunk:
            outs = [
                field_guided_sample(
                    gen,
                    src[i : i + chunk],
                    b_t[i : i + chunk],
                    alphas_cumprod=acp,
                    n_steps=n_steps,
                    guidance_scale=gscale,
                    contrast_id=None if cid is None else cid[i : i + chunk],
                )
                for i in range(0, src.shape[0], chunk)
            ]
            return torch.cat(outs, dim=0)
        return field_guided_sample(
            gen,
            src,
            b_t,
            alphas_cumprod=acp,
            n_steps=n_steps,
            guidance_scale=gscale,
            contrast_id=cid,
        )
