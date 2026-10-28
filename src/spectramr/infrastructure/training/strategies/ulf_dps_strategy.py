"""ULF-consistent diffusion posterior sampling (MICCAI MRIxFields2026, B-2.2).

A learned high-field diffusion **prior** (field-target-conditioned eps-net) is steered
at inference by **diffusion posterior sampling** (DPS, Chung et al. 2023) toward
consistency with the measured ULF observation: each reverse step adds the
data-consistency gradient ``∇_x ||A(x_hat0) - y_ulf||^2`` where ``A`` is the ULF
low-pass forward operator (``ulf_degrade``) and ``y_ulf`` is the 0.1T source image.

Self-contained, composing existing primitives:
- prior training = DDPM eps-matching on the high-field TARGET (reuses
  ``make_alphas_cumprod`` / ``q_sample`` from the field-guided-diffusion strategy);
- score net = ``FieldGuidedScoreUNet`` with ``cond_channels=0`` (unconditional of the
  source — the source enters ONLY through the DPS likelihood, never as a network
  input, so the measurement-consistency term is the sole ULF coupling, anti-facade);
- forward operator = ``ulf_degrade`` (HF -> ULF low-pass), ``y = batch['input']``.

The reverse sampler is the **canonical normalized DPS** (Chung et al. 2023, Alg-1):
each step subtracts a guidance term ``(zeta / ||residual||) * grad`` with the step
size normalised by the residual norm and ``x0_hat`` clamped to the data range. This is
numerically stable for full-size images and the cosine-schedule tail (a fixed-weight
``guidance_weight * sum(residual^2)`` step blows up as ``alpha_bar -> 0``).

``likelihood_weight = 0`` (zeta) disables the guidance (unconstrained prior) — the
one-knob ablation against the ULF-consistent arm.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from .field_guided_diffusion_strategy import make_alphas_cumprod, q_sample
from .reconstruction import ReconstructionTrainingStrategy
from .ulf_lowpass_operator import UlfLowpassOperator, build_ulf_lowpass_operator


def compute_ulf_dps_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    alphas_cumprod: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """DDPM eps-matching on the high-field target (the learned prior)."""
    x0 = batch["target"]
    b_t = batch["field_strength_target"]
    bsz = x0.shape[0]
    timesteps = alphas_cumprod.shape[0]
    t = torch.randint(0, timesteps, (bsz,), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    # cond_channels=0 => no SOURCE conditioning; the prior is over high-field images.
    # contrast_id is NOT source leakage — it conditions the prior on WHICH contrast to
    # synthesize (p(x_HF | field, contrast)); the widened score net uses it only when
    # use_contrast_conditioning is on, else ignores it (raises if on but absent, #15).
    # Shared by GenerativeRefinerStrategy training, so this one edit threads both arms.
    eps = model(x_t, timesteps=t, field_strength=b_t, contrast_id=batch.get("contrast_id"))
    loss = torch.mean((eps - noise) ** 2)
    return {"loss_total": loss, "loss_eps": loss}


def ulf_dps_sample(
    model: Any,
    y_ulf: torch.Tensor,
    b_t: torch.Tensor,
    *,
    alphas_cumprod: torch.Tensor,
    n_steps: int = 50,
    likelihood_weight: float = 1.0,
    blur_cutoff: float = 0.5,
    contrast_id: torch.Tensor | None = None,
    forward_op: UlfLowpassOperator | None = None,
) -> torch.Tensor:
    """Normalized DPS reverse sampling: high-field prior + ULF data-consistency.

    ``forward_op`` is the strategy's resolved ``A``; ``blur_cutoff`` builds a constant
    one when it is not given.

    ``contrast_id`` (optional, per-sample ``[B]``) conditions the widened score net's
    FiLM (only when the net's ``use_contrast_conditioning`` is on).

    Each step: predict ``eps`` -> ``x0_hat`` (clamped to the data range); take the
    deterministic DDIM (eta=0) prior step; subtract the residual-normalized guidance
    ``(zeta / ||residual||) * grad`` where ``residual = A(x0_hat) - y_ulf``. The
    normalization + clamp keep the step bounded for full-size images / the cosine tail
    (Chung et al. 2023, Alg-1). ``likelihood_weight = 0`` (zeta) => plain prior
    sampling. Needs autograd through the score net, so force ``enable_grad`` even if
    validation runs under ``no_grad``.
    """
    timesteps = alphas_cumprod.shape[0]
    n = max(1, min(int(n_steps), timesteps))
    idx = torch.linspace(timesteps - 1, 0, n, device=y_ulf.device).round().long()
    b_t_vec = b_t.reshape(-1).to(y_ulf.device)
    zeta = float(likelihood_weight)
    op = forward_op if forward_op is not None else UlfLowpassOperator.constant_cutoff(blur_cutoff)
    x = torch.randn_like(y_ulf)
    with torch.enable_grad():
        for i in range(n):
            t = int(idx[i])
            x = x.detach().requires_grad_(True)
            t_b = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            eps = model(x, timesteps=t_b, field_strength=b_t_vec, contrast_id=contrast_id)
            a_t = alphas_cumprod[t]
            x0_hat = (x - (1.0 - a_t).sqrt() * eps) / a_t.sqrt().clamp_min(1e-3)
            x0_hat = x0_hat.clamp(-1.0, 2.0)  # data range guard (images are [0,1])
            residual = op.apply(x0_hat, contrast_id) - y_ulf  # A(x0)-y
            # Deterministic DDIM prior step.
            if i < n - 1:
                a_next = alphas_cumprod[int(idx[i + 1])]
                x_prior = a_next.sqrt() * x0_hat + (1.0 - a_next).sqrt() * eps
            else:
                x_prior = x0_hat
            if zeta > 0.0:
                # grad of sum_b ||r_b||^2 is per-sample (samples are independent under
                # the per-sample forward op), so the gradient is correct; normalise by
                # the PER-SAMPLE residual norm (not one batch scalar) so the DPS step
                # size is decoupled across the batch.
                grad = torch.autograd.grad(residual.pow(2).sum(), x)[0]
                rnorm = (
                    residual.detach()
                    .pow(2)
                    .flatten(1)
                    .sum(1)
                    .sqrt()
                    .clamp_min(1e-8)
                    .view(-1, 1, 1, 1)
                )
                x = (x_prior.detach() - (zeta / rnorm) * grad).detach()
            else:
                x = x_prior.detach()
    # Final estimate is scored by the image metrics (SSIM/PSNR/LPIPS) — project into
    # the [0,1] data range (the post-guidance step can leave it, unlike x0_hat).
    return x.detach().clamp(0.0, 1.0)


class UlfDpsStrategy(ReconstructionTrainingStrategy):
    """Train a high-field diffusion prior; sample with ULF-consistent DPS (B-2.2)."""

    #: Loss ownership (mrixfields review 2026-09-03): computes its objective inline and folds NOTHING else: any other declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = False

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("ulf_dps", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "ulf_dps", None)
        self._dps_timesteps = int(getattr(cfg, "timesteps", 1000)) if cfg else 1000
        self._dps_schedule = str(getattr(cfg, "beta_schedule", "cosine")) if cfg else "cosine"
        self._dps_sampling_steps = int(getattr(cfg, "sampling_steps", 50)) if cfg else 50
        self._dps_likelihood_weight = float(getattr(cfg, "likelihood_weight", 1.0)) if cfg else 1.0
        self._ulf_lowpass = build_ulf_lowpass_operator(self.config)
        self._acp_cache: dict[tuple, torch.Tensor] = {}

    def _alphas_cumprod(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (self._dps_timesteps, self._dps_schedule, str(device), str(dtype))
        acp = self._acp_cache.get(key)
        if acp is None:
            acp = make_alphas_cumprod(
                self._dps_timesteps, self._dps_schedule, device=device, dtype=dtype
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
                "UlfDpsStrategy requires a mapping batch (dict/TrainingBatch) from the mrixfields dataset; "
                f"got {type(batch)!r}."
            )
        acp = self._alphas_cumprod(batch["target"].device, batch["target"].dtype)
        return compute_ulf_dps_loss(self.env.generator, batch, alphas_cumprod=acp)

    def validation_step(
        self,
        input_batch: Any,
        target_batch: Any,
        field_strength_target: Any = None,
        contrast_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Fold the per-sample target field into batch_context (the gated seam only
        forwards fields the signature declares)."""
        bc = kwargs.pop("batch_context", None) or {}
        if field_strength_target is not None:
            bc["field_strength_target"] = field_strength_target
        if contrast_id is not None:
            bc["contrast_id"] = contrast_id
        return super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation = ULF-consistent DPS sampling. The ULF source is the DPS
        measurement ``y``; the high-field prior + likelihood produce the estimate."""
        y_ulf = batch_context.get("input", input_batch)
        b_t = batch_context.get("field_strength_target", kwargs.get("field_strength_target"))
        if b_t is None:
            raise ValueError(
                "UlfDpsStrategy validation requires per-sample 'field_strength_target' "
                "(set data.expose_field_strength_target on the mrixfields dataset). "
                "Got field_strength_target=None."
            )
        acp = self._alphas_cumprod(y_ulf.device, y_ulf.dtype)
        gen = self.env.generator
        n_steps = getattr(self, "_dps_sampling_steps", 50)
        zeta = getattr(self, "_dps_likelihood_weight", 1.0)
        # val_chunk_size: the DPS reverse loop retains an autograd graph PER STEP (the
        # data-consistency gradient), so a full val batch at 256x256 can OOM (worse
        # than the no_grad cold-diffusion case that already did). Micro-batch when set.
        cid = batch_context.get("contrast_id", kwargs.get("contrast_id"))
        if cid is not None and not torch.is_tensor(cid):
            cid = torch.as_tensor(cid, device=y_ulf.device)
        val_cfg = getattr(self.config, "validation", None)
        chunk = int((val_cfg.loader.chunk_size if val_cfg else 0) or 0)
        b_t = b_t if torch.is_tensor(b_t) else torch.as_tensor(b_t, device=y_ulf.device)
        if chunk > 0 and y_ulf.shape[0] > chunk:
            outs = [
                ulf_dps_sample(
                    gen,
                    y_ulf[i : i + chunk],
                    b_t[i : i + chunk],
                    alphas_cumprod=acp,
                    n_steps=n_steps,
                    likelihood_weight=zeta,
                    forward_op=self._ulf_lowpass,
                    contrast_id=None if cid is None else cid[i : i + chunk],
                )
                for i in range(0, y_ulf.shape[0], chunk)
            ]
            return torch.cat(outs, dim=0)
        return ulf_dps_sample(
            gen,
            y_ulf,
            b_t,
            alphas_cumprod=acp,
            n_steps=n_steps,
            likelihood_weight=zeta,
            forward_op=self._ulf_lowpass,
            contrast_id=cid,
        )
