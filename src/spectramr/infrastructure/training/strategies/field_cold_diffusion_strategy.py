"""Field-degradation cold diffusion (MICCAI MRIxFields2026, B-2.4).

Cold diffusion (Bansal et al. 2022) replaces Gaussian noising with an arbitrary
*deterministic* degradation. Here the degradation is **field down-conversion**: a
progressive low-pass whose cutoff shrinks from ``max_cutoff`` (t=0, sharp high-field)
to ``min_cutoff`` (t=T, ULF resolution loss). A restorer ``R_phi(x_t, t)`` is trained
to map each degraded image back to the high-field target, and the image is recovered
by the improved cold-diffusion sampling rule

    x_hat_0 = R(x_t, t);  x_{t-1} = x_t - D_t(x_hat_0) + D_{t-1}(x_hat_0).

Lineage: the project's experiment_11 cold diffusion uses a k-space-undersampling
degradation; this is the same algorithm with a field-down-conversion operator,
built self-contained on the physics SSOT (``fft2c`` via ``_lowpass_otf``) rather
than the k-space-specific machinery. Pure helpers hold the testable math.
``degradation: gaussian`` swaps ``D_t`` for isotropic Gaussian noising (the b24 comparator).

SCOPE / LIMITATION: ``compute_field_cold_diffusion_loss`` trains the restorer to
invert a SYNTHETIC low-pass of the high-field **target** (it ignores ``batch['input']``).
At validation ``cold_diffusion_sample`` starts the reverse chain from the REAL ULF
``batch['input']``, which is not exactly a synthetic low-pass of the target (different
noise/contrast). Validation-on-real-ULF is thus a TRANSFER test of the restorer, and
the synthetic field-down-conversion is a surrogate for real ULF degradation — an
acknowledged approximation, not an implied exact train/val match.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

from .reconstruction import ReconstructionTrainingStrategy
from .ulf_map_strategy import _lowpass_otf


def _cutoff_at(t: int, n_steps: int, min_cutoff: float, max_cutoff: float) -> float:
    """Low-pass cutoff at step ``t`` — GEOMETRIC (log-linear) from max at t=0 to min
    at t=n_steps.

    A Gaussian OTF is near-identity for large cutoff and collapses near the small end,
    so a *linear* cutoff schedule removes ~80% of the image-space degradation in the
    last 2 steps (a near-degenerate middle). Geometric interpolation gives roughly
    equal per-step spectral-energy increments, so the multi-step path is meaningful.
    """
    if n_steps <= 0:
        return max_cutoff
    frac = min(max(t / n_steps, 0.0), 1.0)
    return max_cutoff * (min_cutoff / max_cutoff) ** frac


def field_degrade(
    x: torch.Tensor, t: int, n_steps: int, min_cutoff: float, max_cutoff: float
) -> torch.Tensor:
    """Deterministic field down-conversion ``D_t``: progressive low-pass (real out).

    ``t=0`` is near-identity (cutoff=max_cutoff); ``t=n_steps`` is the ULF end
    (cutoff=min_cutoff). FFT round-trip via the physics SSOT.
    """
    if t <= 0:
        return x
    cutoff = _cutoff_at(t, n_steps, min_cutoff, max_cutoff)
    h_otf = _lowpass_otf(x.shape[-2], x.shape[-1], cutoff, x.device, torch.float32)
    return ifft2c(h_otf * fft2c(x)).real


def _restore(
    restorer: Any,
    x_t: torch.Tensor,
    t_norm: torch.Tensor,
    contrast_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Call the t-conditioned restorer ``R(x_t, t)`` and unwrap its output.

    ``contrast_id`` (when present) is threaded to the restorer so a
    contrast-conditioned ``FieldVelocityUNet`` sees it; contrast-blind restorers
    absorb it via ``**kwargs`` (no-op).
    """
    out = restorer(x_t, field_strength=t_norm, contrast_id=contrast_id)
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, dict):
        out = out.get("reconstruction", next(iter(out.values())))
    return out


def compute_field_cold_diffusion_loss(
    restorer: Any,
    target: torch.Tensor,
    train_steps: int,
    min_cutoff: float,
    max_cutoff: float,
    contrast_id: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Cold-diffusion training loss: restore a randomly-degraded target.

    Samples ``t ~ U{1..train_steps}`` per batch, degrades the high-field target to
    ``x_t = D_t(target)``, and regresses ``R(x_t, t) -> target`` (L1). The training
    curriculum is parameterised by ``train_steps`` independently of the sampler step
    count, so collapsing the sampler (the one-knob ablation) does NOT also change the
    training degradation distribution or the FiLM time-conditioning exposure.

    ``contrast_id`` is threaded to the restorer so a contrast-conditioned model
    conditions the render on the acquisition contrast (T1w/T2w/FLAIR).
    """
    b = target.shape[0]
    t = int(torch.randint(1, train_steps + 1, (1,)).item())
    x_t = field_degrade(target, t, train_steps, min_cutoff, max_cutoff)
    t_norm = torch.full((b,), t / train_steps, device=target.device, dtype=target.dtype)
    pred = _restore(restorer, x_t, t_norm, contrast_id)
    loss = torch.mean(torch.abs(pred - target))
    return {"loss_total": loss, "loss_restore": loss}


@torch.no_grad()
def cold_diffusion_sample(
    restorer: Any,
    x_start: torch.Tensor,
    n_steps: int,
    min_cutoff: float,
    max_cutoff: float,
    contrast_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Improved cold-diffusion sampling (Bansal Alg. 2) from the ULF image ``x_T``."""
    x = x_start
    for t in range(n_steps, 0, -1):
        t_norm = torch.full((x.shape[0],), t / n_steps, device=x.device, dtype=x.dtype)
        x0_hat = _restore(restorer, x, t_norm, contrast_id)
        d_t = field_degrade(x0_hat, t, n_steps, min_cutoff, max_cutoff)
        d_tm1 = field_degrade(x0_hat, t - 1, n_steps, min_cutoff, max_cutoff)
        x = x - d_t + d_tm1
    return x


def _sigma_at(t: int, n_steps: int, sigma_min: float, sigma_max: float) -> float:
    """Gaussian noise std at step ``t``: 0 at t=0, geometric from min (t=1) to max (t=T)."""
    if t <= 0:
        return 0.0
    if n_steps <= 1:
        return sigma_max
    frac = min((t - 1) / (n_steps - 1), 1.0)
    return sigma_min * (sigma_max / sigma_min) ** frac


def compute_gaussian_diffusion_loss(
    restorer: Any,
    target: torch.Tensor,
    train_steps: int,
    sigma_min: float,
    sigma_max: float,
    contrast_id: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """As :func:`compute_field_cold_diffusion_loss` with ``D_t(x) = x + sigma_t * eps``."""
    b = target.shape[0]
    t = int(torch.randint(1, train_steps + 1, (1,)).item())
    sigma = _sigma_at(t, train_steps, sigma_min, sigma_max)
    x_t = target + sigma * torch.randn_like(target)
    t_norm = torch.full((b,), t / train_steps, device=target.device, dtype=target.dtype)
    pred = _restore(restorer, x_t, t_norm, contrast_id)
    loss = torch.mean(torch.abs(pred - target))
    return {"loss_total": loss, "loss_restore": loss}


@torch.no_grad()
def gaussian_diffusion_sample(
    restorer: Any,
    x_start: torch.Tensor,
    n_steps: int,
    sigma_min: float,
    sigma_max: float,
    seed: int = 0,
    contrast_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bansal Alg. 2 under Gaussian noising, started from the noised ULF image.

    With ``D_s(x0) = x0 + sigma_s * eps_hat`` and ``eps_hat`` read off the current
    iterate, the update ``x - D_t(x0_hat) + D_{t-1}(x0_hat)`` is the deterministic
    DDIM step ``x0_hat + sigma_{t-1} * eps_hat``. The start noise is re-seeded per
    call so every validation pass draws the same noise.
    """
    gen = torch.Generator(device=x_start.device).manual_seed(int(seed))
    noise = torch.randn(x_start.shape, generator=gen, device=x_start.device, dtype=x_start.dtype)
    x = x_start + _sigma_at(n_steps, n_steps, sigma_min, sigma_max) * noise
    for t in range(n_steps, 0, -1):
        t_norm = torch.full((x.shape[0],), t / n_steps, device=x.device, dtype=x.dtype)
        x0_hat = _restore(restorer, x, t_norm, contrast_id)
        eps_hat = (x - x0_hat) / _sigma_at(t, n_steps, sigma_min, sigma_max)
        x = x0_hat + _sigma_at(t - 1, n_steps, sigma_min, sigma_max) * eps_hat
    return x


#: degradation -> (training loss, sampler). Both share the restorer, curriculum and objective.
_DEGRADATION_PATHS: dict[str, tuple[Any, Any]] = {
    "field_lowpass": (compute_field_cold_diffusion_loss, cold_diffusion_sample),
    "gaussian": (compute_gaussian_diffusion_loss, gaussian_diffusion_sample),
}


class FieldColdDiffusionStrategy(ReconstructionTrainingStrategy):
    """Train a field-down-conversion cold-diffusion restorer (B-2.4)."""

    #: Loss ownership (mrixfields review 2026-09-03): computes its objective inline and folds NOTHING else: any other declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = False

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("field_cold_diffusion", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "field_cold_diffusion", None)
        self._fcd_n_steps = int(getattr(cfg, "n_steps", 10)) if cfg is not None else 10
        self._fcd_train_steps = int(getattr(cfg, "train_steps", 10)) if cfg is not None else 10
        self._fcd_min = float(getattr(cfg, "min_cutoff", 0.25)) if cfg is not None else 0.25
        self._fcd_max = float(getattr(cfg, "max_cutoff", 5.0)) if cfg is not None else 5.0
        self._fcd_degradation = str(getattr(cfg, "degradation", "field_lowpass"))
        if self._fcd_degradation not in _DEGRADATION_PATHS:
            raise ValueError(
                f"field_cold_diffusion.degradation={self._fcd_degradation!r} is unknown; "
                f"expected one of {sorted(_DEGRADATION_PATHS)}."
            )
        if self._fcd_degradation == "gaussian":
            self._fcd_schedule = {
                "sigma_min": float(cfg.noise_sigma_min),
                "sigma_max": float(cfg.noise_sigma_max),
            }
            self._fcd_sample_extra = {"seed": int(cfg.sampler_seed)}

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
                "FieldColdDiffusionStrategy requires a mapping batch (dict/TrainingBatch) from the mrixfields "
                f"dataset; got {type(batch)!r}."
            )
        loss_fn, _ = _DEGRADATION_PATHS[self._degradation()]
        return loss_fn(
            self.env.generator,
            batch["target"],
            train_steps=getattr(self, "_fcd_train_steps", 10),
            contrast_id=batch.get("contrast_id"),
            **self._schedule_kwargs(),
        )

    def _degradation(self) -> str:
        return getattr(self, "_fcd_degradation", "field_lowpass")

    def _schedule_kwargs(self) -> dict[str, float]:
        """The schedule of the declared degradation (cutoffs, or noise stds)."""
        schedule = getattr(self, "_fcd_schedule", None)
        if schedule is not None:
            return dict(schedule)
        return {
            "min_cutoff": getattr(self, "_fcd_min", 0.25),
            "max_cutoff": getattr(self, "_fcd_max", 5.0),
        }

    def validation_step(
        self,
        input_batch: Any,
        target_batch: Any,
        contrast_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Fold the per-sample ``contrast_id`` into ``batch_context``.

        The pipeline's gated validation seam forwards only the batch fields the
        strategy's ``validation_step`` signature declares; declaring ``contrast_id``
        makes it available to :meth:`_validation_forward` (mirrors how
        :class:`FieldFlowStrategy` folds in the field strengths).
        """
        bc = kwargs.pop("batch_context", None) or {}
        if contrast_id is not None:
            bc["contrast_id"] = contrast_id
        return super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation = cold-diffusion sampling from the ULF input (the restored
        high-field image scored by the image metrics)."""
        x0 = batch_context.get("input", input_batch)
        contrast_id = batch_context.get("contrast_id", kwargs.get("contrast_id"))
        _, sample_fn = _DEGRADATION_PATHS[self._degradation()]
        return sample_fn(
            self.env.generator,
            x0,
            n_steps=getattr(self, "_fcd_n_steps", 10),
            contrast_id=contrast_id,
            **self._schedule_kwargs(),
            **getattr(self, "_fcd_sample_extra", {}),
        )
