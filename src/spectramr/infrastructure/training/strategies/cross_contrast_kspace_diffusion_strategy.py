"""Cross-Contrast k-Space Cold Diffusion (Integration plan idea 6).

Plan: TODO/integration_plan_ulf_cheap_fast_mri.md §6.

Cold-diffusion in the **frequency domain** that bridges contrasts
(e.g. T1 ↔ T2) without a lossy intermediate image representation.

Forward corruption schedule for ``α(t) ∈ [0, 1]``:

    ``k_t = (1 − α(t)) · k_src + α(t) · k_dst + σ(t) · ε``

with ``ε ~ CN(0, I)`` and a small noise ``σ(t)``.  The reverse step is
the standard cold-diffusion update

    ``k_{t-1} = k_t − D(k_t, t) + D̂(k_t, t)``

where ``D̂`` is the parameterised "destination predictor" — the
generator network in this strategy.

Wired components:

- :meth:`forward_corrupt`         — analytic corruption ``α(t)`` schedule.
- :meth:`reverse_step`            — single cold-diffusion reverse step.
- :meth:`_compute_losses_impl`    — DSM-style L1 between predicted
  destination and the ground-truth ``k_dst`` plus a per-contrast
  weighted MSE on the residual ``k_t − k_src``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from spectramr.data.batch_types import read_batch_field

from .diffusion import DiffusionTrainingStrategy
from .mixins.utils import _callable_accepts_kwarg


class CrossContrastKspaceDiffusionStrategy(DiffusionTrainingStrategy):
    """Cold-diffusion in k-space for cross-contrast translation."""

    #: Loss ownership (issue #1918). mse_loss(dst_hat, k_dst) is a K-SPACE term against the destination contrast,
    #: not an image loss against the target, and the hook routes nowhere.
    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = False

    sigma_max: float = 0.05
    lambda_destination: float = 1.0
    lambda_residual: float = 0.5

    @staticmethod
    def alpha_schedule(t: torch.Tensor) -> torch.Tensor:
        return t.clamp(0.0, 1.0)

    def forward_corrupt(
        self,
        k_src: torch.Tensor,
        k_dst: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        a = self.alpha_schedule(t).view(-1, *([1] * (k_src.dim() - 1)))
        if eps is None:
            eps_re = torch.randn_like(k_src.real if torch.is_complex(k_src) else k_src)
            eps_im = torch.randn_like(k_src.real if torch.is_complex(k_src) else k_src)
            eps = torch.complex(eps_re, eps_im) if torch.is_complex(k_src) else eps_re
        sigma_t = self.sigma_max * (a * (1 - a)) * 4.0
        return (1 - a) * k_src + a * k_dst + sigma_t * eps

    @torch.no_grad()
    def reverse_step(
        self,
        k_t: torch.Tensor,
        t: torch.Tensor,
        predict_dst: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        dt: float = 0.05,
    ) -> torch.Tensor:
        a = self.alpha_schedule(t).view(-1, *([1] * (k_t.dim() - 1)))
        a_prev = (a - dt).clamp(0.0, 1.0)
        dst_hat = predict_dst(k_t, t)
        # Move along the schedule by one dt, replacing the destination component.
        return k_t + (a_prev - a) * (dst_hat - (k_t - (1 - a) * 0))

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # F6 (round 6 2026-05-17): orchestrator-kwargs canonical signature.
        # F8b inheritance: read generator from self.env.generator
        # (the StrategyContext dataclass has no ``generator`` field).
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        gen = getattr(self.env, "generator", None)
        if gen is None:
            raise RuntimeError(
                "cross_contrast_kspace_diffusion strategy: env.generator is "
                "None — the DI container did not produce a generator. Check "
                "the ModelBuilder wiring."
            )

        # The cross-contrast bridge is DEFINED on a source/destination k-space
        # pair: `forward_corrupt` interpolates between them. Falling back to the
        # parent here trained a plain reconstruction objective under a
        # cross-contrast arm name and reported success (audit A5; pitfall #16) —
        # the run looked healthy and the mechanism the arm exists to test never
        # fired. Raise instead (pitfall #9).
        k_src = read_batch_field(batch, "kspace_source")
        k_dst = read_batch_field(batch, "kspace_target")
        if k_src is None or k_dst is None:
            # This used to fall through to the parent's plain reconstruction
            # objective, which is the facade shape (pitfall #16): the arm keeps
            # its cross-contrast name, smoke-passes, and trains a vanilla
            # denoiser. No dataset under src/spectramr/data/ emits either key, so
            # the fallthrough was not a rare edge case -- it was the only path.
            missing = [
                name
                for name, value in (("kspace_source", k_src), ("kspace_target", k_dst))
                if value is None
            ]
            raise ValueError(
                "cross_contrast_kspace_diffusion requires batch['kspace_source'] "
                f"and batch['kspace_target'] (paired per-contrast k-space); {missing!r} "
                "missing. The batch supplies "
                f"{sorted(batch) if isinstance(batch, dict) else type(batch).__name__!r}. "
                "No dataset currently produces these keys; a cross-contrast paired "
                "k-space loader must emit them before this strategy can run. Falling "
                "back to the plain reconstruction loss would train a vanilla denoiser "
                "under this arm's name."
            )

        B = k_src.shape[0]
        t = torch.rand(B, device=k_src.device)
        contrast_idx = batch.get("contrast_idx")
        k_t = self.forward_corrupt(k_src, k_dst, t)

        # The bridge interpolant is what the model sees, and it is formed here --
        # ``first_steps/input_prepared`` is captured before the forward pass and
        # holds neither endpoint. Declare the real one; the wrapper emits it
        # under ``diffusion_step`` (non-negotiable 14).
        self._declare_model_input(
            {
                "kspace_interpolant": k_t,
                "kspace_source": k_src,
                "kspace_target": k_dst,
            },
            # Named explicitly rather than left to the config union: this
            # strategy RAISES unless the batch supplies paired k-space (above),
            # so all three are k-space by its own contract, whatever
            # ``data.dataset_type`` happens to say.
            in_kspace_keys={"kspace_interpolant", "kspace_source", "kspace_target"},
            extra={
                "model_input_key": "kspace_interpolant",
                "note": (
                    "Cross-contrast feeds forward_corrupt(k_src, k_dst, t) -- the "
                    "bridge interpolant between the source and destination "
                    "contrasts, formed inside this step. Seeing it beside both "
                    "endpoints is what shows the bridge fired rather than the "
                    "arm collapsing to a plain denoiser (audit A5)."
                ),
            },
        )

        # Introspection, not try/except (SAQ-001): `except TypeError` cannot tell
        # "this generator has no contrast_idx kwarg" from "the forward raised a
        # TypeError three frames down". The second is a real bug, and swallowing
        # it silently retrains the arm as an unconditioned denoiser -- the same
        # facade shape the k-space guard above raises on. `_callable_accepts_kwarg`
        # honours `**kwargs` and caches per underlying function.
        #
        # `getattr(gen, "forward", gen)` matches `_generator_accepts_time`: an
        # nn.Module must be introspected through `.forward` (its `__call__` is
        # `(*args, **kwargs)`, so VAR_KEYWORD would answer True for everything),
        # while a plain callable's own signature IS its forward signature.
        if _callable_accepts_kwarg(getattr(gen, "forward", gen), "contrast_idx"):
            dst_hat = gen(k_t, t, contrast_idx=contrast_idx)
        else:
            dst_hat = gen(k_t, t)

        if torch.is_complex(dst_hat):
            destination = (dst_hat - k_dst).abs().pow(2).mean()
        else:
            destination = F.mse_loss(dst_hat, k_dst.real if torch.is_complex(k_dst) else k_dst)
        destination = self.lambda_destination * destination

        # Residual fidelity: corruption-removed k-space should match k_src*(1-a).
        a = self.alpha_schedule(t).view(-1, *([1] * (k_t.dim() - 1)))
        residual_target = (1 - a) * k_src
        residual_pred = k_t - a * dst_hat
        if torch.is_complex(residual_pred):
            residual = (residual_pred - residual_target).abs().pow(2).mean()
        else:
            residual = F.mse_loss(
                residual_pred,
                (residual_target.real if torch.is_complex(residual_target) else residual_target),
            )
        residual = self.lambda_residual * residual

        out: dict[str, torch.Tensor] = {
            "loss_destination": destination,
            "loss_residual": residual,
            "loss_total": destination + residual,
        }
        # Per-contrast diagnostic breakdown is GPU-sync heavy (.tolist(), .any(),
        # .sum()) so we only emit it at log-interval steps to keep the hot path
        # sync-free. CLAUDE.md "Training-loop performance rules" forbid these
        # syncs in the inner loop.
        log_interval = getattr(self, "_cached_log_interval", None) or 100
        current_step = int(kwargs.get("iteration", 0))
        if contrast_idx is not None and current_step % log_interval == 0:
            # Detach once — diagnostics never need gradient.
            sq_err = (dst_hat - k_dst).abs().pow(2).detach()
            # Mean over non-batch dims so we have a scalar per sample.
            per_sample = sq_err.flatten(1).mean(dim=1) if sq_err.dim() > 1 else sq_err
            for c in contrast_idx.unique().tolist():
                mask = contrast_idx == c
                count = mask.sum().clamp(min=1)
                out[f"loss_contrast_{int(c)}"] = (per_sample * mask.float()).sum() / count
        return out
