r"""Bloch-Manifold-Constrained DPS training strategy (Idea 2 / B-2).

Per Phase-0.5 F6 of ``TODO/VF_CAMPAIGN_IMPLEMENTATION_PLAN.md`` this
strategy is ``(BaseTrainingStrategy, KspaceMixin)`` and **composes** the
diffusion sampler / score-net as *models* — it does **not** inherit the
model-side ``DiffusionMixin`` / ``PhysicsMixin``.

The training half is the *standard* DDPM ε-prediction objective on clean
images (Ho et al. 2020): sample ``t``, form
:math:`x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon`,
and minimise :math:`\lVert\hat\varepsilon_\phi(x_t,t)-\varepsilon\rVert^2`.
This mirrors the existing ``diffusion`` strategy's training pipeline; the
**novelty lives entirely in inference**, where
:class:`~spectramr.models.diffusion.bloch_manifold_dps_sampler.BlochManifoldDPSSampler`
interleaves a DPS measurement-likelihood gradient with the per-voxel
Bloch-manifold projection.

The score network is the configured generator. The sampler and projector
are composed lazily (so config-validation construction stays cheap).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.mixins.kspace import KspaceMixin
from spectramr.infrastructure.training.strategies.mixins.utils import _callable_accepts_kwarg

logger = logging.getLogger(__name__)

_EPS = 1e-8


def _cfg_get(block: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a config sub-block that may be a Pydantic model or dict.

    Until the ``training_mode`` discriminated-union dispatch mounts
    :class:`BlochManifoldDPSConfig` (lead-side wiring), the
    ``training.bloch_manifold_dps`` block arrives as a raw dict via the base
    schema's ``extra='allow'``. Once mounted it is a typed model. This
    helper reads either form so the strategy works in both states.
    """
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _linear_beta_schedule(timesteps: int, device: torch.device) -> torch.Tensor:
    scale = 1000.0 / max(timesteps, 1)
    return torch.linspace(scale * 1e-4, scale * 2e-2, timesteps, device=device, dtype=torch.float32)


class BlochManifoldDPSStrategy(BaseTrainingStrategy, KspaceMixin):
    # Debug-snapshot contract (non-negotiable 14): the model is fed q_sample(x0),
    # not the prepared input; declared here and emitted under the tag.
    snapshot_prepared_is_model_input: bool = False
    snapshot_model_input_tag: str | None = "diffusion_step"
    r"""DDPM training + Bloch-manifold-constrained DPS sampling.

    Training is standard ε-prediction; only the sampling/validation path
    is novel. Composes the score network (the generator) with a
    :class:`BlochManifoldDPSSampler` at inference.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        bm = getattr(self.config.training, "bloch_manifold_dps", None)
        if bm is None:
            raise ValueError(
                "BlochManifoldDPSStrategy requires a training.bloch_manifold_dps "
                "block. Use the diffusion strategy if you do not want the "
                "manifold-projection guidance (no silent fallback, CLAUDE.md #9)."
            )
        self._bm_cfg = bm
        self._sampling_steps = int(_cfg_get(bm, "sampling_steps", 1000))

        # Diffusion timesteps: reuse training.diffusion if present.
        diff = getattr(self.config.training, "diffusion", None)
        diff_ts = (
            _cfg_get(diff, "timesteps", self._sampling_steps)
            if diff is not None
            else self._sampling_steps
        )
        self.num_timesteps = int(diff_ts)
        betas = _linear_beta_schedule(self.num_timesteps, self.device)
        self.register_diffusion_buffers(betas)

        self.prediction_type = str(_cfg_get(bm, "prediction_type", "epsilon"))
        self._sampler = None  # lazy — built on first validation/sample call
        # Diffusion strategies own their validation-image logging.
        self.logs_validation_images_in_step = True
        if self.logging_service is not None:
            self.logging_service.log_info(
                "BlochManifoldDPSStrategy initialised: pulse_sequence={}, "
                "sampler={}, zeta={:.3f}, proj_inner={}, T={}.".format(
                    _cfg_get(bm, "pulse_sequence", "spgr"),
                    _cfg_get(bm, "sampler_type", "ddim"),
                    float(_cfg_get(bm, "likelihood_weight_zeta", 1.0)),
                    int(_cfg_get(bm, "projection_inner_iterations", 10)),
                    self.num_timesteps,
                )
            )

    # ------------------------------------------------------------------ #
    # Diffusion buffers
    # ------------------------------------------------------------------ #
    def register_diffusion_buffers(self, betas: torch.Tensor) -> None:
        """Precompute and store the ε-prediction schedule constants."""
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt((1.0 - alphas_cumprod).clamp_min(0.0))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward diffusion: :math:`x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon`."""
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x0.ndim - 1)))
        sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x0.ndim - 1)))
        return sqrt_ab * x0 + sqrt_1mab * noise

    # ------------------------------------------------------------------ #
    # Standard DDPM training step (the non-novel half)
    # ------------------------------------------------------------------ #
    def _to_real_input(self, x: torch.Tensor) -> torch.Tensor:
        """Coerce a (possibly complex / 5D) batch to real model input.

        Complex tensors are stacked to interleaved real/imag channels; a
        trailing singleton depth (TorchIO 5D) is squeezed.

        This DDPM denoises clean *images* (``x0``). If an arm delivers k-space
        (svd coil-processing on ``dataset_type: kspace``) the target is IFFT'd to
        image domain once here, so ``x0`` / the cached REAL reference is a brain
        rather than a centre-bright k-space blob. The decision is the SSOT
        ``needs_ifft_for_visualization`` (see base), so it is a **no-op** for
        ``rss_image`` arms like exp_p2 whose dataset already emits a magnitude
        image — those need no ``target_channels`` change. A k-space-delivering
        Bloch arm must keep the complex pair (``data.target_channels: 2``).
        """
        if x.ndim == 5 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        x = self._ensure_image_domain_target(x)
        if torch.is_complex(x):
            x = torch.view_as_real(x).permute(0, 1, 4, 2, 3).contiguous()
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        return x.to(torch.float32)

    @staticmethod
    def _forward_score(model: Any, x_t: torch.Tensor, t: torch.Tensor) -> Any:
        """Run the score network, passing ``timesteps`` only when it accepts it.

        Dispatch on introspection, never on ``except TypeError`` (SAQ-001). The
        exception cannot distinguish "this backbone has no ``timesteps`` kwarg"
        from "the forward raised a TypeError several frames down"; retrying the
        second as ``model(x_t)`` hid the traceback *and* trained the score
        network unconditioned on t, which still converges to something and so
        never announced itself. ``_callable_accepts_kwarg`` honours ``**kwargs``
        and caches per underlying function, so this costs one signature lookup
        per model, not one per step.
        """
        if _callable_accepts_kwarg(getattr(model, "forward", model), "timesteps"):
            return model(x_t, timesteps=t)
        return model(x_t)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        r"""Standard DDPM ε-prediction loss on clean images.

        The target is the clean image :math:`x_0`; the strategy adds noise
        at a random timestep and trains the score network to predict it.
        """
        x0 = self._to_real_input(target_batch)
        b = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        self._declare_model_input({"x_t": x_t, "t": t})
        pred = self._forward_score(self.generator_model, x_t, t)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if isinstance(pred, dict):
            pred = pred.get("image", pred.get("sample", next(iter(pred.values()))))

        # Channel-align the prediction to the target/noise (models may emit
        # a different channel count; truncate the larger).
        target = noise if self.prediction_type == "epsilon" else x0
        if pred.shape[1] != target.shape[1]:
            c = min(pred.shape[1], target.shape[1])
            pred, target = pred[:, :c], target[:, :c]
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode="nearest")

        loss = F.mse_loss(pred, target)
        return {"g_total_loss": loss, "diffusion_mse": loss.detach()}

    @torch.no_grad()
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        r"""Smoke-grade validation via a single ε-prediction denoise step.

        The inherited base ``validation_step`` calls the score network as
        ``model(x)`` without ``timesteps``, raising
        ``...forward() missing 1 required positional argument: 'timesteps'`` so
        every validation batch failed (exp_p2, dispatch 20260603_200125 task
        20). Full constrained DPS sampling (``reconstruct``) is the faithful
        inference path but runs ~``sampling_steps`` projected iterations per
        batch — far too slow for the per-iteration validation loop. Instead we
        estimate :math:`\hat x_0` from a single ε-prediction at a fixed
        mid-timestep (the exact forward the training objective uses, so it
        tracks training progress) and score it with the SSOT metrics computer,
        so early stopping can resolve the ``val_*`` monitor.
        """
        x0 = self._to_real_input(target_batch.to(self.device))
        b = x0.shape[0]
        # Deterministic mid-timestep — keeps the val metric comparable run-to-run.
        t = torch.full((b,), self.num_timesteps // 2, device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        pred = self._forward_score(self.generator_model, x_t, t)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if isinstance(pred, dict):
            pred = pred.get("image", pred.get("sample", next(iter(pred.values()))))

        # Align prediction channels/shape to x0 (models may emit a different count).
        if pred.shape[1] != x0.shape[1]:
            c = min(pred.shape[1], x0.shape[1])
            pred, x0, x_t = pred[:, :c], x0[:, :c], x_t[:, :c]
        if pred.shape[-2:] != x0.shape[-2:]:
            pred = F.interpolate(pred, size=x0.shape[-2:], mode="nearest")

        if self.prediction_type == "epsilon":
            shape = (-1, *([1] * (x0.ndim - 1)))
            sqrt_ab = self.sqrt_alphas_cumprod[t].view(*shape)
            sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t].view(*shape)
            x0_hat = (x_t - sqrt_1mab * pred) / sqrt_ab.clamp_min(1e-8)
        else:  # "sample"/"x0" prediction — the network already emits x̂₀
            x0_hat = pred

        # Exercise the Bloch-manifold projection in validation so the primary
        # metric reflects the CONSTRAINED sampler (this arm's whole novelty).
        # Full constrained DPS sampling is too slow per-iteration, but a SINGLE
        # projection of the Tweedie estimate x̂₀ is cheap and isolates the
        # constraint's effect. Gated on the arm's projection schedule, so the
        # unconstrained baseline (``projection_every`` ≫ ``sampling_steps``) is
        # graded on the raw estimate — keeping the constrained-vs-unconstrained
        # comparison real. (exp_p2 pitfall #16: the projection was never called
        # from validation, so the arm and its baseline scored identically.)
        proj_every = int(_cfg_get(self._bm_cfg, "projection_every", 1))
        projection_active = proj_every <= self._sampling_steps
        sampler = self.build_sampler()
        with torch.no_grad():
            x0_eval = sampler.project(x0_hat) if projection_active else x0_hat
            residual = float(sampler.manifold_residual(x0_eval))

        computer = self._get_validation_metrics_computer(self.config)
        computed = computer.compute(x0_eval, x0)
        result = {f"val_{k}": float(v) for k, v in computed.items()}
        # Low for the constrained arm (projected onto the manifold), higher for
        # the unconstrained baseline — the headline ``bloch_manifold_residual``.
        result["val_bloch_manifold_residual"] = residual
        return result

    # ------------------------------------------------------------------ #
    # Sampler composition (the novel half)
    # ------------------------------------------------------------------ #
    def build_sampler(self):
        """Build (and cache) the Bloch-manifold-constrained DPS sampler."""
        if self._sampler is None:
            from spectramr.models.diffusion.bloch_manifold_dps_sampler import (
                BlochManifoldDPSSampler,
            )

            bm = self._bm_cfg
            param_bounds = {
                "T1": tuple(_cfg_get(bm, "param_bounds_T1_ms", (100.0, 5000.0))),
                "T2": tuple(_cfg_get(bm, "param_bounds_T2_ms", (10.0, 2000.0))),
                "M0": tuple(_cfg_get(bm, "param_bounds_M0", (0.0, 2.0))),
            }
            self._sampler = BlochManifoldDPSSampler(
                score_network=self.generator_model,
                sampling_steps=int(_cfg_get(bm, "sampling_steps", 1000)),
                sampler_type=str(_cfg_get(bm, "sampler_type", "ddim")),
                likelihood_weight_zeta=float(_cfg_get(bm, "likelihood_weight_zeta", 1.0)),
                prediction_type=str(_cfg_get(bm, "prediction_type", "epsilon")),
                projection_every=int(_cfg_get(bm, "projection_every", 1)),
                projector_kwargs={
                    "pulse_sequence": str(_cfg_get(bm, "pulse_sequence", "spgr")),
                    "param_bounds": param_bounds,
                    "n_inner_iterations": int(_cfg_get(bm, "projection_inner_iterations", 10)),
                    "gn_damping": float(_cfg_get(bm, "projection_gn_damping", 1e-4)),
                },
            ).to(self.device)
        return self._sampler

    @torch.no_grad()
    def reconstruct(
        self,
        measured_kspace: torch.Tensor,
        coil_sensitivities: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        num_steps: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run constrained DPS reconstruction from acquired k-space."""
        sampler = self.build_sampler()
        return sampler.sample(
            measured_kspace,
            coil_sensitivities=coil_sensitivities,
            mask=mask,
            num_steps=num_steps,
        )


__all__ = ["BlochManifoldDPSStrategy"]
