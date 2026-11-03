"""Complex-valued k-space Gaussian diffusion (experiment_47).

The diffusion state is fully sampled complex k-space in the real-interleaved layout
``[Re0, Im0, Re1, Im1, ...]``. The forward process is the DDPM marginal

    ``k_t = sqrt(abar_t) k_0 + sqrt(1 - abar_t) eps``,  ``eps`` i.i.d. N(0, 1) per channel,

i.e. complex white noise. The centred orthonormal FFT is unitary, so ``eps`` is also white
complex noise in the image and the schedule keeps its image-domain meaning while the network
(``complex_unet``) operates on k-space. Training regresses ``eps``.

Sampling starts from noise and is projected onto the measurement ``y`` (mask ``m``) after
every reverse step, ``k <- m * q_sample(y, t_prev) + (1 - m) * k``, with a final hard
replacement ``k <- m * y + (1 - m) * k``. This is the measurement-free prior with hard data
consistency of score-based accelerated MRI (Song et al., ICLR 2022; Chung and Ye, MedIA 2022).

Both ``k_0`` and ``y`` are divided per sample by ``max |ifft2c(y)|``. The scale is taken from
the measurement so training and sampling see the same one, and the prediction is multiplied
back before scoring.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from spectramr.core.module_utils import unwrap_model
from spectramr.data.batch_types import read_batch_field
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.physics.fft_ops import ifft2c

from .diffusion import DiffusionTrainingStrategy
from .mixins.utils import _callable_accepts_kwarg
from .validation_chunking import forward_in_chunks

#: The data-consistency methods this sampler implements.
SUPPORTED_DC_METHODS: frozenset[str] = frozenset({"hard"})


def flat_rows(x: torch.Tensor) -> torch.Tensor:
    """``(B, C, H, W, D)`` -> ``(B*D, C, H, W)``, the order ``_prepare_validation_data`` uses."""
    if x.dim() != 5:
        return x
    b, c, h, w, d = x.shape
    return x.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)


def _as_complex(x: torch.Tensor) -> torch.Tensor:
    """Real-interleaved ``(B, 2C, H, W)`` -> complex ``(B, C, H, W)``; raises on an odd count."""
    if x.dim() != 4 or x.shape[1] % 2 != 0:
        raise ValueError(
            "complex k-space diffusion expects real-interleaved (B, 2C, H, W) k-space; "
            f"got {tuple(x.shape)}."
        )
    return torch.complex(x[:, 0::2], x[:, 1::2])


def measurement_scale(y: torch.Tensor, floor: float = 1e-8) -> torch.Tensor:
    """Per-sample ``max |ifft2c(y)|`` (coil-RSS), shaped ``(B, 1, 1, 1)``."""
    image = ifft2c(_as_complex(y))
    rss = image.abs().pow(2).sum(dim=1).sqrt()
    return rss.flatten(1).amax(dim=1).clamp_min(floor).view(-1, 1, 1, 1)


class ComplexKspaceDiffusionStrategy(DiffusionTrainingStrategy):
    """Epsilon-prediction DDPM on complex k-space with hard DC in the reverse chain."""

    parallel_compat = frozenset({"replicated"})

    #: Loss ownership (issue #1918). The epsilon MSE is the whole objective, computed inline.
    #: A builder image loss has no clean image to meet here, so none is folded.
    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = False
    empty_loss_stack_is_design: ClassVar[bool] = True

    @classmethod
    def standalone_infer_blocker(cls, config: object) -> str | None:
        return (
            "its sampler needs the acquisition mask and the measurement scale "
            "max |ifft2c(y)| and re-imposes the noised measurement after every reverse step; "
            "the generic diffusion inference path does none of this, so arms are scored by "
            "the training-time validation/test pass"
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._resolved_kspace_diffusion = self._resolve_settings()
        self.logging_service.log_info(
            "[ComplexKspaceDiffusion] resolved: "
            + ", ".join(f"{k}={v}" for k, v in self._resolved_kspace_diffusion.items())
        )

    def _resolve_settings(self) -> dict[str, Any]:
        """Read and validate every knob this strategy depends on; raise on anything else."""
        cfg = self.config
        if self._declared_prediction_type() != "epsilon":
            raise ConfigurationError(
                "ComplexKspaceDiffusionStrategy trains epsilon-prediction only; declare "
                "training.diffusion.prediction_type: epsilon."
            )
        if cfg.training.diffusion.condition_on_input is True:
            raise ConfigurationError(
                "ComplexKspaceDiffusionStrategy is a measurement-free prior steered by data "
                "consistency; training.diffusion.condition_on_input: true is not implemented."
            )
        output_domain = str(getattr(cfg.training, "output_domain", "") or "").lower()
        if output_domain != "kspace":
            raise ConfigurationError(
                "ComplexKspaceDiffusionStrategy predicts k-space; training.output_domain must "
                f"be 'kspace', got {output_domain!r}."
            )
        dc = cfg.physics.data_consistency
        method = str(dc.method).lower()
        if not dc.enabled or method not in SUPPORTED_DC_METHODS:
            raise ConfigurationError(
                "ComplexKspaceDiffusionStrategy projects onto the measurement at every reverse "
                "step; it needs physics.data_consistency.enabled: true and method in "
                f"{sorted(SUPPORTED_DC_METHODS)}, got enabled={dc.enabled}, method={method!r}."
            )
        return {
            "prediction_type": "epsilon",
            "forward_process": "complex_white_gaussian_kspace",
            "dc_method": method,
            "scale": "max_abs_ifft2c_measurement",
            "timesteps": int(self.num_timesteps),
            "noise_schedule": str(self.beta_schedule),
        }

    def _call_generator(self, gen: Any, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``gen(x_t, t)``, with the horizon passed when the forward takes it."""
        fwd = getattr(unwrap_model(gen), "forward", gen)
        if _callable_accepts_kwarg(fwd, "max_timesteps"):
            out = gen(x_t, t, max_timesteps=self.num_timesteps)
        else:
            out = gen(x_t, t)
        if isinstance(out, tuple):
            out = out[0]
        if out.shape != x_t.shape:
            raise ValueError(
                f"complex k-space diffusion: the generator returned {tuple(out.shape)} for a "
                f"{tuple(x_t.shape)} state; epsilon must have the state's shape."
            )
        return out

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        gen = getattr(self.env, "generator", None)
        k0 = read_batch_field(batch, "target")
        y = read_batch_field(batch, "input")
        if gen is None or k0 is None or y is None:
            raise ValueError(
                "complex k-space diffusion needs env.generator, batch['target'] (fully sampled "
                "k-space) and batch['input'] (the measurement); got "
                f"generator={gen is not None}, target={k0 is not None}, input={y is not None}."
            )
        k0 = flat_rows(k0).to(self.device, non_blocking=True)
        y = flat_rows(y).to(self.device, non_blocking=True)
        scale = measurement_scale(y)
        x0 = k0 / scale
        t = self.scheduler.sample_timesteps(x0.shape[0])
        eps = torch.randn_like(x0)
        x_t = self.scheduler.q_sample(x0, t, eps)
        self._declare_model_input(
            {"model_input": x_t, "target": x0},
            in_kspace_keys={"model_input", "target"},
            extra={
                "model_input_key": "model_input",
                "provenance": dict(self._resolved_kspace_diffusion),
                "note": (
                    "The network sees k_t = sqrt(abar_t) k_0/s + sqrt(1 - abar_t) eps, formed "
                    "inside this step, with s = max |ifft2c(y)|; 'first_steps/input_prepared' "
                    "is the loader's measurement, not the model input."
                ),
            },
        )
        eps_hat = self._call_generator(gen, x_t, t)
        loss = F.mse_loss(eps_hat, eps)
        return {"loss_total": loss, "loss_eps_kspace": loss.detach()}

    @torch.no_grad()
    def sample_with_dc(self, gen: Any, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reverse-sample k-space pinned to ``y`` on ``mask``; returns k-space at y's scale.

        The parent's chain runs it, with each network call chunked by
        ``validation.loader.chunk_size`` so a flattened volume never meets the U-Net at once.
        """
        scale = measurement_scale(y)
        y_n = y / scale
        keep = 1.0 - mask
        chunk = int(self.config.validation.loader.chunk_size)

        def project(x: torch.Tensor, t_prev: int) -> torch.Tensor:
            if t_prev < 0:
                return mask * y_n + keep * x
            t_vec = torch.full((x.shape[0],), t_prev, dtype=torch.long, device=x.device)
            return mask * self.scheduler.q_sample(y_n, t_vec) + keep * x

        def forward(x: torch.Tensor, t: torch.Tensor, _latent: bool, kw: dict) -> torch.Tensor:
            return forward_in_chunks(
                lambda xc, tc, _kw: self._call_generator(gen, xc, tc), x, t, kw, chunk
            )

        x = self._sample_pixel_epsilon(y_n, {}, y_n.shape, forward, project=project)
        return x * scale

    def _generate_validation_prediction(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        timestep: torch.Tensor,
        batch_data: Any,
        kwargs: dict,
        *,
        start_timestep: int | None = None,
    ) -> torch.Tensor:
        """Sample from noise under hard DC on the loader's measurement and mask."""
        del target_batch, timestep, kwargs, start_timestep
        # The parent's per-batch scoring state; the zero-filled baseline is this measurement.
        self._zf_measurement = self._rung_mask = None
        self._rung_acquired = self._rung_measured_maps = self._ensemble_members = None
        mask = read_batch_field(batch_data, "mask") if batch_data is not None else None
        if mask is None:
            raise ValueError(
                "complex k-space diffusion validation needs the loader's sampling mask "
                "(batch['mask']); it is not inferred from the measurement's non-zeros."
            )
        mask = flat_rows(mask).to(device=input_batch.device, dtype=input_batch.dtype)
        if mask.shape[0] != input_batch.shape[0] or mask.shape[-2:] != input_batch.shape[-2:]:
            raise ValueError(
                f"sampling mask {tuple(mask.shape)} does not align with the measurement "
                f"{tuple(input_batch.shape)}."
            )
        self._zf_measurement, self._rung_mask = input_batch, mask
        return self.sample_with_dc(unwrap_model(self.generator_model), input_batch, mask)

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        batch_data: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """One level, the loader's acceleration: the sampler conditions on that measurement only.

        The parent's acceleration cascade re-labels the same 4x measurement as 2x/8x/32x on a
        non-cold arm, so it is not run here.
        """
        prep = self._prepare_validation_data(
            (input_batch, target_batch), input_batch, target_batch, batch_data
        )
        input_batch, target_batch, scale_factor = prep
        self.generator_model.eval()
        with torch.no_grad():
            timestep = torch.zeros(input_batch.shape[0], dtype=torch.long, device=self.device)
            prediction = self._generate_validation_prediction(
                input_batch, target_batch, timestep, batch_data, kwargs
            )
            return self._compute_validation_metrics(
                prediction,
                target_batch,
                input_batch,
                timestep,
                batch_data,
                scale_factor,
                batch_idx=kwargs.get("batch_idx", 0),
            )
