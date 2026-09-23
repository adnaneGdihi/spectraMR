"""Cold Diffusion Inference Strategy

Dedicated strategy for Cold Diffusion models, particularly those for MRI reconstruction.
Avoids standard DDPM sampling and image-domain normalization that can corrupt k-space data.
"""

import logging
from typing import Any

import torch
from torch import nn

from spectramr.config.schemas.enums import TrainingModeTypes
from spectramr.core.module_utils import unwrap_model
from spectramr.data.transforms.normalization import KSpaceNormalizationSpec
from spectramr.infrastructure.physics.coil_sensitivity import (
    estimate_smaps_calibrated,
    prepare_smaps_for_kspace_conditioning,
    resolve_estimation_settings,
)
from spectramr.infrastructure.physics.sampling import create_kspace_accelerator
from spectramr.models.diffusion.kspace_process import (
    inject_reverse_step_noise,
    validate_sampler_determinism,
)

from .base_inference_strategy import BaseInferenceStrategy

logger = logging.getLogger(__name__)


class ColdDiffusionInferenceStrategy(BaseInferenceStrategy):
    """Inference strategy for Cold Diffusion models.

    Implements the iterative 'repairs' logic of Cold Diffusion:
    x_{t-1} = D(x_0_hat, t-1) + (x_t - D(x_0_hat, t))
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ):
        """__init__.

        Args:
            model (nn.Module): Description.
            device (torch.device): Description.
            config (Optional[dict[str, Any]]): Description.
        """
        super().__init__(model, device, config)

        # Extract diffusion config
        # Handle various config locations
        self.diffusion_config = self.config.get("diffusion") or {}
        if not self.diffusion_config and "training" in self.config:
            training_cfg = self.config["training"]
            if isinstance(training_cfg, dict):
                self.diffusion_config = training_cfg.get("diffusion") or {}

        self.num_timesteps = self.diffusion_config.get("timesteps", 1000)
        self.sampling_steps = self.diffusion_config.get("sampling_steps", 50)
        # Only k-space masking is implemented by the reverse loop below. Per
        # pitfall #15 an advertised-but-unimplemented degradation must RAISE,
        # not silently no-op: a YAML asking for ``degradation: blur`` previously
        # set this attribute and was then ignored, so the run masked k-space
        # while the config author believed a blur schedule was driving it.
        self.degradation_type = self.diffusion_config.get("degradation", "kspace_mask")
        if self.degradation_type != "kspace_mask":
            raise ValueError(
                "ColdDiffusionInferenceStrategy only implements the "
                f"'kspace_mask' degradation; got {self.degradation_type!r}. "
                "Remove the 'degradation' key or implement the schedule."
            )

        # C6 determinism contract, read from the SAME model.model_kwargs keys the
        # generator-side sampler uses, validated identically at construction.
        sampler_kwargs = self.config.get("model", {}).get("model_kwargs", {}) or {}
        self.sampler_sigma = float(sampler_kwargs.get("sampler_sigma", 0.0))
        seed = sampler_kwargs.get("sampler_seed")
        self.sampler_seed = None if seed is None else int(seed)
        self.selection_rule = sampler_kwargs.get("selection_rule", "fixed")
        validate_sampler_determinism(self.sampler_sigma, self.selection_rule)
        self._sampler_generator: torch.Generator | None = None
        if self.sampler_sigma > 0:
            # Warn ONCE at construction, not per inference call (A9/C7): σ>0
            # changes the score distribution, so any conformal trust calibration
            # fitted at a different σ (or seed policy) is no longer exchangeable
            # with these reconstructions and must be refitted.
            logger.warning(
                "sampler_sigma=%.4g > 0: reconstructions carry seeded reverse-step "
                "noise (sampler_seed=%s). Conformal trust calibrations fitted at a "
                "different sigma/seed are not exchangeable with these outputs — "
                "recalibrate before reusing any trust certificate.",
                self.sampler_sigma,
                self.sampler_seed,
            )

        self.accelerator = self._resolve_accelerator()

    def _resolve_accelerator(self):
        """Resolve the k-space accelerator that drives the reverse loop.

        The reverse cold-diffusion loop MUST degrade with the *same* accelerator
        the generator trained on (identical ``mask_type`` / ``seed`` /
        ``center_fraction`` / schedule). Degrading with a different pattern
        restores k-space from masks the network never learned to invert, so the
        iterate wanders off-manifold toward a measurement-independent DC blob —
        the 2026-06-09 training-side fix (``project_exp11_true_cold_diffusion_fix``)
        that the inference path had never received.

        Resolution order (SSOT first, loose config last):
          1. ``model.accelerator`` — the ``cold_diffusion.py`` model carries one.
          2. ``model.kspace_process`` — the ``KSpaceColdDiffusionGenerator`` SSOT
             holds its trained forward process here (NOT under ``accelerator``);
             reuse its own cached, trained accelerator object.
          3. a fresh accelerator built from explicit ``acceleration`` config.

        Returns:
            The accelerator exposing ``get_acceleration_mask``, or ``None``.
        """
        model_accelerator = getattr(self.model, "accelerator", None)
        if model_accelerator is not None:
            return model_accelerator

        kspace_process = getattr(self.model, "kspace_process", None)
        if kspace_process is not None:
            mask_generator = getattr(kspace_process, "mask_generator", None)
            mask_type = getattr(kspace_process, "mask_type", None)
            if mask_generator is not None and hasattr(mask_generator, "_get_accelerator"):
                return mask_generator._get_accelerator(mask_type)

        accel_config = (
            self.config.get("acceleration")
            or self.config.get("acceleration_config")
            or self.diffusion_config.get("acceleration")
        )
        if not isinstance(accel_config, dict):
            return None

        accel_kwargs = dict(accel_config)
        acceleration_type = accel_kwargs.pop(
            "acceleration_type", accel_kwargs.pop("type", "variable_density")
        )
        accel_kwargs.pop("num_timesteps", None)

        try:
            return create_kspace_accelerator(
                acceleration_type=acceleration_type,
                num_timesteps=self.num_timesteps,
                **accel_kwargs,
            )
        except Exception as exc:
            logger.warning("Failed to build k-space accelerator: %s", exc)
            return None

    @staticmethod
    def _assert_trained_width(model_input: torch.Tensor, generator: object) -> None:
        """Fail loudly when the sampled stack is not the width training used.

        The reverse loop's failure mode without this check is invisible:
        ``FourierBridgeNetwork`` rebuilds its ``ChannelAdapter`` for whatever
        channel count arrives, so a mismatched stack is silently squeezed
        through an untrained 1x1 convolution and the reconstruction merely
        looks bad (#1326, pitfalls #9/#16).  A raise converts that into a
        diagnosable error.

        Only asserted for an even ``in_channels``.  A real-interleaved arm
        pairs ``C`` data channels with ``C`` map channels, giving ``2 * C`` --
        but an odd ``in_channels`` (the two ``in_channels: 1`` arms) is a real
        magnitude field that the estimation block lifts to *one* complex coil,
        so its trained stack is ``1 + 2 = 3``, not ``2``.  Encoding the even
        case only keeps the guard exact rather than approximately right.

        The expected width is **not** unconditionally ``2 * in_channels``: an
        internal-DC backbone is fitted at ``1x`` and receives no maps, so the
        doubling is gated on the same resolver every other site uses. Hard-coding
        it made this guard a fourth owner of the width rule, and the one that
        would have fired on the six ``diff_varnet``/``diff_varnet_kan`` arms
        precisely when they were finally being sampled correctly.
        """
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            model_expects_smaps_concat,
        )

        in_channels = getattr(generator, "in_channels", None)
        if not isinstance(in_channels, int) or in_channels % 2 or model_input.dim() != 4:
            return
        expected = 2 * in_channels if model_expects_smaps_concat(generator) else in_channels
        if model_input.shape[1] != expected:
            raise ValueError(
                f"[ColdDiffusionInference] S-map conditioning produced a "
                f"{model_input.shape[1]}-channel stack, but the network was "
                f"trained on {expected} (= 2 x in_channels={in_channels}). "
                "Sampling with the wrong width does not raise inside the "
                "backbone -- it silently builds an untrained ChannelAdapter "
                "(#1326). Check that the input tensor's channel count matches "
                "model.in_channels."
            )

    def _expand_mask(self, mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """_expand_mask.

        Args:
            mask (torch.Tensor): Description.
            target (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(0)
        if mask.shape[1] == 1 and target.shape[1] > 1:
            mask = mask.expand(target.shape[0], target.shape[1], -1, -1)
        elif mask.shape[0] != target.shape[0]:
            mask = mask.expand(target.shape[0], *mask.shape[1:])
        return mask

    @property
    def training_mode(self) -> TrainingModeTypes:
        """training_mode.

        Returns:
            TrainingModeTypes: Description.
        """
        return TrainingModeTypes.DIFFUSION

    @property
    def kspace_norm(self) -> KSpaceNormalizationSpec:
        """The SSOT k-space normalization spec resolved from the run config.

        Resolved from the SAME knobs training used (``kspace_percentile`` /
        ``log_scaling`` / ``kspace_scale_domain``). Before issue #572 this path
        read ``normalization_kwargs`` — the IMAGE-normalization block, a
        different percentile — and never applied ``log_scaling``, so a model
        trained on log-compressed k-space was fed uncompressed k-space and its
        output decoded with the wrong inverse.
        """
        spec = getattr(self, "_kspace_norm", None)
        if spec is None:
            spec = KSpaceNormalizationSpec.from_data_config(self.config.get("data", {}))
            self._kspace_norm = spec
        return spec

    def compute_kspace_scale(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Robust scale for a batched ``(B, C, H, W)`` k-space tensor."""
        return self.kspace_norm.compute_scale(input_tensor, channel_dim=1)

    def denormalize_kspace(self, output_tensor: torch.Tensor) -> torch.Tensor:
        """Undo :meth:`preprocess_input` — decompress, then restore the scale."""
        scale = getattr(self, "_last_scale", None)
        if scale is None:
            return output_tensor
        return self.kspace_norm.denormalize(output_tensor, scale, channel_dim=1)

    def preprocess_input(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Preprocess k-space/image input.

        Applies exactly the normalization the training transform applied, so the
        network sees the distribution it was trained on (issue #572).
        """
        input_tensor = input_tensor.to(self.device).detach()

        spec = self.kspace_norm
        if spec.enabled:
            # channel_dim=1: inference tensors are (B, C, H, W) with real/imag
            # interleaved along C. The scale therefore uses EVERY coil, not the
            # first one -- the old code read channels 0 and 1 only.
            normed, scale = spec.normalize(input_tensor, channel_dim=1)
            self._last_scale = scale
            logger.info(
                "Normalized input k-space: scale=%.4f (p%.0f, domain=%s, log=%s)",
                float(scale),
                spec.percentile * 100,
                spec.scale_domain,
                spec.log_scaling,
            )
            return normed

        return input_tensor

    def run_inference(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run Cold Diffusion sampling loop.

        For reconstruction, input_tensor is treated as the initial (fully degraded) state x_T.
        """
        self.model.eval()

        # Opt-in chainwise monitoring hook (papers' trajectory trust layer).
        # Popped BEFORE the loop so it is never forwarded to the model as a
        # model kwarg. Fires (step_idx, pred_x0, current_mask) once per strided
        # step AFTER data consistency (and C6 noise), in the NORMALIZED k-space
        # domain — κ_s is relative, so the scale cancels. Default None keeps
        # this loop byte-identical to the pre-hook behavior.
        step_callback = kwargs.pop("step_callback", None)

        # C6: fresh noise stream per call — reseeding at entry (never per step)
        # makes this loop a deterministic function of (input, mask, sampler_seed).
        if self.sampler_sigma > 0:
            self._sampler_generator = torch.Generator(device="cpu")
            if self.sampler_seed is not None:
                self._sampler_generator.manual_seed(self.sampler_seed)
            else:
                # A fresh torch.Generator has a FIXED default state; None must
                # mean genuinely nondeterministic, so seed from system entropy.
                self._sampler_generator.seed()

        # x_t starts as the undersampled input
        x_t = input_tensor.clone()
        measurement_mask = kwargs.get("mask")
        if measurement_mask is not None:
            measurement_mask = measurement_mask.to(self.device)

        # Cold Diffusion sampling loop (often using a stride)
        stride = self.num_timesteps // self.sampling_steps
        timesteps = list(range(0, self.num_timesteps, stride))[::-1]

        # [STATIC S-MAP CONDITIONING]
        # Gate on the GENERATOR'S OWN conditioning contract -- never on a magic
        # channel count.  Training and validation concatenate S-maps for *every*
        # ``kspace_cold_diffusion`` arm (``_prepare_diffusion_inputs`` /
        # ``_generate_validation_prediction``), so the width the network was
        # fitted on is ``2 * in_channels`` whenever ``condition_with_smaps`` is
        # on -- and it defaults to on.  The retired ``model.in_channels == 16``
        # gate matched only the four 16-channel arms in the corpus; the other 91
        # were sampled with half the trained width.  That does not raise:
        # ``FourierBridgeNetwork`` lazily *rebuilds* its ``ChannelAdapter`` to
        # fit whatever arrives, so the whole reverse loop ran through a
        # randomly-initialised, never-trained 1x1 projection (#1326, pitfalls
        # #9/#16).  ``energy_probe``/``forward_probe`` ask the same resolver
        # -- this is a further reader of one rule, not a new spelling.
        #
        # It must be the RESOLVED contract, not the arm's declaration: the six
        # ``diff_varnet``/``diff_varnet_kan`` arms declare
        # ``condition_with_smaps: true`` and are still built at ``1x``, because
        # those backbones run their own data consistency internally. Gating on
        # the declaration handed them a ``2x`` stack.
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            model_expects_smaps_concat,
        )

        generator = unwrap_model(self.model)
        smaps = None
        if model_expects_smaps_concat(generator):
            acs_kspace_t = input_tensor
            h, w = acs_kspace_t.shape[-2], acs_kspace_t.shape[-1]
            if not torch.is_complex(acs_kspace_t):
                b, c = acs_kspace_t.shape[0], acs_kspace_t.shape[1]
                c2 = c // 2
                if c2 < 1:
                    # Single-channel magnitude data: treat as 1-channel complex
                    # with a zero imaginary part.  Mirrors the validation path
                    # exactly; without it the two ``in_channels: 1`` arms reach
                    # ``view(b, 0, 2, h, w)`` and die inside view_as_complex.
                    acs_kspace_t = torch.complex(acs_kspace_t, torch.zeros_like(acs_kspace_t))
                else:
                    acs_kspace_t = torch.view_as_complex(
                        acs_kspace_t.view(b, c2, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()
                    )

            # Configured method + sub-knobs, exactly as training/validation
            # resolve them.  Hardcoding ``estimate_csm_power_iter`` here made
            # ``physics.coil_processing.estimation`` a silent no-op at sampling
            # time (non-negotiable 8 / pitfall #15) -- 59 of the 95 cold-
            # diffusion arms declare that block and one of them asks for
            # ``espirit``.  The ACS confinement binds harder here than in
            # validation: validation calibrates from the fully-sampled
            # reference, while sampling only ever sees the undersampled input,
            # so the aliased periphery must be excluded before calibration.
            _method, _est_kwargs = resolve_estimation_settings(self.config)
            smaps = estimate_smaps_calibrated(
                acs_kspace_t,
                method=_method,
                out_size=(h, w),
                **_est_kwargs,
            )

            # Domain translation
            if not torch.is_complex(x_t) and torch.is_complex(smaps):
                smaps = (
                    torch.view_as_real(smaps)
                    .permute(0, 1, 4, 2, 3)
                    .reshape(smaps.shape[0], -1, *smaps.shape[2:])
                )
            elif torch.is_complex(x_t) and not torch.is_complex(smaps):
                smaps = torch.complex(smaps, torch.zeros_like(smaps))

        logger.info(
            f"Starting Cold Diffusion sampling with {len(timesteps)} steps (stride={stride})."
        )

        with torch.no_grad():
            for i, step_idx in enumerate(timesteps):
                t = torch.full((x_t.shape[0],), step_idx, device=self.device, dtype=torch.long)
                kspace_shape = x_t.shape[1:]
                current_mask = None
                if self.accelerator is not None:
                    current_mask = self.accelerator.get_acceleration_mask(
                        kspace_shape, step_idx, device=self.device
                    )
                    current_mask = self._expand_mask(current_mask, x_t)

                # 1. Predict x_0 from current state x_t
                model_kwargs = dict(kwargs)
                if current_mask is not None:
                    model_kwargs["mask"] = current_mask
                model_kwargs.setdefault("kspace_measured", input_tensor)

                model_input = x_t
                if smaps is not None:
                    # [DOMAIN] ``x_t`` is k-space; the maps are image-domain.
                    # Mirror the training/validation path in
                    # ``DiffusionTrainingStrategy`` exactly -- FFT the maps,
                    # match their level to ``x_t`` and cap their amplitude --
                    # or sampling feeds the network a stack the training run
                    # never showed it.  Prepared per step, because ``x_t`` (the
                    # level reference) changes as the reverse process runs.
                    smaps_k, _ = prepare_smaps_for_kspace_conditioning(
                        smaps, model_input, channel_dim=1
                    )
                    model_input = torch.cat([model_input, smaps_k], dim=1)
                    self._assert_trained_width(model_input, generator)

                model_out = self.model(model_input, t, **model_kwargs)
                if isinstance(model_out, tuple):
                    pred_x0 = model_out[0]
                else:
                    pred_x0 = model_out

                # 2. Apply Hard Data Consistency if mask available
                if measurement_mask is not None:
                    measurement_mask = self._expand_mask(measurement_mask, pred_x0)
                    pred_x0 = pred_x0 * (1.0 - measurement_mask) + input_tensor * measurement_mask

                # C6: optional reverse-step noise AFTER the hard-DC pin, masked
                # off the observed support so measured lines stay exactly
                # data-consistent. With no measurement_mask nothing is pinned
                # (the DC step above is skipped too), so the noise covers all of
                # k-space. σ=0 (default): no draw, byte-identical loop.
                if self.sampler_sigma > 0:
                    pred_x0 = inject_reverse_step_noise(
                        pred_x0,
                        self.sampler_sigma,
                        self._sampler_generator,
                        exclude_support=measurement_mask,
                    )

                if step_callback is not None:
                    step_callback(step_idx, pred_x0, current_mask)

                if step_idx == 0:
                    x_t = pred_x0
                    break

                # 3. Step logic: x_{t-1} = D(pred_x0, t-1)
                # For k-space masking, D(x, t) = x * mask_t
                next_step = timesteps[i + 1] if i + 1 < len(timesteps) else 0
                if self.accelerator is not None:
                    next_mask = self.accelerator.get_acceleration_mask(
                        kspace_shape, next_step, device=self.device
                    )
                    next_mask = self._expand_mask(next_mask, pred_x0)
                    x_t = pred_x0 * next_mask
                else:
                    x_t = pred_x0

                if i % 10 == 0:
                    logger.debug(f"Sampling step {i}/{len(timesteps)} (t={step_idx})")

        # The loop pinned the measured bins at every step (the last included), so
        # the predict-time hook in ``BaseInferenceStrategy.infer`` must not
        # project a second time; it records who did instead.
        if measurement_mask is not None:
            self.note_data_consistency_applied("ColdDiffusionInferenceStrategy.run_inference")

        # Denormalize output to physical k-space scale. preprocess_input stored
        # the percentile divisor in self._last_scale; reverse it before return.
        # See findings booklet 2026-05-05 N-2.
        # Full inverse: decompress the log1p BEFORE restoring the scale. A bare
        # ``* scale`` leaves the prediction log-compressed (issue #572).
        x_t = self.denormalize_kspace(x_t)

        return x_t

    def postprocess_output(self, output_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Postprocess output, avoiding inappropriate clamping."""
        # IF we normalized k-space in preprocessing, we might want to un-normalize
        # but usually we want normalized results for display.
        # However, we must NOT do (x+1)/2 and clamp [0,1] if it's k-space.

        is_kspace = (
            self.config.get("model", {}).get("input_type") == "kspace"
            or self.config.get("data", {}).get("dataset_type") == "kspace"
        )

        if is_kspace:
            # Return raw k-space tensor (can be > 1 or < 0)
            return output_tensor.detach().cpu()

        # Standard postprocessing for images
        output_tensor = output_tensor.detach().cpu()
        output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
        return output_tensor
