"""Cold Diffusion Inference Strategy

Predict for k-space cold diffusion. The reverse process is the generator's own
``sample()``, reached through the same chunked call validation makes
(:func:`~spectramr.models.diffusion.reverse_path.sample_reverse_path`), so a
validation metric describes what predict returns (#2067). This strategy owns
only what precedes and follows that call: the k-space normalization the training
transform applied, and the coil maps the backbone is conditioned on.
"""

import logging
from typing import Any

import torch
from torch import nn

from spectramr.config.schemas.enums import TrainingModeTypes
from spectramr.config.schemas.validation import ValidationLoaderConfigSchema
from spectramr.core.module_utils import unwrap_model
from spectramr.data.transforms.normalization import KSpaceNormalizationSpec
from spectramr.infrastructure.physics.coil_sensitivity import estimate_smaps_from_measurement
from spectramr.models.diffusion.acquired_line_budget import acquired_lines_of
from spectramr.models.diffusion.reverse_path import (
    resolve_reverse_steps,
    sample_reverse_path,
    takes_coil_maps,
)

from .base_inference_strategy import BaseInferenceStrategy

logger = logging.getLogger(__name__)

#: DC methods whose reverse-loop projection pins the observed bins exactly, so
#: the predict-time hard projection would be a second, identical application.
_PINNING_DC_METHODS = frozenset({"hard"})


def _section(config: dict[str, Any], *path: str) -> Any:
    node: Any = config
    for key in path:
        node = node.get(key) if isinstance(node, dict) else None
    return node


class ColdDiffusionInferenceStrategy(BaseInferenceStrategy):
    """Inference strategy for k-space cold diffusion: one call into ``generator.sample()``."""

    #: :meth:`preprocess_input` normalises and :meth:`denormalize_kspace` inverts.
    owns_kspace_normalization: bool = True

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ):
        """Resolve the horizon and validate the declared degradation.

        Args:
            model: The trained generator, possibly DDP-wrapped.
            device: Device the input is moved to.
            config: The run config as a dict.
        """
        super().__init__(model, device, config)

        self.diffusion_config = self.config.get("diffusion") or {}
        if not self.diffusion_config and "training" in self.config:
            training_cfg = self.config["training"]
            if isinstance(training_cfg, dict):
                self.diffusion_config = training_cfg.get("diffusion") or {}

        self._horizon = self._resolve_horizon()
        # The generator's sample() implements only k-space masking. An
        # advertised-but-unimplemented degradation must raise (pitfall #15).
        self.degradation_type = self.diffusion_config.get("degradation", "kspace_mask")
        if self.degradation_type != "kspace_mask":
            raise ValueError(
                "ColdDiffusionInferenceStrategy only implements the "
                f"'kspace_mask' degradation; got {self.degradation_type!r}. "
                "Remove the 'degradation' key or implement the schedule."
            )
        # Warned once here, not per call (A9/C7): sigma>0 changes the score
        # distribution, so a conformal trust calibration fitted at another sigma
        # or seed is not exchangeable with these reconstructions.
        generator = unwrap_model(self.model)
        sigma = float(getattr(generator, "sampler_sigma", 0.0) or 0.0)
        if sigma > 0:
            logger.warning(
                "sampler_sigma=%.4g > 0: reconstructions carry seeded reverse-step "
                "noise (sampler_seed=%s). Conformal trust calibrations fitted at a "
                "different sigma/seed are not exchangeable with these outputs — "
                "recalibrate before reusing any trust certificate.",
                sigma,
                getattr(generator, "sampler_seed", None),
            )

    def _resolve_horizon(self) -> int | None:
        """The trained model's own horizon, checked against the configured one.

        Returns ``None`` when neither is known; :attr:`num_timesteps` raises on
        use rather than assuming DDPM's 1000 steps for a 29-rung model.
        """
        model_t = getattr(unwrap_model(self.model), "num_timesteps", None)
        config_t = self.diffusion_config.get("timesteps")
        model_t = model_t if isinstance(model_t, int) else None
        if model_t is not None and config_t is not None and int(config_t) != model_t:
            raise ValueError(
                f"The model was built with num_timesteps={model_t} but the config "
                f"declares training.diffusion.timesteps={config_t}; inference would "
                "walk a different ladder than training did."
            )
        return model_t if model_t is not None else (int(config_t) if config_t else None)

    @property
    def num_timesteps(self) -> int:
        """The diffusion horizon T the reverse loop walks."""
        if self._horizon is None:
            raise ValueError(
                "ColdDiffusionInferenceStrategy has no diffusion horizon: the model "
                "carries no num_timesteps and the config no training.diffusion.timesteps."
            )
        return self._horizon

    @property
    def sampling_steps(self) -> int | None:
        """Reverse steps, resolved by the rule validation uses; ``None`` = the generator's own."""
        return resolve_reverse_steps(
            _section(self.config, "validation", "sampling", "steps"),
            self.diffusion_config.get("sampling_steps"),
            self.num_timesteps,
        )

    @property
    def chunk_size(self) -> int:
        """Rows per ``sample()`` call: ``validation.loader.chunk_size``, as validation reads it."""
        declared = _section(self.config, "validation", "loader", "chunk_size")
        if declared is None:
            return int(ValidationLoaderConfigSchema.model_fields["chunk_size"].default)
        return int(declared)

    @property
    def rung_process(self) -> Any:
        """The generator's forward process, which owns the rung masks (``None`` without one)."""
        return getattr(unwrap_model(self.model), "kspace_process", None)

    @property
    def acquisition_line_dim(self) -> int | None:
        """The plane dim the forward process counts R over (-2 / -1), or ``None`` without one."""
        return getattr(self.rung_process, "line_dim", None)

    def _check_measured_lines(self, measured_lines: torch.Tensor, measurement: torch.Tensor) -> None:
        """The declared support must fit the plane and contain every line the input carries.

        Raises:
            ValueError: no line axis, a wrong shape, or a nonzero input line outside the support.
        """
        dim = self.acquisition_line_dim
        if dim is None:
            raise ValueError(
                "measured_lines was passed but this arm's forward process has no line axis "
                "(kspace_process.line_dim is None); nothing would read it."
            )
        expected = (measurement.shape[0], measurement.shape[dim])
        if tuple(measured_lines.shape) != expected:
            raise ValueError(
                f"measured_lines has shape {tuple(measured_lines.shape)}; the input has "
                f"{expected[1]} lines along dim {dim} for {expected[0]} rows, expected {expected}."
            )
        outside = acquired_lines_of(measurement, dim) & ~measured_lines.to(measurement.device).bool()
        if bool(outside.any()):
            rows = sorted({int(r) for r in outside.any(dim=-1).nonzero().flatten().tolist()})
            lines = sorted({int(c) for c in outside.any(dim=0).nonzero().flatten().tolist()})
            raise ValueError(
                f"the input carries data on lines {lines[:8]} (rows {rows[:8]}) outside the "
                "declared acquired support (ISMRMRD kspace_encoding_step_1). The header does "
                "not describe this input, or reads maximum as a count (M4Raw); check with "
                "scripts/physics/measure_phase_encode_axis.py."
            )

    @property
    def training_mode(self) -> TrainingModeTypes:
        """The paradigm this strategy serves."""
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

    def volume_kspace_scale(self, volume: torch.Tensor, *, per_slice: bool) -> torch.Tensor:
        """``[S]`` scales for an ``(S, C, H, W)`` volume, on the unit training measured them on.

        Training normalises one subject at a time: the whole volume on the per-group
        index, one slice on the slice-level index. Measuring per predict batch made the
        output depend on ``batch_size``, and on a different set of slices than training.
        """
        spec = self.kspace_norm
        volume = volume.to(self.device).detach()
        if per_slice:
            return torch.stack(
                [spec.compute_scale(volume[s : s + 1], channel_dim=1) for s in range(len(volume))]
            ).reshape(-1)
        return spec.compute_scale(volume, channel_dim=1).reshape(1).expand(len(volume))

    def preprocess_input(
        self, input_tensor: torch.Tensor, kspace_scale: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        """Apply exactly the normalization the training transform applied (issue #572).

        ``kspace_scale`` (``[B]``, from :meth:`volume_kspace_scale`) is the scale of the
        volume these slices came from; without it the scale is measured on this batch.
        """
        input_tensor = input_tensor.to(self.device).detach()

        spec = self.kspace_norm
        if spec.enabled:
            # channel_dim=1: real/imag interleaved along C, so the scale uses every coil.
            given = (
                None
                if kspace_scale is None
                else kspace_scale.to(self.device).reshape(-1, *([1] * (input_tensor.dim() - 1)))
            )
            normed, scale = spec.normalize(input_tensor, scale=given, channel_dim=1)
            self._last_scale = scale
            logger.info(
                "Normalized input k-space: scale=%.4f (p%.0f, domain=%s, log=%s)",
                float(scale.flatten()[0]),
                spec.percentile * 100,
                spec.scale_domain,
                spec.log_scaling,
            )
            return normed

        return input_tensor

    def _estimate_smaps(self, measurement: torch.Tensor) -> torch.Tensor:
        """Image-domain coil maps from the measurement, in the measurement's layout.

        Calibrated through the configured estimation method, as training and
        validation are. Validation calibrates from the fully-sampled reference;
        predict has only the undersampled input, so the estimator's ACS
        confinement is what keeps the aliased periphery out of the maps.
        """
        smaps = estimate_smaps_from_measurement(
            measurement, config=self.config, log_scaled=self.kspace_norm.log_compressed
        )
        if not torch.is_complex(measurement) and torch.is_complex(smaps):
            smaps = (
                torch.view_as_real(smaps)
                .permute(0, 1, 4, 2, 3)
                .reshape(smaps.shape[0], -1, *smaps.shape[2:])
            )
        return smaps

    def run_inference(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Reconstruct through ``generator.sample()``, the path validation scores.

        Args:
            input_tensor: Normalized measured k-space ``[B, C, H, W]``.
            **kwargs: ``mask`` (observed support; inferred from the non-zeros
                when absent), ``start_timestep`` (the rung the input is degraded
                at; ``None`` starts at ``T - 1``), ``contrast_idx`` and
                ``reference`` (per-row conditioning), and ``measured_lines``
                (``[B, N]`` bool: the phase-encode lines the scan acquired, from
                its ISMRMRD encoding limits). The reverse rungs count R over
                them, as training did (#2572); omitted, they count over the full
                plane.

        Raises:
            AttributeError: The model exposes no ``sample()``.
            ValueError: ``step_callback`` is passed; the reverse path has no
                per-step hook, and dropping a requested monitor in silence would
                report a trajectory that was never observed.
            TypeError: Any other keyword, which nothing on the path would read.
        """
        self.model.eval()
        if kwargs.pop("step_callback", None) is not None:
            raise ValueError(
                "step_callback is not supported: predict reconstructs through "
                "generator.sample(), which exposes no per-step hook (#2067)."
            )
        mask = kwargs.pop("mask", None)
        start_timestep = kwargs.pop("start_timestep", None)
        contrast_idx = kwargs.pop("contrast_idx", None)
        reference = kwargs.pop("reference", None)
        measured_lines = kwargs.pop("measured_lines", None)
        if kwargs:
            raise TypeError(
                f"ColdDiffusionInferenceStrategy.run_inference got {sorted(kwargs)}, "
                "which the reverse path does not read."
            )

        generator = unwrap_model(self.model)
        if not hasattr(generator, "sample"):
            raise AttributeError(
                f"{type(generator).__name__} exposes no sample(); cold-diffusion "
                "predict reconstructs through the generator's own reverse path."
            )
        if mask is not None:
            mask = mask.to(self.device)
        if measured_lines is not None:
            self._check_measured_lines(measured_lines, input_tensor)
            measured_lines = measured_lines.to(self.device)
        model_type = (self.config.get("model") or {}).get("model_type")
        smaps = self._estimate_smaps(input_tensor) if takes_coil_maps(model_type) else None

        with torch.no_grad():
            x0 = sample_reverse_path(
                generator,
                input_tensor,
                mask=mask,
                steps=self.sampling_steps,
                chunk_size=self.chunk_size,
                smaps=smaps,
                start_timestep=start_timestep,
                contrast_idx=contrast_idx,
                reference=reference,
                measured_lines=measured_lines,
            )

        # A hard-DC reverse loop given the mask pins those bins at every step, so
        # the predict-time projection in ``infer`` would apply the same pin twice.
        # Without a mask the loop pins an inferred support, which is not the
        # declared one, so the projection is left to raise for its missing mask.
        physics = getattr(generator, "_reverse_physics", None)
        unconstrained = getattr(physics, "reverse_dc", None) == "unconstrained"
        if (
            mask is not None
            and not unconstrained
            and getattr(generator, "_dc_method", None) in _PINNING_DC_METHODS
        ):
            self.note_data_consistency_applied("ColdDiffusionInferenceStrategy.run_inference")

        # Full inverse: decompress the log1p before restoring the scale (#572).
        return self.denormalize_kspace(x0)

    def postprocess_output(self, output_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return k-space unclamped; clamp image output to ``[0, 1]``."""
        is_kspace = (
            self.config.get("model", {}).get("input_type") == "kspace"
            or self.config.get("data", {}).get("dataset_type") == "kspace"
        )
        output_tensor = output_tensor.detach().cpu()
        if is_kspace:
            return output_tensor
        return torch.clamp(output_tensor, 0.0, 1.0)
