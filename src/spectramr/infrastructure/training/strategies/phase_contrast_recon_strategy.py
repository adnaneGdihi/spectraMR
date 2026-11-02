"""Phase-contrast reconstruction: undersampled velocity-encoded k-space to images and velocity.

The model (``flow_kt_unrolled``) reconstructs per-encoding complex images from a
readout slice of k-space ``[B, C, E, T, H, W]`` with its mask and coil maps.
Velocity is never a network output: it is decoded from the images by the physics,
so it is in m/s by construction. This PR trains supervised, against reference
images (``image_ref``); the self-supervised mode for reference-free scans comes
with their loader.

Training scores the images (``phase_contrast_image_nmse``) and the encoded phase
(``phase_contrast_phasor``, wrap- and background-free). Validation reports
per-slice metrics and hands slices to :class:`FlowVolumeAccumulator`, which scores
divergence and mass conservation per subject once a volume is complete.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import torch

from spectramr.config.schemas.enums import Regime, Task
from spectramr.infrastructure.physics.fft_ops import sense_adjoint
from spectramr.infrastructure.physics.signal_models.flow_encoding import (
    encoding_directions,
    phase_difference,
    wrap_phase,
)
from spectramr.models.capabilities import StrategyCapabilities

from .base import BaseTrainingStrategy
from .phase_contrast_volume_validation import FlowVolumeAccumulator

_LOSSES = ("phase_contrast_image_nmse", "phase_contrast_phasor")


def velocity_error(
    images: torch.Tensor, reference: torch.Tensor, venc: torch.Tensor
) -> torch.Tensor:
    """Decoded-velocity error (m/s), ``[B, n_dir, T, H, W]``, against reference images.

    Both decodes carry the same background phase, so wrapping their difference
    cancels it exactly: no background fit, and no polynomial order to keep in step
    with the acquisition. ``venc`` is per sample, ``[B, n_dir]``.
    """
    d_pred = phase_difference(images[:, 1:], images[:, :1])
    d_ref = phase_difference(reference[:, 1:], reference[:, :1])
    return wrap_phase(d_pred - d_ref) * venc[:, :, None, None, None] / math.pi


class PhaseContrastReconStrategy(BaseTrainingStrategy):
    """Supervised reconstruction of phase-contrast cine k-space slices."""

    capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities(
        workflows=frozenset({Regime.FLOW}),
        tasks=frozenset({Task.RECONSTRUCTION, Task.PARAMETER_MAPPING}),
        emitted_metrics=frozenset(
            {
                "val_loss",
                "val_image_nmse",
                "val_zf_image_nmse",
                "val_phasor",
                "val_velocity_rmse_roi",
                "val_divergence_normalised",
                "val_mass_conservation",
                "val_velocity_rmse_roi_volume",
            }
        ),
    )
    inline_losses: ClassVar[frozenset[str] | None] = frozenset(_LOSSES)
    folds_image_losses: ClassVar[bool | None] = False
    # Validation assembles each subject's volume from consecutive slices; any
    # multi-process backend shards the validation loader and splits volumes.
    parallel_compat = frozenset()

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("phase_contrast_recon",))
        pc = getattr(self.config.data, "phase_contrast", None)
        if pc is None or not pc.enabled:
            raise ValueError(
                "PhaseContrastReconStrategy needs data.phase_contrast.enabled: true "
                "(the encoding scheme it decodes velocity with)."
            )
        self._n_dir = encoding_directions(pc.encoding_scheme)
        defaults = type(pc).model_fields
        inert = [
            f"data.phase_contrast.{name}"
            for name in ("voxel_area", "through_plane_axis", "flux_conservation")
            if getattr(pc, name) != defaults[name].default
        ]
        if inert:
            raise ValueError(
                f"{inert} are read only by phase_contrast_flow; here voxel size and "
                "measurement planes come from the data. Remove them."
            )
        loader = self.config.validation.loader if self.config.validation else None
        if loader is not None and (
            loader.num_batches is not None or loader.num_samples is not None
        ):
            raise ValueError(
                "PhaseContrastReconStrategy scores whole subject volumes, so validation "
                "cannot be capped (validation.loader.num_batches / num_samples). Use "
                "fewer validation subjects instead."
            )
        self._weights = {name: self._get_loss_weight(name) for name in _LOSSES}
        if self._weights["phase_contrast_image_nmse"] <= 0.0:
            raise ValueError(
                "losses.physics.lambda_phase_contrast_image_nmse must be > 0: the phasor "
                "term alone can be lowered by shrinking the reconstructed image."
            )
        from spectramr.models.losses.flow_losses import (
            PhaseContrastImageNMSELoss,
            PhaseContrastPhasorLoss,
        )

        self._loss_fns = {
            "phase_contrast_image_nmse": PhaseContrastImageNMSELoss(),
            "phase_contrast_phasor": PhaseContrastPhasorLoss(),
        }
        self._volumes = FlowVolumeAccumulator()

    def declared_metric_keys(self) -> frozenset[str]:
        return frozenset(f"loss_{name}" for name in _LOSSES)

    def _reconstruct(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """``(images, image_ref)``, both complex ``[B, E, T, H, W]``."""
        reference = batch.get("image_ref")
        if reference is None:
            raise ValueError(
                "supervised phase-contrast training needs image_ref on the batch; this "
                "dataset has no fully sampled reference"
            )
        images = self.env.generator(batch.input, batch.mask, batch.coil_maps)
        return images, reference

    def _losses(self, images: torch.Tensor, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: fn(images, reference) for name, fn in self._loss_fns.items()}

    def train_step(
        self,
        batch: Any,
        epoch: int,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generator step that bypasses the base 4-D batch guards (input is 6-D complex)."""
        if not self.generator_model.training:
            self.generator_model.train()

        def g_closure() -> torch.Tensor:
            losses, metrics_tensor = self._compute_losses(None, None, epoch, batch=batch, **kwargs)
            detached = {k: v.detach() if hasattr(v, "detach") else v for k, v in losses.items()}
            self._last_step_metrics = self._handle_anomalies(detached, metrics_tensor, epoch)
            return losses["g_total_loss"]

        return [
            {
                "optimizer": self.env.opt_g,
                "closure": g_closure,
                "model": self.generator_model,
                "name": "generator",
            }
        ]

    def _compute_losses_impl(self, input_batch=None, target_batch=None, epoch=0, **kwargs) -> dict:
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        images, reference = self._reconstruct(batch)
        terms = self._losses(images, reference)
        total = sum(self._weights[n] * terms[n] for n in _LOSSES)
        self._snapshot(batch, images, reference, step=int(kwargs.get("iteration", 0)))
        return {"g_total_loss": total, **{f"loss_{n}": t.detach() for n, t in terms.items()}}

    @staticmethod
    def _mid_frame(x: torch.Tensor) -> torch.Tensor:
        """``|x|`` of the reference encoding at the middle frame, ``[B, 1, H, W]``."""
        return x[:, 0, x.shape[2] // 2].abs()[:, None]

    def _snapshot(
        self, batch: Any, images: torch.Tensor, reference: torch.Tensor, *, step: int
    ) -> None:
        # The loader's k-space is both input_raw and the model's input; the images go
        # in as 4-D mid-frame magnitudes under the model_output tag, so the generic
        # hook does not render [B, E, T, H, W] along the wrong axis (rule 14).
        self.save_debug_snapshot(
            {
                "input_raw": batch.input,
                "input_prepared": batch.input,
                "model_output": self._mid_frame(images),
                "target": self._mid_frame(reference),
            },
            step=step,
            tag="model_output",
            in_kspace_keys={"input_raw", "input_prepared"},
            log_scaled_keys=set(),
            domain_exact=True,
        )

    def validation_step(
        self, input_batch, target_batch, *, batch_data: Any = None, **kwargs
    ) -> dict:
        batch = batch_data
        self.env.generator.eval()
        with torch.no_grad():
            images, reference = self._reconstruct(batch)
            terms = self._losses(images, reference)
            zero_filled = sense_adjoint(
                batch.input, batch.coil_maps[:, :, None, None], batch.mask[:, None]
            )[:, 0]
            err = velocity_error(images, reference, batch.get("venc"))
            vessel = batch.get("vessel_mask")[:, None, None]  # [B, 1, 1, H, W]
            # 0/0 = NaN when a slice misses the vessel; the loop drops NaN per key.
            rmse = (
                (err.pow(2) * vessel).sum() / (vessel.sum() * err.shape[1] * err.shape[2])
            ).sqrt()
            truth = batch.get("velocity")
        self._volumes.add(
            list(batch.get("subject_id")),
            batch.get("slice_index"),
            batch.get("n_slices"),
            batch.get("voxel_size_mm"),
            {
                "velocity": (truth + err) if truth is not None else None,
                "velocity_true": truth,
                "vessel_mask": batch.get("vessel_mask"),
                "inlet_mask": batch.get("inlet_mask"),
                "outlet_mask": batch.get("outlet_mask"),
            },
        )
        self._last_visual_pred = self._mid_frame(images)
        self._last_visual_target = self._mid_frame(reference)
        nmse = self._loss_fns["phase_contrast_image_nmse"]
        return {
            "val_loss": sum(self._weights[n] * terms[n] for n in _LOSSES),
            "val_image_nmse": terms["phase_contrast_image_nmse"],
            "val_zf_image_nmse": nmse(zero_filled, reference),
            "val_phasor": terms["phase_contrast_phasor"],
            "val_velocity_rmse_roi": rmse,
        }

    def reset_validation_metrics(self) -> None:
        super().reset_validation_metrics()
        self._volumes.reset()

    def finalize_validation(self) -> dict[str, float]:
        return {**super().finalize_validation(), **self._volumes.finalize()}


__all__ = ["PhaseContrastReconStrategy", "velocity_error"]
