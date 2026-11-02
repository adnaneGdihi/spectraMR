"""Artifact-robust reconstruction: train on corrupted inputs, detect the corruption.

``ArtifactRobustGenerator`` wraps a U-Net over interleaved multi-coil k-space.
In training mode every forward corrupts its input with the artifacts declared in
``model_kwargs.artifact_augmentation`` (:mod:`artifact_corruption`), while the
loss target stays clean, so the network learns to undo them. A small detector
reads the coil images and predicts which artifacts are present; its sigmoid
scores enter the U-Net as constant input planes, and its BCE against the applied
labels is an auxiliary loss the reconstruction strategy folds into the objective.
In eval mode the input is left alone unless the strategy's robustness pass forces
one artifact (:meth:`ArtifactRobustGenerator.forced_artifact`).
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.core.metrics.coil_manifold import as_interleaved
from spectramr.models.generators.artifact_corruption import (
    ArtifactCorruptor,
    fold_depth,
    parse_artifact_augmentation,
    unfold_depth,
)
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)

#: The auxiliary loss key this model stamps while training.
DETECTOR_LOSS_KEY = "artifact_detector_bce"

#: The fixed seed of the robustness pass, so every validation scores one corruption.
EVAL_SEED = 1234


@register_model(
    name="artifact_robust_generator",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="kspace",
    accepts_complex=False,
    expects_real_imag_interleaved=True,
    strict_kwargs=True,
)
class ArtifactRobustGenerator(nn.Module, IGenerator):
    """U-Net trained on artifact-corrupted k-space, conditioned on a learned artifact detector."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        features: Sequence[int] = (32, 64, 128, 256, 512),
        artifact_augmentation: dict[str, Any] | None = None,
    ) -> None:
        """Build the detector, the corruptor and the U-Net.

        Args:
            in_channels: Interleaved coil channels of the k-space input (``2C``).
            out_channels: Interleaved coil channels of the k-space output.
            features: U-Net widths per level; ``depth = len(features) - 1``.
            artifact_augmentation: The required corruption block (see
                :func:`parse_artifact_augmentation`).
        """
        super().__init__()
        if in_channels % 2:
            raise ValueError(
                f"in_channels must be 2 x coils (real/imag interleaved), got {in_channels}."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spec = parse_artifact_augmentation(artifact_augmentation)
        self.corruptor = ArtifactCorruptor(self.spec)
        n_artifacts = len(self.spec.artifacts)
        self.artifact_detector = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, n_artifacts),
        )
        from spectramr.models.reconstruction.unet import UNet, UNetConfig

        widths = tuple(int(f) for f in features)
        if len(widths) < 2:
            raise ValueError(f"features needs at least two widths, got {list(widths)}.")
        # A UNetConfig, not UNet(features=...): the kwargs path builds its default widths.
        self.base_generator = UNet(
            config=UNetConfig(
                in_channels=in_channels + n_artifacts,
                out_channels=out_channels,
                features=widths,
                depth=len(widths) - 1,
            )
        )
        self._aux: dict[str, torch.Tensor] = {}
        logger.info("[artifact_robust_generator] artifact_augmentation: %s", self.spec.record())

    @contextlib.contextmanager
    def forced_artifact(self, name: str) -> Iterator[None]:
        """Corrupt every sample with ``name`` at ``eval_severity``, also in eval mode."""
        if self.spec.eval_severity is None:
            raise ValueError(
                "artifact_augmentation.eval_severity is null; the robustness pass is off."
            )
        with self.corruptor.forced(name, self.spec.eval_severity, EVAL_SEED):
            yield

    def robustness_eval_artifacts(self) -> tuple[str, ...]:
        """The artifacts the strategy scores a corrupted pass for; empty when off."""
        return () if self.spec.eval_severity is None else self.spec.names

    @property
    def auxiliary_loss_keys(self) -> frozenset[str]:
        """Loss keys :meth:`auxiliary_losses` stamps during training."""
        return frozenset({DETECTOR_LOSS_KEY}) if self.spec.detector_loss_weight > 0 else frozenset()

    def auxiliary_losses(self) -> dict[str, torch.Tensor]:
        """Weighted auxiliary terms from the latest training forward."""
        return dict(self._aux)

    def provenance_record(self) -> dict[str, Any]:
        """The corruption policy as applied, for the run provenance."""
        return {"artifact_augmentation": self.spec.record()}

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Reconstruct interleaved k-space ``[B, 2C, H, W]`` (or ``[..., D]``).

        Extra strategy kwargs (mask, coil maps) are not read: this model owns no
        data-consistency layer.
        """
        del kwargs
        x4, layout = fold_depth(x)
        self._aux = {}
        corrupt = self.training or self.corruptor.active
        if corrupt:
            with torch.no_grad():
                x4, images, labels = self.corruptor.corrupt(x4)
        else:
            images = self.corruptor.coil_images(x4)
        logits = self.artifact_detector(as_interleaved(images).to(x4.dtype))
        if self.training and self.spec.detector_loss_weight > 0:
            bce = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))
            self._aux[DETECTOR_LOSS_KEY] = self.spec.detector_loss_weight * bce
        scores = torch.sigmoid(logits)[:, :, None, None].expand(-1, -1, *x4.shape[-2:])
        out = self.base_generator(torch.cat([x4, scores.to(x4.dtype)], dim=1))
        return unfold_depth(out, layout)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Same spatial extent, ``out_channels`` channels."""
        return (input_shape[0], self.out_channels, *input_shape[2:])

    @property
    def name(self) -> str:
        """Return model name."""
        return "artifact_robust_generator"


__all__ = ["DETECTOR_LOSS_KEY", "ArtifactRobustGenerator"]
