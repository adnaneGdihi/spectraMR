"""Predict-time k-space: the normalisation and the estimator the arm was trained and scored with.

Training divides k-space by a robust scale (``KSpaceNormalizationTransform``), and
an R2R arm that declares ``validation.scoring.recorrupted_inference`` is scored on
the mean of ``J`` forwards over fresh recorruptions rather than on one forward
(#2329). Predict did neither: it fed raw scanner-unit k-space to a network trained
near unit scale and ran one forward (#2581). This module gives predict both,
through the owners validation uses: :class:`KSpaceNormalizationSpec` for the scale
and :class:`RecorruptedInference` for the mean.

Two differences from validation cannot be removed, so they are stamped instead:

* the scale is measured on the **input**, because a deployed scan has no target
  (training measures it on the subject's ``kspace``, the target);
* coil maps estimated here (``coil_maps_missing: estimate`` with no file on disk)
  come from the single input repetition, not from the group's phase-aligned mean.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from spectramr.core.metrics.coil_manifold import as_complex_coils, as_interleaved
from spectramr.data.signal_domain import is_kspace_dataset_type
from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

logger = logging.getLogger(__name__)

__all__ = ["PredictKSpace", "PreparedVolume"]

#: The validation-context key the null-space estimator reads maps from.
_MAPS_KEY = "sensitivity_complex"


@dataclass
class PreparedVolume:
    """One input file in the model's layout, with what the estimator needs per slice."""

    tensor: torch.Tensor
    scale: torch.Tensor
    per_slice: dict[str, torch.Tensor] = field(default_factory=dict)
    shared: dict[str, torch.Tensor] = field(default_factory=dict)

    def batch_data(self, start: int, end: int) -> dict[str, torch.Tensor]:
        """The validation-batch fields for slices ``[start, end)``."""
        data = {k: v[start:end] for k, v in self.per_slice.items()}
        data.update(self.shared)
        data["kspace_scale"] = self.scale[start:end]
        return data


class PredictKSpace:
    """Normalise, run the declared estimator, de-normalise; resolved once per run."""

    def __init__(self, config: Any, strategy: Any) -> None:
        owns = bool(getattr(strategy, "owns_kspace_normalization", False))
        spec = KSpaceNormalizationSpec.from_data_config(config.data)
        self.data = config.data
        self.owner = "strategy" if owns else "pipeline"
        self.norm: KSpaceNormalizationSpec | None = spec if spec.enabled and not owns else None
        self.interleave = is_kspace_dataset_type(config.data.dataset_type) and not owns
        self.per_slice_scale = bool(getattr(config.data, "slice_level_records", False))
        self.spec, self.estimator = self._resolve_estimator(config, owns)
        self.sources: dict[str, dict[str, str]] = {}

    @staticmethod
    def _resolve_estimator(config: Any, owns: bool) -> tuple[Any, Any]:
        from spectramr.infrastructure.training.strategies.mixins.recorrupted_inference import (
            RecorruptedInference,
            resolve_recorrupted_inference,
        )

        spec = resolve_recorrupted_inference(config)
        if spec is None:
            return None, None
        if owns:
            raise ValueError(
                "validation.scoring.recorrupted_inference wraps one forward of a "
                "reconstruction network; this arm's inference strategy runs its own "
                "sampler, which the estimator cannot wrap."
            )
        return spec, RecorruptedInference(spec)

    # -- per file -----------------------------------------------------------

    def prepare(self, tensor: torch.Tensor, file_id: str) -> PreparedVolume:
        """``(S, C, H, W)`` input -> normalised model layout plus raw-unit side inputs.

        Maps and ``Sigma_n`` are taken before the divide: ``Sigma_n`` is in raw
        units (the estimator divides its draw by ``kspace_scale``) and the maps
        are scale-free.
        """
        n = tensor.shape[0]
        per_slice, shared = self._estimator_inputs(tensor, file_id)
        if self.norm is None:
            scale = torch.ones(n)
        else:
            scale = self._scale(tensor)
            tensor, _ = self.norm.normalize(tensor, scale=scale.reshape(-1, 1, 1, 1), channel_dim=1)
        if self.interleave and torch.is_complex(tensor):
            tensor = as_interleaved(tensor)
        return PreparedVolume(tensor, scale.cpu(), per_slice, shared)

    def _scale(self, tensor: torch.Tensor) -> torch.Tensor:
        assert self.norm is not None
        if self.per_slice_scale:
            return torch.stack(
                [
                    self.norm.compute_scale(tensor[s : s + 1], channel_dim=1)
                    for s in range(len(tensor))
                ]
            ).reshape(-1)
        # One subject is one volume off the per-group loader, and one scale.
        return self.norm.compute_scale(tensor, channel_dim=1).reshape(1).expand(len(tensor))

    def _estimator_inputs(
        self, tensor: torch.Tensor, file_id: str
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        per_slice: dict[str, torch.Tensor] = {}
        shared: dict[str, torch.Tensor] = {}
        if self.spec is None:
            return per_slice, shared
        coils = tensor if torch.is_complex(tensor) else as_complex_coils(tensor, name="input")
        record: dict[str, str] = {}
        if self.spec.target_mode == "nullspace_r2r":
            per_slice[_MAPS_KEY], record["coil_maps"] = self._maps(coils, file_id)
        if "self_calibrated" in (self.spec.covariance_source, self.spec.whitening):
            from spectramr.data.datasets.m4raw_dataset import fit_scan_noise_covariance

            shared["noise_covariance"], _ = fit_scan_noise_covariance(
                coils, support_policy=self.data.noise_fit_support_policy
            )
            record["noise_covariance"] = "self_calibrated:input"
        self.sources[file_id] = record
        return per_slice, shared

    def _maps(self, coils: torch.Tensor, file_id: str) -> tuple[torch.Tensor, str]:
        root = getattr(self.data, "coil_maps_root", None)
        if root is None:
            raise ValueError(
                "the nullspace_r2r estimator splits along the coil maps and this arm "
                "declares no data.coil_maps_root; validation read them from the loader."
            )
        from spectramr.data.datasets.m4raw_coil_map_source import CoilMapSource
        from spectramr.data.metadata.path_resolver import PathResolver

        source = CoilMapSource.declared(
            PathResolver.resolve(root),
            self.data.coil_maps_missing,
            recipe=self.data.coil_maps_recipe,
            acs_size=self.data.coil_maps_acs_size,
        )
        maps, origin = source.maps_for(file_id, [coils])
        return maps, origin if origin != "estimated" else "estimated:input_repetition"

    # -- per batch ----------------------------------------------------------

    def run(
        self,
        forward: Callable[[torch.Tensor], torch.Tensor],
        batch: torch.Tensor,
        batch_data: dict[str, torch.Tensor],
        *,
        batch_key: int,
    ) -> torch.Tensor:
        """One forward, or the declared ``J``-sample mean (seeded by ``batch_key``)."""
        if self.estimator is None:
            return forward(batch)
        return self.estimator.predict(forward, batch, batch_data, batch_key=batch_key)

    def denormalize(self, output: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Back to scanner units, the inverse of :meth:`prepare`'s divide."""
        if self.norm is None:
            return output
        return self.norm.denormalize(
            output, scale.reshape(-1, 1, 1, 1).to(output.device), channel_dim=1
        )

    def provenance(self) -> dict[str, Any]:
        """What predict did to k-space, for the result dict and the run summary."""
        norm = self.norm
        return {
            "kspace_normalised": norm is not None,
            "kspace_normalisation_owner": self.owner,
            "kspace_scale_source": "input" if norm is not None else None,
            "kspace_scale_granularity": (
                None if norm is None else ("slice" if self.per_slice_scale else "volume")
            ),
            "kspace_normalization": None
            if norm is None
            else {
                "percentile": norm.percentile,
                "log_scaling": norm.log_scaling,
                "scale_domain": norm.scale_domain,
            },
            "recorrupted_inference": None
            if self.spec is None
            else {
                "samples": self.spec.samples,
                "target_mode": self.spec.target_mode,
                "alpha": self.spec.alpha,
                "angle_deg": self.spec.angle_deg,
                "covariance_source": self.spec.covariance_source,
                "whitening": self.spec.whitening,
                "seed": self.spec.seed,
                "batch_key": "index of the batch's first slice in its file",
            },
            "estimator_inputs": dict(self.sources),
        }
