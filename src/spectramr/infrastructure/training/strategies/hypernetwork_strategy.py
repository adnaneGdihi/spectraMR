r"""LCAH -- acquisition-hypernetwork training with a Lipschitz certificate (M3).

Trains :class:`~spectramr.models.encoders.lcah_encoder.LCAHEncoder`, whose
hypernetwork :math:`h_\psi` maps the continuous acquisition vector
:math:`\boldsymbol\varphi=(\mathrm{TE},\mathrm{TR},\mathrm{TI},\alpha,B_0)` to FiLM
modulation for a spectral-normalised target :math:`f_\theta`.

The scientific content is the *certificate*, so it is computed in the module-level
:func:`compute_lcah_loss` (unit-testable without a trainer) rather than buried in
the strategy: with both networks spectral-normalised the prediction drift obeys
:math:`\|\Delta f\| \le L_w L_h\|\Delta\varphi\|`, and the certified radius at an
unseen :math:`\varphi` is reported at validation for free.

``lipschitz_weight > 0`` adds a hinge penalty on the empirical product
:math:`L_wL_h` above ``lipschitz_target``, which tightens the certificate at a
small fidelity cost. It is off by default: the certificate is *valid* either way,
only looser.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from spectramr.config.schemas.training.acq_hypernetwork import ACQUISITION_VECTOR_FIELDS

from .mixins.reconstruction import _as_aux_mapping
from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)

#: Batch key of the per-sample header dict that ``data.acquisition_metadata`` emits.
ACQUISITION_METADATA_KEY = "acquisition_metadata"


def acquisition_vector(batch: Any, acquisition_key: str = "acquisition") -> torch.Tensor:
    r"""The ``[B, 5]`` acquisition vector :math:`(\mathrm{TE},\mathrm{TR},\mathrm{TI},\alpha,B_0)`.

    Read from ``batch[acquisition_key]`` when the pipeline supplies the vector
    directly, else stacked from the ``acquisition_metadata`` dict that
    ``data.acquisition_metadata.enabled`` attaches to every sample (collated as
    one dict of columns, or one dict per sample), in the
    :data:`ACQUISITION_VECTOR_FIELDS` order.

    Raises:
        KeyError: when the batch carries neither, or the dict lacks a field. The
            hypernetwork has no meaningful behaviour without the vector, so this
            raises rather than substituting zeros (pitfall #9).
    """
    if acquisition_key in batch:
        return batch[acquisition_key]
    meta = batch.get(ACQUISITION_METADATA_KEY) if hasattr(batch, "get") else None
    # The image collate passes non-tensor values through as one dict per sample.
    if isinstance(meta, list | tuple) and meta and all(isinstance(m, dict) for m in meta):
        meta = {k: [m[k] for m in meta] for k in meta[0] if all(k in m for m in meta)}
    if not isinstance(meta, dict):
        keys = sorted(batch) if hasattr(batch, "keys") else type(batch).__name__
        raise KeyError(
            f"LCAH requires the acquisition vector under batch key '{acquisition_key}' "
            f"or a '{ACQUISITION_METADATA_KEY}' dict; got keys {keys!r}. Enable "
            "`data.acquisition_metadata` so the pipeline emits it."
        )
    columns = []
    for field in ACQUISITION_VECTOR_FIELDS:
        name = field.upper()
        if name not in meta:
            raise KeyError(
                f"'{ACQUISITION_METADATA_KEY}' lacks '{name}' (has {sorted(meta)!r}); "
                "list it under data.acquisition_metadata.fields."
            )
        columns.append(torch.as_tensor(meta[name], dtype=torch.float32).reshape(-1))
    return torch.stack(columns, dim=-1)


def require_finite_acquisition(phi: torch.Tensor) -> None:
    """Raise when the vector carries NaN, the lenient metadata reader's stand-in for a missing field."""
    if not bool(torch.isfinite(phi).all()):
        raise ValueError(
            "LCAH acquisition vector is not finite: the metadata source supplied no value "
            "for at least one of TE/TR/TI/FA/B0 (strict=false falls back to NaN). Point "
            "data.acquisition_metadata.source at a reader that has the scan header, or "
            "give the field under data.acquisition_metadata.defaults."
        )


def compute_lcah_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    acquisition_key: str = "acquisition",
    lambda_l1: float = 1.0,
    lipschitz_weight: float = 0.0,
    lipschitz_target: float | None = None,
) -> dict[str, torch.Tensor]:
    r"""L1 reconstruction under acquisition conditioning, plus a Lipschitz budget.

    Args:
        model: An :class:`LCAHEncoder`-like module accepting ``(x, acquisition)``
            and exposing ``lipschitz_hyper()`` / ``lipschitz_target()``.
        batch: Mapping with ``input``, ``target`` and the acquisition vector.
        acquisition_key: Batch key carrying the ``[B, acq_dim]`` vector.
        lambda_l1: Weight on the L1 reconstruction term.
        lipschitz_weight: Weight on the Lipschitz-budget hinge (0 disables).
        lipschitz_target: Budget for the product ``L_w * L_h``.

    Returns:
        Loss dict with ``loss_total`` plus the grad-carrying
        ``prediction``/``target_image`` pair consumed by the loss-SSOT seam.

    Raises:
        KeyError: when the acquisition vector is absent from the batch. The
            hypernetwork has no meaningful behaviour without it, so this raises
            rather than substituting zeros (pitfall #9).
    """
    x = batch["input"]
    y = batch["target"]
    phi = _batched(acquisition_vector(batch, acquisition_key), x)
    pred = model(x, phi)

    loss_l1 = F.l1_loss(pred, y)
    out: dict[str, torch.Tensor] = {
        "loss_total": lambda_l1 * loss_l1,
        "loss_l1": loss_l1,
        "prediction": pred,
        "target_image": y,
    }

    if lipschitz_weight > 0.0 and lipschitz_target is not None:
        # Differentiable surrogate for the spectral-norm product: the power
        # iteration inside `spectral_norm_product` is detached, so penalise the
        # Frobenius-norm product instead, which upper-bounds it and carries grad.
        product = torch.ones((), device=pred.device, dtype=pred.dtype)
        for param in model.parameters():
            if param.ndim >= 2:
                product = product * param.flatten(1).norm()
        penalty = F.relu(product - lipschitz_target)
        out["loss_total"] = out["loss_total"] + lipschitz_weight * penalty
        out["loss_lipschitz_budget"] = penalty

    return out


def _batched(phi: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast a shared ``[acq_dim]`` vector over the batch, on ``x``'s device and dtype."""
    phi = phi.to(device=x.device, dtype=x.dtype)
    if phi.ndim == 1:
        phi = phi.unsqueeze(0).expand(x.shape[0], -1)
    return phi


class AcquisitionHypernetworkStrategy(ReconstructionTrainingStrategy):
    """Train a Lipschitz-certified acquisition hypernetwork (LCAH, M3)."""

    def _setup_strategy_specific_components(self) -> None:
        """Bind the ``training.acq_hypernetwork`` block onto the strategy."""
        self._verify_strategy_config(expected_modes=("acq_hypernetwork", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "acq_hypernetwork", None)
        if cfg is None:
            raise ValueError(
                "AcquisitionHypernetworkStrategy requires a "
                "`training.acq_hypernetwork` config block (schema "
                "TrainingConfigAcqHypernetwork). Declare it in the arm YAML."
            )
        self._lcah_acquisition_key = str(getattr(cfg, "acquisition_key", "acquisition"))
        self._lcah_lambda_l1 = float(getattr(cfg, "lambda_l1", 1.0))
        self._lcah_lipschitz_weight = float(getattr(cfg, "lipschitz_weight", 0.0))
        self._lcah_lipschitz_target = getattr(cfg, "lipschitz_target", None)
        self._lcah_report_radius = bool(getattr(cfg, "report_certified_radius", True))
        self._lcah_train_phi: list[torch.Tensor] = []

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Run the FiLM-conditioned forward and fold the declarative image losses."""
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if batch is None or not hasattr(batch, "get"):
            raise ValueError(
                "AcquisitionHypernetworkStrategy requires a mapping batch "
                f"(dict/TrainingBatch) carrying the acquisition vector; got {type(batch)!r}."
            )
        phi = acquisition_vector(batch, self._lcah_acquisition_key)
        if not getattr(self, "_lcah_phi_checked", False):
            # Once, not per step: the check is a host sync (perf rule 9).
            require_finite_acquisition(phi)
            self._lcah_phi_checked = True
        # The base step hands the pre_model-adapted tensors as input/target_batch;
        # the raw batch carries only the acquisition vector, so the model trains on
        # the same input validation grades.
        if isinstance(input_batch, torch.Tensor) and isinstance(target_batch, torch.Tensor):
            batch = {"input": input_batch, "target": target_batch, self._lcah_acquisition_key: phi}
        out = compute_lcah_loss(
            self.env.generator,
            batch,
            acquisition_key=self._lcah_acquisition_key,
            lambda_l1=self._lcah_lambda_l1,
            lipschitz_weight=self._lcah_lipschitz_weight,
            lipschitz_target=self._lcah_lipschitz_target,
        )
        # Remember the training acquisition vectors so validation can report the
        # certified radius against the actual training support. Detached and kept
        # on-device: no `.cpu()`/`.item()` in the training loop (perf rule 9).
        if self._lcah_report_radius:
            self._lcah_train_phi.append(phi.detach().reshape(-1, phi.shape[-1]))

        pred = out.pop("prediction", None)
        target_image = out.pop("target_image", None)
        if pred is not None and target_image is not None:
            aux = self._apply_builder_image_losses(pred, target_image, out)
            if aux is not None:
                out["loss_total"] = out["loss_total"] + aux
            self._last_prediction = pred
            self._last_target = target_image
        return out

    def certified_radius_for(self, phi: torch.Tensor) -> torch.Tensor:
        r"""Certified drift radius :math:`L_wL_h\min_i\|\varphi-\varphi_i\|` at ``phi``.

        Args:
            phi: Query acquisition vector ``[acq_dim]``.

        Returns:
            The certified worst-case prediction drift; zero at a training point.

        Raises:
            RuntimeError: when no training acquisition vectors have been seen, or
                the bound model does not expose the certificate API.
        """
        if not self._lcah_train_phi:
            raise RuntimeError(
                "No training acquisition vectors recorded; run at least one "
                "training step before requesting a certified radius."
            )
        model = self.env.generator
        if not hasattr(model, "certified_radius"):
            raise RuntimeError(
                f"Model {type(model).__name__} exposes no `certified_radius`; "
                "the LCAH certificate needs a spectral-normalised hypernetwork "
                "(model: lcah_encoder)."
            )
        # Collapse the accumulated rows to the distinct protocols, which also bounds
        # the buffer: a protocol seen every step is stored once.
        support = torch.unique(torch.cat(self._lcah_train_phi, dim=0), dim=0)
        self._lcah_train_phi = [support]
        return model.certified_radius(phi.to(support.device), support)

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        *,
        batch_data: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validate under the batch's acquisition vector and report the certified radius."""
        raw = _as_aux_mapping(batch_data) if batch_data is not None else {}
        phi = acquisition_vector(raw, self._lcah_acquisition_key)
        require_finite_acquisition(phi)
        context = dict(
            kwargs.pop("batch_context", None) or {"use_dc": False, "measured_kspace": None}
        )
        context["acquisition"] = phi
        metrics = super().validation_step(
            input_batch, target_batch, batch_data=batch_data, batch_context=context, **kwargs
        )
        if self._lcah_report_radius and isinstance(metrics, dict):
            metrics.update(self._certificate_metrics(phi))
        return metrics

    def _validation_forward(
        self,
        input_batch: torch.Tensor,
        batch_context: dict[str, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward the pre_model-adapted input under its acquisition vector, as training does."""
        phi = batch_context["acquisition"]
        if phi.ndim > 1 and phi.shape[0] not in (1, input_batch.shape[0]):
            raise ValueError(
                f"validation batch carries {phi.shape[0]} acquisition vector(s) for "
                f"{input_batch.shape[0]} input rows; refusing to broadcast a protocol "
                "onto rows it was not recorded for."
            )
        phi = _batched(phi, input_batch)
        if phi.shape[0] != input_batch.shape[0]:
            phi = phi.expand(input_batch.shape[0], -1)
        return self.env.generator(input_batch, phi)

    def _certificate_metrics(self, phi: torch.Tensor) -> dict[str, float]:
        """Mean and max certified radius over the batch's protocols, plus ``L_w L_h``."""
        if not self._lcah_train_phi:
            if not getattr(self, "_lcah_radius_skip_logged", False):
                self._lcah_radius_skip_logged = True
                logger.info(
                    "[LCAH] validation before any training step: no training support, "
                    "certified radius not reported for this pass."
                )
            return {}
        model = self.env.generator
        rows = phi.reshape(-1, phi.shape[-1]) if phi.ndim > 1 else phi.unsqueeze(0)
        radii = torch.stack([self.certified_radius_for(row) for row in rows])
        lipschitz = float(model.lipschitz_hyper() * model.lipschitz_target())
        return {
            "val_lcah_certified_radius": float(radii.mean()),
            "val_lcah_certified_radius_max": float(radii.max()),
            "val_lcah_lipschitz_product": lipschitz,
        }


__all__ = [
    "AcquisitionHypernetworkStrategy",
    "acquisition_vector",
    "compute_lcah_loss",
    "require_finite_acquisition",
]
