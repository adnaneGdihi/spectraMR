r"""DL-BAE -- dispersion-latent Bloch autoencoder training (M4).

Trains :class:`~spectramr.models.physics_ae.disp_bloch_ae.DispersionBlochAutoencoder`
under two terms:

.. math::

   \mathcal L = \lambda_{\mathrm{DC}}\,\mathcal L_{\mathrm{DC}}
              + \lambda_{\mathrm{mono}}\,\mathcal L_{\mathrm{mono}} ,

the multi-field data consistency that forces one field-invariant latent to
explain every observed field, and the :math:`\partial T_1/\partial B_0 \ge 0`
hinge that keeps the recovered dispersion physical.

The science lives in the module-level :func:`compute_dispersion_bloch_ae_loss` so
it is testable without a trainer -- the same seam as ``bloch_field`` and
``operator_id_bch``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from spectramr.models.losses.dispersion_monotonicity_loss import DispersionMonotonicity
from spectramr.models.losses.image.multifield_data_consistency import (
    MultiFieldDataConsistency,
)

from .reconstruction import ReconstructionTrainingStrategy


def compute_dispersion_bloch_ae_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    data_consistency: MultiFieldDataConsistency,
    monotonicity: DispersionMonotonicity,
    data_consistency_weight: float = 1.0,
    monotonicity_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    r"""Multi-field data consistency + the :math:`\partial T_1/\partial B_0\ge0` hinge.

    Args:
        model: A :class:`DispersionBlochAutoencoder`-like module returning
            ``(reconstruction, latent)`` and exposing ``relaxation_maps``.
        batch: Mapping carrying the multi-field stack. ``target`` is used when
            present, otherwise the autoencoder reconstructs its own ``input``.
        data_consistency: The registered multi-field fidelity module.
        monotonicity: The registered monotonicity-hinge module.
        data_consistency_weight: Weight on the fidelity term.
        monotonicity_weight: Weight on the monotonicity hinge.

    Returns:
        Loss dict with ``loss_total`` plus the grad-carrying
        ``prediction``/``target_image`` pair consumed by the loss-SSOT seam.

    Raises:
        KeyError: when the batch carries no ``input`` stack.
    """
    if "input" not in batch:
        raise KeyError(
            f"DL-BAE requires the multi-field stack under batch key 'input'; "
            f"got keys {sorted(batch)!r}."
        )
    x = batch["input"]
    # Autoencoder: the target IS the observed stack unless the arm supplies a
    # separate (e.g. denoised) reference.
    y = batch.get("target", x)

    recon, latent = model(x)
    loss_dc = data_consistency(recon, y)

    t1_maps, _ = model.relaxation_maps(latent)
    fields = getattr(model, "fields", None)
    loss_mono = monotonicity(t1_maps, fields)

    total = data_consistency_weight * loss_dc + monotonicity_weight * loss_mono
    return {
        "loss_total": total,
        "loss_multifield_data_consistency": loss_dc,
        "loss_dispersion_monotonicity": loss_mono,
        "prediction": recon,
        "target_image": y,
    }


class DispersionBlochAEStrategy(ReconstructionTrainingStrategy):
    """Train the dispersion-latent Bloch autoencoder (DL-BAE, M4)."""

    #: Both physics terms are computed inline at the weights in
    #: ``training.dispersion_bloch_ae``; the fold skips them so a declared
    #: ``image_losses`` entry is not added a second time (#1710).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset(
        {"multifield_data_consistency", "dispersion_monotonicity"}
    )
    folds_image_losses: ClassVar[bool | None] = True

    def _setup_strategy_specific_components(self) -> None:
        """Bind the ``training.dispersion_bloch_ae`` block and build the loss modules."""
        self._verify_strategy_config(expected_modes=("dispersion_bloch_ae", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "dispersion_bloch_ae", None)
        if cfg is None:
            raise ValueError(
                "DispersionBlochAEStrategy requires a `training.dispersion_bloch_ae` "
                "config block (schema TrainingConfigDispersionBlochAE). Declare it "
                "in the arm YAML."
            )
        fields = tuple(float(b) for b in getattr(cfg, "fields_present", ()))
        n_pools = int(getattr(cfg, "num_pools", 1))
        required = 2 * n_pools + 1
        if len(fields) < required:
            # Belt and braces: the Tier-1 check catches this at audit time, but a
            # strategy constructed programmatically must not train a rank-deficient
            # fit to a meaningless optimum (pitfall #9).
            raise ValueError(
                f"DL-BAE is under-determined: num_pools={n_pools} needs M >= {required} "
                f"distinct fields, but fields_present names {len(fields)} ({fields!r})."
            )
        self._dlbae_dc_weight = float(getattr(cfg, "data_consistency_weight", 1.0))
        self._dlbae_mono_weight = float(getattr(cfg, "monotonicity_weight", 1.0))
        self._dlbae_data_consistency = MultiFieldDataConsistency()
        self._dlbae_monotonicity = DispersionMonotonicity()

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Run the physics-decoder autoencoder and fold the declarative image losses."""
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if batch is None or not hasattr(batch, "get"):
            raise ValueError(
                "DispersionBlochAEStrategy requires a mapping batch (dict/TrainingBatch) "
                f"carrying the multi-field stack; got {type(batch)!r}."
            )
        out = compute_dispersion_bloch_ae_loss(
            self.env.generator,
            batch,
            data_consistency=self._dlbae_data_consistency,
            monotonicity=self._dlbae_monotonicity,
            data_consistency_weight=self._dlbae_dc_weight,
            monotonicity_weight=self._dlbae_mono_weight,
        )
        pred = out.pop("prediction", None)
        target_image = out.pop("target_image", None)
        if pred is not None and target_image is not None:
            aux = self._apply_builder_image_losses(pred, target_image, out)
            if aux is not None:
                out["loss_total"] = out["loss_total"] + aux
            self._last_prediction = pred
            self._last_target = target_image
        return out


__all__ = ["DispersionBlochAEStrategy", "compute_dispersion_bloch_ae_loss"]
