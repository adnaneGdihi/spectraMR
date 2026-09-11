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

from typing import Any

import torch
import torch.nn.functional as F

from .reconstruction import ReconstructionTrainingStrategy


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
    if acquisition_key not in batch:
        raise KeyError(
            f"LCAH requires the acquisition vector under batch key "
            f"'{acquisition_key}'; got keys {sorted(batch)!r}. Enable "
            "`data.expose_acquisition_params` so the pipeline emits it."
        )
    x = batch["input"]
    y = batch["target"]
    phi = batch[acquisition_key]
    if phi.ndim == 1:
        phi = phi.unsqueeze(0).expand(x.shape[0], -1)
    pred = model(x, phi.to(dtype=x.dtype))

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
            phi = batch[self._lcah_acquisition_key]
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
        support = torch.cat(self._lcah_train_phi, dim=0)
        return model.certified_radius(phi.to(support.device), support)


__all__ = ["AcquisitionHypernetworkStrategy", "compute_lcah_loss"]
