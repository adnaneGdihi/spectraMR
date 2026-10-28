"""Heteroscedastic ULF restoration (MICCAI MRIxFields2026, B-2.9).

The generator predicts a 2-channel ``[mean, log-variance]``; training minimises the
Gaussian NLL plus a prior anchoring the predicted variance to the physically
calibrated ULF noise level :math:`\\sigma(B_0)` (Hoult, per-sample from the batch's
source field). Validation returns the MEAN channel so the image metrics score the
restored image (not the 2-channel raw output — avoiding a metric/shape mismatch).

Pure helper :func:`compute_heteroscedastic_ulf_losses` holds the testable math.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from spectramr.infrastructure.training.utils.metric_views import MetricViews
from spectramr.models.losses.heteroscedastic_losses import heteroscedastic_ulf_loss

from .reconstruction import ReconstructionTrainingStrategy


def _hoult_sigma_per_sample(
    field_strength: torch.Tensor | None,
    sigma_floor: float,
    max_sigma: float = 0.2,
    source_field: float | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Per-sample ULF noise std on the [0,1] data scale (Hoult, CLAMPED).

    The per-sample source field is ``batch['field_strength']`` when a dataset stamps
    it, else the declared ``source_field`` (the paired ULF route stamps only the
    3T target). Neither present raises: the anchor is never silently the 3T floor.

    The raw Hoult amplification ``(3/B0)^1.75`` is ~370x at 0.1 T — an ABSOLUTE
    pre-normalisation factor that is meaningless once each image is renormalised to
    [0,1]. We clamp the std to ``max_sigma`` so the variance anchor stays on the data
    scale: 64 mT (sigma 41.9 raw) saturates at ``max_sigma``.
    """
    if field_strength is None:
        if source_field is None:
            raise ValueError(
                "HeteroscedasticULFStrategy needs the source field: batch['field_strength'] "
                "is absent and training.heteroscedastic_ulf.source_field is unset."
            )
        # Built on the loss's device with a fill, so the CUDA path neither mixes devices nor syncs.
        field_strength = torch.full((1,), float(source_field), device=device)
    b0 = field_strength.reshape(-1).float().clamp_min(1e-6)
    return (sigma_floor / (b0 / 3.0) ** 1.75).clamp(max=max_sigma)


def _as_mean_logvar(pred: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(pred, tuple):
        pred = pred[0]
    if isinstance(pred, dict):
        pred = pred.get("reconstruction", next(iter(pred.values())))
    if pred.shape[1] < 2:
        raise ValueError(
            "HeteroscedasticULFStrategy needs a 2-channel [mean, logvar] output; "
            f"got {pred.shape[1]} channel(s). Set model.out_channels: 2."
        )
    return pred[:, 0:1], pred[:, 1:2]


def compute_heteroscedastic_ulf_losses(
    generator: Any,
    batch: dict[str, Any],
    sigma_floor: float = 0.05,
    lambda_var_prior: float = 0.1,
    max_sigma: float = 0.2,
    lambda_mean: float = 0.0,
    logvar_min: float = -10.0,
    logvar_max: float = 2.0,
    source_field: float | None = None,
) -> dict[str, torch.Tensor]:
    """Gaussian NLL + Hoult variance prior over the 2-channel prediction, plus the
    anti-degeneracy mean-anchor + logvar-clamp guards (#20)."""
    x = batch["input"]
    y = batch["target"]
    mean, logvar = _as_mean_logvar(generator(x))
    sigma_t = _hoult_sigma_per_sample(
        batch.get("field_strength"), sigma_floor, max_sigma, source_field=source_field, device=y.device
    )
    return heteroscedastic_ulf_loss(
        mean,
        logvar,
        y,
        sigma_t,
        lambda_var_prior=lambda_var_prior,
        lambda_mean=lambda_mean,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )


class HeteroscedasticULFStrategy(ReconstructionTrainingStrategy):
    """Train a mean+variance ULF restorer with a Hoult-anchored NLL (B-2.9)."""

    #: Loss ownership (mrixfields review 2026-09-03): computes its objective inline and folds NOTHING else: any other declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"heteroscedastic_ulf"})
    folds_image_losses: ClassVar[bool | None] = False

    #: The generator emits a 2-channel ``[mean, logvar]`` for a 1-channel target
    #: and this strategy self-computes the Gaussian NLL from both channels
    #: (``_as_mean_logvar`` enforces the contract). Tells the base train_step
    #: width guard to defer rather than raise ``[DomainMismatch] outputs 2 ...
    #: target provided 1`` (the b29_heteroscedastic_ulf crash).
    predicts_distribution_params: ClassVar[bool] = True

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("heteroscedastic_ulf", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "heteroscedastic_ulf", None)
        self._het_sigma_floor = (
            float(getattr(cfg, "sigma_floor", 0.05)) if cfg is not None else 0.05
        )
        self._het_lambda = float(getattr(cfg, "lambda_var_prior", 0.1)) if cfg is not None else 0.1
        self._het_max_sigma = float(getattr(cfg, "max_sigma", 0.2)) if cfg is not None else 0.2
        # Anti-degeneracy guards (#20): a direct mean L1 anchor + a logvar clamp so the
        # NLL cannot be minimised by hedging with a large variance and a lazy mean.
        self._het_lambda_mean = float(getattr(cfg, "lambda_mean", 0.0)) if cfg is not None else 0.0
        self._het_logvar_min = (
            float(getattr(cfg, "logvar_min", -10.0)) if cfg is not None else -10.0
        )
        self._het_logvar_max = float(getattr(cfg, "logvar_max", 2.0)) if cfg is not None else 2.0
        src = getattr(cfg, "source_field", None) if cfg is not None else None
        self._het_source_field = float(src) if src is not None else None
        if getattr(self, "logging_service", None):
            self.logging_service.log_info(
                f"[B-2.9] guards: lambda_mean={self._het_lambda_mean}, "
                f"logvar_clamp=[{self._het_logvar_min}, {self._het_logvar_max}]"
            )

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if batch is None or not hasattr(batch, "get"):  # dict OR TrainingBatch (canonical pipeline)
            raise ValueError(
                "HeteroscedasticULFStrategy requires a mapping batch (dict/TrainingBatch) from the mrixfields "
                f"dataset; got {type(batch)!r}."
            )
        return compute_heteroscedastic_ulf_losses(
            self.env.generator,
            batch,
            sigma_floor=getattr(self, "_het_sigma_floor", 0.05),
            lambda_var_prior=getattr(self, "_het_lambda", 0.1),
            max_sigma=getattr(self, "_het_max_sigma", 0.2),
            lambda_mean=getattr(self, "_het_lambda_mean", 0.0),
            logvar_min=getattr(self, "_het_logvar_min", -10.0),
            logvar_max=getattr(self, "_het_logvar_max", 2.0),
            source_field=getattr(self, "_het_source_field", None),
        )

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **_: Any
    ) -> torch.Tensor:
        """Validation prediction = the full 2-channel ``[mean, logvar]`` output.

        The B-2.9 headline is calibration, so validation scores ``gaussian_nll``
        (which consumes the predicted variance) — that requires both channels. The
        arm therefore validates on ``gaussian_nll`` (not SSIM, which would need the
        mean alone); the variance prior's effect is thus measured, not invisible
        (pitfall #18). ``mean`` is recoverable as channel 0 for any mean-only viz.
        """
        x0 = batch_context.get("input", input_batch)
        out = self.env.generator(x0)
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("reconstruction", next(iter(out.values())))
        return out

    def _metric_transform_views(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        metrics_config: Any = None,
        *,
        batch_data: Any = None,
    ) -> MetricViews:
        """No-op: the 2-channel output is a ``[mean, logvar]`` distribution head,
        NOT complex real/imag.

        The base heuristic reads ``pred.shape[1] == 2`` as complex and auto-selects
        the ``magnitude`` transform, which does ``torch.complex(mean, logvar)`` on
        the prediction and ``torch.complex(target[:,0], target[:,1])`` on the
        1-channel target — an ``IndexError`` that fell back to a raw PSNR of -14
        (the b29_heteroscedastic_ulf / b29_ablate_var_prior crash, pitfall #18).
        ``gaussian_nll`` consumes the 2-channel prediction directly (it slices
        ``[mean, logvar]`` itself), so the head must reach the metric untransformed.
        Overriding the views hook also covers ``_apply_metric_transforms``, which
        delegates to it.
        """
        del metrics_config, batch_data
        return MetricViews(pred, target)

    def _prediction_for_visualization(self, predictions: torch.Tensor) -> torch.Tensor:
        """Visualise the MEAN channel (channel 0), not ``sqrt(mean**2 + logvar**2)``.

        ``_validation_forward`` returns the full 2-channel ``[mean, logvar]`` head so
        ``gaussian_nll`` can consume the predicted variance. The image/TensorBoard path
        must instead show the restored image: the base ``to_magnitude`` treats a
        2-channel tensor as complex and RSS-blends it into ``sqrt(mean**2 + logvar**2)``
        (issue #371 - the b29 dark-brain-on-grey wrong-channel viz, near input-independent
        because the logvar is anchored to a flat Hoult ``sigma``). Slice channel 0 so the
        saved ``fake_images`` are the mean reconstruction. Symmetric with the
        ``_metric_transform_views`` no-op above.
        """
        if predictions.dim() >= 2 and predictions.shape[1] >= 2:
            return predictions[:, 0:1]
        return predictions
