"""Model Validation Mixin Module.

This module provides a reusable mixin for model validation, handling
batch unpacking, prediction generation, and metric computation.
"""

import functools
import logging
import traceback
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any

import torch

from spectramr.core.metrics.data_range_contract import DataRangeContractError
from spectramr.core.metrics.registry import MetricsRegistry
from spectramr.core.metrics.sense_image_metrics import SenseMapsMissingError
from spectramr.infrastructure.builders.leaf.adapter_builders import apply_pre_model
from spectramr.infrastructure.training.utils.training_utils import clamp_to_range

from ..validation_chunking import reconstruction_output
from .recorrupted_inference import RecorruptedInference, resolve_recorrupted_inference
from .test_slice_feed import feed_test_slice_writer
from .utils import pick_present
from .validation_context import (
    SUPPLIED_FIELDS,
    compute_context_metrics,
    context_metric_names,
    fold_depth,
    project_prediction,
    with_measurement,
)

logger = logging.getLogger(__name__)


class _FallbackRangeUnresolvedError(Exception):
    """The fallback PSNR has no right-domain number to report, and has said why.

    Raised when no data_range can be resolved, or when the metric transform that
    puts the output in the scored domain failed. Internal control flow only.
    Distinct from a generic exception so the fallback's own ``except`` cannot
    mistake either case for "the fallback itself is broken" and log the wrong
    diagnosis.
    """


class ModelValidationMixin(ABC):
    #: Metrics the input-as-prediction baseline is graded on. Intersected with
    #: the arm's configured specs by ``ValidationMetricsComputer.compute(only=)``,
    #: never added to them (one owner of "which metrics does this arm grade on").
    # ssim included: the one-virtual-coil EI pair selects on val_ssim and reads it
    # against the zero-filled adjoint, which needs the same column on the baseline.
    _ZF_BASELINE_METRICS: tuple[str, ...] = ("psnr", "ssim", "hfen")
    #: The object-only pair, added by the strategies whose cohorts read it (EI, SSDU, n2n).
    _ZF_OBJECT_METRICS: tuple[str, ...] = ("robust_mri_psnr", "clinical_ssim")
    #: Reported once per strategy instance when the baseline cannot be formed.
    _zf_baseline_not_applicable_reported: bool = False
    #: The coil-manifold projection announces itself once (pitfall 15's provenance leg).
    _coil_projection_logged: bool = False
    #: ``validation.scoring.recorrupted_inference``, resolved on the first step:
    #: ``False`` until then, ``None`` when the knob is off.
    _recorrupted_inference: RecorruptedInference | bool | None = False

    # Overwritten per instance by BaseTrainingStrategy; the empty default keeps bare stubs valid.
    adapter_chains: Any = MappingProxyType({})

    @classmethod
    def routes_context_metric(cls, name: str) -> bool:
        """Whether this path hands ``name`` the context it declared it needs.

        Read by the ``context_metrics_routed`` audit check. A strategy with its own
        ``validation_step`` does not inherit this, and a context metric configured on
        one raises there for want of its maps instead of producing a column.
        """
        return set(MetricsRegistry.needs(name)) <= SUPPLIED_FIELDS

    def _validation_loss(
        self,
        raw_prediction: torch.Tensor,
        target_batch: torch.Tensor,
        batch_context: dict[str, Any],
        batch_data: Any,
    ) -> torch.Tensor | None:
        """The strategy's own ``val_loss``, or ``None`` for the generic magnitude L1.

        A strategy whose training objective is not a prediction-vs-target distance
        (EI, SSDU) overrides this so ``val_loss`` is on the training loss's scale.
        """
        return None

    def _scoring_estimator(self, config: Any) -> RecorruptedInference | None:
        """The Monte-Carlo estimator the arm declares, resolved and announced once."""
        if self._recorrupted_inference is False:
            # No validation block declares no knob, as for coil_manifold_projection.
            declared = config is not None and config.validation is not None
            spec = resolve_recorrupted_inference(config) if declared else None
            self._recorrupted_inference = None if spec is None else RecorruptedInference(spec)
            if spec is not None:
                logger.info(
                    "[validation] recorrupted_inference: ON (%s); every scored output, "
                    "pred_alt included, is the mean over the recorruptions.",
                    spec.describe(),
                )
        return self._recorrupted_inference or None

    def _zero_filled_baseline_metrics(
        self, input_batch, target_batch, computer, transform_fn, val_config
    ) -> dict[str, float]:
        """``val_zf_<metric>``: the input graded against the target like a prediction.

        For a loader-masked k-space arm the input IS the zero-filled measurement;
        for an image-to-image arm it is the identity baseline. Both answer "did
        the model beat what it was handed?". Not applicable (reported once, then
        silent) when the input cannot be brought to the target's shape by the
        arm's own metric transform -- an SR arm whose input is a smaller image.
        """
        if input_batch is None or target_batch is None:
            return {}
        try:
            zf_in = input_batch
            zf_tgt = target_batch
            if zf_in.ndim == 5:
                b_dim, c_dim, h_dim, w_dim, d_dim = zf_in.shape
                zf_in = zf_in.permute(0, 4, 1, 2, 3).reshape(b_dim * d_dim, c_dim, h_dim, w_dim)
            if zf_tgt.ndim == 5:
                b_dim, c_dim, h_dim, w_dim, d_dim = zf_tgt.shape
                zf_tgt = zf_tgt.permute(0, 4, 1, 2, 3).reshape(b_dim * d_dim, c_dim, h_dim, w_dim)
            if transform_fn:
                zf_in, zf_tgt = transform_fn(zf_in, zf_tgt, val_config)
            if zf_in.shape != zf_tgt.shape:
                raise ValueError(f"input {tuple(zf_in.shape)} vs target {tuple(zf_tgt.shape)}")
            baseline = computer.compute(zf_in, zf_tgt, only=self._ZF_BASELINE_METRICS)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            if not self._zf_baseline_not_applicable_reported:
                self._zf_baseline_not_applicable_reported = True
                logger.info(
                    "[ModelValidationMixin] zero-filled/identity baseline not applicable "
                    "for this arm (%s); no val_zf_* columns will be reported.",
                    exc,
                )
            return {}
        return {f"val_zf_{k}": v for k, v in baseline.items()}

    """Mixin for standardized model validation logic.

    Provides a template method `validation_step` that orchestrates:
    1. Batch unpacking
    2. Prediction generation (via hook)
    3. Metric computation
    4. Error handling
    """

    #: True when ``_validation_forward`` reads the measurement from the batch
    #: context instead of the tensor it is handed (EI, SSDU); the second-repetition
    #: forward then gets a context that carries ``input_alt`` as its measurement.
    validation_reads_context_measurement: bool = False

    def _alternate_validation_context(
        self, batch_context: dict[str, Any], alternate: torch.Tensor
    ) -> dict[str, Any]:
        """The context the forward on ``input_alt`` runs on (#2593)."""
        if not self.validation_reads_context_measurement:
            return batch_context
        return with_measurement(batch_context, alternate)

    def _context_metric_prediction(
        self, raw: torch.Tensor, batch_context: dict[str, Any]
    ) -> tuple[torch.Tensor, str]:
        """``(tensor, domain)`` the context metrics grade, from the folded raw output.

        The model's output as it stands, in ``model.output_type``. A strategy whose
        output is not what those metrics read overrides this: the Rep-BV metrics
        read coil k-space, and an image network's SENSE-combined ``x_hat``
        has to be re-expanded to ``F S x_hat`` before they can grade it (#2509).
        """
        del batch_context
        return raw, self.state.config.model.output_type

    @abstractmethod
    def _validation_forward(
        self,
        input_batch: torch.Tensor,
        batch_context: dict[str, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass for validation.

        Args:
            input_batch: Input tensor
            batch_context: Batch context dictionary
            **kwargs: Additional arguments

        Returns:
            Generated output tensor
        """
        pass

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        *,
        batch_data: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Performs a single validation step with evaluation metrics.

        Args:
            input_batch: Optional pre-unpacked input
            target_batch: Optional pre-unpacked target
            batch_data: The whole validation batch. Declared by name because the
                loop forwards it only to a ``validation_step`` that asks for it
                (``select_validation_extra_fields``); it is read only for the
                fields metrics declare in ``needs`` and for the coil maps the
                ``coil_manifold_projection`` knob projects onto.
            **kwargs: Additional context (e.g. batch_context)

        Returns:
            Dictionary of computed metrics
        """
        batch = (input_batch, target_batch)
        # POP (not get): _forward below re-forwards **kwargs to _validation_forward
        # and also passes batch_context positionally. With .get, a caller that supplies
        # batch_context= (the cross-field / field-flow / field-bridge validation_step
        # overrides do) would hit "multiple values for argument 'batch_context'".
        batch_context = kwargs.pop("batch_context", None)

        # Unpack if needed
        if input_batch is None or target_batch is None:
            # Assume self._unpack_batch exists (from BatchPreparationMixin)
            unpack_fn = self._unpack_batch
            if unpack_fn:
                lr, hr = unpack_fn(batch)
                input_batch = pick_present(input_batch, lr)
                target_batch = pick_present(target_batch, hr)

        # Fail gracefully if still missing
        if input_batch is None or target_batch is None:
            logger.warning(
                "[ModelValidationMixin] _unpack_batch returned None for input/target. "
                "Batch type=%s, keys=%s. Returning empty metrics.",
                type(batch).__name__,
                list(batch.keys()) if isinstance(batch, dict) else "N/A",
            )
            return {}

        # Same pre_model chain as train_step: validation grades the tensors the model trained on.
        input_batch, target_batch = apply_pre_model(self.adapter_chains, input_batch, target_batch)

        self.generator_model.eval()
        # Prepare context if missing
        if batch_context is None:
            batch_context = {"use_dc": False, "measured_kspace": None}

        config = self.state and self.state.config

        def _forward(x: torch.Tensor, ctx: dict[str, Any] | None = None) -> torch.Tensor:
            out = self._validation_forward(x, batch_context if ctx is None else ctx, **kwargs)
            # Tuple-returning models (DisentangledMRI, VAE) and dict-returning ones.
            return reconstruction_output(out)

        # The arm's estimator: the model itself, or its mean over recorruptions.
        estimator = self._scoring_estimator(config)
        if estimator is not None and self.validation_reads_context_measurement:
            # The estimator recorrupts the tensor it hands the forward; a strategy
            # that reads y from the context would run J identical forwards.
            raise ValueError(
                f"{type(self).__name__} reads its measurement from the batch context, so "
                "validation.scoring.recorrupted_inference would average J copies of one "
                "output. Drop the knob for this arm."
            )

        def _estimate(
            x: torch.Tensor, batch_key: int | None = None, ctx: dict[str, Any] | None = None
        ) -> torch.Tensor:
            if estimator is None:
                return _forward(x, ctx)
            return estimator.predict(lambda z: _forward(z, ctx), x, batch_data, batch_key=batch_key)

        # The batch index reseeds the draws, so every epoch scores this batch under
        # the same recorruptions; the pred_alt call below continues the stream.
        hr_fakes = _estimate(input_batch, kwargs.get("batch_idx"))

        if hr_fakes is None:
            logger.warning(
                "[ModelValidationMixin] _validation_forward returned None. Cannot compute metrics."
            )
            return {}

        # Ensure tensors are on the same device
        if target_batch is not None and hr_fakes.device != target_batch.device:
            hr_fakes = hr_fakes.to(target_batch.device)

        # Optional output range enforcement
        training_config = getattr(config, "training", None)
        if training_config and getattr(training_config, "enforce_output_range", False):
            hr_fakes = clamp_to_range(hr_fakes, enable=True, telemetry=False)

        # Depth folds into the batch once, here, so the projection, the context
        # metrics and the image metrics all see the (B, C, H, W) layout.
        hr_fakes, _ = fold_depth(hr_fakes)
        input_batch, _ = fold_depth(input_batch)
        if target_batch is not None:
            target_batch, _ = fold_depth(target_batch)

        # Compute Metrics
        metrics = {}
        val_config = config.validation if config else None
        compute_img_metrics = val_config.scoring.enable_image_metrics if val_config else True

        # The network's own output, kept before any projection: the certificate
        # metrics read THAT (a residual measured after projecting would be 0);
        # every other metric, image or context, grades the projected output.
        raw_prediction = hr_fakes
        projector = None
        if val_config is not None and val_config.scoring.coil_manifold_projection:
            hr_fakes, maps_key = project_prediction(hr_fakes, batch_data, config.model.output_type)

            def projector(x: torch.Tensor) -> torch.Tensor:
                return project_prediction(x, batch_data, config.model.output_type)[0]

            if not self._coil_projection_logged:
                self._coil_projection_logged = True
                logger.info(
                    "[validation] coil_manifold_projection: ON, maps from batch key %r; "
                    "every metric grades the projected output except the null-space "
                    "certificate, which reads the raw one.",
                    maps_key,
                )

        # Attached by the held-out test pass only: every test slice is rendered and guarded.
        test_writer = getattr(self, "_test_slice_writer", None)
        if test_writer is not None and target_batch is not None:
            feed_test_slice_writer(
                self,
                test_writer,
                input_batch=input_batch,
                prediction=hr_fakes,
                target=target_batch,
                val_config=val_config,
                batch_data=batch_data,
                batch_index=kwargs.get("batch_idx"),
            )

        if compute_img_metrics and target_batch is not None:
            computer = self.validation_metrics_computer
            if computer and not computer.get_metric_names():
                # Every requested name is strategy-emitted (#2839). Skipping here
                # also keeps the fallback PSNR below from scoring what was not asked for.
                computer = None
            context_names: tuple[str, ...] = ()
            if computer:
                # Outside the `try` below on purpose: a context metric missing its
                # maps is a wiring defect and must fail the batch, not degrade to
                # the fallback PSNR. Run first, since `compute` clears
                # `last_not_applicable` per call and the full pass should own it.
                context_names = context_metric_names(computer)
                if context_names:
                    graded, domain = self._context_metric_prediction(raw_prediction, batch_context)

                    def _graded_forward(x: torch.Tensor) -> torch.Tensor:
                        alt_context = self._alternate_validation_context(batch_context, x)
                        return self._context_metric_prediction(
                            fold_depth(_estimate(x, ctx=alt_context))[0], alt_context
                        )[0]

                    metrics.update(
                        compute_context_metrics(
                            computer,
                            context_names,
                            graded,
                            target_batch,
                            input_batch,
                            batch_data,
                            config.model,
                            _graded_forward,
                            projected=projector,
                            signal_domain=domain,
                            image_prediction=(
                                raw_prediction if config.model.output_type == "image" else None
                            ),
                        )
                    )
            if computer:
                try:
                    views = self._metric_transform_views(
                        hr_fakes, target_batch, val_config, batch_data=batch_data
                    )

                    # Evidential UNets output 4 parameters (mean, var, alpha, beta)
                    # We only want to compute structural metrics on the predicted mean
                    if (
                        getattr(config.model, "model_type", "") == "evidential_unet"
                        and views.pred.shape[1] > views.target.shape[1]
                    ):
                        views = views._replace(pred=views.pred[:, : views.target.shape[1]])

                    if context_names:
                        plain = [n for n in computer.get_metric_names() if n not in context_names]
                        computed = views.compute(computer, only=plain)
                    else:
                        computed = views.compute(computer)

                    # Apply prefix 'val_'
                    for k, v in computed.items():
                        metrics[f"val_{k}"] = v

                    # Zero-filled / identity baseline (cohort review 2026-09-02,
                    # T0.7): the INPUT, graded exactly like the prediction on the
                    # cheap full-reference metrics the arm already computes. A
                    # model scoring below its own input is the failure every
                    # dashboard was green on (experiment_11_attention_none,
                    # 2026-08-18); the cold-diffusion path emits its own from the
                    # masked measurement and says so with ``_owns_zero_filled_baseline``.
                    if not getattr(self, "_owns_zero_filled_baseline", False):
                        metrics.update(
                            self._zero_filled_baseline_metrics(
                                input_batch,
                                target_batch,
                                computer,
                                functools.partial(
                                    self._apply_metric_transforms, batch_data=batch_data
                                ),
                                val_config,
                            )
                        )
                except torch.cuda.OutOfMemoryError:
                    # NN#3 (WS-7 SM-002): never swallow CUDA OOM — it leaves the
                    # allocator in a corrupt state and the fallback metric path
                    # below would OOM again. Propagate so the run fails loudly
                    # instead of silently reporting degraded validation metrics.
                    raise
                except (DataRangeContractError, SenseMapsMissingError):
                    # Not a metric failure for the fallback to paper over: the
                    # fallback scores the untransformed target and would read the
                    # contradiction as "no range", leaving the run without val_psnr;
                    # a SENSE transform without maps would be scored as RSS.
                    raise
                except Exception as e:
                    logger_service = self.logging_service
                    tb_str = traceback.format_exc()
                    if logger_service:
                        logger_service.log_warning(
                            f"[ModelValidationMixin] Validation metrics computation failed: {e}\n"
                            f"  pred shape={hr_fakes.shape}, dtype={hr_fakes.dtype}\n"
                            f"  target shape={target_batch.shape}, dtype={target_batch.dtype}\n"
                            f"  Traceback:\n{tb_str}"
                        )
                    else:
                        logger.warning(
                            "[ModelValidationMixin] Validation metrics failed: %s\n%s",
                            e,
                            tb_str,
                        )

                    # Fallback: compute basic PSNR and SSIM so we never return empty metrics
                    try:
                        # Scored in the domain the computer would have scored in:
                        # the raw tensors of a k-space arm are interleaved k-space,
                        # and an image PSNR on them is a finite number in the wrong
                        # domain (#2603). If the transform itself is what failed,
                        # there is no right-domain tensor to score, so NaN.
                        try:
                            fallback_views = self._metric_transform_views(
                                hr_fakes, target_batch, val_config, batch_data=batch_data
                            )
                        except Exception as view_exc:
                            logger.warning(
                                "[ModelValidationMixin] Fallback val_psnr is NaN: the "
                                "metric transform that maps this arm's output "
                                "(model.output_type=%s) into the scored domain failed "
                                "(%s), and the untransformed tensors are not scored "
                                "in its place.",
                                getattr(getattr(config, "model", None), "output_type", None),
                                view_exc,
                            )
                            metrics["val_psnr"] = float("nan")
                            raise _FallbackRangeUnresolvedError from None
                        pred_for_fallback = fallback_views.pred.float()
                        target_for_fallback = fallback_views.target.float()

                        # A 2-channel prediction is real/imag for MOST strategies, but a
                        # distribution head emits [mean, logvar]. RSS-ing THOSE computes
                        # sqrt(mean^2 + logvar^2) -- mixing an intensity with a log-variance
                        # -- and since logvar is near-constant across the image it drags the
                        # graded prediction toward an input-independent field (the b29
                        # wrong-channel metric). The strategy already declares which it is.
                        _dist_head = bool(getattr(self, "predicts_distribution_params", False))

                        # Ensure same number of channels via magnitude conversion
                        if pred_for_fallback.shape[1] != target_for_fallback.shape[1]:
                            if pred_for_fallback.shape[1] == 2:
                                pred_for_fallback = (
                                    pred_for_fallback[:, 0:1]
                                    if _dist_head
                                    else torch.sqrt(
                                        pred_for_fallback[:, 0:1] ** 2
                                        + pred_for_fallback[:, 1:2] ** 2
                                    )
                                )
                            if target_for_fallback.shape[1] == 2:
                                target_for_fallback = torch.sqrt(
                                    target_for_fallback[:, 0:1] ** 2
                                    + target_for_fallback[:, 1:2] ** 2
                                )
                            # If still mismatched, take first channel
                            if pred_for_fallback.shape[1] != target_for_fallback.shape[1]:
                                pred_for_fallback = pred_for_fallback[:, :1]
                                target_for_fallback = target_for_fallback[:, :1]

                        mse = torch.nn.functional.mse_loss(
                            pred_for_fallback, target_for_fallback
                        ).item()

                        # The SAME data_range policy, declared range and scale
                        # contract the registered `psnr` uses, read off the computer
                        # that just failed, so `val_psnr` means one thing whichever
                        # branch produced it. See docs/metrics_reference.rst, "How
                        # the image range is resolved".
                        from spectramr.core.metrics.evaluation_metrics import (
                            resolve_image_data_range,
                        )
                        from spectramr.core.metrics.outcome import (
                            MetricNotApplicableError,
                        )

                        try:
                            data_range = resolve_image_data_range(
                                target_for_fallback,
                                computer.data_range,
                                metric_name="val_psnr",
                                contract=computer.scale_contract,
                            )
                        except MetricNotApplicableError as dr_exc:
                            # No contract holds, so there is no PSNR to report.
                            logger.warning(
                                "[ModelValidationMixin] Fallback val_psnr skipped: %s",
                                dr_exc,
                            )
                            metrics["val_mse"] = mse
                            raise _FallbackRangeUnresolvedError from None

                        if mse > 0:
                            psnr = 10.0 * torch.log10(torch.tensor(data_range**2 / mse)).item()
                        else:
                            psnr = float("nan")

                        metrics["val_psnr"] = psnr
                        metrics["val_mse"] = mse
                        logger.info(
                            "[ModelValidationMixin] Fallback metrics (same data_range "
                            "policy as the registered psnr): val_psnr=%.2f, val_mse=%.6f",
                            psnr,
                            mse,
                        )
                    except _FallbackRangeUnresolvedError:
                        pass
                    except DataRangeContractError:
                        # The data path contradicts its declared normalization;
                        # logging it here would leave the run without val_psnr (#2241).
                        raise
                    except Exception as fallback_e:
                        logger.warning(
                            "[ModelValidationMixin] Fallback metrics also failed: %s",
                            fallback_e,
                        )

        # ---- Validation loss for the learning-curve figure (fig_1_2) --------
        # The image-metric block above emits val_ssim/psnr/lpips but NO scalar
        # loss, so validation_metrics.csv carried no loss column and
        # fig_1_2_learning_curves had a train-only series. Emit a paradigm-agnostic
        # validation reconstruction loss (magnitude-matched L1 on the prediction /
        # target already in hand) as ``val_loss``. It is always computable here
        # without the strategy's batch dict; strategies with their own validation
        # loss (standard / diffusion) override validation_step and set ``val_loss``
        # there, so this never double-writes. Monitoring scalar only — never crash
        # validation for it (the .item() sync is off the hot training loop).
        if compute_img_metrics and target_batch is not None and "val_loss" not in metrics:
            try:
                own_loss = self._validation_loss(raw_prediction, target_batch, batch_context, batch_data)
                if own_loss is not None:
                    metrics["val_loss"] = own_loss.detach()
                    return metrics
                pred_l = hr_fakes.float()
                tgt_l = target_batch.float()
                if pred_l.shape[1] != tgt_l.shape[1]:
                    if pred_l.shape[1] == 2:
                        # See the fallback path above: [mean, logvar] is not real/imag,
                        # so RSS here corrupts val_loss the same way it corrupted PSNR.
                        pred_l = (
                            pred_l[:, 0:1]
                            if getattr(self, "predicts_distribution_params", False)
                            else torch.sqrt(pred_l[:, 0:1] ** 2 + pred_l[:, 1:2] ** 2)
                        )
                    if tgt_l.shape[1] == 2:
                        tgt_l = torch.sqrt(tgt_l[:, 0:1] ** 2 + tgt_l[:, 1:2] ** 2)
                    if pred_l.shape[1] != tgt_l.shape[1]:
                        # An image against coil k-space: grade both in the declared
                        # scoring domain, not on channel 0 of each (|x| vs Re k_0).
                        views = self._metric_transform_views(
                            hr_fakes, target_batch, val_config, batch_data=batch_data
                        )
                        pred_l, tgt_l = views.pred.float(), views.target.float()
                    if pred_l.shape != tgt_l.shape:
                        raise ValueError(
                            f"prediction {tuple(pred_l.shape)} and target {tuple(tgt_l.shape)} "
                            "share no domain under validation.scoring.output_transform"
                        )
                metrics["val_loss"] = torch.nn.functional.l1_loss(pred_l, tgt_l).detach()
            except Exception as exc:  # monitoring scalar — never fatal to validation
                logger.warning("[ModelValidationMixin] val_loss computation skipped: %s", exc)

        return metrics
