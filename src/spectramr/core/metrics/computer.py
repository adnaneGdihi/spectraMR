"""Validation Metrics Computer.

Provides a configurable system for computing validation metrics using the SSOT registry.
"""

import inspect
import logging
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import torch

from spectramr.core.metrics.complex_input import as_phase_bearing
from spectramr.core.metrics.data_range_contract import (
    DataRangeContractError,
    QuantileScaleContract,
)
from spectramr.core.metrics.metric_directions import metric_higher_is_better
from spectramr.core.metrics.outcome import (
    MetricContractError,
    MetricNotApplicableError,
    NotApplicableReason,
)
from spectramr.core.metrics.registry import MetricsRegistry
from spectramr.core.metrics.scalar_transfer import fuse_to_host
from spectramr.core.metrics.types import (
    MetricMode,
    MetricSpec,
    ValidationMetricsConfig,
)

_POSITIONAL = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)
_NAMEABLE = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)


@dataclass(frozen=True, slots=True)
class MetricCallShape:
    """What a metric's callable will actually accept.

    The registry's ``requires_reference`` flag is a *declaration*; this is the
    *implementation*. They disagree for 69 of the 211 registered metrics, all in
    the same direction -- the metric declares ``requires_reference=False`` and
    its ``forward`` still takes ``target`` as a required positional, ignoring it.
    Dispatching on the declaration alone (the shape a naive reading of the
    registry suggests) hands those 69 one argument too few and converts a working
    measurement into a swallowed ``TypeError`` -> NaN. So we dispatch on the
    signature and use the declaration only as a *cross-check*.

    Attributes:
        accepts_target: The callable takes a second positional tensor.
        accepted_kwargs: The keyword names it will tolerate, or ``None`` when it
            declares ``**kwargs`` and tolerates all of them.
    """

    accepts_target: bool
    accepted_kwargs: frozenset[str] | None


def resolve_metric_call_shape(fn: Any) -> MetricCallShape:
    """Introspect ``fn`` once and describe how it may be called.

    Args:
        fn: The bound ``forward`` (or the plain callable, for function metrics).

    Returns:
        The :class:`MetricCallShape`. An uninspectable callable (C-implemented
        or wrapped without ``__wrapped__`` -- 5 of the 211 metrics, e.g. ``fid``)
        is described as the permissive full-reference shape, which is exactly how
        it was called before this seam existed. That keeps the unknown case
        behaviour-preserving instead of guessing a narrower contract for it.
    """
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return MetricCallShape(accepts_target=True, accepted_kwargs=None)

    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        # ``*args`` swallows a target whatever the named parameters say.
        accepts_target = True
    else:
        accepts_target = sum(1 for p in params if p.kind in _POSITIONAL) >= 2

    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        accepted: frozenset[str] | None = None
    else:
        accepted = frozenset(p.name for p in params if p.kind in _NAMEABLE)
    return MetricCallShape(accepts_target=accepts_target, accepted_kwargs=accepted)


class ValidationMetricsComputer:
    """Configurable validation metrics computer.

    Computes user-defined metrics during validation with support for:
    - User-configured metric selection
    - Optimization direction (maximize/minimize) per metric
    - Weighted composite scoring
    - Integration with MetricsRegistry (SSOT)
    """

    def __init__(
        self,
        config: ValidationMetricsConfig | None = None,
        device: str = "cpu",
        domain: str = "image",
        data_range: float | None = None,
        scale_contract: QuantileScaleContract | None = None,
    ):
        """Initialize the validation metrics computer.

        Args:
            config: Validation metrics configuration. If None, uses defaults.
            device: Device for metric computation.
            domain: "image" or "kspace". Passed to metrics for domain-aware computation.
            data_range: Explicit data range override (e.g. 2.0). If None, metrics use their defaults.
            scale_contract: The percentile scale the data path declares, handed to
                the metrics that resolve an image ``data_range`` when none is declared.
        """
        self.config = config or ValidationMetricsConfig.from_dict(None)
        self.device = device
        self.domain = domain
        self.data_range = data_range
        self.scale_contract = scale_contract

        # Build metric lookup for quick access
        self._metric_specs: dict[str, MetricSpec] = {
            spec.name: spec for spec in self.config.metrics if spec.enabled
        }

        # Cache for stateful metric instances
        self._metric_instances: dict[str, Any] = {}
        # Resolved once per (metric, callable) -- `inspect.signature` on every
        # metric of every validation batch would be a per-step cost in the hot
        # path (non-negotiable 9).
        self._call_shapes: dict[tuple[str, str], MetricCallShape] = {}

        # Metric names already warned about (log-once dedupe): a metric that fails
        # every validation batch (e.g. LPIPS with no backend on a stale env) must not
        # spam hundreds of identical warnings into the run log — warn once, then NaN.
        self._warned_metrics: set[str] = set()
        # Log-once dedupe for the distribution-head narrowing in _align_prediction.
        self._narrowed_metrics: set[str] = set()
        # Log-once dedupe for declared not-applicable outcomes, keyed on
        # (metric, reason): the same metric excluded for a NEW reason is new
        # information and must not be suppressed by the first one.
        self._not_applicable_seen: set[tuple[str, NotApplicableReason]] = set()
        #: Metric name -> the reason it was N/A on the most recent ``compute``.
        #: Read by callers that need to distinguish "undefined on this input"
        #: from "crashed", which a bare NaN cannot express. Overwritten each
        #: call; a metric that scores normally is absent.
        self.last_not_applicable: dict[str, NotApplicableReason] = {}

    def reset(self) -> None:
        """Reset all cached metrics (clearing internal states like FID features)."""
        self._metric_instances.clear()
        self._call_shapes.clear()

    def _invoke_metric(
        self,
        name: str,
        callable_: Any,
        shape_source: Any,
        preds: torch.Tensor,
        targets: torch.Tensor,
        compute_kwargs: dict[str, Any],
        *,
        slot: str,
    ) -> Any:
        """Call one metric with exactly the arguments it accepts.

        Args:
            name: Registry name, used for the cache key and error text.
            callable_: What to actually call (the instance, or its ``update``).
            shape_source: The callable to introspect -- ``forward`` for an
                ``nn.Module`` metric, otherwise ``callable_`` itself.
            preds: Prediction tensor, already aligned by ``_align_prediction``.
            targets: Reference tensor. Passed only when the signature takes one.
            compute_kwargs: The computer's context kwargs (``domain``, and
                ``data_range`` when the user configured one).
            slot: ``"call"`` or ``"update"`` -- disambiguates the shape cache.

        Returns:
            Whatever the metric returned.

        Raises:
            MetricContractError: The registry declares the metric needs a reference but
                its signature cannot take one. Dropping the target silently would
                report a no-reference number under a full-reference name -- a
                wrong number, which this framework ranks above a crash. Measured
                0/211 at the time of writing, so this raise is a ratchet against
                a future regression, not a live break.
            MetricNotApplicableError: A *user-declared* kwarg (``data_range``)
                cannot be passed to this metric. Silently falling back to the
                metric's own default range would substitute a default for a
                declared value (non-negotiable 3b), so it is reported as a
                declared N/A instead.
        """
        key = (name, slot)
        cached = self._call_shapes.get(key)
        if cached is None:
            cached = resolve_metric_call_shape(shape_source)
            self._call_shapes[key] = cached

        if MetricsRegistry.requires_reference(name) and not cached.accepts_target:
            raise MetricContractError(
                f"Metric '{name}' declares requires_reference=True but its "
                f"signature accepts no reference argument. Refusing to compute a "
                f"no-reference number and report it under a full-reference name. "
                f"Fix the declaration or the signature -- they are the same claim."
            )

        if cached.accepted_kwargs is None:
            kwargs = compute_kwargs
        else:
            kwargs = {k: v for k, v in compute_kwargs.items() if k in cached.accepted_kwargs}
            dropped = set(compute_kwargs) - set(kwargs)
            if "data_range" in dropped:
                raise MetricNotApplicableError(
                    name,
                    NotApplicableReason.DECLARED_KWARG_UNSUPPORTED,
                    "the run declares metrics.data_range but this metric's "
                    "signature cannot accept it; computing against its built-in "
                    "default would silently ignore the declared range",
                )

        args = (preds, targets) if cached.accepts_target else (preds,)
        return callable_(*args, **kwargs)

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        *,
        only: Collection[str] | None = None,
        complex_pair: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute the configured validation metrics.

        Args:
            predictions: Model predictions [B, C, H, W]
            targets: Ground truth targets [B, C, H, W]
            complex_pair: The phase-preserving ``(prediction, target)`` the
                metric transform produced before taking ``abs()``. Metrics
                registered ``requires_complex`` are scored on it; every other
                metric on ``predictions``/``targets``. When ``None`` the
                phase-sensitive metrics fall back to ``predictions``/``targets``
                and raise :class:`PhaseBlindInputError` if those carry no phase.
            only: Restrict this call to the named subset of the *configured*
                specs. ``None`` (the default) runs every enabled spec, i.e.
                the historical behaviour. This is an INTERSECTION, never a
                widening: a name that is not already a configured spec is not
                computed, because a caller that could ADD metrics here would
                be a second owner of "which metrics does this arm grade on"
                (non-negotiable 17). Its purpose is a cheap second pass over
                the same batch -- the zero-filled baseline in the diffusion
                validation path grades a *different tensor* against the same
                target and must not pay for the arm's full metric set, which
                on a perceptual arm means a second LPIPS per rung per batch.

                ``last_not_applicable`` is cleared per call regardless (the
                attribute means "the metrics THIS call ran"), so a subset call
                placed AFTER a full one narrows what ``training_loop.py``'s
                N/A reporter can still see. Subset first, full second.

            **kwargs: Additional arguments passed to metric functions

        Returns:
            Dictionary of metric name -> value
        """
        metrics: dict[str, float] = {}
        # Per-call, not cumulative: a metric that was N/A on the previous batch
        # and scores on this one must not still be reported as excluded.
        self.last_not_applicable.clear()

        # Real scalar tensors are parked here and transferred in ONE sync after
        # the loop. `.item()` per metric per batch cost N syncs per validation
        # batch (~500 for 10 metrics over 50 batches) and, worse, serialised the
        # metrics: each sync drained the queue, so metric k+1 could not start
        # launching while metric k was still running.
        pending_keys: list[str] = []
        pending_scalars: list[torch.Tensor] = []

        # Ensure tensors are on correct device and detached
        predictions = predictions.detach().to(self.device)
        targets = targets.detach().to(self.device)
        phase_source = (
            (predictions, targets)
            if complex_pair is None
            else (complex_pair[0].detach().to(self.device), complex_pair[1].detach().to(self.device))
        )

        # Add domain context to kwargs for metrics that support it
        compute_kwargs = {**kwargs, "domain": self.domain}

        # Add data_range if configured (overrides metric default)
        if self.data_range is not None:
            compute_kwargs["data_range"] = self.data_range
        if self.scale_contract is not None:
            compute_kwargs["scale_contract"] = self.scale_contract

        for spec in self.config.metrics:
            if not spec.enabled:
                continue
            if only is not None and spec.name not in only:
                continue

            # Outside the ``try``: a phase metric fed a magnitude image is a
            # wiring defect, and the broad handler below would turn it into NaN.
            if MetricsRegistry.requires_complex(spec.name):
                metric_preds = as_phase_bearing(spec.name, phase_source[0], "prediction")
                metric_targets = as_phase_bearing(spec.name, phase_source[1], "target")
            else:
                metric_preds, metric_targets = predictions, targets

            try:
                # Use SSOT registry with caching
                if spec.name not in self._metric_instances:
                    self._metric_instances[spec.name] = MetricsRegistry.get(
                        spec.name, device=self.device
                    )

                metric_inst = self._metric_instances[spec.name]

                # A distribution head (e.g. the heteroscedastic [mean, logvar] output of
                # B-2.9) is deliberately wider than its target: `gaussian_nll` consumes
                # both channels, every image metric grades the mean. All metrics share
                # one (pred, target) pair, so the split has to happen per metric, here.
                # Without it PSNR/SSIM assert to NaN and NRMSE/NMSE/MAE/MSE silently
                # BROADCAST the log-variance channel into the error, returning a finite,
                # plausible, meaningless score (pitfall #18).
                pred_for_metric = self._align_prediction(
                    metric_inst, metric_preds, metric_targets, spec.name
                )

                # [SSOT] Check if metric is stateful/summary-based (e.g., FID)
                # These metrics accumulate during the loop and compute only once at finalize()
                is_summary = getattr(metric_inst, "summarize", False)

                if is_summary:
                    # Update internal state, but don't perform expensive full-pass computation
                    if hasattr(metric_inst, "update"):
                        self._invoke_metric(
                            spec.name,
                            metric_inst.update,
                            metric_inst.update,
                            pred_for_metric,
                            metric_targets,
                            compute_kwargs,
                            slot="update",
                        )
                    else:
                        # Fallback for raw torchmetrics or functions
                        self._invoke_metric(
                            spec.name,
                            metric_inst,
                            getattr(metric_inst, "forward", metric_inst),
                            pred_for_metric,
                            metric_targets,
                            compute_kwargs,
                            slot="call",
                        )

                    # Return zero during the loop; actual value comes from finalize()
                    metrics[spec.name] = 0.0
                else:
                    # Incremental metric (PSNR, SSIM, etc.); compute normally
                    value = self._invoke_metric(
                        spec.name,
                        metric_inst,
                        getattr(metric_inst, "forward", metric_inst),
                        pred_for_metric,
                        metric_targets,
                        compute_kwargs,
                        slot="call",
                    )

                    # Defer the host transfer for REAL SCALAR tensors only, so
                    # this stays value-identical to the `.item()` it replaces:
                    #   - non-scalar tensor -> `.item()` raises, caught below,
                    #     recorded NaN. Meaning "a metric returned a vector" is
                    #     still a defect, not silently averaged into a plausible
                    #     number (pitfall #18).
                    #   - complex scalar -> `.item()` yields a Python complex and
                    #     `float()` raises, likewise NaN. Which projection is
                    #     meaningful (.real? .abs()?) is the metric's call.
                    # The placeholder holds the dict slot so the fused write-back
                    # cannot reorder the metrics relative to the spec order.
                    if (
                        isinstance(value, torch.Tensor)
                        and value.numel() == 1
                        and not torch.is_complex(value)
                    ):
                        metrics[spec.name] = float("nan")
                        pending_keys.append(spec.name)
                        pending_scalars.append(value.detach())
                        continue

                    # Handle scalar tensors
                    if hasattr(value, "item"):
                        value = value.item()
                    metrics[spec.name] = float(value)

            except KeyError as exc:
                # RAISE (#173). This used to warn-and-`continue`, which produced a
                # silently missing CSV column for the whole run -- indistinguishable
                # from "that metric was never any good on this data". It fired for
                # six shipped `compute_*` flags naming unregistered metrics, so the
                # noise was routine and nobody read it.
                #
                # Both upstream surfaces are now gated: `metrics.compute` names are
                # validated when the strategy resolves them, and dangling legacy
                # flags are filtered (with one explicit log) before selection. A
                # name reaching this point has passed both, so a KeyError here is a
                # defect in the wiring, not a user typo -- exactly the case that
                # must not degrade quietly (pitfall #9).
                raise KeyError(
                    f"Metric '{spec.name}' reached the computer but is not "
                    "registered. Upstream selection should have rejected it; this "
                    "is a wiring defect, not a config typo. Registered names are "
                    "in MetricsRegistry."
                ) from exc
            except MetricContractError:
                # The registry declaration and the implementation disagree.
                # Re-raised untouched: this is a wiring defect, and degrading it
                # to NaN below would silence the single condition the check was
                # added to make visible (pitfall #9, and the same policy the
                # KeyError branch above states).
                raise
            except MetricNotApplicableError as na:
                # A DECLARED not-applicable, not a crash. `outcome.py` exists to
                # keep these three apart, and its own docstring names collapsing
                # them "pitfall #9 wearing a numeric disguise" -- yet the broad
                # handler below used to catch this and log it as "Failed to
                # compute metric 'X'", which is the one reading that is false.
                #
                # The value is still NaN, because that IS the contract for a
                # non-OK outcome (`MetricOutcome.__post_init__` enforces it). What
                # changes is that the machine-readable reason survives, so a
                # consumer can say WHY there is no number without reading a
                # traceback. Recorded per (metric, reason) rather than per metric:
                # the same metric going N/A for a new reason is new information.
                key = (spec.name, na.reason)
                if key not in self._not_applicable_seen:
                    self._not_applicable_seen.add(key)
                    logging.getLogger(__name__).warning(
                        "Metric '%s' is NOT APPLICABLE to this input (%s): %s. "
                        "Reporting NaN -- this is a declared exclusion, not a "
                        "failed computation. Further identical reports for this "
                        "metric/reason are suppressed.",
                        spec.name,
                        na.reason,
                        na.detail,
                    )
                self.last_not_applicable[spec.name] = na.reason
                metrics[spec.name] = float("nan")
            except DataRangeContractError:
                # A ValueError, but not the shape mismatch the branch below
                # rewraps every ValueError as; keep its type and message.
                raise
            except ValueError as e:
                # F-METRICFAIL / 2026-05-20 — a ValueError from a metric
                # is virtually always a shape/dtype mismatch between
                # pred and target (smoke run 20260519 emitted "Failed
                # to compute metric 'psnr': Predictions and target
                # must have same shape" repeatedly). Storing 0.0 made
                # the broken metric appear in the CSV as a real-but-
                # terrible reconstruction. Re-raise so the underlying
                # YAML / pipeline bug surfaces immediately
                # (CLAUDE.md pitfall #9, no silent fallbacks).
                raise ValueError(
                    f"Metric '{spec.name}' raised ValueError "
                    f"(typically a shape mismatch between pred and "
                    f"target): {e}. Fix the pipeline that produced "
                    f"the predictions / targets rather than ignoring "
                    f"the metric. CLAUDE.md pitfall #9."
                ) from e
            except Exception as e:
                # Non-ValueError: log warning AND store NaN so the
                # bad metric is visible downstream rather than masquerading
                # as a finite 0.0. CSV / plots / early-stopping treat
                # NaN as "not computed" and will skip it cleanly.
                # Warn once per metric name (log-once dedupe): a metric that fails
                # every batch would otherwise flood the log with identical lines.
                if spec.name not in self._warned_metrics:
                    logging.getLogger(__name__).warning(
                        f"Failed to compute metric '{spec.name}': {e} "
                        "(further identical failures for this metric are suppressed)"
                    )
                    self._warned_metrics.add(spec.name)
                metrics[spec.name] = float("nan")

        if pending_keys:
            # The single sync. Deliberately outside the per-metric `try`: an
            # error surfacing here is a device fault, not "metric X is bad", and
            # attributing it to whichever metric happens to be next would be a
            # worse lie than letting it propagate (pitfall #9).
            for key, host_value in zip(pending_keys, fuse_to_host(pending_scalars), strict=True):
                metrics[key] = host_value

        return metrics

    def _align_prediction(
        self,
        metric_inst: Any,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        """Give a shape-strict metric the channels it grades.

        Only ever narrows a prediction that is *wider* than its target in the channel
        dim, and only for a metric that declares ``REQUIRES_MATCHING_SHAPES`` (the
        default). A metric that consumes the whole distribution head — ``gaussian_nll``
        — declares otherwise and receives the prediction untouched. Any other shape
        disagreement is left alone so the metric raises: narrowing it would be the
        silent fallback this method exists to remove (pitfall #9).
        """
        if not getattr(metric_inst, "REQUIRES_MATCHING_SHAPES", True):
            return predictions
        if not (isinstance(predictions, torch.Tensor) and isinstance(targets, torch.Tensor)):
            return predictions
        if predictions.shape == targets.shape:
            return predictions
        # Narrow ONLY the channel axis, and only when every other axis already agrees.
        same_rank = predictions.ndim == targets.ndim and predictions.ndim >= 2
        if not same_rank:
            return predictions
        others_match = (
            predictions.shape[:1] == targets.shape[:1]
            and predictions.shape[2:] == targets.shape[2:]
        )
        if not (others_match and predictions.shape[1] > targets.shape[1]):
            return predictions
        c = targets.shape[1]
        if name not in self._narrowed_metrics:
            logging.getLogger(__name__).info(
                "Metric '%s' grades the leading %d of %d prediction channels "
                "(distribution head detected: pred %s vs target %s).",
                name,
                c,
                predictions.shape[1],
                tuple(predictions.shape),
                tuple(targets.shape),
            )
            self._narrowed_metrics.add(name)
        return predictions[:, :c]

    def finalize(self) -> dict[str, float]:
        """Finalize all stateful/summary metrics and return their final results.

        Returns:
            Dictionary of metric name -> finalized value
        """
        summary_results: dict[str, float] = {}

        for spec in self.config.metrics:
            if not spec.enabled:
                continue

            metric_inst = self._metric_instances.get(spec.name)
            if metric_inst and getattr(metric_inst, "summarize", False):
                try:
                    # Metrics usually have a .compute() method
                    if hasattr(metric_inst, "compute"):
                        value = metric_inst.compute()

                        # Handle None or invalid values. A summary metric that
                        # produced nothing is "not computed" — emit NaN, not a
                        # finite 0.0 that CSV / plots / early-stopping would
                        # treat as a real (terrible) score. This matches the
                        # NaN-means-not-computed contract in ``compute()`` above
                        # (the non-ValueError branch stores ``float("nan")``),
                        # and downstream gates (plateau_monitor,
                        # loss_schedule_controller) skip non-finite values.
                        if value is None:
                            summary_results[spec.name] = float("nan")
                        else:
                            if hasattr(value, "item"):
                                value = value.item()
                            summary_results[spec.name] = float(value)
                    else:
                        # No ``compute`` method to finalise: not computed → NaN.
                        summary_results[spec.name] = float("nan")
                except Exception as e:
                    logging.getLogger(__name__).warning(
                        f"Failed to finalize summary metric '{spec.name}': {e}"
                    )
                    # Exception during finalize → not computed → NaN (not 0.0).
                    summary_results[spec.name] = float("nan")

        return summary_results

    def get_direction(self, metric_name: str) -> MetricMode:
        """Get optimization direction for a metric.

        Resolution is: an explicit per-metric spec (the user dictating
        ``direction:`` in YAML) first, then the registry/SSOT resolver
        :func:`~spectramr.core.metrics.metric_directions.metric_higher_is_better`,
        which **raises** rather than assuming a direction it cannot derive.

        This used to end in ``DEFAULT_METRIC_DIRECTIONS.get(name, MetricMode.MAX)``
        — a silent default that resolved every lower-is-better metric absent from
        that 63-entry legacy table (``lpips``, ``fid``, ``nll_bits_per_dim``,
        ``val_hfen``, …) to MAX. :meth:`is_improvement` then read ``current > best``
        for them, so best-checkpoint selection and early stopping retained the
        checkpoint with the WORST score. See issue #208.

        Args:
            metric_name: Name of the metric

        Returns:
            MetricMode.MAX or MetricMode.MIN

        Raises:
            UnknownMetricDirectionError: the key resolves to no declared
                direction. Declare it rather than letting it default.
        """
        # An explicit spec is the user dictating the direction — always wins.
        if metric_name in self._metric_specs:
            return self._metric_specs[metric_name].direction

        return MetricMode.MAX if metric_higher_is_better(metric_name) else MetricMode.MIN

    def is_improvement(
        self,
        current: float,
        best: float,
        metric_name: str | None = None,
    ) -> bool:
        """Check if current value is an improvement over best.

        Args:
            current: Current metric value
            best: Best metric value so far
            metric_name: Name of metric. If None, uses primary_metric.

        Returns:
            True if current is better than best
        """
        if metric_name is None:
            metric_name = self.config.primary_metric

        direction = self.get_direction(metric_name)

        if direction == MetricMode.MAX:
            return current > best
        else:
            return current < best

    def get_primary_metric(self) -> str:
        """Get the primary metric name."""
        return self.config.primary_metric

    def get_metric_names(self) -> list[str]:
        """Get list of configured metric names."""
        return [spec.name for spec in self.config.metrics if spec.enabled]


def create_validation_metrics_computer(
    config: Any | None = None,
    device: str = "cpu",
    domain: str = "image",
    data_range: float | None = None,
    scale_contract: QuantileScaleContract | None = None,
) -> ValidationMetricsComputer:
    """Factory function to create ValidationMetricsComputer from various config types.

    Args:
        config: Can be:
            - None: Uses defaults
            - dict: Parsed as ValidationMetricsConfig
            - list[str]: Simple list of metric names
            - ValidationMetricsConfig: Used directly
        device: Device for computation
        domain: "image" or "kspace". Passed to metrics for domain-aware computation.
        data_range: Explicit data range override.
        scale_contract: Declared percentile scale, see ``ValidationMetricsComputer``.

    Returns:
        Configured ValidationMetricsComputer instance
    """
    if config is None:
        metrics_config = ValidationMetricsConfig.from_dict(None)
    elif isinstance(config, ValidationMetricsConfig):
        metrics_config = config
    elif isinstance(config, list):
        metrics_config = ValidationMetricsConfig.from_list(config)
    elif isinstance(config, dict):
        metrics_config = ValidationMetricsConfig.from_dict(config)
    else:
        # Try to extract from training config object or validation config object
        metrics = None
        primary = "psnr"

        if hasattr(config, "metrics"):
            # Scenario 1: config is the validation sub-config or metrics config
            metrics = config.metrics
            primary = config.primary_metric if hasattr(config, "primary_metric") else "psnr"
        elif getattr(config, "validation", None) is not None:
            # Scenario 2: config is the top-level training config
            val_config = config.validation
            metrics = val_config.scoring.compute
            # The schema default IS "psnr", so the old hasattr fallback was
            # restating it; the field cannot be absent.
            primary = val_config.scoring.primary

        if metrics is not None:
            metrics_config = ValidationMetricsConfig.from_dict(
                {
                    "metrics": metrics,
                    "primary_metric": primary,
                }
            )
        else:
            # Fallback for complex objects that don't match or None
            metrics_config = ValidationMetricsConfig.from_dict(None)

    return ValidationMetricsComputer(
        metrics_config,
        device=device,
        domain=domain,
        data_range=data_range,
        scale_contract=scale_contract,
    )
