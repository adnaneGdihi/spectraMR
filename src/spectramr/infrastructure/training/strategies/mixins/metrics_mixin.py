"""Metrics Mixin

This module contains the logic for metrics tracking and aggregation.
"""

import logging
from collections.abc import Mapping
from typing import Any

import torch

from spectramr.core.metrics.data_range_contract import QuantileScaleContract
from spectramr.core.metrics.flag_map import schema_flag_to_metric
from spectramr.core.metrics.registry import MetricsRegistry
from spectramr.core.metrics.scalar_transfer import fuse_to_host
from spectramr.core.metrics.sense_image_metrics import sense_scoring_pair
from spectramr.infrastructure.training.loop_state import resolve_loop_iteration
from spectramr.infrastructure.training.strategies.mixins.utils import (
    _get_config_value,
    _scoring_leaf,
    pick_present,
    reduce_on_device,
)
from spectramr.infrastructure.training.utils.data_adapters import TorchIOAdapter
from spectramr.infrastructure.training.utils.metric_transform import (
    IMPLEMENTED_METRIC_TRANSFORMS,
    MetricTransformResolution,
    image_magnitude,
    kspace_rss_magnitude,
    resolve_metric_transform,
)
from spectramr.infrastructure.training.utils.metric_views import (
    MetricViews,
    real_stacked_to_complex,
)

from .validation_context import scoring_maps

logger = logging.getLogger(__name__)


def report_case_id(step: int, cascade_level: float | None = None) -> str:
    """Label one recorded validation case.

    The acceleration rung belongs in the label. Cascading validation calls this
    seam once per level with the SAME training iteration, so ``f"val_step{step}"``
    alone collides across rungs: ``cases_index.json`` then carries several
    indistinguishable ``val_step12000`` entries and nothing downstream can tell an
    R=2 case from an R=32 one — which silently mixes acceleration into any
    per-case comparison (e.g. ``scripts/diagnostics/spectral_sharpness.py``).
    The tie is also what made ``ReportCaseRecorder._evict_median`` crash on
    ``list.remove``'s ``==`` fallback over numpy arrays.

    Matches the saved-PNG convention (``validation_R32x``) so a case row and its
    render carry the same rung tag.
    """
    if cascade_level is None:
        return f"val_step{step}"
    tag = f"{cascade_level:.0f}" if float(cascade_level).is_integer() else f"{cascade_level:g}"
    return f"val_step{step}_R{tag}x"


def _identity_values(batch_data: Any, key: str) -> list[str]:
    """The per-sample values of ``key`` in a collated batch, as strings.

    ``ImageCollateStrategy`` stacks tensors but leaves non-tensor values as a
    plain per-sample list, so a string subject key arrives here as
    ``["a", "b"]`` for a batch of two. A dataset that publishes nothing returns
    ``[]`` — absent, which the caller reports as an absent column rather than
    inventing a placeholder.
    """
    if isinstance(batch_data, dict):
        raw = batch_data.get(key)
    elif batch_data is not None:
        raw = getattr(batch_data, key, None)
    else:
        raw = None
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    return [str(raw)]


def summarize_batch_identity(batch_data: Any) -> dict[str, Any]:
    """Which samples a per-case row averaged over — ``file_id``/``contrast``/``batch_size``.

    The metrics fed to the sink are means over a **batch**, so naming a single
    contrast would assert something the numbers do not support: the validation
    loader shuffles, so one batch can hold a T1 volume and a FLAIR one. A
    homogeneous batch reports its one contrast; a mixed batch reports every
    contrast present, joined with ``|``, so no row is ever read as belonging to
    a contrast it only partly covers.

    ``batch_size`` is emitted alongside for the same reason — it is what tells a
    reader the row is a mean over N volumes rather than one measurement. Set
    ``validation.loader.batch_size: 1`` and each row becomes exactly one volume.

    Returns ``{}`` when the dataset publishes no identity at all, so the columns
    are simply absent instead of filled with a placeholder that reads as data.
    """
    ids = _identity_values(batch_data, "file_id")
    contrasts = _identity_values(batch_data, "contrast")
    out: dict[str, Any] = {}
    if ids:
        out["file_id"] = ids[0] if len(ids) == 1 else "|".join(ids)
        out["batch_size"] = len(ids)
    if contrasts:
        unique = sorted(set(contrasts))
        out["contrast"] = unique[0] if len(unique) == 1 else "|".join(unique)
        out.setdefault("batch_size", len(contrasts))
    return out


def feed_report_case_recorder(
    recorder,
    *,
    predictions,
    targets,
    inputs,
    metrics,
    step,
    sink=None,
    cascade_level: float | None = None,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Push the first sample of a validation batch into the report recorder.

    Tensors may be torch or numpy; reduced to single-channel magnitude on
    CPU. Reuses tensors the validation-image path already moved to CPU, so it
    adds no GPU sync. No-op when the recorder is None/disabled.

    When ``sink`` (a :class:`PerCallMetricSink`) is enabled its per-case metric
    row is appended too, reusing the same already-detached metrics dict — so the
    unbounded ``per_call_metrics.csv`` is populated even when the (bounded) image
    recorder is disabled (``n_report_cases=0``).

    ``cascade_level`` tags the case with its acceleration rung — see
    :func:`report_case_id`.

    ``context`` carries the row's identity — acceleration, timestep, contrast,
    file id — into the sink as its own CSV columns. It is forwarded verbatim and
    NOT merged into ``metrics``: the sink coerces metrics to float, which would
    turn ``heldout`` into ``1.0`` and drop ``contrast`` on the floor. The
    recorder ignores it; only the sink writes a table with columns.
    """
    rec_on = recorder is not None and getattr(recorder, "enabled", False)
    sink_on = sink is not None and getattr(sink, "enabled", False)
    if not rec_on and not sink_on:
        return
    case_id = report_case_id(step, cascade_level)
    clean_metrics = {
        k: float(v)
        for k, v in (metrics or {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    if sink_on:
        sink.observe(
            case_id=case_id,
            metrics=clean_metrics,
            split="val",
            step=step,
            context=context,
        )
    if not rec_on:
        return
    import numpy as _np

    def _magnitude_np(t):
        """Detached CPU float32 magnitude array at its native rank (or None)."""
        if t is None:
            return None
        import torch as _torch

        # No ``except Exception: pass`` around the host copy (2026-09-03): a
        # tensor that cannot be detached or moved raised into a swallowed branch
        # and came back through ``np.asarray`` as whatever it happened to be.
        if isinstance(t, _torch.Tensor):
            if _torch.is_complex(t):
                t = _torch.abs(t)
            t = t.detach().to("cpu").float().numpy()
        return _np.asarray(t, dtype=_np.float32)

    def _to_mag_np(t):
        arr = _magnitude_np(t)
        if arr is None:
            return None
        while arr.ndim > 2:  # (B,C,H,W) -> first sample, channel-RSS
            if arr.ndim == 4:
                arr = _np.sqrt((arr[0] ** 2).sum(axis=0) + 1e-8)
            else:
                arr = arr[0]
        return arr

    def _to_mag_volume(t):
        """Preserve a genuine ``[Z, H, W]`` slice stack, or None.

        Only fires for an unambiguous slice axis: a 5-D ``[B, C, Z, H, W]``
        tensor (RSS the coils, keep the Z slices) or a 3-D ``[Z, H, W]`` volume.
        A 4-D ``[B, C, H, W]`` is a 2-D slice with coils — NOT a volume — so it
        returns None rather than fabricating depth from the coil axis.
        """
        arr = _magnitude_np(t)
        if arr is None or arr.ndim < 3:
            return None
        if arr.ndim == 5:  # [B,C,Z,H,W] -> sample 0 -> RSS coils -> [Z,H,W]
            arr = _np.sqrt((arr[0] ** 2).sum(axis=0) + 1e-8)
        elif arr.ndim == 4:  # [B,C,H,W] -> 2-D + coils, no genuine slice axis
            return None
        if arr.ndim != 3 or arr.shape[0] < 2:
            return None
        return arr

    arrays = {
        "input": _to_mag_np(inputs),
        "prediction": _to_mag_np(predictions),
        "target": _to_mag_np(targets),
    }
    if getattr(recorder, "record_volumes", False):
        for name, tensor in (
            ("input", inputs),
            ("prediction", predictions),
            ("target", targets),
        ):
            vol = _to_mag_volume(tensor)
            if vol is not None:
                arrays[f"{name}_volume"] = vol

    recorder.observe(
        case_id=case_id,
        arrays=arrays,
        metrics={k: float(v) for k, v in (metrics or {}).items() if isinstance(v, (int, float))},
        domain={},
    )


class MetricsMixin:
    """Mixin for metrics tracking and aggregation."""

    @property
    def training_metrics_computer(self) -> Any:
        """Lazy-loaded evaluation metrics computer for training."""
        return self._get_training_metrics_computer(self.config)

    @property
    def validation_metrics_computer(self) -> Any:
        """Lazy-loaded evaluation metrics computer for validation."""
        return self._get_validation_metrics_computer(self.config)

    def _get_training_metrics_computer(self, config: Any) -> Any:
        """Get or create the training metrics computer (lazy loaded)."""
        if hasattr(self, "_training_computer") and self._training_computer is not None:
            return self._training_computer

        # Build metric list from config flags
        enabled_metrics = []
        metrics_s = config.metrics if hasattr(config, "metrics") else None

        if metrics_s:
            # ✅ SSOT: Delegate to the comprehensive extractor (same as validation computer)
            # This ensures training and validation computers share the same metric→flag mapping
            enabled_metrics = self._extract_metrics_from_config(metrics_s)
        else:
            enabled_metrics = ["mse", "mae", "psnr", "ssim"]

        if hasattr(self, "logging_service"):
            self.logging_service.log_info(
                f"[Metrics Builder] Training metrics mapped from config: {enabled_metrics}"
            )

        domain = "image"
        if metrics_s and hasattr(metrics_s, "domain"):
            if metrics_s.domain:
                domain = metrics_s.domain

        if domain == "image":
            physics = config.physics if hasattr(config, "physics") else None
            if physics is not None:
                if hasattr(physics, "kspace") and hasattr(physics.kspace, "enable_kspace_recon"):
                    if physics.kspace.enable_kspace_recon:
                        domain = "kspace"

        data_range = None
        # ✅ SSOT: Direct access to data.data_range (trust schema default None)
        if hasattr(self.config, "data"):
            data_range = self.config.data.processing.data_range

        if data_range is None and metrics_s and hasattr(metrics_s, "data_range"):
            data_range = metrics_s.data_range

        from spectramr.core.metrics.computer import create_validation_metrics_computer

        self._training_computer = create_validation_metrics_computer(
            config=enabled_metrics,
            device=self.device,
            domain=domain,
            data_range=data_range,
            scale_contract=self._metric_scale_contract(data_range),
        )
        return self._training_computer

    def _metric_scale_contract(self, data_range: float | None) -> QuantileScaleContract | None:
        """The declared percentile scale both computers hand their image metrics (#2241).

        Logs once which of the three sources sets the image ``data_range``: the
        declared value, the normalization contract, or the per-batch sign test.
        """
        from spectramr.infrastructure.training.utils.metric_data_range import (
            scale_contract_from_config,
        )

        contract = scale_contract_from_config(getattr(self, "config", None))
        if not getattr(self, "_metric_range_source_logged", False):
            self._metric_range_source_logged = True
            if data_range is not None:
                logger.info("[Metrics] image data_range %g: declared", data_range)
            elif contract is not None:
                logger.info(
                    "[Metrics] image data_range %g: %s", contract.anchor, contract.describe()
                )
            else:
                logger.info(
                    "[Metrics] image data_range: inferred per batch from the target's sign "
                    "([0, 1] or [-1, 1]); no range or image-domain scale is declared"
                )
        return contract

    def _registry_backed_validation_metrics(self, declared: list[str]) -> list[str]:
        """Validate ``validation.scoring.compute`` against registry-or-emitted; return the registry part.

        ``metrics.compute`` has been gated against ``MetricsRegistry`` since #173;
        this list was not, and 13 arms used it to name keys their strategy's own
        ``validation_step`` writes (``field_mse`` on the qMRI arms, ``val_loss``
        on a path that writes it, see :func:`universal_validation_keys`)
        -- honest keys that the registry computer would have raised on at the
        first batch had it ever been built for those strategies. A strategy now
        DECLARES those keys in ``capabilities.emitted_metrics``; a name in
        neither set is an error when the strategy has declared its set, and an
        INFO census line when it has not (the ratchet, cohort review
        2026-09-02, T0.5). Emitted names are removed from the computer's list:
        the strategy produces them itself.
        """
        from spectramr.core.metrics.registry import MetricsRegistry
        from spectramr.core.metrics.validation_names import partition_validation_compute

        emitted = frozenset(getattr(type(self).capabilities, "emitted_metrics", frozenset()))
        registry_names, unknown = partition_validation_compute(
            declared,
            emitted,
            MetricsRegistry.is_registered,
            universal=universal_validation_keys(type(self)),
        )
        if unknown and emitted:
            raise ValueError(
                f"validation.scoring.compute names {len(unknown)} metric(s) that are neither "
                f"registered nor emitted by {type(self).__name__}: {sorted(unknown)}. "
                f"Emitted by this strategy: {sorted(emitted)}. Fix the name, register the "
                "metric, or declare it in the strategy's capabilities.emitted_metrics."
            )
        if unknown:
            logger.info(
                "[metrics] validation.scoring.compute names %s, which are not registered; "
                "%s declares no emitted_metrics, so they are left to its own validation_step "
                "(declare capabilities.emitted_metrics to make this an error).",
                sorted(unknown),
                type(self).__name__,
            )
        return registry_names

    def _get_validation_metrics_computer(self, config: Any) -> Any:
        """Get or create the validation metrics computer (lazy loaded)."""
        if hasattr(self, "_validation_computer") and self._validation_computer is not None:
            return self._validation_computer

        val_config = config.validation if hasattr(config, "validation") else None
        val_metrics = val_config.scoring.compute if val_config else None
        if val_metrics:
            val_metrics = self._registry_backed_validation_metrics(list(val_metrics))

        if val_metrics is None:
            # ✅ SSOT: Direct access to metrics config
            train_metrics_config = config.metrics if hasattr(config, "metrics") else None
            if train_metrics_config:
                val_metrics = self._extract_metrics_from_config(train_metrics_config)

        metrics_config = None
        if val_metrics is not None:
            metrics_config = {"metrics": val_metrics}
            if val_config:
                # ✅ SSOT: Direct access with default (schema has default="psnr")
                # Schema default is "psnr"; the field cannot be absent.
                primary = val_config.scoring.primary
                if primary:
                    metrics_config["primary_metric"] = primary
        elif val_config:
            metrics_config = val_config
        else:
            # ✅ SSOT: Direct access to metrics config
            train_metrics_config = config.metrics if hasattr(config, "metrics") else None
            if train_metrics_config:
                enabled_train_metrics = self._extract_metrics_from_config(train_metrics_config)
                metrics_config = {"metrics": enabled_train_metrics}

        domain = None
        if val_config:
            # Canonical leaf is validation.scoring.domain; the flat spelling is a
            # retired fold and reads None on every real config (same defect as
            # the transform resolver below).
            explicit_domain = _scoring_leaf(val_config, "domain", None)
            if explicit_domain:
                domain = explicit_domain

        if domain is None:
            # ✅ SSOT: Direct access to metrics config
            train_metrics_config = config.metrics if hasattr(config, "metrics") else None
            if train_metrics_config:
                train_domain = _get_config_value(train_metrics_config, "domain", None)
                if train_domain:
                    domain = train_domain

        if domain is None:
            physics = _get_config_value(config, "physics", None)
            if physics is not None:
                if isinstance(physics, dict):
                    kspace = physics.get("kspace", {})
                    if isinstance(kspace, dict) and kspace.get("enable_kspace_recon", False):
                        domain = "kspace"
                elif hasattr(physics, "kspace"):
                    if _get_config_value(physics.kspace, "enable_kspace_recon", False):
                        domain = "kspace"

        if domain is None:
            domain = "image"

        data_range = None
        # ✅ SSOT: Direct access to data.data_range
        if hasattr(self.config, "data"):
            data_range = self.config.data.processing.data_range

        if data_range is None and val_config:
            data_range = _get_config_value(val_config, "data_range", None)

        from spectramr.core.metrics.computer import create_validation_metrics_computer

        self._validation_computer = create_validation_metrics_computer(
            config=metrics_config,
            device=self.device,
            domain=domain,
            data_range=data_range,
            scale_contract=self._metric_scale_contract(data_range),
        )

        self._validate_metrics_domain_consistency(config)

        return self._validation_computer

    def _extract_metrics_from_config(self, metrics_s: Any) -> list[str]:
        """Extract list of enabled metrics from metrics configuration object or dict."""
        enabled_metrics = []

        # `metrics.compute` wins outright when present. It is not merged with the
        # compute_* flags: merging would mean the list understates what the arm
        # measures, and the point of the list is that it IS the answer. The
        # flags remain the fallback for the ~800 arms that predate it.
        declared = getattr(metrics_s, "compute", None)
        if declared:
            # An EXPLICIT name is validated strictly and raises (#173). Unlike the
            # legacy compute_* flags -- which are skipped with one warning above,
            # because 734 arms carry a dangling one and failing them at load helps
            # nobody -- a name written into `compute:` is a deliberate statement
            # about what this arm measures. A typo there used to be logged once
            # inside the computer and then produce a silently missing CSV column
            # for the whole run, which is indistinguishable from "that metric was
            # never any good on this data".
            #
            # This lives here, not on the schema: `config/` cannot import
            # `core.metrics.registry` (core already imports `config.schemas.enums`,
            # so the reverse edge would be a cycle). Strategy construction is
            # still before the first batch, which is what matters.
            unknown = [m for m in declared if not MetricsRegistry.is_registered(m)]
            if unknown:
                raise ValueError(
                    f"metrics.compute names {len(unknown)} metric(s) that are not "
                    f"registered: {sorted(unknown)}.\n"
                    "A name that is not in MetricsRegistry can never produce a "
                    "column, so the run would report a silently missing metric "
                    "rather than a failure. Fix the name, or register the metric "
                    "with @register_metric."
                )
            return list(declared)

        if isinstance(metrics_s, dict):
            for metric_name, enabled_flag in metrics_s.items():
                if enabled_flag is True:
                    enabled_metrics.append(metric_name)
        elif hasattr(metrics_s, "__dict__"):
            # Flag -> metric name, DERIVED from MetricsConfigSchema rather than
            # hand-listed (#340). The hand-written version had 43 entries against a
            # schema of 86, and the 43 were not a deliberate policy -- 22 of the
            # missing flags resolve to a metric that IS registered, so an arm could
            # set `compute_wm2max: true`, receive a `losses.csv` column for it from
            # `training_loop._CSV_METRIC_NAME_MAP` (78 entries, also hand-written),
            # and never have the metric selected. A header over a permanently empty
            # column is worse than no column: it reads as a measurement that came
            # back blank. Deriving both from the schema makes a flag reachable by
            # construction, so the two maps cannot drift apart again.
            compute_flags = schema_flag_to_metric()
            # Flag-selected names are filtered against the registry HERE, not left
            # to fail one-by-one inside the computer (#173/#660). 17 of the schema's
            # 86 flags name a metric that is not registered -- measured through
            # `MetricsRegistry.is_registered`, which resolves the 296-entry ALIAS
            # table as well; counting against `_metrics` alone (as CLAUDE.md's
            # snippet does) calls 4 more dangling than really are, because `fwhm`,
            # `gsr`, `ndc` and `volume_similarity` are aliases of canonical names.
            # Each used to reach `MetricsRegistry.get`, raise KeyError, and be
            # logged-and-skipped per metric -- yielding a silently missing CSV
            # column -- while compute_precision_recall failed differently again, as
            # an `UnknownMetricDirectionError` whose message is about DIRECTIONS and
            # says nothing about the real cause.
            #
            # These are legacy booleans, not user typos, so a dangling FLAG warns
            # rather than raising; an explicit name in `metrics.compute` is a
            # deliberate statement and is validated strictly above. The one flag
            # that would have broken the corpus, `compute_advanced_metrics`, never
            # reaches here: it names no metric at all and is excluded at the SSOT
            # (`flag_map.NON_METRIC_FLAGS`). It defaults **True**, so treating it as
            # a dangling metric name would fire this warning on EVERY arm -- and
            # warnings exit 2 under `audit --strict` (non-negotiable #4).
            unresolvable: list[str] = []
            for flag_attr, metric_name in compute_flags.items():
                # ✅ SSOT: hasattr guards, getattr for dynamic access
                if hasattr(metrics_s, flag_attr) and getattr(metrics_s, flag_attr):
                    if not MetricsRegistry.is_registered(metric_name):
                        unresolvable.append(f"{flag_attr} -> '{metric_name}'")
                        continue
                    enabled_metrics.append(metric_name)

            if unresolvable and not getattr(self, "_dangling_flags_warned", False):
                # Once per run, naming every one: a per-metric warning inside the
                # computer scrolled past and read as an incident rather than a
                # standing config-surface defect.
                logger.warning(
                    "[Metrics] %d enabled compute_* flag(s) name no registered "
                    "metric and were skipped: %s. These flags cannot produce a "
                    "column; remove them from the arm or register the metric.",
                    len(unresolvable),
                    ", ".join(sorted(unresolvable)),
                )
                self._dangling_flags_warned = True

            # NOTE: the no-reference (NR) metric battery (``metrics.nr``) is
            # DELIBERATELY NOT selected here. NR metrics are research-mode /
            # validation-pending and run ONLY through the offline meta-evaluation
            # harness (``spectramr meta-evaluate --nr-battery``), which synthesises
            # its own MetricContext per degraded sample. They must never execute
            # during training. The ``nr_metrics_research_mode`` audit check warns
            # if a training config lists them. See docs/no_reference_metrics.rst.

        return enabled_metrics if enabled_metrics else ["psnr", "ssim"]

    def _validate_metrics_domain_consistency(self, config: Any) -> None:
        """Validate that training and validation metrics use consistent domains."""
        if not hasattr(self, "logging_service"):
            return

        train_domain_actual = "image"
        metrics_config = _get_config_value(config, "metrics", None)
        if metrics_config:
            train_domain = _get_config_value(metrics_config, "domain", None)
            if train_domain:
                train_domain_actual = train_domain
            train_transform = _get_config_value(metrics_config, "transform", None)
            if train_transform == "ifft_magnitude":
                train_domain_actual = "image"
        else:
            physics = _get_config_value(config, "physics", None)
            if physics is not None:
                if isinstance(physics, dict):
                    kspace = physics.get("kspace", {})
                    if isinstance(kspace, dict) and kspace.get("enable_kspace_recon", False):
                        train_domain_actual = "kspace"
                elif hasattr(physics, "kspace"):
                    if _get_config_value(physics.kspace, "enable_kspace_recon", False):
                        train_domain_actual = "kspace"

        val_domain_actual = "image"
        val_config = _get_config_value(config, "validation", None)
        if val_config:
            val_domain = _get_config_value(val_config, "domain", None)
            if val_domain:
                val_domain_actual = val_domain
            else:
                val_domain_actual = train_domain_actual

            val_transform = _get_config_value(val_config, "output_transform", None)
            if val_transform in ("ifft_magnitude", "ifft_target_magnitude", "sense_magnitude"):
                val_domain_actual = "image"
        else:
            val_domain_actual = train_domain_actual

        if train_domain_actual != val_domain_actual:
            self.logging_service.log_warning(
                f"[Metrics Domain Mismatch] Training metrics computed in '{train_domain_actual}' domain, "
                f"but validation metrics computed in '{val_domain_actual}' domain."
            )

    def _prediction_for_visualization(self, predictions: torch.Tensor) -> torch.Tensor:
        """Reduce the raw validation prediction to the tensor that should be VISUALISED.

        The visualization sibling of :meth:`_apply_metric_transforms`. Default: identity.
        A distribution-head strategy whose generator emits e.g. a 2-channel
        ``[mean, logvar]`` (``HeteroscedasticULFStrategy``) overrides this to return the
        point estimate (channel 0), so the saved ``fake_images`` / TensorBoard previews
        show the restored image instead of the ``sqrt(mean**2 + logvar**2)`` channel-RSS
        blend that ``to_magnitude`` produces for any even-channel tensor (issue #371 - the
        b29 dark-brain-on-grey wrong-channel viz). The metric and viz reducers must be
        overridden together for a head that is NOT complex real/imag.
        Default is no longer identity (#390/#709). It delegates to
        :class:`VisualizationReducer`, which resolves "is this a distribution
        head?" through the SAME declaration the width guard in
        ``BaseTrainingStrategy.train_step`` uses -- ``model_type ==
        "evidential_unet"`` or ``predicts_distribution_params``. The width guard
        already had to know; the visualization side asked separately and got a
        different answer, which is why ``evidential_unet`` rendered its four
        parameters as ``sqrt(Σ params²)`` while ``heteroscedastic_ulf``, whose
        strategy overrides this method, rendered correctly.

        For anything that is NOT a distribution head this is still identity, and
        that matters: a 2-channel COMPLEX tensor must reach ``to_magnitude``
        intact. Taking channel 0 of one would show the real part and call it the
        image.
        """
        from spectramr.infrastructure.training.utils.visualization_reducer import (
            VisualizationReducer,
        )

        # `getattr`, not `self.config`: a bare mixin constructed in a unit test
        # has no config, and the safe answer there is "not a distribution head"
        # -> identity, i.e. exactly the previous behaviour. A real strategy
        # always has one.
        return VisualizationReducer.from_config(getattr(self, "config", None), self).point_estimate(
            predictions
        )

    def _log_metric_transform_once(self, resolution: MetricTransformResolution) -> None:
        """Announce which declaration decided the metric transform.

        Pitfall #15's third obligation: a wired knob must reach provenance. This
        function runs per validation batch, so the line is emitted once per
        resolved value and then only when it changes — enough to answer "which
        knob fired?" from a run log, which is precisely the question #927 and
        #931 could not be answered from.
        """
        if getattr(self, "_metric_transform_logged", None) == resolution:
            return
        self._metric_transform_logged = resolution
        logger.info("[metrics] transform: %s", resolution.describe())

    def _apply_metric_transforms(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        metrics_config: Any,
        *,
        batch_data: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply configured domain transforms before metric computation.

        Returns only the magnitude pair. A caller that scores a metric set
        should use :meth:`_metric_transform_views` instead, so phase-sensitive
        metrics receive the complex image rather than its magnitude (#867).
        """
        views = self._metric_transform_views(pred, target, metrics_config, batch_data=batch_data)
        return views.pred, views.target

    def _metric_transform_views(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        metrics_config: Any,
        *,
        batch_data: Any = None,
        smaps: torch.Tensor | None = None,
    ) -> MetricViews:
        """Apply the configured transform, keeping the complex image it passes through.

        ``batch_data`` is the validation batch, read only by ``sense_magnitude``
        for its coil maps; the other transforms need nothing beyond the tensors.
        ``smaps`` hands ``ifft_sense_adjoint`` an explicit map set in place of the
        strategy's own, which is how cold-diffusion validation re-scores a
        prediction with measurement-only maps; no other transform reads it.
        """
        # NOTE: the metric-combine convention comes ONLY from the
        # metrics/validation transform — NOT from physics.coil_processing.combine,
        # which is the DATA-LOAD combine (a different pipeline stage: a model may
        # receive multi-coil input yet have its output metric-combined, and SENSE
        # needs smaps that exist only post-model). For SENSE metrics set
        # validation.scoring.output_transform: ifft_sense_adjoint.
        #
        # Which name wins is decided by ``resolve_metric_transform`` and nowhere
        # else — the audit's ``metric_domain_matches_loss_output`` asks the same
        # resolver, so the check that says "a transform bridges these domains"
        # and the code that runs one can no longer disagree (pitfall 13b).
        #
        # ``metrics_config`` is misnamed by history: the validation callers pass
        # ``config.validation`` and ``_compute_training_metrics`` passes the
        # ``metrics`` block, which is how ``metrics.transform`` reaches the
        # dispatcher on the training path (and only there). Both land in the
        # first slot, so that asymmetry is preserved exactly, not resolved here.
        #
        # ``getattr``, not ``self.config``: a bare mixin constructed in a unit
        # test has no config, and the old code only reached for one when the
        # passed-in block yielded no name — so requiring it here would break
        # callers that never needed it (same reason as ``_reduce_for_metrics``).
        own_config = getattr(self, "config", None)
        resolution = resolve_metric_transform(
            None if isinstance(metrics_config, str) else metrics_config,
            _get_config_value(own_config, "metrics", None),
            caller_override=metrics_config if isinstance(metrics_config, str) else None,
            fallback_validation_config=_get_config_value(own_config, "validation", None),
        )
        self._log_metric_transform_once(resolution)

        if resolution.suppressed:
            # ``domain: none`` outranks the auto-magnitude gate below: the user
            # asked for metrics on the raw tensors, not for a combine we picked.
            return MetricViews(pred, target)

        if not resolution.is_implemented:
            # Pitfall #9, and the actual defect behind #931. This chain has no
            # ``else``, so a name no branch matched fell out of the bottom and
            # the tensors were returned unchanged — 146 arms declare
            # ``ifft_mag_combine`` / ``ifft_mag``, which no branch implements,
            # and were told nothing. Deliberately NOT aliased onto
            # ``ifft_magnitude``: 112 of those arms output images, where an IFFT
            # produces a Fourier magnitude rather than the combine they meant.
            raise ValueError(
                f"Unknown metric transform {resolution.name!r} declared at "
                f"{resolution.source}. Implemented: "
                f"{sorted(IMPLEMENTED_METRIC_TRANSFORMS)}. Remove the key, or "
                "name one of those."
            )

        transform_name = resolution.name
        if transform_name is None:
            if torch.is_complex(pred) or (pred.dim() == 4 and pred.shape[1] == 2):
                transform_name = "magnitude"

        if transform_name is None:
            return MetricViews(pred, target)

        complex_pred: torch.Tensor | None = None
        complex_target: torch.Tensor | None = None

        if transform_name == "ifft_magnitude":
            from spectramr.infrastructure.physics.fft_ops import ifft2c

            # [FIX] For multi-contrast single-coil data (e.g., 8ch target =
            # 4 contrasts × 1 coil × 2 R/I), RSS over "4 complex channels"
            # would mix different contrasts (T1/FLAIR/etc.), producing a PSF
            # blob instead of anatomy.  Slice to one contrast's R/I pair.
            #
            # Only applies when virtual-coil compression is *active* — i.e.
            # ``coil_processing_mode`` is one of {svd, compress}. With
            # ``coil_processing_mode: none`` the channel count equals the
            # number of physical coils × 2 (R/I); ``num_virtual_coils`` is
            # ignored by the data pipeline in that mode and using it here
            # would slice off real coil information (the experiment_11 May
            # 2026 "doubled brain" symptom: 8ch single-contrast 4-coil data
            # was sliced to first 2 coils, leaving a non-uniform spatial
            # sensitivity that reads as brain + faint echo).
            data_cfg = _get_config_value(self.config, "data", None)
            coil_mode = str(_get_config_value(data_cfg, "coil_processing_mode", "") or "").lower()
            num_vc = _get_config_value(data_cfg, "num_virtual_coils", None)
            virtual_coils_active = coil_mode in {"svd", "compress"}
            if virtual_coils_active and num_vc is not None and num_vc > 0:
                channels_per_contrast = 2 * num_vc  # R/I per virtual coil
                if pred.shape[1] > channels_per_contrast:
                    logger.debug(
                        "[ifft_magnitude] Multi-contrast detected: slicing "
                        "%d → %d ch (first contrast, %d virtual coils, "
                        "coil_mode=%s)",
                        pred.shape[1],
                        channels_per_contrast,
                        num_vc,
                        coil_mode,
                    )
                    pred = pred[:, :channels_per_contrast]
                    target = target[:, :channels_per_contrast]

            # Pair real-stacked channels [R0, I0, R1, I1, ...] into complex
            # BEFORE IFFT. Calling ifft2c on a REAL tensor treats each channel
            # as an independent real-valued k-space; the IFFT of a real signal
            # is Hermitian-symmetric, and its magnitude is centro-symmetric —
            # i.e. the brain superimposed with its 180°-rotated copy. That is
            # the experiment_11 / experiment_130 / cross-contrast May-2026
            # "doubled brain" symptom in the saved validation_R*.png images.
            pred_kc = real_stacked_to_complex(pred)
            target_kc = real_stacked_to_complex(target)

            if torch.is_complex(pred_kc):
                # Per-coil complex images: without smaps there is no
                # phase-preserving coil combine, so phase metrics see each coil.
                complex_pred = ifft2c(pred_kc)
                complex_target = ifft2c(target_kc)
                pred = torch.abs(complex_pred)
                target = torch.abs(complex_target)
            else:
                # Single-channel real fallback. iFFT on a strictly-real tensor
                # has a Hermitian-symmetric output and its magnitude is
                # centro-symmetric — that is the experiment_11 "doubled brain"
                # symptom in the validation_R*.png images. Calling ifft2c here
                # was the previous behaviour and *guaranteed* an artefact, so
                # we now skip the transform and treat the 1-channel tensor as
                # already-magnitude (the most common upstream cause of this
                # branch is a strategy that magnitudes early). We still warn
                # so a mis-set ``data.target_channels`` is visible in the log
                # — the warning is the actionable signal, not the centro-
                # symmetric magnitude.
                logger.warning(
                    "[ifft_magnitude] pred has %d channel(s) — cannot pair as "
                    "complex; treating as already-magnitude image (skipping "
                    "iFFT) to avoid the centro-symmetric 'doubled brain' "
                    "artefact. Check data.target_channels (must equal "
                    "channels-per-target-contrast).",
                    pred.shape[1],
                )
                # If pred is already real, just keep magnitude (no-op for
                # nonneg tensors; abs() makes the contract explicit).
                pred = pred.abs()
                target = target.abs()

            if pred.shape[1] > 1:
                pred = torch.sqrt(torch.sum(pred**2, dim=1, keepdim=True))
                target = torch.sqrt(torch.sum(target**2, dim=1, keepdim=True))

        elif transform_name == "ifft_target_magnitude":
            # Image-domain prediction, k-space target: only the target crosses
            # domains (#2325). ``metric_transform`` documents why it is declared.
            pred = image_magnitude(pred)
            target = kspace_rss_magnitude(target)

        elif transform_name == "sense_magnitude":
            # The validation batch's scoring maps (#2664), else its own maps; not
            # ``_current_smaps``, which only the diffusion strategies set. Missing
            # maps raise inside the pair.
            maps, _ = scoring_maps(batch_data, pred) if batch_data is not None else (None, None)
            sense_pred, sense_target = sense_scoring_pair(pred, target, maps)
            complex_pred = sense_pred if torch.is_complex(sense_pred) else None
            complex_target = sense_target
            pred = sense_pred.abs()
            target = sense_target.abs()

        elif transform_name == "ifft_sense_adjoint":
            from spectramr.infrastructure.physics.fft_ops import sense_adjoint

            # Sensitivity maps are populated per strategy. Prefer the per-step
            # ``_current_smaps`` over the validation-scoped ``_cached_smaps`` and
            # accept a map only when its batch broadcasts against the current
            # prediction — otherwise a stale validation-batch map crashes
            # ``sense_adjoint`` at dim 0. See ``_select_batch_compatible_smaps``.
            if smaps is None:
                smaps = self._select_batch_compatible_smaps(pred.shape[0])
            coils = pred.shape[1] if torch.is_complex(pred) else pred.shape[1] // 2
            if smaps is None and coils > 1 and not getattr(self, "_sense_no_maps_warned", False):
                self._sense_no_maps_warned = True
                logger.warning(
                    "[ifft_sense_adjoint] no sensitivity maps for a %d-coil prediction: the "
                    "metrics grade per-coil iFFT magnitudes, not the SENSE-combined image the "
                    "transform names. Further occurrences are not repeated.",
                    coils,
                )
            complex_pred = sense_adjoint(pred, smaps=smaps)
            complex_target = sense_adjoint(target, smaps=smaps)
            pred = torch.abs(complex_pred)
            target = torch.abs(complex_target)

        elif transform_name == "magnitude":
            if torch.is_complex(pred):
                complex_pred, complex_target = pred, target
                pred = torch.abs(pred)
                target = torch.abs(target)
            elif pred.dim() == 4 and pred.shape[1] == 2:
                # Gated on the PREDICTION being 2-ch real-stacked complex, but the target
                # used to be indexed at [:, 1] unconditionally: a 1-channel magnitude
                # target raised IndexError, which ModelValidationMixin swallows in a bare
                # `except Exception` -> the run silently drops to a psnr/mse-only metric
                # set -> the configured early-stopping monitor key is never emitted ->
                # nothing ever stops training (mrixfields_b29 burned the full 150k iters
                # / ~23 h GPU while val_psnr diverged -9 -> -21 dB). Resolve the target on
                # its OWN shape.
                complex_pred = torch.complex(pred[:, 0], pred[:, 1]).unsqueeze(1)
                pred = complex_pred.abs()
                if target.dim() == 4 and target.shape[1] == 2:
                    complex_target = torch.complex(target[:, 0], target[:, 1]).unsqueeze(1)
                    target = complex_target.abs()
                elif torch.is_complex(target):
                    complex_target = target
                    target = torch.abs(target)
                elif target.dim() == 4 and target.shape[1] == 1:
                    # A real 1-channel reference has no phase: complex_target
                    # stays None and a phase metric raises in the computer.
                    target = torch.abs(target)
                else:
                    # Comparing |pred| against a target we cannot map to a magnitude
                    # produces a meaningless metric, so refuse to guess. NOTE: this is
                    # not yet fail-loud end-to-end — ModelValidationMixin still catches
                    # every non-OOM exception and degrades to a psnr/ssim fallback set
                    # (issue #181). Until that lands the raise is authoritative only for
                    # direct callers; in the live validation path it downgrades the
                    # metric set rather than failing the run.
                    raise ValueError(
                        "metrics transform='magnitude': prediction is 2-channel real-stacked "
                        f"complex but target has shape {tuple(target.shape)}; expected a "
                        "2-channel real-stacked, complex, or 1-channel magnitude target."
                    )

        return MetricViews(pred, target, complex_pred, complex_target)

    def _select_batch_compatible_smaps(self, batch: int) -> torch.Tensor | None:
        """Return sensitivity maps whose batch broadcasts against ``batch``.

        Prefers the per-step ``_current_smaps`` over the validation-scoped
        ``_cached_smaps``. Validation writes BOTH to the validation batch
        (``diffusion.py`` ``_generate_validation_prediction``, lines 3325-3326)
        while a training step refreshes only ``_current_smaps`` (line 2227); so
        after a validation pass ``_cached_smaps`` holds a STALE validation batch.
        Used unchecked in the next training-metric step it broadcasts against the
        smaller training prediction and crashes ``sense_adjoint`` at dim 0 — the
        experiment_11 EMA-ablation iter-1001 ``tensor a (2) vs b (36)`` failure.

        A map is accepted only when its leading dim is 1 (broadcastable) or
        equal to ``batch``. An incompatible map is dropped with a loud warning
        so the caller falls back to per-coil iFFT magnitude rather than silently
        degrading (CLAUDE.md #9/#10). ``_cached_smaps`` is only ever written
        alongside an identical ``_current_smaps``, so preferring the per-step map
        never loses information — it additionally keeps SENSE training metrics
        alive across the validation→training boundary instead of disabling them.
        """
        for attr in ("_current_smaps", "_cached_smaps"):
            smaps = getattr(self, attr, None)
            if smaps is None:
                continue
            if smaps.shape[0] in (1, batch):
                return smaps
            logger.warning(
                "[ifft_sense_adjoint] %s batch %d != pred batch %d — stale "
                "validation-scoped sensitivity maps leaking into a metric step; "
                "dropping SENSE-adjoint smaps (per-coil iFFT magnitude). Clear "
                "_cached_smaps after validation to avoid this.",
                attr,
                smaps.shape[0],
                batch,
            )
        return None

    def _compute_training_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        config: Any = None,
        current_step: int = 0,
        batch_data: Any = None,
    ) -> dict[str, float]:
        """Compute reconstruction quality metrics during training.

        ``batch_data`` is the training batch, read only by ``sense_magnitude`` for
        its coil maps.
        """
        metrics: dict[str, float] = {}

        metrics_config = _get_config_value(config, "metrics", None)
        enable_tracking = _get_config_value(metrics_config, "enable_tracking", True)
        if not enable_tracking:
            return metrics

        train_metric_interval = _get_config_value(metrics_config, "train_metric_interval", 100)

        # Fire on the cadence, AND unconditionally on the first and final
        # iteration -- mirroring the CSV row gate at `training_loop.py`, which
        # carries `or is_first_iteration or is_last_iteration` for exactly this
        # reason.
        #
        # Without the override this throttle CANNOT FIRE AT ALL on a run whose
        # budget is under the cadence: over iterations 1..40 with the schema
        # default interval of 100 the satisfying set is empty, so every `train_*`
        # column in `training_metrics.csv` stays blank while the loss columns
        # fill normally. Measured on `experiment_11_attention_none`.
        #
        # That is a REGRESSION INTRODUCED BY A CORRECT FIX, which is why it went
        # unnoticed: `self.env.step` used to be a frozen 0, so
        # `0 % train_metric_interval == 0` was true on EVERY step and the
        # throttle was a facade (pitfall #16). `resolve_loop_iteration` now
        # returns the live 1-based iteration -- correct, but it flipped short
        # runs from always-on to never, and the short-run case was never
        # re-checked. Short runs are not exotic: every smoke run, every
        # `--max-iterations` debug override and any budget under 100 hits this.
        #
        # `max_iterations` is the budget as DECLARED, which makes the final-step
        # leg an approximation: an epoch-bounded run can stop before reaching it,
        # in which case only the first-iteration leg fires and the curve is a
        # single point. The loop knows its own last iteration and the mixin does
        # not, so this is the best available signal here -- a one-point curve
        # still beats a structurally empty column.
        #
        # A non-positive interval has no periodic firing rather than raising
        # `ZeroDivisionError`; first/final still produce data.
        training_config = _get_config_value(config, "training", None)
        max_iterations = _get_config_value(training_config, "max_iterations", 0)

        on_interval = train_metric_interval > 0 and current_step % train_metric_interval == 0
        is_first = current_step == 1
        is_last = (
            isinstance(max_iterations, int)
            and max_iterations > 0
            and current_step == max_iterations
        )
        if not (on_interval or is_first or is_last):
            return metrics

        # Guard: some generators (e.g. DisentangledMRIGenerator) return a dict.
        # Extract the primary reconstruction tensor before any shape operations.
        if isinstance(pred, dict):
            pred = pred.get(
                "reconstruction",
                pred.get("output", next(iter(pred.values()))),
            )
        if isinstance(target, dict):
            target = target.get(
                "reconstruction",
                target.get("target", next(iter(target.values()))),
            )

        if pred.shape != target.shape:
            try:
                pred, target = TorchIOAdapter.ensure_channel_match(pred, target)
            except Exception as _exc:
                logger.debug("Suppressed exception: %s", _exc)

        if pred.shape != target.shape:
            return metrics

        pred, target = self._training_metric_inputs(pred, target)
        views = self._metric_transform_views(pred, target, metrics_config, batch_data=batch_data)

        with torch.no_grad():
            raw_metrics = views.compute(self.training_metrics_computer)

        return {f"train_{k}": v for k, v in raw_metrics.items()}

    def _training_metric_inputs(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Put the training pair in the units validation scores in; identity unless overridden."""
        return pred, target

    @torch.no_grad()
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Default validation step using ValidationMetricsComputer.

        Returns ``{}`` without a forward pass when the computer has no metrics,
        i.e. every name in ``validation.scoring.compute`` is one the strategy
        emits itself; its output need not be an image (#2839).
        """
        computer = self.validation_metrics_computer
        if not computer.get_metric_names():
            return {}
        with torch.no_grad():
            # Ensure model is in eval mode
            self.env.generator.eval()

            input_batch = input_batch.to(self.device, non_blocking=True)
            target_batch = target_batch.to(self.device, non_blocking=True)

            output = self.env.generator(input_batch)
            metrics = self._score_validation_output(output, target_batch)

            self.env.generator.train()

            return metrics

    def _score_validation_output(self, output: Any, target_batch: torch.Tensor) -> dict[str, float]:
        """Unpack a generator output and compute the validation metrics on it."""
        if isinstance(output, tuple):
            output = output[0]
        elif isinstance(output, dict):
            output = output.get("reconstruction", output.get("output", output))

        if output.device != target_batch.device:
            output = output.to(target_batch.device)

        val_config = _get_config_value(self.config, "validation", None)
        views = self._metric_transform_views(output, target_batch, val_config)
        return views.compute(self.validation_metrics_computer)

    def reset_validation_metrics(self) -> None:
        """Reset validation metrics computer."""
        if hasattr(self, "_validation_computer") and self._validation_computer:
            self._validation_computer.reset()

    def finalize_validation(self) -> dict[str, float]:
        """Finalize stateful validation metrics."""
        if hasattr(self, "_validation_computer") and self._validation_computer:
            return self._validation_computer.finalize()
        return {}

    def get_primary_metric(self) -> str:
        """Get the primary validation metric name."""
        return self.validation_metrics_computer.get_primary_metric()

    def is_metric_improvement(
        self,
        current: float,
        best: float,
        metric_name: str | None = None,
    ) -> bool:
        """Check if current value is an improvement over best."""
        return self.validation_metrics_computer.is_improvement(current, best, metric_name)

    @staticmethod
    def _convert_metrics_to_floats(
        metrics: dict[str, Any],
    ) -> dict[str, float]:
        """Convert all metrics to float scalars, with one host transfer.

        Each entry is reduced by :func:`reduce_on_device` (real part of complex,
        ``mean()`` of non-scalar), and the tensor scalars cross to the host in a
        single fused transfer rather than one ``.item()`` per entry (audit
        TRAIN-1). The strategies that score on the host call this; the per-step
        ``_handle_anomalies`` does not, so it never syncs. Raises TypeError for
        an unconvertible entry.
        """
        converted: dict[str, float] = {}
        tensor_keys: list[str] = []
        tensor_scalars: list[torch.Tensor] = []
        for key, value in metrics.items():
            reduced = reduce_on_device(key, value)
            if isinstance(reduced, torch.Tensor):
                # float64 gives the fused stack one dtype under AMP (mixed fp16/
                # fp32) and widens losslessly, so values match a per-key .item().
                tensor_keys.append(key)
                tensor_scalars.append(reduced.double())
            else:
                converted[key] = reduced
        if tensor_scalars:
            # The transfer mechanism lives in core so `core.metrics.computer` can
            # share it; the reduction stays with the caller, because mean-over-
            # batch is right for a loss and wrong for a metric.
            for key, host_val in zip(tensor_keys, fuse_to_host(tensor_scalars), strict=True):
                converted[key] = float(host_val)
        return converted

    def _log_validation_images_to_tensorboard(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        inputs: torch.Tensor,
        metrics: dict,
    ) -> None:
        """Log validation predictions and targets to TensorBoard as images."""
        if not hasattr(self, "logging_service") or self.logging_service is None:
            return

        try:
            # The TRAINING iteration, not a call counter (#585). `step` used to be
            # `validation_step_count` -- bumped once per validation call (and once
            # per cascade level) -- with a fallback to `self.env.current_step` that
            # can NEVER fire: `current_step` exists on neither `TrainingEnvironment`
            # nor the builder's environment, so the `hasattr` is always False and
            # the label was always the counter. Saved images were therefore
            # numbered 0,1,2,... while the run reported iterations 500,1000,1500,
            # and nothing tied a picture to the checkpoint that produced it.
            #
            # `resolve_loop_iteration` is the existing SSOT for this (6+ strategies
            # already use it, and `diffusion.py` overrode this very method to reach
            # it); it degrades to 0 for a mixin constructed standalone in a test.
            step = resolve_loop_iteration(self)

            # ✅ SSOT: Extract logging config from strategy's immutable config (TrainingSettings frozen=True)
            logging_s = self.config.logging if hasattr(self.config, "logging") else None
            if logging_s and not (logging_s.images.log_validation if logging_s else True):
                return

            # ✅ SSOT: Extract max_images_per_batch from logging schema (default: 4)
            max_images = 4  # Schema default
            if logging_s:
                max_images = logging_s.images.max_per_batch

            def _ensure_4d(t: torch.Tensor) -> torch.Tensor:
                """Ensure tensor is 4D BCHW format for image logging.

                Handles:
                - 2D: (H, W) → (1, 1, H, W)
                - 3D: (C, H, W) → (1, C, H, W)
                - 4D: (B, C, H, W) → unchanged
                - 5D: (B, C, H, W, D) → (B*D, C, H, W) [flatten batch & depth]
                """
                if t.dim() == 2:
                    return t.unsqueeze(0).unsqueeze(0)
                if t.dim() == 3:
                    return t.unsqueeze(0)
                if t.dim() == 4:
                    return t
                if t.dim() == 5:
                    # Handle volumetric data: (B, C, H, W, D) → (B*D, C, H, W)
                    b, c, h, w, d = t.shape
                    return t.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
                return t

            def to_magnitude(t):
                """Reduce a possibly-multi-channel image-domain tensor to
                single-channel magnitude.

                Image-domain visualization MUST NOT assume that an even
                channel count means (R, I) interleaved — paired-modality
                data (ULF/HF, T1/T2, …) also has even channel count and
                pairing-as-complex would mix the modalities into a
                ``sqrt(M_a² + M_b²)`` blend (the "doubled and odd"
                regression). For genuine real-stacked complex tensors,
                channel-RSS is mathematically equivalent to the
                per-pair-magnitude+coil-RSS chain (see
                ``debug_snapshot._render_image_preview`` for the proof).
                """
                if torch.is_complex(t):
                    return torch.abs(t)
                # Multi-channel real tensor → RSS along channel axis.
                if t.dim() == 5 and t.shape[1] > 1:
                    return torch.sqrt((t**2).sum(dim=1, keepdim=True) + 1e-8)
                if t.dim() == 4 and t.shape[1] > 1:
                    return torch.sqrt((t**2).sum(dim=1, keepdim=True) + 1e-8)
                return t

            def kspace_to_image(ksp):
                """Convert k-space to image domain.

                Applies IFFT when the model operates in k-space domain.
                Uses the authoritative domain inference utility (SSOT)
                from domain_inference.py. See docs/DOMAIN_HANDLING_RULES.md.

                Also checks metrics.transform for explicit IFFT/SENSE requests.

                Handles both 4D and 5D volumetric data.
                Supports SENSE adjoint when configured via metrics.transform.
                Output is always 4D or less (never 5D).
                """
                try:
                    # ✅ SSOT: Use authoritative domain inference
                    from spectramr.infrastructure.training.utils.domain_inference import (
                        infer_output_domain,
                    )

                    output_domain = infer_output_domain(self.config)
                    is_kspace = output_domain == "kspace"

                    # Override: metrics.transform with 'ifft' always forces IFFT
                    if not is_kspace:
                        _metrics_cfg = _get_config_value(self.config, "metrics", None)
                        _transform = _get_config_value(_metrics_cfg, "transform", None)
                        if _transform and "ifft" in str(_transform).lower():
                            is_kspace = True

                    if not is_kspace:
                        # F14 (2026-05-17 round 7): refuse to silently render
                        # a k-space-shaped tensor as if it were image-domain.
                        # The pre-F14 ``return to_magnitude(ksp)`` produced
                        # the "white spot in the middle" symptom for every
                        # config whose ``infer_output_domain`` returned
                        # ``"image"`` while the model's forward actually
                        # emitted k-space (the experiment_11 fourier-bridge /
                        # cold-diffusion / cross_contrast cohort). The
                        # magnitude plot of k-space is dominated by the DC
                        # spike at the centred origin → bright center pixel.
                        # CLAUDE.md #9 forbids silent fallbacks; the audit
                        # must observe and raise.
                        looks_like_kspace = torch.is_complex(ksp) or (
                            ksp.dim() in (4, 5) and ksp.shape[1] >= 2 and ksp.shape[1] % 2 == 0
                        )
                        if looks_like_kspace:
                            raise ValueError(
                                f"[MetricsMixin.kspace_to_image] "
                                f"infer_output_domain returned 'image' but "
                                f"tensor shape={tuple(ksp.shape)} "
                                f"(dtype={ksp.dtype}, is_complex={torch.is_complex(ksp)}) "
                                f"is k-space-shaped (complex or even-channel "
                                f"real-stacked). Rendering this as a "
                                f"magnitude image produces the DC-spike "
                                f"'white spot in the middle' artifact. Fix "
                                f"the YAML's model.model_domain / "
                                f"model.input_type, or set "
                                f"metrics.transform='ifft' to force IFFT "
                                f"on the visualization path. "
                                f"See TODO/audit/smoke_audit_20260516.md §F14."
                            )
                        # Image-domain output → magnitude only, no IFFT
                        return to_magnitude(ksp)

                    from spectramr.infrastructure.physics.fft_ops import ifft2c

                    # Check if SENSE adjoint is configured
                    _metrics_cfg = _get_config_value(self.config, "metrics", None)
                    _vis_transform = _get_config_value(_metrics_cfg, "transform", None)
                    if _vis_transform == "ifft_sense_adjoint":
                        from spectramr.infrastructure.physics.fft_ops import sense_adjoint

                        # pick_present, not ``a or b`` — see the :515 note above
                        # (tensor-truthiness RuntimeError on multi-element smaps).
                        smaps = pick_present(
                            getattr(self, "_cached_smaps", None),
                            getattr(self, "_current_smaps", None),
                        )
                        mag = torch.abs(sense_adjoint(ksp, smaps=smaps))
                        # RSS coil combination for multi-coil data
                        if mag.shape[1] > 1:
                            mag = torch.sqrt((mag**2).sum(dim=1, keepdim=True))
                        return mag

                    if torch.is_complex(ksp):
                        img = ifft2c(ksp)
                    # Handle 5D volumetric k-space [B, C, H, W, D]
                    elif ksp.dim() == 5 and ksp.shape[1] >= 2 and ksp.shape[1] % 2 == 0:
                        b, c, h, w, d = ksp.shape
                        ksp_flat = ksp.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
                        img_channels = []
                        for i in range(0, c, 2):
                            ksp_complex = torch.complex(ksp_flat[:, i], ksp_flat[:, i + 1])
                            img_complex = ifft2c(ksp_complex).unsqueeze(1)
                            img_channels.append(img_complex)
                        img = torch.cat(img_channels, dim=1)
                    # Handle 4D k-space [B, C, H, W]
                    elif ksp.dim() == 4 and ksp.shape[1] >= 2 and ksp.shape[1] % 2 == 0:
                        b, c, h, w = ksp.shape
                        img_channels = []
                        for i in range(0, c, 2):
                            ksp_complex = torch.complex(ksp[:, i], ksp[:, i + 1])
                            img_complex = ifft2c(ksp_complex).unsqueeze(1)
                            img_channels.append(img_complex)
                        img = torch.cat(img_channels, dim=1)
                    elif ksp.dim() == 4 and ksp.shape[1] == 1:
                        # Single real channel claimed to be k-space → treat as
                        # Re part with Im=0 and IFFT. See train.py
                        # ``_kspace_to_image`` for the full rationale.
                        ksp_complex = torch.complex(ksp.float(), torch.zeros_like(ksp).float())
                        img = ifft2c(ksp_complex)
                    else:
                        # Loud-fail (CLAUDE.md #9): refuse to silently render
                        # a tensor with an unsupported layout as if it were
                        # image domain. The strategy/YAML must be corrected.
                        raise ValueError(
                            "[MetricsMixin.kspace_to_image] cannot IFFT tensor "
                            f"with shape={tuple(ksp.shape)} dtype={ksp.dtype} — "
                            "neither complex, even-channel real-stacked, nor "
                            "single-channel real."
                        )
                    mag = torch.abs(img)
                    # Apply RSS coil combination → single-channel magnitude
                    # Prevents downstream to_magnitude() from misinterpreting
                    # coil magnitudes as real/imaginary pairs.
                    if mag.shape[1] > 1:
                        mag = torch.sqrt((mag**2).sum(dim=1, keepdim=True))
                    return mag
                except ValueError:
                    # Loud-fail: do not mask SSOT-domain mismatches behind a
                    # silent magnitude fallback (CLAUDE.md #9).
                    raise
                except Exception as e:
                    # F14 (2026-05-17 round 7): re-raise instead of falling
                    # back to ``to_magnitude(ksp)``. The pre-F14 fallback
                    # caught any non-ValueError exception (TypeError from a
                    # shape mismatch, RuntimeError from an OOM during IFFT,
                    # ImportError if ifft2c failed to load, …) and rendered
                    # the raw k-space magnitude as the "image" — producing
                    # the DC-spike center-bright artifact seen in 2026-05-16
                    # mosaics. Per CLAUDE.md #9: failures must surface, not
                    # be papered over with a misleading visualization.
                    import logging as _logging

                    _logging.getLogger(__name__).error(
                        f"[MetricsMixin.kspace_to_image] IFFT failed: {e}. "
                        f"Refusing to silently fall back to magnitude — that "
                        f"would render k-space as image (DC-spike artifact). "
                        f"shape={ksp.shape}, dtype={ksp.dtype}. "
                        f"See TODO/audit/smoke_audit_20260516.md §F14."
                    )
                    raise

            images_dict = {}

            # ── Domain-aware visualization ───────────────────────────────────
            # kspace_to_image uses P2 (model-type lookup) which gives the RIGHT
            # answer for predictions, but targets come from the DataLoader and
            # may still be in k-space even when the MODEL is image-domain.
            # Use needs_ifft_for_visualization to decouple the two decisions.
            try:
                from spectramr.infrastructure.training.utils.domain_inference import (
                    needs_ifft_for_visualization,
                )

                _ifft_preds, _ifft_targets = needs_ifft_for_visualization(self.config)
            except Exception:
                # Fallback: unified kspace_to_image handles both
                _ifft_preds = _ifft_targets = None

            def _to_vis(tensor, force_ifft: bool | None):
                """Convert tensor to visualizable image magnitude.

                Args:
                    tensor: Input tensor (predictions, targets, or inputs).
                    force_ifft: If True force IFFT; if False skip; if None use
                        kspace_to_image heuristic.

                .. note::

                    Audit-2026-05-14 §1 surfaced 18 experiments whose
                    validation *real* images were flagged as
                    ``centro_symmetric``. Investigation (round-3) showed
                    two distinct root causes:

                    1. ``experiment_40_zero_shot_cold_diffusion`` — the
                       saved tensor is a single-channel ``|kspace|``
                       magnitude (phase-stripped). No IFFT can recover
                       the image; the rings rendered directly are
                       diagnostically correct but visually confusing.
                       The fix lives in the dataset / strategy that
                       emits the validation target — it must hand the
                       image-domain reference here, NOT the k-space
                       magnitude. See audit doc §6 follow-up F9.
                    2. ``experiment_11_attention_*`` /
                       ``kspace_cold_diffusion`` cohort — the audit
                       heuristic over-flagged natural anatomy: dark
                       background dominates the mean-abs-diff, pushing
                       csym above 0.88 even when the image is only
                       bilaterally (left-right) symmetric. The fix
                       lives in the audit heuristic (require csym to
                       exceed both LR and TB mirror scores by a margin).

                    So this ``_to_vis`` site stays as-is: silently
                    IFFT-ing a phase-stripped single-channel tensor
                    would produce a *Hermitian-symmetric* "doubled
                    brain" output — the same failure mode it was
                    intended to fix. The defensive behaviour is to
                    fall through to ``img = t`` so the user sees the
                    raw k-space rings (diagnostic) instead of a
                    plausible-looking-but-wrong reconstruction.
                """
                if force_ifft is True:
                    # config flag says k-space (dataset_type=kspace), but a
                    # strategy may already hand an IMAGE-domain target/prediction
                    # here (e.g. exp_p7 HamiltonianAcquisition). IFFT'ing that
                    # produces a k-space DC blob in the saved real/fake PNG
                    # (smoke 2026-06-15). Build the complex candidate, then VETO
                    # the IFFT when the magnitude lacks a k-space DC signature.
                    from spectramr.infrastructure.physics.fft_ops import ifft2c
                    from spectramr.infrastructure.training.utils.domain_inference import (
                        looks_like_kspace,
                    )

                    t = tensor
                    if isinstance(t, dict):
                        t = t.get("reconstruction", t.get("output", next(iter(t.values()))))
                    if torch.is_complex(t):
                        kcand = t
                    elif t.dim() == 4 and t.shape[1] >= 2 and t.shape[1] % 2 == 0:
                        kcand = torch.stack(
                            [torch.complex(t[:, i], t[:, i + 1]) for i in range(0, t.shape[1], 2)],
                            dim=1,
                        )
                    else:
                        return to_magnitude(t)  # odd/real channels: not k-space-able

                    img = ifft2c(kcand) if looks_like_kspace(kcand.abs()) else kcand
                    mag = torch.abs(img)
                    if mag.shape[1] > 1:
                        mag = torch.sqrt((mag**2).sum(dim=1, keepdim=True))
                    return mag
                elif force_ifft is False:
                    # Fast path: image domain, just take magnitude
                    return to_magnitude(tensor)
                else:
                    # Original unified heuristic (preserves old behaviour)
                    return kspace_to_image(tensor)

            # kspace_to_image already returns magnitude — do NOT re-apply to_magnitude
            # as it would treat even-channel images as complex pairs (RSS combine)
            # Reduce a distribution head (e.g. [mean, logvar]) to its point estimate
            # first so it is not RSS-blended into sqrt(mean^2 + logvar^2) (issue #371).
            pred_img = _to_vis(self._prediction_for_visualization(predictions), _ifft_preds)
            pred_mag = _ensure_4d(pred_img)
            images_dict["val/predictions"] = pred_mag[:max_images]

            target_img = _to_vis(targets, _ifft_targets)
            target_mag = _ensure_4d(target_img)
            images_dict["val/targets"] = target_mag[:max_images]

            # ✅ SSOT: Extract log_difference_images from config schema (default: True)
            if logging_s and (logging_s.images.log_difference if logging_s else True):
                diff = _ensure_4d(torch.abs(pred_mag - target_mag))
                images_dict["val/difference"] = diff[:max_images]

            # ✅ SSOT: Extract log_input_images from config schema (default: False)
            if logging_s and (logging_s.images.log_input if logging_s else False):
                input_img = _to_vis(inputs, _ifft_targets)  # inputs same domain as targets
                input_mag = _ensure_4d(input_img)
                images_dict["val/inputs"] = input_mag[:max_images]

            self.logging_service.log_images_batch(images_dict, step, max_images=max_images)

            recorder = getattr(self, "_report_case_recorder", None)
            sink = getattr(self, "_per_case_metric_sink", None)
            if (recorder is not None and getattr(recorder, "enabled", False)) or (
                sink is not None and getattr(sink, "enabled", False)
            ):
                _input_vis = _to_vis(inputs, _ifft_targets)
                feed_report_case_recorder(
                    recorder,
                    predictions=pred_mag,
                    targets=target_mag,
                    inputs=_input_vis,
                    metrics=metrics,
                    step=step,
                    sink=sink,
                )

            # Also save images to filesystem if metrics service available
            try:
                if hasattr(self, "metrics_service") and self.metrics_service is not None:
                    epoch = getattr(self, "current_epoch", 0)
                    self.logging_service.log_info(
                        f"[MixinValidation] Saving {target_mag.shape[0]} images to disk "
                        f"(epoch={epoch}, step={step}, metrics_service={type(self.metrics_service).__name__}, "
                        f"save_images={getattr(self.metrics_service, 'save_images', 'unknown')})"
                    )
                    real_paths, fake_paths = self.metrics_service.save_images_batch(
                        real_images=target_mag,
                        fake_images=pred_mag,
                        prefix="validation",
                        epoch=epoch,
                        step=step,
                        max_images=max_images,
                    )
                    if real_paths or fake_paths:
                        self.logging_service.log_info(
                            f"[MixinValidation] Saved {len(real_paths)} real + {len(fake_paths)} fake images"
                        )
                    else:
                        # No paths returned likely means save_images=False in config (intended behavior)
                        pass
                else:
                    # ✅ FAIL LOUD if metrics_service is expected but not available
                    missing_attr = not hasattr(self, "metrics_service")
                    if missing_attr:
                        self.logging_service.log_error(
                            "[MixinValidation] CRITICAL: metrics_service attribute missing on strategy. "
                            "This indicates a training infrastructure issue - metrics_service must be passed from bootstrap. "
                            "Validation images will NOT be saved."
                        )
                    else:
                        self.logging_service.log_error(
                            "[MixinValidation] CRITICAL: metrics_service is None on strategy. "
                            "DI resolution failed or bootstrap did not register IMetricsService. "
                            "Validation images will NOT be saved."
                        )
            except Exception as e:
                self.logging_service.log_warning(
                    f"[MixinValidation] Failed to save validation images: {e}"
                )

            self.validation_step_count = getattr(self, "validation_step_count", 0) + 1

        except Exception as e:
            if hasattr(self, "logging_service"):
                self.logging_service.log_warning(f"Failed to log validation images: {e}")


def universal_validation_keys(strategy_cls: type) -> frozenset[str]:
    """The undeclared keys ``strategy_cls``'s validation path writes (#2838).

    ``MetricsMixin.validation_step`` returns only the registry computer's keys,
    so a strategy that resolves to it earns no ``val_loss`` exemption. Writing
    ``val_loss`` there instead would need a forward pass and an image-shaped
    output, which that path deliberately does not assume (#2839). An override
    is trusted to write ``val_loss``, as ``ModelValidationMixin`` and the
    diffusion/standard strategies do; one that only extends this default
    without writing it is not detected here.
    """
    from spectramr.core.metrics.validation_names import UNIVERSAL_VALIDATION_KEYS

    if getattr(strategy_cls, "validation_step", None) is MetricsMixin.validation_step:
        return frozenset()
    return UNIVERSAL_VALIDATION_KEYS
