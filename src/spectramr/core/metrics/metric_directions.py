"""Single source of truth for metric *optimization direction*.

Every metric in the registry must expose a boolean ``higher_is_better``
attribute — it is part of the :class:`~spectramr.core.metrics.registry.IMetric`
protocol and is consumed by the reporting layer (training-curve direction,
results tables, ablation strips) and by the meta-evaluation ranker
(``metric_adapter`` previously defaulted a *missing* attribute to ``False``,
which silently ranked PSNR/SSIM as "lower is better").

Two ways a metric declares its direction:

1. **Self-declared** — the metric class sets ``higher_is_better`` itself
   (class attribute or property). ~22 metrics do this today. These own
   their value and are NOT listed here (a direction is defined exactly
   once, never duplicated).
2. **Centrally declared** — for the large family of metrics that subclass
   ``BaseMetric`` (which historically defined no direction), the value
   lives in :data:`METRIC_HIGHER_IS_BETTER` below and is injected onto the
   class by the ``@register_metric`` decorator at registration time.

This map lives in ``core/`` (the architectural home for metrics) because
``core/`` may not import ``scripts/``, where the sim2rank ``METRIC_SPECS``
list independently annotates direction. ``METRIC_SPECS`` is a downstream
consumer; ``tests/unit/core/metrics/test_metric_directions.py`` asserts the
two never disagree, so this map remains the single source of truth.

Adding a new metric: either set ``higher_is_better`` on the class, or add an
entry here. The registry gate tests
(``tests/contracts/test_metric_registry.py``,
``tests/unit/core/metrics/test_metric_registry_health.py``) fail loudly if a
registered metric ends up with neither.
"""

from __future__ import annotations

# name -> True if a larger value indicates better reconstruction quality.
# Sourced from scripts/sim2rank/metrics_list.py::METRIC_SPECS (the authoritative
# annotated list); zipper_detection has no spec entry (artifact-detection metric,
# higher == more zipper artifact == worse).
METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    "att_mae": False,
    "auprc": True,
    "auroc": True,
    "average_surface_distance": False,
    "bat": False,
    "bland_altman_bias": False,
    "blockiness": False,
    "brenner_focus": True,
    "cbf_rmse": False,
    "cc_snr": True,
    "cjv": False,
    # clinical_ssim subclasses SSIMMetric and inherits its injected direction
    # (True); listing it here would define the direction twice.
    "cnr": True,
    "coefficient_of_variation": False,
    "cohen_kappa": True,
    "complex_hfen": False,
    "cosine_similarity": True,
    "crlb": False,
    "cw_ssim": True,
    "detection_sensitivity": True,
    "detection_specificity": True,
    "dice": True,
    "dists": False,
    "divergence": False,
    "dss": True,
    "dvf_mae": False,
    "edge_preservation_index": True,
    "efc": False,
    "empirical_coverage": True,  # ensemble interval mean +/- k*std covers the target
    "expected_calibration_error": False,
    "fber": True,
    "fid": False,
    "focal_frequency": False,
    "folding_fraction": False,
    "frd": False,
    "freq_domain_snr": True,
    "fsim": True,
    "ghosting_ratio": False,
    "gibbs_ringing": False,
    "gmsd": False,
    "gradient_entropy": False,
    "gradient_error": False,
    "haarpsi": True,
    "hd95": False,
    "high_freq_energy_ratio": True,
    "icc_3_1": True,
    "immerkaer_noise": False,
    "inception_score": True,
    "intensity_entropy": True,
    "iou": True,
    "ipen": False,
    "iw_ssim": True,
    "kid": False,
    "kspace_error": False,
    "ktrans": True,
    "limits_of_agreement_lower": True,
    "limits_of_agreement_upper": False,
    "lpips": False,
    "lpips_alex": False,
    "mad": False,
    "mae": False,
    "mass_conservation": True,
    "mdsi": False,
    "med_fid": False,
    "mlv": True,
    "mmd_metric": False,
    "ms_gmsd": False,
    "ms_ssim": True,
    "mse": False,
    "mutual_information": True,
    "ndc_diffusion": True,
    "negative_voxels": False,
    "net_flow_error": False,
    "nll_bits_per_dim": False,
    "nmse": False,
    "normalized_gradient_squared": True,
    "nr_iqa": True,
    "nrmse": False,
    "nrmse_l2": False,
    "pdm": False,
    "peak_velocity_error": False,
    "pearson": True,
    "phase_mse": False,
    "power_spectrum_consistency": False,
    "psnr": True,
    "qi1": False,
    "radial_k_error": False,
    "residual_whiteness": True,
    "rfs": False,
    "rmse": False,
    "robust_mri_psnr": True,
    "sliced_wasserstein": False,
    "snr": True,
    "spectral_linewidth": False,
    "spike_detection": False,
    "spike_percentage": False,
    "srsim": True,
    "ssim": True,
    "st_mad": False,
    "target_registration_error": False,
    "through_plane_fwhm": False,
    "total_variation": False,
    "uqi": True,
    "velocity_rmse": False,
    "vif": True,
    "vif_p": True,
    "vnr": True,
    "volume_consistency": True,
    "vsi": True,
    "wash_slope": True,
    "wasserstein_1d": False,
    "wm2max": True,
    "zipper_detection": False,
}


# ---------------------------------------------------------------------------
# Non-registry monitor keys
# ---------------------------------------------------------------------------
# Strategy-emitted scalars that are legal ``best_metric_name`` /
# ``early_stopping.metric`` targets but are NOT registered metrics (they are
# produced inside a ``validation_step``, not by an ``@register_metric`` class).
# They have no class to carry ``higher_is_better``, so their direction is
# DECLARED here rather than guessed from the name.
#
# Why this table exists at all: the previous resolver ended in
# ``return not ("loss" in kl or "error" in kl)`` — a substring guess. It got
# ``val_chamfer``, ``val_t1_mape``, ``phys_eq_residual`` and
# ``rdsm_quadratic_residual`` WRONG (all are lower-is-better, all were guessed
# "higher"), which silently inverts best-checkpoint selection. A guess is a
# silent fallback (pitfall #9); a declaration is not.
NON_REGISTRY_METRIC_DIRECTIONS: dict[str, bool] = {
    # aggregate objectives — lower is better
    "loss": False,
    "val_combined": False,
    "val_g_total_loss": False,
    "score_field_loss_x": False,
    "crlb_objective": False,
    # physics / geometry residuals — lower is better
    "phys_eq_residual": False,
    "rdsm_quadratic_residual": False,
    # `val_field_bias` is DELIBERATELY absent. It is the SIGNED offset
    # `mean(pred_field - field)`, and it used to be declared lower-is-better with
    # the comment "|bias| → 0 is the goal" — a declaration that contradicted the
    # quantity actually emitted, so minimising it selected the MOST NEGATIVE field
    # estimate on nine arms (#230). "bias" conventionally means signed, so the
    # emitter is right and the declaration was wrong; the fix is that no direction
    # exists for it. `metric_higher_is_better` therefore RAISES on it, which is the
    # intended outcome: a signed bias is a diagnostic, not a selection target.
    # Monitor `val_field_abs_bias` instead.
    "val_field_abs_bias": False,  # |mean(pred - field)| → 0 is the goal
    "val_chamfer": False,  # Chamfer distance
    # The guarded selection composite (strategies/selection_rules.py): a penalised
    # ring-NRMSE ratio, lower is better. The guard flag is 1.0 when a guard trips.
    "guarded_ring_fidelity_mean": False,
    "guarded_ring_fidelity_mean_partial": False,
    "selection_guard_tripped": False,
    "val_t1_mape": False,  # mean absolute percentage error
    # conformal coverage / certificates — higher is better (coverage → 1 - alpha)
    "coverage_at_alpha": True,
    "empirical_coverage_alpha_01": True,
    # ``empirical_coverage`` is a registered metric since 2026-09
    # (``quantitative/conformal_risk.py``); its direction lives in
    # METRIC_HIGHER_IS_BETTER above, where the registry injects it.
    # Monitor keys emitted by specific strategies. Each direction below is taken
    # from the emitting code or the arm's own declared mode, never inferred from
    # the name:
    #   kspace_consistency_tta — spectra_tta_strategy.py:187 emits the
    #     KspaceConsistencyTTALoss data-consistency term, i.e. a loss.
    #   entropy — exp_tta_sotadit.yaml declares `mode: min` alongside it
    #     (test-time entropy MINIMIZATION).
    #   no_crash_lowering — a cosim wiring smoke: 1 = the GEMM lowered onto the
    #     SPECTRA array without crashing. Higher is better.
    "val_kspace_consistency_tta": False,
    "kspace_consistency_tta": False,
    # Disentanglement readings of DisentangledTrainingStrategy.finalize_validation
    # (core/metrics/disentanglement.py): a style code should predict the contrast, a content
    # code should not (leak = accuracy - chance), and a swap should return the content and
    # the style it was given (relative error 0). `val_style_contrast_chance` and
    # `val_probe_split_by_subject` are references, not quantities to optimise: undeclared.
    "val_style_contrast_acc": True,
    "val_content_contrast_leak": False,
    "val_swap_content_distance": False,
    "val_swap_style_distance": False,
    "entropy": False,
    "perplexity": False,
    "hallucination_rate": False,
    # trajectory-monitor summary keys (core/metrics/trajectory_metrics.py):
    # per-step data-consistency residuals and admissible-tube excursions are
    # all discrepancies — lower is better.
    "trajectory_kappa_final": False,
    "trajectory_kappa_max": False,
    "trajectory_excursion": False,
    "trajectory_max_violation_ratio": False,
    "hallucination_severity": False,  # CVaR of the excursion tail
    "dkw_slack": False,  # DKW inequality slack — a tighter (smaller) bound is better
    "pac_bayes_bound": False,  # a tighter bound is better
    "mmd_paired_distribution_match": False,  # MMD is a discrepancy
    "per_pair_consistency_residual": False,
    "nll": False,
    "certified_recall_lower_bound": True,
    "no_crash_lowering": True,
    # ---------------------------------------------------------------
    # Names carried over from the legacy ``types.DEFAULT_METRIC_DIRECTIONS``
    # table. None of these is a registered metric (no @register_metric class),
    # so they have no other home. Values are preserved verbatim from the legacy
    # table so this consolidation is behaviour-preserving for them.
    # ---------------------------------------------------------------
    "correlation": True,
    "ndc": True,
    "volume_similarity": True,
    "sfnr": True,
    "tsnr": True,  # temporal SNR
    "pe_cross_corr": True,
    # NOTE: legacy declared "blur" as higher-is-better. That looks wrong (more
    # blur is worse), but it is not a registered metric and no experiment YAML
    # monitors it, so the value is preserved rather than silently flipped.
    # Tracked in the audit notes; flip it in a dedicated change with a test.
    "blur": True,
    "l1": False,
    "fwhm": False,
    "dvars": False,
    "gcor": False,
    "gsr": False,
    "spike_percent": False,
    "neg_voxels": False,
    "rase": False,
    "sam": False,
    "fd": False,
    "aor": False,
    "piesno": False,
    "medicalnet_distance": False,
}

# Prefixes stripped when a monitor key wraps a metric name (``val_psnr`` →
# ``psnr``). Ordered longest-first so ``val_`` never shadows a longer prefix.
_MONITOR_PREFIXES: tuple[str, ...] = ("val_", "train_", "test_", "best_", "eval_")


class UnknownMetricDirectionError(KeyError):
    """Raised when a monitor key's optimization direction cannot be resolved.

    Deliberately fatal. Defaulting an unknown key to "higher is better" is what
    made ``best_metric_name: lpips`` retain the *worst* checkpoint — the failure
    is invisible because the run completes normally and only the saved weights
    are wrong.
    """


def _registered_direction(name: str) -> bool | None:
    """Read ``higher_is_better`` off the registered metric class, or None.

    The registry is the true SSOT: ``@register_metric`` already injects the
    direction onto every class (from :data:`METRIC_HIGHER_IS_BETTER` or the
    decorator's ``direction=`` kwarg), and ~95 metrics *self-declare* it and are
    therefore absent from the static map above. A map-only lookup cannot see
    them — that is exactly how ``hfen`` (lower-is-better, self-declared) came to
    resolve as higher-is-better despite this module's own docstring citing it as
    a metric that could never do so.

    ``higher_is_better`` may be a plain bool class attribute or a ``property``;
    the latter needs an instance to evaluate.
    """
    # Local import: registry imports this module, so a module-level import here
    # would be circular.
    from spectramr.core.metrics.registry import MetricsRegistry

    metric_cls = MetricsRegistry._metrics.get(name)
    if metric_cls is None:
        return None
    for base in metric_cls.__mro__:
        declared = base.__dict__.get("higher_is_better")
        if declared is None:
            continue
        if isinstance(declared, bool):
            return declared
        # A property/descriptor — instantiate to read it.
        try:
            return bool(MetricsRegistry.get(name).higher_is_better)
        except Exception:  # an un-instantiable metric is not a direction source
            return None
    return None


def _direct_lookup(key: str) -> bool | None:
    """Exact-match the key against every declared direction source."""
    if key in METRIC_HIGHER_IS_BETTER:
        return METRIC_HIGHER_IS_BETTER[key]
    if key in NON_REGISTRY_METRIC_DIRECTIONS:
        return NON_REGISTRY_METRIC_DIRECTIONS[key]
    return _registered_direction(key)


def resolve_direction(key: str) -> bool | None:
    """Resolve a monitor key's direction, or ``None`` if it is undeclared.

    The non-fatal sibling of :func:`metric_higher_is_better`, for callers that
    sweep *arbitrary* columns rather than making a decision from a named metric —
    e.g. ``MetricsTracker._update_running_best``, which folds every numeric key in
    a training record (``epoch``, ``lr``, ``grad_norm``, whatever a strategy
    happened to log).

    Such a caller must **skip** what it cannot resolve, not guess: a "best
    ``grad_norm``" recorded under an assumed direction is meaningless, and making
    it fatal would crash training on the first unrecognised log column. Decision
    paths (checkpoint retention, early stopping, leaderboard ranking) use the
    strict function instead — there, an assumed direction silently keeps the
    wrong weights.
    """
    try:
        return metric_higher_is_better(key)
    except UnknownMetricDirectionError:
        return None


def metric_higher_is_better(key: str) -> bool:
    """Resolve a metric's optimization direction. **Raises** if it cannot.

    Resolution order (each step is a *declaration*, never a guess):

    1. exact match against the registry / :data:`METRIC_HIGHER_IS_BETTER` /
       :data:`NON_REGISTRY_METRIC_DIRECTIONS`;
    2. the same, after stripping a monitor prefix (``val_`` / ``train_`` / …),
       so ``val_lpips`` resolves through ``lpips``;
    3. the longest known metric name appearing as a substring of the key, so
       suffixed columns (``psnr_mean``, ``val_psnr_2x``) still resolve;
    4. otherwise :class:`UnknownMetricDirectionError`.

    Step 4 used to be ``return not ("loss" in key or "error" in key)``. That
    guess is why ``best_metric_name: lpips`` maximized LPIPS. An unresolvable
    monitor key is a config error, not a coin flip — declare it in
    :data:`NON_REGISTRY_METRIC_DIRECTIONS` or set ``direction:`` explicitly on
    the metric spec.

    This is the single resolver shared by the metrics tracker, early stopping
    and the campaign leaderboard ranker.
    """
    kl = key.lower()

    resolved = _direct_lookup(kl)
    if resolved is not None:
        return resolved

    for prefix in _MONITOR_PREFIXES:
        if kl.startswith(prefix):
            resolved = _direct_lookup(kl[len(prefix) :])
            if resolved is not None:
                return resolved
            break

    # A column named ``*_loss`` is an OBJECTIVE, and an objective is minimized --
    # whatever metric its stem happens to name. This has to sit ahead of the
    # token-run fallback below, because that fallback resolves ``ssim_loss`` by
    # matching the run ``ssim`` and returns the METRIC's direction: higher. The
    # measured consequence was ``final_metrics.json`` reporting the *maximum*
    # ``ssim_loss`` as that column's best. Confirmed on ``ssim_loss``,
    # ``ms_ssim_loss``, ``dice_loss`` and ``psnr_loss`` -- every loss named after a
    # higher-is-better metric. ``val_loss`` / ``g_total_loss`` were never affected,
    # which is why this survived: the common names resolve correctly by accident.
    #
    # Deliberately AFTER both exact lookups, so a metric genuinely registered under
    # a ``*_loss`` name with a declared direction still wins. This reads the name's
    # own declared semantics; it is not the "not ('loss' in key)" guess step 4
    # documents, which applied to every unresolvable key rather than to a suffix
    # the emitter chose.
    if kl == "loss" or kl.endswith("_loss"):
        return False

    # Token-run fallback, so a decorated column (``psnr_mean``, ``val_psnr_2x``,
    # ``val_robust_mri_psnr``) still resolves.
    #
    # Matching is on contiguous runs of WHOLE underscore-delimited tokens, never
    # on raw character substrings. A raw-substring match silently finds ``mad``
    # inside ``made`` (and ``sam`` inside ``sample``, ``fd`` inside ``fdr``), so
    # a nonsense key like ``totally_made_up_metric`` would resolve to the
    # direction of MAD instead of raising. Longest run wins, so ``psnr`` beats
    # ``snr`` and ``robust_mri_psnr`` beats ``psnr``.
    tokens = kl.split("_")
    runs = [
        "_".join(tokens[i:j]) for i in range(len(tokens)) for j in range(i + 1, len(tokens) + 1)
    ]
    for run in sorted(runs, key=len, reverse=True):
        resolved = _direct_lookup(run)
        if resolved is not None:
            return resolved

    raise UnknownMetricDirectionError(
        f"Cannot resolve an optimization direction for monitor key {key!r}. "
        "Refusing to assume 'higher is better' — that silently retains the WORST "
        "checkpoint for any lower-is-better metric. Fix by either: (a) registering "
        "the metric with @register_metric(..., direction='higher'|'lower'); "
        "(b) adding it to NON_REGISTRY_METRIC_DIRECTIONS in "
        "spectramr/core/metrics/metric_directions.py; or (c) declaring "
        "`direction: min|max` on the metric spec in the experiment YAML."
    )


__all__ = [
    "METRIC_HIGHER_IS_BETTER",
    "NON_REGISTRY_METRIC_DIRECTIONS",
    "UnknownMetricDirectionError",
    "metric_higher_is_better",
    "resolve_direction",
]
