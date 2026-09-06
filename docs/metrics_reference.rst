.. _metrics_reference:

========================================
Metrics Registry — Mathematical Reference
========================================

.. sectionauthor:: spectraMR Research

Metrics are registered with the ``@register_metric`` decorator and resolved
through the ``MetricsRegistry`` singleton. The

.. note::

   **This page is hand-written and not exhaustive.** It documents 115 of the
   211 metrics the registry holds, so a name's absence here is **not** evidence
   that it does not exist -- check the registry before concluding one is
   unavailable. Counts are deliberately not restated in the prose above: a frozen
   number in a hand-maintained page drifts silently, and this one had -- it read "100+".

   .. code-block:: python

      from spectramr.core.metrics.registry import MetricsRegistry

      MetricsRegistry.list_available()
      len(MetricsRegistry.list_available())

   Tracked as issue #1643 -- these pages should be generated from the
   registries, the way ``docs/config_key_reference.rst`` already is.
sim2rank sweep selects **121** of them (see
:ref:`sim2rank-zero-metric-review`). On 2026-05-25 three registry metrics
whose input contract is not a 2-D image pair were dropped from the sweep
because they returned NaN on *every* 2-D evaluation: ``through_plane_fwhm``
(needs a 3-D volume), ``persistence_diameter`` (consumes an MRF trajectory),
and ``topological_mask_certificate`` (consumes a sparse k-space mask). All
three remain registered for their native input contracts.
All metrics implement ``__call__(prediction, target, **kwargs) → float``,
plus ``name`` and ``higher_is_better`` properties.

.. contents:: Table of Contents
   :depth: 2
   :local:

Derived validation keys: the zero-filled baseline
================================================

Not every ``val_*`` key names a registered metric. The diffusion validation
path emits three families per rung:

``val_<k>``
   the model's score on registered metric ``k``.

``val_zf_<k>``
   the same registered metric, computed on the **zero-filled baseline** --
   the k-space the model was actually handed (``masked_input``), decompressed
   and denormalised through the same seam as the prediction, then put through
   the same ``validation.scoring.output_transform``. Restricted to ``psnr``
   and ``hfen``, intersected with the metrics the arm already grades on: a
   baseline for a metric the arm does not compute has nothing to be compared
   against, and a second pass over a full metric set would double every
   perceptual metric per rung per batch.

``val_zf_delta_<k>``
   the signed ``val_<k> - val_zf_<k>``. It is deliberately **not** oriented to
   "positive is better": ``metric_higher_is_better`` resolves an unknown key
   through the longest registered metric name inside it, so
   ``val_zf_delta_hfen`` resolves via ``hfen`` to *lower is better* -- which is
   correct for a raw difference and backwards for a gain.

The baseline is derived from the *input*, never from the target. Validation
masks the noisy single repetition the dataloader delivered, so a
target-derived zero-filled reconstruction would be a noise-free idealisation
no reconstructor can reach -- it would read higher than the model on a healthy
run, which is the opposite of what a baseline is for.

Arms that record no measurement -- every non-cold-diffusion arm -- emit none of
these keys, on every batch and every rank. Absence is uniform by construction,
because the distributed all-reduce packs ``sorted(keys)`` positionally.

Strict-duplicate registration (post-2026-05-09 audit)
=====================================================

Per CLAUDE.md rule #9 (silent fallbacks forbidden), ``MetricsRegistry``
refuses to silently overwrite a registration:

* Re-registering the **same** class under the **same** name is idempotent.
* Registering a **different** class under the **same** canonical name
  raises :py:class:`ValueError` immediately at import time.
* Re-binding an alias to a **different** canonical name also raises.

The pre-fix behaviour was a "warn-then-overwrite" branch that depended
on import order: e.g. ``hfen`` was sometimes resolved to
``evaluation_metrics.HFEN`` (5D-aware L2) and sometimes to
``hfen.HFENMetric`` (4D L1) within the same training run. The strict
guard makes that class of bug impossible.

Two registered metrics regained visibility at the same audit pass:
``feature_fidelity_index`` and ``fabrication_rate`` — their host module
``hallucination_metrics`` was missing from
``src/spectramr/core/metrics/__init__.py`` so the decorators never fired.

See ``TODO/audit/00_implementation_tracker.md`` and the regression tests
under ``tests/unit/models/test_registry_strict_duplicates.py``.

.. _metric-direction-ssot:

Optimization direction (``higher_is_better``) — single source of truth
======================================================================

Every registered metric must expose a boolean ``higher_is_better``. It is
part of the :class:`IMetric` protocol and is consumed by the reporting
layer (training-curve direction, results tables, ablation strips) and by
the meta-evaluation ranker. A *missing* attribute used to default to
``False`` in ``metric_adapter``, which silently ranked PSNR/SSIM as
"lower is better" — exactly the kind of silent fallback CLAUDE.md rule #9
forbids.

A metric declares its direction in **exactly one** of two places:

* **Self-declared** — the class sets ``higher_is_better`` itself (class
  attribute or property). ~22 metrics do this.
* **Centrally declared** — for the large family that subclass
  ``BaseMetric`` (which defines no direction), the value lives in
  :data:`spectramr.core.metrics.metric_directions.METRIC_HIGHER_IS_BETTER`
  and is injected onto the class by the ``@register_metric`` decorator at
  registration time.

The map lives in ``core/`` because ``core/`` may not import ``scripts/``,
where the sim2rank ``METRIC_SPECS`` list independently annotates direction.
``METRIC_SPECS`` is a downstream consumer;
``tests/unit/core/metrics/test_metric_directions.py`` asserts the two never
disagree, so the core map remains authoritative.

The registry gate tests
(``tests/contracts/test_metric_registry.py`` and
``tests/unit/core/metrics/test_metric_registry_health.py``) fail loudly if a
registered metric ends up with neither a self-declared nor a mapped
direction.

.. note::

   ``auroc`` and ``cohen_kappa`` are ``DOMAIN_SPECIFIC`` (they need discrete
   label inputs); on a continuous ``(pred, target)`` image pair the target
   collapses to a single class and the score is undefined. They are excluded
   from the per-image reconstruction sweep and from the reflexive
   finite-scalar contract check.

Perceptual-metric input finiteness guard
========================================

Perceptual metrics (:class:`~spectramr.core.metrics.evaluation_metrics.LPIPS`,
:class:`~spectramr.core.metrics.evaluation_metrics.FID`) range-normalise their
inputs and expand them to three channels before a frozen VGG / Inception
backbone. The historical normalisers used a silent ``torch.clamp``, which
leaves ``NaN`` / ``Inf`` **unchanged** — so a model that diverged and emitted
non-finite predictions had its ``NaN`` forwarded into torchmetrics, where it
surfaced as a misleading *"values in range [nan, nan] … expected [-1, 1]"*
range error (the 2026-06-24 ``exp_hm_06`` / ``exp_hm_10`` Hyper-Mamba crash).

The single source of truth for the *finiteness* half of the perceptual-metric
input contract is
:func:`spectramr.core.metrics.metric_input_prep.assert_finite_metric_input`.
``LPIPS`` and ``FID`` call it at the metric boundary, **before** any shape
handling or feature accumulation, and it raises a precise, actionable error
that names the metric, the offending field (``preds`` / ``target``), and the
``NaN`` / ``Inf`` counts, attributing the failure to model instability rather
than to the metric:

.. code-block:: text

   ValueError: Metric 'lpips' received non-finite 'preds' (12 NaN, 0 Inf of
   65536 elements). The model produced unstable (NaN/Inf) output; fix model /
   training stability rather than the metric — silently normalising NaN would
   hide the divergence (CLAUDE.md pitfall #9).

This keeps a diverged run failing loudly at the right layer (pitfall #9, no
silent fallbacks; pitfall #10, a non-finite metric must not pass). Regression
tests: ``tests/unit/core/metrics/test_metric_input_prep.py``,
``test_lpips_finite_guard.py``, ``test_fid_finite_guard.py``.

Perceptual-metric backend availability (2026-07-07)
===================================================

torchmetrics is genuinely optional (undeclared conflicts: it requires
``huggingface-hub<1.0``). When it cannot be imported, the torchmetrics-backed
metrics do **not** fabricate a ``0.0`` — they record ``NaN`` ("not computed").
Two refinements make that failure mode safe:

* **LPIPS has a fallback.** :class:`~spectramr.core.metrics.evaluation_metrics.LPIPS`
  falls back to the ``lpips`` package (AlexNet, the reference impl already used by
  the ``lpips_alex`` challenge metric) when torchmetrics is unavailable, so
  ``val_lpips`` is a real number rather than a run-long ``NaN``. A one-time warning
  names the backend swap (AlexNet values are *not* comparable to the torchmetrics
  VGG backbone). ``ms_ssim`` / ``uqi`` / ``kid`` / ``fid`` have no fallback and still
  raise → ``NaN``.
* **Fail-fast at config-health.**
  :meth:`~spectramr.infrastructure.validation.config_health_checker.ConfigHealthChecker.check_metric_backend_available`
  raises (severity ``error``) at pre-flight audit if a requested ``validation.metrics``
  entry has no importable backend and no fallback — a broken env fails at startup, not
  after wasting a 150k-iter run. The ``static_normalize`` step also clamps ``x*2-1`` to
  ``[-1,1]`` so an un-normalised model output cannot NaN the backbone once the env is
  restored. Regression tests: ``test_torchmetrics_fallback_raises.py``,
  ``test_config_health_checker.py`` (``TestMetricBackendAvailable``),
  ``test_computer.py`` (``test_repeated_metric_failure_warns_once``).

The durable resolution on the cluster is an env re-sync:
``pip install -e ".[dev]"``.

.. note::

   This specific conflict is **resolved**. It was caused by ``torchmetrics``
   failing to import under ``huggingface-hub>=1.0``, and the standing advice
   here used to be to pin ``huggingface-hub<1.0``. Re-measured 2026-08-29:
   ``torchmetrics 1.9.0`` imports cleanly against ``huggingface-hub 1.27.0``,
   so that pin would now *downgrade* a working environment. The failure mode
   the section describes — a dependency that is installed and
   version-satisfied yet raises on import — is real and still worth
   understanding; only the pin is obsolete. ``verify_dependencies.py
   --import-check`` is what detects it.

Usage
=====

.. code-block:: python

   from spectramr.core.metrics.registry import get_metric, compute_metric

   psnr = get_metric("psnr")
   score = psnr(prediction, target)

   # Or one-shot
   score = compute_metric("ssim", prediction, target)

.. code-block:: yaml

   # In experiment YAML
   metrics:
     compute_psnr: true
     compute_ssim: true
     best_metric_name: val_robust_mri_psnr
     best_metric_mode: max

.. _flag-coverage-ssot:

How a ``compute_*`` flag reaches a CSV column
=============================================

Two things have to agree for a flag to produce a number: something must *select*
the metric for computation, and something must *declare a column* for it in
``losses.csv``. Until 2026-08-04 those were two hand-written dictionaries —
43 entries in ``MetricsMixin._extract_metrics_from_config`` and 78 in
``pipelines.training_loop._CSV_METRIC_NAME_MAP`` — and the gap between them was
not a policy. **22 flags sat in the header map alone while naming a metric that
is registered and computable**, so enabling one produced a column header with
nothing ever written under it (#340).

That failure mode is worse than a missing column, and worth naming: an empty
column *under a header* reads as "we measured it and it came back blank", not
"we never selected it". It is pitfall #16 at the artifact layer.

Both maps now derive from a single function:

.. code-block:: python

   from spectramr.core.metrics.flag_map import schema_flag_to_metric

   schema_flag_to_metric()   # {flag: metric_name} for every metric-selecting flag

Coverage comes from ``MetricsConfigSchema.model_fields`` rather than a literal, so
a flag added to the schema is reachable by construction. Names still resolve
through :func:`~spectramr.core.metrics.flag_map.metric_for_flag` (identity plus a
one-entry alias table). The per-batch builder keeps its own narrower
``BUILDER_METRIC_FLAGS`` policy — it deliberately excludes offline / distribution
metrics such as FID — but it takes its *names* from the same resolver.

.. rubric:: Flags that select nothing, on purpose

``flag_map.NON_METRIC_FLAGS`` holds ``compute_*`` booleans that are not metric
selectors at all. Today that is ``compute_advanced_metrics``, a legacy master
switch. The distinction is load-bearing rather than cosmetic: it **defaults
True**, so if it were treated as an ordinary flag whose name happens to be
unregistered, the mixin's dangling-flag warning would fire on every arm in the
corpus — and warnings exit 2 under ``audit --strict`` (non-negotiable #4).

The other 16 dangling flags (``compute_blur``, ``compute_dvars``, ``compute_fd``,
``compute_gcor``, ``compute_pe_cross_corr``, ``compute_precision_recall``, …) all
default False and are left **visible**: they name a metric that is simply not
registered yet, which is the open half of #340. Silencing them would hide the
work rather than do it.

.. warning::

   Reachability is ``MetricsRegistry.is_registered``, which consults the
   296-entry **alias** table. Counting against ``MetricsRegistry._metrics``
   alone — as the census snippet in ``CLAUDE.md`` does — reports 4 more dangling
   flags than really are: ``compute_fwhm``, ``compute_gsr``, ``compute_ndc`` and
   ``compute_volume_similarity`` each name an alias of a canonical metric and are
   perfectly selectable. The census measures *canonical* coverage, which is what
   the drain-to-``metrics.compute`` migration tracks; it is not a reachability
   figure and should not be quoted as one.

.. _metric-reachability:

What "reachable from a config" actually requires
================================================

Two independent things, and they fail separately:

1. **The name survives validation.** ``MetricsMixin._extract_metrics_from_config``
   raises on any ``metrics.compute`` entry that is not
   ``MetricsRegistry.is_registered`` (#173), so a typo fails at strategy
   construction rather than becoming a missing column.
2. **The metric constructs.** ``ValidationMetricsComputer`` then calls
   ``MetricsRegistry.get(name, device=self.device)``. That is the *only* way a
   metric is built from config, and it supplies exactly one kwarg.

Step 2 is where reachability was quietly broken for eight metrics.

.. rubric:: The ``nn.Module`` signature trap

``torch.nn.Module.__init__`` is declared ``(*args, **kwargs)`` but raises
``TypeError: ... got an unexpected keyword argument`` for **any** kwarg, unless
the subclass sets ``call_super_init`` (default ``False``). It is a signature that
lies, kept that way for backward compatibility.

``MetricsRegistry.get`` filters kwargs to the constructor's signature so that a
generic ``get(name, device=...)`` does not crash metrics whose ``__init__`` never
declared ``device``. It read the advertised signature, saw ``**kwargs``, and
forwarded everything — correct for a class that defines its own ``__init__``, and
exactly wrong for one that inherits ``nn.Module``'s.

The consequence was not a degraded score but a dead run: ``velocity_rmse``,
``peak_velocity_error``, ``net_flow_error``, ``vnr``, ``cbf_rmse``, ``att_mae``,
``negative_voxels`` and ``ndc_diffusion`` were registered, workflow-tagged and
selectable, and **crashed in validation** for any arm that asked for them. This
is the mechanism behind #340's observation that a flow or perfusion arm "cannot
be graded on its own physics".

The filter now resolves which class in the MRO actually *owns* ``__init__``:

.. code-block:: python

   init_owner = next((c for c in metric_cls.__mro__ if "__init__" in c.__dict__), object)
   if init_owner is object or init_owner is torch.nn.Module:
       accepted = {}          # inherited-only __init__ takes nothing

Ownership is ground truth; the signature is hearsay. ``device`` is still passed
to every constructor that genuinely declares it.

.. rubric:: What remains unreachable, and why

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Metric
     - Class
     - Reason
   * - ``super_nyquist_fidelity``
     - needs ctor args
     - Requires ``sr_scale`` (or ``voxel_mm`` + ``effective_voxel_mm``) to have a
       passband at all, and refuses to guess. ``MetricSpec`` carries no kwargs
       dict, so there is nowhere in YAML to supply one — usable only from
       sim2rank, which constructs it directly.
   * - ``frd``, ``rfs``
     - optional extra
     - Need a radiomics backend (``pyradiomics``). Reachable wherever the extra
       is installed; absence is environmental, not a registry defect.

``tests/unit/config/test_metrics_schema_coverage.py::TestEveryRegisteredMetricIsReachable``
gates both halves and is green, so it can finally ratchet — the next metric added
with a broken constructor turns it red. Its predecessor,
``test_no_new_registry_orphans``, asserted that every metric needs a ``compute_*``
flag; that property is **backwards** under the drain-to-``metrics.compute``
migration, and it was red on clean ``dev``, so it could never have signalled
anything (#343).

---

.. _sim2rank-zero-metric-review:

Sim2Rank zero-metric review (2026-05-24)
========================================

In the ``sim2rank_20260524_132408`` per-axis table
(``metric_evaluations_per_axis.csv``) **15 metrics scored identically zero on
every degradation axis**. The per-axis value is a metric's *discriminative
power* — how monotonically it responds to increasing degradation — so a metric
that is zero everywhere contributes nothing to the ranking. Diagnosis grouped
the 15 into three families, each defined by a *physical dimension that a single
real magnitude image cannot carry*:

.. list-table:: Zero-metric families and root causes
   :header-rows: 1
   :widths: 18 30 52

   * - Family
     - Metrics
     - Root cause
   * - Distributional
     - ``fid``, ``kid``, ``inception_score``, ``mmd_metric``,
       ``sliced_wasserstein``, ``wasserstein_1d``,
       ``kernelised_stein_discrepancy``
     - Need a *population* of samples, not one (degraded, clean) pair. The
       per-axis loop only ran the per-image sweep, so these ``SUMMARY`` metrics
       were never fed a population and defaulted to ``0``. A second, latent bug:
       the engine called every distributional metric with the torchmetrics
       ``update()/compute()`` accumulator pattern, but the direct comparators
       (sliced-Wasserstein, W1, MMD, KSD) return their distance from
       ``__call__(preds, target)`` and yielded ``None`` → NaN → 0.
   * - Phase
     - ``ipen``, ``phase_mse``
     - A magnitude image has no phase. The engine promoted them to
       ``complex(real, 0)``, so ``angle()`` was a flat zero field and the metric
       was identically zero — even though both metrics already carry a Fourier
       fallback for real input.
   * - k-space / physics
     - ``g_factor``, ``asymptotic_gfactor``, ``through_plane_fwhm``
     - ``g_factor`` returns a constant ``1.0`` when called without sensitivity
       maps (and its computation ignored the acceleration); ``asymptotic_gfactor``
       returned an init-time constant. ``through_plane_fwhm`` needs a 3-D volume.

The unifying insight: each zero-metric needs a dimension the magnitude sweep
discarded (phase, the coil/k-space axis, or the population axis). The
**Fourier bridge** is the inverse operation that re-lifts magnitude back into
the domain each metric requires.

The Fourier bridge
------------------

``scripts/sim2rank/fourier_bridge.py`` (:class:`FourierBridge`) reuses the
canonical physics primitives (``fft2c``/``ifft2c``, ``create_synthetic_csm``,
``sense_forward``, :class:`MaskGenerator`) — never a raw ``torch.fft`` — to
synthesize the discarded dimensions. For M4Raw the real phase and coil maps are
preferred (see :ref:`sim2rank-native-data-eval`); the bridge is the fallback for
magnitude-only inputs:

* ``to_complex_image(mag)`` attaches a smooth, deterministic background phase
  :math:`\varphi` (a stand-in for B0 / receiver phase), forming
  :math:`x = |x|\,e^{i\varphi}` without changing the magnitude.
* ``image_phase_reference(mag)`` returns the centred k-space
  :math:`\mathcal{F}(|x|\,e^{i\varphi})`. The forward FFT couples *magnitude
  structure* into k-space phase, so two different magnitude images have
  measurably different k-space phases — the property that restores signal to
  ``ipen`` / ``phase_mse``.
* ``to_multicoil_kspace(mag, num_coils, accel)`` synthesizes birdcage
  sensitivity maps and a severity-tied undersampling mask, producing the SENSE
  forward :math:`y = M F S x` for the parallel-imaging metrics.

Population-based distributional evaluation
------------------------------------------

Distributional metrics now receive a *different evaluation strategy*: at each
severity :math:`\theta` on each axis we degrade a **population** of reference
images and compare its distribution to the clean population via
``Sim2RankEngine.evaluate_distributional`` (wired per axis by
``evaluate_axis_distributional``). The engine dispatches by interface —
``summarize=True`` metrics (FID, KID, IS) use the accumulator pattern;
``summarize``-falsy metrics (sliced-Wasserstein, W1, MMD, KSD) are called
directly as two-population comparators. As :math:`\theta` grows the
distributions diverge and the distance rises.

Severity-tied g-factor
-----------------------

The g-factor is an *encoding-property* metric — it measures parallel-imaging
noise amplification from the sampling + coil geometry, independent of the image.
It is therefore only meaningful on undersampling axes, where the engine maps
severity to an acceleration :math:`R \in [1, R_{\max}]`, synthesizes coil maps,
and evaluates a textbook SENSE g-factor

.. math::

   g_k = \sqrt{[(E^H E)^{-1}]_{kk}\,[E^H E]_{kk}},
   \qquad E \in \mathbb{C}^{n_c \times R},

which rises as :math:`R \to n_c`. ``asymptotic_gfactor`` uses the
Marchenko–Pastur worst-case bound :math:`1+\sqrt{c}` with :math:`c = R/n_c`. On
non-undersampling axes both stay constant — which is physically correct.
``through_plane_fwhm`` is intentionally left for volumetric (3-D) evaluation; it
cannot be measured from a 2-D slice, and as of 2026-05-25 it is removed from the
sim2rank ``METRIC_SPECS`` sweep (it raised on every 2-D call → NaN). It stays in
the registry for true 3-D evaluation.

.. note::

   **No-reference registry ↔ sweep coherence (2026-07-24 NR audit).**
   ``g_factor`` is registered ``requires_reference=False``: its ``__call__``
   ignores ``target`` and measures the coil/sampling geometry alone, so the
   registry flag now agrees with its ``NO_REFERENCE`` type in ``METRIC_SPECS``
   (it previously fell to the implicit ``requires_reference=True`` default — a
   two-SSOT drift, now guarded by
   ``tests/unit/test_sim2rank_canonical_counts.py::test_no_reference_specs_agree_with_registry``).
   Three further ``requires_reference=False`` metrics —
   ``composed_spectral_norm_bound`` and ``max_layer_spectral_norm``
   (``lipschitz_bound_metrics``) and ``srf_bound`` — are deliberately **not** in
   the sweep: they score a *model / operator* property (spectral-norm and
   super-resolution-factor bounds), not a degraded image, so ramping artifact
   severity :math:`\theta` through them is undefined. They stay in the registry
   for model-analysis use. Every one of the remaining registered no-reference
   metrics (the ``nr_*`` / ``no_reference_*`` / ``physics_nr_metrics`` battery)
   is both registered and swept.

.. _sim2rank-native-data-eval:

Native-data evaluation (M4Raw, 2026-05-25)
------------------------------------------

The Fourier bridge described above *synthesizes* phase and coil maps — but for
M4Raw the pipeline already computes them in **real** form and then discards
them. ``synthesize_pseudo_gt`` loads the native complex multi-coil k-space,
averages repetitions, estimates **ESPIRiT** coil sensitivities, SENSE-combines
to a complex (phase-bearing) image, and only then takes the magnitude. The real
complex image and the real coil maps existed one step before the ``.abs()``.

The engine now **prefers the real quantities when supplied**, falling back to
the bridge only for magnitude-only inputs (synthetic phantoms, fastMRI
magnitude):

* **Phase metrics** (``ipen``, ``phase_mse``): the degradation sweep emits the
  SENSE-combined complex degraded image per timestep
  (``DegradationSweep.sweep_combined(..., return_complex=True)``), and the
  reference complex GT comes from ``coil_combine_sense``. ``Sim2RankEngine``
  feeds these complex tensors straight to the metric, which takes
  :math:`\angle(\cdot)` of the *real image-domain phase* — degradation-sensitive
  and physically meaningful, unlike the bridge's magnitude-independent synthetic
  phase.
* **Parallel-imaging metrics** (``g_factor``): the engine uses the real ESPIRiT
  ``smaps`` instead of a synthetic birdcage array, and caps the severity-tied
  acceleration at the real coil count :math:`n_c` (the SENSE g-factor is
  degenerate, :math:`\equiv 1`, for :math:`R > n_c`; M4Raw has 4 coils).

Threaded via ``evaluate_sweep(..., smaps=, complex_degraded=, complex_gt=)`` and
``evaluate_timestep(..., smaps=, complex_deg=, complex_gt=)``. All real tensors
are moved onto the engine device, so the path is correct on CUDA — which is now
the **canonical backend** (``--device`` defaults to ``auto``: CUDA when
available, else CPU), because the native path adds GPU-bound ESPIRiT SVD,
complex k-space degradation, and Inception feature extraction. The
central-slice limitation still stands: ``synthesize_pseudo_gt`` keeps only the
central slice, so ``through_plane_fwhm`` and other 3-D metrics remain deferred
until the volume is carried through.

Newly added IQA metrics (2026-05-24)
------------------------------------

The 2026 MRI-IQA survey [ReassessFR2025]_ flags several widely used closed-form
full-reference metrics this framework lacked. Five are added by wrapping the
validated ``piq`` implementations (optional ``[iqa]`` extra; skipped gracefully
when ``piq`` is absent, like the radiomic metrics): ``haarpsi`` (higher better),
``mdsi`` (lower), ``vsi`` (higher), ``dss`` (higher), ``ms_gmsd`` (lower).

The gradient-based no-reference family is grounded in the MRI motion-autofocus
literature:

* ``gradient_entropy`` — Shannon entropy of the gradient-magnitude histogram;
  the *Entropy Focus Criterion* validated for MRI motion correction
  [Atkinson1997]_.
* ``normalized_gradient_squared`` (**new**) — the lower-cost autofocus
  counterpart [McGee2000]_, gradient energy normalized by image energy:

  .. math::

     \mathrm{NGS} = \frac{\sum_i (\partial_x I)^2 + (\partial_y I)^2}
                         {\sum_i I^2 + \epsilon},

  intensity-scale invariant, higher for sharper images.
* ``gradient_error`` — full-reference Sobel gradient-magnitude :math:`L_1`,
  already present; pinned by regression tests.

Extended no-reference artifact-detection sweep (2026-05-25)
-----------------------------------------------------------

A broader blind-IQA pass added classical computer-vision and MRI artifact
measures that flag a specific degradation **without a clean reference**,
complementing the learned blind metrics (BRISQUE, NIQE). All collapse complex /
multi-coil input to a single grayscale channel
(``spectramr.core.metrics.no_reference_extended``):

.. list-table:: No-reference artifact-detection metrics
   :header-rows: 1
   :widths: 26 12 62

   * - Metric
     - Direction
     - Detects / definition
   * - ``brenner_focus``
     - higher = sharper
     - Brenner gradient :math:`\sum (I_{x+2}-I_x)^2` over image energy
       [Brenner1971]_; blur lowers it.
   * - ``immerkaer_noise``
     - higher = noisier
     - Fast noise-:math:`\sigma` via a structure-insensitive Laplacian mask
       [Immerkaer1996]_.
   * - ``mlv``
     - higher = sharper
     - Maximum Local Variation spread (max abs 8-neighbour difference)
       [Bahrami2014]_.
   * - ``intensity_entropy``
     - — (information)
     - Shannon entropy of the intensity histogram.
   * - ``blockiness``
     - higher = more grid
     - Ratio of on-grid to off-grid gradient energy (blocking / aliasing
       grid) [Wang2000]_.
   * - ``high_freq_energy_ratio``
     - higher = sharper
     - Fraction of spectral energy above a radial cutoff (real-valued
       feature FFT, not k-space).
   * - ``gibbs_ringing``
     - higher = more ringing
     - Laplacian oscillation energy in a band beside strong edges; flags
       Fourier-truncation ringing.

Four further full-reference / no-reference metrics from ``piq`` were wired:
``iw_ssim`` (information-weighted SSIM, upsampled to :math:`\ge 161`),
``srsim`` (spectral-residual similarity), ``vif_p`` (pixel-domain VIF), and
``total_variation`` (no-reference; rises with noise).

.. note::

   Metrics still considered for future addition (not yet wired): the learned
   blind metrics CLIP-IQA, MUSIQ, PaQ-2-PiQ, MetaIQA and PieAPP — all require
   pretrained-weight downloads (available via the ``pyiqa`` toolbox), which is
   impractical on the offline cluster.

VF-campaign reporting metrics
-----------------------------------------------------------

The Virtual-Fiducial campaign (`experiments/inprogress/vf/`) referenced metric
names that the plan claimed were unregistered. Two are genuine
``(prediction, target)`` quality metrics and were added to the registry:

.. list-table:: VF-campaign quality metrics
   :header-rows: 1
   :widths: 26 12 62

   * - Metric (arm)
     - Direction
     - Definition / home
   * - ``hallucination_rate`` (vf_08)
     - lower = fewer fabrications
     - **Alias** of ``fabrication_rate`` (``hallucination_metrics``):
       :math:`|F_{\text{pred}} \setminus F_{\text{target}}| / |F_{\text{pred}}|`,
       the fraction of predicted structure absent from the target. Not a
       duplicate implementation — the same quantity under the arm's name.
   * - ``banding_region_mse`` (vf_10)
     - lower = better
     - Region-restricted MSE over a banding ROI supplied via the
       ``region_mask`` / ``banding_mask`` kwarg
       (``spectramr.core.metrics.region_metrics``). Without a mask it warns and
       returns a full-FOV MSE (an honest, flagged degradation).

.. note::

   **Stale plan claims corrected.** ``g_factor`` was already registered
   (``physics_nr_metrics``), so it was not re-added. ``param_count`` and
   ``inference_latency_ms`` (referenced by Method B) are **model-level**
   quantities, *not* ``(prediction, target)`` metrics — registering them in the
   metric registry breaks the registry-wide finite-scalar contract
   (``tests/contracts/test_metric_registry.py``). They are reported through the
   profiling path instead — :class:`spectramr.core.metrics.performance.PerformanceMetrics`
   already tracks ``parameters``, ``forward_time`` and ``throughput``. Method B's
   listing them under ``metrics:`` is a benign warn-skip; moving those two names
   to a reporting/profiling context is the recommended follow-up.

---

Image Quality Metrics
=====================

.. note::

   This section used to embed rendered per-metric maps computed on a real M4Raw
   FLAIR slice at 4x Cartesian undersampling. Those images are **not published**:
   ``docs/figures/`` has never been tracked -- a blanket ``*.png`` rule in
   ``.gitignore`` covers the whole path -- so every directive resolved to nothing
   and rendered as a broken image. The category headings below are kept, because
   they enumerate what is actually registered. Regenerate the plots locally
   against your own data if you want them.

**Reference IQA Metrics** — PSNR, SSIM (2 degradations), GMSD, HFEN, FSIM, MS-SSIM per scale, VIF:


**Error Metrics** — MSE, MAE, RMSE, NMSE, NRMSE, Phase MSE, error distribution, relative error:


**No-Reference, Artifact & Statistical Metrics** — SNR, CNR, EFC, FBER, Gradient Entropy, Ghosting, Spike Detection, Pearson/Cosine, Power Spectrum:


**Segmentation & Quantitative Metrics** — Dice, IoU, Tofts Ktrans, CRLB, g-factor:


Peak Signal-to-Noise Ratio (PSNR)
----------------------------------

**Registry name:** ``psnr`` — **Higher is better** ✓

The most widely used image quality metric in MRI reconstruction:

.. math::

   \text{PSNR} = 10 \cdot \log_{10}\left(\frac{L^2}{\text{MSE}}\right)
   = 10 \cdot \log_{10}\left(\frac{L^2}{\frac{1}{N}\sum_i(x_i - \hat{x}_i)^2}\right)

where :math:`L` is the dynamic range of the signal (typically 1.0 for
normalized MRI).

**Graded per sample, then averaged** (issue #1347). :math:`N` above ranges over
one image's voxels, never over a batch's:

.. math::

   \text{PSNR}_{\text{batch}}
   = \frac{1}{B}\sum_{b=1}^{B} 10 \cdot \log_{10}
     \left(\frac{L_b^2}{\text{MSE}_b}\right)

The log is concave, so reducing MSE over a whole batch and taking the logarithm
once is *not* the mean of the per-image scores. Doing that made the published
number a function of how the loader happened to group the images: **14.3 dB**
across ``batch_size`` 1 → 24 on a heterogeneous 24-image set, with the
predictions held bit-identical. The effect is driven by heterogeneity across
samples and nearly vanishes on uniform synthetic data (0.005 dB), which is why
no fixture caught it.

:math:`L_b` carries the same rule wherever the range is a **per-image peak** —
``domain="kspace"`` and ``use_target_max`` both resolve it per sample, so the
loudest spectrum in a batch no longer sets the reference for every other image.
The default range is the one deliberate exception: it resolves a *contract*
(``[0, 1]`` vs ``[-1, 1]``) from the sign of the data, and a per-sample sign
test would read an all-positive sample of a ``[-1, 1]`` dataset as ``[0, 1]``
and halve its range. On mixed-sign data that contract can still differ between
two batch compositions; declare ``metrics.data_range`` to pin it.

The epoch value is weighted by each batch's **sample** count, not by batch
count, so a short final batch under ``drop_last=False`` no longer weighs as much
as a full one. Both conventions are stamped into ``provenance.json`` under
``metric_aggregation`` — this change restates numbers the corpus has already
recorded, and nothing else in the artifact would say so.

.. note::

   Metrics that are a *ratio* of batch-level reductions (``nrmse``, ``nmse``)
   and the sibling log metrics ``robust_mri_psnr`` and ``snr`` still reduce over
   the whole batch. They inherit the sample-weighted epoch mean but keep their
   own Jensen term.


Robust MRI PSNR
---------------

**Registry name:** ``robust_mri_psnr`` — **Higher is better** ✓

A modified PSNR that clips outlier voxels before computing MSE, reducing
sensitivity to rare high-error pixels at the boundary:

.. math::

   \text{PSNR}_{\text{robust}} = 10 \cdot \log_{10}\left(\frac{L^2}{\text{MSE}_{\text{clip}}}\right)

where :math:`\text{MSE}_{\text{clip}}` excludes the top/bottom 1% of
error values.


Structural Similarity Index (SSIM)
-----------------------------------

**Registry name:** ``ssim`` — **Higher is better** ✓

Combines luminance, contrast, and structure comparisons in local windows:

.. math::

   \text{SSIM} = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}

where :math:`\mu`, :math:`\sigma` are local statistics within an
:math:`11 \times 11` Gaussian-weighted window.


Multi-Scale SSIM
----------------

**Registry name:** ``ms_ssim`` — **Higher is better** ✓

Computes SSIM at :math:`M = 5` scales via iterative downsampling:

.. math::

   \text{MS-SSIM} = [l_M]^{\alpha_M} \cdot \prod_{j=1}^{M} [cs_j]^{\beta_j}

Captures quality at multiple spatial frequencies simultaneously.


Complex Wavelet SSIM (CW-SSIM)
-------------------------------

**Registry name:** ``cw_ssim`` — **Higher is better** ✓

Operates in the complex wavelet domain, making it insensitive to small
geometric translations — useful for MRI where sub-pixel shifts are common:

.. math::

   \text{CW-SSIM} = \frac{2|\sum_i c_i^x \cdot \overline{c_i^y}| + K}{\sum_i |c_i^x|^2 + \sum_i |c_i^y|^2 + K}

where :math:`c^x, c^y` are complex wavelet coefficients.


Feature Similarity Index (FSIM)
-------------------------------

**Registry name:** ``fsim`` — **Higher is better** ✓

Uses phase congruency and gradient magnitude as primary features:

.. math::

   \text{FSIM} = \frac{\sum_i S_L(i) \cdot PC_m(i)}{\sum_i PC_m(i)}

where :math:`S_L(i)` is the combined similarity at pixel :math:`i` and
:math:`PC_m(i)` is the maximum phase congruency (used as importance weight).


Visual Information Fidelity (VIF)
---------------------------------

**Registry name:** ``vif`` — **Higher is better** ✓

Based on the natural scene statistics model in the wavelet domain:

.. math::

   \text{VIF} = \frac{\sum_{j \in \text{scales}} I(\mathbf{C}^{N,j}; \mathbf{F}^{N,j} | s^{N,j})}{\sum_{j \in \text{scales}} I(\mathbf{C}^{N,j}; \mathbf{E}^{N,j} | s^{N,j})}

where :math:`I(\cdot;\cdot)` denotes mutual information in each wavelet
subband.


HaarPSI (Haar Wavelet Perceptual Similarity)
--------------------------------------------

**Module:** ``spectramr.infrastructure.reporting.metrics.haarpsi``
**Function:** ``haar_psi(pred, target, *, data_range, c, alpha)`` — **Higher is better** ✓

Two-scale Haar wavelet decomposition with logit-weighted similarity:

.. math::

   \text{HaarPSI} = \sigma\!\left(\alpha \cdot
   \frac{\sum_{s,\theta} W_{s,\theta} \cdot \text{logit}(S_{s,\theta})}
        {\sum_{s,\theta} W_{s,\theta}}\right)

where :math:`W_{s,\theta}` are local wavelet-magnitude weights and
:math:`S_{s,\theta}` are per-scale similarity maps.  HaarPSI ranks among the strongest
MRI IQA correlates [Reisenhofer2018]_.

.. code-block:: python

   from spectramr.infrastructure.reporting.metrics.haarpsi import haar_psi
   import numpy as np
   score = haar_psi(pred_image, ref_image, data_range=1.0)  # float in [0, 1]


LPIPS (Learned Perceptual)
--------------------------

**Registry name:** ``lpips`` — **Lower is better** ✗

Deep perceptual distance using VGG features (same formulation as the loss):

.. math::

   d(x, \hat{x}) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \| w_l \odot (\phi_l(x) - \phi_l(\hat{x})) \|_2^2

.. note::
   LPIPS, FID, KID, DISTS and the perceptual losses all wrap an
   ImageNet-pretrained backbone that expects 3-channel RGB. Inputs
   with other channel counts (1-channel grayscale, 2/4-channel
   complex MRI, multi-coil) are routed through
   :func:`spectramr.core.metrics.channel_adapter.adapt_to_rgb`. See the
   :ref:`channel_adapter` section below for the available modes and
   trade-offs.


Gradient Magnitude Similarity Deviation (GMSD)
-----------------------------------------------

**Registry name:** ``gmsd`` — **Lower is better** ✗

Measures the standard deviation of the gradient magnitude similarity map:

.. math::

   \text{GMSD} = \sqrt{\frac{1}{N}\sum_i \left(\text{GMS}(i) - \overline{\text{GMS}}\right)^2}

where :math:`\text{GMS}(i) = \frac{2 m_r(i) m_d(i) + c}{m_r^2(i) + m_d^2(i) + c}`
and :math:`m_r, m_d` are gradient magnitudes of reference and distorted images.


HFEN (High-Frequency Error Norm)
---------------------------------

**Registry name:** ``hfen`` — **Lower is better** ✗

Applies LoG filter before L2 comparison, emphasizing edge fidelity:

.. math::

   \text{HFEN} = \frac{\|\text{LoG}(x) - \text{LoG}(\hat{x})\|_2}{\|\text{LoG}(x)\|_2}


Cubical PH-W2 distance (topology)
---------------------------------

**Registry name:** ``cubical_ph_w2_distance`` (aliases ``ph_w2_distance``,
``topology_w2``) — **Lower is better** ✗

**Module:** :mod:`spectramr.core.metrics.quantitative.cubical_ph_w2_distance`

The Wasserstein-2 distance between the sublevel-set cubical persistence
diagrams of prediction and target, taken per ``(batch, channel)`` slice and
averaged. It is the validation-side reading of the ``cubical_ph_w2`` loss:
:class:`~spectramr.models.losses.cubical_ph_w2_loss.CubicalPHWassersteinLoss`
owns the diagram routine and the matching, and the metric calls it without a
gradient, so the validation number of an arm that trains on the term is
its own objective. It is ``0`` at ``pred == target`` and bounded by the L2 error
(Cohen-Steiner, :math:`W_2 \le \|\hat{x} - x\|_2`). Added for the
``geomamba_ulf`` cohort (2026-09-03), whose claim is topological but whose arms
selected on ``val_psnr`` alone.

Needs the ``[topology]`` extra (``gudhi`` + ``POT``). Without it the
constructor raises the loss's ``ImportError`` (it does not return ``0.0``),
and the direction stays readable off the class, so ``val_cubical_ph_w2_distance``
still resolves as lower-is-better for checkpoint selection on a box without
the extra.


.. _tissue_segmentation_metrics:

Tissue-segmentation agreement (structural fidelity)
----------------------------------------------------

**Module:** :mod:`spectramr.core.metrics.tissue_segmentation`

These grade *image synthesis / translation* (e.g. ULF→HF field translation) on the
question PSNR cannot answer: **does the synthesized image still support the same
tissue partition as the real one?**

They do **not** compare against a reference segmentation — the ULF/HF cohort ships no
labels and no brain masks. Instead they segment the **prediction and the target with
the same deterministic segmenter** and measure whether the two label maps agree. This
is the standard downstream-task-consistency protocol (SynthSeg-Dice, as used by
SynthSR / LF-SynthSR).

:class:`~spectramr.core.metrics.tissue_segmentation.OtsuTissueSegmenter` computes an
Otsu brain mask and then a 3-class Otsu tissue partition *within* it (a CSF/GM/WM
proxy). It is deterministic, pure-torch, and range-agnostic — the histogram spans the
image's own ``[min, max]``, so it works on ``[-1, 1]``-normalised MRI. Any bias the
segmenter has cancels in the prediction-vs-target comparison.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Registry name
     - Direction
     - Measures
   * - ``tissue_dice``
     - Higher ✓
     - Mean Dice over the 3 tissue classes, ``seg(pred)`` vs ``seg(target)``.
   * - ``brain_mask_dice``
     - Higher ✓
     - Dice of the Otsu brain masks — does the predicted head occupy the same FOV?
   * - ``tissue_volume_similarity``
     - Higher ✓
     - Mean per-class :math:`1 - |V_{pred} - V_{target}| / V_{target}`.
   * - ``tissue_hd95``
     - Lower ✗
     - 95th-percentile Hausdorff distance between tissue boundaries, in **pixels**
       (MONAI-backed). Boundary-sensitive where Dice is area-sensitive.

**Relationship to** ``synthseg_dice``. That metric's local backend
(:class:`~spectramr.core.metrics.quantitative.segmentation.LabelDiceBackend`) bins by
intensity **quantile**, so its classes are equal-population *by construction*: it sees
spatial arrangement only and is invariant to any monotone intensity change, and its
class volumes can never disagree. The Otsu family derives thresholds from the
histogram, so class sizes are data-driven and both blur and volume error register.
The two are complementary — run both.

**Caveats (state these when citing the numbers):**

* **Slice-Dice, not volume-Dice.** With ``data.sampling.enable_slice_2d: true`` the validation loop
  passes one axial slice at a time, so the reported figure is a mean of *per-slice*
  Dice; small slices carry the same weight as large ones. Evaluate a representative
  slice set — the val loader is **unshuffled**, so a small
  ``validation.num_validation_batches`` grades only the first N adjacent slices of one
  volume.
* **Independent thresholds.** Prediction and target are each segmented with their own
  Otsu thresholds, so the Dice is invariant to a global *affine* intensity change by
  design (it isolates structure, which PSNR/MAE conflate with intensity calibration).
  Non-linear contrast errors and blur are penalised.
* **On a non-anatomical contrast, "tissue" is a misnomer.** The 3-class partition
  approximates CSF/GM/WM on T1w/T2w/FLAIR; on an ADC map it is a generic intensity
  partition — still a valid agreement readout, but not tissue. The registry names are
  deliberately neutral (``tissue_*``, never ``gm_*``/``wm_*``).

**Degenerate inputs never yield NaN.** ``pipelines/train.py`` accumulates validation
metrics with a bare running sum and no non-finite guard, so a single ``NaN`` would
poison the metric for the whole evaluation pass. Slices whose *target* brain mask falls
below :data:`~spectramr.core.metrics.tissue_segmentation.MIN_FOREGROUND_FRACTION` carry no
anatomy and are dropped from the batch average. For ``tissue_hd95``, a class present in
the target but **absent from the prediction** scores the FOV diagonal (the worst
distance the image admits) — averaging over only the finite entries would otherwise hand
a fully collapsed prediction a distance of ``0.0``, i.e. a perfect score for the exact
failure the metric exists to catch.

.. automodule:: spectramr.core.metrics.tissue_segmentation
   :members:
   :undoc-members:
   :show-inheritance:


.. _channel_adapter:

Multi-channel inputs to ImageNet-pretrained metrics
----------------------------------------------------

LPIPS, FID, KID, DISTS, and the VGG-based perceptual losses all run
ImageNet-pretrained backbones (VGG19 / InceptionV3) that expect a
``(B, 3, H, W)`` real-valued RGB input. MRI data is rarely RGB:

* **1 channel** — magnitude images (the common case).
* **2 channels** — interleaved real/imag (single-coil complex).
* **4 channels** — typical multi-coil complex (2 coils × 2) or
  multi-contrast stacks.
* **8 channels** — 4 coils × {real, imag}.

The :func:`spectramr.core.metrics.channel_adapter.adapt_to_rgb` helper makes
the conversion explicit. Each metric class accepts a
``channel_mode`` constructor argument; the default ``"auto"`` is
MRI-aware and never silently truncates channels.

.. list-table:: Channel-mode reference
   :header-rows: 1
   :widths: 18 32 50

   * - Mode
     - When to use
     - Behavior
   * - ``auto`` (default)
     - General MRI workflows; mixed grayscale / complex
     - ``C == 1`` → replicate to 3. ``C == 3`` → as-is. Even ``C > 1``
       (or already-complex) → coil-wise root-sum-of-squares magnitude →
       replicate. **Raises** ``ValueError`` for odd ``C != 1, 3``.
   * - ``grayscale_mean``
     - Multi-contrast stacks where channels are independent images;
       odd channel counts
     - Mean across the channel dim → 1 channel → replicate to 3.
       Always succeeds. Loses inter-channel structure.
   * - ``complex_rss``
     - You know the data is interleaved (real, imag) pairs and want
       to enforce that interpretation
     - Even ``C`` → reshape into ``C/2`` complex coils → RSS magnitude
       → replicate. Raises on odd ``C`` or non-complex non-even input.
   * - ``replicate``
     - Strict 1-channel input
     - Replicate to 3. Raises for any other ``C``.
   * - ``passthrough``
     - Strict 3-channel RGB input
     - As-is. Raises for any other ``C``.

**Why this exists.** Before this adapter, ``LPIPS`` used
``preds = preds[:, :3]`` to truncate to the first 3 channels — silently
discarding channel 4 and beyond. ``FID`` and ``KID`` only handled
``C == 1``, so 4-channel inputs reached InceptionV3 with the wrong
shape and produced meaningless features. Both are silent-fallback
anti-patterns of the kind CLAUDE.md item #9 forbids.

**API:**

.. autofunction:: spectramr.core.metrics.channel_adapter.adapt_to_rgb

.. autoclass:: spectramr.core.metrics.channel_adapter.ChannelMode
   :members:
   :undoc-members:

**Example:**

.. code-block:: python

   from spectramr.core.metrics.evaluation_metrics import LPIPS

   # Default — works for 1, 3, even-C MRI complex; raises on odd C != 1, 3
   metric = LPIPS(device="cuda:0")

   # Multi-contrast stack: 4 independent contrasts → mean to grayscale
   metric = LPIPS(device="cuda:0", channel_mode="grayscale_mean")

   # Strict 3-channel — raises if you ever feed it grayscale by mistake
   metric = LPIPS(device="cuda:0", channel_mode="passthrough")


---

Error Metrics
=============

MSE, MAE, RMSE
---------------

**Registry names:** ``mse``, ``mae``, ``rmse`` — **Lower is better** ✗

.. math::

   \text{MSE} = \frac{1}{N}\sum_i (x_i - \hat{x}_i)^2, \quad
   \text{MAE} = \frac{1}{N}\sum_i |x_i - \hat{x}_i|, \quad
   \text{RMSE} = \sqrt{\text{MSE}}


Normalized MSE (NMSE)
---------------------

**Registry name:** ``nmse`` — **Lower is better** ✗

Scale-invariant error normalization:

.. math::

   \text{NMSE} = \frac{\|x - \hat{x}\|_2^2}{\|x\|_2^2}


Phase MSE
---------

**Registry name:** ``phase_mse`` — **Lower is better** ✗

Measures phase accuracy for complex-valued reconstructions:

.. math::

   \text{Phase-MSE} = \frac{1}{N}\sum_i \left(\angle x_i - \angle \hat{x}_i\right)^2

where angles are wrapped to :math:`[-\pi, \pi]`.


---

Signal & Noise Metrics
======================

Signal-to-Noise Ratio (SNR)
----------------------------

**Registry name:** ``snr`` — **Higher is better** ✓

.. math::

   \text{SNR} = 10 \cdot \log_{10}\left(\frac{\text{Var}[\text{signal}]}{\text{Var}[\text{noise}]}\right)


Contrast-to-Noise Ratio (CNR)
------------------------------

**Registry name:** ``cnr`` — **Higher is better** ✓

Measures contrast between two tissue regions relative to noise:

.. math::

   \text{CNR} = \frac{|\mu_A - \mu_B|}{\sigma_{\text{noise}}}


Entropy Focus Criterion (EFC)
-----------------------------

**Registry name:** ``efc`` — **Lower is better** ✗

Shannon entropy of the voxel intensity distribution (lower = sharper focus):

.. math::

   \text{EFC} = -\frac{1}{\log N_{\max}} \sum_i p_i \log p_i

where :math:`p_i` is the normalized intensity of each voxel and
:math:`N_{\max}` is the maximum possible value.


Foreground-Background Energy Ratio (FBER)
-----------------------------------------

**Registry name:** ``fber`` — **Higher is better** ✓

Ratio of mean energy inside versus outside the brain mask:

.. math::

   \text{FBER} = \frac{\text{median}(|x_{\text{fg}}|^2)}{\text{median}(|x_{\text{bg}}|^2)}


---

Generative Model Metrics
=========================

Fréchet Inception Distance (FID)
---------------------------------

**Registry name:** ``fid`` — **Lower is better** ✗

Compares distributions of Inception-v3 features between real and generated
samples:

.. math::

   \text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)

where :math:`(\mu_r, \Sigma_r)` and :math:`(\mu_g, \Sigma_g)` are the
mean and covariance of Inception features for real and generated sets.


Kernel Inception Distance (KID)
-------------------------------

**Registry name:** ``kid`` — **Lower is better** ✗

An unbiased alternative to FID using the squared MMD with polynomial kernel:

.. math::

   \text{KID} = \text{MMD}^2(p_r, p_g) = \mathbb{E}[k(x,x')] - 2\mathbb{E}[k(x,y)] + \mathbb{E}[k(y,y')]

where :math:`k` is the cubic polynomial kernel on Inception features.


Model-level profiling (param_count, inference_latency_ms)
---------------------------------------------------------

**Registry names:** ``param_count`` (aliases ``params``, ``num_params``) and
``inference_latency_ms`` (aliases ``latency``, ``latency_ms``) — **Lower is
better** ✗ for both.

Unlike every other metric on this page, these are functions of the *model
object*, not of a ``(prediction, target)`` image pair. ``param_count`` returns
the trainable-parameter count (pass ``trainable_only=False`` for the full
count); ``inference_latency_ms`` returns the mean forward-pass latency in
milliseconds over ``n_runs`` (with ``n_warmup`` warm-up passes and CUDA
synchronisation). Both are passed the model via a ``model`` kwarg (or a
``module`` / ``net`` / ``generator`` alias) and an ``input_tensor`` probe, and
return ``nan`` when those are absent.

Because their input contract is not an image pair, they are annotated
``MetricType.DOMAIN_SPECIFIC`` in ``scripts/sim2rank/metrics_list.py`` so they
are excluded from both swept buckets (``PER_IMAGE_SPECS``, ``SUMMARY_SPECS``)
and the registry-contract harness (:doc:`audit_ladder_user_guide`) skips the
synthetic ``(pred, target)`` forward call. They are consumed by the reporting /
profiling pipeline, not the per-image validation loop, and back the
``metadata.secondary_metrics`` references in the HyperMamba VF arm.


Functional / BOLD temporal metrics (tsnr, temporal_fidelity)
------------------------------------------------------------

**Registry names:** ``tsnr`` (aliases ``temporal_snr``, ``TemporalSNR``) and
``temporal_fidelity`` (aliases ``temporal_correlation``, ``TemporalFidelity``)
— **Higher is better** ✓ for both. Workflow-tagged ``mri_functional``;
``temporal_fidelity`` additionally carries ``mri_dynamic``.

``tsnr`` is :math:`\mathrm{mean}_t(S) / \mathrm{std}_t(S)` over a de-drifted
voxel time-course (a linear drift is removed first — scanner drift is a
low-frequency artifact, not noise, and leaving it in inflates the denominator).
``temporal_fidelity`` is the mean Pearson correlation between predicted and
reference voxel time-courses.

**Report the pair, never ``tsnr`` alone.** ``tsnr`` is maximised by destroying
the signal it exists to protect: temporal smoothing raises it monotonically
while collapsing ``temporal_fidelity``. Measured on dynamics near the temporal
Nyquist (~4 frames/cycle), a moving average moves tSNR 11.6 → 39.8 while
fidelity goes 0.82 → **−0.45**. Against *slow* dynamics the same smoothing is a
genuinely good denoiser and improves both — which is why the trade-off only
appears when the dynamics are faster than the kernel.

Both read ``[B, T, H, W]`` real series with time on axis 1 and require
**T ≥ 2**: a single frame has no temporal axis to measure. As with the
model-profiling metrics above, their input contract is not an image pair, so
they are annotated ``MetricType.DOMAIN_SPECIFIC`` in
``scripts/sim2rank/metrics_list.py`` — excluded from ``PER_IMAGE_SPECS`` /
``SUMMARY_SPECS``, and the registry-contract harness skips the synthetic
``(pred, target)`` forward call instead of hard-failing on the ``nan`` they
correctly return for it.

A ``nan`` from either is "not measured", never a score. It means the input was
not a usable series (single frame, complex, or fewer than 3 dims), or that no
voxel cleared the foreground / variance guard — for ``tsnr`` the foreground is
an Otsu threshold, which adapts to the actual air fraction rather than assuming
a fixed one (real EPI is 60–80% air).


Fréchet Radiomic Distance (FRD)
-------------------------------

**Registry name:** ``frd`` — **Lower is better** ✗

Medical-imaging-specific FID using radiomic features (GLCM, shape, etc.)
instead of Inception features:

.. math::

   \text{FRD} = \|\mu_r^{\text{rad}} - \mu_g^{\text{rad}}\|^2 + \text{Tr}(\Sigma_r^{\text{rad}} + \Sigma_g^{\text{rad}} - 2(\Sigma_r^{\text{rad}} \Sigma_g^{\text{rad}})^{1/2})


---

K-Space & Spectral Metrics
============================

K-Space Error
-------------

**Registry name:** ``kspace_error`` — **Lower is better** ✗

Relative error in k-space:

.. math::

   \text{KSE} = \frac{\|\mathcal{F}(x) - \mathcal{F}(\hat{x})\|_2}{\|\mathcal{F}(x)\|_2}


Complex HFEN
-------------

**Registry name:** ``complex_hfen`` — **Lower is better** ✗

HFEN applied to complex-valued data (both real and imaginary channels).


Power Spectrum Consistency
--------------------------

**Registry name:** ``power_spectrum_consistency`` — **Higher is better** ✓

Correlation between radially-averaged power spectra:

.. math::

   \text{PSC} = \text{corr}\left(P_r(|k|),\ P_g(|k|)\right)

where :math:`P(|k|) = \langle |\mathcal{F}(x)(k)|^2 \rangle_{|k|=\text{const}}`.


---

Segmentation Metrics
=====================

Dice Similarity Coefficient
----------------------------

**Registry name:** ``dice`` — **Higher is better** ✓

.. math::

   \text{Dice} = \frac{2|X \cap Y|}{|X| + |Y|}


Intersection Over Union (IoU)
------------------------------

**Registry name:** ``iou`` — **Higher is better** ✓

.. math::

   \text{IoU} = \frac{|X \cap Y|}{|X \cup Y|} = \frac{\text{Dice}}{2 - \text{Dice}}


Hausdorff Distance 95th Percentile (HD95)
------------------------------------------

**Registry name:** ``hd95`` — **Lower is better** ✗

A surface-distance metric robust to outliers — uses the 95th percentile
of the directed Hausdorff distances in both directions:

.. math::

   \text{HD95}(A, B) = \max\left(h_{0.95}(A, B),\ h_{0.95}(B, A)\right)

where :math:`h_{0.95}(A, B) = \text{percentile}_{0.95}\{\min_{b \in B} d(a, b) : a \in A\}`.

The 95th percentile suppresses the extreme outliers of standard Hausdorff
distance, making HD95 clinically preferred for lesion segmentation challenges
(BraTS, WMH).


WMH Dice Evaluator
-------------------

**Class:** :class:`~spectramr.core.metrics.wmh_dice_evaluator.WMHDiceEvaluator` —
**File:** ``src/spectramr/core/metrics/wmh_dice_evaluator.py``

A downstream clinical evaluator that measures White-Matter-Hyperintensity (WMH)
overlap between reconstructed FLAIR volumes and ground-truth WMH annotations.

**Adapter pattern:** The evaluator does not ship a learned segmenter. Instead it
wraps any ``WMHSegmenterAdapter``-compliant backend:

.. code-block:: python

   from spectramr.core.metrics.wmh_dice_evaluator import WMHDiceEvaluator, LazySegmenterAdapter

   # Diagnostic baseline (percentile thresholding — smoke tests only)
   evaluator = WMHDiceEvaluator(segmenter=LazySegmenterAdapter(percentile=0.97))

   # Batched (B, C, D, H, W) volumes
   scores = evaluator(pred_volume, target_volume, mask=brain_mask)
   # → Tensor of shape (B,) with per-sample Dice scores

**Dice formula:**

.. math::

   \text{WMH-Dice} = \frac{2|\text{seg}_{pred} \cap \text{seg}_{target}| + \varepsilon}
                         {|\text{seg}_{pred}| + |\text{seg}_{target}| + \varepsilon}

where :math:`\varepsilon` is the smoothing constant (default ``1.0``).

.. note::

   ``LazySegmenterAdapter`` is a diagnostic baseline only. For clinical
   evaluation, plug in **LST-AI** or **SynthSeg-WMH** via the adapter
   protocol. The ``segment()`` method must accept a 3-D magnitude volume
   and an optional brain mask.

**Input requirements:**

- ``pred``, ``target``: shape ``(B, 1, D, H, W)`` — FLAIR magnitude volumes
- ``mask``: optional brain mask, shape ``(B, 1, D, H, W)`` or ``(B, D, H, W)``
- For multi-channel inputs (``C > 1``), channel 0 is used as a magnitude proxy

**Design rationale:** Plan §11 P5 deliverable. Separates reconstruction quality
(PSNR/SSIM) from downstream clinical utility (WMH lesion detection), enabling
end-to-end evaluation without coupling the reconstruction pipeline to a
specific segmentation model.

**Saturation guard (2026-05 fix):** ``LazySegmenterAdapter.segment``
uses ``volume >= threshold`` rather than ``volume > threshold``
whenever the top-percentile cutoff equals the image maximum. The
strict-``>`` form produces an empty segmentation when a bright block
is binary-valued (the percentile lands *at* the block intensity),
which collapses Dice to ``smooth/smooth = 1.0`` for two completely
disjoint bright blocks — a textbook false-positive.


---

Artifact Detection Metrics
============================

Ghosting Ratio
--------------

**Registry name:** ``ghosting_ratio`` — **Lower is better** ✗

Detects phase-encode ghosting by comparing signal in ghost regions to the
foreground:

.. math::

   \text{GR} = \frac{\bar{I}_{\text{ghost}} - \bar{I}_{\text{bg}}}{\bar{I}_{\text{fg}}}


Spike Detection
---------------

**Registry names:** ``spike_detection``, ``spike_percentage`` — **Lower is better** ✗

Identifies voxels with intensity exceeding :math:`\mu + n\sigma` threshold
in k-space, indicative of gradient coil spikes.


Zipper Detection
----------------

**Registry name:** ``zipper_detection`` — **Lower is better** ✗

Detects periodic banding artifacts from RF interference along the
frequency-encode direction.


---

Cascading Acceleration Validation Metrics
==========================================

For diffusion strategies (e.g., Experiment 11), validation runs at multiple
acceleration levels simultaneously. Each base metric is emitted once per level
with a ``_{R}x`` suffix. These are tracked separately in TensorBoard and
the training CSV.

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Column Pattern
     - Levels
     - Example Columns
   * - ``val_psnr_{R}x``
     - 2, 8, 32
     - ``val_psnr_2x``, ``val_psnr_8x``, ``val_psnr_32x``
   * - ``val_ssim_{R}x``
     - 2, 8, 32
     - ``val_ssim_2x``, ``val_ssim_8x``, ``val_ssim_32x``
   * - ``val_mse_{R}x``
     - 2, 8, 32
     - ``val_mse_2x``, ``val_mse_8x``, ``val_mse_32x``
   * - ``val_mae_{R}x``
     - 2, 8, 32
     - ``val_mae_2x``, ``val_mae_8x``, ``val_mae_32x``
   * - ``val_psnr_scaled_{R}x``
     - 2, 8, 32
     - After normalization to [0, 1] before PSNR
   * - ``val_timestep_mean_{R}x``
     - 2, 8, 32
     - Mean diffusion timestep sampled during this cascade level
   * - ``val_pred_mean_{R}x``
     - 2, 8, 32
     - Mean prediction intensity (stability check)

**Best metric selection:** ``val_lpips`` (``mode: min``) is used for
checkpoint selection in Experiment 11, as it captures perceptual quality
better than PSNR on k-space data. Configured via:

.. code-block:: yaml

   checkpoint:
     best_metric_name: val_lpips
     best_metric_mode: min

---

Intraclass Correlation Coefficient (ICC)
=========================================

**File:** ``src/spectramr/core/metrics/icc.py`` — **Higher is better** ✓

Measures inter-rater or test-retest reliability of quantitative MRI
parameter maps (T1, T2, PD):

.. math::

   \text{ICC}(2,1) = \frac{MS_R - MS_E}{MS_R + (k-1)MS_E + k(MS_C - MS_E)/n}

where :math:`MS_R, MS_C, MS_E` are the mean squares for rows, columns,
and error terms of a two-way mixed-effects ANOVA.

Used in quantitative MRI pipeline validation (``qmap_physics_advanced``
experiments) to quantify reproducibility across scan sessions.

---

Quantitative MRI Metrics
=========================

Tofts Model (Ktrans)
--------------------

**Registry name:** ``ktrans`` — **Higher is better** ✓

Pharmacokinetic parameter from the Tofts model for DCE-MRI:

.. math::

   C_t(t) = K^{\text{trans}} \int_0^t C_p(\tau) e^{-k_{ep}(t-\tau)} d\tau

where :math:`C_t` is tissue concentration, :math:`C_p` is the arterial
input function, and :math:`k_{ep} = K^{\text{trans}}/v_e`.


Cramér-Rao Lower Bound (CRLB)
------------------------------

**Registry name:** ``crlb`` — **Lower is better** ✗

Theoretical lower bound on the variance of any unbiased estimator,
calculated from the Fisher Information Matrix:

.. math::

   \text{Var}[\hat{\theta}_i] \geq [I(\theta)^{-1}]_{ii}

where :math:`I(\theta)_{ij} = -\mathbb{E}\left[\frac{\partial^2 \log p(x|\theta)}{\partial \theta_i \partial \theta_j}\right]`.


Conformal Risk Control on Parameter Maps (qCRC)
-----------------------------------------------

**Registry names:** ``conformal_risk_control`` (aliases ``qCRC``,
``crc_radius``) — **Lower is better** ✗ — and
``qmap_conformal_coverage`` (alias ``qCRC_coverage``) — **Higher is
better** ✓.

qCRC lifts Conformal Risk Control [Angelopoulos2024]_ onto the *geodesic*
error functional of quantitative ``(M0, T1, T2)`` maps on the Bloch
relaxation manifold (see
:class:`~spectramr.infrastructure.physics.manifolds.BlochRelaxationManifold`).
Instead of certifying coverage of pixel intensities, it certifies a
geodesic tolerance for the *parameter map itself* via the nested
geodesic prediction set

.. math::

   \mathcal{C}_\lambda(y) = \{\, \theta' : d_g(\theta', \hat{\theta}(y)) \le \lambda \,\}.

The batch (calibration cohort of :math:`n` scans) yields per-voxel
geodesic residuals :math:`r_{i,v} = d_g(\theta_i, \hat{\theta}_i)`. The
miscoverage loss :math:`\ell_i(\lambda) = \operatorname{mean}_v
\mathbf{1}[r_{i,v} > \lambda]` is monotone non-increasing in
:math:`\lambda` because the geodesic balls are nested
(:math:`\lambda_1 \le \lambda_2 \Rightarrow \mathcal{C}_{\lambda_1}
\subseteq \mathcal{C}_{\lambda_2}`), so CRC selects

.. math::

   \hat{\lambda} = \inf\Big\{ \lambda : \hat{R}_n(\lambda) \le
   \alpha - \tfrac{B - \alpha}{n} \Big\},
   \qquad \hat{R}_n(\lambda) = \operatorname{mean}_i \ell_i(\lambda),

which guarantees :math:`\mathbb{E}[\ell(\mathcal{C}_{\hat\lambda},
\theta_{n+1})] \le \alpha` distribution-free and finite-sample with loss
bound :math:`B = 1`. The trivial limit :math:`\alpha = 0` forces full
coverage (``lambda_hat`` is the largest residual); a perfect
reconstruction certifies a zero radius. Calibration size: the slack
:math:`(B-\alpha)/n` falls below a tolerance :math:`\varepsilon` once
:math:`n \ge (B-\alpha)/\varepsilon` (e.g. :math:`\alpha = 0.1`,
:math:`\varepsilon = 0.01` needs :math:`n \ge 90` calibration scans).

.. [Angelopoulos2024] A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, and
   T. Schuster. "Conformal Risk Control." *ICLR*, 2024. arXiv:2208.02814.


Empirical Coverage of a Reverse-Sample Ensemble
-----------------------------------------------

**Registry name:** ``empirical_coverage`` — **Higher is better** ✓

The fraction of target pixels that fall inside the per-pixel interval
:math:`\bar{x} \pm k\,\sigma` of an ensemble of ``N`` stochastic
cold-diffusion reverse samples, where :math:`\bar{x}` is the pixelwise
ensemble mean the validation path grades and :math:`\sigma` the pixelwise
sample standard deviation over the members (carried on
``MetricContext.ensemble_std``; ``k`` is the constructor argument, default
2). The count is :func:`~spectramr.core.metrics.quantitative.conformal_risk.coverage_fraction`,
the same function the qCRC coverage above uses at its calibrated radius, so
the two numbers share one definition of "inside" (the boundary counts). It
is not a certificate: without calibration there is no finite-sample
guarantee, only the observed hit rate that a conformal claim is checked
against. Without an ``ensemble_std`` the metric is declared not applicable
(``NaN`` plus one warning) rather than reported as a number.

The diffusion strategy produces it on a cold-diffusion arm that sets
``validation.sampling.enable_multistep_cold: true`` and
``validation.sampling.ensemble_samples: N`` (``2 <= N <= 4``) with
``model.model_kwargs.sampler_sigma > 0``: the multi-step sampler runs ``N``
times per input with one C6 noise stream per member (``seed_offset`` 0 to
``N-1`` on ``sampler_seed``), the k-space mean goes on as the prediction,
and ``val_ensemble_std_mean`` and ``val_empirical_coverage`` are written per
rung (``validation.sampling.coverage_k`` sets ``k``), with their cascade
``_mean`` like the other validation metrics. The load is refused when the
declared sigma is 0 and the run is refused when the resolved sampler's sigma
is 0, because ``N`` identical members would report a spread of zero.


---

Statistical Metrics
====================

Pearson Correlation
-------------------

**Registry name:** ``pearson`` — **Higher is better** ✓

.. math::

   r = \frac{\sum_i (x_i - \bar{x})(\hat{x}_i - \bar{\hat{x}})}{\sqrt{\sum_i (x_i - \bar{x})^2} \sqrt{\sum_i (\hat{x}_i - \bar{\hat{x}})^2}}


Cosine Similarity
-----------------

**Registry name:** ``cosine_similarity`` — **Higher is better** ✓

.. math::

   \cos\theta = \frac{x \cdot \hat{x}}{\|x\| \cdot \|\hat{x}\|}


Gradient Entropy
----------------

**Registry name:** ``gradient_entropy`` — **Lower is better** ✗

Entropy of the gradient magnitude histogram; lower values indicate sharper,
more focused images:

.. math::

   H_\nabla = -\sum_b p_b \log_2 p_b

where :math:`p_b` is the normalized histogram bin of :math:`|\nabla x|`.


---

Full Registry Table
===================

.. list-table::
   :header-rows: 1
   :widths: 22 22 8 48

   * - Name
     - Class
     - ↑/↓
     - Category
   * - ``bat``
     - ``BAT``
     - ↑
     - Segmentation
   * - ``cc_snr``
     - ``CCSNR``
     - ↑
     - Signal/Noise
   * - ``cjv``
     - ``CJV``
     - ↓
     - Signal/Noise
   * - ``clinical_ssim``
     - ``ClinicalSSIM``
     - ↑
     - Image Quality
   * - ``cnr``
     - ``CNR``
     - ↑
     - Signal/Noise
   * - ``complex_hfen``
     - ``ComplexHFEN``
     - ↓
     - K-Space
   * - ``cosine_similarity``
     - ``CosineSimilarity``
     - ↑
     - Statistical
   * - ``crlb``
     - ``CRLB``
     - ↓
     - Quantitative MRI
   * - ``cw_ssim``
     - ``CWSSIM``
     - ↑
     - Image Quality
   * - ``dice``
     - ``Dice``
     - ↑
     - Segmentation
   * - ``divergence``
     - ``Divergence``
     - ↓
     - Statistical
   * - ``efc``
     - ``EFC``
     - ↓
     - Signal/Noise
   * - ``fber``
     - ``FBER``
     - ↑
     - Signal/Noise
   * - ``fid``
     - ``FID``
     - ↓
     - Generative
   * - ``frd``
     - ``FrechetRadiomicDistance``
     - ↓
     - Generative
   * - ``freq_domain_snr``
     - ``FrequencyDomainSNR``
     - ↑
     - Signal/Noise
   * - ``fsim``
     - ``FSIM``
     - ↑
     - Image Quality
   * - ``ghosting_ratio``
     - ``GhostingRatio``
     - ↓
     - Artifact Detection
   * - ``gmsd``
     - ``GMSD``
     - ↓
     - Image Quality
   * - ``gradient_entropy``
     - ``GradientEntropy``
     - ↓
     - Statistical
   * - ``gradient_error``
     - ``GradientError``
     - ↓
     - Error
   * - ``hd95``
     - ``HD95``
     - ↓
     - Segmentation
   * - ``hfen``
     - ``HFEN``
     - ↓
     - Image Quality
   * - ``inception_score``
     - ``InceptionScoreMetric``
     - ↑
     - Generative
   * - ``icc``
     - ``ICC``
     - ↑
     - Statistical
   * - ``iou``
     - ``IoU``
     - ↑
     - Segmentation
   * - ``ipen``
     - ``IPEN``
     - ↓
     - Statistical
   * - ``kid``
     - ``KID``
     - ↓
     - Generative
   * - ``kspace_error``
     - ``KSpaceError``
     - ↓
     - K-Space
   * - ``ktrans``
     - ``ToftsModel``
     - ↑
     - Quantitative MRI
   * - ``lpips``
     - ``LPIPS``
     - ↓
     - Image Quality
   * - ``mad``
     - ``MAD``
     - ↓
     - Error
   * - ``mae``
     - ``MAE``
     - ↓
     - Error
   * - ``mass_conservation``
     - ``ConsistencyOfMass``
     - ↑
     - Quantitative MRI
   * - ``med_fid``
     - ``MedFID``
     - ↓
     - Generative
   * - ``ms_ssim``
     - ``MSSSIM``
     - ↑
     - Image Quality
   * - ``mse``
     - ``MSE``
     - ↓
     - Error
   * - ``ndc_diffusion``
     - ``NDC``
     - ↓
     - Diffusion
   * - ``negative_voxels``
     - ``NegativeVoxelCount``
     - ↓
     - Quantitative MRI
   * - ``nmse``
     - ``NMSE``
     - ↓
     - Error
   * - ``nr_iqa``
     - ``NRIQA``
     - ↑
     - Image Quality
   * - ``nrmse``
     - ``NRMSE``
     - ↓
     - Error
   * - ``pdm``
     - ``PDM``
     - ↓
     - Statistical
   * - ``pearson``
     - ``PearsonCorrelation``
     - ↑
     - Statistical
   * - ``phase_mse``
     - ``PhaseMSE``
     - ↓
     - Error
   * - ``power_spectrum_consistency``
     - ``PowerSpectrumConsistency``
     - ↑
     - K-Space
   * - ``psnr``
     - ``PSNR``
     - ↑
     - Image Quality
   * - ``rfs``
     - ``RadiomicFeatureStability``
     - ↑
     - Statistical
   * - ``rmse``
     - ``RMSE``
     - ↓
     - Error
   * - ``robust_mri_psnr``
     - ``RobustMRI_PSNR``
     - ↑
     - Image Quality
   * - ``snr``
     - ``SNR``
     - ↑
     - Signal/Noise
   * - ``spectral_linewidth``
     - ``SpectralLinewidth``
     - ↓
     - K-Space
   * - ``spike_detection``
     - ``SpikeDetection``
     - ↓
     - Artifact Detection
   * - ``spike_percentage``
     - ``SpikePercentage``
     - ↓
     - Artifact Detection
   * - ``ssim``
     - ``SSIMMetric``
     - ↑
     - Image Quality
   * - ``st_mad``
     - ``STMAD``
     - ↓
     - Statistical
   * - ``uqi``
     - ``UQI``
     - ↑
     - Image Quality
   * - ``vif``
     - ``VIF``
     - ↑
     - Image Quality
   * - ``vnr``
     - ``VelocityNoiseRatio``
     - ↑
     - Diffusion/Flow
   * - ``wash_slope``
     - ``WashSlope``
     - ↑
     - Quantitative MRI
   * - ``wmh_dice``
     - ``WMHDiceEvaluator``
     - ↑
     - Segmentation (Clinical)
   * - ``zipper_detection``
     - ``ZipperDetection``
     - ↓
     - Artifact Detection


---

==============================================
Perfusion & Dynamic Contrast-Enhanced Metrics
==============================================

**File:** ``src/spectramr/core/metrics/perfusion_metrics.py``

These metrics evaluate DCE-MRI (Dynamic Contrast-Enhanced) and ASL
(Arterial Spin Labelling) reconstruction quality through pharmacokinetic
model parameters.


WashSlope
----------

**Registry key:** ``wash_slope`` | **Aliases:** ``WashSlope``

Measures the rate of contrast agent uptake (**wash-in**) and
elimination (**wash-out**) from tissue over a DCE time series.

.. math::

   \text{Wash-in slope} = \frac{S_{peak} - S_{baseline}}
                                {\Delta t \cdot (t_{peak} - t_{baseline})}

   \text{Wash-out slope} = \frac{S_{peak} - S_{last}}
                                {\Delta t \cdot (t_{last} - t_{peak})}

**Input:**

.. code-block:: python

   from spectramr.core.metrics.perfusion_metrics import WashSlope

   metric = WashSlope(temporal_resolution=2.0)   # seconds per frame
   result = metric(
       dce_curve,        # Tensor[batch, voxels, time]
       baseline_frames=5
   )
   # result: {"wash_in_slope": Tensor, "wash_out_slope": Tensor}

**Clinical thresholds (breast DCE):**

- Wash-in slope > 100%/min → high suspicion of malignancy
- Type I curve (monotonic increase) → benign
- Type III curve (wash-out > 10%/min) → malignant


KtransVe (Tofts Model)
-----------------------

**Registry key:** ``ktrans_ve``

Fits the **Extended Tofts Model** to DCE time-activity curves:

- :math:`K^{trans}` — plasma→tissue volume transfer constant (min⁻¹)
- :math:`v_e` — extravascular extracellular volume fraction
- :math:`v_p` — plasma volume fraction (Extended Tofts only)

.. math::

   C_t(t) = K^{trans} \int_0^t C_p(\tau) e^{-K^{trans}(t-\tau)/v_e} d\tau
           + v_p C_p(t)

where :math:`C_p(t)` is the arterial input function (AIF).

.. code-block:: python

   from spectramr.core.metrics.perfusion_metrics import KtransVeMetric

   metric = KtransVeMetric(temporal_resolution=2.0, extended_tofts=True)
   params = metric(tissue_curves, aif_curve)
   # params: {"ktrans": Tensor, "ve": Tensor, "vp": Tensor}


BATMetric (Bolus Arrival Time)
-------------------------------

**Registry key:** ``bat``

Detects first significant contrast arrival using a threshold on the
signal enhancement derivative:

.. math::

   \text{BAT} = \min \left\{ t : \frac{dS}{dt}(t) > \mu_{baseline} + 3\sigma_{baseline} \right\}


---

=================================
MR Spectroscopy (MRS) Metrics
=================================

**File:** ``src/spectramr/core/metrics/spectroscopy_metrics.py``


SpectralLinewidth (FWHM)
--------------------------

**Registry key:** ``spectral_linewidth`` | **Aliases:** ``FWHM``

Full Width at Half Maximum of spectral peaks (Hz). Narrower linewidth
= better B0 homogeneity.

.. math::

   \text{FWHM} = \Delta f \cdot N_{\text{pts at half-max}}

.. code-block:: python

   from spectramr.core.metrics.spectroscopy_metrics import SpectralLinewidth

   metric = SpectralLinewidth(spectral_resolution=1.0, peak_window_hz=50.0)
   fwhm = metric(spectrum)   # Tensor[batch, freq_points]

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Quality
     - FWHM
     - Notes
   * - Excellent
     - < 0.1 ppm
     - < 12.8 Hz at 3T
   * - Clinical pass
     - < 0.15 ppm
     - < 19.2 Hz at 3T
   * - Fail (reshim)
     - > 0.2 ppm
     - > 25.6 Hz at 3T


CRLBMetric (Cramér-Rao Lower Bound)
--------------------------------------

**Registry key:** ``crlb``

Minimum variance lower bound on metabolite fitting uncertainty:

.. math::

   \text{CRLB}(\hat{\theta}_i) = \sqrt{[F^{-1}(\theta)]_{ii}}

where :math:`F` is the Fisher information matrix. CRLB < 20% is the
accepted clinical threshold for reliable quantification.


FreqSNR
--------

**Registry key:** ``freq_snr``

Signal-to-noise ratio in the frequency domain:

.. math::

   \text{SNR}_{freq} = \frac{\max_f |S(f)|}{\sigma_{noise}}

where :math:`\sigma_{noise}` is estimated from a signal-free spectral region.


---

===========================================================
Fréchet Radiomic Distance (FRD) & Radiomic Feature Stability
===========================================================

**File:** ``src/spectramr/core/metrics/radiomic.py`` | **Dependency:** ``pip install pyradiomics``

Radiomic-based image quality metrics that avoid the ImageNet-centric
bias of FID and LPIPS by using medically meaningful texture features.


FRD (Fréchet Radiomic Distance)
---------------------------------

**Registry key:** ``frd``

Analogous to FID but using PyRadiomics features:

.. math::

   \text{FRD} = \| \mu_r - \mu_g \|^2 +
   \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)

where :math:`(\mu_r, \Sigma_r)` are mean/covariance of PyRadiomics features
from real images, and :math:`(\mu_g, \Sigma_g)` from reconstructions.

**Feature groups:** ``firstorder`` (intensity statistics), ``glcm``
(texture), ``gldm`` (dependence), ``shape2D`` (morphology).

.. code-block:: python

   from spectramr.core.metrics.radiomic import FRDMetric

   frd = FRDMetric(feature_groups=["firstorder", "glcm", "gldm", "shape2D"])
   score = frd.compute(real_images, generated_images)
   # Lower is better: < 10 excellent, 10–50 acceptable, > 50 significant artefacts


RFS (Radiomic Feature Stability)
----------------------------------

**Registry key:** ``rfs``

Mean ICC between original and reconstructed radiomic features:

.. math::

   \text{RFS} = \frac{1}{|\mathcal{F}|} \sum_{f \in \mathcal{F}}
   \text{ICC}(x_f^{real}, x_f^{recon})

- RFS > 0.9 → excellent (features reproducible across methods)
- RFS 0.75–0.9 → good
- RFS < 0.75 → unstable (caution for downstream analysis)


---

================================
High-Frequency Error Norm (HFEN)
================================

**File:** ``src/spectramr/core/metrics/hfen.py``

**Registry key:** ``hfen`` | **Aliases:** ``HFEN``, ``HighFrequencyErrorNorm``

Applies a **Laplacian-of-Gaussian (LoG)** filter then computes normalized L1:

.. math::

   \text{HFEN} = \frac{\| \text{LoG}_\sigma(\hat{x}) - \text{LoG}_\sigma(x) \|_1}
                      {\| \text{LoG}_\sigma(x) \|_1 + \epsilon}

The LoG kernel:

.. math::

   \text{LoG}(r) = \frac{r^2 - 2\sigma^2}{\sigma^4}
   \exp\!\left(-\frac{r^2}{2\sigma^2}\right)

- **Lower is better** — 0.0 = perfect edge reproduction
- **Scale-selective** — ``sigma`` controls targeted spatial frequencies

.. code-block:: python

   from spectramr.core.metrics.hfen import HFENMetric

   hfen = HFENMetric(sigma=1.5, kernel_size=15)
   score = hfen(pred, target)    # Tensor[B, 1, H, W]

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - σ
     - Targets
     - Clinical Application
   * - 0.5–1.0
     - Very fine edges (< 2px)
     - Trabecular bone, vessel walls
   * - 1.5 (default)
     - Standard edges (2–4px)
     - Brain sulci, cartilage, tumour margins
   * - 2.0–3.0
     - Coarse boundaries (> 4px)
     - Organ boundaries, large lesions

**Config:**

.. code-block:: yaml

   metrics:
     compute_hfen: true
     hfen_sigma: 1.5


Field-domain metrics and the parametrization guard
===================================================

Field-valued claims (a B0 off-resonance map in Hz, a spiral trajectory deviation
in cycles/FOV) are graded against an *independently known* reference. The danger
is the **inert-mechanism facade** (pitfall #16): a model that advertises a field
output but actually emits an *image* (e.g. a corrected bSSFP frame) smoke-PASSes
while the field claim is measured by nothing. The guard makes this un-gradeable
rather than silently-wrong.

The comparability guard
-----------------------

``spectramr.core.metrics._guards.field_comparability`` implements the precondition
``comparable(u, v) = same shape AND same kind AND same units``. A metric grades
only when the verdict is ``ok``; otherwise it returns ``NaN`` (a *skip*, never a
silent grade, never a hard raise that would break unrelated validation):

.. code-block:: python

   from spectramr.core.metrics._guards import field_comparability

   verdict = field_comparability(
       estimate, reference,
       estimate_kind="field", reference_kind="field",
       estimate_units="Hz", reference_units="Hz",
   )
   if not verdict.ok:
       return float("nan")   # e.g. a complex *image* graded as a Hz field

``b0_field_rmse``
-----------------

Absolute RMS error (Hz) between an estimated B0 field and a real reference
(``spectramr.core.metrics.b0_field_rmse``; alias ``b0_rmse_hz``; lower is better).
A complex tensor is an image — not a real field — so it is reported as
``kind="image"`` and the guard skips it. The metric is the headline for the
phase-cycled bSSFP arm ``exp_vf_29`` and fires in
``ConcreteMultiAcquisitionStrategy.validation_step`` (as ``val_b0_field_rmse``)
for every B0 method that has a real reference.

``k_space_trajectory_rmse``
---------------------------

RMS error (cycles/FOV) between an estimated trajectory deviation ``Δk̂`` and the
*measured* deviation ``Δk_true = k_measured - k_nominal``
(``spectramr.core.metrics.k_space_trajectory_rmse``; alias ``traj_rmse``; lower is
better). It reuses the same guard with ``kind="trajectory"``,
``units="cycles_per_fov"``: a Cartesian per-readout-line ``Δk [B, 2, H]`` graded
against a spiral ``Δk(t) [n_samp, 2]`` is a shape mismatch and is skipped — the
diff_trajectory_opt facade the metric exists to block. The headline metric for
the spiral-trajectory-recovery arm ``exp_vf_35`` (problem formulation in
``docs/superpowers/specs/2026-06-08-spiral-trajectory-recovery-vf35.md``); its
static twin is the model's declared ``trajectory_parametrization`` capability.

The two-lock units invariant
----------------------------

The Hz-vs-Hz invariant is locked twice so a later config edit cannot re-open the
facade:

* **Static (Tier-1 audit)** — ``check_b0_field_metric_requires_hz_model``: if a
  config requests ``b0_field_rmse``, the chosen model must declare
  ``output_field_units == "Hz"`` on its ``@register_model`` capabilities. The
  ``bssfp_b0_regressor`` head does; the old ``bssfp_peak_finder`` (image output)
  does not, so wiring it to this metric fails the audit.
* **Runtime** — ``field_comparability`` above, on the realised tensor.

The model recovers ΔB0 from the banding via a frozen elliptical-signal-model
phase-cycle DFT prior plus a learned residual; see the problem formulation in
``docs/superpowers/specs/2026-06-08-bssfp-banding-b0-vf29.md`` and the synthesis
in ``spectramr.infrastructure.physics.multi_acquisition`` (``bssfp_banding`` /
``invert_bssfp_banding``).

The trajectory metric is locked the same way:

* **Static** — ``check_trajectory_metric_requires_matching_parametrization``: a
  config requesting ``k_space_trajectory_rmse`` must use a model declaring
  ``trajectory_parametrization == "spiral"`` (the ``spiral_trajectory_estimator``
  does; the Cartesian ``diff_trajectory_opt`` does not).
  ``check_trajectory_recon_requires_measured_emit`` additionally requires the
  dataset to emit the paired trajectories when ``read_measured_trajectory`` is set.
* **Runtime** — ``field_comparability`` with ``kind="trajectory"``, on the tensor.

The ``trajectory_recon`` paradigm (``TrajectoryReconstructionStrategy`` +
``spiral_trajectory_estimator``) supervises ``Δk̂ = G_θ(k_nominal)`` against the
measured deviation; only the GIRF path is differentiable (the NUFFT does not
backprop to the trajectory), so ``gradient_entropy`` is a no-grad sharpness
diagnostic, never a θ-objective. See
``docs/superpowers/specs/2026-06-08-spiral-trajectory-recovery-vf35.md``.


References
==========

1. Wang, Z., et al. "Image Quality Assessment: From Error Visibility to
   Structural Similarity." IEEE TIP, 2004.

2. Heusel, M., et al. "GANs Trained by a Two Time-Scale Update Rule
   Converge to a Local Nash Equilibrium." NeurIPS, 2017.

3. Zhang, R., et al. "The Unreasonable Effectiveness of Deep Features as
   a Perceptual Metric." CVPR, 2018.

4. Esteban, O., et al. "MRIQC: Advancing the Automatic Prediction of
   Image Quality in MRI from Unseen Sites." PLoS ONE, 2017.

5. Tofts, P.S. "Modeling Tracer Kinetics in Dynamic Gd-DTPA MR Imaging."
   JMRI, 1997.

6. Huttenlocher, D.P., et al. "Comparing Images Using the Hausdorff Distance."
   IEEE TPAMI, 1993. (HD95 surface distance)

7. Mendrik, A.M., et al. "MRBrainS Challenge: Online Evaluation Framework
   for Brain Image Segmentation in 3T MRI Data." Computational Intelligence
   and Neuroscience, 2015. (WMH segmentation evaluation protocol)

8. Shrout, P.E. and Fleiss, J.L. "Intraclass Correlations: Uses in Assessing
   Rater Reliability." Psychological Bulletin, 1979. (ICC)

9. Schmidt, P., et al. "An Automated Tool for Detection of FLAIR-Hyperintense
   White-Matter Lesions in Multiple Sclerosis." NeuroImage, 2012.
   (Background for WMH Dice evaluator baseline)

10. Ravishankar, S., Bresler, Y. "MR Image Reconstruction From Highly
    Undersampled k-Space Data by Dictionary Learning." IEEE TMI, 2011.
    (HFEN metric definition)

.. rubric:: Sim2Rank zero-metric review (2026-05-24)

.. [Atkinson1997] Atkinson, D., Hill, D.L.G., Stoyle, P.N.R., Summers, P.E.,
   Keevil, S.F. "Automatic Correction of Motion Artifacts in Magnetic
   Resonance Images Using an Entropy Focus Criterion." IEEE TMI 16(6), 1997.
   (gradient-entropy / Entropy Focus Criterion)

.. [McGee2000] McGee, K.P., Manduca, A., Felmlee, J.P., Riederer, S.J., Ehman,
   R.L. "Image Metric-Based Correction (Autocorrection) of Motion Effects:
   Analysis of Image Metrics." JMRI 11(2), 2000.
   (normalized gradient-squared autofocus measure)

.. [ReassessFR2025] "A Study of Why We Need to Reassess Full Reference Image
   Quality Assessment with Medical Images." J. Imaging Inform. Med., 2025.
   (full-reference IQA panel: HaarPSI, MDSI, VSI, DSS, MS-GMSD, IW-SSIM)

.. [Reisenhofer2018] Reisenhofer, R., Bosse, S., Kutyniok, G., Wiegand, T.
   "A Haar Wavelet-Based Perceptual Similarity Index for Image Quality
   Assessment." Signal Processing: Image Communication 61, 2018. (HaarPSI)

.. rubric:: Extended no-reference artifact-detection sweep (2026-05-25)

.. [Brenner1971] Brenner, J.F., et al. "An Automated Microscope for Cytologic
   Research." J. Histochem. Cytochem. 19(11), 1971. (Brenner focus measure)

.. [Immerkaer1996] Immerkaer, J. "Fast Noise Variance Estimation." Computer
   Vision and Image Understanding 64(2), 1996.

.. [Bahrami2014] Bahrami, K., Kot, A.C. "A Fast Approach for No-Reference Image
   Sharpness Assessment Based on Maximum Local Variation." IEEE Signal
   Processing Letters 21(6), 2014. (MLV)

.. [Wang2000] Wang, Z., Bovik, A.C., Evan, B.L. "Blind Measurement of Blocking
   Artifacts in Images." ICIP, 2000. (blockiness)

11. Zwanenburg, A., et al. "The Image Biomarker Standardization Initiative:
    Standardized Quantitative Radiomics for High-Throughput Image-based
    Phenotyping." Radiology, 2020. (PyRadiomics / FRD basis)

12. Parker, G.J.M., et al. "Experimentally-Derived Functional Form for
    a Population-Averaged High-Temporal-Resolution Arterial Input Function
    for Dynamic Contrast-Enhanced MRI." MRM, 2006. (DCE-MRI AIF)
