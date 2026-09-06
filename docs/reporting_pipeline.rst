Reporting pipeline (Nature-grade figures)
=========================================


.. contents:: On this page
   :local:
   :depth: 2

The end-of-training reporting pipeline (:mod:`spectramr.infrastructure.reporting`)
emits publication-grade figures, tables, and a provenance manifest into
``<run_dir>/report/``.

.. _reporting-single-entry-point:

One entry point, two tiers
--------------------------

``generate_report`` is the **only** end-of-training report path, reached from
``pipelines/train.py::_maybe_run_reporting``. What a run gets depends on whether
it configured reporting:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Config
     - Output
   * - No ``reporting:`` block, or ``enabled: false``
     - **Tables only.** ``tab_run_summary`` as ``.csv`` + ``.md`` + ``.tex``. No
       figures, no HTML, no optional dependencies.
   * - ``reporting.enabled: true``
     - The full pipeline: figures, tables, QC HTML, manifest.

.. rubric:: Why this needed fixing

It used to be two paths arranged backwards. ``MetricsReportGenerator`` ran from
``pipelines/training_loop.py`` on **every** run with no config gate at all, while
``generate_report`` sat behind ``reporting.enabled``, which **defaults False**.
The legacy generator therefore always ran and the canonical pipeline almost
never did.

The legacy call site is gone. The class itself remains — ``scripts/render_full_
reporting_pipeline.py`` builds its auto-report section from it — so this is an
SSOT claim about the *training hook*, not a deletion.

.. rubric:: Why the floor needed a new table

The obvious floor is "run the ``default`` preset's tables and skip the figures".
That emits **nothing**. Both preset tables are publication tables:
``tab_2_1_main_results`` has rows = methods and ``tab_2_4_dataset_descriptor``
needs a cohort descriptor, and a single training run has neither, so both
correctly return ``None``. Verified on a real run directory carrying 27 rows of
metrics: zero files written.

``tab_run_summary`` (:mod:`spectramr.infrastructure.reporting.tables.run_summary`)
is built for exactly this case. It consumes what the aggregator already recovers
from any run — ``training_metrics.csv``, ``validation_metrics.csv``,
``final_metrics.json`` — and emits one row per ``(split, metric)``:

.. code-block:: text

    split,metric,final,best,best_step,n,direction
    val,val_psnr,22.0,25.0,20,3,higher
    train,loss,0.5,0.3,20,3,lower

``best`` follows the metric-direction SSOT via ``resolve_direction`` — the
non-fatal resolver. An unrecognised metric name leaves ``best`` and ``direction``
**empty** rather than defaulting to max; defaulting to max is what made
``best_metric_name: lpips`` maximise LPIPS (#208). The table returns ``None``
rather than writing a header over no rows, on the same reasoning as
:ref:`flag-coverage-ssot`.

Enable the full pipeline with a ``reporting:`` block:

.. code-block:: yaml

    reporting:
      enabled: true
      task: reconstruction        # default | reconstruction | synthesis | super_resolution | gan | diffusion | vae | calibration
      style: nature               # nature (default) | ieee
      formats: [pdf, png]         # subset of {pdf, png, tiff, eps, svg}
      dpi: 600
      panel_labels: true
      submission_bundle: false    # emit submission/ with 600-dpi TIFF + captions
      tikz: false                 # also emit LaTeX-native TikZ/pgfplots figures
      emit_manifest: true
      n_report_cases: 6
      case_selection: best_median_worst

Each ``task`` selects a default figure/table preset (``pipeline.TASK_PRESETS``);
an unknown value raises rather than silently falling back to ``default``, and an
arm with no explicit ``figures:`` list relies entirely on its preset. The
**conformal-calibration / certification arms** (PAC-Bayes / pathology-recall
certificates, validation badges) use ``task: calibration`` — its preset grades a
*certificate* (coverage reliability, significance, agreement), not a
reconstruction. There is no separate ``certification`` task: it is the same
preset, so those arms set ``task: calibration``.

House style
-----------

``style: nature`` applies ``styles/nature.mplstyle`` — sans-serif (Helvetica/Arial),
7 pt base / 5 pt floor, no gridlines, bold lowercase panel letters, the colour-blind-safe
Okabe–Ito cycle, and vector + 600-dpi raster. Figure widths follow Nature columns
(89 mm single / 120 mm 1.5 / 183 mm double) via :func:`spectramr.infrastructure.reporting.style.column_width`.

Axis labels and panel titles resolve through
:func:`spectramr.infrastructure.reporting.style.pretty_label` — the same SSOT idea as
``METHOD_COLOURS``: ``psnr`` renders as "PSNR (dB)" and ``val_ssim`` as
"validation SSIM" in every figure; unknown keys degrade to lowercase words
(cosmetic fallback — labels are not registered-option enums, so pitfall #9
does not apply).

Every figure passes through
:func:`spectramr.infrastructure.reporting.style.save_figure`, which scrubs
non-renderable C0/C1 control characters from all text artists (via
:func:`spectramr.infrastructure.reporting.style.sanitize_figure_text`) immediately
before rasterising. This is defence-in-depth against a corrupt data-derived
label — e.g. an arm name carrying stray ``\x7f``/``\x80`` bytes flowing into
``suptitle`` — which otherwise makes ``fig.savefig`` emit a "Glyph N missing from
font(s) DejaVu Sans" ``UserWarning``. Control characters carry no display meaning,
so stripping them is a normalisation, not a silent fallback (pitfalls #9/#10).

Data sources
------------

:func:`spectramr.infrastructure.reporting.aggregator.aggregate` builds the tidy
long-format frame every plotter consumes. It merges, per run directory:

- ``logs/training_metrics.csv`` → ``split="train"`` rows (per-step)
- ``logs/validation_metrics.csv`` → ``split="val"`` rows (per-step)
- ``final_eval.json`` → ``split="test"`` rows (per-subject where available)
- ``final_metrics.json`` (the per-arm campaign contract) → ``split="best"``
  rows (``ssim_best`` → metric ``ssim``) and ``split="final"`` scalars
- ``run_summary.json`` (run root, or legacy ``logs/``) → ``split="run"`` facts:
  ``params_m`` (millions), ``iterations_per_sec``, ``duration_min``,
  ``effective_batch``

The last two were added 2026-07-01: without them the headline Pareto (cost axis
``params_m``), the computational profile, and the run-summary card could never
fire on a real run — the training loop writes ``final_metrics.json`` /
``run_summary.json``, not ``final_eval.json``. Missing artifacts contribute no
rows; a *present-but-corrupt* JSON logs a warning and is skipped.

TikZ / pgfplots export
----------------------

``tikz: true`` additionally emits LaTeX-native figures under
``report/figures/tikz/`` via :mod:`spectramr.infrastructure.reporting.tikz_export`:
per-metric training/validation curves (``curve_<metric>.tex``) and the headline
Pareto (``pareto_<metric>_vs_<cost>.tex``). Each figure is written twice —

- ``<stem>.tikz`` — the bare ``tikzpicture``, ``\input``-able inside a paper;
- ``<stem>.tex`` — a ``standalone`` wrapper, compilable with
  ``tectonic -X compile <stem>.tex``.

Emission is pure string generation: **no LaTeX is required at write time**
(matplotlib's PGF backend is deliberately not used — it shells out to a live
``latex`` process for text metrics). Methods keep their Okabe–Ito colours via
generated ``\definecolor`` lines, and TikZ artifacts are recorded in
``report_manifest.json`` like any other figure.

Report-case contract
---------------------

Qualitative figures need image data. When ``logging.report_cases.enabled`` is true, the
validation loop records best/median/worst cases (by the primary metric) to
``<run_dir>/report_cases/`` (``case_*.npz`` + ``cases_index.json``). ``generate_report``
auto-discovers them — no manual wiring needed.

Metric-key resolution: ``validation.primary_metric`` is a bare name (``psnr``), but the
feed seam stores the validation metrics dict verbatim, whose keys carry a monitor prefix
(``val_psnr``). :class:`~spectramr.infrastructure.reporting.cases.recorder.ReportCaseRecorder`
resolves the ranking key against the ``metric_directions`` monitor-prefix SSOT
(bare name, then each ``val_``/``train_``/… prefix), so ``best``/``median``/``worst`` rank
by the true metric. Before this fix an exact-match lookup silently missed the prefixed
keys, every case scored ``0.0``, and the labels collapsed to insertion order (a 14.5 dB
case was labelled ``best`` while a 17.2 dB case was labelled ``worst``).

Figure catalog
--------------

The ``PLOTTERS`` registry (:mod:`spectramr.infrastructure.reporting.plotters`) is the
authoritative figure set; ``list_available()`` lists every id. **By default the
report attempts the task preset** — ``TASK_PRESETS[<task>]["figures"]``, with
``contact_sheet`` rendered last since it composites the others. Pass an explicit
``figures`` list (or set ``reporting.figures`` in YAML) to override it.

Until issue #719 the code instead attempted *every* registered plotter, while
this page and ``ReportingSettings.figures`` ("None = preset default") both
promised the preset — three parties, and only the code said "all 44". The
practical cost was not wrong output (data-less plotters soft-skip) but that
``task:`` was **inert for figures**: every arm produced the same 44 render
attempts regardless of what it declared. Sizes now differ as intended —
``default`` 11, ``reconstruction`` 20, ``gan`` 7 (before the QC union below).

``reporting.qc_figures`` is **additive, not a filter**. Four presets (``gan``,
``diffusion``, ``vae``, ``calibration``) list no ``qc_*`` id at all, so once the
preset became authoritative a pure filter would have left the knob advertising
an inclusion it never performed (pitfall #15). With ``qc_figures: true`` the
three QC plotters are unioned into the preset; with ``false`` they are stripped
wherever they appear. An **explicit** ``figures`` list is never topped up —
explicit means explicit.

``TASK_PRESETS[<task>]`` also drives the table set and qualitative-case routing.

Each plotter is responsible for a **legible, self-describing** figure — labelled
axes, a units-bearing colorbar or legend, and a provenance stamp (git SHA) — and
returns ``None`` (soft-skip) rather than an empty or malformed canvas when its
input is absent. Layouts adapt to the data volume: e.g. the failure gallery packs
1-3 cases compactly instead of a mostly-empty wide strip. A plotter that renders
against representative data is exercised by
``tests/unit/infrastructure/reporting/test_plotters_smoke.py`` (the registry-driven
anti-facade guard), so a registered-but-silent figure fails CI.

*Core (Phase 1) — data-driven from the aggregator frame:*

===================================  =======================================================
ID                                   Shows
===================================  =======================================================
``fig_1_1_headline_pareto``          Quality vs cost (``params_m``) Pareto scatter
``fig_1_2_learning_curves``          Train/val loss vs step (log-y), ±1σ seed bands
``fig_1_3_loss_decomposition``       Per-component loss stack over steps
``fig_1_4_residual_diagnostics``     Residual QQ / histogram / vs-target
``fig_1_5_predicted_vs_true``        Predicted-vs-target calibration scatter
``fig_1_9_ablation_strip``           Per-ablation Δ vs the full model
``fig_1_11_stratified_performance``  Metric by subgroup (age / site / pathology)
``fig_1_12_failure_gallery``         Worst-case montage
``fig_1_15_computational_profile``   Params / throughput / wall-time / memory panels
``fig_1_16_run_summary_card``        Run facts + best-metrics card (triage at a glance)
``fig_1_17_metric_correlation``      Spearman ρ heatmap across validation metrics
``fig_1_18_train_val_gap``           Matched train/val curves with shaded generalization gap
===================================  =======================================================

*Generative-paradigm diagnostics:*

=============================  ===================================================
``gen_gan_diagnostics``        D/G loss balance, D-accuracy, gradient-penalty norm
``gen_vae_diagnostics``        KL per latent (posterior-collapse), recon vs KL
``gen_diffusion_diagnostics``  Loss-per-timestep, PSNR vs NFE / CFG scale
=============================  ===================================================

*MRI-specific (need recorded cases — see below):*

==========================  ==========================================================
``mri_recon_panel``         Cases × [input, prediction, target, error] with ROI insets
``kspace_error_spectrum``   Radial k-space error vs spatial frequency
``mri_a1_sr_triptych``      LR / SR / HR triptych (+ optional spectrum)
``mri_a2_synthesis_c2c``    Contrast-to-contrast synthesis panel
``mri_a7_kspace_recon``     Zero-filled / recon / reference + k-space
``mri_a11_cohort_table``    Cohort descriptor rendered as a figure
``mri_a12_fiducial_check``  Virtual-fiducial localization check
==========================  ==========================================================

*Publication statistics:*

========================  ===================================================
``metric_distribution``   Raincloud + paired-bootstrap significance brackets
``bland_altman``          Agreement (bias ± 1.96σ limits)
``significance_matrix``   Pairwise Holm–Bonferroni-corrected p-values
``calibration_coverage``  Reliability diagram (empirical vs nominal coverage)
``forest_plot``           Effect size Δ vs baseline ± bootstrap CI
``ablation_heatmap``      Knob × knob → metric heatmap
``acceleration_sweep``    Metric vs acceleration R with ±1σ seed bands
``contact_sheet``         Thumbnail montage / index of every other PNG
========================  ===================================================

*Novel-paradigm physics diagnostics (bespoke domain frames):*

==========================================  =================================================
``fig_2_15_active_acquisition_trajectory``  PSNR vs acquisition step, per acquisition policy
``fig_b3_bloch_consistency_residual``       Bloch-equation residual decay per tissue
``fig_b7_qmap_riemannian_vs_euclidean``     qMap error (Riemannian vs Euclidean geometry)
``fig_c1_beltrami_field``                   Beltrami coefficient magnitude + phase
``fig_c2_spd_geodesic``                     Geodesic distance trajectory on Sym⁺(n)
``fig_c3_teichmuller_schedule``             Teichmüller radial schedule r(t)
``fig_c4_fingerprint_embedding``            MRF fingerprint embedding scatter (clusters)
==========================================  =================================================

*Quality-control (QC) report:*

=====================  =============================================================
``qc_group_strip``     Group IQM distribution: one horizontal box + jittered strip
                       per metric/loss the run computed, Tukey-outlier (1.5·IQR)
                       points flagged and labelled by ``case_id``
``qc_subject_mosaic``  Per-case QC mosaic: prediction (foreground-mask contour),
                       target, error map, background-noise reportlet
``qc_carpet``          Cases × image-column error carpet + per-case spike/ghost strip
=====================  =============================================================

``qc_group_strip`` discovers its metric set from the data — preferring
``per_call_metrics.csv`` (one row per validation-case observation, emitted when
``reporting.per_call_metrics`` is on), then the recorded-case ``predictions_df``,
then the aggregate frame — so it tracks whatever metrics/losses the run computed
rather than a fixed IQM list. The mosaic and carpet consume the recorded
``report_cases`` arrays (or, in the standalone ``spectramr report`` path, the
downloaded ``real_images``/``fake_images`` PNG pairs). All three are bundled into
the self-contained ``qc_report.html`` when ``reporting.html_report`` is on
(interactive group plots when :mod:`plotly` is installed, static-PNG fallback
otherwise). Gated per-run by ``reporting.qc_figures``.

To report a whole cohort at once,
:func:`spectramr.infrastructure.reporting.generate_reports` (CLI:
``spectramr report --exp-dir <root> --recursive``) discovers every run beneath a
root via :func:`spectramr.infrastructure.reporting.discover_run_dirs`, reports each,
and writes a top-level ``report_index.html`` linking them.

Cohort task-ablation bar figures
--------------------------------

Where ``--recursive`` emits one report *per arm*, the ``cohort_task_ablation_bars``
plotter emits one figure *per task* that places every arm side by side —
purpose-built for a multi-task, method-vs-ablation cohort like ``mrixfields2026``.
Each figure is a vertical stack of sub-panels (one validation metric each; SSIM /
PSNR / LPIPS by default, since the scales differ), and within every panel the bars
are grouped by **method family** (``bNN``) with the full method(s) drawn first and
their one-knob **ablation** control(s) immediately after, coloured by role. The
per-arm scalar is the **best-over-iterations** validation value (max, or min for
``lower_is_better`` metrics), matching each run's ``restore_best_weights``
early-stopping semantics. Degenerate / collapsed arms (negative PSNR, near-zero
SSIM, identity collapse) are shown honestly with their real value and hatched —
never hidden.

Assembly lives in :mod:`spectramr.infrastructure.reporting.cohort_ablation`:
:func:`~spectramr.infrastructure.reporting.cohort_ablation.discover_cohort_arms`
derives ``task`` / ``family`` / ``role`` self-contained from each arm's
``resolved_config.json`` (``metadata.tags.task``, the ``bNN`` token, the
``_ablate_`` suffix);
:func:`~spectramr.infrastructure.reporting.cohort_ablation.best_val_metrics` reuses
the :func:`~spectramr.infrastructure.reporting.aggregator.aggregate` SSOT reader to
pull best validation scalars; and
:func:`~spectramr.infrastructure.reporting.cohort_ablation.generate_task_ablation_figures`
dispatches the registered plotter once per task. The plotter itself is generic —
it consumes a tidy ``[family, role, arm, metric, value]`` frame and returns
``None`` (soft-skip) on the standard aggregator frame, so it never disturbs the
default ``spectramr report`` "plot all" pass. Across a whole cohort, drive it
with ``spectramr report --exp-dir <cohort-root> --recursive``, which walks every
experiment directory beneath the root::

    spectramr report --exp-dir <cohort-root> --recursive

The driver never calls ``savefig`` — saving happens inside the registered plotter
via :func:`~spectramr.infrastructure.reporting.style.save_figure`, so the C1
plotting-SSOT ratchet holds.

Interactive layer (plotly)
--------------------------

When ``reporting.interactive`` is on (default) **and** :mod:`plotly` is installed,
``qc_report.html`` gains an interactive layer *beside* the static PNGs — the PNGs
stay as the print/archival copy and the offline fallback. The builders live in
:mod:`spectramr.infrastructure.reporting.interactive`; each returns an inline
``<div>`` or ``None`` (plotly absent or no data), so the section degrades cleanly
to the embedded PNG. Every interactive panel carries a short **"How to read this"**
caption for explainability.

*Self-contained & offline.* plotly.js is inlined **once** at the top of the report
(:func:`interactive.plotly_util.plotlyjs_script`); every div is emitted with
``include_plotlyjs=False``. The report therefore renders fully offline — there is
**no CDN fetch** (the previous ``include_plotlyjs="cdn"`` path is gone), matching
the base64-PNG self-containment.

Interactive figures (:mod:`interactive.figures`):

- **Group IQM** — hoverable box+strip of every metric at once; hover shows the
  ``case_id``; the box whiskers delimit the Tukey (1.5·IQR) fences; axis labels
  carry ↑/↓ "better" arrows.
- **Training dynamics** — train (solid) vs validation (dotted) curves with a
  metric dropdown and a linear/log-y toggle; legend-click isolates a method.
- **Per-case metric distribution** — a violin+box+points raincloud with a metric
  dropdown, split by method when several are present.
- **Method agreement** — Bland-Altman between the two best-represented methods
  (bias ± 1.96σ), metric dropdown; soft-skips with fewer than two methods.

MRIQC-style slice viewers (:mod:`interactive.viewer_2d`, :mod:`interactive.viewer_3d`):

- **2-D per-subject viewer** — a slider scrubs the recorded subjects
  (best→median→worst) while layer buttons flick the panel between Prediction,
  Target and \|Error\| (the before/after reportlet). The slider drives plotly
  *frames* (swapping the heatmap ``z``) and the buttons drive trace *visibility*,
  so the two controls are orthogonal. Always available (2-D cases).
- **3-D volume viewer** — the same two-control pattern over the **Z slices** of a
  single volume (the MRIQC mosaic navigation). It renders only when a case carries
  a 3-D ``*_volume`` array; on single-slice acquisitions (e.g. M4Raw) it
  **soft-skips with an explicit note** — volumes are never fabricated from
  unrelated slices.

*Recording volumes.* The 3-D viewer needs 3-D data. Set
``reporting.record_volumes`` (opt-in, default off) to make the validation feeder
preserve a ``[Z, H, W]`` stack (``input_volume``/``prediction_volume``/
``target_volume``) alongside the 2-D representative. It fires only for an
unambiguous slice axis — a 5-D ``[B, C, Z, H, W]`` tensor (coils RSS-collapsed,
slices kept) or a 3-D ``[Z, H, W]`` volume; a 4-D ``[B, C, H, W]`` (a 2-D slice
with coils) records **no** volume. On single-slice data the knob is a no-op.
Set ``reporting.interactive: false`` (or ``spectramr report --no-interactive``) to
force the static-only report.

The three ``fig_1_16``–``fig_1_18`` figures (2026-07-01) are data-driven from the
aggregator alone — no case recording needed. ``fig_1_17`` reads an absolute
Spearman ρ near 1 off-diagonal as "these two metrics are redundant";
``fig_1_18`` matches metrics
across splits by base name (``train_ssim``/``ssim`` ↔ ``val_ssim``) and shades
the gap band — widening = overfitting, validation *below* train from step one =
non-exchangeable splits (leakage / site shift).

Direct-call renderers (not registry-dispatched)
-----------------------------------------------

Four renderers deliberately sit **outside** the ``PLOTTERS``/``TABLES`` registries
because their signatures don't match the ``make(df, out_path, **kw)`` contract
(``dispatch`` always calls ``fn(df, out_path, ...)``, so registering them would
crash it). They are called directly from the certification / acquisition paths:

- :func:`plotters.certificate_summary.render_certificate_summary` — R1/R2/R4/R5 +
  validation-badge 2×3 grid, from the conformal-calibration JSON artefacts.
- :func:`plotters.ksd_certificate.render_ksd_certificate` — KSD goodness-of-fit
  one-page PDF, from a ``KSDDefensibilityResult``.
- :func:`plotters.learnable_acquisition_pareto.render_learnable_acquisition_pareto`
  — thin wrapper over ``headline_pareto`` taking an ``ArmResult`` list.
- :func:`tables.certificate_table.write_certificate_table` — per-certificate
  LaTeX + Markdown table from a ValidationBadge payload.

Full synthetic gallery
----------------------

To render **every** registered plotter + table + the direct-call renderers + the
TikZ bundle on synthetic fixtures (browsable in one ``GALLERY.md``)::

the maintainers run a gallery generator over synthetic fixtures. That generator
and its fixtures are development scaffolding and are not distributed; what *is*
distributed is the guarantee it was built to check. The anti-facade
guard that every registered figure actually fires on well-formed input is enforced
by ``tests/unit/infrastructure/reporting/test_plotters_smoke.py``
(``test_every_registered_plotter_renders`` sweeps ``list_available()``;
``test_registry_fixture_map_covers_every_registered_id`` fails if a newly
registered figure has no coverage fixture).

Empty-input contract
--------------------

Every plotter ``make`` and table ``make`` is a **total function on bad input**:
given an empty :class:`~pandas.DataFrame`, a frame missing the ``metric`` column,
or one lacking the requested metric/cost rows, it returns ``None`` (a "no data"
signal) rather than raising. The driver (:func:`pipeline.generate_report` via
``plotters.dispatch`` / ``tables.dispatch``) additionally soft-fails each
component in a ``try/except`` so a single bad figure cannot abort a long run's
wrap-up. The guard must precede any ``df["metric"]`` access — a guard placed
*after* the subscript raises ``KeyError: 'metric'`` on the empty frame before it
can return ``None`` (regression covered by
``tests/unit/reporting/test_reporting_plotters.py`` for ``headline_pareto``,
``ablation_strip``, and ``ablation_table``).

.. _per-case-metrics-are-per-call:

``per_call_metrics.csv`` is per *call*, not per sample
-------------------------------------------------------

The name overpromises, and a reader who trusts it draws a conclusion the data
cannot support. Each row is **one validation call** carrying the **batch-aggregate**
metrics, tagged with a ``case_id`` that
:func:`~spectramr.infrastructure.training.strategies.mixins.metrics_mixin.report_case_id`
derives from the training step (plus the cascade rung, when there is one). It is not
one row per sample, because the run computes metrics batch-wise and no per-sample
value exists to record.

That mismatch produced a real false result (#503). On ``exp_vf_01`` the file held
eight rows whose ``psnr`` was the same number — ``48.71364215`` — and
``qc_group_strip`` drew them as a box-and-whisker. The panel asserted a tight
agreement across eight cases that was one scalar repeated eight times, while its
siblings on the same report (``fig_1_5_predicted_vs_true``, ``mri_a1_sr_triptych``,
``mri_a11_cohort_table``) honestly reported *"skipped (no data)"*.

Two guards now stand between that data and a chart:

#. **No distribution without spread.** ``qc_group_strip`` drops any metric whose
   values have zero range, and returns ``None`` (soft-skip, matching its siblings)
   when nothing survives. The drop is **logged with the metric names** — a silently
   missing panel is indistinguishable from a broken plotter.
#. **No fabricated quartiles.** Below four observations the box is omitted rather
   than collapsed to ``q1 = med = q3``. The old code drew it anyway at
   ``max(q3 - q1, 1e-9)`` wide, and that clamp is the subtle half of the bug: a
   zero-width box would have been invisible, whereas a 1e-9 sliver renders as an
   extremely *tight* distribution — the most confident claim on the chart, from the
   least data. The median tick and the min-max span still draw, because those are
   things the data does say.

Each row now also carries a **context block** — identity columns written between
``step`` and the metrics, in the fixed order declared by
:data:`~spectramr.infrastructure.reporting.cases.metric_sink.CONTEXT_COLUMNS`:

========================= ======================================================
``acceleration_level``    The cascade rung this pass evaluated (2, 8, 32, …)
``acceleration_realized`` What the schedule decoded that timestep back to
``timestep``              The diffusion timestep the rung actually ran at
``heldout``               ``True`` for an out-of-distribution severity point
``contrast``              ``T1`` / ``T2`` / ``FLAIR`` — see the mixed-batch note
``file_id``               The source file stem(s) the row averaged over
``batch_index``           Position of the batch within the validation loop
``batch_size``            How many volumes the row's metrics are a mean over
========================= ======================================================

The rung was previously encoded *only* inside ``case_id``, which is why a
45-batch × 3-rung sweep wrote 135 rows under three distinct labels: every
number was present, and nothing said which volume or which acceleration
produced any one of them. ``case_id`` keeps its spelling — PNG filenames and
downstream parsers are built on it — and the context columns are additive.

Two honesty constraints hold in that block. A ``None`` value is **omitted**
rather than written as ``0``, because ``0`` is a real timestep (the clean end of
the diffusion schedule) and a plausible wrong value is worse than a visible gap.
``acceleration_realized`` is blank for exactly this reason on the linear-fallback
path, where there is no schedule to invert: an empty cell there means *nothing
could decode it*, never *it matched the request*.
And because the validation loader shuffles, one batch can hold a T1 volume and a
FLAIR one — ``contrast`` then reports **every** contrast present, joined with
``|``, rather than picking one and attributing the whole mean to it.

That block reaches the CSV through one call: ``feed_report_case_recorder``
passes ``context=`` to whatever sink it is handed, **unconditionally** -- not only
when the mapping is non-empty. Accepting the keyword is therefore part of the
metric-sink contract, and nothing in the type system says so: ``sink`` is an
untyped parameter defaulting to ``None``, so a sink that omits it imports, lints
and type-checks clean, then raises ``TypeError`` the first time validation runs --
hours into a training job. ``test_every_metric_sink_in_src_accepts_the_context_kwarg``
is the ratchet for that. It discriminates by *shape* rather than by the name
``observe``, because three unrelated ``observe`` protocols live in this tree -- the
metric sink (``split`` / ``step``), the image recorder (``arrays`` / ``domain``)
and the inference-artifact writer (``prediction`` / ``target``) -- and only the
first is ever handed a context.

**Still open (narrowed):** metrics are still computed batch-wise, so a row is a
mean over ``batch_size`` volumes rather than a per-sample measurement — the
artifact keeps its name for that reason, and renaming it would ratify batch
aggregates as the intended content. What changed is that the row now *says* what
it aggregated over. Setting ``validation.loader.batch_size: 1`` makes each row
exactly one volume, which is the per-sample granularity without per-sample
metric computation.

.. _validation-cascade-levels:

The validation ladder is declared, not fixed
--------------------------------------------

``acceleration_level`` comes from the cascade ladder, which is a config knob:

.. code-block:: yaml

   validation:
     cascade:
       levels: [2, 4, 8]      # omit for the framework default (2, 8, 32)

It used to be a module constant in
:mod:`spectramr.core.cascading_validation`, so an arm could widen
``undersampling.acceleration_range`` for *training* while *validation* stayed
pinned at 2/8/32 with no spelling to say otherwise — the two halves of the loop
disagreed by construction. Both the strategy and the
``validation_cascade_levels_in_range`` audit check now read one resolver,
:func:`~spectramr.core.cascading_validation.resolve_cascade_levels`; the check
previously held its own copy of the ladder beside a docstring conceding the two
"should be updated in lockstep".

Levels are deduplicated and sorted ascending (the accel-gap readout subtracts
the last rung from the first, so "first rung is the mildest" is an invariant of
the consumers, not of the author's typing order). An integral rung stays an
``int`` so the flat ``val_<metric>_<R>x`` names keep their spelling —
``val_psnr_2x``, not ``val_psnr_2.0x`` — which matters because the L4
input-dependence gate and the accel-gap stamp look those names up and do not
raise on a miss. An empty ladder, a sub-1x rung, a non-finite value or a boolean
are **refused at load time** rather than silently repaired.

Under ``undersampling.schedule_type: step`` a rung that is not in
``undersampling.acceleration_range`` has no timestep inverse and is skipped at
runtime; ``spectramr audit`` warns about that before the launch.

Artifacts
---------

``report_manifest.json`` records every emitted artifact (id, path, format, sha256) plus
provenance (git commit, seed, dataset, timestamp). ``submission_bundle: true`` additionally
writes ``submission/`` with a 600-dpi TIFF and a caption ``.txt`` per figure.

Plotting SSOT and the ``report`` command
----------------------------------------

The reporting pipeline is the **single source of truth for figures**: every plot is
produced by a registered plotter under ``plotters/`` (plus the plotly ``interactive/``
layer), styled through ``style.py``, and orchestrated by :func:`pipeline.generate_report`.
The only sanctioned exceptions are the sim2rank meta-evaluation figures under
``core/metrics/meta_evaluation/`` and dataset EDA under ``data/eda/`` — reached by the
separate ``spectramr meta-evaluate`` / EDA paths, never by ``spectramr report``. A ratchet test,
``tests/unit/infrastructure/reporting/test_plotting_ssot_guard.py``, fails on any new
``savefig`` / ``write_html`` / ``write_image`` / ``plt.show`` added outside
``infrastructure/reporting/`` (with a frozen baseline of pre-existing emitters).

``spectramr report`` renders **every applicable figure by default** —
:func:`pipeline.generate_report` requests all registered plotters and each soft-skips
(returns ``None``) when the run lacks its data. Pass ``--figures a,b,c`` to opt into a
subset. Unlike the end-of-training hook, the ``report`` command deliberately does *not*
inherit a ``reporting.figures`` subset from ``--config`` (that would silently narrow the
SSOT report).

Domain-artifact convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Physics / geometry figures whose data the aggregator frame cannot carry
(``fig_2_15_active_acquisition_trajectory``, ``fig_b3_bloch_consistency_residual``,
``fig_b7_qmap_riemannian_vs_euclidean``, ``fig_c1_beltrami_field`` .. ``fig_c4``) are fed
from ``<run>/report_artifacts/`` when present: ``<fig_id>.csv`` becomes that plotter's
DataFrame; ``active_acquisition/*.csv`` → ``csv_paths``; ``qmap_slices.npz`` → ``slices``.
Absent artifacts leave the figure to soft-skip. ``mri_a12_fiducial_check`` is fed the
recorded image cases like the other case plotters. The certificate figures
(``certificate_summary``, ``ksd_certificate``, ``learnable_acquisition_pareto``) are
registered via soft-skipping adapters and render when their certificate / result / arms
payload is routed.

Validation loss in the learning curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ModelValidationMixin.validation_step`` emits a ``val_loss`` (a magnitude-matched L1 on
the validation prediction/target) alongside the image metrics, so
``fig_1_2_learning_curves`` (and its interactive twin) draw a validation-loss curve rather
than a train-only series. The interactive builder strips the ``val_`` prefix so a
validation series pairs with its training twin on one panel (solid = train, dotted = val),
and the aggregator drops entirely-empty metric columns so a phantom pre-seeded ``val_loss``
header never renders.
