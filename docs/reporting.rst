Reporting: figures, tables and manifests
========================================

.. contents:: On this page
   :local:
   :depth: 2

The reporting pipeline (:mod:`spectramr.infrastructure.reporting`) turns a
finished run into publication-grade figures, tables and a provenance manifest
under ``<run_dir>/report/``. It runs at the end of training when the experiment
configures it, and on demand through ``spectramr report``.

What a report contains
----------------------

.. code-block:: text

    <experiment_dir>/report/
    |-- figures/
    |   |-- fig_1_2_learning_curves.pdf
    |   |-- fig_1_3_loss_decomposition.pdf
    |   |-- fig_1_15_computational_profile.pdf
    |   |-- mri_a11_cohort_table.pdf
    |   `-- *.meta.json                        # per-figure provenance sidecars
    |-- tables/
    |   |-- tab_2_1_main_results.{md,tex}
    |   `-- tab_2_4_dataset_descriptor.{md,tex}
    |-- qc_report.html
    |-- report_manifest.json
    `-- report_summary.md                      # human-readable index

Every figure stamps ``git=<sha>  .  seed=<n>  .  data=<version>`` in the
bottom-right corner and writes a sidecar ``*.meta.json``, so "how was this number
computed?" is answerable in one lookup.

.. _reporting-single-entry-point:

One entry point, two tiers
--------------------------

:func:`~spectramr.infrastructure.reporting.generate_report` is the only
end-of-training report path, reached from the training pipeline's wrap-up hook
(:func:`spectramr.infrastructure.reporting.run_hook.maybe_run_reporting`). What a
run gets depends on whether it configured reporting:

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

The unconfigured floor is a table of its own rather than "the ``default``
preset's tables without the figures", because that combination emits nothing.
Both preset tables are publication tables — ``tab_2_1_main_results`` has
rows = methods, and ``tab_2_4_dataset_descriptor`` needs a cohort descriptor —
and a single training run has neither, so both correctly return ``None``.

``tab_run_summary``
(:mod:`spectramr.infrastructure.reporting.tables.run_summary`) is built for
exactly this case. It consumes what the aggregator already recovers from any run
— ``training_metrics.csv``, ``validation_metrics.csv``, ``final_metrics.json`` —
and emits one row per ``(split, metric)``:

.. code-block:: text

    split,metric,final,best,best_step,n,direction
    val,val_psnr,22.0,25.0,20,3,higher
    train,loss,0.5,0.3,20,3,lower

``best`` follows the metric-direction SSOT through ``resolve_direction``, the
non-fatal resolver. An unrecognised metric name leaves ``best`` and ``direction``
**empty** rather than defaulting to "higher is better" — that default would
report the *worst* value as best for any lower-is-better metric, so a run with
``best_metric_name: lpips`` would be scored upside down. The table returns
``None`` rather than writing a header over no rows, on the same reasoning as
:ref:`flag-coverage-ssot`.

Enabling reporting
------------------

Add a ``reporting:`` block to the experiment YAML. Every key below is a field of
``ReportingSettings``, which **rejects unknown keys at load time** — a
misspelled key is an error, never a silently ignored one.

.. code-block:: yaml

    reporting:
      enabled: true
      task: reconstruction        # default | reconstruction | synthesis | super_resolution
                                  # | gan | diffusion | vae | calibration
      method_name: my_baseline    # label used in plots and tables
      out_subdir: report          # subdirectory of <experiment_dir> to write into

      # House style and output formats
      style: nature               # nature (default) | ieee
      formats: [pdf, png]         # subset of {pdf, png, tiff, eps, svg}
      dpi: 600
      panel_labels: true
      submission_bundle: false    # also emit submission/ (600-dpi TIFF + captions)
      tikz: false                 # also emit LaTeX-native TikZ/pgfplots figures
      emit_manifest: true

      # Qualitative cases
      n_report_cases: 6
      case_selection: best_median_worst   # | random | first
      record_volumes: false       # keep [Z, H, W] stacks for the 3-D viewer

      # Optional layers (all default on)
      qc_figures: true            # union the QC plotters into the preset
      html_report: true           # assemble qc_report.html
      interactive: true           # plotly layer inside qc_report.html
      per_call_metrics: true      # emit <run_dir>/per_call_metrics.csv

      # Override the task preset (omit, or null, to use the preset)
      figures:
        - fig_1_2_learning_curves
        - fig_1_3_loss_decomposition
        - fig_1_5_predicted_vs_true
        - mri_a7_kspace_recon
        - mri_a11_cohort_table
      tables:
        - tab_2_1_main_results
        - tab_2_4_dataset_descriptor

      # Metrics for the main-results table (tab_2_1). Names resolve against the
      # metrics registry or the report-time IQMs listed further down.
      metrics: [psnr, ssim, ms_ssim, lpips, vif, fsim, haarpsi, gmsd, rmse, nmse, hfen]

      # Optional cohort descriptor for the dataset table and cohort figure
      cohort:
        n_total: 100
        train_n: 60
        val_n: 20
        test_n: 20
        age_mean: 42.5
        age_std: 11.2
        sex_split: 52F/48M
        scanners: Siemens 3T
        field_strength: 3T
        pathology: healthy
        split_rule: subject-level (no slice leakage)

      # Optional hyperparameter table rows
      hyperparameters:
        - name: learning_rate
          distribution: log-uniform
          range: 1e-5..1e-3
          final: 1e-4
          criterion: best val_psnr

      # Optional baseline runs folded into the same tables
      extra_runs:
        - experiments/outputs/baseline_v1

      fail_on_error: false        # true: a plotter exception aborts the wrap-up

Every key is optional, ``enabled`` included — it defaults to ``false``. Without
``enabled: true`` the run gets the unconfigured floor described above, and an arm
with no ``reporting:`` block at all gets the same.

Schema field reference
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 20 18 40

   * - Key
     - Type
     - Default
     - Meaning
   * - ``enabled``
     - bool
     - ``false``
     - Master switch. ``false`` makes the end-of-training hook a no-op beyond the
       unconfigured floor.
   * - ``task``
     - enum
     - ``default``
     - Preset that selects the default figure and table set. One of ``default``,
       ``reconstruction``, ``synthesis``, ``super_resolution``, ``gan``,
       ``diffusion``, ``vae``, ``calibration``.
   * - ``method_name``
     - str
     - experiment dir name
     - Label this run carries in every plot legend and table row.
   * - ``out_subdir``
     - str
     - ``report``
     - Subdirectory of the experiment output dir to write into.
   * - ``style``
     - enum
     - ``nature``
     - House style: ``nature`` or ``ieee``.
   * - ``formats``
     - list[str]
     - ``[pdf, png]``
     - Figure output formats, a subset of ``{pdf, png, tiff, eps, svg}``.
   * - ``dpi``
     - int
     - ``600``
     - Raster DPI for ``png`` / ``tiff`` outputs. Accepts 72-1200.
   * - ``panel_labels``
     - bool
     - ``true``
     - Stamp bold lowercase panel letters (a, b, c).
   * - ``submission_bundle``
     - bool
     - ``false``
     - Also write ``submission/`` with a 600-dpi TIFF and a caption ``.txt`` per
       figure.
   * - ``tikz``
     - bool
     - ``false``
     - Also emit LaTeX-native TikZ/pgfplots figures.
   * - ``emit_manifest``
     - bool
     - ``true``
     - Write ``report_manifest.json`` (artifact index plus provenance).
   * - ``n_report_cases``
     - int
     - ``6``
     - How many qualitative cases to record and render.
   * - ``case_selection``
     - enum
     - ``best_median_worst``
     - Which validation cases the recorder keeps: ``best_median_worst``,
       ``random`` or ``first``.
   * - ``record_volumes``
     - bool
     - ``false``
     - Preserve full 3-D ``[Z, H, W]`` volumes for the interactive volumetric
       viewer. A no-op on single-slice acquisitions.
   * - ``qc_figures``
     - bool
     - ``true``
     - Union the QC plotters into the figure set. Additive, not a filter — see
       below.
   * - ``html_report``
     - bool
     - ``true``
     - Assemble the self-contained ``qc_report.html``.
   * - ``interactive``
     - bool
     - ``true``
     - Emit the plotly layer inside the HTML report. Falls back to static PNGs
       when plotly is not installed.
   * - ``per_call_metrics``
     - bool
     - ``true``
     - Emit ``<run_dir>/per_call_metrics.csv``. Read
       :ref:`what a row of that file is <per-case-metrics-are-per-call>` before
       plotting it.
   * - ``figures``
     - list[str]
     - preset
     - Explicit figure ids, overriding the task preset.
   * - ``tables``
     - list[str]
     - preset
     - Explicit table ids, overriding the task preset.
   * - ``metrics``
     - list[str]
     - auto-detect
     - Metrics the main-results table includes. Names must resolve in the
       metrics registry or among the report-time IQMs.
   * - ``cohort``
     - dict
     - unset
     - Free-form cohort descriptor consumed by ``tab_2_4_dataset_descriptor``
       and ``mri_a11_cohort_table``.
   * - ``hyperparameters``
     - list[dict]
     - unset
     - Rows for ``tab_2_3_hyperparameters``; each carries ``name``,
       ``distribution``, ``range``, ``final`` and ``criterion``.
   * - ``extra_runs``
     - list[str]
     - unset
     - Other experiment directories to fold into the same tables, e.g.
       baselines.
   * - ``fail_on_error``
     - bool
     - ``false``
     - When ``false``, a plotter exception logs a warning and the rest of the
       report still generates. Set ``true`` in CI.

Each ``task`` selects a default figure and table preset
(``pipeline.TASK_PRESETS``); an unknown value raises rather than silently
falling back to ``default``, and an arm with no explicit ``figures:`` list relies
entirely on its preset. The **conformal-calibration / certification arms**
(PAC-Bayes and pathology-recall certificates, validation badges) use
``task: calibration`` — its preset grades a *certificate* (coverage reliability,
significance, agreement), not a reconstruction. There is no separate
``certification`` task: it is the same preset, so those arms set
``task: calibration``.

Generating a report by hand
---------------------------

``spectramr report`` runs the same orchestrator as the end-of-training hook
against an existing output directory:

.. code-block:: bash

    spectramr report \
      --exp-dir experiments/outputs/my_baseline \
      --task reconstruction \
      --method my_baseline \
      --cohort-json my_cohort.json \
      --seed 42 \
      --dataset-version m4raw_v1

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Effect
   * - ``--exp-dir`` / ``-e``
     - Required. The experiment output directory, or a cohort root with
       ``--recursive``.
   * - ``--recursive`` / ``-r``
     - Treat ``--exp-dir`` as a cohort root: report every run beneath it and
       write a linking ``report_index.html``.
   * - ``--config`` / ``-c``
     - Reuse a config's ``reporting:`` block (style, formats, dpi, tables,
       ``qc_figures``, ``html_report``) for parity with the training hook.
       Explicit flags win over it.
   * - ``--task``
     - Task preset. Defaults to the config's ``reporting.task``, else
       ``default``.
   * - ``--method``
     - Method label for plots.
   * - ``--figures``
     - Comma-separated figure ids to render, e.g.
       ``--figures fig_1_2_learning_curves,qc_group_strip``.
   * - ``--out-subdir``
     - Subdirectory under ``--exp-dir`` to write into.
   * - ``--seed``, ``--dataset-version``
     - Values stamped onto figure metadata.
   * - ``--cohort-json``
     - A JSON file of cohort fields (``train_n``, ``val_n``, ``age_mean``, ...).
   * - ``--html`` / ``--no-html``
     - Emit or skip the self-contained QC HTML report.
   * - ``--interactive`` / ``--no-interactive``
     - Emit or skip the interactive plotly layer.
   * - ``--qc`` / ``--no-qc``
     - Include or skip the QC figures.

``--config`` reuses that YAML's ``reporting:`` block — style, formats, dpi,
panel labels, method name, tables, metrics, hyperparameters, extra runs, output
subdirectory, manifest, submission bundle, TikZ, QC, HTML and cohort — so a
hand-run report matches what the end-of-training hook would have written.
Explicit CLI flags override it.

``figures`` is the one key ``--config`` deliberately does **not** inherit,
because the ``report`` command is the plotting SSOT: it renders the task preset
(see :ref:`reporting-figure-selection`) unless you narrow it explicitly with
``--figures``. The end-of-training hook still honours ``reporting.figures``
per run.

House style
-----------

``style: nature`` applies ``styles/nature.mplstyle`` — sans-serif
(Helvetica/Arial), 7 pt base with a 5 pt floor, no gridlines, bold lowercase
panel letters, the colour-blind-safe Okabe-Ito cycle, and vector plus 600-dpi
raster output. Figure widths follow Nature columns (89 mm single, 120 mm 1.5,
183 mm double) via
:func:`spectramr.infrastructure.reporting.style.column_width`.

Axis labels and panel titles resolve through
:func:`spectramr.infrastructure.reporting.style.pretty_label`, the same SSOT idea
as ``METHOD_COLOURS``: ``psnr`` renders as "PSNR (dB)" and ``val_ssim`` as
"validation SSIM" in every figure. Unknown keys degrade to lowercase words; this
is a cosmetic fallback, since labels are not a registered-option enum.

Every figure passes through
:func:`spectramr.infrastructure.reporting.style.save_figure`, which scrubs
non-renderable C0/C1 control characters from all text artists (via
:func:`spectramr.infrastructure.reporting.style.sanitize_figure_text`)
immediately before rasterising. This is defence in depth against a corrupt
data-derived label — an arm name carrying stray ``\x7f`` / ``\x80`` bytes flowing
into ``suptitle`` — which otherwise makes ``fig.savefig`` emit a "Glyph N missing
from font(s) DejaVu Sans" ``UserWarning``. Control characters carry no display
meaning, so stripping them is a normalisation rather than a silent fallback.

Data sources
------------

:func:`spectramr.infrastructure.reporting.aggregator.aggregate` builds the tidy
long-format frame every plotter consumes. It merges, per run directory:

- ``logs/training_metrics.csv`` -> ``split="train"`` rows (per-step)
- ``logs/validation_metrics.csv`` -> ``split="val"`` rows (per-step)
- ``final_eval.json`` -> ``split="test"`` rows (per-subject where available)
- ``final_metrics.json`` (the per-arm campaign contract) -> ``split="best"``
  rows (``ssim_best`` becomes metric ``ssim``) and ``split="final"`` scalars
- ``run_summary.json`` (run root, or ``logs/``) -> ``split="run"`` facts:
  ``params_m`` (millions), ``iterations_per_sec``, ``duration_min``,
  ``effective_batch``

The last two matter because the training loop writes ``final_metrics.json`` and
``run_summary.json``, not ``final_eval.json``: without them the headline Pareto's
cost axis (``params_m``), the computational profile and the run-summary card have
no data to fire on. Missing artifacts contribute no rows; a *present-but-corrupt*
JSON logs a warning and is skipped.

Report-case contract
--------------------

Qualitative figures need image data. When ``logging.report_cases.enabled`` is
true, the validation loop records best/median/worst cases (by the primary metric)
to ``<run_dir>/report_cases/`` (``case_*.npz`` plus ``cases_index.json``).
``generate_report`` auto-discovers them — no manual wiring needed.

Metric-key resolution matters here. ``validation.primary_metric`` is a bare name
(``psnr``), but the feed seam stores the validation metrics dict verbatim, and
its keys carry a monitor prefix (``val_psnr``).
:class:`~spectramr.infrastructure.reporting.cases.recorder.ReportCaseRecorder`
resolves the ranking key against the ``metric_directions`` monitor-prefix SSOT —
bare name first, then each ``val_`` / ``train_`` prefix — so ``best``, ``median``
and ``worst`` rank by the true metric rather than by insertion order.

.. _reporting-figure-selection:

Figure catalog
--------------

The ``PLOTTERS`` registry
(:mod:`spectramr.infrastructure.reporting.plotters`) is the authoritative figure
set; ``list_available()`` lists every id. **By default the report renders the
task preset** — ``TASK_PRESETS[<task>]["figures"]``, with ``contact_sheet``
rendered last since it composites the others. Presets differ in size and
membership, which is what makes ``task:`` meaningful. Pass an explicit
figure list to override the preset entirely: ``--figures`` on the command line,
or ``reporting.figures`` in the YAML the end-of-training hook reads.
``spectramr report --config`` does not inherit ``reporting.figures``.

``reporting.qc_figures`` is **additive, not a filter**. Several presets list no
``qc_*`` id at all, so a pure filter would leave the knob advertising an
inclusion it never performed. With ``qc_figures: true`` the three QC plotters are
unioned into the preset; with ``false`` they are stripped wherever they appear.
An **explicit** ``figures`` list is never topped up — explicit means explicit.

``TASK_PRESETS[<task>]`` also drives the table set and qualitative-case routing.

Each plotter is responsible for a **legible, self-describing** figure — labelled
axes, a units-bearing colorbar or legend, and a provenance stamp — and returns
``None`` (soft-skip) rather than an empty or malformed canvas when its input is
absent. Layouts adapt to the data volume: the failure gallery, for instance,
packs 1-3 cases compactly instead of stretching a mostly-empty wide strip. That
every registered figure actually renders against representative data is enforced
by ``tests/unit/infrastructure/reporting/test_plotters_smoke.py``:
``test_every_registered_plotter_renders`` sweeps ``list_available()``, and
``test_registry_fixture_map_covers_every_registered_id`` fails if a newly
registered figure has no coverage fixture. A registered-but-silent figure
therefore fails CI.

*Core — data-driven from the aggregator frame:*

===================================  =======================================================
ID                                   Shows
===================================  =======================================================
``fig_1_1_headline_pareto``          Quality vs cost (``params_m``) Pareto scatter
``fig_1_2_learning_curves``          Train/val loss vs step (log-y), +/-1 sigma seed bands
``fig_1_3_loss_decomposition``       Per-component loss stack over steps
``fig_1_4_residual_diagnostics``     Residual QQ / histogram / vs-target
``fig_1_5_predicted_vs_true``        Predicted-vs-target calibration scatter
``fig_1_9_ablation_strip``           Per-ablation delta vs the full model, 95% CI
``fig_1_11_stratified_performance``  Metric by subgroup (age / site / pathology), with n
``fig_1_12_failure_gallery``         Worst-case montage
``fig_1_15_computational_profile``   Params / throughput / wall-time / memory panels
``fig_1_16_run_summary_card``        Run facts + best-metrics card (triage at a glance)
``fig_1_17_metric_correlation``      Spearman rho heatmap across validation metrics
``fig_1_18_train_val_gap``           Matched train/val curves, shaded generalization gap
===================================  =======================================================

``fig_1_17`` reads an absolute Spearman rho near 1 off-diagonal as "these two
metrics are redundant". ``fig_1_18`` matches metrics across splits by base name
(``train_ssim`` / ``ssim`` against ``val_ssim``) and shades the gap band: a
widening band is overfitting, while validation sitting *below* train from step
one indicates non-exchangeable splits (leakage or site shift).

*Generative-paradigm diagnostics:*

=============================  ===================================================
``gen_gan_diagnostics``        D/G loss balance, D-accuracy, gradient-penalty norm
``gen_vae_diagnostics``        KL per latent (posterior collapse), recon vs KL
``gen_diffusion_diagnostics``  Loss-per-timestep, PSNR vs NFE / CFG scale
=============================  ===================================================

*MRI-specific (need recorded cases):*

==========================  ==========================================================
``mri_recon_panel``         Cases x [input, prediction, target, error] with ROI insets
``kspace_error_spectrum``   Radial k-space error vs spatial frequency
``mri_a1_sr_triptych``      LR / SR / HR triptych, optional radial spectrum
``mri_a2_synthesis_c2c``    Contrast-to-contrast panel, 2-D joint histogram and MI
``mri_a7_kspace_recon``     Zero-filled / recon / reference plus k-space error
``mri_a11_cohort_table``    Cohort descriptor rendered as a figure
``mri_a12_fiducial_check``  Virtual-fiducial localization check
==========================  ==========================================================

``mri_a12_fiducial_check`` regenerates the canonical ``VirtualFiducial``
(:mod:`spectramr.infrastructure.physics.virtual_fiducial`) grid at
``pred.shape[-2:]`` and uses it as the ROI mask, so the pipeline does not need to
plumb an extra fiducial-mask field through the case contract — the same
``{pred, target, ...}`` payload the other case plotters take works unchanged. It
outlines the ROI in cyan and reports full-image PSNR against ROI PSNR per case.
Where the YAML sets ``physics.virtual_fiducial``, use the same ``grid_spacing``
and ``sigma`` as the figure's defaults (``grid_spacing=16``, ``sigma=2.0``), or
pass overrides through the plotter's ``**kwargs``.

*Publication statistics:*

========================  ===================================================
``metric_distribution``   Raincloud + paired-bootstrap significance brackets
``bland_altman``          Agreement (bias +/- 1.96 sigma limits)
``significance_matrix``   Pairwise Holm-Bonferroni-corrected p-values
``calibration_coverage``  Reliability diagram (empirical vs nominal coverage)
``forest_plot``           Effect size delta vs baseline, bootstrap CI
``ablation_heatmap``      Knob x knob -> metric heatmap
``acceleration_sweep``    Metric vs acceleration R, +/-1 sigma seed bands
``contact_sheet``         Thumbnail montage / index of every other PNG
========================  ===================================================

*Novel-paradigm physics diagnostics (bespoke domain frames):*

==========================================  =================================================
``fig_2_15_active_acquisition_trajectory``  PSNR vs acquisition step, per acquisition policy
``fig_b3_bloch_consistency_residual``       Bloch-equation residual decay per tissue
``fig_b7_qmap_riemannian_vs_euclidean``     qMap error (Riemannian vs Euclidean geometry)
``fig_c1_beltrami_field``                   Beltrami coefficient magnitude and phase
``fig_c2_spd_geodesic``                     Geodesic distance trajectory on Sym+(n)
``fig_c3_teichmuller_schedule``             Teichmueller radial schedule r(t)
``fig_c4_fingerprint_embedding``            MRF fingerprint embedding scatter (clusters)
==========================================  =================================================

*Quality-control (QC) report:*

=====================  =============================================================
``qc_group_strip``     Group IQM distribution: one horizontal box + jittered strip
                       per metric/loss the run computed, Tukey-outlier (1.5x IQR)
                       points flagged and labelled by ``case_id``
``qc_subject_mosaic``  Per-case QC mosaic: prediction (foreground-mask contour),
                       target, error map, background-noise reportlet
``qc_carpet``          Cases x image-column error carpet, per-case spike/ghost strip
=====================  =============================================================

``qc_group_strip`` discovers its metric set from the data — preferring
``per_call_metrics.csv``, then the recorded-case ``predictions_df``, then the
aggregate frame — so it tracks whatever metrics and losses the run computed
rather than a fixed IQM list. The mosaic and carpet consume the recorded
``report_cases`` arrays (or, in the standalone ``spectramr report`` path, the
recovered ``real_images`` / ``fake_images`` PNG pairs). All three are bundled
into ``qc_report.html`` when ``reporting.html_report`` is on, and gated per run by
``reporting.qc_figures``.

To report a whole cohort at once,
:func:`spectramr.infrastructure.reporting.generate_reports` (CLI:
``spectramr report --exp-dir <root> --recursive``) discovers every run beneath a
root via :func:`spectramr.infrastructure.reporting.discover_run_dirs`, reports
each, and writes a top-level ``report_index.html`` linking them.

Table catalog
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - ID
     - Table
     - Output
   * - ``tab_2_1_main_results``
     - Methods x metrics, mean +/- std, best and second-best marked,
       Holm-corrected p-values
     - ``.md`` + ``.tex``
   * - ``tab_2_2_ablation``
     - Variant x delta-metric against the full model
     - ``.md`` + ``.tex``
   * - ``tab_2_3_hyperparameters``
     - name / distribution / range / final / criterion
     - ``.md`` + ``.tex``
   * - ``tab_2_4_dataset_descriptor``
     - n / age / sex / scanner / pathology / split rule
     - ``.md`` + ``.tex``
   * - ``tab_run_summary``
     - One row per ``(split, metric)`` for a single run
     - ``.csv`` + ``.md`` + ``.tex``

Metrics available to the tables
-------------------------------

Names in ``reporting.metrics`` resolve against the framework metrics registry
(:mod:`spectramr.core.metrics`) — see :doc:`metrics_reference` for the registered
set — or against the report-time IQMs below.

Tier-1 radiologist-correlate IQMs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These live outside the core registry because they operate on numpy-style
prediction/target arrays at report time
(:mod:`spectramr.infrastructure.reporting.metrics`):

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Metric
     - Reference
     - Module
   * - **GMSD**
     - Xue *et al.*, IEEE TIP, 2014
     - ``reporting/metrics/gmsd.py``
   * - **HaarPSI**
     - Reisenhofer *et al.*, SPIC, 2018
     - ``reporting/metrics/haarpsi.py``
   * - **VIF**
     - Sheikh & Bovik, IEEE TIP, 2006
     - ``reporting/metrics/vif_fsim.py``
   * - **FSIM**
     - Zhang *et al.*, IEEE TIP, 2011
     - ``reporting/metrics/vif_fsim.py``

Report-time registered metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:mod:`spectramr.core.metrics.report_step_metrics` registers a further set aimed
at reporting and agreement analysis. All are decorated with ``@register_metric``,
so they resolve from any YAML's ``reporting.metrics`` block:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Registry name (aliases)
     - Definition
   * - ``dists`` (``DISTS``)
     - Ding *et al.*, 2020 — DISTS via the existing loss implementation
   * - ``mutual_information`` (``mi``, ``MI``)
     - Pluim *et al.*, IEEE TMI, 2003
   * - ``edge_preservation_index`` (``epi``)
     - Sobel-edge gradient correlation
   * - ``radial_k_error``
     - Mean ``||F.x_hat - F.x||`` binned by radial k
   * - ``average_surface_distance`` (``asd``)
     - Symmetric mean surface distance
   * - ``target_registration_error`` (``tre``)
     - Mean Euclidean landmark distance
   * - ``folding_fraction``
     - Voxel-wise ``Pr(det J <= 0)``
   * - ``dvf_mae``
     - Mean absolute error on a displacement vector field
   * - ``through_plane_fwhm``
     - LSF-derived real z-resolution
   * - ``bland_altman_bias`` (``ba_bias``)
     - Mean (prediction - target)
   * - ``limits_of_agreement_upper`` / ``_lower``
     - mean +/- 1.96 sigma
   * - ``icc_3_1`` (``ICC``)
     - Shrout & Fleiss, 1979 — ICC(3,1)
   * - ``coefficient_of_variation`` (``cv``)
     - ``sigma / |mu|``
   * - ``detection_sensitivity`` / ``detection_specificity``
     - ``TP/(TP+FN)`` and ``TN/(TN+FP)``
   * - ``auroc`` / ``auprc``
     - scikit-learn implementations
   * - ``expected_calibration_error`` (``ece``)
     - Standard 15-bin ECE
   * - ``nll_bits_per_dim`` (``bpd``)
     - NLL / log 2 per element
   * - ``wasserstein_1d`` (``w1``)
     - ``scipy.stats.wasserstein_distance``
   * - ``sliced_wasserstein`` (``sw``)
     - Random-projection 1-D Wasserstein
   * - ``mmd_metric``
     - Multi-bandwidth Gaussian-kernel MMD squared
   * - ``residual_whiteness`` (``whiteness``)
     - Spectral flatness of the denoising residual
   * - ``cohen_kappa`` (``kappa``)
     - Weighted kappa for Likert reads

Radiomics-based hallucination audit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Function
     - Use
   * - ``concordance_correlation_coefficient(a, b)``
     - Lin's CCC for per-feature agreement
   * - ``feature_preservation_profile(real, synth)``
     - Median CCC, fraction with CCC >= 0.85, per-IBSI-family medians,
       Bland-Altman bias per feature
   * - ``hallucination_index(real, synth, test_retest_std=...)``
     - Fraction of texture features whose deviation exceeds
       ``k * sigma_test-retest``

Both consume tidy radiomic feature DataFrames (rows = subjects, columns = IBSI
feature names); generate them with the PyRadiomics-backed
:mod:`spectramr.core.metrics.radiomic`.

Interactive layer (plotly)
--------------------------

When ``reporting.interactive`` is on (the default) **and** :mod:`plotly` is
installed, ``qc_report.html`` gains an interactive layer *beside* the static
PNGs — the PNGs stay as the print/archival copy and the offline fallback. The
builders live in :mod:`spectramr.infrastructure.reporting.interactive`; each
returns an inline ``<div>`` or ``None`` (plotly absent, or no data), so the
section degrades cleanly to the embedded PNG. Every interactive panel carries a
short **"How to read this"** caption.

*Self-contained and offline.* plotly.js is inlined **once** at the top of the
report (:func:`interactive.plotly_util.plotlyjs_script`); every div is emitted
with ``include_plotlyjs=False``. The report renders fully offline, with no CDN
fetch, matching the base64-PNG self-containment.

Interactive figures (:mod:`interactive.figures`):

- **Group IQM** — hoverable box and strip of every metric at once; hover shows
  the ``case_id``; the box whiskers delimit the Tukey (1.5x IQR) fences; axis
  labels carry up/down "better" arrows.
- **Training dynamics** — train (solid) against validation (dotted) curves with a
  metric dropdown and a linear/log-y toggle; a legend click isolates a method.
- **Per-case metric distribution** — a violin + box + points raincloud with a
  metric dropdown, split by method when several are present.
- **Method agreement** — Bland-Altman between the two best-represented methods
  (bias +/- 1.96 sigma), with a metric dropdown; soft-skips with fewer than two
  methods.

MRIQC-style slice viewers (:mod:`interactive.viewer_2d`,
:mod:`interactive.viewer_3d`):

- **2-D per-subject viewer** — a slider scrubs the recorded subjects
  (best to median to worst) while layer buttons flick the panel between
  Prediction, Target and absolute Error (the before/after reportlet). The slider
  drives plotly *frames* (swapping the heatmap ``z``) and the buttons drive trace
  *visibility*, so the two controls are orthogonal. Always available for 2-D
  cases.
- **3-D volume viewer** — the same two-control pattern over the **Z slices** of a
  single volume (the MRIQC mosaic navigation). It renders only when a case
  carries a 3-D ``*_volume`` array; on single-slice acquisitions (M4Raw, for
  example) it **soft-skips with an explicit note** — volumes are never fabricated
  from unrelated slices.

*Recording volumes.* The 3-D viewer needs 3-D data. Set
``reporting.record_volumes`` (opt-in, default off) to make the validation feeder
preserve a ``[Z, H, W]`` stack (``input_volume`` / ``prediction_volume`` /
``target_volume``) alongside the 2-D representative. It fires only for an
unambiguous slice axis — a 5-D ``[B, C, Z, H, W]`` tensor (coils RSS-collapsed,
slices kept) or a 3-D ``[Z, H, W]`` volume; a 4-D ``[B, C, H, W]`` (a 2-D slice
with coils) records **no** volume. On single-slice data the knob is a no-op. Set
``reporting.interactive: false`` (or ``spectramr report --no-interactive``) to
force the static-only report.

Cohort task-ablation bar figures
--------------------------------

Where ``--recursive`` emits one report *per arm*, the
``cohort_task_ablation_bars`` plotter emits one figure *per task* placing every
arm side by side — purpose-built for a multi-task, method-vs-ablation cohort.
Each figure is a vertical stack of sub-panels (one validation metric each; SSIM,
PSNR and LPIPS by default, since the scales differ), and within every panel the
bars are grouped by **method family** with the full method drawn first and its
one-knob **ablation** controls immediately after, coloured by role. The per-arm
scalar is the **best-over-iterations** validation value (max, or min for
lower-is-better metrics), matching each run's ``restore_best_weights``
early-stopping semantics. Degenerate or collapsed arms (negative PSNR, near-zero
SSIM, identity collapse) are shown honestly with their real value and hatched —
never hidden.

Assembly lives in
:mod:`spectramr.infrastructure.reporting.cohort_ablation`:
:func:`~spectramr.infrastructure.reporting.cohort_ablation.discover_cohort_arms`
derives ``task`` / ``family`` / ``role`` self-contained from each arm's
``resolved_config.json``;
:func:`~spectramr.infrastructure.reporting.cohort_ablation.best_val_metrics`
reuses the :func:`~spectramr.infrastructure.reporting.aggregator.aggregate` SSOT
reader to pull best validation scalars; and
:func:`~spectramr.infrastructure.reporting.cohort_ablation.generate_task_ablation_figures`
dispatches the registered plotter once per task. The plotter itself is generic:
it consumes a tidy ``[family, role, arm, metric, value]`` frame and returns
``None`` (soft-skip) on the standard aggregator frame, so it never disturbs the
default ``spectramr report`` pass. Drive it across a cohort with:

.. code-block:: bash

    spectramr report --exp-dir <cohort-root> --recursive

The driver never calls ``savefig`` — saving happens inside the registered plotter
via :func:`~spectramr.infrastructure.reporting.style.save_figure`, so the
plotting-SSOT ratchet holds.

TikZ / pgfplots export
----------------------

``tikz: true`` additionally emits LaTeX-native figures under
``report/figures/tikz/`` via
:mod:`spectramr.infrastructure.reporting.tikz_export`: per-metric
training/validation curves (``curve_<metric>.tex``) and the headline Pareto
(``pareto_<metric>_vs_<cost>.tex``). Each figure is written twice —

- ``<stem>.tikz`` — the bare ``tikzpicture``, ``\input``-able inside a paper;
- ``<stem>.tex`` — a ``standalone`` wrapper, compilable with
  ``tectonic -X compile <stem>.tex``.

Emission is pure string generation: **no LaTeX is required at write time**
(matplotlib's PGF backend is deliberately not used, because it shells out to a
live ``latex`` process for text metrics). Methods keep their Okabe-Ito colours
via generated ``\definecolor`` lines, and TikZ artifacts are recorded in
``report_manifest.json`` like any other figure.

Artifacts and provenance
------------------------

``report_manifest.json`` records every emitted artifact (id, path, format,
sha256) plus provenance (git commit, seed, dataset, timestamp).
``submission_bundle: true`` additionally writes ``submission/`` with a 600-dpi
TIFF and a caption ``.txt`` per figure.

.. _per-case-metrics-are-per-call:

``per_call_metrics.csv`` is per *call*, not per sample
------------------------------------------------------

Each row is **one validation call** carrying the **batch-aggregate** metrics,
tagged with a ``case_id`` that
:func:`~spectramr.infrastructure.training.strategies.mixins.metrics_mixin.report_case_id`
derives from the training step (plus the cascade rung, when there is one). It is
not one row per sample, because the run computes metrics batch-wise and no
per-sample value exists to record.

Reading it as a per-sample distribution produces false confidence. Eight rows
that all carry the same ``psnr`` are one scalar recorded eight times, and drawn
naively as a box-and-whisker they assert a tight agreement across eight cases
that the data never contained. Two guards stand between that data and a chart:

#. **No distribution without spread.** ``qc_group_strip`` drops any metric whose
   values have zero range, and returns ``None`` (soft-skip, matching its
   siblings) when nothing survives. The drop is **logged with the metric names**
   — a silently missing panel is indistinguishable from a broken plotter.
#. **No fabricated quartiles.** Below four observations the box is omitted rather
   than collapsed to ``q1 = med = q3``. Clamping the width instead — drawing the
   box at ``max(q3 - q1, 1e-9)`` — is the subtle trap: a zero-width box would be
   invisible, whereas a 1e-9 sliver renders as an extremely *tight* distribution,
   the most confident claim on the chart from the least data. The median tick and
   the min-max span still draw, because those are things the data does say.

Each row also carries a **context block** — identity columns written between
``step`` and the metrics, in the fixed order declared by
:data:`~spectramr.infrastructure.reporting.cases.metric_sink.CONTEXT_COLUMNS`:

========================= ======================================================
``acceleration_level``    The cascade rung this pass evaluated (2, 8, 32, ...)
``acceleration_realized`` What the schedule decoded that timestep back to
``timestep``              The diffusion timestep the rung actually ran at
``heldout``               ``True`` for an out-of-distribution severity point
``contrast``              ``T1`` / ``T2`` / ``FLAIR`` — see the mixed-batch note
``file_id``               The source file stem(s) the row averaged over
``batch_index``           Position of the batch within the validation loop
``batch_size``            How many volumes the row's metrics are a mean over
========================= ======================================================

``case_id`` encodes the step and, where there is one, the cascade rung — but not
which volume or which acceleration produced a given number. The context columns make
both explicit; ``case_id`` keeps its spelling, because PNG filenames and
downstream parsers are built on it, and the columns are purely additive.

Two honesty constraints hold in that block. A ``None`` value is **omitted**
rather than written as ``0``, because ``0`` is a real timestep (the clean end of
the diffusion schedule) and a plausible wrong value is worse than a visible gap.
``acceleration_realized`` is blank for exactly this reason on the linear-fallback
path, where there is no schedule to invert: an empty cell there means *nothing
could decode it*, never *it matched the request*. And because the validation
loader shuffles, one batch can hold a T1 volume and a FLAIR one — ``contrast``
then reports **every** contrast present, joined with ``|``, rather than picking
one and attributing the whole mean to it.

That block reaches the CSV through one call: ``feed_report_case_recorder`` passes
``context=`` to whatever sink it is handed, **unconditionally**, not only when
the mapping is non-empty. Accepting the keyword is therefore part of the
metric-sink contract, and nothing in the type system says so: ``sink`` is an
untyped parameter defaulting to ``None``, so a sink that omits it imports, lints
and type-checks clean, then raises ``TypeError`` the first time validation runs —
hours into a training job.
``test_every_metric_sink_in_src_accepts_the_context_kwarg`` is the ratchet for
that. It discriminates by *shape* rather than by the name ``observe``, because
three unrelated ``observe`` protocols live in this tree — the metric sink
(``split`` / ``step``), the image recorder (``arrays`` / ``domain``) and the
inference-artifact writer (``prediction`` / ``target``) — and only the first is
ever handed a context.

**Limitation.** Metrics are computed batch-wise, so a row is a mean over
``batch_size`` volumes rather than a per-sample measurement; the artifact keeps
its name for that reason, since renaming it would ratify batch aggregates as the
intended content. What the context block adds is that the row now *says* what it
aggregated over. Setting ``validation.loader.batch_size: 1`` makes each row
exactly one volume, which is per-sample granularity without per-sample metric
computation.

.. _validation-cascade-levels:

The validation ladder is declared, not fixed
--------------------------------------------

``acceleration_level`` comes from the cascade ladder, which is a config knob:

.. code-block:: yaml

   validation:
     cascade:
       levels: [2, 4, 8]      # omit for the framework default (2, 8, 32)

Both the training strategy and the ``validation_cascade_levels_in_range`` audit
check read one resolver,
:func:`~spectramr.core.cascade_levels.resolve_cascade_levels`, so an arm that
widens ``undersampling.acceleration_range`` for *training* cannot leave
*validation* pinned at a different ladder without saying so.

Levels are deduplicated and sorted ascending — the accel-gap readout subtracts
the last rung from the first, so "first rung is the mildest" is an invariant of
the consumers rather than of the author's typing order. An integral rung stays an
``int`` so the flat ``val_<metric>_<R>x`` names keep their spelling
(``val_psnr_2x``, not ``val_psnr_2.0x``), which matters because the L4
input-dependence gate and the accel-gap stamp look those names up and do not
raise on a miss. An empty ladder, a sub-1x rung, a non-finite value or a boolean
are **refused at load time** rather than silently repaired.

Under ``undersampling.schedule_type: step`` a rung that is not in
``undersampling.acceleration_range`` has no timestep inverse and is skipped at
runtime; ``spectramr audit`` warns about that before the launch.

Empty input and soft failure
----------------------------

Every plotter ``make`` and table ``make`` is a **total function on bad input**:
given an empty :class:`~pandas.DataFrame`, a frame missing the ``metric`` column,
or one lacking the requested metric/cost rows, it returns ``None`` (a "no data"
signal) rather than raising. The guard must precede any ``df["metric"]`` access —
a guard placed *after* the subscript raises ``KeyError: 'metric'`` on the empty
frame before it can return ``None``. That ordering is covered by
``tests/unit/reporting/test_reporting_plotters.py`` for ``headline_pareto``,
``ablation_strip`` and ``ablation_table``.

On top of that, the driver
(:func:`~spectramr.infrastructure.reporting.pipeline.generate_report`, through
``plotters.dispatch`` and ``tables.dispatch``) soft-fails each component in a
``try``/``except``, so a single bad figure cannot abort a long run's wrap-up. The
end-of-training hook is wrapped the same way: a reporting bug cannot break
training. Set ``fail_on_error: true`` when you want a reporting failure to be
loud — CI, or a strict regeneration.

Extending the pipeline
----------------------

Adding a new figure type
~~~~~~~~~~~~~~~~~~~~~~~~

#. Create ``my_figure.py`` in ``plotters/`` with the signature
   ``make(df, out_path, *, metadata=None, **kwargs) -> Path | None``.
#. Add it to the registration block in ``plotters/__init__.py``.
#. Add it to the relevant task preset in ``pipeline.py``'s ``TASK_PRESETS``, or
   invoke it explicitly through ``reporting.figures`` in YAML.

A newly registered figure with no coverage fixture fails
``test_registry_fixture_map_covers_every_registered_id``, so step 1 and its test
fixture land together.

Plotting SSOT
~~~~~~~~~~~~~

The reporting pipeline is the **single source of truth for figures**: every plot
is produced by a registered plotter under ``plotters/`` (plus the plotly
``interactive/`` layer), styled through ``style.py``, and orchestrated by
:func:`~spectramr.infrastructure.reporting.pipeline.generate_report`. The only
sanctioned exceptions are the sim2rank meta-evaluation figures under
``core/metrics/meta_evaluation/`` and dataset EDA under ``data/eda/``, reached by
the separate ``spectramr meta-evaluate`` and EDA paths, never by
``spectramr report``. A ratchet test,
``tests/unit/infrastructure/reporting/test_plotting_ssot_guard.py``, fails on any
new ``savefig`` / ``write_html`` / ``write_image`` / ``plt.show`` added outside
``infrastructure/reporting/``.

Direct-call renderers
~~~~~~~~~~~~~~~~~~~~~

Four renderers deliberately sit **outside** the ``PLOTTERS`` / ``TABLES``
registries because their signatures do not match the ``make(df, out_path, **kw)``
contract (``dispatch`` always calls ``fn(df, out_path, ...)``, so registering
them would crash it). They are called directly from the certification and
acquisition paths:

- :func:`plotters.certificate_summary.render_certificate_summary` — R1/R2/R4/R5
  plus validation-badge 2x3 grid, from the conformal-calibration JSON artefacts.
- :func:`plotters.ksd_certificate.render_ksd_certificate` — KSD goodness-of-fit
  one-page PDF, from a ``KSDDefensibilityResult``.
- :func:`plotters.learnable_acquisition_pareto.render_learnable_acquisition_pareto`
  — a thin wrapper over ``headline_pareto`` taking an ``ArmResult`` list.
- :func:`tables.certificate_table.write_certificate_table` — per-certificate
  LaTeX and Markdown table from a ValidationBadge payload.

Domain-artifact convention
~~~~~~~~~~~~~~~~~~~~~~~~~~

Physics and geometry figures whose data the aggregator frame cannot carry
(``fig_2_15_active_acquisition_trajectory``,
``fig_b3_bloch_consistency_residual``, ``fig_b7_qmap_riemannian_vs_euclidean``,
``fig_c1_beltrami_field`` through ``fig_c4_fingerprint_embedding``) are fed from
``<run>/report_artifacts/`` when present: ``<fig_id>.csv`` becomes that plotter's
DataFrame; ``active_acquisition/*.csv`` becomes ``csv_paths``; ``qmap_slices.npz``
becomes ``slices``. Absent artifacts leave the figure to soft-skip.
``mri_a12_fiducial_check`` is fed the recorded image cases like the other case
plotters. The certificate figures (``certificate_summary``, ``ksd_certificate``,
``learnable_acquisition_pareto``) are registered through soft-skipping adapters
and render when their certificate, result or arms payload is routed.

Validation loss in the learning curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ModelValidationMixin.validation_step`` emits a ``val_loss`` (a
magnitude-matched L1 on the validation prediction/target) alongside the image
metrics, so ``fig_1_2_learning_curves`` and its interactive twin draw a
validation-loss curve rather than a train-only series. The interactive builder
strips the ``val_`` prefix so a validation series pairs with its training twin on
one panel (solid = train, dotted = validation), and the aggregator drops
entirely-empty metric columns so a phantom pre-seeded ``val_loss`` header never
renders.

Quality-assurance checklist
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Check
     - Where it is enforced
   * - ``spectramr report -e <dir>`` regenerates a figure reproducibly
     - the sidecar ``*.meta.json`` records git SHA and seed
   * - Each figure renders at journal column width
     - per-plotter ``figsize`` through ``style.column_width``
   * - Colour-blind-safe palette
     - Okabe-Ito (8 hues), ``style.py``
   * - Legible in greyscale print
     - line styles and markers selected per method, not colour alone
   * - Numerical claims match the underlying CSV
     - tables read from ``final_eval.json`` and the aggregator output directly
   * - No figure depends on a deleted log file
     - the aggregator soft-fails on missing artifacts; plotters return ``None``
