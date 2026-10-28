r"""Global style tokens and color dictionary for the reporting pipeline.

Per `TODO/report_step` Phase 0: enforce a single source of truth for fonts,
colours, sizes, and output formats. Plotter modules are forbidden from
setting colour values locally.

Usage:
    >>> from spectramr.infrastructure.reporting.style import use_default_style
    >>> use_default_style()
    >>> # ... build figures ...
"""

from __future__ import annotations

import logging
import re
import zlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

_LOG = logging.getLogger(__name__)

_STYLE_DIR = Path(__file__).resolve().parent / "styles"
#: ``poster`` and ``slides`` keep the Nature column geometry -- the figure is
#: scaled up where it is placed -- and change only text and stroke weight
#: relative to the figure, which is what reading distance demands.
_STYLE_FILES = {
    "ieee": _STYLE_DIR / "ieee.mplstyle",
    "nature": _STYLE_DIR / "nature.mplstyle",
    "poster": _STYLE_DIR / "poster.mplstyle",
    "slides": _STYLE_DIR / "slides.mplstyle",
}

# Nature column widths (mm). Plotters size figures by profile, not magic inches.
NATURE_WIDTHS_MM = {"single": 89.0, "onehalf": 120.0, "double": 183.0}


def column_width(profile: str = "single") -> float:
    """Return a Nature column width in inches for the given profile.

    Args:
        profile: One of ``single`` (89 mm), ``onehalf`` (120 mm), ``double`` (183 mm).

    Raises:
        ValueError: on an unknown profile (no silent fallback).
    """
    if profile not in NATURE_WIDTHS_MM:
        raise ValueError(
            f"unknown width profile {profile!r}; choose from {sorted(NATURE_WIDTHS_MM)}"
        )
    return NATURE_WIDTHS_MM[profile] / 25.4


# Okabe–Ito colour-blind-safe palette + framework-wide method-colour
# dictionary. Plotters look up colours by *method name* (or fall back to
# the cycler order) — this keeps a single method visually consistent
# across every figure in the paper.
OKABE_ITO = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
]

METHOD_COLOURS: dict[str, str] = {
    # Reserved baselines
    "baseline": "#000000",
    "zero_filled": "#000000",
    "cs_sense": "#999999",
    # Diffusion family
    "diffusion": "#0072B2",
    "cold_diffusion": "#56B4E9",
    "ldm": "#009E73",
    "edm": "#56B4E9",
    "rectified_flow": "#0072B2",
    "stochastic_interpolants": "#009E73",
    # Score / posterior samplers
    "dps": "#0072B2",
    "pi_gdm": "#56B4E9",
    "dds": "#009E73",
    # GAN family
    "gan": "#E69F00",
    "cycle_gan": "#D55E00",
    # Reconstruction unrolled
    "varnet": "#CC79A7",
    "modl": "#CC79A7",
    # ULF physics arms
    "caur": "#E69F00",
    "xfield_fm": "#D55E00",
    "scas": "#CC79A7",
    "kspace_inr": "#0072B2",
    "girf_aware": "#999999",
    "field_probe_coupled": "#56B4E9",
    "diffeomorphic_recon": "#009E73",
    "inverse_bloch_phase": "#CC79A7",
    "spin_sde": "#E69F00",
    # Multi-contrast techniques
    "ssl_frozen_anatomy": "#0072B2",
    "vendor_soft_prompt": "#D55E00",
    "cfg_time_varying": "#009E73",
    "latent_residual_head": "#56B4E9",
    "bloch_bottleneck": "#CC79A7",
    # Acquisition arms
    "loupe": "#E69F00",
    "pilot": "#D55E00",
    "bald": "#0072B2",
    # Federated + DP
    "federated_dp_conformal": "#009E73",
    # Multi-param mapping
    "multi_param_mapping": "#CC79A7",
    "mrf_baseline": "#999999",
    # ----- Beautify backlog item 2.3 expansion (2026-05-14) -----
    # Per-strategy entries so any registered model in
    # ``src/infrastructure/training/strategy_factory.py`` gets a
    # deterministic colour at headline-plot time. Grouped by paradigm
    # family so a reader can see the colour design at a glance.
    # ----- Paradigm modes (canonical anchors) -----
    "reconstruction": "#009E73",
    "vae": "#0072B2",
    "vqvae": "#56B4E9",
    "ssl": "#E69F00",
    "mae": "#D55E00",
    "masked": "#D55E00",
    "flow_matching": "#56B4E9",
    # ----- Diffusion variants -----
    "kspace_cold_diffusion": "#56B4E9",
    "graph_cold_diffusion": "#009E73",
    "disentangled_diffusion": "#0072B2",
    "qspace_diffusion": "#56B4E9",
    "noise_adaptive_score": "#0072B2",
    "cross_contrast_kspace_diffusion": "#009E73",
    # ----- Disentangled / multi-task -----
    "disentangled": "#CC79A7",
    "disentangled_vae": "#0072B2",
    "multi_task": "#999999",
    "dual_task": "#999999",
    "multi": "#E69F00",
    "multi_contrast_contrastive": "#56B4E9",
    # ----- Physics / Bloch / ULF -----
    "physics_driven": "#0072B2",
    "pinn": "#56B4E9",
    "padnet": "#009E73",
    "pma_varnet": "#CC79A7",
    "concomitant_aware_recon": "#E69F00",
    "bloch_consistent_denoising": "#56B4E9",
    "cycle_bloch": "#009E73",
    "cycle_bloch_digital_twin": "#0072B2",
    "coord_kspace_gen": "#D55E00",
    "mri_slam": "#999999",
    "geomamba_ulf": "#E69F00",
    "hssc": "#56B4E9",
    "qsm_pipeline": "#009E73",
    # ----- Domain adaptation / federation / swarm -----
    "domain_adaptation": "#D55E00",
    "adaptation": "#D55E00",
    "federated": "#009E73",
    "swarm": "#56B4E9",
    "swarm_learning": "#56B4E9",
    # ----- Acquisition / active learning -----
    "active_acquisition": "#E69F00",
    "learnable_acquisition": "#D55E00",
    "acq_hypernetwork": "#D55E00",
    # ----- TTA / TTO -----
    "test_time_adaptation": "#CC79A7",
    "test_time_optimization": "#CC79A7",
    "tto": "#CC79A7",
    "ttt": "#CC79A7",
    # ----- SR / mesh / volumetric -----
    "guided_sr": "#0072B2",
    "mesh": "#999999",
    "slice_to_volume": "#56B4E9",
    "volumetric": "#56B4E9",
    # ----- Meta / motion / synth -----
    "meta_learning": "#CC79A7",
    "motion_meta": "#CC79A7",
    "synthetic_pathology_aug": "#E69F00",
    # ----- Calibration / certificates -----
    "calibration": "#999999",
    "hardware_warp_calibration": "#999999",
    "validation_badge": "#000000",
    "distillation": "#56B4E9",
    # ----- Misc primitives -----
    "diff_siren": "#0072B2",
    "low_rank_sparse": "#D55E00",
    "noise_to_noise": "#E69F00",
    "hybrid": "#999999",
    "jepa": "#56B4E9",
    "virtual_fiducial": "#D55E00",
    "vf_admm": "#D55E00",
    "privileged": "#0072B2",
    "privileged_learning": "#0072B2",
    "adversarial_robustness": "#CC79A7",
    "pnp": "#999999",
    "pnp_red": "#999999",
    # ----- SFC / conformal geometry (audit_plan_novel 2026) -----
    "conformal_diffusion_recon": "#0072B2",
    "cortical_conformal_recon": "#56B4E9",
    "teichmuller_cold_diffusion": "#009E73",
    # ----- fMRI directions (audit_plan_novel_fmri 2026) -----
    "cortical_conformal_fmri_recon": "#0072B2",
    "spatiotemporal_adaptive_sfc_recon": "#56B4E9",
    "adaptive_sfc_hssc": "#009E73",
    "hrf_manifold_diffusion": "#E69F00",
    "riemannian_dfc_diffusion": "#D55E00",
    # ----- MR-fingerprinting (audit_plan_novel_fmri 2026 — MRF half) -----
    "conformal_mrf_dictless_recon": "#0072B2",
    "spatiotemporal_mrf_recon": "#56B4E9",
    "riemannian_mrf_diffusion": "#009E73",
    "riemannian_bloch_diffusion": "#E69F00",
    "crlb_mrf_pulse_design": "#D55E00",
    "cross_scanner_mrf_harmonisation": "#CC79A7",
    # ----- Physics equivariance / Beltrami distortion correction -----
    "physics_equivariant_ssl": "#0072B2",
    "phys_eq_ssl": "#0072B2",
    "beltrami_epi_distortion": "#56B4E9",
    "beltrami_motion_correction": "#009E73",
    "bloch_equivariant_translation": "#E69F00",
    # ----- Information-bottleneck active acquisition -----
    "ib_active_acquisition": "#D55E00",
    "ib_bald": "#CC79A7",
    # ----- Score-field tomography -----
    "score_field_tomography": "#999999",
    # ----- 2026-07-01 coverage pass: every STRATEGY_CLASS_PATHS key gets a
    # deterministic colour (test_method_colours_coverage was blind to these
    # while it pointed at the pre-refactor src/ path). Grouped by family. -----
    # Diffusion / score / bridge samplers
    "ambient_diffusion": "#0072B2",
    "cross_modal_diffusion": "#56B4E9",
    "levy_diffusion": "#009E73",
    "resetting_diffusion": "#0072B2",
    "x_diffusion": "#56B4E9",
    "tissue_diffusion_pretrain": "#009E73",
    "field_guided_diffusion": "#0072B2",
    "field_cold_diffusion": "#56B4E9",
    "bloch_manifold_dps": "#0072B2",
    "twin_dps": "#56B4E9",
    "ulf_dps": "#009E73",
    "doob_bridge": "#0072B2",
    "bloch_schrodinger_bridge": "#56B4E9",
    "flow_matching_pfode": "#0072B2",
    # GAN / VAE hybrids
    "beta_vae_gan": "#E69F00",
    "progressive_gan": "#D55E00",
    "generative": "#E69F00",
    # SSL / self-supervised reconstruction
    "ssdu": "#E69F00",
    "robust_ssdu": "#D55E00",
    "self_supervised_reconstruction": "#E69F00",
    "equivariant_imaging": "#56B4E9",
    "robust_ei": "#0072B2",
    "recoverability_vib": "#CC79A7",
    # Physics / field arms
    "bloch_field": "#0072B2",
    "field_bridge": "#56B4E9",
    "field_flow": "#009E73",
    "field_fno": "#0072B2",
    "field_wiener": "#999999",
    "koopman_field": "#56B4E9",
    "monotone_field": "#009E73",
    "mccann_field_path": "#CC79A7",
    "cross_field_translation": "#D55E00",
    "field_conditioned_inr": "#E69F00",
    "heteroscedastic_ulf": "#E69F00",
    "ulf_map": "#0072B2",
    "ulf_redegrad_tta": "#CC79A7",
    "heisenberg_phase": "#56B4E9",
    "symplectic_bloch": "#009E73",
    # Geometry / manifold / algebraic
    "fisher_rao_flow": "#0072B2",
    "fisher_rao_geodesic": "#56B4E9",
    "hyperbolic_cardiac": "#009E73",
    "hodge_motion": "#CC79A7",
    "sheaf_kspace": "#0072B2",
    "sheaf_multi_contrast": "#56B4E9",
    "spectral_triple": "#999999",
    "tropical_mrf": "#E69F00",
    "tropical_quantitative_maps": "#D55E00",
    "sle_compressed_sensing": "#009E73",
    "scattering_besov": "#56B4E9",
    "primary_ideal": "#999999",
    "koopman_fmri": "#0072B2",
    # Federated / causal
    "frontdoor_federated": "#009E73",
    "frontdoor_scanner": "#56B4E9",
    "stein_federated": "#0072B2",
    # Acquisition / trajectory
    "hamiltonian_acquisition": "#E69F00",
    "hjb_trajectory": "#D55E00",
    "hjb_waveform": "#CC79A7",
    "multi_acquisition": "#E69F00",
    "trajectory_recon": "#0072B2",
    # Calibration / certificates / robustness
    "equivariance_conformal": "#999999",
    "phys_residual_conformal": "#999999",
    "corruption_calibration": "#999999",
    "certified_robustness": "#CC79A7",
    # Synthesis / translation
    "brenier_synthesis": "#E69F00",
    "paired_synthesis": "#D55E00",
    "steerable_synthesis": "#CC79A7",
    # TTA
    "spectra_tta": "#CC79A7",
    # Misc / infrastructure arms
    "universal_reconstruction": "#009E73",
    "operator_id": "#999999",
    "operator_id_bch": "#999999",
    "lora_modulation": "#56B4E9",
    "data_efficiency_harness": "#999999",
    "confluence": "#0072B2",
    "ib_vf": "#D55E00",
    "vf_consistency_distillation": "#56B4E9",
    "sparse_frame": "#E69F00",
    "cartoon_texture_safe": "#009E73",
    # ----- Unpaired translation (GAN family) -----
    # `cyclegan` is the registered strategy name; `cycle_gan` above is the
    # spelling the older figures used. Both map to the same colour ON PURPOSE --
    # one method must not change colour because a caller spelled it differently.
    "cyclegan": "#D55E00",
    "cut": "#E69F00",
    "generative_refiner": "#CC79A7",
    # ----- Physics: Bloch / field -----
    "bloch_synth": "#56B4E9",
    "dispersion_bloch_ae": "#56B4E9",
    "field_cocycle": "#009E73",
    # ----- Quantitative / non-anatomical endpoints -----
    # These grade a physical quantity (a spectrum, a perfusion constant, a
    # velocity), not an image, so they are deliberately far apart in hue from
    # the reconstruction families above.
    "mrs_quantification": "#0072B2",
    "perfusion_kinetic": "#D55E00",
    "phase_contrast_flow": "#CC79A7",
    # ----- Acquisition-side -----
    "quality_matching": "#999999",
}


_ACTIVE_STYLE = "nature"


def use_default_style(style: str | None = None, extra_rc: dict[str, object] | None = None) -> None:
    """Apply a named house style + optional per-figure overrides.

    Args:
        style: ``"nature"``, ``"ieee"``, ``"poster"`` or ``"slides"``. When ``None``, re-apply the
            most-recently-selected style — so plotters that call this with no
            argument inherit the style chosen by ``generate_report``.
        extra_rc: Optional dict of additional rcParams overrides.

    Raises:
        ValueError: on an unknown explicit style name (no silent fallback).
    """
    global _ACTIVE_STYLE
    if style is None:
        style = _ACTIVE_STYLE
    elif style not in _STYLE_FILES:
        raise ValueError(f"unknown report style {style!r}; choose from {sorted(_STYLE_FILES)}")
    else:
        _ACTIVE_STYLE = style
    style_file = _STYLE_FILES[style]
    if style_file.exists():
        plt.style.use(str(style_file))
    else:
        mpl.rcParams.update(
            {
                "font.family": "sans-serif" if style == "nature" else "serif",
                "font.size": 7 if style == "nature" else 9,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": style != "nature",
                "savefig.dpi": 600 if style == "nature" else 300,
                "savefig.bbox": "tight",
            }
        )
    if extra_rc:
        mpl.rcParams.update(extra_rc)


_PANEL_LABELS_ENABLED = True


def set_panel_labels(enabled: bool) -> None:
    """Globally enable/disable panel lettering (driven by ``reporting.panel_labels``).

    Plotters call :func:`panel_label` unconditionally; this flag lets the report
    pipeline honour the ``panel_labels`` knob without per-plotter branching.
    """
    global _PANEL_LABELS_ENABLED
    _PANEL_LABELS_ENABLED = bool(enabled)


def panel_label(
    ax,
    letter: str,
    *,
    weight: str = "bold",
    size: float = 8.0,
    x: float = -0.02,
    y: float = 1.04,
) -> None:
    """Stamp a bold lowercase panel letter at the top-left of ``ax`` (Nature convention).

    No-op when panel labels are globally disabled via :func:`set_panel_labels`.
    """
    if not _PANEL_LABELS_ENABLED:
        return
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=size,
        fontweight=weight,
        va="bottom",
        ha="right",
    )


#: Colours an UNREGISTERED method may never be given. Black is the reserved
#: identity of ``baseline`` / ``zero_filled``: handing it to an unknown method
#: both loses the distinction between two unknowns and disguises one as the
#: baseline, which is the single most misreadable thing a comparison figure can do.
_FALLBACK_PALETTE = [c for c in OKABE_ITO if c != "#000000"]


def colour_for(method: str, fallback_index: int | None = None) -> str:
    """Return the canonical colour for a method name.

    An unregistered method used to fall back to ``OKABE_ITO[0]`` -- black -- for
    *every* caller that did not pass an index. Nine registered strategies had no
    entry (#720), so a figure comparing two of them drew both in the baseline's
    colour: not a missing colour but a wrong one, and silently. The fallback is
    now derived from the name, so two unknowns differ and neither can be mistaken
    for the baseline.

    This stays a fallback rather than a raise. A colour is cosmetic in a way an
    ``attention_type`` is not (pitfall #9 governs registered-option enums), and
    ``test_method_colours_coverage`` is the ratchet that keeps the table complete.

    Args:
        method: Method identifier (case-insensitive, underscores allowed).
        fallback_index: Explicit index into the non-black palette. ``None``
            derives a stable index from the method name -- ``crc32``, not
            ``hash()``, because ``hash()`` is salted per process and would give
            one method different colours in two figures of the same report.
    """
    key = method.lower().strip()
    registered = METHOD_COLOURS.get(key)
    if registered is not None:
        return registered
    if fallback_index is None:
        fallback_index = zlib.crc32(key.encode("utf-8"))
    return _FALLBACK_PALETTE[fallback_index % len(_FALLBACK_PALETTE)]


# Human-readable axis / title labels for the metric keys that appear in the
# tidy frame. Same SSOT rationale as METHOD_COLOURS: one dictionary keeps
# "psnr" rendering as "PSNR (dB)" in every figure. The fallback (underscores
# to spaces) is intentionally non-raising — labels are cosmetic, unlike
# registered-option enums (pitfall #9 does not apply).
_PRETTY_LABELS: dict[str, str] = {
    # image-quality metrics
    "psnr": "PSNR (dB)",
    "ssim": "SSIM",
    "ms_ssim": "MS-SSIM",
    "nmse": "NMSE",
    "nrmse": "NRMSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "mse": "MSE",
    "lpips": "LPIPS",
    "hfen": "HFEN",
    "vif": "VIF",
    "fsim": "FSIM",
    "haarpsi": "HaarPSI",
    "gmsd": "GMSD",
    "fid": "FID",
    "dice": "Dice",
    # loss terms
    "loss": "total loss",
    "g_total_loss": "generator total loss",
    "d_total_loss": "discriminator total loss",
    "g_loss": "generator loss",
    "d_loss": "discriminator loss",
    "l1": "L1 loss",
    "l2": "L2 loss",
    "kl": "KL divergence",
    "perceptual": "perceptual loss",
    "adversarial": "adversarial loss",
    # run-level facts (split="run" rows from run_summary.json)
    "params_m": "parameters (M)",
    "flops_g": "FLOPs (G)",
    "iterations_per_sec": "throughput (it/s)",
    "duration_min": "wall-clock (min)",
    "duration_sec": "wall-clock (s)",
    "effective_batch": "effective batch size",
    # bookkeeping
    "final_loss": "final loss",
    "early_stopping_best_value": "early-stopping best value",
    "early_stopping_best_iteration": "early-stopping best iteration",
    "learning_rate": "learning rate",
    "step": "training step",
    "epoch": "epoch",
    "gradient_norm_g": "generator gradient norm",
    "gradient_norm_d": "discriminator gradient norm",
    "gpu_mem_gb": "GPU memory (GB)",
    "cpu_mem_gb": "CPU memory (GB)",
    "coverage_at_alpha": "coverage @ α",
}

_SPLIT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("val_", "validation "),
    ("train_", "training "),
    ("test_", "test "),
)


def pretty_label(name: str) -> str:
    """Return the canonical human-readable label for a metric / axis key.

    Known keys map through :data:`_PRETTY_LABELS`; ``val_``/``train_``/``test_``
    prefixes are expanded recursively (``val_ssim`` → ``"validation SSIM"``);
    unknown keys degrade to lowercase words (never raises — see the SSOT note
    on :data:`_PRETTY_LABELS`).
    """
    low = str(name).strip().lower()
    if low in _PRETTY_LABELS:
        return _PRETTY_LABELS[low]
    for prefix, words in _SPLIT_PREFIXES:
        if low.startswith(prefix) and len(low) > len(prefix):
            return words + pretty_label(low[len(prefix) :])
    return low.replace("_", " ").strip()


# ============================================================
# Phase 2 logging-backlog helpers (PR — beautify_logging_and_reporting)
# ============================================================


def styled_figure(fn):
    """Decorator: apply ``use_default_style()`` before ``fn`` runs.

    Plan: TODO/backlog_beautify_logging_and_reporting.md §Phase 2.1.

    Wraps any plotter entry point so it always honours the project's
    matplotlib style — without each plotter remembering to call
    ``use_default_style()`` itself.
    """
    import functools

    @functools.wraps(fn)
    def _wrap(*args, **kwargs):
        use_default_style()
        return fn(*args, **kwargs)

    return _wrap


def add_metadata_footer(
    fig,
    *,
    run_id: str | None = None,
    commit_sha: str | None = None,
    timestamp: str | None = None,
    fig_id: str | None = None,
) -> None:
    """Stamp a one-line metadata footer at the bottom of ``fig``.

    Plan: TODO/backlog_beautify_logging_and_reporting.md §Phase 2.2.

    Format:
        ``fig 3 · run-2026-05-08-... · @ a1b2c3d · 2026-05-08 14:32 UTC``

    Reviewers can trace any figure back to the exact run + commit.
    """
    parts: list[str] = []
    if fig_id:
        parts.append(f"fig {fig_id}")
    if run_id:
        parts.append(run_id)
    if commit_sha:
        parts.append(f"@ {commit_sha[:7]}")
    if timestamp:
        parts.append(timestamp)
    if not parts:
        return
    fig.text(
        0.5,
        0.005,
        " · ".join(parts),
        ha="center",
        va="bottom",
        fontsize=5,
        color="#888888",
    )


def caption_for(plot_id: str, **kwargs) -> str:
    """Project-wide IEEE-style caption template.

    Plan: TODO/backlog_beautify_logging_and_reporting.md §Phase 2.4.

    The plot_id selects a one-sentence template; kwargs are formatted in.
    """
    templates = {
        "headline_pareto": (
            "**Headline Pareto.** {metric} (y) vs. {cost} (x); markers on the "
            "non-dominated frontier are highlighted. Each method evaluated "
            "on {n_test} held-out scans."
        ),
        "main_results": (
            "**Main results.** Mean ± std over {n_seeds} seeds; "
            "best in **bold**, second-best *underlined*. Significance "
            "computed with paired bootstrap and Holm-Bonferroni correction."
        ),
        "certificate_summary": (
            "**Regulatory certificates.** R1 conformal coverage, R2 CHD slack, "
            "R4 pathology-recall lower bound, R5 PAC-Bayes upper bound, V.3 "
            "validation badge. {n_calibration} calibration samples."
        ),
        "ablation_strip": (
            "**Ablation.** {metric} as a function of {axis}; baseline shown "
            "as a dashed reference line."
        ),
        "run_summary_card": (
            "**Run summary.** Provenance, compute profile, and best validation "
            "metrics for run {run_id}."
        ),
        "metric_correlation": (
            "**Metric correlation.** Spearman rank correlation between "
            "validation metrics across training; |ρ| ≈ 1 off-diagonal flags a "
            "redundant (or degenerate) metric."
        ),
        "train_val_gap": (
            "**Generalization gap.** Matched train/validation {metric} over "
            "training; the shaded band is the train-val gap (widening = "
            "overfitting)."
        ),
    }
    tpl = templates.get(plot_id)
    if tpl is None:
        return ""
    try:
        return tpl.format(**kwargs)
    except KeyError as e:
        return f"{tpl}  (missing key: {e})"


# ============================================================
# Phase 2.5 colorbar helper (beautify_logging_and_reporting §2.5)
# ============================================================


def attach_colorbar(
    im,
    ax,
    label: str,
    units: str | None = None,
    *,
    shrink: float = 0.8,
    num_ticks: int = 5,
    vmin_zero_floor: bool = False,
) -> object:
    """Attach a colorbar to ``ax`` for image ``im`` with consistent formatting.

    Args:
        im: The ``AxesImage`` returned by ``ax.imshow(...)``.
        ax: The matplotlib axes to attach the colorbar to.
        label: Quantity name (e.g. ``"PSNR"``).
        units: Optional units string (e.g. ``"dB"``). Appended as
            ``label [units]``.
        shrink: Vertical shrink ratio (default ``0.8`` — matches IEEE
            single-column figures).
        num_ticks: Target number of ticks. ``MaxNLocator(num_ticks)``
            chooses sensible positions.
        vmin_zero_floor: When ``True`` and the colormap's ``vmin`` is at
            or below zero, silence the bottom-truncation artefact by
            clamping ``vmin`` to zero. Useful for SSIM / PSNR / FSIM
            plots where negative values are physically meaningless.

    Returns:
        The created ``Colorbar`` instance (so callers can further tweak
        font sizes, label positions, etc.).
    """
    import matplotlib.ticker as mticker

    fig = ax.get_figure()
    formatted_label = f"{label} [{units}]" if units else label
    if vmin_zero_floor:
        vmin, vmax = im.get_clim()
        if vmin is not None and vmin <= 0:
            im.set_clim(vmin=0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=shrink)
    cbar.set_label(formatted_label)
    cbar.locator = mticker.MaxNLocator(num_ticks)
    cbar.update_ticks()
    return cbar


# ============================================================
# Phase 2.6 PNG + PDF dual emit (beautify_logging_and_reporting §2.6)
# ============================================================

# C0 (minus TAB/LF) + DEL + C1 control characters. DejaVu Sans — the house
# font — has no glyph for these, so any leaking into a figure label makes
# ``fig.savefig`` emit a "Glyph N missing from font(s)" ``UserWarning`` (the
# 2026-07 report: a corrupt arm name reaching ``suptitle`` produced glyphs
# 127/128). They carry zero display meaning, so stripping them is a
# normalisation, not a silent fallback (pitfalls #9/#10).
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def sanitize_text_for_font(text: str) -> str:
    """Strip non-renderable C0/C1 control characters from ``text``.

    Keeps ordinary whitespace (TAB, LF) and every renderable Unicode
    character (µ, ±, en-dash, …). Returns ``text`` unchanged when it holds
    no control characters, so the common path allocates nothing new.
    """
    return _CONTROL_CHARS_RE.sub("", text)


def sanitize_figure_text(fig) -> int:
    """Normalise every ``Text`` artist on ``fig`` in place.

    Walks all titles, axis/tick labels, legend entries, table cells, and
    annotations, replacing any that carry control characters with a
    sanitised copy. Returns the number of artists rewritten (``0`` on the
    clean common path). Call immediately before rasterising/serialising a
    figure so data-derived labels can never emit a missing-glyph warning.
    """
    from matplotlib.text import Text

    changed = 0
    for artist in fig.findobj(Text):
        raw = artist.get_text()
        if not raw:
            continue
        cleaned = sanitize_text_for_font(raw)
        if cleaned != raw:
            artist.set_text(cleaned)
            changed += 1
    if changed:
        _LOG.debug(
            "sanitize_figure_text: stripped control characters from %d text artist(s)",
            changed,
        )
    return changed


def save_plot(
    fig,
    out_path: str | Path,
    *,
    formats: tuple[str, ...] | None = None,
    dpi: int = 600,
    metadata=None,
) -> Path | None:
    """Save ``fig`` in every requested format, write its sidecar, return the primary path.

    The one way a plotter ends: ``out_path``'s stem names the files and the first
    format is the primary. ``formats=None`` means the one format ``out_path``
    names, so a direct call with ``out.png`` still gets ``out.png``.
    """
    out_path = Path(out_path)
    if formats is None:
        suffix = out_path.suffix.lstrip(".").lower()
        formats = (suffix,) if suffix else ("pdf", "png")
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        from .metadata import write_sidecar

        write_sidecar(primary, metadata)
    return primary


def save_figure(
    fig,
    out_dir: str | Path,
    stem: str,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    close: bool = True,
) -> dict[str, Path]:
    """Save a figure in each requested format.

    Data-derived labels (arm names, metric keys, provenance strings) are
    scrubbed of non-renderable control characters via
    :func:`sanitize_figure_text` before saving, so a corrupt label can
    never emit matplotlib's "Glyph N missing from font" warning.

    Args:
        formats: any of ``pdf`` (vector), ``png`` (review raster),
            ``tiff`` (600 dpi LZW, Nature submission), ``eps`` (vector), ``svg``.
        dpi: raster DPI for png/tiff.
        close: close the figure afterwards to release memory.

    Raises:
        ValueError: on an unknown format token (no silent skip).
    """
    from pathlib import Path

    allowed = {"pdf", "png", "tiff", "eps", "svg"}
    unknown = set(formats) - allowed
    if unknown:
        raise ValueError(f"unknown figure format(s) {sorted(unknown)}; allowed {sorted(allowed)}")
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    # Defence-in-depth: normalise any control characters that leaked into a
    # data-derived label before matplotlib tries (and fails) to render them.
    sanitize_figure_text(fig)
    written: dict[str, Path] = {}
    for ext in formats:
        path = target_dir / f"{stem}.{ext}"
        if ext in {"png", "tiff"}:
            save_kw: dict[str, object] = {"dpi": dpi}
            if ext == "tiff":
                save_kw["pil_kwargs"] = {"compression": "tiff_lzw"}
            fig.savefig(path, **save_kw)
        else:
            fig.savefig(path)  # vector
        written[ext] = path
    if close:
        import matplotlib.pyplot as _plt

        _plt.close(fig)
    return written
