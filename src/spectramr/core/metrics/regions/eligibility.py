"""The ROI eligibility gate: which metrics may be scored on which regions.

A fastMRI+ pathology bbox is ~40x40 px. On an ROI that small, roughly half the
metric pool is **ill-defined**:

* **MS-SSIM** needs >= 161 px/side for its 5 dyadic scales.
* **NIQE / BRISQUE** fit natural-scene statistics on patches.
* **LPIPS / DISTS / FID / KID** compute deep-network statistics; on 40 px of
  content those statistics are meaningless.
* **Spectral metrics** -- an image-space crop's FFT is a convolution with a sinc,
  **not** a restriction of k-space. The crop does not compute "the spectrum of the
  ROI"; it computes a smeared spectrum of the whole slice.
* **Physics NR metrics** (``g_factor``, ``ndcr``, ``prp_snr``, …) consume the
  *measurement* (k-space, coil maps, sampling mask). Restricting them to an image
  ROI is a category error, not an approximation.

Without a gate, every one of those still returns a *number*, and the per-segment
leaderboard becomes noise dressed as science. So: the gate returns NaN **and a
machine-readable reason**, and the metric object is **never invoked**.

Three tiers
-----------

``MAP_THEN_MASK`` (preferred) -- compute the full-FOV spatial error map, then
average over the mask. Exact for MSE/MAE/PSNR, and it keeps correct neighbourhood
statistics at the ROI boundary for SSIM: the SSIM map at an ROI pixel uses an 11x11
Gaussian neighbourhood that may extend *outside* the ROI, which is correct -- that
is the structure the pixel actually sits in. A crop truncates it, and for a 40x40
ROI the 11-px boundary band is ~65% of the pixels.

``CROP_THEN_COMPUTE`` -- tight bbox crop, no resampling, no padding, no zeroing of
out-of-region pixels inside the bbox (zeroing manufactures a hard edge that LPIPS
and NIQE will happily "detect"). Required for metrics with no spatial map. Because
the bbox of a non-rectangular region (GM, WM) is a strict superset of the region,
crop-tier metrics are **ineligible on non-rectangular regions** -- cropping would
score the bbox, not the tissue.

``NOT_RESTRICTABLE`` -- NaN + reason. The metric is never called.

No safe default
---------------

There is deliberately **no fallback tier**. Defaulting to ``CROP`` would silently
run new metrics on 40 px crops; defaulting to ``NOT_RESTRICTABLE`` would make them
silently vanish from every regional leaderboard. Both are fabrications. An
undeclared metric therefore raises :class:`KeyError` -- loud, and fixable in one
line.

The normalisation trap
----------------------

PSNR's peak, SSIM's ``data_range``, NMSE's denominator. If ``data_range`` is
recomputed *from the ROI*, then "SSIM on GM" and "SSIM on the bbox" use different
C1/C2 and are **not comparable across regions** -- the metric has silently changed
per region. The rule: **error terms are masked; normalisation constants are
computed once on the full slice.** Every policy states which it does, and that
string is stamped on every output row.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from spectramr.core.metrics.outcome import NotApplicableReason
from spectramr.core.metrics.regions.types import RegionMask

if TYPE_CHECKING:
    # Resolved at runtime by the module-level __getattr__ below (PEP 562), which keeps
    # the table from freezing against a half-registered registry. Declared here so type
    # checkers still see the public name.
    ROI_POLICY: dict[str, RoiPolicy]

__all__ = [
    "ROI_POLICY",
    "Eligibility",
    "Normalisation",
    "RoiPolicy",
    "RoiSupport",
    "evaluate_eligibility",
    "min_side_for",
    "policy_for",
]


class RoiSupport(StrEnum):
    """How (or whether) a metric can be restricted to a region."""

    MAP_THEN_MASK = "map"
    CROP_THEN_COMPUTE = "crop"
    NOT_RESTRICTABLE = "none"


class Normalisation(StrEnum):
    """Where a metric's normalisation constants come from.

    ``FULL_SLICE`` is the only value that keeps scores comparable across regions.
    ``NOT_APPLICABLE`` is for metrics with no normalisation constant at all.

    ``ROI_LOCAL`` is the honest label for the crop tier. The crop tier hands the metric
    *only* the crop, and the piq family then derives its own ``data_range`` from what it
    is handed (``perceptual_piq.py``: ``scale = t.amax().clamp_min(1e-8)``) -- so HaarPSI
    on a dark GM ROI and HaarPSI on a bright lesion ROI use *different* constants.
    Verified: ``haarpsi(p, t)`` and ``haarpsi(0.2*p, 0.2*t)`` return the identical
    0.884248. The same holds for ``robust_mri_psnr`` (percentile of its input) and
    ``clinical_ssim`` (auto-mask at 5% of its input's max).

    These rows used to be stamped ``FULL_SLICE``, which was simply false -- the crop tier
    has no mechanism to receive a full-slice constant. A stamp that lies is worse than no
    stamp: it is an unverified provenance claim on every output row. Until a full-slice
    ``data_range`` is threaded into the crop call, say what actually happened, and treat
    crop-tier scores as **not comparable across regions**.
    """

    FULL_SLICE = "full_slice"
    ROI_LOCAL = "roi_local"
    NOT_APPLICABLE = "none"


@dataclass(frozen=True, slots=True)
class RoiPolicy:
    """What a single metric is allowed to do on a region."""

    support: RoiSupport
    justification: str
    min_roi_side: int = 0
    min_roi_px: int = 0
    normalisation: Normalisation = Normalisation.NOT_APPLICABLE

    def __post_init__(self) -> None:
        if not self.justification:
            raise ValueError("every RoiPolicy needs a human justification")
        if self.support is RoiSupport.NOT_RESTRICTABLE and (self.min_roi_side or self.min_roi_px):
            raise ValueError(
                "a NOT_RESTRICTABLE policy cannot carry a size floor -- size is not "
                "why it is ineligible."
            )


@dataclass(frozen=True, slots=True)
class Eligibility:
    """The verdict for one (metric, region) pair. A pure function of policy+geometry."""

    metric: str
    region_id: str
    eligible: bool
    support: RoiSupport
    n_px: int
    min_side: int
    reason: NotApplicableReason | None = None
    detail: str = ""
    normalisation: Normalisation = Normalisation.NOT_APPLICABLE

    def as_row(self) -> dict[str, object]:
        """One row of the run's exclusion table."""
        return {
            "metric": self.metric,
            "region": self.region_id,
            "n_px": self.n_px,
            "min_side": self.min_side,
            "eligible": self.eligible,
            "tier": str(self.support),
            "reason": str(self.reason) if self.reason else "",
            "detail": self.detail,
            "normalisation": str(self.normalisation),
        }


# ---------------------------------------------------------------------------
# The policy table
#
# Built from rules that are *defensible from the metric's definition*, not from
# guesses about its receptive field. Per-metric overrides carry hard evidence.
# ---------------------------------------------------------------------------

# Metrics with a genuine per-pixel error map, for which reductions.py implements a
# (maps, reduce) pair AND a test proves reduce(maps(p, t), ones) == the registry
# metric. Nothing enters this tier without that proof -- otherwise the region tier
# would be silently computing a *different* metric under the same name.
_MAP_TIER: dict[str, Normalisation] = {
    "mse": Normalisation.NOT_APPLICABLE,
    "mae": Normalisation.NOT_APPLICABLE,
    "rmse": Normalisation.NOT_APPLICABLE,
    "nmse": Normalisation.FULL_SLICE,
    "psnr": Normalisation.FULL_SLICE,
    "ssim": Normalisation.FULL_SLICE,
}

# Metric groups, declared HERE in core.
#
# The sim2rank METRIC_SPECS categories live in `scripts/`, and `src/` may never
# import `scripts/` (layer rule). So the membership is declared in core and a test
# (tests/unit/core/metrics/regions/test_eligibility.py) asserts it stays in sync
# with METRIC_SPECS -- drift fails the suite rather than silently un-declaring a
# metric.

_NOT_RESTRICTABLE_GROUPS: dict[str, tuple[str, tuple[str, ...]]] = {
    "fov_level": (
        "consumes the background/air of the full FOV, which a tissue ROI contains none "
        "of. fber Otsu-splits into air vs tissue -- inside a lesion crop there is no air, "
        "so it splits tissue against tissue or returns the fabricated constant 1.0. "
        "ghosting_ratio measures signal at a displacement of FOV/2 *of the tensor it is "
        "handed*: on a 40x40 crop that is 20 px, a physically different displacement from "
        "the acquisition's, and when the crop is all-signal its background mask is empty "
        "and it returns a fabricated 0.0 with status OK.",
        ("fber", "ghosting_ratio"),
    ),
    "spectral": (
        "an image-space crop's FFT is a convolution with a sinc, not a restriction "
        "of k-space: the crop does not compute the ROI's spectrum, it computes a "
        "smeared spectrum of the whole slice.",
        (
            "bpci",
            "csr",
            "dss",
            "focal_frequency",
            "hfrw",
            "ksk",
            "kspace_error",
            "power_spectrum_consistency",
            "radial_k_error",
            "sher",
            "super_nyquist_fidelity",
        ),
    ),
    "phase": (
        "consumes image/k-space phase, which the engine synthesises through a "
        "Fourier bridge; the same sinc argument as the spectral family applies -- an "
        "image crop does not restrict the phase measurement.",
        ("ipen", "phase_mse"),
    ),
    "distributional": (
        "a population statistic over a feature distribution (FID/KID/IS class): "
        "defined over a *sample of images*, not over a sub-region of one.",
        (
            "fid",
            "frd",
            "inception_score",
            "kernelised_stein_discrepancy",
            "kid",
            "med_fid",
            "mmd_metric",
            "rfs",
            "sliced_wasserstein",
            "srf_bound",
            "wasserstein_1d",
        ),
    ),
    "mri_physics": (
        "consumes the measurement or its geometry (k-space / trajectory / coil maps / "
        "sampling mask); an image-space ROI cannot restrict a k-space measurement.",
        (
            "asymptotic_gfactor",
            "g_factor",
            "k_space_trajectory_rmse",
            "nema_cnr",
            "nema_snr",
            "qi1",
            "residual_whiteness",
            "spike_detection",
            "topological_mask_certificate",
            "wm2max",
        ),
    ),
    "model_level": (
        "a property of the MODEL or of a whole sample of paired measurements, not of "
        "an image region. 'inference_latency_ms restricted to grey matter' is not a "
        "quantity -- it is a category error.",
        (
            "adaptive_ece",
            "adversarial_psnr_drop",
            "bland_altman_bias",
            "brier_score",
            "classwise_ece",
            "composed_spectral_norm_bound",
            "conformal_risk_control",
            "equivariance_defect",
            "expected_calibration_error",
            "galois_h1_certificate",
            "gaussian_nll",
            "icc_3_1",
            "inference_latency_ms",
            "input_dependence_spread",
            "limits_of_agreement_lower",
            "limits_of_agreement_upper",
            "max_layer_spectral_norm",
            "mrf_persistence_stability",
            "nll_bits_per_dim",
            "ood_acceleration_shift",
            "pac_bayes_certificate",
            "param_count",
            "qmap_conformal_coverage",
            "subgroup_psnr_disparity",
            "subgroup_ssim_gap",
        ),
    ),
    "parametric_map": (
        "defined on a parametric / non-magnitude map (perfusion, diffusion, flow, "
        "B0 field, MRS, qMRI), not on a magnitude ROI.",
        (
            "adc_mae",
            "att_mae",
            "b0_field_rmse",
            "bat",
            "bloch_manifold_residual",
            "cbf_rmse",
            "cc_snr",
            "cosine_preservation_score",
            "crlb",
            "cross_scanner_t1t2_concordance",
            "divergence",
            "freq_domain_snr",
            "geodesic_fc_error",
            "geodesic_mrf_parameter_error",
            "geodesic_qmap_error",
            "ktrans",
            "mass_conservation",
            "ndc_diffusion",
            "negative_voxels",
            "net_flow_error",
            "peak_velocity_error",
            "phys_residual_consistency",
            "spectral_linewidth",
            "spike_percentage",
            "velocity_rmse",
            "vnr",
            "wash_slope",
        ),
    ),
    "hallucination": (
        "requires a reference distribution / posterior sample set, not a sub-region "
        "of a single image.",
        ("fabrication_rate", "feature_fidelity_index"),
    ),
    "segmentation": (
        "consumes label maps (segmentation overlap / boundary distance / volume "
        "agreement). Restricting a Dice or HD95 to a spatial ROI is a DIFFERENT claim "
        "from the whole-structure one this metric makes -- a lesion bbox severs the "
        "structure and its boundary, so HD95 loses the boundary points it measures and "
        "volume similarity measures a truncated volume -- and needs its own written "
        "justification before it is enabled; an empty table costs nothing, a wrong one "
        "costs a result.",
        (
            "auprc",
            "auroc",
            "average_surface_distance",
            "brain_mask_dice",
            "cohen_kappa",
            "detection_sensitivity",
            "detection_specificity",
            "dice",
            "dvf_mae",
            "folding_fraction",
            "hd95",
            "iou",
            "synthseg_dice",
            "synthseg_dice_risk",
            "target_registration_error",
            "tissue_dice",
            "tissue_hd95",
            "tissue_volume_similarity",
            "volume_consistency",
        ),
    ),
    "temporal_series": (
        "consumes a time axis the (pred, target) image pair does not carry: these read "
        "[B, T, H, W] with T >= 2 on axis 1 and are undefined on the single frame a "
        "spatial ROI restricts to. Regional tSNR is a real and standard fMRI quantity, "
        "but it is obtained by handing the region in as an explicit foreground mask "
        "(TemporalSNR's mask= kwarg), NOT by cropping: tsnr Otsu-splits air vs tissue "
        "to locate its foreground, so inside an all-tissue ROI it splits tissue against "
        "tissue and fabricates one -- the same trap as fber in fov_level above.",
        (
            "temporal_fidelity",
            "tsnr",
        ),
    ),
    "topology": (
        "persistent homology is a global invariant: a crop cuts connected components "
        "and loops at its border and creates new ones, so the diagram of a crop is not "
        "the restriction of the image's diagram. persistence_diameter is not computed on "
        "an image at all but on a [T, d] signal trajectory.",
        ("cubical_ph_w2_distance", "persistence_diameter"),
    ),
    "volumetric": (
        "needs a through-plane axis ([D, H, W]); a region restricts one 2-D slice, on "
        "which the through-plane line-spread function is undefined.",
        ("through_plane_fwhm",),
    ),
    "pre_masked": (
        "already a region metric: it averages over the banding / artefact mask its "
        "caller supplies. Restricting it to another ROI would intersect the two masks, "
        "which is a different claim; the regional squared error is mse in the map tier.",
        ("banding_region_mse",),
    ),
    "task_observer": (
        "a channelized Hotelling observer estimates its channel covariance from 32 px "
        "tiles of the residual and scales the inserted lesion by the target's 95th "
        "percentile. A ~40 px lesion box holds one tile, below the two realisations the "
        "estimate needs, and a crop makes the lesion contrast ROI-local, so the regional "
        "d' answers a different question from the full-slice one. Declined until a "
        "regional observer is specified.",
        ("task_based_detectability",),
    ),
}

# Deep-network perceptual metrics: backbones downsample ~5x, so below ~64 px/side
# the deepest feature map is a single cell and the comparison degenerates. Declared
# conservatively and stated, rather than guessed precisely.
# Only lpips and dists run an ImageNet backbone (~5 dyadic downsamples), so only they
# need the 64 px floor. The rest of this group is CLOSED-FORM -- haarpsi is a Haar wavelet
# index, mdsi gradient+chromaticity, vsi a spectral-residual saliency index, mad/st_mad
# Larson & Chandler MAD on 16x16 blocks, pdm Daly's VDP. All of them compute cleanly on a
# 40x40 crop (measured: 0.7443 / 0.3122 / 0.9732), and the gate declined them anyway on a
# rationale ("backbones downsample ~5x") that does not apply to them -- silently dropping
# six metrics at exactly the ~40 px lesion scale that IS the experiment. Their real floors
# come from min_side_for().
_DEEP_NET_MIN_SIDE = 64
_DEEP_NET = ("lpips", "lpips_alex", "dists")
_PERCEPTUAL = (
    "dists",
    "haarpsi",
    "lpips",
    "lpips_alex",
    "mad",
    "mdsi",
    "pdm",
    "st_mad",
    "vsi",
)

# No-reference statistics over the prediction alone. NSS-style metrics fit a model
# on local patches; below a few thousand pixels the fit is dominated by its own
# variance.
_PATCH_STAT_MIN_PX = 1024
_NO_REFERENCE = (
    "bemd_ier",
    "blockiness",
    "brenner_focus",
    "brisque",
    "efc",
    "gibbs_ringing",
    "gradient_entropy",
    "high_freq_energy_ratio",
    "immerkaer_noise",
    "intensity_entropy",
    "laplacian_variance",
    "mlv",
    "niqe",
    "normalized_gradient_squared",
    "nr_iqa",
    "pgle",
    "pid",
    "tenengrad_variance",
    "total_variation",
    "wlms",
    "wpde",
    "zipper_detection",
)

# Scalar full-reference reductions with no spatial map exposed -- well-defined on a
# crop, so they crop.
_CROP_SCALAR = (
    # structural
    "clinical_ssim",
    "cw_ssim",
    "fsim",
    "gmsd",
    "iw_ssim",
    "ms_gmsd",
    "ms_ssim",
    "srsim",
    "uqi",
    # edge / high-frequency
    "ciea",
    "complex_hfen",
    "edge_preservation_index",
    "gradient_error",
    "gri",
    "hfen",
    "pcd",
    "rar",
    "vif",
    "vif_p",
    # pixel-error variants with no map implemented (see _MAP_TIER for the rest).
    # nrmse_l2 shares nrmse's global denominator (||target||), which has no verified
    # per-pixel (maps, reduce) decomposition in reductions.py -- so it crops, it does
    # not map. FULL_SLICE would be a lie here (the crop tier has no full-slice constant).
    "nrmse",
    "nrmse_l2",
    "robust_mri_psnr",
    # statistical scalars genuinely defined on a sub-image
    "cjv",
    "cnr",
    "coefficient_of_variation",
    "cosine_similarity",
    "mutual_information",
    "pearson",
    "snr",
)

# Crop-tier size floors backed by hard evidence, not by priors.
#
# A hand-kept table is the wrong shape for this: it drifts from the metrics it is
# supposed to describe, and it did. `_MIN_SIDE_OVERRIDES` declared ms_ssim's 161 and
# stopped there -- but `iw_ssim` carries the SAME 161 floor, declared on its own class
# (perceptual_piq.py: `_MIN_SIZE = 161`), and `_PIQMetric._maybe_upsample` BILINEARLY
# INTERPOLATES a smaller pair up to it. So `iw_ssim` on a 40x40 lesion crop returned a
# confident 0.9323 -- computed on invented high-frequency detail, which is precisely the
# fabrication this package exists to decline (see crop.py's module docstring). A fastMRI+
# lesion box is ~40 px, so that was every lesion cell in the leaderboard.
#
# So: read the floor OFF THE METRIC where the metric declares one (`_MIN_SIZE`), and keep
# the table only for backends that raise or fabricate WITHOUT declaring a floor. Each
# entry below is a measured value, not a prior.
_MIN_SIDE_OVERRIDES: dict[str, int] = {
    # 5 dyadic scales x an 11-tap window => (11-1)*2**4 + 1 = 161 px/side. The sim2rank
    # engine now READS this floor (via min_side_for) rather than keeping its own copy.
    "ms_ssim": 161,
    # piq raises "Invalid size of the input images, expected at least 41x41".
    "vif_p": 41,
    # piq raises "... at least 17x17".
    "ms_gmsd": 17,
    # piq raises "Kernel size can't be greater than actual input size" below ~21.
    "srsim": 21,
    # NIQE's patch loop is `range(0, h - patch_size + 1, stride)` with patch_size=96
    # (no_reference_metrics.py:201). Below 96 px the loop yields NOTHING and the metric
    # returns a FABRICATED all-zero feature vector -> a hard 0.0 with status OK.
    "niqe": 96,
    # Larson & Chandler MAD works on 16x16 blocks (perceptual_advanced.py:334).
    "mad": 16,
    "st_mad": 16,
}


def min_side_for(key: str, default: int = 0) -> int:
    """The strictest floor known for ``key``: the metric's own, the measured one, or the default.

    `_PIQMetric` subclasses declare `_MIN_SIZE` on the class and silently BILINEARLY
    UPSAMPLE anything smaller (perceptual_piq.py `_maybe_upsample`). Reading that value
    straight off the registered class keeps the gate and the metric from drifting apart:
    a new piq metric that carries a floor is covered the day it is registered, with no
    edit here. `MetricsRegistry._metrics` maps name -> class, so nothing is instantiated.

    This is the **only** size-floor table in the system, and it is deliberately public:
    the sim2rank engine applies the same floor to the FULL-FOV sweep. It used to keep a
    private `_MIN_SPATIAL = {"ms_ssim": 161}` -- one rule, two tables, already drifted
    (`iw_ssim` carries the same 161 floor and fabricates below it, and the engine's table
    had never heard of it). A floor is a property of the METRIC, not of the region, so it
    lives here and both callers read it.

    Returns 0 when nothing declares a floor -- "no floor known", not "no floor exists".
    """
    from spectramr.core.metrics.registry import MetricsRegistry

    cls = MetricsRegistry._metrics.get(key)
    declared = int(getattr(cls, "_MIN_SIZE", 0) or 0) if cls is not None else 0
    return max(declared, _MIN_SIDE_OVERRIDES.get(key, 0), default)


def _build_policy_table() -> dict[str, RoiPolicy]:
    """Assemble ROI_POLICY from the declared groups + the live registry."""
    from spectramr.core.metrics.registry import MetricsRegistry, list_available

    table: dict[str, RoiPolicy] = {}

    def put(key: str, policy: RoiPolicy) -> None:
        if key in table:
            raise ValueError(f"{key!r} is declared in two ROI-policy groups")
        table[key] = policy

    # (1) Declared not-restrictable groups. Each justification is an argument from
    #     the metric's definition, not an opinion about its receptive field.
    for why, keys in _NOT_RESTRICTABLE_GROUPS.values():
        for key in keys:
            put(key, RoiPolicy(support=RoiSupport.NOT_RESTRICTABLE, justification=why))

    # (2) Map tier -- only where reductions.py implements it AND a test proves
    #     reduce(maps(p, t), ones) == the unrestricted registry metric.
    for key, norm in _MAP_TIER.items():
        put(
            key,
            RoiPolicy(
                support=RoiSupport.MAP_THEN_MASK,
                justification=(
                    "has a per-pixel error map; the map is reduced over the mask. "
                    "Verified against the unrestricted metric on an all-ones mask."
                ),
                normalisation=norm,
            ),
        )

    # (3) Crop tier.
    for key in _PERCEPTUAL:
        put(
            key,
            RoiPolicy(
                support=RoiSupport.CROP_THEN_COMPUTE,
                justification=(
                    "deep-network feature comparison with no spatial map; scores the "
                    "tight bbox crop."
                ),
                min_roi_side=min_side_for(key, _DEEP_NET_MIN_SIDE if key in _DEEP_NET else 0),
                normalisation=Normalisation.ROI_LOCAL,
            ),
        )
    for key in _NO_REFERENCE:
        put(
            key,
            RoiPolicy(
                support=RoiSupport.CROP_THEN_COMPUTE,
                justification=(
                    "no-reference statistic over the prediction; scores the tight "
                    "bbox crop of the prediction alone."
                ),
                min_roi_side=min_side_for(key),
                min_roi_px=_PATCH_STAT_MIN_PX,
                normalisation=Normalisation.NOT_APPLICABLE,
            ),
        )
    for key in _CROP_SCALAR:
        put(
            key,
            RoiPolicy(
                support=RoiSupport.CROP_THEN_COMPUTE,
                justification=(
                    "scalar full-reference reduction with no spatial map exposed; "
                    "scores the tight bbox crop."
                ),
                min_roi_side=min_side_for(key),
                normalisation=Normalisation.ROI_LOCAL,
            ),
        )

    # (4) The strongest rule in the file, and the only AUTO-derived one: a metric
    #     that needs a MetricContext consumes the *measurement* (k-space, coil maps,
    #     sampling mask). An image-space crop is not a k-space restriction -- that is
    #     a category error, not an approximation. Full stop.
    #
    #     Derived straight from the registry, so it cannot drift out of sync with the
    #     metric's own declaration, and a NEW context metric is covered the day it is
    #     registered. Applied last so it overrides any group a metric was placed in.
    context_policy = RoiPolicy(
        support=RoiSupport.NOT_RESTRICTABLE,
        justification=(
            "needs a MetricContext (k-space / coil maps / sampling mask): an "
            "image-space ROI cannot restrict a k-space measurement."
        ),
    )
    for key in list_available():
        if MetricsRegistry.needs_context(key):
            table[key] = context_policy

    return table


_POLICY_TABLE: dict[str, RoiPolicy] | None = None
_BUILT_AT_REGISTRY_SIZE = -1


def _registry_size() -> int:
    """How many metrics are registered right now. O(1) -- this sits on the hot path."""
    from spectramr.core.metrics.registry import MetricsRegistry

    return len(MetricsRegistry._metrics)


def policy_table() -> dict[str, RoiPolicy]:
    """The policy table, built lazily and rebuilt whenever the registry has grown.

    **Deliberately not a module-level constant.** It used to be
    ``ROI_POLICY = _build_policy_table()``, evaluated at import -- and it read a registry
    that was still filling. ``spectramr/core/metrics/__init__.py`` imports this package
    partway through its own registration pass, so the table froze at **137 of 201**
    metrics. Two things then went quietly wrong, and both depended on nothing but import
    order, which is why the suite passed in one invocation and failed in another:

    * every floor read off a metric CLASS came out as **0**, because the class was not in
      the registry yet. ``iw_ssim`` declares ``_MIN_SIZE = 161``; its policy floor was 0;
      so the gate **accepted a 40x40 lesion crop**, the piq backend bilinearly upsampled
      it to 161x161, and the metric returned a confident score computed on interpolated
      detail. That is the exact fabrication this package exists to decline, running inside
      the thing built to prevent it.
    * the 27 context metrics (``ndcr``, ``are``, ``csoe``, ``prp_snr``, …) register late,
      so they were **absent** from the table and ``policy_for`` raised ``KeyError``
      mid-sweep rather than declining them.

    Rebuilding when the registry size changes makes it self-healing: whenever the table is
    consulted, it reflects the registry as it is *now*, not as it was during import.
    """
    global _POLICY_TABLE, _BUILT_AT_REGISTRY_SIZE
    n = _registry_size()
    if _POLICY_TABLE is None or n != _BUILT_AT_REGISTRY_SIZE:
        _POLICY_TABLE = _build_policy_table()
        _BUILT_AT_REGISTRY_SIZE = n
    return _POLICY_TABLE


def __getattr__(name: str) -> object:
    """PEP 562: keep ``ROI_POLICY`` as a public name, but resolve it lazily.

    A plain module constant would re-freeze the bug above for anyone who does
    ``from ... import ROI_POLICY`` at import time.
    """
    if name == "ROI_POLICY":
        return policy_table()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def policy_for(metric: str) -> RoiPolicy:
    """The declared policy for ``metric``.

    Raises:
        KeyError: when the metric has no declared policy. There is **no default** --
            "assume crop" would silently score new metrics on 40 px crops, and
            "assume none" would make them silently vanish from every regional
            leaderboard. Both are fabrications; a raise is a one-line fix.
    """
    try:
        return policy_table()[metric]
    except KeyError:
        raise KeyError(
            f"metric {metric!r} has no declared ROI policy. Declare one in "
            "core/metrics/regions/eligibility.py -- there is deliberately no default, "
            "because every possible default silently fabricates or silently drops."
        ) from None


def evaluate_eligibility(metric: str, region: RegionMask) -> Eligibility:
    """Is ``metric`` defined on ``region``? A pure function of policy + geometry.

    No tensors, no metric calls -- so a run can emit its **complete exclusion table
    before it computes anything**.
    """
    policy = policy_for(metric)
    n_px, min_side = region.n_px, region.min_side

    def verdict(
        eligible: bool,
        reason: NotApplicableReason | None = None,
        detail: str = "",
    ) -> Eligibility:
        return Eligibility(
            metric=metric,
            region_id=region.region_id,
            eligible=eligible,
            support=policy.support,
            n_px=n_px,
            min_side=min_side,
            reason=reason,
            detail=detail,
            normalisation=policy.normalisation,
        )

    if policy.support is RoiSupport.NOT_RESTRICTABLE:
        return verdict(
            False,
            NotApplicableReason.MEASUREMENT_NOT_ROI_RESTRICTABLE,
            policy.justification,
        )

    # NOTE: the `full` identity region gets NO exemption from the size floors.
    #
    # It used to short-circuit to `eligible=True` here, on the reasoning that full IS the
    # unrestricted metric. That reasoning holds for the *rectangularity* rule (full is a
    # filled box, so it passes anyway) but NOT for the size floors, and the exemption
    # opened a fabrication hole at exactly the place it would be least noticed: on a
    # 128x128 slice, `full` handed ms_ssim and iw_ssim an image below their 161 px floor,
    # both of which bilinearly upsample and return a confident score computed on invented
    # detail -- and `full` is the control cell that every regional ranking is compared
    # against. Meanwhile the sim2rank engine, reading the same floor, declined the very
    # same call. One rule, two answers, and the fabricating one fed the leaderboard.
    #
    # A floor is a property of the metric, not of the region. It applies everywhere.
    if policy.min_roi_side and min_side < policy.min_roi_side:
        return verdict(
            False,
            NotApplicableReason.ROI_TOO_SMALL_SIDE,
            f"{min_side} < {policy.min_roi_side} px minimum side",
        )

    if policy.min_roi_px and n_px < policy.min_roi_px:
        return verdict(
            False,
            NotApplicableReason.ROI_TOO_SMALL_AREA,
            f"{n_px} < {policy.min_roi_px} px minimum area",
        )

    # The crop-tier / non-rectangular hazard: the bbox of a non-rectangular region
    # is a strict superset of the region, so cropping would score the BBOX, not the
    # tissue. Better to decline than to answer a different question.
    if policy.support is RoiSupport.CROP_THEN_COMPUTE and not region.is_rectangular:
        return verdict(
            False,
            NotApplicableReason.ROI_NOT_RECTANGULAR,
            "crop-tier metric on a non-rectangular region: the bbox is a strict "
            "superset of the region, so the crop would score the bbox, not the region.",
        )

    return verdict(True)
