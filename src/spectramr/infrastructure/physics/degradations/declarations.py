"""Declared facts about the taxonomy as a whole: image components, collinear pairs, theta=0 pedestals."""

from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "BOTH",
    "DECLARED_PHYSICAL_FOV_MM",
    "DECLARED_THETA_ZERO_PEDESTAL",
    "KNOWN_COLLINEAR_PAIRS",
    "MAG",
    "MAGNITUDE",
    "PHASE",
    "PHASE_ONLY",
    "is_known_collinear",
]

MAGNITUDE: str = "magnitude"
PHASE: str = "phase"

BOTH = frozenset({MAGNITUDE, PHASE})
MAG = frozenset({MAGNITUDE})
PHASE_ONLY = frozenset({PHASE})


# ──────────────────────────────────────────────────────────────────────
# Declared non-orthogonal axis pairs (issue #245)
#
# The taxonomy is *presented* as a set of orthogonal degradation axes, and the
# per-family aggregations (Borda sum, CDRS variance penalty) assume as much. The
# pairs below are the ones that measurably are NOT orthogonal — |cos| of their
# theta=1 residuals exceeds 0.9 — and cannot be made so without fabricating
# physics the 2-D magnitude data cannot carry. They are declared here rather than
# hidden in a test's skip list, so a reader of the results can see the double
# counting, and so a NEW collinear pair fails the orthogonality test loudly.
#
# The one pair that WAS a genuine duplicate (t2star_blur / slice_profile, |cos| =
# 0.998 — the same isotropic Gaussian blur registered twice) is not on this list:
# it was deleted. See the D27 note in ``slice2d/blur.py``.
# ──────────────────────────────────────────────────────────────────────

# The LOW-PASS FAMILY. Four axes with genuinely different physics and genuinely
# different operators — a Gaussian PSF (D12), an anisotropic through-plane PVE
# proxy (D13), a circular k-space window (D9), and a centre-peaked undersampling
# mask that at high R keeps almost only the k-space centre (D4) — but whose
# theta=1 residuals are all, to leading order, "the high-frequency content,
# removed". They cannot be made orthogonal; they can only be declared. Their
# pairwise |cos| runs 0.79-0.95 and is anatomy- and grid-size-dependent, so the
# pairs that cross the 0.9 guard on one image sit just under it on another.
_LOW_PASS_FAMILY = (
    "Both are low-pass operators. D9 (gibbs) is a circular k-space window, D12 "
    "(t2star_blur) an isotropic Gaussian PSF, D13 (slice_pve) an anisotropic "
    "readout-axis blur, and D4 (vd_undersamp) at high R keeps little but the "
    "k-space centre. Different physics, different operators — but every one of "
    "their residuals is dominated by the same thing: the missing high frequencies. "
    "This is a real limit of the taxonomy, not a bug to be tuned away, and it is "
    "declared so the per-family aggregations can be read with it in mind."
)

_CS_RECON_RESIDUAL = (
    "cs_recon_artifacts reconstructs a Cartesian PE-line undersampled scan with a 1 % of p99 "
    "wavelet threshold, so its result stays data-consistent at the acquired lines and its residual "
    "is the incoherent PE-line aliasing of the undersampling itself (|cos| 0.92-0.93 on the guard "
    "phantom); the prior only removes part of it and adds blocky Haar texture. A stronger "
    "threshold would separate them but its dips against R grow (measured), so the shared "
    "'aliasing of the same lines' residual is declared."
)

KNOWN_COLLINEAR_PAIRS: Mapping[tuple[str, str], str] = {
    ("slice_pve", "t2star_blur"): _LOW_PASS_FAMILY,
    ("gibbs", "t2star_blur"): _LOW_PASS_FAMILY,
    ("gibbs", "vd_undersamp"): _LOW_PASS_FAMILY,
    ("iq_ghost", "stimulated_echo"): (
        "Both add a low-amplitude replica of the object to itself — D25 a "
        "conjugate mirror, D26 a PE-shifted copy. On near-symmetric anatomy, and "
        "with D26's shift capped at 6 px, both residuals reduce towards a scalar "
        "multiple of the object. Raising D26's displacement to a clinically "
        "realistic fraction of the FOV would separate them; that is a severity "
        "recalibration, tracked separately."
    ),
    ("cartesian_undersamp", "cs_recon_artifacts"): _CS_RECON_RESIDUAL,
    ("cs_recon_artifacts", "vd_cartesian"): _CS_RECON_RESIDUAL,
    ("recon_apodization", "zero_fill_interp"): (
        "Both are separable k-space low-passes of the reconstruction: a smooth Hamming/Hann/Fermi "
        "taper versus a hard central box. Their residuals are the same removed high frequencies "
        "(|cos| 0.94 on the guard phantom, 0.90 with the Fermi window); they differ in the "
        "truncation shape (taper: blur without ringing, box: sinc ringing), which the magnitude "
        "residual of a smooth object barely resolves. Neither is collinear with gibbs "
        "(|cos| 0.66 and 0.53, a radial cut-off) or resolution_snr (0.82 and 0.78)."
    ),
    ("cartesian_undersamp", "vd_cartesian"): (
        "Both are incoherent Cartesian PE-line undersampling with a dense k-space "
        "centre: D4b draws the outer lines with a variable-density profile "
        "(1+|ky|)^-alpha, D3 draws them uniformly at random, but at matched R the "
        "aliasing residual of each is the same broadband incoherent-Cartesian "
        "pattern (|cos| ~0.99). Steepening D4b's density toward a low-pass profile "
        "would separate them but re-introduce the resolution-loss facade this axis "
        "was repaired (advisor) to remove, so the near-duplication is declared "
        "rather than manufactured away."
    ),
    ("iq_ghost", "sar_limited_flip"): (
        "Both residuals reduce towards a smooth multiple of the object on the sweep image: "
        "iq_ghost's central replica and sar_limited_flip's gain at the effective echo "
        "(|cos| ~0.94). The ky-dependent PSF part of sar_limited_flip is what distinguishes it."
    ),
    ("adc_clipping", "sar_limited_flip"): (
        "Both residuals are dominated by a smooth change of the object's signal level on the "
        "sweep image: clipping removes part of the k-space centre, and sar_limited_flip lowers "
        "the signal at the effective echo (|cos| ~0.95). They differ in the high-frequency part "
        "(clipping ringing versus the echo train's ky-dependent PSF)."
    ),
    ("chemical_shift_phase", "sar_limited_flip"): (
        "Both residuals are a smooth change of the object's signal level on the sweep images: "
        "the water-fat phase change without a fat map, and the signal lost at the effective "
        "echo (|cos| 0.89 in 2-D, 0.95 in multislice and 3-D). The echo train's ky-dependent "
        "PSF and a real fat distribution separate them."
    ),
    ("flow_voids", "inflow_enhancement"): (
        "Both act only inside the same seeded vessel map (one draw per sample from the seed), "
        "voids scaling the vessel signal down by washout and inflow scaling it up by fresh "
        "magnetisation, so their residuals share one support with opposite signs (|cos| ~0.99 "
        "in multislice and 3-D). They differ in sign, velocity law and slice-order dependence, "
        "which a measured vessel map or a signed score separates."
    ),
    # slice_crosstalk saturates each slice by its neighbours' profile tails: a smooth per-slice
    # gain below 1, so on the volume phantom its residual is a multiple of |x|, like these.
    ("adc_clipping", "slice_crosstalk"): (
        "On the volume phantom both residuals are a loss of object signal close to a multiple "
        "of |x|: clipping removes part of the k-space centre, crosstalk lowers each slice by a "
        "smooth saturation gain (|cos| ~0.91, multislice). Clipping adds the ringing of the cut "
        "samples; crosstalk varies with the slice order and gap."
    ),
    ("chemical_shift_phase", "slice_crosstalk"): (
        "On the volume phantom, with no separate fat map, chemical_shift_phase's in/opposed-phase "
        "signal change is a smooth change of the object's level, as is crosstalk's per-slice "
        "saturation gain (|cos| ~0.95, multislice). They separate with a real fat distribution "
        "or a slice order that changes the crosstalk interval."
    ),
    ("sar_limited_flip", "slice_crosstalk"): (
        "sar_limited_flip lowers the signal at the effective echo (a gain on the object) and "
        "crosstalk lowers each slice by a smooth saturation gain (|cos| ~0.98, multislice). The "
        "echo train's ky-dependent PSF and the slice-order dependence are what separate them."
    ),
}


def is_known_collinear(a: str, b: str) -> bool:
    """True when ``(a, b)`` is a declared non-orthogonal pair (order-insensitive)."""
    return tuple(sorted((a, b))) in KNOWN_COLLINEAR_PAIRS


# ──────────────────────────────────────────────────────────────────────
# Declared non-identity clean anchors (issue #246)
#
# ``D(x, theta=0)`` must equal ``x``. Every statistic in the sweep is anchored on
# that clean baseline, so a degrader that quietly corrupts its own theta=0 output
# biases everything downstream. ``gibbs`` (a k-space window that zeroed 22.8% of
# the samples at its nominal "no truncation" setting) and ``radial_undersamp``
# (200 spokes that do not cover the grid) both did exactly that, with pedestals
# of 0.055 and 0.069 — larger than the ENTIRE theta=1 distortion of
# ``partial_fourier``. Both are fixed; the invariant is now pinned for every axis
# by ``TestCleanAnchor``.
#
# One axis has a legitimately non-identity theta=0, and it is DECLARED here
# rather than silently tolerated by the test: ``complex_gaussian``'s severity IS
# the SNR, and its SeveritySpec declares theta=0 to be **40 dB**, not infinity —
# a clean clinical scanner, not a noiseless one. The residual is bounded and
# intentional, so the bound is written down.
# ──────────────────────────────────────────────────────────────────────

DECLARED_THETA_ZERO_PEDESTAL: Mapping[str, float] = {
    # name -> max tolerated rel-L2 of D(x, 0) against x
    "complex_gaussian": 0.02,  # 40 dB SNR at theta=0, per its SeveritySpec
}


#: Axes whose effect is set by the physical FOV, not the pixel grid: name -> the in-plane FOV (mm)
#: a suite must give its phantom. The calibrated dielectric standing wave varies on the RF
#: wavelength (332 mm at 3 T), so on a few-cm unit-voxel phantom it vanishes, as it physically does.
DECLARED_PHYSICAL_FOV_MM: Mapping[str, float] = {"dielectric_b1": 192.0}
