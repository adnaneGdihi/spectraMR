"""The reconstruction family: g-factor noise, residual aliasing, CS artefacts, apodisation, zero-filling, gradwarp.

Every kernel is native in the encoded geometry it needs. Apodisation, zero-filling and CS act on each
slice's own 2-D k-space, so a multislice stack is exactly the slice kernel run per slice.
"""

from __future__ import annotations

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations.catalogue.recon_conditions import (
    gfactor_condition,
    pi_residual_condition,
)
from spectramr.infrastructure.physics.degradations.catalogue.recon_cs import (
    cs_recon_2d,
    cs_recon_volume,
)
from spectramr.infrastructure.physics.degradations.catalogue.recon_gradwarp import (
    gradwarp_residual_2d,
    gradwarp_residual_volume,
)
from spectramr.infrastructure.physics.degradations.catalogue.recon_kspace import (
    apodization_2d,
    apodization_volume,
    zero_fill_2d,
    zero_fill_volume,
)
from spectramr.infrastructure.physics.degradations.catalogue.recon_sense import (
    gfactor_noise_2d,
    gfactor_noise_volume,
    pi_residual_2d,
    pi_residual_volume,
)
from spectramr.infrastructure.physics.degradations.declarations import BOTH
from spectramr.infrastructure.physics.degradations.families import MS, VOL, exact
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionMode
from spectramr.infrastructure.physics.degradations.spec import (
    ActivationGate,
    DegradationSpec,
    Realisation,
    Requirement,
    SliceKernel,
    VoxelSliceKernel,
    entry,
)
from spectramr.infrastructure.physics.severity import PhysicalParam, SeveritySpec

__all__ = ["ENTRIES"]

_P = PhysicalParam
_N = Realisation.NATIVE
_SENSE_NOTE = (
    " R = (R_pe, R_slice) is the profile's acceleration (kwarg acceleration, default (2, 1)); R_slice "
    "aliases slices (multislice) or kz (3-D) with a CAIPI shift of 1 when both exceed 1. Coil maps are "
    "the coil_maps kwarg, else seeded synthetic Gaussian-lobe maps; the noise covariance is the "
    "profile's coil_covariance, else identity."
)


def _spec(
    code, name, kernel_2d, volume, *, description, severity, mode_decls=None, **kw
) -> DegradationSpec:
    """Declare an axis by its 2-D kernel and a volume impl (or per-mode ``(impl, realisation)`` decls)."""
    decls = mode_decls or {MS: (volume, _N), VOL: (volume, _N)}
    return entry(
        code=code,
        name=name,
        family="reconstruction",
        description=description,
        severity=severity,
        affects=BOTH,
        origin=kw.pop("origin", ArtifactOrigin.ACQUISITION),
        implementations={
            AcquisitionMode.SLICE_2D: kernel_2d,
            **{m: impl for m, (impl, _) in decls.items()},
        },
        realisation={AcquisitionMode.SLICE_2D: _N, **{m: r for m, (_, r) in decls.items()}},
        requires=kw.pop("requires", frozenset()),
        condition=kw.pop("condition", None),
        active_when=kw.pop("active_when", frozenset()),
    )


_ACCEL = frozenset({ActivationGate.ACCELERATED})

ENTRIES = (
    _spec(
        "D64",
        "gfactor_noise",
        VoxelSliceKernel(gfactor_noise_2d),
        gfactor_noise_volume,
        description="SENSE g-factor noise amplification at the scanner's acceleration",
        severity=SeveritySpec(
            (_P("noise_rms", "x p99 signal", 0.0, 0.1),),
            notes="complex noise of std sigma_0 g(r) sqrt(R) added to the image, sigma_0 = 0.1 theta x "
            "p99(|x|); g = sqrt([(S^H Psi^-1 S)^-1]_rr [S^H Psi^-1 S]_rr) over the aliased voxel set "
            "(Pruessmann 1999); R = 1 gives g = 1." + _SENSE_NOTE,
        ),
        condition=gfactor_condition,
        active_when=_ACCEL,
    ),
    _spec(
        "D65",
        "pi_residual_aliasing",
        VoxelSliceKernel(pi_residual_2d),
        pi_residual_volume,
        description="Residual aliasing of SENSE unfolded with miscalibrated coil maps",
        severity=SeveritySpec(
            (_P("coil_map_error", "max |eps| of S' = S (1 + eps)", 0.0, 0.2),),
            notes="the coil images plus S (m' - m): the exact unfolding m of the data aliased with the "
            "true maps S, subtracted from the unfolding m' with S' = S (1 + eps); eps a smooth seeded "
            "real field per coil with max |eps| = 0.2 theta."
            + _SENSE_NOTE
            + " Needs multi-coil input (the channels are the coils). Without the coil_maps kwarg the "
            "residual is taken against seeded synthetic maps the data are not consistent with; pass "
            "coil_maps for a physically matched residual.",
        ),
        requires=frozenset({Requirement.MULTICOIL}),
        condition=pi_residual_condition,
        active_when=_ACCEL,
    ),
    _spec(
        "D66",
        "cs_recon_artifacts",
        SliceKernel(cs_recon_2d),
        cs_recon_volume,
        description="Compressed-sensing reconstruction artefacts (ISTA, Haar wavelet prior)",
        severity=SeveritySpec(
            (_P("acceleration", "x", 1.0, 6.0),),
            notes="theta sets R = 1 + 5 theta on nested variable-density masks ((ky) or (kz, ky), alpha 3, "
            "8 % calibration). The Haar soft threshold (coarse band excepted) is 0.01 p99(|x|) times "
            "min(1, theta / 0.05): fixed above a short ramp, so theta -> 0+ is continuous and only R "
            "grows over most of the axis (a threshold that ramps with theta competes with R and made "
            "the severity dip, measured over seeds). 20 ISTA iterations from the zero-filled image, "
            "data consistency through fft_ops. Multislice runs the 2-D kernel per slice with one "
            "ky mask, but the lambda reference p99 is the subject's, not each slice's; volumes use "
            "the 3-D Haar.",
        ),
        mode_decls={MS: exact(cs_recon_2d), VOL: (cs_recon_volume, _N)},
        active_when=_ACCEL,
    ),
    _spec(
        "D67",
        "recon_apodization",
        SliceKernel(apodization_2d),
        apodization_volume,
        description="Reconstruction k-space apodisation filter (low-pass)",
        severity=SeveritySpec(
            (_P("window_strength", "fraction of the window applied", 0.0, 1.0),),
            notes="k-space multiplied by w_theta = 1 - theta (1 - w), w the window kwarg in {hamming, "
            "hann, fermi} (default hamming), separable over the encoded axes (ky, kx; kz, ky, kx in 3-D).",
        ),
        mode_decls={MS: exact(apodization_2d), VOL: (apodization_volume, _N)},
    ),
    _spec(
        "D68",
        "zero_fill_interp",
        SliceKernel(zero_fill_2d),
        zero_fill_volume,
        description="Reduced-matrix acquisition zero-filled to the full matrix",
        severity=SeveritySpec(
            (_P("truncated_fraction", "fraction of each encoded axis", 0.0, 0.5),),
            notes="the central 1 - f of every encoded axis is acquired (round(n (1 - f)) samples about DC), "
            "the rest zero-filled; f = 0.5 theta. Encoded axes are (ky, kx), or (kz, ky, kx) in 3-D.",
        ),
        mode_decls={MS: exact(zero_fill_2d), VOL: (zero_fill_volume, _N)},
    ),
    _spec(
        "D69",
        "gradwarp_residual",
        VoxelSliceKernel(gradwarp_residual_2d),
        gradwarp_residual_volume,
        description="Residual geometric distortion after vendor gradient-warp correction",
        severity=SeveritySpec(
            (
                _P(
                    "correction_error",
                    "fraction of the gradient-nonlinearity coefficients",
                    0.0,
                    0.5,
                ),
            ),
            notes="net displacement -eps d_c(r) in one resampling, d_c the order 2..3 polynomial "
            "displacement of gradient_nonlinearity at theta = 1 (same seed, same coefficients), "
            "eps = 0.5 theta (eps = 1: no correction, so 0.5 leaves half of the distortion: wrong coil-coefficient file or scanner mismatch); mm -> voxels by voxel_mm (each axis's own FOV without it).",
        ),
        origin=ArtifactOrigin.SCANNER,
    ),
)
