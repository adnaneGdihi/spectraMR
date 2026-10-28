"""The sequence family of the catalogue: field-of-view wrap, TSE and inversion-recovery trains, radial off-resonance."""

from __future__ import annotations

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations.catalogue.sequence_conditions import (
    t1_shot_condition,
    tse_condition,
)
from spectramr.infrastructure.physics.degradations.catalogue.sequence_fov import (
    degrade_fov_foldover,
    fov_foldover_volume,
)
from spectramr.infrastructure.physics.degradations.catalogue.sequence_offres import (
    degrade_offres_blur_noncartesian,
    offres_blur_volume,
)
from spectramr.infrastructure.physics.degradations.catalogue.sequence_t1 import (
    degrade_t1_shot_blur,
    t1_shot_blur_volume,
)
from spectramr.infrastructure.physics.degradations.catalogue.sequence_tse import (
    degrade_tse_t2_blur,
    tse_t2_blur_volume,
)
from spectramr.infrastructure.physics.degradations.declarations import BOTH
from spectramr.infrastructure.physics.degradations.families import MS, VOL, exact, native_2d
from spectramr.infrastructure.physics.degradations.spec import Realisation
from spectramr.infrastructure.physics.severity import PhysicalParam, SeveritySpec

__all__ = ["ENTRIES"]

_P = PhysicalParam

ENTRIES = (
    native_2d(
        code="D56",
        name="fov_foldover",
        family="sequence",
        fn=degrade_fov_foldover,
        description="Wrap-around from a reduced phase-encode field of view",
        severity=SeveritySpec(
            (_P("fov_reduction", "fraction of the object's extent", 0.0, 0.5),),
            notes="the prescribed FOV along the axis is FOV' = E x (1 - 0.5 theta) with E the object's "
            "extent: the span of the support of the projection sum |x| over the other axes "
            "(readout, coils, slices) above 10% of that profile's maximum, measured on the device, "
            "so a background noise floor of a few per cent does not widen it and the wrap scales "
            "with the anatomy. The window is centred on the support's midpoint (as a technologist "
            "centres the prescription). Pixel spacing is fixed and only the line count drops: the "
            "output is the FOV'-wide window holding the aliased sum over k = -2..2 of x(y + k FOV') "
            "(fractional Fourier shifts, exact for FOV' >= E / 2), and zero outside it (those pixels "
            "are not acquired); the window edge is a hard pixel mask and fractional shifts ring on "
            "sharp edges, so severity has small steps in theta. For a smooth object the effect jumps "
            "at theta = 0+ by the size of the folded sub-10% tails (they wrap in, FOV' = E at the 10% "
            "extent; theta = 0 is the declared bypass): for a sigma = 4 px Gaussian the effect at theta = 1e-4 is 13% of that at theta = 1. No interpolation or "
            "magnification. A point at y0 outside the window reappears at y0 - FOV' with the "
            "same intensity. Kwarg axes in {pe, slice, both} (default pe); slice / both apply "
            "the rule along kz with the slab extent, which only a 3-D acquisition encodes, and "
            "raise in slice_2d and multislice_2d.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.ACQUISITION,
        modes={
            MS: exact(degrade_fov_foldover),
            VOL: (fov_foldover_volume, Realisation.NATIVE),
        },
    ),
    native_2d(
        code="D57",
        name="tse_t2_blur",
        family="sequence",
        fn=degrade_tse_t2_blur,
        description="T2 decay across the turbo-spin-echo train",
        severity=SeveritySpec(
            (_P("train_length_over_t2", "ETL x ESP / T2", 0.0, 2.0),),
            notes="line n is acquired at echo e(n) = floor(rank x min(ETL, lines) / lines) of the "
            "train (a train longer than the lines leaves its last echoes unused) and weighted "
            "exp(-(TE_e - TE_centre) / T2_eff), TE_e = (e + 1) ESP, T2_eff = ETL x ESP / (2 theta), "
            "so ETL x ESP / T2_eff = 2 theta; the centre line has weight 1 (contrast at TE_eff). "
            "Kwargs etl (default 16), esp_ms (default 10), order in {linear, centric} (default "
            "linear in 2-D; a volume takes the geometry's line order): linear is asymmetric "
            "(edge enhancement on one side), centric decays towards the edges (blur). A TSE "
            "profile supplies etl and esp_ms. 3-D weights the (ky, kz) plane by line order.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.ACQUISITION,
        modes={
            MS: exact(degrade_tse_t2_blur),
            VOL: (tse_t2_blur_volume, Realisation.NATIVE),
        },
        condition=tse_condition,
    ),
    native_2d(
        code="D58",
        name="t1_shot_blur",
        family="sequence",
        fn=degrade_t1_shot_blur,
        description="T1 recovery across an inversion-recovery readout train",
        severity=SeveritySpec(
            (_P("readout_train_over_t1", "N_train x TR_ro / T1", 0.0, 1.0),),
            notes="after an inversion (M_0 = 1 - 2 exp(-TI / T1), TI to the first readout pulse) a train of "
            "N_train = theta T1 / TR_ro pulses drives Mz towards M_ss = (1 - E1) / (1 - E1 cos alpha), "
            "line j of L inner-loop lines seeing M_p = M_ss + (M_0 - M_ss)(E1 cos alpha)^p, "
            "p = j N_train / L, E1 = exp(-TR_ro / T1); weights 1 + (M_p - M_centre) / M_ss, the "
            "deviation from constant weighting scaled by the positive steady state (never singular, "
            "any T1). Kwargs t1_ms (default 1000), ti_ms (900), fa_deg (5), tr_ro_ms (7), order as "
            "for tse_t2_blur; the inner loop is ky (kz for kz_inner). A profile of sequence mprage, "
            "gre or spoiled_gre supplies ti_ms, fa_deg and the readout repetition (GRE tr_ms, MPRAGE "
            "esp_ms); any other sequence has no such readout, uses the defaults and stamps "
            "profile_ti_fa_ignored in the realised kwargs.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.ACQUISITION,
        modes={
            MS: exact(degrade_t1_shot_blur),
            VOL: (t1_shot_blur_volume, Realisation.NATIVE),
        },
        condition=t1_shot_condition,
    ),
    native_2d(
        code="D59",
        name="offres_blur_noncartesian",
        family="sequence",
        fn=degrade_offres_blur_noncartesian,
        description="Off-resonance blur of a radial readout",
        severity=SeveritySpec(
            (_P("peak_offresonance_x_readout", "cycles", 0.0, 2.0),),
            notes="peak |df| x T_ro = 2 theta phase cycles across the readout (T_ro = kwarg t_ro_ms, "
            "default 5; only the product matters). A seeded smooth df field (or kwarg b0_map_hz, "
            "whose pattern is scaled to that peak) modulates a time-segmented NUFFT of a golden-angle "
            "radial trajectory; the change it makes to the gridding reconstruction is added to the "
            "object. requires is empty: kwarg trajectory (with readout_times_s) is optional and "
            "absent means a deterministic golden-angle trajectory of ceil(pi/2 N) spokes. A volume is "
            "a stack of stars with fully sampled kz, exact per slice of the 3-D df field.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.SCANNER,
        modes={
            MS: exact(degrade_offres_blur_noncartesian, volume_kwargs=("b0_map_hz",)),
            VOL: (offres_blur_volume, Realisation.NATIVE),
        },
    ),
)
