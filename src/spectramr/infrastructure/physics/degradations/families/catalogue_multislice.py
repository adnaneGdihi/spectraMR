"""The catalogue's multislice and through-plane axes (no SLICE_2D: they need the slice axis)."""

from __future__ import annotations

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations.catalogue.multislice_conditions import (
    off_resonance_condition,
    spin_history_condition,
    tr_condition,
)
from spectramr.infrastructure.physics.degradations.catalogue.multislice_crosstalk import (
    slice_crosstalk,
)
from spectramr.infrastructure.physics.degradations.catalogue.multislice_history import (
    spin_history,
)
from spectramr.infrastructure.physics.degradations.catalogue.multislice_offres import (
    off_resonance_slice_shift,
)
from spectramr.infrastructure.physics.degradations.catalogue.multislice_pve import (
    through_plane_pve,
)
from spectramr.infrastructure.physics.degradations.declarations import BOTH, MAG
from spectramr.infrastructure.physics.degradations.families import MS, VOL
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionMode
from spectramr.infrastructure.physics.degradations.spec import DegradationSpec, Realisation, entry
from spectramr.infrastructure.physics.severity import PhysicalParam, SeveritySpec

__all__ = ["ENTRIES"]

_P = PhysicalParam
_NATIVE = Realisation.NATIVE
_S2D = AcquisitionMode.SLICE_2D
_NEEDS_D = (
    "{} moves or mixes content between slices, so it needs the slice axis; a lone 2-D slice "
    "has no neighbours"
)


def _spec(code: str, name: str, family: str, modes, origin, affects, **kw) -> DegradationSpec:
    return entry(
        code=code,
        name=name,
        family=family,
        origin=origin,
        affects=affects,
        implementations=modes,
        realisation=dict.fromkeys(modes, _NATIVE),
        requires=frozenset(),
        active_when=frozenset(),
        **kw,
    )


ENTRIES: tuple[DegradationSpec, ...] = (
    _spec(
        "D60",
        "slice_crosstalk",
        "field",
        {MS: slice_crosstalk},
        ArtifactOrigin.ACQUISITION,
        MAG,
        description="Inter-slice RF crosstalk: neighbouring slice profiles saturate each slice",
        severity=SeveritySpec(
            (_P("profile_transition_sigma", "x slice thickness", 0.0, 0.5),),
            notes="Slice profile = rect(thk) convolved with a Gaussian of std sigma = theta*thk/2, "
            "overlapping the adjacent slabs; the gap is voxel_mm[0] - slice_thickness_mm "
            "(thickness = pitch when unset). Slice k is scaled by prod over neighbours "
            "(1 - s exp(-dt/T1)), s the slab-mean of the neighbour's profile, dt from the slice "
            "order and TR (default 500 ms, profile tr_ms overrides), T1 1000 ms. SEQUENTIAL is "
            "block-by-block (line_times): neighbours are a whole slice acquisition apart, "
            "recover almost fully, so it saturates far less than INTERLEAVED. The effect is "
            "continuous from theta=0.",
        ),
        condition=tr_condition,
        unsupported_detail={
            _S2D: "a single 2-D slice has no neighbouring slice to saturate it (the digital "
            "twin's rf_crosstalk stage is the in-plane proxy)",
            AcquisitionMode.VOLUME_3D: "a 3-D slab is excited once as a whole, so there are no "
            "separate slice excitations to saturate each other",
        },
    ),
    _spec(
        "D61",
        "through_plane_pve",
        "field",
        {MS: through_plane_pve, VOL: through_plane_pve},
        ArtifactOrigin.ACQUISITION,
        BOTH,
        description="Through-plane partial volume: slice-profile averaging (kz window in 3-D)",
        severity=SeveritySpec(
            (
                _P("profile_transition_sigma", "x slice thickness (multislice)", 0.0, 0.5),
                _P("kz_window_weight", "fraction of a Hamming kz window (3-D)", 0.0, 1.0),
            ),
            notes="Multislice: the stack convolved along D with rect(thk) convolved with a "
            "Gaussian of std theta*thk/2, each tap the profile averaged over a slice extent. "
            "3-D: kz weights 1 - theta (1 - hamming(f)). theta=0 is the identity, continuous.",
        ),
        condition=None,
        unsupported_detail={
            _S2D: "it needs the slice axis D to average across; slice_pve is the 2-D "
            "readout-blur proxy"
        },
    ),
    _spec(
        "D62",
        "spin_history",
        "motion",
        {MS: spin_history, VOL: spin_history},
        ArtifactOrigin.PATIENT,
        BOTH,
        description="Spin history: magnetisation carried between slices by through-plane motion",
        severity=SeveritySpec(
            (_P("peak_through_plane_displacement", "mm", 0.0, 5.0),),
            notes="One rigid dz per shot (8 shots), |dz| <= 5 mm*theta. A spin from slice j gets signal "
            "gain 1 + a q^n (q = E1 cos a): a = M_0/M_ss - 1, M_0 = 1 - (1 - M_ss cos a) exp(-dt'/T1) at its first "
            "excitation, M_n = M_ss + (M_0 - M_ss)(E1 cos a)^n after n more; defaults TR 50 ms, "
            "flip 30 deg, T1 1000 ms (profile tr_ms and fa_deg override). 3-D: only voxels "
            "sourced outside the slab change (inflow, M_0 = 1).",
        ),
        condition=spin_history_condition,
        unsupported_detail={_S2D: _NEEDS_D.format("spin history")},
    ),
    _spec(
        "D63",
        "off_resonance_slice_shift",
        "field",
        {MS: off_resonance_slice_shift, VOL: off_resonance_slice_shift},
        ArtifactOrigin.SCANNER,
        BOTH,
        description="Off-resonance shift of the excited slice along the slice-select axis",
        severity=SeveritySpec(
            (_P("peak_off_resonance", "Hz", 0.0, 200.0),),
            notes="dz = thk*df/BW_rf with BW_rf 1000 Hz (G_ss = BW_rf/(gamma_bar thk)); a seeded "
            "smooth field of peak 200 Hz*B0/3 T, or b0_map_hz (the measured field at theta=1). "
            "A profile's B0/3 scaling (realised as profile_field_scale) applies to the seeded "
            "field only and is not applied to a measured map, which is already in Hz at its own "
            "field; an explicit peak_hz or field_scale with a map raises. Multislice resamples "
            "along D (linear); 3-D moves only the slab boundary (thk = the slab). A profile "
            "gradient_max_mT_per_m below G_ss raises.",
        ),
        condition=off_resonance_condition,
        unsupported_detail={_S2D: _NEEDS_D.format("the slice-select shift")},
    ),
)
