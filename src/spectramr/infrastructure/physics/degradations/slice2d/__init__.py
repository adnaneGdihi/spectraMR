"""The native 2-D degradation kernels, ``fn(x, theta, seed=0, **kwargs)`` on ``(B, C, H, W)``."""

from spectramr.infrastructure.physics.degradations.slice2d.blur import (
    degrade_resolution_snr,
    degrade_slice_thickness_pve,
    degrade_t2star_blur,
)
from spectramr.infrastructure.physics.degradations.slice2d.fields import (
    degrade_bias_field,
    degrade_coil_sensitivity_miscal,
    degrade_susceptibility,
)
from spectramr.infrastructure.physics.degradations.slice2d.geometric import degrade_geometric_warp
from spectramr.infrastructure.physics.degradations.slice2d.kspace_phase import (
    degrade_concomitant_phase,
    degrade_iq_ghost,
    degrade_nyquist_ghost,
    degrade_stimulated_echo,
)
from spectramr.infrastructure.physics.degradations.slice2d.motion import (
    degrade_pulsatile_motion,
    degrade_random_motion,
    degrade_sudden_motion,
    degrade_translation_drift,
)
from spectramr.infrastructure.physics.degradations.slice2d.noise import (
    degrade_complex_gaussian_noise,
    degrade_rician_noise,
)
from spectramr.infrastructure.physics.degradations.slice2d.sampling_cartesian import (
    degrade_cartesian_undersampling,
    degrade_partial_fourier,
    degrade_variable_density_cartesian,
)
from spectramr.infrastructure.physics.degradations.slice2d.sampling_vd_radial import (
    degrade_radial_undersampling,
    degrade_variable_density_2d,
    degrade_variable_density_undersampling,
)

__all__ = [
    "degrade_bias_field",
    "degrade_cartesian_undersampling",
    "degrade_coil_sensitivity_miscal",
    "degrade_complex_gaussian_noise",
    "degrade_concomitant_phase",
    "degrade_geometric_warp",
    "degrade_iq_ghost",
    "degrade_nyquist_ghost",
    "degrade_partial_fourier",
    "degrade_pulsatile_motion",
    "degrade_radial_undersampling",
    "degrade_random_motion",
    "degrade_resolution_snr",
    "degrade_rician_noise",
    "degrade_slice_thickness_pve",
    "degrade_stimulated_echo",
    "degrade_sudden_motion",
    "degrade_susceptibility",
    "degrade_t2star_blur",
    "degrade_translation_drift",
    "degrade_variable_density_2d",
    "degrade_variable_density_cartesian",
    "degrade_variable_density_undersampling",
]
