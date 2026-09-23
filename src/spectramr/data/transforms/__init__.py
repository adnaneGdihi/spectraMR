# Registry population. The @register_transform decorator only runs when the
# defining module is imported, so a transform that nothing else imports would
# be registered nowhere and stay unreachable from YAML -- exactly the state
# this registry exists to end. Import them here, unconditionally, and keep this
# list in sync when adding a config-declarable transform.
# (Same maintenance contract as data/adapters/__init__.py.)
from . import foreground_mask_extractor as _foreground_mask_extractor  # noqa: F401
from . import graph_transform as _graph_transform  # noqa: F401
from . import homomorphic_bias_field as _homomorphic_bias_field  # noqa: F401
from . import joint_rotation as _joint_rotation  # noqa: F401
from . import espirit_sensitivity as _espirit_sensitivity  # noqa: F401
from . import phase_residual as _phase_residual  # noqa: F401
from . import qmap_ulf_operator_transform as _qmap_ulf_operator_transform  # noqa: F401
from . import scout_acquisition as _scout_acquisition  # noqa: F401
from . import slab_transforms as _slab_transforms  # noqa: F401
from . import slice_profile as _slice_profile  # noqa: F401
from . import synthetic_ulf_simulator as _synthetic_ulf_simulator  # noqa: F401
from .benchmarking_degradations import (
    B1TransmitInhomogeneity,
    KSpaceComplexGaussianNoise,
    KSpaceGhosting,
    KSpaceSpikes,
)
from .kspace_coil_transforms import CoilCombineTransform, ComplexToRealTransform
from .normalization import (
    KSpaceNormalizationTransform,
    NormalizationConfig,
    NormalizationStrategy,
    compute_magnitude,
    denormalize_percentile,
    normalize_minmax,
    normalize_percentile,
    normalize_tensor,
    normalize_zscore,
)
from .registry import (
    TRANSFORM_REGISTRY,
    RegisteredTransform,
    build_transform,
    get_transform,
    list_transforms,
    register_transform,
    transforms_producing,
)
from .sle_trajectory import (
    build_sle_kspace_mask,
    kappa_to_dimension,
    sample_sle_trace,
)
from .tio_physics import PhysicsInformedMasking
from .transforms import (
    AddGaussianNoise,
    ApplyMask,
    B0Inhomogeneity,
    LogMagnitudeScaling,
    LowFieldNoise,
    PhysicsTransform,
    SimulateKSpace,
)

__all__ = [
    "AddGaussianNoise",
    "ApplyMask",
    "B0Inhomogeneity",
    "B1TransmitInhomogeneity",
    "CoilCombineTransform",
    "ComplexToRealTransform",
    "KSpaceComplexGaussianNoise",
    "KSpaceGhosting",
    "KSpaceNormalizationTransform",
    "KSpaceSpikes",
    "LogMagnitudeScaling",
    "LowFieldNoise",
    "NormalizationConfig",
    "NormalizationStrategy",
    "PhysicsInformedMasking",
    "PhysicsTransform",
    "SimulateKSpace",
    "build_sle_kspace_mask",
    "compute_magnitude",
    "denormalize_percentile",
    "kappa_to_dimension",
    "normalize_minmax",
    "normalize_percentile",
    "normalize_tensor",
    "normalize_zscore",
    "sample_sle_trace",
]
