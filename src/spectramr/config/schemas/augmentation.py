"""Data augmentation configuration schema."""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from .renames import reject_renamed_keys


class AugmentationConfigSchema(BaseModel):
    """Data augmentation configuration.

    Defines augmentation strategies and probabilities.

    Example:
        >>> config = AugmentationConfigSchema(
        ...     enabled=True,
        ...     probability=0.8,
        ...     intensity_range=(0.0, 1.0),
        ... )
    """

    # An unknown key raises: a misspelled transform flag would otherwise train
    # without the augmentation the YAML advertises.
    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }

    _reject_renamed = model_validator(mode="before")(
        classmethod(reject_renamed_keys("data.augmentation"))  # type: ignore[arg-type]
    )

    enabled: bool = Field(
        default=False,
        description="Enable data augmentation",
    )
    probability: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Probability of applying augmentation",
    )

    # Spatial augmentations
    enable_rotation: bool = Field(
        default=False,
        description="Enable rotation augmentation",
    )
    rotation_range: tuple[float, float] = Field(
        default=(-15, 15),
        description="Rotation angle range in degrees",
    )
    prob_rotate: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability of rotation",
    )

    enable_flip: bool = Field(
        default=False,
        description="Enable flip augmentation",
    )
    prob_flip: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability of flip",
    )
    flip_axes: list[int] = Field(
        default_factory=list,
        description="Axes to flip (e.g., [0, 1] for horizontal/vertical)",
    )

    enable_elastic_deformation: bool = Field(
        default=False,
        description="Enable elastic deformation",
    )
    elastic_alpha: float = Field(
        default=34.0,
        ge=0,
        description="Alpha parameter for elastic deformation",
    )
    elastic_sigma: float = Field(
        default=4.0,
        ge=0,
        description="Sigma parameter for elastic deformation",
    )

    # Intensity augmentations
    enable_noise: bool = Field(
        default=False,
        description="Enable noise augmentation",
    )
    noise_std: float = Field(
        default=0.01,
        ge=0,
        description="Standard deviation of noise",
    )

    enable_brightness: bool = Field(
        default=False,
        description="Enable brightness adjustment",
    )
    brightness_range: tuple[float, float] = Field(
        default=(0.8, 1.2),
        description="Brightness scaling range",
    )

    enable_contrast: bool = Field(
        default=False,
        description="Enable contrast adjustment",
    )
    contrast_range: tuple[float, float] = Field(
        default=(0.8, 1.2),
        description="Contrast scaling range",
    )

    enable_gamma: bool = Field(
        default=False,
        description="Enable gamma correction",
    )
    gamma_range: tuple[float, float] = Field(
        default=(0.8, 1.2),
        description="Gamma correction range",
    )

    # Realistic Degradations
    enable_rician_noise: bool = Field(
        default=False,
        description="Enable Rician noise (MRI-specific)",
    )
    rician_noise_level: float = Field(
        default=0.05,
        ge=0,
        description="Noise level for Rician noise",
    )

    enable_motion_blur: bool = Field(
        default=False,
        description="Enable 3D motion blur",
    )
    motion_blur_intensity: float = Field(
        default=0.3,
        ge=0,
        description="Intensity of motion blur",
    )

    enable_bias_field: bool = Field(
        default=False,
        description="Enable bias field (B1) inhomogeneity simulation",
    )

    enable_forward_operator: bool = Field(
        default=False,
        description=(
            "Replace the input, with probability prob_forward_operator, by a ULF "
            "image simulated from the HF target through simulate_ulf_from_hf "
            "(B0 shift, Maxwell, T2* blur, Rician noise). Train chain only: "
            "validation keeps the measured input."
        ),
    )
    prob_forward_operator: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Per-subject probability of the forward-operator replacement",
    )
    forward_operator_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "SimulateULFFromHF constructor kwargs (field_strength_t, target_snr_db, "
            "b0_distortion_strength, t2_star_blur_sigma, ...). An unknown key raises "
            "when the pipeline is built; source_image/output_image are fixed."
        ),
    )

    @model_validator(mode="after")
    def _forward_operator_keys_are_owned(self) -> "AugmentationConfigSchema":
        owned = {"source_image", "output_image", "keep_source", "p"} & set(
            self.forward_operator_kwargs
        )
        if owned:
            raise ValueError(
                f"data.augmentation.forward_operator_kwargs may not set {sorted(owned)}: "
                "the augmentation always reads target and writes input, and its "
                "probability is prob_forward_operator."
            )
        if self.forward_operator_kwargs and not self.enable_forward_operator:
            raise ValueError(
                "data.augmentation.forward_operator_kwargs is set but "
                "enable_forward_operator is false, so nothing reads it."
            )
        return self

    enable_b0_distortion: bool = Field(
        default=False,
        description="Enable B0 geometric distortion (susceptibility artifacts)",
    )
    b0_max_displacement: float = Field(
        default=2.0,
        ge=0,
        description="Maximum pixel displacement for B0 distortion",
    )
    b0_prob: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Probability of B0 distortion",
    )
    bias_field_coefficients: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Coefficient range for bias field polynomial",
    )

    enable_blur: bool = Field(
        default=False,
        description="Enable random blur (point spread function simulation)",
    )
    blur_std: tuple[float, float] = Field(
        default=(0.5, 2.0),
        description="Standard deviation range for Gaussian blur",
    )

    enable_ghosting: bool = Field(
        default=False,
        description="Enable random ghosting artifacts",
    )
    num_ghosts: tuple[int, int] = Field(
        default=(4, 10),
        description="Range for number of ghosts",
    )
    ghosting_intensity: tuple[float, float] = Field(
        default=(0.5, 1.0),
        description="Intensity of ghosting artifacts",
    )
    ghosting_axes: tuple[int, ...] = Field(
        default=(0, 1),
        description="Axes along which ghosting can occur",
    )

    enable_spike: bool = Field(
        default=False,
        description="Enable random spike artifacts (k-space)",
    )
    num_spikes: int = Field(
        default=1,
        ge=0,
        description="Number of spikes",
    )
    spike_intensity: tuple[float, float] = Field(
        default=(0.1, 1.0),
        description="Intensity of spikes",
    )

    enable_anisotropy: bool = Field(
        default=False,
        description="Enable random anisotropy (simulation of low-res axes)",
    )
    anisotropy_downsampling: tuple[float, float] = Field(
        default=(1.5, 5.0),
        description="Downsampling factor range for anisotropy",
    )

    enable_kspace_undersampling_augmentation: bool = Field(
        default=False,
        description="Enable k-space undersampling as augmentation",
    )
    undersampling_factor_augmentation: int = Field(
        default=2,
        ge=1,
        description="Acceleration factor for k-space undersampling",
    )

    # Advanced augmentations
    enable_mixup: bool = Field(
        default=False,
        description="Enable mixup augmentation",
    )
    mixup_alpha: float = Field(
        default=0.2,
        ge=0,
        description="Alpha parameter for mixup",
    )

    enable_cutmix: bool = Field(
        default=False,
        description="Enable cutmix augmentation",
    )
    cutmix_alpha: float = Field(
        default=0.2,
        ge=0,
        description="Alpha parameter for cutmix",
    )

    # Custom transform config
    transforms: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom transforms configuration",
    )
    intensity_range: tuple[float, float] = Field(
        default=(0.0, 1.0),
        description="Input intensity range (for normalization)",
    )


__all__ = ["AugmentationConfigSchema"]
