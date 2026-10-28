"""Centralized collation strategy selection logic.

This module handles strategy selection for data batch collation.
Replaces hardcoded selection logic in builders with a single source of truth.

Author: spectraMR Team
Date: 2026-02-15
"""

import logging

from spectramr.config.schemas.collation import CollationConfigSchema

logger = logging.getLogger(__name__)

__all__ = ["CollationStrategySelector"]


def _image_kwargs(config, enable_slab_mode, patch_size) -> dict:
    kwargs = {
        "padding_mode": config.padding_mode,
        "padding_value": config.padding_value,
        "allow_variable_shapes": config.allow_variable_shapes,
    }
    # Auto-derive squeeze_depth if not specified by user
    if config.squeeze_depth_dim is None:
        # Squeeze depth dimension if patch_size[-1] == 1 (2D data)
        squeeze_depth = patch_size[-1] == 1 if patch_size and len(patch_size) == 3 else False
    else:
        # User explicitly specified squeeze behavior
        squeeze_depth = config.squeeze_depth_dim
    kwargs["squeeze_depth_dim"] = squeeze_depth
    # ImageCollateStrategy uses this in the final NaN check; forward the wired YAML knob
    # (collation.validate_nans) so validate_nans=False passes NaNs through unchanged
    # instead of replacing them with zeros.
    kwargs["validate_nans"] = config.validate_nans
    return kwargs


def _slab_kwargs(config, enable_slab_mode, patch_size) -> dict:
    # SlabCollateStrategy fixes both itself (pads to the batch's shape and keeps depth for
    # its own flatten), so passing them raised "multiple values" at loader build. A config
    # that asks for the opposite is refused rather than ignored.
    if config.allow_variable_shapes:
        raise ValueError(
            "collation.allow_variable_shapes=true is not supported by the 'slab' "
            "strategy, which always pads to the batch's shape."
        )
    if config.squeeze_depth_dim:
        raise ValueError(
            "collation.squeeze_depth_dim=true is not supported by the 'slab' "
            "strategy, which keeps depth for its own channel flatten."
        )
    kwargs = {"padding_mode": config.padding_mode, "padding_value": config.padding_value}
    if enable_slab_mode:
        # Was the literal "middle" -- an advertised three-value knob (middle/flatten/keep)
        # with no config route, so no arm could pick (B6). Default unchanged.
        kwargs["target_mode"] = config.slab_target_mode
        kwargs["flat_input"] = True  # Flatten depth to channels
    kwargs["validate_nans"] = config.validate_nans
    return kwargs


#: Strategy name -> the builder of its ``__init__`` kwargs. A strategy not listed takes none.
_STRATEGY_KWARGS = {"image": _image_kwargs, "slab": _slab_kwargs}


class CollationStrategySelector:
    """Centralized selector for collation strategy based on user config and data type.

    This class encapsulates all logic for choosing which collation strategy to use
    when creating data loaders. It provides:

    1. User choice override (explicit config.collation.strategy)
    2. Auto-selection based on dataset_type and enable_slab_mode
    3. Compatibility validation with warnings
    4. Parameter derivation (squeeze_depth from patch_size)
    5. Transparent logging of all decisions

    Benefits:
        - Single source of truth for collation selection
        - Testable in isolation (no builder dependencies)
        - Consistent selection across train/val/test
        - User-configurable with sensible auto-selection fallback

    Example Usage:
        ```python
        from spectramr.config.schemas.collation import CollationConfigSchema
        from spectramr.data.collation.strategy_selector import CollationStrategySelector
        from spectramr.data.collation import CollateStrategyFactory

        config = CollationConfigSchema(strategy="image", padding_mode="zeros")

        strategy_name, kwargs = CollationStrategySelector.select_strategy(
            config=config,
            dataset_type="kspace",
            enable_slab_mode=False,
            patch_size=(256, 256, 1),
        )

        # strategy_name = "image"
        # kwargs = {"padding_mode": "zeros", "squeeze_depth_dim": True, ...}

        strategy = CollateStrategyFactory.create(strategy_name, **kwargs)
        ```
    """

    # Auto-selection mapping: dataset_type → strategy_name
    # Used ONLY when user doesn't explicitly specify strategy
    AUTO_SELECT_MAP = {
        "kspace": "image",  # K-space data uses image collation
        "kspace_cold_diffusion": "image",  # Cold diffusion k-space
        "image": "image",  # 2D images
        "nifti": "image",  # 3D NIfTI volumes
        "nifti_paired": "image",  # Paired NIfTI for translation
        "contrast_aware_paired": "image",  # Paired with per-contrast normalization
        "npy_slice": "image",  # Pre-paired numpy slices
        "fastmri": "image",  # FastMRI HDF5
        "m4raw": "image",  # M4Raw k-space
        "dicom": "image",  # DICOM series
        "synthetic": "image",  # Synthetic data
        # E1/E2/E5 + Phase-4 families — all return input/target tio.Subjects
        # (same contract as nifti_paired), so they collate with "image" rather
        # than silently falling through to "robust" (pitfall #9).
        "bart_kspace": "image",  # BART .cfl/.hdr non-Cartesian k-space (E2)
        "ismrmrd_kspace": "image",  # ISMRMRD measured-trajectory k-space
        "bids_paired": "image",  # BIDS low/high-field paired NIfTI (E5)
        "png_paired": "image",  # Paired-PNG super-resolution (brats_sr)
        "field_ref": "image",  # NIfTI B0/B1 field-reference
        "oracle_bssfp": "image",  # Phase-cycled bSSFP + real Hz B0
        "mrixfields": "image",  # MRIxFields multi-field paired translation
        "quantitative": "image",  # qMRI parameter-map regression (Phase 4b)
        "cine": "image",  # Cine 4D (Phase 4c)
        "pde_synthetic": "image",  # PDE operator benchmarks (Burgers/Darcy)
        "graph": "graph",  # Graph data
        "graph_mri": "graph",  # Graph-based MRI
        "sequence": "sequence",  # Sequential data
    }

    # Compatibility mapping: strategy → compatible dataset_types
    # Used for validation warnings (not blocking)
    COMPATIBILITY_MAP = {
        "image": {
            "kspace",
            "kspace_cold_diffusion",
            "image",
            "nifti",
            "nifti_paired",
            "contrast_aware_paired",
            "npy_slice",
            "fastmri",
            "m4raw",
            "dicom",
            "synthetic",
            "bart_kspace",
            "ismrmrd_kspace",
            "bids_paired",
            "png_paired",
            "field_ref",
            "oracle_bssfp",
            "mrixfields",
            "quantitative",
            "cine",
            "pde_synthetic",
        },
        "slab": {"nifti", "kspace", "image", "nifti_paired"},  # 3D data only
        "graph": {"graph", "graph_mri"},
        "sequence": {"sequence"},
        "physics": {"kspace", "fastmri", "m4raw"},  # Non-Cartesian physics
        "robust": {"*"},  # Works with anything (fallback)
        # "universal" lived here and CollateStrategyFactory has no such class
        # (audit A8). It passed the Literal, passed this map, then died in the
        # factory with "Unknown data type: universal" — advertised, validated,
        # unbuildable. Zero arms declared it, so it was latent rather than
        # broken. `_assert_vocabularies_agree` below stops the next one.
        "mixed": {"*"},  # Multi-modal (handles any combination)
    }

    @staticmethod
    def _assert_vocabularies_agree() -> None:
        """COMPATIBILITY_MAP and the factory must name the same strategies.

        They are two tables for one concept and the permissive one is consulted
        first, so a name present here and absent there passes validation and
        then dies in construction — which is exactly how "universal" survived
        (A8). Called at selection time, so the check runs on the live path
        rather than only under test.
        """
        from spectramr.data.collation.strategies import CollateStrategyFactory

        buildable = set(CollateStrategyFactory.STRATEGIES)
        advertised = set(CollationStrategySelector.COMPATIBILITY_MAP)

        # THREE vocabularies, not two. The schema `Literal` is what a YAML is
        # validated against, and it kept advertising "universal" after this map
        # and the factory had both dropped it -- so the check below passed while
        # a schema-legal arm would still have died in construction. That is the
        # exact hole this guard exists to close, one layer up.
        import typing

        from spectramr.config.schemas.collation import CollationConfigSchema

        declared = {
            a
            for a in typing.get_args(
                typing.get_args(CollationConfigSchema.model_fields["strategy"].annotation)[0]
            )
            if isinstance(a, str)
        }
        if declared - buildable:
            raise ValueError(
                f"Collation strategies accepted by CollationConfigSchema but "
                f"not constructible by CollateStrategyFactory: "
                f"{sorted(declared - buildable)}. A schema-legal value that "
                "cannot be built passes config validation and then fails at "
                "construction (non-negotiable #8)."
            )
        if advertised - buildable:
            raise ValueError(
                f"Collation strategies advertised in COMPATIBILITY_MAP but not "
                f"constructible by CollateStrategyFactory: "
                f"{sorted(advertised - buildable)}. Register the class, or drop "
                "the entry — an advertised name that cannot be built passes "
                "validation and fails at construction."
            )

    @staticmethod
    def select_strategy(
        config: CollationConfigSchema,
        dataset_type: str,
        enable_slab_mode: bool = False,
        patch_size: tuple[int, int, int] | None = None,
    ) -> tuple[str, dict]:
        """Select collation strategy based on user config and auto-selection rules.

        Selection priority:
            1. User-specified config.strategy (highest priority)
            2. Auto-select "slab" if enable_slab_mode=True
            3. Auto-select from dataset_type using AUTO_SELECT_MAP
            4. Fallback to "robust" if dataset_type not recognized

        Args:
            config: CollationConfigSchema with user choices
            dataset_type: Type of dataset (e.g., "kspace", "image", "nifti")
            enable_slab_mode: Whether 2.5D slab mode is enabled (overrides auto-select)
            patch_size: 3D patch size tuple (W, H, D) for auto-squeeze derivation

        Returns:
            Tuple of (strategy_name, strategy_kwargs):
                - strategy_name: Name of collation strategy to use
                - strategy_kwargs: Dict of __init__ parameters for strategy instantiation

        Raises:
            ValueError: If user-specified strategy is invalid

        Example:
            >>> config = CollationConfigSchema(strategy=None)  # Auto-select
            >>> strategy, kwargs = CollationStrategySelector.select_strategy(
            ...     config=config,
            ...     dataset_type="kspace",
            ...     enable_slab_mode=False,
            ...     patch_size=(256, 256, 1),
            ... )
            >>> print(strategy)
            'image'
            >>> print(kwargs)
            {'padding_mode': 'zeros', 'squeeze_depth_dim': True, ...}
        """
        # The two strategy vocabularies must agree before either is trusted.
        CollationStrategySelector._assert_vocabularies_agree()

        # STEP 1: User explicitly specified strategy? (highest priority)
        if config.strategy is not None:
            strategy = config.strategy
            if config.log_strategy_selection:
                logger.info(
                    f"[CollationStrategySelector] User-specified strategy: '{strategy}' "
                    f"(dataset_type='{dataset_type}')"
                )

            # Validate compatibility — raises on an incompatible pair (A13).
            CollationStrategySelector._validate_compatibility(strategy, dataset_type)
        else:
            # STEP 2: Auto-select based on slab_mode first (overrides dataset_type)
            if enable_slab_mode:
                strategy = "slab"
                if config.log_strategy_selection:
                    logger.info(
                        f"[CollationStrategySelector] Auto-selected 'slab' "
                        f"(enable_slab_mode=True, dataset_type='{dataset_type}')"
                    )
            else:
                # STEP 3: Auto-select from dataset_type
                strategy = CollationStrategySelector.AUTO_SELECT_MAP.get(
                    dataset_type,
                    "robust",  # Fallback to robust if unknown
                )
                if config.log_strategy_selection:
                    logger.info(
                        f"[CollationStrategySelector] Auto-selected '{strategy}' "
                        f"for dataset_type='{dataset_type}'"
                    )

        # STEP 4: Build strategy __init__ kwargs
        kwargs = CollationStrategySelector._build_strategy_kwargs(
            strategy=strategy,
            config=config,
            enable_slab_mode=enable_slab_mode,
            patch_size=patch_size,
        )

        if config.log_strategy_selection:
            logger.info(
                f"[CollationStrategySelector] Final selection: "
                f"strategy='{strategy}', kwargs={kwargs}"
            )

        return strategy, kwargs

    @staticmethod
    def _validate_compatibility(strategy: str, dataset_type: str) -> None:
        """Validate that strategy is compatible with dataset_type.

        This is the one place the subsystem knows the pair is wrong, so it
        raises rather than logging (audit A13, pitfall #10). It previously
        warned "Proceeding anyway (user-specified choice)" — but the collation
        strategy decides how samples are stacked, and stacking them the wrong
        way does not fail loudly downstream; it produces a batch of the wrong
        shape or the wrong contents, which surfaces as a model error far from
        the cause, if at all.

        Measured before flipping: **105 arms declare data.collation.strategy,
        every one of them 'image', and ZERO form an incompatible pair.** So
        this raises for nobody today; it is a guard against the next arm.

        A pairing that is deliberate rather than mistaken has an escape that
        does not require lying about the dataset: 'robust' accepts every type
        by construction.

        Args:
            strategy: Collation strategy name
            dataset_type: Dataset type identifier

        Raises:
            ValueError: strategy is unknown, or is incompatible with
                dataset_type.
        """
        if strategy not in CollationStrategySelector.COMPATIBILITY_MAP:
            raise ValueError(
                f"Unknown collation strategy: '{strategy}'. "
                f"Valid strategies: {sorted(CollationStrategySelector.COMPATIBILITY_MAP.keys())}"
            )

        valid_types = CollationStrategySelector.COMPATIBILITY_MAP[strategy]

        if dataset_type not in valid_types and "*" not in valid_types:
            raise ValueError(
                f"Collation strategy '{strategy}' is not compatible with "
                f"dataset_type '{dataset_type}'. Compatible types for "
                f"'{strategy}': {sorted(valid_types)}. A wrong pairing does not "
                "fail loudly downstream — it stacks samples the wrong way and "
                "surfaces as a model error far from the cause. Pick a "
                "compatible strategy, or use 'robust', which accepts every "
                "dataset_type by construction."
            )

    @staticmethod
    def _build_strategy_kwargs(
        strategy: str,
        config: CollationConfigSchema,
        enable_slab_mode: bool,
        patch_size: tuple[int, int, int] | None,
    ) -> dict:
        """Build __init__ kwargs for strategy instantiation.

        Builds parameters based on:
            1. User config (padding_mode, padding_value, etc.)
            2. Auto-derivation (squeeze_depth from patch_size)
            3. Strategy-specific requirements

        Args:
            strategy: Strategy name
            config: CollationConfigSchema with user parameters
            enable_slab_mode: Whether slab mode is enabled
            patch_size: 3D patch size tuple for squeeze derivation

        Returns:
            Dict of kwargs to pass to CollateStrategyFactory.create()

        Example:
            >>> kwargs = CollationStrategySelector._build_strategy_kwargs(
            ...     strategy="image",
            ...     config=CollationConfigSchema(padding_mode="zeros"),
            ...     enable_slab_mode=False,
            ...     patch_size=(256, 256, 1),
            ... )
            >>> print(kwargs)
            {
                'padding_mode': 'zeros',
                'padding_value': 0.0,
                'allow_variable_shapes': False,
                'squeeze_depth_dim': True,  # Auto-derived from patch_size
                'validate_nans': True,
            }
        """
        build = _STRATEGY_KWARGS.get(strategy)
        if build is None:
            return {}
        return build(config, enable_slab_mode, patch_size)
