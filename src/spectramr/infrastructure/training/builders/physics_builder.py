"""Physics Builder

Creates MRI physics operators (FFT, k-space masks, data consistency, coil sensitivity).
"""

import logging
from typing import Any

from spectramr.config.settings import TrainingSettings

from .base import Builder

logger = logging.getLogger(__name__)


class PhysicsBuilder(Builder):
    """Builds MRI physics components.

    Creates physics operators needed for MRI reconstruction: FFT transforms,
    k-space undersampling masks, data consistency operators, and coil sensitivity
    estimators.

    Attributes:
        _config: Training configuration
        _device: Device where physics components will be used
        _components: Dictionary of created physics components

    Example:
        >>> builder = PhysicsBuilder(config, torch.device("cuda"))
        >>> physics = (builder
        ...     .build_fft_transformer()
        ...     .build_data_consistency()
        ...     .build())
        >>> fft = physics["fft"]
    """

    def __init__(self, config: TrainingSettings, device: Any):
        """Initialize PhysicsBuilder.

        Args:
            config: Immutable training configuration
            device: Device for physics components
        """
        self._config = config
        self._device = device
        self._components: dict[str, Any] = {}

    def build_fft_transformer(self) -> "PhysicsBuilder":
        """Create FFT/IFFT transformer.

        Always creates FFT transformer as it's fundamental for MRI reconstruction.

        Returns:
            self: For method chaining
        """
        try:
            from spectramr.infrastructure.physics.fft_ops import FFTTransformer

            self._components["fft"] = FFTTransformer()  # norm="ortho" by default
            logger.info("Created FFT transformer")
        except Exception as e:
            logger.error(f"Failed to create FFT transformer: {e}")
            raise

        return self

    def build_data_consistency(self) -> "PhysicsBuilder":
        """[DEPRECATED] Create data consistency operator.

        DC is now integrated directly into the models (e.g. KSpaceColdDiffusionGenerator)
        to support learnable parameters and maintain architectural SSOT.

        Returns:
            self: For method chaining
        """
        logger.debug("PhysicsBuilder.build_data_consistency skipped: DC is model-integrated")
        return self

    def build_coil_sensitivity(self) -> "PhysicsBuilder":
        """No-op. Coil sensitivity is owned by the DATA path, not by this builder.

        This method was unreachable dead code, not a live failure. It read
        ``self._config.physics.parallel_imaging.enabled``, but ``parallel_imaging``
        is not a field on ANY config schema (verified across every class in
        ``config/schemas/``) and ``settings.physics`` is ``None`` unless a config
        supplies a ``physics:`` block. Both guards therefore returned early on
        every call, and the body below them never ran::

            from ...coil_sensitivity import ESPIRiTSensitivity   # never existed
            self._components["coil_sens"] = ESPIRiTSensitivity()

        So the broken import never raised and its ``except Exception`` never
        warned -- the defect was invisible from the outside precisely because the
        dead knob above it kept the dead import unreachable.

        Restoring the import would fix nothing: ``_components["coil_sens"]`` was
        the ONLY reference to that key tree-wide, so a working estimator would be
        constructed and discarded.

        The owner is already elected. ``estimate_smaps`` is called live from
        ``data_pipeline_director.py:290``, smaps reach the strategies on the batch
        as ``_current_smaps`` / ``gen_kwargs["smaps"]``, and
        ``tests/unit/test_coil_sensitivity.py`` records the older
        ``CoilSensitivityEstimationService`` as "deprecated (superseded by
        estimate_smaps)". That is a duplicate owner, so this route goes rather
        than being wired (non-negotiable 17).

        Kept as an explicit no-op rather than deleted because ``director.py:171``
        chains it, and a builder step that vanishes from a fluent chain is harder
        to notice than one that says why it does nothing.
        """
        return self

    def validate(self) -> "PhysicsBuilder":
        """Validate that required physics components are created.

        Returns:
            self: For method chaining

        Raises:
            ValueError: If FFT transformer is missing
        """
        if "fft" not in self._components:
            raise ValueError("FFT transformer is required but not created")

        logger.info(f"Physics validation passed ({len(self._components)} components created)")
        return self

    def build(self) -> dict[str, Any]:
        """Return all physics components.

        Returns:
            Dict[str, Any]: Copy of physics components dictionary

        Raises:
            ValueError: If validation fails
        """
        if "fft" not in self._components:
            raise ValueError("FFT transformer not created. Call build_fft_transformer() first.")

        return dict(self._components)  # Return copy for immutability
