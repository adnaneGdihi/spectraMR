"""Base Inference Strategy

Abstract base class for paradigm-specific inference strategies.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from spectramr.config.schemas.enums import TrainingModeTypes
from spectramr.core.module_utils import unwrap_model
from spectramr.infrastructure.checkpointing.restore import (
    CheckpointPartMissingError,
    read_part,
    weights,
)
from spectramr.infrastructure.inference.predict_data_consistency import (
    PredictDataConsistency,
)


def _has_ema(checkpoint_path: str) -> bool:
    try:
        read_part(checkpoint_path, "ema")
    except CheckpointPartMissingError:
        return False
    return True


class BaseInferenceStrategy(ABC):
    """Abstract base class for inference strategies.

    This class defines the interface that all paradigm-specific inference
    strategies must implement. Each strategy handles the unique requirements
    of different training paradigms (GAN, diffusion, VAE, etc.).
    """

    # ── Phase 2 of TODO/audit/data_layer_unification_plan.md ───────────
    #
    # Per-paradigm opt-in for sliding-window inference tiling. When True,
    # ``DataPipelineDirector.build(mode="infer", ...)`` may wrap the
    # dataloader with ``tio.GridSampler`` + ``tio.GridAggregator`` (see
    # ``src/data/builders/inference_tiling.py``) so 3D volumes that
    # don't fit in memory can be processed tile-by-tile.
    #
    # Subclasses opt in by overriding this attribute to ``True``. The
    # **default is False** because the naive "tile → forward once →
    # aggregate" pattern is wrong for paradigms with iterative reverse
    # samplers (diffusion, cold-diffusion, latent-diffusion). Those
    # paradigms can opt in only after implementing per-tile reverse
    # loops + aggregation — see Phase 2 §11 risk #6 for the design notes.
    #
    # Compatibility matrix (proposed, refined as paradigms opt in):
    #   GAN / Reconstruction / VAE / Domain Adaptation / PnP-RED → True
    #   Latent Diffusion / Cold Diffusion / MAE                  → False
    #   GeoMamba ULF (position-aware)                            → False (verify)
    #   SSL pretrain                                             → False (no inference)
    supports_grid_tiling: bool = False

    #: Whether ``infer`` applies the arm's k-space normalisation itself. When
    #: False the predict pipeline normalises (#2581); a strategy that also did
    #: would divide twice, so exactly one of the two owns it per strategy.
    owns_kspace_normalization: bool = False

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ):
        """Initialize the inference strategy.

        Args:
            model: The trained model to use for inference
            device: Device to run inference on
            config: Configuration dictionary for the strategy
        """
        self.model = model
        self.device = device
        self.config = config or {}

        # Phase 6 of TODO/audit/data_layer_unification_plan.md —
        # build the adapter chain at inference-strategy init so the
        # YAML's ``adapters:`` block is honored at inference time, not
        # just during training. Without this, a YAML that declares
        # ``adapters: {pre_model: [rss_coils_to_magnitude]}`` would
        # silently skip the adapter at inference and produce
        # wrong-shape tensors (CLAUDE.md #9).
        from spectramr.infrastructure.builders.leaf.adapter_builders import (
            AdapterChainBuilder,
        )

        # The ``config`` dict here is the strategy-specific subdict; the
        # adapters block lives on the parent settings under ``adapters``.
        # We support both: a top-level ``adapters`` key in the strategy
        # config and a wrapped ``settings.adapters`` shape.
        adapters_cfg = None
        if isinstance(self.config, dict):
            adapters_cfg = self.config.get("adapters")
        elif hasattr(self.config, "adapters"):
            adapters_cfg = self.config.adapters
        self.adapter_chains: dict[str, list[Any]] = AdapterChainBuilder(adapters_cfg).build()

        # ``physics.data_consistency.apply_at_predict``: the test-time hard
        # projection onto the measurement. Resolved once, here, for every
        # strategy; ``None`` when the knob is off so the off state carries no
        # projection code path at all. The hook is in ``infer`` below.
        self.predict_dc: PredictDataConsistency | None = PredictDataConsistency.from_config(
            self.config
        )

    def apply_adapters(self, hook: str, x: Any) -> Any:
        """Apply the declared adapter chain at ``hook`` at inference time.

        Mirror of :py:meth:`BaseTrainingStrategy.apply_adapters` —
        when a YAML declares ``adapters: {pre_model: [...], post_model:
        [...], pre_metric: [...]}``, those chains MUST fire at
        inference, not only at training. Otherwise the model sees
        differently-shaped tensors between the two paths and silently
        produces wrong outputs (CLAUDE.md #9).

        No-op when no adapter chain is declared at ``hook``.

        Args:
            hook: insertion point name (``"pre_model"``, ``"post_model"``,
                ``"pre_metric"``, etc.).
            x: tensor (or dict of tensors) at this stage of the pipeline.

        Returns:
            The transformed tensor / dict.
        """
        from spectramr.infrastructure.builders.leaf.adapter_builders import (
            apply_chain,
        )

        chain = self.adapter_chains.get(hook, [])
        if not chain:
            return x
        return apply_chain(chain, x)

    @property
    @abstractmethod
    def training_mode(self) -> TrainingModeTypes:
        """Return the training mode this strategy handles."""
        pass

    @abstractmethod
    def preprocess_input(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Preprocess input tensor for inference.

        Args:
            input_tensor: Raw input tensor
            **kwargs: Additional preprocessing parameters

        Returns:
            Preprocessed tensor ready for model inference
        """
        pass

    @abstractmethod
    def run_inference(
        self, input_tensor: torch.Tensor, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Run inference using the trained model.

        Args:
            input_tensor: Preprocessed input tensor
            **kwargs: Additional inference parameters

        Returns:
            Model output tensor, optionally with metadata
        """
        pass

    @abstractmethod
    def postprocess_output(
        self, output_tensor: torch.Tensor, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Postprocess model output.

        Args:
            output_tensor: Raw model output
            **kwargs: Additional postprocessing parameters

        Returns:
            Postprocessed output tensor, optionally with metadata
        """
        pass

    @torch.no_grad()
    def infer(
        self, input_tensor: torch.Tensor, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Complete inference pipeline.

        Args:
            input_tensor: Raw input tensor
            **kwargs: Additional parameters for preprocessing/inference/postprocessing

        Returns:
            Final inference result
        """
        # ``measured_kspace`` is consumed HERE and never forwarded. Cold
        # diffusion hands its remaining kwargs to the model's forward, so a
        # kwarg only the projection reads must not reach ``run_inference``.
        # ``mask`` stays: it is already in the strategies' kwarg vocabulary.
        measured_kspace = kwargs.pop("measured_kspace", None)
        # A scale measured over the whole volume by a strategy that owns the
        # normalisation (``volume_kspace_scale``); only its preprocessing reads it.
        kspace_scale = kwargs.pop("kspace_scale", None)
        if self.predict_dc is not None:
            self.predict_dc.begin()

        # Preprocess
        if kspace_scale is None:
            processed_input = self.preprocess_input(input_tensor, **kwargs)
        else:
            processed_input = self.preprocess_input(input_tensor, kspace_scale=kspace_scale, **kwargs)

        # Run inference
        raw_output = self.run_inference(processed_input, **kwargs)
        output = raw_output if isinstance(raw_output, torch.Tensor) else raw_output[0]

        # Hard data consistency at predict, BEFORE postprocessing: postprocess
        # moves to CPU, rescales and clamps (GAN maps [-1, 1] to [0, 1]), and
        # the projection must see the model's output on the measurement's
        # scale. Off-knob: ``predict_dc`` is None and nothing here runs.
        output = self._project_onto_measurement(
            output, mask=kwargs.get("mask"), measured_kspace=measured_kspace
        )

        # Postprocess
        return self.postprocess_output(output, **kwargs)

    def _project_onto_measurement(
        self,
        output: torch.Tensor,
        *,
        mask: torch.Tensor | None,
        measured_kspace: torch.Tensor | None,
    ) -> torch.Tensor:
        """The predict-time projection hook (one owner: ``PredictDataConsistency``)."""
        if self.predict_dc is None:
            return output
        return self.predict_dc.finalize(
            output,
            mask=mask,
            measured_kspace=measured_kspace,
            strategy=type(self).__name__,
        )

    def note_data_consistency_applied(self, by: str) -> None:
        """Record that this strategy's own loop pinned the measurement this call.

        A sampler that applies DC per reverse step (diffusion under ``method:
        hard``, cold diffusion given a mask, physics-driven) calls this once
        after its loop so the hook in :meth:`infer` does not project a second
        time and the run summary names who did.
        """
        if self.predict_dc is not None:
            self.predict_dc.note_applied(by)

    def predict_dc_provenance(self) -> dict[str, Any]:
        """What this run can claim about DC at predict, for the results stamp."""
        if self.predict_dc is None:
            return {"apply_at_predict": False}
        return self.predict_dc.provenance()

    def load_checkpoint(self, checkpoint_path: str, **kwargs) -> None:
        """Load the run's EMA weights when it kept them (``use_ema``, default on), else
        its generator; strict, into the bare model."""
        part = (
            "ema" if self.config.get("use_ema", True) and _has_ema(checkpoint_path) else "generator"
        )
        weights(checkpoint_path, part, unwrap_model(self.model), policy="exact")
        self.model.eval()

    def get_strategy_info(self) -> dict[str, Any]:
        """Get information about this inference strategy."""
        return {
            "training_mode": self.training_mode.value,
            "strategy_class": self.__class__.__name__,
            "config": self.config,
        }

    def validate_mixin(self, mixin_name: str, required_attrs: list) -> None:
        """Validate that a mixin has been properly initialized.

        Args:
            mixin_name: Name of the mixin for error messages
            required_attrs: List of attributes that should exist on self

        Raises:
            RuntimeError: If mixin is not properly initialized
            AttributeError: If required attributes are missing
        """
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(
                    f"{mixin_name} not properly initialized: missing attribute '{attr}'. "
                    f"Ensure {mixin_name}.__init__() was called."
                )

            # Validate that attribute is not None
            attr_value = getattr(self, attr)
            if attr_value is None:
                raise RuntimeError(
                    f"{mixin_name} initialization incomplete: attribute '{attr}' is None. "
                    f"Check {mixin_name}.__init__() for proper initialization."
                )

    @torch.no_grad()
    def infer_single(
        self, input_data: torch.Tensor | Any, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Run inference on a single input.

        Default implementation delegates to infer().
        Override for custom single-inference behavior.

        Args:
            input_data: Single input tensor or file path
            **kwargs: Additional parameters

        Returns:
            Inference result
        """
        # If input_data is not a tensor, subclasses should override this method
        if not isinstance(input_data, torch.Tensor):
            # TODO: Implement infer_single for non-tensor inputs in subclasses
            msg = f"{self.__class__.__name__} must implement infer_single() for non-tensor inputs"
            raise NotImplementedError(msg)
        return self.infer(input_data, **kwargs)

    @torch.no_grad()
    def infer_batch(
        self,
        input_data_list: list,
        batch_size: int = 4,
        **kwargs,
    ) -> list:
        """Run inference on a batch of inputs.

        Default implementation processes inputs sequentially using infer_single().
        Override for custom batch-inference behavior.

        Args:
            input_data_list: List of inputs (tensors or file paths)
            batch_size: Batch size for processing
            **kwargs: Additional parameters

        Returns:
            List of inference results
        """
        results = []
        for input_data in input_data_list:
            result = self.infer_single(input_data, **kwargs)
            results.append(result)
        return results
