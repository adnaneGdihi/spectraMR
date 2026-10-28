"""Model Management Service

Comprehensive service for managing model lifecycle, versioning, checkpoints,
and metadata throughout the spectraMR project.
"""

import json
import logging
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from spectramr.infrastructure.services import IModelManagementService
from spectramr.models.model_cards import ModelCard, PerformanceMetrics, TrainingDetails


@dataclass
class ModelInfo:
    """Information about a registered model."""

    model_id: str
    model_type: str
    version: str
    status: str  # 'active', 'archived', 'deprecated'
    created_at: str
    updated_at: str
    checkpoint_dir: str
    export_dir: str
    metadata: dict[str, Any]
    model_card_path: str | None = None


class ModelManagementService(IModelManagementService):
    """Comprehensive model management service.

    Provides complete lifecycle management for models including:
    - Registration and versioning
    - Checkpoint management
    - Model export and deployment
    - Model cards and documentation
    - Metadata tracking
    """

    def __init__(
        self,
        base_dir: str = "./models",
        logging_service: Any | None = None,
    ):
        """Initialize the model management service.

        Args:
            base_dir: Base directory for model storage
            logging_service: Optional logging service for tracking

        """
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "registry"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.exports_dir = self.base_dir / "exports"
        self.cards_dir = self.base_dir / "cards"

        # Create directories
        for dir_path in [
            self.models_dir,
            self.checkpoints_dir,
            self.exports_dir,
            self.cards_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Dependencies
        self.logging_service = logging_service
        self.logger = logging.getLogger(__name__)

        # In-memory registry
        self._model_registry: dict[str, ModelInfo] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load model registry from disk."""
        registry_file = self.models_dir / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file) as f:
                    data = json.load(f)
                    for model_id, info_dict in data.items():
                        self._model_registry[model_id] = ModelInfo(**info_dict)
            except Exception as e:
                self.logger.warning(f"Failed to load model registry: {e}")

    def _save_registry(self) -> None:
        """Save model registry to disk atomically."""
        registry_file = self.models_dir / "registry.json"
        try:
            data = {model_id: asdict(info) for model_id, info in self._model_registry.items()}
            # [FORENSIC FIX] Atomic write to prevent corruption on crash
            from spectramr.shared.utils.safe_io import atomic_save_json

            atomic_save_json(data, registry_file)
        except Exception as e:
            self.logger.error(f"Failed to save model registry: {e}")

    def register_model(
        self,
        model: torch.nn.Module,
        model_type: str,
        version: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a model with metadata and versioning.

        Args:
            model: The PyTorch model to register
            model_type: Type/category of the model
            version: Version string for the model
            metadata: Additional metadata for the model

        Returns:
            Unique model ID

        """
        model_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        # Create model info
        model_info = ModelInfo(
            model_id=model_id,
            model_type=model_type,
            version=version,
            status="active",
            created_at=now,
            updated_at=now,
            checkpoint_dir=str(self.checkpoints_dir / model_id),
            export_dir=str(self.exports_dir / model_id),
            metadata=metadata or {},
        )

        # Store in registry
        self._model_registry[model_id] = model_info
        self._save_registry()

        # Create directories
        Path(model_info.checkpoint_dir).mkdir(exist_ok=True)
        Path(model_info.export_dir).mkdir(exist_ok=True)

        self.logger.info(
            "Registered model %s (%s v%s)",
            model_id,
            model_type,
            version,
        )

        return model_id

    def get_model(self, model_id: str) -> torch.nn.Module | None:
        """Retrieve a model by ID.

        Note: This returns None as models are typically loaded from checkpoints
        rather than stored in memory. Use ``checkpointing.restore`` instead.

        Args:
            model_id: Unique model identifier

        Returns:
            None (models should be loaded from checkpoints)

        """
        if model_id not in self._model_registry:
            self.logger.warning(f"Model {model_id} not found in registry")
            return None

        # Models are typically loaded from checkpoints, not stored in memory
        return None

    def list_models(
        self,
        model_type: str | None = None,
        version: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """list models with optional filtering.

        Args:
            model_type: Filter by model type
            version: Filter by version
            status: Filter by status ('active', 'archived', 'deprecated')

        Returns:
            list of model information dictionaries

        """
        models: list[dict[str, Any]] = []

        for model_info in self._model_registry.values():
            # Apply filters
            if model_type and model_info.model_type != model_type:
                continue
            if version and model_info.version != version:
                continue
            if status and model_info.status != status:
                continue

            models.append(asdict(model_info))

        return models

    def update_model_metadata(
        self,
        model_id: str,
        metadata: dict[str, Any],
    ) -> bool:
        """Update model metadata.

        Args:
            model_id: Unique model identifier
            metadata: Metadata to update

        Returns:
            True if successful, False otherwise

        """
        if model_id not in self._model_registry:
            self.logger.warning(f"Model {model_id} not found")
            return False

        model_info = self._model_registry[model_id]
        model_info.metadata.update(metadata)
        model_info.updated_at = datetime.now().isoformat()

        self._save_registry()
        self.logger.info(f"Updated metadata for model {model_id}")

        return True

    def export_model(
        self,
        model_id: str,
        export_format: str,
        export_path: str | None = None,
        model: torch.nn.Module | None = None,
        input_shape: tuple | None = None,
    ) -> str:
        """Export model in specified format.

        Args:
            model_id: Unique model identifier
            export_format: Export format ('onnx', 'torchscript', 'jit')
            export_path: Optional custom export path
            model: Model instance to export (required if not already loaded)
            input_shape: Input shape for ONNX export (e.g. (1, 1, 256, 256))

        Returns:
            Path to exported model

        """
        if model_id not in self._model_registry:
            raise ValueError(f"Model {model_id} not registered")

        model_info = self._model_registry[model_id]

        if export_path is None:
            # Add extension if missing
            if not export_format.startswith("."):
                ext = f".{export_format}"
            else:
                ext = export_format

            fname = f"{model_info.model_type}_v{model_info.version}{ext}"
            export_path = os.path.join(model_info.export_dir, fname)

        if model is None:
            # We can't export without the model architecture in memory.
            # In a full DI system, we might use a ModelFactory here.
            # For now, we require the model instance to be passed.
            raise ValueError("Model instance must be provided for export")

        self.logger.info(
            f"Exporting model {model_id} to {export_format}: {export_path}",
        )

        # Ensure export dir exists
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        try:
            model.eval()

            if export_format.lower() == "onnx":
                trace_input = self._resolve_trace_input(model, model_info, input_shape)

                torch.onnx.export(
                    model,
                    trace_input,
                    export_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "output": {0: "batch_size"},
                    },
                )

            elif export_format.lower() in ["torchscript", "jit", "pt"]:
                # Use scripting if possible, tracing for complex control flow
                try:
                    scripted_model = torch.jit.script(model)
                    scripted_model.save(export_path)
                except Exception as e:
                    self.logger.warning(f"Scripting failed ({e}), falling back to trace")
                    trace_input = self._resolve_trace_input(model, model_info, input_shape)
                    traced_model = torch.jit.trace(model, trace_input)
                    traced_model.save(export_path)

            else:
                raise ValueError(f"Unsupported export format: {export_format}")

            self.logger.info(f"Export successful: {export_path}")

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            raise RuntimeError(f"Failed to export model: {e}")

        return export_path

    def _resolve_trace_input(
        self,
        model: torch.nn.Module,
        model_info: ModelInfo,
        input_shape: tuple | None,
    ) -> torch.Tensor:
        """Resolve a representative input tensor for ONNX/TorchScript tracing.

        Resolution order:
        1. Explicit ``input_shape`` parameter → ``torch.randn(input_shape)``
        2. DI-backed ``get_sample_batch()`` from configured data pipeline
        3. Model-type-aware default shape (last resort)
        """
        device = next(model.parameters()).device

        # 1. Explicit shape always wins — this is not a "dummy" fallback,
        #    it is an intentional caller specification.
        if input_shape is not None:
            self.logger.info(f"Using explicit input_shape {input_shape} for tracing")
            return torch.randn(input_shape, device=device)

        # 2. Try fetching a real sample from the data pipeline via DI
        try:
            from spectramr.config.settings import TrainingSettings
            from spectramr.infrastructure.di.di_container import resolve_service
            from spectramr.shared.utils.data_utils import get_sample_batch

            config = resolve_service(TrainingSettings)
            sample = get_sample_batch(config, device=device)
            self.logger.info("Using dynamic sample batch (shape=%s) for tracing", sample.shape)
            return sample
        except Exception as e:
            self.logger.warning(
                "Dynamic sample fetch failed (%s), using model-type default shape", e
            )

        # 3. Model-type-aware default (last resort)
        channels = (
            2 if ("complex" in model_info.model_type or "gan" in model_info.model_type) else 1
        )
        fallback_shape = (1, channels, 256, 256)
        self.logger.warning(f"Falling back to default input_shape {fallback_shape}")
        return torch.randn(fallback_shape, device=device)

    def create_model_card(
        self,
        model_id: str,
        training_details: dict[str, Any],
        performance_metrics: dict[str, Any] | None = None,
        output_dir: str | None = None,
    ) -> str:
        """Create comprehensive model card.

        Args:
            model_id: Unique model identifier
            training_details: Training configuration details
            performance_metrics: Optional performance metrics
            output_dir: Optional directory to save the model card
                (defaults to self.cards_dir)

        Returns:
            Path to saved model card

        """
        if model_id not in self._model_registry:
            raise ValueError(f"Model {model_id} not registered")

        model_info = self._model_registry[model_id]

        # Create training details
        training_details_obj = TrainingDetails(**training_details)

        # Create performance metrics if provided
        performance_metrics_obj = None
        if performance_metrics:
            performance_metrics_obj = PerformanceMetrics(**performance_metrics)

        # Create model card
        model_card = ModelCard(
            model_name=f"{model_info.model_type}_{model_id}",
            model_type=model_info.model_type,
            version=model_info.version,
        )

        model_card.set_training_details(training_details_obj)
        if performance_metrics_obj:
            model_card.set_performance_metrics(performance_metrics_obj)

        # Add model metadata
        model_card.add_additional_info("model_id", model_id)
        model_card.add_additional_info("created_at", model_info.created_at)
        model_card.add_additional_info("status", model_info.status)

        # Save model card
        card_filename = f"{model_info.model_type}_v{model_info.version}_card.json"
        output_path = Path(output_dir) if output_dir else self.cards_dir
        card_path = output_path / card_filename
        model_card.save_json(card_path)

        # Update model info
        model_info.model_card_path = str(card_path)
        model_info.updated_at = datetime.now().isoformat()
        self._save_registry()

        self.logger.info(f"Created model card for {model_id}: {card_path}")
        return str(card_path)

    def get_model_card(self, model_id: str) -> Any | None:
        """Retrieve model card.

        Args:
            model_id: Unique model identifier

        Returns:
            ModelCard object or None if not found

        """
        if model_id not in self._model_registry:
            return None

        model_info = self._model_registry[model_id]
        if not model_info.model_card_path or not os.path.exists(
            model_info.model_card_path,
        ):
            return None

        try:
            return ModelCard.load_json(model_info.model_card_path)
        except Exception as e:
            self.logger.error(f"Failed to load model card for {model_id}: {e}")
            return None

    def delete_model(self, model_id: str) -> bool:
        """Delete model and all associated artifacts.

        Args:
            model_id: Unique model identifier

        Returns:
            True if successful, False otherwise

        """
        if model_id not in self._model_registry:
            self.logger.warning(f"Model {model_id} not found")
            return False

        model_info = self._model_registry[model_id]

        try:
            # Remove directories
            for dir_path in [model_info.checkpoint_dir, model_info.export_dir]:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)

            # Remove from registry
            del self._model_registry[model_id]
            self._save_registry()

            self.logger.info(f"Deleted model {model_id} and all artifacts")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete model {model_id}: {e}")
            return False
