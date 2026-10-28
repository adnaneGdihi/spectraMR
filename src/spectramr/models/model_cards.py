"""Model Cards for spectraMR trained models.

This module provides comprehensive documentation for trained models,
including performance metrics, training details, and usage guidelines.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from spectramr.shared.utils.safe_io import safe_torch_load

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""

    model_name: str
    model_type: str
    version: str
    created_date: str
    framework: str = "PyTorch"
    license: str = "MIT"

    def to_dict(self) -> dict[str, Any]:
        """to_dict.

        Returns:
            dict[str, Any]: Description.
        """
        return asdict(self)


@dataclass
class TrainingDetails:
    """Training configuration and details."""

    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    loss_functions: dict[str, Any]
    dataset: str
    data_split: dict[str, float]
    hardware: str
    training_time: float | None = None
    early_stopping: bool = False
    normalization: str | None = None  # Data normalization method
    loss_cocktail: dict[str, float] | None = None  # Loss function weights

    def to_dict(self) -> dict[str, Any]:
        """to_dict.

        Returns:
            dict[str, Any]: Description.
        """
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Model performance metrics."""

    psnr: float | None = None
    ssim: float | None = None
    mse: float | None = None
    mae: float | None = None
    fid: float | None = None
    inception_score: float | None = None

    # Medical-specific metrics
    structural_similarity: float | None = None
    feature_similarity: float | None = None
    peak_snr: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """to_dict.

        Returns:
            dict[str, Any]: Description.
        """
        return asdict(self)


@dataclass
class ModelLimitations:
    """Known limitations and biases."""

    resolution_limits: str | None = None
    modality_restrictions: str | None = None
    anatomical_coverage: str | None = None
    training_data_bias: str | None = None
    performance_degradation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """to_dict.

        Returns:
            dict[str, Any]: Description.
        """
        return asdict(self)


@dataclass
class UsageGuidelines:
    """Guidelines for using the model."""

    recommended_use_cases: list[str]
    input_format: str
    output_format: str
    preprocessing_requirements: str | None = None
    postprocessing_suggestions: str | None = None
    performance_expectations: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """to_dict.

        Returns:
            dict[str, Any]: Description.
        """
        return asdict(self)


class ModelCard:
    """Comprehensive model documentation card."""

    def __init__(self, model_name: str, model_type: str, version: str = "1.0.0"):
        """__init__.

        Args:
            model_name (str): Description.
            model_type (str): Description.
            version (str): Description.
        """
        self.metadata = ModelMetadata(
            model_name=model_name,
            model_type=model_type,
            version=version,
            created_date=datetime.now().isoformat(),
        )
        self.training_details: TrainingDetails | None = None
        self.performance_metrics: PerformanceMetrics | None = None
        self.limitations: ModelLimitations | None = None
        self.usage_guidelines: UsageGuidelines | None = None
        self.additional_info: dict[str, Any] = {}

    def set_training_details(self, details: TrainingDetails):
        """Set training configuration details."""
        self.training_details = details

    def set_performance_metrics(self, metrics: PerformanceMetrics):
        """Set model performance metrics."""
        self.performance_metrics = metrics

    def set_limitations(self, limitations: ModelLimitations):
        """Set known limitations and biases."""
        self.limitations = limitations

    def set_usage_guidelines(self, guidelines: UsageGuidelines):
        """Set usage guidelines and recommendations."""
        self.usage_guidelines = guidelines

    def add_additional_info(self, key: str, value: Any):
        """Add additional information to the model card."""
        self.additional_info[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert model card to dictionary."""
        card_dict = {
            "metadata": self.metadata.to_dict(),
            "additional_info": self.additional_info,
        }

        if self.training_details:
            card_dict["training_details"] = self.training_details.to_dict()

        if self.performance_metrics:
            card_dict["performance_metrics"] = self.performance_metrics.to_dict()

        if self.limitations:
            card_dict["limitations"] = self.limitations.to_dict()

        if self.usage_guidelines:
            card_dict["usage_guidelines"] = self.usage_guidelines.to_dict()

        return card_dict

    def save_json(self, filepath: str | Path):
        """Save model card as JSON."""
        filepath = Path(filepath)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def save_yaml(self, filepath: str | Path):
        """Save model card as YAML with fallback to JSON if YAML unavailable."""
        filepath = Path(filepath)

        try:
            # Check if yaml is actually available
            if "yaml" not in globals():
                raise ImportError("yaml module not available")

            # Try to save as YAML
            with open(filepath, "w", encoding="utf-8") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        except (NameError, AttributeError, ImportError, ModuleNotFoundError) as e:
            # YAML not available, fallback to JSON
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("YAML not available (%s), falling back to JSON", e)

            # Change extension to .json if it was .yaml
            if filepath.suffix.lower() == ".yaml":
                json_filepath = filepath.with_suffix(".json")
            else:
                json_filepath = filepath

            with open(json_filepath, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, filepath: str | Path) -> ModelCard:
        """Load model card from JSON."""
        filepath = Path(filepath)
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def load_yaml(cls, filepath: str | Path) -> ModelCard:
        """Load model card from YAML with fallback to JSON if YAML unavailable."""
        filepath = Path(filepath)

        try:
            # Try to load as YAML
            with open(filepath, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        except (NameError, AttributeError, ImportError, ModuleNotFoundError) as e:
            # YAML not available, try loading as JSON
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"YAML not available ({e}), trying to load as JSON")

            # Try loading as JSON
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                return cls.from_dict(data)
            except (json.JSONDecodeError, FileNotFoundError) as json_error:
                # If JSON also fails, try with .json extension
                json_filepath = filepath.with_suffix(".json")
                if json_filepath.exists():
                    with open(json_filepath, encoding="utf-8") as f:
                        data = json.load(f)
                    return cls.from_dict(data)
                raise json_error

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCard:
        """Create model card from dictionary."""
        metadata = data["metadata"]
        card = cls(
            model_name=metadata["model_name"],
            model_type=metadata["model_type"],
            version=metadata["version"],
        )

        # Restore metadata
        card.metadata = ModelMetadata(**metadata)

        # Restore training details
        if "training_details" in data:
            card.training_details = TrainingDetails(**data["training_details"])

        # Restore performance metrics
        if "performance_metrics" in data:
            card.performance_metrics = PerformanceMetrics(**data["performance_metrics"])

        # Restore limitations
        if "limitations" in data:
            card.limitations = ModelLimitations(**data["limitations"])

        # Restore usage guidelines
        if "usage_guidelines" in data:
            card.usage_guidelines = UsageGuidelines(**data["usage_guidelines"])

        # Restore additional info
        card.additional_info = data.get("additional_info", {})

        return card

    def generate_markdown_report(self) -> str:
        """Generate a markdown report of the model card."""
        lines = []

        # Header
        lines.append(f"# Model Card: {self.metadata.model_name}")
        lines.append("")

        # Metadata
        lines.append("## Model Information")
        lines.append(f"- **Model Type**: {self.metadata.model_type}")
        lines.append(f"- **Version**: {self.metadata.version}")
        lines.append(f"- **Created**: {self.metadata.created_date}")
        lines.append(f"- **Framework**: {self.metadata.framework}")
        lines.append(f"- **License**: {self.metadata.license}")
        lines.append("")

        # Training Details
        if self.training_details:
            lines.append("## Training Details")
            td = self.training_details
            lines.append(f"- **Epochs**: {td.epochs}")
            lines.append(f"- **Batch Size**: {td.batch_size}")
            lines.append(f"- **Learning Rate**: {td.learning_rate}")
            lines.append(f"- **Optimizer**: {td.optimizer}")
            lines.append(f"- **Dataset**: {td.dataset}")
            lines.append(f"- **Hardware**: {td.hardware}")
            if td.training_time:
                lines.append(f"- **Training Time**: {td.training_time:.2f} hours")
            if td.normalization:
                lines.append(f"- **Normalization**: {td.normalization}")
            if td.loss_cocktail:
                lines.append("- **Loss Cocktail**:")
                for loss_name, weight in td.loss_cocktail.items():
                    lines.append(f"  - {loss_name}: {weight}")
            lines.append("")

        # Performance Metrics
        if self.performance_metrics:
            lines.append("## Performance Metrics")
            pm = self.performance_metrics
            if pm.psnr:
                lines.append(f"- **PSNR**: {pm.psnr:.2f} dB")
            if pm.ssim:
                lines.append(f"- **SSIM**: {pm.ssim:.4f}")
            if pm.mse:
                lines.append(f"- **MSE**: {pm.mse:.6f}")
            if pm.mae:
                lines.append(f"- **MAE**: {pm.mae:.6f}")
            if pm.fid:
                lines.append(f"- **FID**: {pm.fid:.2f}")
            if pm.structural_similarity:
                lines.append(
                    f"- **Structural Similarity**: {pm.structural_similarity:.4f}",
                )
            lines.append("")

        # Usage Guidelines
        if self.usage_guidelines:
            lines.append("## Usage Guidelines")
            ug = self.usage_guidelines
            lines.append("### Recommended Use Cases")
            for use_case in ug.recommended_use_cases:
                lines.append(f"- {use_case}")
            lines.append("")
            lines.append(f"### Input Format: {ug.input_format}")
            lines.append(f"### Output Format: {ug.output_format}")
            if ug.preprocessing_requirements:
                lines.append(f"### Preprocessing: {ug.preprocessing_requirements}")
            if ug.postprocessing_suggestions:
                lines.append(f"### Postprocessing: {ug.postprocessing_suggestions}")
            lines.append("")

        # Limitations
        if self.limitations:
            lines.append("## Limitations and Biases")
            lim = self.limitations
            if lim.resolution_limits:
                lines.append(f"- **Resolution Limits**: {lim.resolution_limits}")
            if lim.modality_restrictions:
                lines.append(
                    f"- **Modality Restrictions**: {lim.modality_restrictions}",
                )
            if lim.anatomical_coverage:
                lines.append(f"- **Anatomical Coverage**: {lim.anatomical_coverage}")
            if lim.training_data_bias:
                lines.append(f"- **Training Data Bias**: {lim.training_data_bias}")
            if lim.performance_degradation:
                lines.append(
                    f"- **Performance Degradation**: {lim.performance_degradation}",
                )
            lines.append("")

        # Additional Information
        if self.additional_info:
            lines.append("## Additional Information")
            if self.additional_info.get("model_size"):
                lines.append(
                    f"- **Memory Footprint**: {self.additional_info['model_size']}",
                )
            if self.additional_info.get("inference_time"):
                lines.append(
                    f"- **Inference Latency**: {self.additional_info['inference_time']:.2f} ms",
                )
            if self.additional_info.get("checkpoint_path"):
                lines.append(
                    f"- **Checkpoint Path**: {self.additional_info['checkpoint_path']}",
                )
            lines.append("")

        return "\n".join(lines)


class ModelCardGenerator:
    """Automated model card generation from training results."""

    def __init__(self, output_dir: str | Path = "./model_cards"):
        """__init__.

        Args:
            output_dir (str | Path): Description.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_from_training(
        self,
        model_name: str,
        model_type: str,
        config: Any,
        metrics: dict[str, Any],
        checkpoint_path: str | Path | None = None,
    ) -> ModelCard:
        """Generate model card from training results."""
        # Create model card
        card = ModelCard(model_name, model_type)

        # Set training details
        training_details = TrainingDetails(
            epochs=config.epochs if hasattr(config, "epochs") else 100,
            batch_size=config.batch_size if hasattr(config, "batch_size") else 8,
            learning_rate=(config.learning_rate if hasattr(config, "learning_rate") else 0.0002),
            optimizer=config.optimizer if hasattr(config, "optimizer") else "adam",
            loss_functions={
                "generator": config.gan_loss if hasattr(config, "gan_loss") else "bce",
                "discriminator": "bce",
                "l1_weight": (config.l1_weight if hasattr(config, "l1_weight") else 100.0),
            },
            dataset=config.dataset if hasattr(config, "dataset") else "MRI Dataset",
            data_split={"train": 0.7, "val": 0.2, "test": 0.1},
            hardware="GPU" if torch.cuda.is_available() else "CPU",
            normalization="Min-Max normalization to [0,1] range",
            loss_cocktail={
                "adversarial": 1.0,
                "l1": config.l1_weight if hasattr(config, "l1_weight") else 100.0,
                "perceptual": config.perceptual_weight,
            },
        )
        card.set_training_details(training_details)

        # Set performance metrics
        performance_metrics = PerformanceMetrics(
            psnr=metrics.get("psnr"),
            ssim=metrics.get("ssim"),
            mse=metrics.get("mse"),
            mae=metrics.get("mae"),
            fid=metrics.get("fid"),
            structural_similarity=metrics.get("structural_similarity"),
            peak_snr=metrics.get("peak_snr"),
        )
        card.set_performance_metrics(performance_metrics)

        # Set usage guidelines
        usage_guidelines = UsageGuidelines(
            recommended_use_cases=[
                "Medical image super-resolution",
                "MRI image enhancement",
                f"{model_type} architecture evaluation",
            ],
            input_format="3D NIfTI files (.nii/.nii.gz) or 2D numpy arrays",
            output_format="Super-resolved 3D NIfTI files or 2D numpy arrays",
            preprocessing_requirements="Images should be normalized to [0,1] range",
            postprocessing_suggestions="Apply appropriate windowing for medical visualization",
            performance_expectations=f"Expected PSNR: {metrics.get('psnr', 'N/A')}",
        )
        card.set_usage_guidelines(usage_guidelines)

        # Set limitations
        limitations = ModelLimitations(
            resolution_limits="Optimized for 256x256 to 512x512 resolution scaling",
            modality_restrictions="Trained on T1-weighted MRI, may not generalize to other modalities",
            anatomical_coverage="Optimized for brain MRI, performance may vary for other anatomies",
            training_data_bias="Performance may be biased toward training dataset characteristics",
            performance_degradation="May show reduced performance on images with severe artifacts",
        )
        card.set_limitations(limitations)

        # Add additional information
        card.add_additional_info(
            "checkpoint_path",
            str(checkpoint_path) if checkpoint_path else None,
        )
        card.add_additional_info(
            "model_size",
            self._estimate_model_size(checkpoint_path),
        )
        card.add_additional_info("inference_time", metrics.get("inference_time"))

        return card

    def _estimate_model_size(self, checkpoint_path: str | Path | None) -> str | None:
        """Estimate model size from checkpoint."""
        if not checkpoint_path or not Path(checkpoint_path).exists():
            return None

        try:
            checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
            total_params = 0

            for _key, value in checkpoint.items():
                if isinstance(value, torch.Tensor):
                    total_params += value.numel()

            # Estimate size in MB (assuming float32)
            size_mb = (total_params * 4) / (1024 * 1024)
            return f"{size_mb:.2f} MB"
        except Exception:
            return None

    def save_model_card(
        self,
        card: ModelCard,
        format: str = "json",
        filename: str | None = None,
    ):
        """Save model card to file."""
        if filename is None:
            filename = f"{card.metadata.model_name}_v{card.metadata.version}"

        if format == "json":
            filepath = self.output_dir / f"{filename}.json"
            card.save_json(filepath)
        elif format == "yaml":
            filepath = self.output_dir / f"{filename}.yaml"
            card.save_yaml(filepath)
        elif format == "markdown":
            filepath = self.output_dir / f"{filename}.md"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(card.generate_markdown_report())
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.debug(f"Model card saved to: {filepath}")
        return filepath


def create_model_card_for_experiment(
    experiment_name: str,
    model_type: str,
    config: Any,
    metrics: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path = "./model_cards",
) -> ModelCard:
    """Convenience function to create and save a model card for an experiment."""
    generator = ModelCardGenerator(output_dir)
    card = generator.generate_from_training(
        experiment_name,
        model_type,
        config,
        metrics,
        checkpoint_path,
    )

    # Save in multiple formats
    generator.save_model_card(card, format="json")
    generator.save_model_card(card, format="markdown")

    return card
