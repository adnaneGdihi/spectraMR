"""Comprehensive unit tests for interface definitions.

Tests cover:
- Service interfaces: ILoggingService, IDeviceService, ICheckpointService, etc.
- Dataset interfaces: IDataLoader, IImageTransformer, IAugmentationStrategy, etc.
- Protocol compliance and abstract method definitions
- Interface inheritance and method signatures
"""

import inspect
from abc import ABC
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from spectramr.domain.interfaces.checkpoint_service_interface import ICheckpointService
from spectramr.domain.interfaces.dataset_interfaces import (
    IAugmentationStrategy,
    ICacheManager,
    IDataLoader,
    IFileDiscoverer,
    IImageTransformer,
)
from spectramr.domain.interfaces.memory_optimization_interface import (
    IMemoryOptimizationService,
)
from spectramr.domain.interfaces.service_interfaces import (
    IConfigurationService,
    IDeviceManager,
    ILoggingService,
    IMetricsService,
    ITrainingOrchestrator,
)
from spectramr.domain.interfaces.service_interfaces import (
    IMemoryOptimizationService as IMemoryOptService,
)


class TestILoggingService:
    """Test ILoggingService interface."""

    def test_is_abstract_class(self):
        """Test that ILoggingService is an abstract class."""
        assert inspect.isabstract(ILoggingService)
        assert issubclass(ILoggingService, ABC)

    def test_abstract_methods(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = ILoggingService.__abstractmethods__

        expected_methods = {
            "setup",
            "get_logger",
            "log_info",
            "log_warning",
            "log_error",
            "log_critical",
            "log_debug",
            "log_images_batch",
            "epoch",
            "step",
            "set_epoch",
            "set_step",
            "log_images",
        }

        assert abstract_methods == expected_methods

    def test_method_signatures(self):
        """Test method signatures are correct."""
        # Check setup method
        setup_sig = inspect.signature(ILoggingService.setup)
        assert "log_dir" in setup_sig.parameters
        assert "log_level" in setup_sig.parameters
        assert setup_sig.parameters["log_level"].default == "INFO"

        # Check log_info method
        log_info_sig = inspect.signature(ILoggingService.log_info)
        assert "message" in log_info_sig.parameters
        assert "model_type" in log_info_sig.parameters
        assert log_info_sig.parameters["model_type"].default == ""

    def test_can_create_concrete_implementation(self):
        """Test that we can create a concrete implementation."""

        class ConcreteLoggingService(ILoggingService):
            def setup(
                self,
                log_dir: str,
                log_level: str = "INFO",
                config: dict[str, Any] | None = None,
            ) -> None:
                pass

            def get_logger(self, name: str) -> Any:
                return Mock()

            def log_info(
                self,
                message: str,
                model_type: str = "",
                epoch: int = -1,
                extra: dict[str, Any] | None = None,
            ) -> None:
                pass

            def log_warning(
                self,
                message: str,
                model_type: str = "",
                epoch: int = -1,
                extra: dict[str, Any] | None = None,
            ) -> None:
                pass

            def log_error(
                self,
                message: str,
                model_type: str = "",
                epoch: int = -1,
                extra: dict[str, Any] | None = None,
            ) -> None:
                pass

            def log_critical(
                self,
                message: str,
                model_type: str = "",
                epoch: int = -1,
                extra: dict[str, Any] | None = None,
            ) -> None:
                pass

            def log_debug(
                self,
                message: str,
                model_type: str = "",
                epoch: int = -1,
                extra: dict[str, Any] | None = None,
            ) -> None:
                pass

            def log_images_batch(
                self,
                images: torch.Tensor,
                step: int,
                prefix: str | None = None,
                **kwargs: Any,
            ) -> None:
                pass

            @property
            def epoch(self) -> int:
                return 0

            @property
            def step(self) -> int:
                return 0

            def set_epoch(self, epoch: int) -> None:
                pass

            def set_step(self, step: int) -> None:
                pass

            def log_images(
                self,
                images: Any,
                title: str,
                step: int | None = None,
                **kwargs: Any,
            ) -> None:
                pass

        # Should not raise an exception
        service = ConcreteLoggingService()
        assert isinstance(service, ILoggingService)


class TestIDeviceManager:
    """Test IDeviceManager interface."""

    def test_is_abstract_class(self):
        """Test that IDeviceManager is an abstract class."""
        assert inspect.isabstract(IDeviceManager)
        assert issubclass(IDeviceManager, ABC)

    def test_abstract_methods(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = IDeviceManager.__abstractmethods__

        expected_methods = {
            "get_device",
            "move_to_device",
            "is_cuda_available",
        }

        assert abstract_methods == expected_methods

    def test_can_create_concrete_implementation(self):
        """Test that we can create a concrete implementation."""

        class ConcreteDeviceManager(IDeviceManager):
            def get_device(self) -> torch.device:
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")

            def move_to_device(self, data: Any) -> Any:
                return data

            def is_cuda_available(self) -> bool:
                return False

        service = ConcreteDeviceManager()
        assert isinstance(service, IDeviceManager)


class TestICheckpointService:
    """Test ICheckpointService interface."""

    def test_is_abstract_class(self):
        """Test that ICheckpointService is an abstract class."""
        assert inspect.isabstract(ICheckpointService)
        assert issubclass(ICheckpointService, ABC)

    def test_abstract_methods(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = ICheckpointService.__abstractmethods__

        expected_methods = {
            "save_checkpoint",
            "load_checkpoint",
            "find_latest_checkpoint",
        }

        assert abstract_methods == expected_methods

    def test_method_signatures(self):
        """Test method signatures are correct."""
        save_sig = inspect.signature(ICheckpointService.save_checkpoint)
        assert "model" in save_sig.parameters
        assert "optimizer" in save_sig.parameters
        assert "epoch" in save_sig.parameters
        assert "loss" in save_sig.parameters
        assert "file_path" in save_sig.parameters

    def test_can_create_concrete_implementation(self):
        """Test that we can create a concrete implementation."""

        class ConcreteCheckpointService(ICheckpointService):
            def save_checkpoint(
                self,
                model: Any,
                optimizer: Any,
                epoch: int,
                loss: float,
                file_path: str,
                **kwargs: Any,
            ) -> None:
                pass

            def load_checkpoint(
                self,
                model: Any,
                optimizer: Any,
                file_path: str,
                **kwargs: Any,
            ) -> dict[str, Any] | None:
                return None

            def find_latest_checkpoint(self, checkpoint_dir: str) -> str | None:
                return None

        service = ConcreteCheckpointService()
        assert isinstance(service, ICheckpointService)


class TestIMemoryOptimizationService:
    """Test IMemoryOptimizationService interface."""

    def test_is_abstract_class(self):
        """Test that IMemoryOptimizationService is an abstract class."""
        assert inspect.isabstract(IMemoryOptimizationService)
        assert issubclass(IMemoryOptimizationService, ABC)

    def test_abstract_methods(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = IMemoryOptimizationService.__abstractmethods__

        expected_methods = {
            "optimize_memory",
        }

        assert abstract_methods == expected_methods

    def test_can_create_concrete_implementation(self):
        """Test that we can create a concrete implementation."""

        class ConcreteMemoryOptimizationService(IMemoryOptimizationService):
            def optimize_memory(self):
                pass

        service = ConcreteMemoryOptimizationService()
        assert isinstance(service, IMemoryOptimizationService)


class TestIMetricsService:
    """Test IMetricsService interface."""

    def test_is_abstract_class(self):
        """Test that IMetricsService is an abstract class."""
        assert inspect.isabstract(IMetricsService)
        assert issubclass(IMetricsService, ABC)

    def test_abstract_methods(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = IMetricsService.__abstractmethods__

        expected_methods = {
            "update_metrics",
            "get_current_metrics",
            "reset_metrics",
            "log_metrics",
        }

        assert abstract_methods == expected_methods

    def test_can_create_concrete_implementation(self):
        """Test that we can create a concrete implementation."""

        class ConcreteMetricsService(IMetricsService):
            def update_metrics(self, metrics: dict[str, Any]) -> None:
                pass

            def get_current_metrics(self) -> dict[str, Any]:
                return {}

            def reset_metrics(self) -> None:
                pass

            def log_metrics(self, step: int) -> None:
                pass

        service = ConcreteMetricsService()
        assert isinstance(service, IMetricsService)


class TestIDataLoader:
    """Test IDataLoader interface."""

    def test_is_abstract_class(self):
        """Test that IDataLoader is an abstract class."""
        assert inspect.isabstract(IDataLoader)
        assert issubclass(IDataLoader, ABC)

    def test_abstract_methods(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = IDataLoader.__abstractmethods__

        expected_methods = {
            "load_image_pair",
        }

        assert abstract_methods == expected_methods

    def test_method_signatures(self):
        """Test method signatures are correct."""
        sig = inspect.signature(IDataLoader.load_image_pair)
        assert "lr_path" in sig.parameters
        assert "hr_path" in sig.parameters

    def test_can_create_concrete_implementation(self):
        """Test that we can create a concrete implementation."""

        class ConcreteDataLoader(IDataLoader):
            def load_image_pair(
                self, lr_path: str, hr_path: str
            ) -> tuple[Image.Image, Image.Image]:
                return Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))

        loader = ConcreteDataLoader()
        assert isinstance(loader, IDataLoader)


class TestIImageTransformer:
    """Test IImageTransformer interface."""

    def test_is_abstract_class(self):
        """Test that IImageTransformer is an abstract class."""
        assert inspect.isabstract(IImageTransformer)
        assert issubclass(IImageTransformer, ABC)

    def test_abstract_methods(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = IImageTransformer.__abstractmethods__

        expected_methods = {
            "transform_to_tensor",
        }

        assert abstract_methods == expected_methods

    def test_can_create_concrete_implementation(self):
        """Test that we can create a concrete implementation."""

        class ConcreteImageTransformer(IImageTransformer):
            def transform_to_tensor(
                self, lr_img: Image.Image, hr_img: Image.Image
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return torch.randn(3, 64, 64), torch.randn(3, 64, 64)

        transformer = ConcreteImageTransformer()
        assert isinstance(transformer, IImageTransformer)


class TestIAugmentationStrategy:
    """Test IAugmentationStrategy interface."""

    def test_is_abstract_class(self):
        """Test that IAugmentationStrategy is an abstract class."""
        assert inspect.isabstract(IAugmentationStrategy)
        assert issubclass(IAugmentationStrategy, ABC)

    def test_abstract_methods(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = IAugmentationStrategy.__abstractmethods__

        expected_methods = {
            "apply_augmentations",
        }

        assert abstract_methods == expected_methods

    def test_can_create_concrete_implementation(self):
        """Test that we can create a concrete implementation."""

        class ConcreteAugmentationStrategy(IAugmentationStrategy):
            def apply_augmentations(
                self, lr_tensor: torch.Tensor, hr_tensor: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return lr_tensor, hr_tensor

        strategy = ConcreteAugmentationStrategy()
        assert isinstance(strategy, IAugmentationStrategy)


class TestICacheManager:
    """Test ICacheManager interface."""

    def test_is_abstract_class(self):
        """Test that ICacheManager is an abstract class."""
        assert inspect.isabstract(ICacheManager)
        assert issubclass(ICacheManager, ABC)

    def test_abstract_methods(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = ICacheManager.__abstractmethods__

        expected_methods = {
            "get_cached_item",
            "cache_item",
            "is_enabled",
        }

        assert abstract_methods == expected_methods

    def test_can_create_concrete_implementation(self):
        """Test that we can create a concrete implementation."""

        class ConcreteCacheManager(ICacheManager):
            def get_cached_item(self, idx: int) -> Any | None:
                return None

            def cache_item(self, idx: int, data: Any) -> None:
                pass

            def is_enabled(self) -> bool:
                return True

        manager = ConcreteCacheManager()
        assert isinstance(manager, ICacheManager)


class TestIFileDiscoverer:
    """Test IFileDiscoverer interface."""

    def test_is_abstract_class(self):
        """Test that IFileDiscoverer is an abstract class."""
        assert inspect.isabstract(IFileDiscoverer)
        assert issubclass(IFileDiscoverer, ABC)

    def test_abstract_methods(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = IFileDiscoverer.__abstractmethods__

        expected_methods = {
            "find_image_files",
        }

        assert abstract_methods == expected_methods

    def test_can_create_concrete_implementation(self):
        """Test that we can create a concrete implementation."""

        class ConcreteFileDiscoverer(IFileDiscoverer):
            def find_image_files(self, directory: str) -> list[str]:
                return []

        discoverer = ConcreteFileDiscoverer()
        assert isinstance(discoverer, IFileDiscoverer)


class TestIConfigurationService:
    """Test IConfigurationService interface."""

    def test_is_abstract_class(self):
        """Test that IConfigurationService is an abstract class."""
        assert inspect.isabstract(IConfigurationService)
        assert issubclass(IConfigurationService, ABC)

    def test_abstract_methods(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = IConfigurationService.__abstractmethods__

        expected_methods = {
            "load_config",
            "validate_config",
            "update_config",
        }

        assert abstract_methods == expected_methods

    def test_can_create_concrete_implementation(self):
        """Test that we can create a concrete implementation."""

        class ConcreteConfigurationService(IConfigurationService):
            def load_config(self, path: str) -> dict[str, Any]:
                return {}

            def validate_config(self, config: dict[str, Any]) -> bool:
                return True

            def update_config(
                self, config: dict[str, Any], updates: dict[str, Any]
            ) -> dict[str, Any]:
                return config

        service = ConcreteConfigurationService()
        assert isinstance(service, IConfigurationService)


class TestITrainingOrchestrator:
    """Test ITrainingOrchestrator interface."""

    def test_is_abstract_class(self):
        """Test that ITrainingOrchestrator is an abstract class."""
        assert inspect.isabstract(ITrainingOrchestrator)
        assert issubclass(ITrainingOrchestrator, ABC)

    def test_abstract_methods(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = ITrainingOrchestrator.__abstractmethods__

        expected_methods = {
            "orchestrate_training",
            "setup_training",
            "cleanup_training",
            "finalize_training",
            "run_full_training_pipeline",
        }

        assert abstract_methods == expected_methods

    def test_can_create_concrete_implementation(self):
        """Test that we can create a concrete implementation."""

        class ConcreteTrainingOrchestrator(ITrainingOrchestrator):
            def orchestrate_training(self, config: dict[str, Any]) -> dict[str, Any]:
                return {}

            def setup_training(self, config: dict[str, Any]) -> None:
                pass

            def cleanup_training(self) -> None:
                pass

            def finalize_training(self, session: Any) -> None:
                """Implementation required by Interface update."""
                pass

            def run_full_training_pipeline(self, config: Any) -> dict[str, Any]:
                """Implementation required by Interface update."""
                return {}

        orchestrator = ConcreteTrainingOrchestrator()
        assert isinstance(orchestrator, ITrainingOrchestrator)


class TestInterfaceInheritance:
    """Test interface inheritance and multiple interface implementation."""

    def test_multiple_interface_inheritance(self):
        """Test that a class can implement multiple interfaces."""

        class MultiInterfaceService(
            ILoggingService,
            IDeviceManager,
            IMetricsService,
        ):
            # ILoggingService methods
            def setup(
                self,
                log_dir: str,
                log_level: str = "INFO",
                config: dict[str, Any] | None = None,
            ) -> None:
                pass

            def get_logger(self, name: str) -> Any:
                return Mock()

            def log_info(
                self,
                message: str,
                model_type: str = "",
                epoch: int = -1,
                extra: dict[str, Any] | None = None,
            ) -> None:
                pass

            def log_warning(
                self,
                message: str,
                model_type: str = "",
                epoch: int = -1,
                extra: dict[str, Any] | None = None,
            ) -> None:
                pass

            def log_error(
                self,
                message: str,
                model_type: str = "",
                epoch: int = -1,
                extra: dict[str, Any] | None = None,
            ) -> None:
                pass

            def log_critical(
                self,
                message: str,
                model_type: str = "",
                epoch: int = -1,
                extra: dict[str, Any] | None = None,
            ) -> None:
                pass

            def log_debug(
                self,
                message: str,
                model_type: str = "",
                epoch: int = -1,
                extra: dict[str, Any] | None = None,
            ) -> None:
                pass

            def log_images_batch(
                self,
                images: torch.Tensor,
                step: int,
                prefix: str | None = None,
                **kwargs: Any,
            ) -> None:
                pass

            @property
            def epoch(self) -> int:
                return 0

            @property
            def step(self) -> int:
                return 0

            def set_epoch(self, epoch: int) -> None:
                pass

            def set_step(self, step: int) -> None:
                pass

            def log_images(
                self,
                images: Any,
                title: str,
                step: int | None = None,
                **kwargs: Any,
            ) -> None:
                pass

            # IDeviceManager methods
            def get_device(self) -> torch.device:
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")

            def move_to_device(self, data: Any) -> Any:
                return data

            def is_cuda_available(self) -> bool:
                return False

            # IMetricsService methods
            def update_metrics(self, metrics: dict[str, Any]) -> None:
                pass

            def get_current_metrics(self) -> dict[str, Any]:
                return {}

            def reset_metrics(self) -> None:
                pass

            def log_metrics(self, step: int) -> None:
                pass

        service = MultiInterfaceService()

        # Should be instance of all interfaces
        assert isinstance(service, ILoggingService)
        assert isinstance(service, IDeviceManager)
        assert isinstance(service, IMetricsService)

    def test_interface_protocol_compliance(self):
        """Test that interfaces follow protocol design principles."""
        # All interfaces should be abstract
        interfaces = [
            ILoggingService,
            IDeviceManager,
            ICheckpointService,
            IMemoryOptService,
            IMetricsService,
            IDataLoader,
            IImageTransformer,
            IAugmentationStrategy,
            ICacheManager,
            IFileDiscoverer,
            IConfigurationService,
            ITrainingOrchestrator,
        ]

        for interface in interfaces:
            assert inspect.isabstract(interface)
            assert issubclass(interface, ABC)
            # Should have abstract methods
            assert len(interface.__abstractmethods__) > 0


if __name__ == "__main__":
    pytest.main([__file__])
