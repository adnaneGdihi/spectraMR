from typing import Any

import pytest

from spectramr.domain.exceptions import ConfigurationError
from spectramr.domain.interfaces.service_interfaces import (
    ICheckpointService,
    IDeviceService,
    ILoggingService,
)
from spectramr.infrastructure.di.di_container import DIContainer
from spectramr.infrastructure.di.service_audit import audit_services, verify_startup_services
from spectramr.infrastructure.services import IMetricsService


class MockDeviceService(IDeviceService):
    def get_device(self) -> Any:
        return "cpu"

    def is_cuda_available(self) -> bool:
        return False

    def move_to_device(self, obj: Any) -> Any:
        return obj


class MockLoggingService(ILoggingService):
    def log_info(self, msg: str, level: str = "info"):
        pass

    def log_error(self, msg: str):
        pass

    def log_critical(self, msg: str):
        pass

    def log_warning(self, msg: str):
        pass

    def setup(self):
        pass

    def log_debug(self, msg: str):
        pass

    def get_logger(self) -> Any:
        return None

    def step(self):
        pass

    def set_step(self, step: int):
        pass

    def epoch(self):
        pass

    def set_epoch(self, epoch: int):
        pass

    def log_images(self, name: str, images: Any, step: int):
        pass

    def log_images_batch(self, name: str, images: Any, step: int):
        pass


class MockCheckpointService(ICheckpointService):
    def save_checkpoint(self, path: str, state_dict: dict):
        pass

    def load_checkpoint(self, path: str) -> dict:
        return {}

    def get_latest_checkpoint(self) -> str:
        return ""

    def find_latest_checkpoint(self) -> str:
        return ""


class MockMetricsService(IMetricsService):
    def log_metric(self, name: str, value: Any, step: int):
        pass

    def log_metrics(self, metrics: dict, step: int):
        pass

    def get_current_metrics(self) -> dict:
        return {}

    def reset_metrics(self):
        pass

    def update_metrics(self, metrics: dict):
        pass


def setup_healthy_container() -> DIContainer:
    """Setup a healthy container with all required services."""
    container = DIContainer()
    container.register(IDeviceService, implementation=MockDeviceService())
    container.register(ILoggingService, implementation=MockLoggingService())
    container.register(ICheckpointService, implementation=MockCheckpointService())
    return container


def test_audit_services_healthy():
    """Test auditing a healthy container."""
    container = setup_healthy_container()

    # Optional service
    container.register(IMetricsService, implementation=MockMetricsService())

    result = audit_services(container)

    assert result.is_healthy
    assert result.missing_services == 0
    assert result.health_status == "🟢 HEALTHY"
    assert "DI Container Service Audit Report" in result.summary()


def test_audit_services_missing_required():
    """Test auditing when a required service is missing."""
    container = DIContainer()
    container.register(IDeviceService, implementation=MockDeviceService())
    # Missing Logging and Checkpoint service

    result = audit_services(container)

    assert not result.is_healthy
    assert result.missing_services == 2
    assert "🟡 DEGRADED (2 missing)" in result.health_status
    assert "IDeviceService" in result.summary()

    # Check errors recorded
    assert any("Required service ILoggingService" in err for err in result.errors)
    assert any("Required service ICheckpointService" in err for err in result.errors)


def test_audit_services_missing_optional():
    """Test auditing when an optional service is missing but required are present.

    Optional missing services do NOT produce an entry in ``result.errors``
    — that list is reserved for required-service failures so callers can
    treat its emptiness as a startup-safety signal. A missing optional
    service is recorded in ``result.registrations`` with
    ``registered=False`` instead.
    """
    container = setup_healthy_container()

    result = audit_services(container)

    # Still healthy even if optional services (e.g., IMetricsService) are missing
    assert result.is_healthy
    assert result.missing_services == 0
    # Required-services error list stays clean for missing-optional case.
    assert not any("IMetricsService" in err for err in result.errors)
    # But the registration record IS captured for visibility.
    assert any(
        reg.name == "IMetricsService" and not reg.registered
        for reg in result.registrations
    )


def test_audit_services_none_container():
    """Test behavior with a None container."""
    with pytest.raises(ValueError, match="Container cannot be None"):
        audit_services(None)


def test_verify_startup_services_healthy():
    """Test verify_startup_services on a healthy container."""
    container = setup_healthy_container()

    # Should not raise
    assert verify_startup_services(container, fail_on_missing=True)


def test_verify_startup_services_missing_required():
    """Test verify_startup_services raises on missing required."""
    container = DIContainer()
    # Missing all

    with pytest.raises(ConfigurationError, match="Service audit failed"):
        verify_startup_services(container, fail_on_missing=True)


def test_verify_startup_services_no_fail():
    """Test verify_startup_services returns False instead of raising if fail_on_missing=False."""
    container = DIContainer()

    is_healthy = verify_startup_services(container, fail_on_missing=False)
    assert not is_healthy
