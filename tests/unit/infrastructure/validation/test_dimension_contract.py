"""Tests for the dimension-contract Layer-1(a) report check + coverage helper.

Covers ``ConfigHealthChecker.check_model_contract_declared`` (the report-only
audit check that surfaces undeclared ModelCapabilities) and the pure
``model_contract_status`` helper in ``scripts/audit/contract_coverage.py``.

The check tests find declared / undeclared models *dynamically* from the live
registry so they stay green as the high-traffic back-fill lands.
"""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import pytest

from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker
from tests.utils.config_block_stub import block_stub
from tests.utils.repo_scripts import require_repo_file


def _cfg(model_type):
    return SimpleNamespace(model=SimpleNamespace(model_type=model_type, model_kwargs={}))


@pytest.fixture(scope="module")
def registry():
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY, get_model_capabilities

    populate_model_registry()
    return MODEL_REGISTRY, get_model_capabilities


def test_check_reports_an_undeclared_model(registry):
    model_registry, get_caps = registry
    undeclared = next((n for n in model_registry if get_caps(n) is None), None)
    assert undeclared is not None, "expected at least one unannotated model"
    r = ConfigHealthChecker().check_model_contract_declared(_cfg(undeclared))
    # Report-only: never gates (passed=True, info), but surfaces the gap.
    assert r.passed is True
    assert r.severity == "info"
    assert r.category == "dimension_contract"
    assert "undeclared" in r.message


def test_check_passes_a_fully_declared_model(registry):
    model_registry, get_caps = registry

    def fully_declared(name):
        c = get_caps(name)
        return (
            c is not None
            and c.spatial_dims is not None
            and (c.input_domain is not None or c.output_domain is not None)
        )

    declared = next((n for n in model_registry if fully_declared(n)), None)
    if declared is None:
        pytest.skip("no fully-declared model in registry")
    r = ConfigHealthChecker().check_model_contract_declared(_cfg(declared))
    assert r.passed is True
    assert "declares its dimension contract" in r.message


def test_check_skips_unknown_model():
    r = ConfigHealthChecker().check_model_contract_declared(_cfg("definitely_not_a_real_model_xyz"))
    assert r.passed is True
    assert "not in registry" in r.message


def test_check_handles_missing_model_section():
    r = ConfigHealthChecker().check_model_contract_declared(SimpleNamespace())
    assert r.passed is True
    assert r.severity == "info"


# --------------------------------------------------------------------------
# contract_coverage.model_contract_status (pure helper)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cc():
    path = require_repo_file("scripts/audit/contract_coverage.py")
    spec = importlib.util.spec_from_file_location("contract_coverage", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_model_contract_status_all_undeclared(cc):
    from spectramr.models.capabilities import ModelCapabilities

    status = cc.model_contract_status(ModelCapabilities())
    assert status == {
        "spatial_dims": False,
        "input_domain": False,
        "output_domain": False,
    }


def test_model_contract_status_partial(cc):
    from spectramr.models.capabilities import ModelCapabilities

    caps = ModelCapabilities(spatial_dims=(2,), input_domain="kspace")
    status = cc.model_contract_status(caps)
    assert status["spatial_dims"] is True
    assert status["input_domain"] is True
    assert status["output_domain"] is False


# --------------------------------------------------------------------------
# Layer 1(b): _data_is_dangerous + check_dangerous_data_requires_contract
# --------------------------------------------------------------------------


def _full_cfg(model_type, patch_size=(64, 64), dataset_type="m4raw", coil_mode=None):
    return SimpleNamespace(
        model=SimpleNamespace(model_type=model_type, model_kwargs={}),
        data=SimpleNamespace(
            patch_size=patch_size,
            dataset_type=dataset_type,
            coil_processing_mode=coil_mode,
        ),
    )


def test_data_is_dangerous_3d():
    assert ConfigHealthChecker._data_is_dangerous(_full_cfg("x", patch_size=(32, 32, 32))) == "3D"


def test_data_is_dangerous_kspace():
    cfg = _full_cfg("x", dataset_type="m4raw_kspace")
    assert ConfigHealthChecker._data_is_dangerous(cfg) == "complex/k-space"


def test_data_not_dangerous_when_coil_combined_to_image():
    # rss_image collapses k-space to a 2D magnitude image → not dangerous.
    cfg = _full_cfg("x", dataset_type="m4raw_kspace", coil_mode="rss_image")
    assert ConfigHealthChecker._data_is_dangerous(cfg) is None


def test_dangerous_data_undeclared_model_flagged(registry):
    model_registry, get_caps = registry
    undeclared = next(n for n in model_registry if get_caps(n) is None)
    cfg = _full_cfg(undeclared, patch_size=(32, 32, 32))  # 3D
    r = ConfigHealthChecker().check_dangerous_data_requires_contract(cfg)
    assert r.severity == "info"  # report-only during rollout
    assert "no dimension contract" in r.message
    assert "3D" in r.message


def test_dangerous_data_declared_model_passes():
    # complex_unet declares spatial_dims → out of the default-deny bucket.
    cfg = _full_cfg("complex_unet", dataset_type="m4raw_kspace")
    r = ConfigHealthChecker().check_dangerous_data_requires_contract(cfg)
    assert "declares a dimension contract" in r.message


def test_dangerous_data_trivial_allows_undeclared(registry):
    model_registry, get_caps = registry
    undeclared = next(n for n in model_registry if get_caps(n) is None)
    # 2D + a genuinely image-domain dataset (not a k-space type like m4raw).
    cfg = _full_cfg(undeclared, patch_size=(64, 64), dataset_type="generic_image")
    r = ConfigHealthChecker().check_dangerous_data_requires_contract(cfg)
    assert "trivial" in r.message.lower()


# --------------------------------------------------------------------------
# Layer 1(c): check_metric_domain_matches_loss_output
# --------------------------------------------------------------------------


def _metric_cfg(loss_out, metric_domain, transform=None):
    return SimpleNamespace(
        losses=block_stub("losses", output_domain=loss_out),
        metrics=SimpleNamespace(domain=metric_domain, transform=transform),
    )


def test_metric_domain_bridged_by_transform():
    # kspace loss + image metrics is FINE when a transform bridges them.
    cfg = _metric_cfg("kspace", "image", transform="ifft_sense_adjoint")
    r = ConfigHealthChecker().check_metric_domain_matches_loss_output(cfg)
    assert r.passed is True
    assert "bridged" in r.message or "consistent" in r.message


def test_metric_domain_match_passes():
    cfg = _metric_cfg("image", "image")
    r = ConfigHealthChecker().check_metric_domain_matches_loss_output(cfg)
    assert "matches" in r.message


def test_metric_domain_unbridged_mismatch_flagged():
    cfg = _metric_cfg("kspace", "image", transform=None)
    r = ConfigHealthChecker().check_metric_domain_matches_loss_output(cfg)
    assert "differs" in r.message
    assert r.category == "dimension_contract"
