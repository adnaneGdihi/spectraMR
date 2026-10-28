from unittest.mock import patch

import pytest

from spectramr.infrastructure.domain_validation import (
    DomainMismatchWarning,
    DomainValidationResult,
    _loss_domain_of,
    _normalize_loss_name,
    get_domain_report,
    validate_loss_domains,
    verify_startup_loss_domains,
)


@pytest.mark.parametrize(
    "input_name,expected",
    [("mae", "l1"), ("MSE", "l2"), ("unknown_loss", "unknown_loss")],
)
def test_normalize_loss_name(input_name, expected):
    assert _normalize_loss_name(input_name) == expected


@pytest.mark.parametrize(
    "loss_name,expected_domain,expected_agnostic,expected_registered",
    [
        # `complex_l1` was `kspace` in the deleted hand table and is `agnostic`
        # on the decorator -- the single disagreement between the two owners,
        # and the decorator is the one that is right: an elementwise L1 is
        # defined on any tensor.
        ("complex_l1", None, True, True),
        ("mae", None, True, True),
        ("hfen", "image", False, True),
        ("non_existent_loss", None, False, False),
    ],
)
def test_loss_domain_of(loss_name, expected_domain, expected_agnostic, expected_registered):
    domain, agnostic, registered = _loss_domain_of(loss_name)
    assert domain == expected_domain
    assert agnostic is expected_agnostic
    assert registered is expected_registered


@patch("spectramr.infrastructure.domain_validation.LossBuilder")
def test_validate_loss_domains_valid(mock_builder):
    mock_instance = mock_builder.return_value
    mock_instance.get_enabled_losses.return_value = {"complex_l1": 1.0, "l1": 0.5}

    result = validate_loss_domains({}, "kspace")
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert len(result.warnings) == 0


@patch("spectramr.infrastructure.domain_validation.LossBuilder")
def test_validate_loss_domains_warning(mock_builder):
    mock_instance = mock_builder.return_value
    mock_instance.get_enabled_losses.return_value = {"hfen": 1.0}

    # kspace input with hfen loss is a critical error (image domain)
    result = validate_loss_domains({}, "kspace")
    assert result.is_valid is False
    assert len(result.errors) == 1
    assert len(result.warnings) == 1
    assert result.warnings[0].severity == "error"

    # latent input with hfen loss is a warning (image domain but not critical kspace mismatch)
    result = validate_loss_domains({}, "latent")
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert len(result.warnings) == 1
    assert result.warnings[0].severity == "warning"


@pytest.mark.parametrize(
    "loss_name,expected_label",
    [
        # Two different absences. Conflating them under one "unknown" label is
        # what let an un-audited loss read like a deliberately generic one, and
        # the old `fix` text pointed at a table that no longer exists.
        ("custom_unknown_loss", "unregistered"),
        ("bloch_residual", "unannotated"),
    ],
)
@patch("spectramr.infrastructure.domain_validation.LossBuilder")
def test_validate_loss_domains_reports_which_absence(mock_builder, loss_name, expected_label):
    mock_instance = mock_builder.return_value
    mock_instance.get_enabled_losses.return_value = {loss_name: 1.0}

    result = validate_loss_domains({}, "kspace")
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert len(result.warnings) == 1
    assert result.warnings[0].loss_domain == expected_label
    assert "LOSS_DOMAIN_REGISTRY" not in result.warnings[0].fix


def test_validate_loss_domains_invalid_input_domain():
    with pytest.raises(ValueError, match="Invalid input domain"):
        validate_loss_domains({}, "invalid_domain")


@patch("spectramr.infrastructure.domain_validation.LossBuilder")
def test_validate_loss_domains_builder_error(mock_builder):
    mock_builder.side_effect = Exception("Builder failed")
    result = validate_loss_domains({}, "kspace")
    assert result.is_valid is False
    assert len(result.errors) == 1
    assert "Builder failed" in result.errors[0]


@patch("spectramr.infrastructure.domain_validation.validate_loss_domains")
def test_verify_startup_loss_domains_success(mock_validate):
    mock_validate.return_value = DomainValidationResult(
        is_valid=True, errors=[], warnings=[]
    )
    assert verify_startup_loss_domains({}, "kspace") is True


@patch("spectramr.infrastructure.domain_validation.validate_loss_domains")
def test_verify_startup_loss_domains_failure(mock_validate):
    mock_validate.return_value = DomainValidationResult(
        is_valid=False,
        errors=["Critical error"],
        warnings=[DomainMismatchWarning("l", "d", "i", "warning", "msg", "fix")],
    )
    with pytest.raises(RuntimeError, match="Loss domain validation failed"):
        verify_startup_loss_domains({}, "kspace", fail_on_error=True)

    # Should not raise if fail_on_error is False
    assert verify_startup_loss_domains({}, "kspace", fail_on_error=False) is False


def test_registration_is_the_decorators_job():
    """`register_loss_domain` is gone. It mutated a module-level dict that was a
    second owner of the domain a loss declares, had zero callers in `src/` and
    `tests/`, and the registration path is `@register_loss(domain=...)`."""
    import spectramr.infrastructure.domain_validation as dv

    assert not hasattr(dv, "register_loss_domain")
    assert not hasattr(dv, "LOSS_DOMAIN_REGISTRY")


@patch("spectramr.infrastructure.domain_validation.validate_loss_domains")
def test_get_domain_report(mock_validate):
    mock_validate.return_value = DomainValidationResult(
        is_valid=False,
        errors=["Error 1"],
        warnings=[
            DomainMismatchWarning("loss1", "kspace", "image", "warning", "msg", "fix")
        ],
    )
    report = get_domain_report({}, "image")
    assert "Loss Domain Validation Report" in report
    assert "Input Domain: image" in report
    assert "Status: 🔴 INVALID" in report
    assert "CRITICAL ERRORS:" in report
    assert "Error 1" in report
    assert "WARNINGS:" in report
    assert "loss1: msg" in report


@patch("spectramr.infrastructure.domain_validation.LossBuilder")
def test_image_losses_block_is_bridged_not_critical(mock_builder):
    """F35 — hfen/ssim declared under losses.image_losses are auto-bridged to
    image domain by LossBuilder, so they are NOT a critical error on a k-space
    model (regression for experiment_12_physics_cold_diffusion_v2, which was
    rejected at pipeline build with 'Loss hfen requires domain image but model
    uses kspace' despite being correctly placed under image_losses).
    """
    from types import SimpleNamespace

    mock_builder.return_value.get_enabled_losses.return_value = {
        "hfen": 1.0,
        "ssim": 0.5,
    }
    config = SimpleNamespace(
        image_losses=[
            SimpleNamespace(name="hfen", enabled=True),
            SimpleNamespace(name="ssim", enabled=True),
        ]
    )
    result = validate_loss_domains(config, "kspace")
    assert result.is_valid is True
    assert result.errors == []


@patch("spectramr.infrastructure.domain_validation.LossBuilder")
def test_image_loss_outside_image_block_still_critical(mock_builder):
    """An image-domain loss NOT declared under image_losses (no bridge) on a
    k-space model is still a critical error — the F35 demotion is scoped to
    bridged losses only.
    """
    from types import SimpleNamespace

    mock_builder.return_value.get_enabled_losses.return_value = {"hfen": 1.0}
    config = SimpleNamespace(image_losses=[])  # hfen NOT bridged
    result = validate_loss_domains(config, "kspace")
    assert result.is_valid is False
