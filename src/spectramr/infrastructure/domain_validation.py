"""Loss Domain Validation - Phase 2 Runtime Domain Checking.

Validates that loss functions match the input domain at runtime (on bootstrap).
Prevents silent domain mismatches like HFEN (image-domain) receiving k-space data.

Integrates with loss_audit.py for comprehensive validation.
"""

import logging
from dataclasses import dataclass

from spectramr.config.schemas.loss import LossConfigSchema
from spectramr.infrastructure.training.builders.loss_builder import LossBuilder

logger = logging.getLogger(__name__)


# The registry's ``@register_loss(domain=...)`` annotation is the ONE owner of a
# loss's domain (non-negotiable 17). This module used to carry a second,
# hand-maintained table of 36 entries beside it. Characterized before removing,
# as the rule requires: of the 19 names both surfaces annotated, 18 agreed and
# ONE disagreed -- ``complex_l1``, hand-written as ``kspace`` against the
# decorator's ``agnostic``. The decorator is right; an elementwise L1 is defined
# on any tensor, and the hand entry would have flagged a legal declaration. The
# hand table also covered 36 of 220 registered losses, so 184 were invisible to
# it while reading as checked.
#
# ``get_loss_capabilities`` is the documented single retrieval surface, and it
# keeps the distinction this check needs: ``domain=None`` means UNANNOTATED
# (skip, nobody has said) while ``domain_agnostic=True`` is a positive claim
# that skipping is correct. Collapsing those is what let an un-audited loss and
# a deliberately generic one look alike.


def _loss_domain_of(loss_name: str) -> tuple[str | None, bool, bool]:
    """``(domain, is_agnostic, is_registered)`` for ``loss_name``.

    ``domain`` is ``None`` when the loss is registered but carries no
    ``domain=`` annotation -- a different state from "not registered at all",
    and the two get different messages below (non-negotiable 18: absent is a
    state to report, never one to infer).
    """
    from spectramr.models.losses.registry import LossRegistry, get_loss_capabilities

    canonical = _normalize_loss_name(loss_name)
    registered = canonical in LossRegistry.list_available() or canonical in getattr(
        LossRegistry, "_aliases", {}
    )
    caps = get_loss_capabilities(canonical)
    if caps is None:
        return None, False, registered
    domain = getattr(caps.domain, "value", caps.domain)
    return domain, bool(caps.domain_agnostic), registered


# Normalize aliases to canonical names
LOSS_ALIASES = {
    "mae": "l1",
    "mean_absolute_error": "l1",
    "mse": "l2",
    "mean_squared_error": "l2",
    "bce": "l1",
    "cross_entropy": "l1",
    "gan_standard": "l1",
    "gan_vanilla": "l1",
    "lsgan": "l1",
    "r1_regularization": "l1",
    "gan_bce": "l1",
    "contrastive": "l1",
    "domain_adversarial": "l1",
    "deep_supervision": "l1",
    "modality_swap": "l1",
    "uncertainty": "l1",
}


@dataclass
class DomainMismatchWarning:
    """Warning about potential domain mismatch."""

    loss_name: str
    loss_domain: str
    input_domain: str
    severity: str  # "error" or "warning"
    message: str
    fix: str  # Suggested fix


@dataclass
class DomainValidationResult:
    """Result of domain validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[DomainMismatchWarning]


def _normalize_loss_name(loss_name: str) -> str:
    """Normalize loss name to canonical form via aliases."""
    canonical = loss_name.lower()
    return LOSS_ALIASES.get(canonical, canonical)


def validate_loss_domains(config: LossConfigSchema, input_domain: str) -> DomainValidationResult:
    """Validate that all configured losses match input domain.

    Args:
        config: LossConfigSchema instance
        input_domain: Model input domain (kspace, image, or latent)

    Returns:
        DomainValidationResult with validation outcome

    Raises:
        ValueError: If input_domain is invalid
    """
    if input_domain not in ("kspace", "image", "latent"):
        raise ValueError(
            f"Invalid input domain: {input_domain}. Must be 'kspace', 'image', or 'latent'"
        )

    result = DomainValidationResult(is_valid=True, errors=[], warnings=[])

    # F35 / 2026-05-22 — losses declared under ``losses.image_losses`` are
    # evaluated in IMAGE domain via a bridge that LossBuilder auto-inserts
    # (the same contract the static audit's ``loss_domain_consistency`` check
    # already honours). For a k-space model these image-domain losses are
    # therefore EXPECTED and handled — not a domain error. Without this, a
    # correctly-bridged config (e.g. experiment_12_physics_cold_diffusion_v2
    # with hfen/ssim under image_losses) was rejected at pipeline build with
    # "Loss 'hfen' requires domain 'image' but model uses 'kspace'". Collect
    # the bridged set so the critical check below skips them.
    bridged_image_losses: set[str] = set()
    image_losses_block = getattr(config, "image_losses", None) or []
    for _entry in image_losses_block:
        _name = getattr(_entry, "name", None)
        if _name is None and isinstance(_entry, dict):
            _name = _entry.get("name")
        _enabled = getattr(_entry, "enabled", True)
        if isinstance(_entry, dict):
            _enabled = _entry.get("enabled", True)
        if _name and _enabled:
            bridged_image_losses.add(_normalize_loss_name(_name))

    # Create builder to get enabled losses
    class ConfigShim:
        """ConfigShim class."""

        def __init__(self, loss_config):
            """__init__.

            Args:
                loss_config (Any): Description.
            """
            self.losses = loss_config
            self.objectives = None
            self.training = None
            self.training_mode = "reconstruction"
            self.deep_supervision_weight = 0.0

    try:
        shim = ConfigShim(config)
        builder = LossBuilder(shim, device="cpu")  # type: ignore
        enabled_losses = builder.get_enabled_losses()
    except Exception as e:
        result.errors.append(f"Failed to get enabled losses: {e!s}")
        result.is_valid = False
        return result

    # Check each enabled loss against input domain
    for loss_name, weight in enabled_losses.items():
        loss_domain, is_agnostic, is_registered = _loss_domain_of(loss_name)

        if loss_domain is None:
            # Two different absences, and conflating them is what let an
            # un-audited loss read as a deliberately generic one. Neither is an
            # error here: an unregistered NAME is owned by the audit's
            # ``check_declared_losses_registered``, and raising in both places
            # would be two owners of one rule.
            if is_agnostic:
                continue
            result.warnings.append(
                DomainMismatchWarning(
                    loss_name=loss_name,
                    loss_domain="unannotated" if is_registered else "unregistered",
                    input_domain=input_domain,
                    severity="warning",
                    message=(
                        f"Loss '{loss_name}' carries no domain annotation - domain "
                        "validation skipped."
                        if is_registered
                        else f"Loss '{loss_name}' is not in the loss registry - domain "
                        "validation skipped (custom loss?)."
                    ),
                    fix=(
                        f"Add domain=... to @register_loss for '{loss_name}', or "
                        "domain_agnostic=True if it genuinely imposes no constraint."
                        if is_registered
                        else f"Register '{loss_name}' with @register_loss."
                    ),
                )
            )
            continue

        # Agnostic is a positive claim of compatibility; otherwise the loss's
        # own domain must be the one the model feeds it.
        is_compatible = is_agnostic or loss_domain == input_domain

        if not is_compatible:
            error_msg = (
                f"Loss '{loss_name}' requires domain '{loss_domain}' "
                f"but model uses '{input_domain}'"
            )

            # F35 — image-domain losses declared under ``image_losses`` are
            # bridged to image domain by LossBuilder, so a k-space model is
            # fine. Treat as compatible (skip silently) rather than a critical
            # build-blocking error.
            if loss_domain == "image" and _normalize_loss_name(loss_name) in bridged_image_losses:
                continue

            is_critical = loss_domain == "image" and input_domain == "kspace"

            if is_critical:
                severity = "error"
                fix = (
                    f"Set enable_{loss_name}: false or lambda_{loss_name}: 0.0 "
                    f"in your experiment YAML, "
                    f"or use spatial_losses_use_fourier_bridge: true "
                    f"to auto-bridge image losses to k-space"
                )
                logger.error("Domain mismatch (critical): %s — %s", loss_name, error_msg)
                result.errors.append(error_msg)
                result.is_valid = False
            else:
                severity = "warning"
                fix = f"Verify compatibility or set lambda_{loss_name}=0.0 if unsure"

            result.warnings.append(
                DomainMismatchWarning(
                    loss_name=loss_name,
                    loss_domain=loss_domain,
                    input_domain=input_domain,
                    severity=severity,
                    message=error_msg,
                    fix=fix,
                )
            )

    return result


def verify_startup_loss_domains(
    config: LossConfigSchema, input_domain: str, fail_on_error: bool = True
) -> bool:
    """Verify loss domains match input domain on startup (fail-fast).

    Integrates with bootstrap.py validation pipeline.

    Args:
        config: LossConfigSchema instance
        input_domain: Model input domain (kspace, image, or latent)
        fail_on_error: If True, raise RuntimeError on domain errors

    Returns:
        True if validation passed

    Raises:
        RuntimeError: If domain errors found and fail_on_error=True
    """
    result = validate_loss_domains(config, input_domain)

    # Print summary
    status = "🟢" if result.is_valid else "🔴"
    logger.debug(
        f"\n{status} Loss Domain Validation: {input_domain} domain | "
        f"{len(result.errors)} errors, {len(result.warnings)} warnings"
    )

    # Print errors
    if result.errors:
        logger.debug("\n❌ Domain Mismatches (ERRORS):")
        for error in result.errors:
            logger.debug(f"  - {error}")

    # Print warnings
    if result.warnings:
        logger.debug("\n⚠️  Domain Compatibility Warnings:")
        for warning in result.warnings:
            logger.debug(f"  - [{warning.severity.upper()}] {warning.message}")
            logger.debug(f"    Fix: {warning.fix}")

    # Fail fast if errors
    if result.errors and fail_on_error:
        raise RuntimeError(
            f"Loss domain validation failed for '{input_domain}' domain. "
            f"Fix the {len(result.errors)} critical error(s) above and retry."
        )

    return result.is_valid


def get_domain_report(config: LossConfigSchema, input_domain: str) -> str:
    """Generate a detailed domain validation report.

    Args:
        config: LossConfigSchema instance
        input_domain: Model input domain

    Returns:
        Detailed report as multi-line string
    """
    result = validate_loss_domains(config, input_domain)

    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("Loss Domain Validation Report")
    lines.append("=" * 70)
    lines.append(f"Input Domain: {input_domain}")
    lines.append(f"Status: {'🟢 VALID' if result.is_valid else '🔴 INVALID'}")
    lines.append(f"Errors: {len(result.errors)} | Warnings: {len(result.warnings)}")
    lines.append("")

    if result.errors:
        lines.append("CRITICAL ERRORS:")
        for i, error in enumerate(result.errors, 1):
            lines.append(f"  {i}. ❌ {error}")
        lines.append("")

    if result.warnings:
        lines.append("WARNINGS:")
        for warning in result.warnings:
            lines.append(
                f"  ⚠️  [{warning.severity.upper()}] {warning.loss_name}: {warning.message}"
            )
            lines.append(f"      Fix: {warning.fix}")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)
