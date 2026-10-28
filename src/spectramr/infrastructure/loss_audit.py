"""Loss Audit Module - Comprehensive validation of loss configuration and creation.

Follows same pattern as ServiceAudit from Phase 4.
Validates:
1. Loss configuration is well-formed (YAML/Pydantic)
2. Loss builder can create all configured losses
3. Losses can accept required inputs (shapes, dtypes)
4. Losses produce outputs with correct shapes/types
5. Metrics are computed with correct types and keys
"""

import inspect
import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from spectramr.config.schemas.loss import LossConfigSchema
from spectramr.infrastructure.training.builders.loss_builder import LossBuilder

logger = logging.getLogger(__name__)


@dataclass
class LossRegistration:
    """Registration info for a single loss."""

    name: str
    enabled: bool
    weight: float
    has_metrics: bool
    error: str | None = None
    warning: str | None = None
    creation_time_ms: float = 0.0

    def is_healthy(self) -> bool:
        """Check if loss is healthy (no errors)."""
        return self.error is None


@dataclass
class LossAuditResult:
    """Result of auditing losses."""

    total_losses: int = 0
    enabled_losses: int = 0
    losses_with_metrics: int = 0
    all_healthy: bool = True

    losses: dict[str, LossRegistration] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Health indicators
    critical_errors: int = 0
    warnings_count: int = 0

    def __post_init__(self):
        """Calculate health after initialization."""
        self._calculate_health()

    def _calculate_health(self):
        """Recalculate health status from losses."""
        self.total_losses = len(self.losses)
        self.enabled_losses = sum(1 for l in self.losses.values() if l.enabled)
        self.losses_with_metrics = sum(1 for l in self.losses.values() if l.has_metrics)
        self.critical_errors = sum(1 for l in self.losses.values() if l.error)
        self.warnings_count = sum(1 for l in self.losses.values() if l.warning)
        # A top-level build failure (``audit_losses`` except branch) populates
        # ``self.errors`` but leaves ``self.losses`` empty, so ``critical_errors``
        # alone reports a vacuous "healthy". Fold ``self.errors`` into health so
        # the startup gate (``verify_startup_losses``) fails loudly instead of
        # passing a config whose losses did not construct at all (pitfall #10).
        self.all_healthy = self.critical_errors == 0 and not self.errors

    def get_status(self) -> str:
        """Get health status indicator.

        Returns:
            '🟢' if all healthy, '🟡' if warnings, '🔴' if errors
        """
        if self.critical_errors > 0 or self.errors:
            return "🔴"
        elif self.warnings_count > 0:
            return "🟡"
        else:
            return "🟢"

    def get_summary(self) -> str:
        """Get human-readable summary.

        Returns:
            Summary string with key metrics
        """
        status = self.get_status()
        return (
            f"{status} Loss Audit: {self.enabled_losses}/{self.total_losses} enabled, "
            f"{self.losses_with_metrics} with metrics, "
            f"{self.critical_errors} errors, {self.warnings_count} warnings"
        )

    def get_detailed_report(self) -> str:
        """Get detailed report for logging.

        Returns:
            Multi-line detailed report
        """
        lines = [self.get_summary(), ""]

        # Losses section
        if self.losses:
            lines.append("Loss Components:")
            for name, loss in sorted(self.losses.items()):
                status = "✅" if loss.is_healthy() else "❌"
                enabled_str = "enabled" if loss.enabled else "disabled"
                metrics_str = "(with metrics)" if loss.has_metrics else "(no metrics)"
                lines.append(
                    f"  {status} {name}: {enabled_str} (weight={loss.weight}) {metrics_str}"
                )
                if loss.error:
                    lines.append(f"      Error: {loss.error}")
                if loss.warning:
                    lines.append(f"      Warning: {loss.warning}")
            lines.append("")

        # Errors section
        if self.errors:
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  ❌ {error}")
            lines.append("")

        # Warnings section
        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")
            lines.append("")

        # Summary section
        lines.append("Summary:")
        lines.append(f"  Total losses: {self.total_losses}")
        lines.append(f"  Enabled: {self.enabled_losses}")
        lines.append(f"  With metrics: {self.losses_with_metrics}")
        lines.append(f"  Critical errors: {self.critical_errors}")
        lines.append(f"  Warnings: {self.warnings_count}")
        lines.append(f"  Health status: {self.get_status()}")

        return "\n".join(lines)


#: Prefix marking "we could not BUILD a probe for this loss", as distinct from
#: "we probed it and it passed". Never a pass — see :func:`_unsatisfiable_params`.
UNPROBED = "UNPROBED: "


def _is_tensor_annotation(annotation: object) -> bool:
    """Whether the probe's random tensor can legitimately fill this parameter.

    Unannotated is treated as tensor-compatible (benefit of the doubt — most of the older
    losses carry no annotations, and we want their *real* forward errors to surface rather
    than be excused as UNPROBED).
    """
    if annotation is inspect.Parameter.empty:
        return True
    if annotation is torch.Tensor:
        return True
    # ``from __future__ import annotations`` leaves these as strings.
    return "Tensor" in str(annotation)


def _unsatisfiable_params(loss: nn.Module) -> list[str]:
    """Forward parameters the synthetic ``(pred, target)`` probe cannot supply.

    Two ways a loss can be unprobeable, and both are facts of the signature — knowable
    *before* we call it:

    1. **A required parameter beyond the first two** — a sampling mask, a coil map:
       ``data_consistency(pred_image, measured_kspace, mask)``.
    2. **One of the first two slots is not a tensor at all** —
       ``r1_regularization(discriminator: Any, real_images, ...)``. Probing it feeds a random
       tensor into the discriminator slot, and the loss dies on ``discriminator(x)`` with
       "'Tensor' object is not callable". Assuming slot 0 is a prediction and slot 1 a target
       is wrong for ~40 of the 204 registered losses.

    This replaces a substring match on the raised exception text
    (``if "missing" in err and "argument" in err: return None``), which was **fail-open**:
    every loss with a required extra argument passed the audit *by failing it*, and a
    genuinely mis-wired loss that happened to raise "missing … argument" for an unrelated
    reason was reported HEALTHY too. Deciding from the signature means an unprobeable loss is
    reported UNPROBED (honest) and every *real* forward error now surfaces (loud).
    """
    try:
        sig = inspect.signature(loss.forward)
    except (TypeError, ValueError):  # pragma: no cover - C-implemented callable
        return []
    positional = [
        p
        for name, p in sig.parameters.items()
        if name != "self" and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]

    unsatisfiable = [p.name for p in positional[:2] if not _is_tensor_annotation(p.annotation)]
    unsatisfiable += [p.name for p in positional[2:] if p.default is p.empty]
    return unsatisfiable


def _test_loss_forward(loss: nn.Module, batch_size: int = 4) -> str | None:
    """Test if loss can perform forward pass.

    Args:
        loss: Loss module to test
        batch_size: Batch size for testing

    Returns:
        ``None`` when the probe ran and passed; a message prefixed with :data:`UNPROBED`
        when no probe could be built (not a pass — the caller records it as a warning);
        the error text when the probe ran and failed.
    """
    unsatisfiable = _unsatisfiable_params(loss)
    if unsatisfiable:
        return (
            f"{UNPROBED}forward() requires {unsatisfiable} beyond (pred, target); the "
            "synthetic probe cannot supply them. This loss is NOT verified — treat a green "
            "audit as 'did not crash', not 'mechanism exercised' (pitfall #16)."
        )

    loss_name = loss.__class__.__name__.lower()
    # "bridge" covers the iFFT→magnitude/complex wrappers (`_BridgedLoss`,
    # `FourierBridgeLossWrapper`): they reinterpret the channel axis as
    # (coil, real/imag) pairs, so they need the complex probe below — the
    # generic odd-channel real dummy (B,3,H,W) fails their reshape and logs a
    # spurious "shape ... is invalid for input of size" warning, even though
    # the loss is correctly wired (it computes a real, non-zero value on the
    # even multi-coil tensors it sees in training). Routing bridged losses to
    # the complex branch keeps this a clean wiring check (pitfall #10).
    is_complex = (
        "complex" in loss_name
        or "kspace" in loss_name
        or "fft" in loss_name
        or "bridge" in loss_name
    )
    is_evidential = "evidential" in loss_name

    # Probe in eval mode: the synthetic inputs carry NO runtime batch data
    # (no coil-sensitivity maps, masks, etc.). A loss that guards against
    # missing runtime inputs DURING TRAINING — e.g. SENSEAdjointL1Loss raises
    # on smaps=None to prevent the DC-blob silent-zero — must treat this
    # synthetic probe as a legitimate eval/no-coil zero, not a hard failure.
    # eval() keeps the audit a wiring check; the original mode is restored.
    _was_training = loss.training
    loss.eval()
    try:
        # Create dummy inputs
        if is_evidential:
            pred = torch.randn(batch_size, 4, 64, 64)
            # Evidential UNet maps an arbitrary number of channels to 4 parameter channels, target is typically 1 output domain (or standard 2 for complex)
            target = torch.randn(batch_size, 1, 64, 64)
        elif is_complex:
            # Use complex inputs for complex losses to avoid warnings/errors
            pred = torch.randn(batch_size, 2, 64, 64, dtype=torch.complex64)
            target = torch.randn(batch_size, 2, 64, 64, dtype=torch.complex64)
        else:
            pred = torch.randn(batch_size, 3, 64, 64)
            target = torch.randn(batch_size, 3, 64, 64)

        # Test forward
        output = loss(pred, target)

        # Handle tuple output (e.g. loss, metrics)
        if isinstance(output, tuple):
            output = output[0]  # Assume first element is loss

        # Verify output
        if not isinstance(output, torch.Tensor):
            return f"Loss output is {type(output)}, expected Tensor"

        if output.ndim != 0:
            return f"Loss output has shape {output.shape}, expected scalar"

        if not torch.isfinite(output):
            return "Loss output is not finite (NaN or Inf)"

        return None

    except Exception:
        # Retry with 2 channels (Common for Complex/K-Space) if not already tried
        try:
            # Fallback to standard complex shape (B, C, H, W) with complex dtype if not tried
            if not is_complex:
                pred = torch.randn(batch_size, 2, 64, 64, dtype=torch.complex64)
                target = torch.randn(batch_size, 2, 64, 64, dtype=torch.complex64)
            else:
                # If complex failed, try float (maybe it wasn't complex after all?)
                pred = torch.randn(batch_size, 2, 64, 64)
                target = torch.randn(batch_size, 2, 64, 64)

            output = loss(pred, target)
            if isinstance(output, tuple):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                return f"Loss output is {type(output)}, expected Tensor"
            if output.ndim != 0:
                return f"Loss output has shape {output.shape}, expected scalar"
            if not torch.isfinite(output):
                return "Loss output is not finite (NaN or Inf)"
            return None
        except Exception as e:
            # NO substring fail-open here. The "needs a discriminator / mask / coil map"
            # case is decided up-front from the SIGNATURE (:func:`_unsatisfiable_params`),
            # so anything that still raises at this point is a genuine forward failure and
            # must surface as an error. The old text match
            # (``"missing" in err and "argument" in err -> return None``) reported every
            # such loss as HEALTHY, which would have silently rubber-stamped exactly the
            # migration that shrinks STRATEGY_MANAGED_LOSSES.
            return str(e)
    finally:
        loss.train(_was_training)


def _test_metrics_computation(loss: nn.Module) -> tuple[bool, str | None]:
    """Test if loss can compute metrics.

    Args:
        loss: Loss module to test

    Returns:
        (has_metrics, error_message)
    """
    # Probe in eval mode (see _test_loss_forward): a training-only guard such
    # as SENSEAdjointL1Loss's raise-on-missing-smaps must not turn a synthetic
    # no-coil metrics probe into a spurious warning (strict smoke exits 2 on
    # warnings). Original mode restored in the finally below.
    _was_training = loss.training
    loss.eval()
    try:
        loss_name = loss.__class__.__name__.lower()
        is_evidential = "evidential" in loss_name
        is_complex = any(k in loss_name for k in ["complex", "fft", "kspace"])

        if not hasattr(loss, "forward_with_metrics"):
            # Check if forward returns tuple (implied metrics)
            # Create dummy inputs
            if is_evidential:
                pred = torch.randn(2, 4, 64, 64)
                target = torch.randn(2, 1, 64, 64)
            elif is_complex:
                pred = torch.randn(2, 2, 64, 64, dtype=torch.complex64)
                target = torch.randn(2, 2, 64, 64, dtype=torch.complex64)
            else:
                pred = torch.randn(2, 3, 64, 64)
                target = torch.randn(2, 3, 64, 64)

            output = loss(pred, target)
            if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], dict):
                # It returns metrics naturally
                return True, None

            return False, None

        # Create dummy inputs
        if is_evidential:
            pred = torch.randn(2, 4, 64, 64)
            target = torch.randn(2, 1, 64, 64)
        elif is_complex:
            pred = torch.randn(2, 2, 64, 64, dtype=torch.complex64)
            target = torch.randn(2, 2, 64, 64, dtype=torch.complex64)
        else:
            pred = torch.randn(2, 3, 64, 64)
            target = torch.randn(2, 3, 64, 64)

        # Test forward_with_metrics
        loss_val, metrics = loss.forward_with_metrics(pred, target)

        # Verify metrics
        if not isinstance(metrics, dict):
            return True, f"Metrics is {type(metrics)}, expected dict"

        # Check metric values
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                return True, f"Metric '{key}' is {type(value)}, expected numeric"

            if not torch.isfinite(torch.tensor(value)):
                return True, f"Metric '{key}' is not finite"

        return True, None

    except Exception as e:
        return False, str(e)
    finally:
        loss.train(_was_training)


def audit_losses(config: LossConfigSchema) -> LossAuditResult:
    """Audit loss configuration and creation.

    Args:
        config: LossConfigSchema instance

    Returns:
        LossAuditResult with detailed findings
    """
    import time

    result = LossAuditResult()

    # Create builder
    # We construct a minimal shim for LossBuilder because it expects TrainingSettings
    # configured with losses.
    class ConfigShim:
        """ConfigShim class."""

        def __init__(self, loss_config):
            """__init__.

            Args:
                loss_config (Any): Description.
            """
            self.losses = loss_config
            self.objectives = None  # Fallback logic in builder checks this
            self.training = None
            self.training_mode = "reconstruction"  # Default for audit
            self.deep_supervision_weight = 0.0

    try:
        shim = ConfigShim(config)
        # Type ignore: Shim mimics TrainingSettings structure needed by builder
        builder = LossBuilder(shim, device="cpu")  # type: ignore

        # Validate configuration first
        builder.validate_loss_configuration()

        start_build = time.time()
        # Build all losses using the builder
        built_losses = (
            builder.build_reconstruction_losses()
            .build_adversarial_losses()
            .build_physics_losses()
            .build_regularization_losses()
            .build_diffusion_losses()
            .build_latent_losses()
            .build_ssl_losses()
            .build()
        )
        build_time = (time.time() - start_build) * 1000

    except Exception as e:
        result.errors.append(f"Failed to build losses with LossBuilder: {e!s}")
        # Recompute health so the populated ``errors`` list flips ``all_healthy``
        # to False — the success path does this at the end, but an early return
        # here must too, else a total build failure reports vacuously "healthy".
        result._calculate_health()
        return result

    # Get enabled losses from config via builder introspection
    enabled = builder.get_enabled_losses()

    # Audit each enabled loss
    for loss_name, weight in enabled.items():
        # Check if builder created this loss
        loss_module = built_losses.get(loss_name)

        error = None
        # Losses that don't get a standalone audit entry. Three categories:
        #
        # (a) Strategy-managed — built by the training strategy at runtime
        #     (not by LossBuilder), because they need access to gradients,
        #     encoder features, or other state the LossBuilder can't see.
        # (b) Composite-bundled — built INSIDE another loss via lambda
        #     parameters. The composite shows up under its own key
        #     (e.g. ``adversarial``); the sub-loss appears in
        #     ``get_enabled_losses()`` but never in ``built_losses``.
        # (c) Context-requiring — built by LossBuilder, but ``forward()``
        #     needs a strategy-injected ``context`` dict that the synthetic
        #     forward test below cannot supply. Marking them strategy-
        #     managed acknowledges the loss exists and silences the
        #     spurious forward-test failure.
        #
        # F41 (2026-05-23 smoke audit) — the auditor's skip set MUST be a
        # superset of ``LossBuilder``'s skip set, else a config enabling a
        # strategy-managed loss the builder correctly skips (``sim`` /
        # ``smooth`` / ``marker`` / ``style`` / ``content`` / ``padnet_*`` /
        # ``physics_informed`` …) is reported as "LossBuilder did not create
        # it" and ``verify_startup_losses`` raises at startup. The two copies
        # drifted before (F6/E26); the builder set is now the single SSOT
        # (:data:`...loss_builder.STRATEGY_MANAGED_LOSSES`) imported here so
        # they cannot diverge again (re-derived 2026-06-11).
        #
        # Plus one auditor-only entry: ``bloch_consistency`` IS built by the
        # builder, but its ``forward`` needs a strategy-injected ``context``
        # dict the synthetic forward-test below cannot supply.
        from spectramr.infrastructure.training.builders.loss_builder import (
            STRATEGY_MANAGED_LOSSES,
        )

        strategy_managed_losses = STRATEGY_MANAGED_LOSSES | {"bloch_consistency"}
        if loss_name in strategy_managed_losses:
            registration = LossRegistration(
                name=loss_name,
                enabled=True,
                weight=weight,
                has_metrics=False,
                error=None,
                warning="Strategy-managed loss (computed by training strategy, not LossBuilder)",
                creation_time_ms=0.0,
            )
            result.losses[loss_name] = registration
            continue

        if loss_module is None:
            # Map canonical names (builder output) to config errors
            error = f"Loss '{loss_name}' passed enabled check but LossBuilder did not create it. Created: {list(built_losses.keys())}"

        registration = LossRegistration(
            name=loss_name,
            enabled=True,
            weight=weight,
            has_metrics=False,
            error=error,
            creation_time_ms=build_time if loss_module else 0.0,
        )

        # If creation succeeded, test forward pass
        if loss_module is not None:
            forward_error = _test_loss_forward(loss_module)
            if forward_error and forward_error.startswith(UNPROBED):
                # A loss we could not probe is NOT a loss that passed. Record it as a
                # visible, counted warning rather than folding it into the healthy set —
                # the old code returned None here and the audit went green.
                registration.warning = forward_error
            elif forward_error:
                registration.error = forward_error
            else:
                # If forward succeeded, test metrics
                has_metrics, metrics_error = _test_metrics_computation(loss_module)
                registration.has_metrics = has_metrics
                if metrics_error:
                    registration.warning = metrics_error

        result.losses[loss_name] = registration

    # Calculate final health
    result._calculate_health()

    return result


def verify_startup_losses(config: LossConfigSchema) -> bool:
    """Verify loss configuration on startup (fail-fast, registry-only).

    This is a *cheap* pre-flight check: it confirms every loss name in
    the declarative v6.0 ``kspace_losses`` / ``image_losses`` /
    ``complex_losses`` lists resolves to a registered loss, **without
    constructing any loss module**.

    Previously this ran the full ``LossBuilder(...).build()`` chain on
    CPU — instantiating e.g. ``perceptual`` → a VGG19 backbone — only to
    discard the result (the training director builds the real losses
    again later; see ``bootstrap.py`` PHASE 5.5). That made every
    ``train`` / ``audit`` cold start pay a redundant VGG load. The
    list-based names are exactly the ones the builder routes through
    :func:`create_loss`, so a registry-membership check is authoritative
    for them (a missing name would raise ``Unknown loss`` at build time).

    Loss names handled by the builder's composite / strategy paths (GAN
    ``adversarial``, ``r1``, VQ ``commitment``, ...) and the legacy
    flag-based path are validated authoritatively when the director
    constructs them — that build is still fail-loud, so a bad name there
    aborts before training, just slightly later than this pre-check.

    Args:
        config: LossConfigSchema instance.

    Returns:
        True if validation passed.

    Raises:
        RuntimeError: If a configured list-based loss name is not
            registered.
    """
    from spectramr.config.schemas.loss import LOSS_LIST_DOMAINS
    from spectramr.models.losses.registry import LossRegistry

    list_loss_names: list[str] = []
    for attr in LOSS_LIST_DOMAINS:
        for component in getattr(config, attr, None) or []:
            name = getattr(component, "name", None)
            if name:
                list_loss_names.append(name)

    unregistered = [n for n in list_loss_names if not LossRegistry.is_registered(n)]
    if unregistered:
        available = ", ".join(LossRegistry.list_available())
        error_details = "\n".join(f"  - {name}: not a registered loss" for name in unregistered)
        raise RuntimeError(
            f"Loss configuration validation failed:\n{error_details}\n\n"
            f"Registered losses: {available}\n"
            f"Please fix the configured loss name(s) and try again."
        )

    return True


def get_loss_metrics_capability(config: LossConfigSchema) -> dict[str, bool]:
    """Get metrics capability for each configured loss.

    Args:
        config: LossConfigSchema instance

    Returns:
        Dictionary mapping loss name to whether it has metrics
    """
    result = audit_losses(config)
    return {name: loss.has_metrics for name, loss in result.losses.items()}
