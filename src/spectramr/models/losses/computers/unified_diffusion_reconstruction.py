"""Unified Diffusion Loss Computer - SSOT Implementation.

Handles diffusion-specific loss computation including:
- Diffusion process losses (score matching, noise prediction)
- Optional reconstruction loss
- Optional adversarial loss
- Weighting and scheduling
"""

import inspect
import logging
from collections.abc import Mapping
from typing import Any

import torch

from spectramr.models.losses.computers.base import BaseLossComputer, LossOutput
from spectramr.models.losses.weights import is_loss_configured

logger = logging.getLogger(__name__)


# Standard keys produced by diffusion / cold-diffusion / VAE generators
# whose forward() returns a dict instead of a single tensor. Probed in
# this order: the first key whose value is a tensor wins.
_PRED_DICT_KEYS = (
    "pred",
    "prediction",
    "predictions",
    "image",
    "image_out",
    "output",
    "out",
    "x",
    "x0",
    "x_0",
    "x_pred",
    "x_hat",
    "x_recon",
    "sample",
    "samples",
    "reconstruction",
    "denoised",
)


def _unwrap_tensor_arg(name: str, value: Any) -> torch.Tensor:
    """Coerce a model output / target into a Tensor.

    Some generators (latent diffusion, cold diffusion, multi-head VAEs)
    return a ``dict`` of outputs (e.g. ``{"pred": ..., "aux": ...}``).
    The loss layer expects a single tensor — we unwrap the dict by
    picking the first known-prediction key, falling back to the first
    tensor-valued entry. Tuples / lists are unwrapped to their first
    element. Anything else is returned unchanged so the downstream
    ``isinstance`` check still flags it.

    Args:
        name: Identifier used in error messages (e.g. ``"pred"`` /
            ``"target"``).
        value: Candidate tensor, dict, tuple, list, or other.

    Returns:
        A ``torch.Tensor`` if one can be located; otherwise the
        original ``value``.
    """
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for key in _PRED_DICT_KEYS:
            if key in value and isinstance(value[key], torch.Tensor):
                return value[key]
        for v in value.values():
            if isinstance(v, torch.Tensor):
                return v
        raise TypeError(
            f"[loss] {name} arrived as dict with no tensor entry "
            f"(keys={list(value.keys())[:8]}). Update the strategy to "
            "unwrap the model output before calling the loss computer "
            "— see ``_unwrap_tensor_arg`` for the probed keys."
        )
    if isinstance(value, (tuple, list)) and value:
        return _unwrap_tensor_arg(name, value[0])
    return value


def _call_safe_loss(
    loss_fn: Any, pred: torch.Tensor, target: torch.Tensor, **kwargs
) -> torch.Tensor:
    """Invokes a loss function safely by filtering available kwargs according to its signature."""
    pred = _unwrap_tensor_arg("pred", pred)
    target = _unwrap_tensor_arg("target", target)

    if hasattr(loss_fn, "forward"):
        sig = inspect.signature(loss_fn.forward)
    else:
        sig = inspect.signature(loss_fn)

    valid_kwargs = {}
    has_varkw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

    for k, v in kwargs.items():
        if k in sig.parameters or has_varkw:
            valid_kwargs[k] = v

    return loss_fn(pred, target, **valid_kwargs)


def _get_config_value(obj: Any, path: str, default: Any = None) -> Any:
    """Helper to get nested config value."""
    try:
        parts = path.split(".")
        current = obj
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current
    except Exception:
        return default


def _complex_safe_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss safely handling complex tensors.

    Handles:
    - One or both tensors being complex
    - Shape mismatches by reshaping
    - Complex/real conversions
    - Dict / tuple model outputs (via ``_unwrap_tensor_arg``)
    """
    pred = _unwrap_tensor_arg("pred", pred)
    target = _unwrap_tensor_arg("target", target)
    pred_is_complex = torch.is_complex(pred)
    target_is_complex = torch.is_complex(target)

    # If one is complex and other is real, need special handling
    if pred_is_complex != target_is_complex:
        logger.debug(
            f"[MSE Loss] Complex/Real mismatch: pred_complex={pred_is_complex}, "
            f"target_complex={target_is_complex}. Converting both to magnitude."
        )
        if pred_is_complex:
            pred = pred.abs()
        if target_is_complex:
            target = target.abs()
    else:
        # Both same type - use view_as_real if complex
        if pred_is_complex:
            pred = torch.view_as_real(pred)
        if target_is_complex:
            target = torch.view_as_real(target)

    # Final shape check
    if pred.shape != target.shape:
        logger.warning(
            f"[MSE Loss] Shape mismatch after conversion: pred={pred.shape}, target={target.shape}. "
            f"Elements: pred={pred.numel()}, target={target.numel()}"
        )

        # Try to reshape target to match pred if they have same number of elements
        if pred.numel() == target.numel() and pred.shape != target.shape:
            logger.warning(f"[MSE Loss] Reshaping target from {target.shape} to {pred.shape}")
            target = target.reshape(pred.shape)
        elif (
            pred.ndim >= 4
            and target.ndim >= 4
            and pred.shape[0] == target.shape[0]
            and pred.shape[1] == target.shape[1]
            and pred.shape[2:] != target.shape[2:]
        ):
            # Spatial dimension mismatch (e.g., different H×W from n_downsample
            # asymmetry).  Interpolate pred to match target spatial dims.
            logger.warning(
                f"[MSE Loss] Spatial interpolation: pred {pred.shape} → target spatial {target.shape[2:]}"
            )
            pred = torch.nn.functional.interpolate(
                pred.float(),
                size=target.shape[2:],
                mode="bilinear" if pred.ndim == 4 else "trilinear",
                align_corners=False,
            )
        elif pred.numel() != target.numel():
            raise RuntimeError(
                f"[MSE Loss] Cannot reconcile shapes: pred={pred.shape} ({pred.numel()} elements) vs "
                f"target={target.shape} ({target.numel()} elements)"
            )

    loss = torch.nn.functional.mse_loss(pred, target)
    if torch.isnan(loss) or torch.isinf(loss):
        logger.warning(
            f"[MSE Loss] NaN/Inf detected! pred NaN: {pred.isnan().any()}, target NaN: {target.isnan().any()}"
        )
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
    return loss


def _complex_safe_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute L1 loss safely handling complex tensors.

    Handles:
    - One or both tensors being complex
    - Shape mismatches by reshaping
    - Complex/real conversions
    - Dict / tuple model outputs (via ``_unwrap_tensor_arg``)
    """
    pred = _unwrap_tensor_arg("pred", pred)
    target = _unwrap_tensor_arg("target", target)
    pred_is_complex = torch.is_complex(pred)
    target_is_complex = torch.is_complex(target)

    # If one is complex and other is real, need special handling
    if pred_is_complex != target_is_complex:
        logger.debug(
            f"[L1 Loss] Complex/Real mismatch: pred_complex={pred_is_complex}, "
            f"target_complex={target_is_complex}. Converting both to magnitude."
        )
        if pred_is_complex:
            pred = pred.abs()
        if target_is_complex:
            target = target.abs()
    else:
        # Both same type - use view_as_real if complex
        if pred_is_complex:
            pred = torch.view_as_real(pred)
        if target_is_complex:
            target = torch.view_as_real(target)

    # Final shape check
    if pred.shape != target.shape:
        logger.warning(
            f"[L1 Loss] Shape mismatch after conversion: pred={pred.shape}, target={target.shape}. "
            f"Elements: pred={pred.numel()}, target={target.numel()}"
        )

        # Try to reshape target to match pred if they have same number of elements
        if pred.numel() == target.numel() and pred.shape != target.shape:
            logger.warning(f"[L1 Loss] Reshaping target from {target.shape} to {pred.shape}")
            target = target.reshape(pred.shape)
        elif (
            pred.ndim >= 4
            and target.ndim >= 4
            and pred.shape[0] == target.shape[0]
            and pred.shape[1] == target.shape[1]
            and pred.shape[2:] != target.shape[2:]
        ):
            # Spatial dimension mismatch — interpolate pred to target spatial dims
            logger.warning(
                f"[L1 Loss] Spatial interpolation: pred {pred.shape} → target spatial {target.shape[2:]}"
            )
            pred = torch.nn.functional.interpolate(
                pred.float(),
                size=target.shape[2:],
                mode="bilinear" if pred.ndim == 4 else "trilinear",
                align_corners=False,
            )
        elif pred.numel() != target.numel():
            raise RuntimeError(
                f"[L1 Loss] Cannot reconcile shapes: pred={pred.shape} ({pred.numel()} elements) vs "
                f"target={target.shape} ({target.numel()} elements)"
            )

    loss = torch.nn.functional.l1_loss(pred, target)
    if torch.isnan(loss) or torch.isinf(loss):
        logger.warning(
            f"[L1 Loss] NaN/Inf detected! pred NaN: {pred.isnan().any()}, target NaN: {target.isnan().any()}"
        )
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
    return loss


class UnifiedDiffusionLossComputer(BaseLossComputer):
    """Unified diffusion loss computer implementing SSOT pattern.

    Supports multiple diffusion variants:
    - Score matching (predict score)
    - Noise prediction (predict noise)
    - Velocity prediction (predict velocity)
    """

    def __init__(self, config: Any, device: torch.device = torch.device("cpu")):
        """Initialize diffusion loss computer.

        Args:
            config: Configuration with diffusion settings
            device: Compute device
        """
        self.diffusion_loss_fn = None
        self.reconstruction_loss_fn = None
        self.adversarial_loss_fn = None
        self.discriminator = None
        # Registry name of the loss `_initialize_losses` auto-wired into the
        # `reconstruction` slot (None when it was injected externally). Lets
        # `compute()` skip the slot when the SAME term arrives via losses_dict
        # — otherwise complex_l1 is computed twice (once as `reconstruction`
        # at weight 1.0, once as `complex_l1` at its configured lambda).
        self._recon_fallback_name: str | None = None

        super().__init__(config, device)

    def _initialize_losses(self) -> None:
        """Initialize loss functions from config."""
        # Fallback initialization if losses weren't injected
        try:
            # Check for complex L1 (reconstruction)
            # A CONFIGURATION question, asked at construction time where there is
            # no iteration -- so it must not consult the warm-up gate. See
            # ``is_loss_configured``: the weight form returned 0.0 for the
            # warm-up-gated ``l1`` below and the loss was never built at all.
            if is_loss_configured(self._weight_table, "complex_l1") and (
                self.reconstruction_loss_fn is None
            ):
                from spectramr.models.losses.registry import create_loss

                self.reconstruction_loss_fn = create_loss("complex_l1").to(self.device)
                self._recon_fallback_name = "complex_l1"

            # Check for L1 (reconstruction) if complex not used
            if self.reconstruction_loss_fn is None:
                if is_loss_configured(self._weight_table, "l1"):
                    from spectramr.models.losses.registry import create_loss

                    self.reconstruction_loss_fn = create_loss("l1").to(self.device)
                    self._recon_fallback_name = "l1"

            # Check for diffusion specific loss
            if self.diffusion_loss_fn is None:
                # Standard MSE is default, but we can check if a specific one is requested
                pass

            # Adversarial term for an adversarial-diffusion arm.
            #
            # ``adversarial_loss_fn`` used to be set to ``None`` here and
            # assigned NOWHERE, while ``compute()`` guarded its adversarial
            # branch on ``and self.adversarial_loss_fn`` -- so the branch could
            # never be entered, and its body was a bare ``pass`` regardless. An
            # arm could declare a non-zero adversarial weight AND attach a
            # discriminator, satisfy every visible precondition, and receive no
            # adversarial gradient (#1669). Built through ``LossBuilder``, the
            # same owner ``UnifiedGANLossComputer`` uses, so the objective is
            # identical whatever the generator is doing (non-negotiable 17).
            #
            # Only built when the arm actually asks for it: a plain diffusion
            # arm constructs nothing new.
            if self.adversarial_loss_fn is None and is_loss_configured(
                self._weight_table, "adversarial"
            ):
                from spectramr.infrastructure.training.builders.loss_builder import LossBuilder

                built = LossBuilder(self.config, self.device).build_adversarial_losses().build()
                self.adversarial_loss_fn = built.get("adversarial")

        except Exception as _init_exc:
            # F-LOSSINIT-VISIBILITY / 2026-05-20 — the previous ``pass``
            # claimed to "log but don't crash" but never actually logged.
            # A silent init failure here leaves the computer running on
            # the ``compute()`` fallback path without the user knowing
            # WHY their declared loss-builder didn't wire (e.g., missing
            # custom-loss registration, malformed weight schedule). Emit
            # at warning so the failure is at least diagnosable; keep
            # the no-crash behaviour because ``compute()`` does have a
            # functional fallback.
            logger.warning(
                "[%s] loss-builder init partially failed (%s: %s); "
                "compute() will use the registry/default fallback. "
                "If results look wrong, fix the underlying init error.",
                type(self).__name__,
                type(_init_exc).__name__,
                _init_exc,
            )

    def _resolve_diffusion_weight(self) -> float:
        """Resolve the main diffusion-loss weight through its owning knobs.

        The base ``_get_loss_weight("diffusion")`` only knows
        ``losses.<section>.lambda_diffusion`` and falls through to a 1.0
        default — which silently ran a full-k-space MSE at weight 1.0 on
        every cold-diffusion arm even when the YAML declared
        ``training.diffusion.lambda_mse: 0.0`` (an advertised-but-unread
        knob, pitfall #15). Precedence:

        1. a scheduled override (LossScheduleController seam),
        2. explicit ``lambda_diffusion`` in a losses section,
        3. ``losses.diffusion.lambda_mse`` (DiffusionLossesConfig),
        4. ``training.diffusion.lambda_mse`` (the training-SSOT knob),
        5. the historical 1.0 default.

        Both schema ``lambda_mse`` fields default to 1.0, so arms that never
        set the knob keep the exact legacy behaviour.
        """
        override = self._scheduled_override("diffusion")
        if override is not None:
            return override
        for path in (
            "losses.reconstruction.lambda_diffusion",
            "losses.diffusion.lambda_diffusion",
            "losses.diffusion.lambda_mse",
            "training.diffusion.lambda_mse",
        ):
            val = _get_config_value(self.config, path, None)
            if val is not None:
                return float(val)
        return 1.0

    def _get_loss_weight(
        self, loss_name: str, epoch: int = 0, iteration: int = 0, **kwargs: Any
    ) -> float:
        """Route the ``diffusion`` term through its dedicated resolver.

        The gate in ``compute()`` and the weighting in ``_stack_components``
        both call this method, so the ``lambda_mse`` wiring applies to each
        exactly once and they can never disagree.
        """
        if loss_name == "diffusion":
            return self._resolve_diffusion_weight()
        return super()._get_loss_weight(loss_name, epoch=epoch, iteration=iteration, **kwargs)

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        epoch: int = 0,
        iteration: int = 0,
        losses_dict: dict[str, Any] | None = None,
        critic_cond: "Mapping[str, Any] | None" = None,
        **kwargs: Any,
    ) -> LossOutput:
        """Compute diffusion training losses.

        Args:
            pred: Model prediction (noise/score/velocity)
            target: Target (noise/score/velocity)
            epoch: Current epoch
            iteration: Current iteration
            critic_cond: Conditioning forwarded to the ``discriminator(pred)``
                call in the adversarial branch below (#1931). ``None``/``{}``
                reproduces the unconditioned call byte-for-byte.

                **Declared explicitly, not read out of ``**kwargs``**, and that
                is load-bearing: ``_call_safe_loss`` forwards this method's
                ``**kwargs`` to every reconstruction loss whose signature has a
                ``**kwargs`` of its own. A ``critic_cond`` riding in there would
                reach an arbitrary loss as a stray keyword. Naming it in the
                signature captures it here, so it can only ever reach the
                critic.
            **kwargs: timesteps, context, etc.

        Returns:
            LossOutput with total loss and components
        """
        device = target.device
        components = {}

        logger.debug(f"STDOUT: LossComputer Compute. Pred: {pred.shape}, Target: {target.shape}")

        # 1. DIFFUSION LOSS (main loss)
        lambda_diffusion = self._get_loss_weight("diffusion", epoch, iteration)
        if lambda_diffusion > 0:
            if self.diffusion_loss_fn:
                diffusion_loss = _call_safe_loss(self.diffusion_loss_fn, pred, target, **kwargs)
            else:
                # Fallback: MSE loss
                diffusion_loss = _complex_safe_mse(pred, target)
            components["diffusion"] = diffusion_loss

        # 2. RECONSTRUCTION LOSS (optional, at specific timesteps)
        lambda_recon = self._get_loss_weight("reconstruction", epoch, iteration)
        # The `reconstruction` slot is a FALLBACK (`_initialize_losses`
        # auto-wires complex_l1/l1 off their lambdas when nothing was
        # injected). When the SAME term arrives via losses_dict (the
        # LossBuilder path), computing the slot too double-counts it — on
        # exp_11 complex_l1 ran at an effective 2.0 (slot at 1.0 + dynamic
        # component at lambda_complex_l1). Skip the fallback in that case;
        # an externally-injected reconstruction_loss_fn is unaffected.
        _fallback_duplicated = (
            self._recon_fallback_name is not None
            and losses_dict is not None
            and self._recon_fallback_name in losses_dict
        )
        if lambda_recon > 0 and self.reconstruction_loss_fn and not _fallback_duplicated:
            # Only apply at final diffusion steps (low noise)
            timesteps = kwargs.get("timesteps")
            if timesteps is not None and torch.all(timesteps < 50):  # Final 50 steps
                recon_loss = _call_safe_loss(self.reconstruction_loss_fn, pred, target, **kwargs)
                components["reconstruction"] = recon_loss

        # 3. ADVERSARIAL LOSS (optional, for adversarial diffusion)
        lambda_adv = self._get_loss_weight("adversarial", epoch, iteration)
        discriminator = kwargs.get("discriminator", self.discriminator)

        if lambda_adv > 0 and discriminator is not None:
            # This branch used to be a bare ``pass`` guarded by an
            # ``and self.adversarial_loss_fn`` that could never be true --
            # ``adversarial_loss_fn`` is set to ``None`` in ``__init__`` and
            # assigned nowhere. So an arm could declare a non-zero adversarial
            # weight AND attach a discriminator, satisfy every precondition, and
            # receive no adversarial gradient at all. The comment said "usually
            # handled in strategy"; ``DiffusionTrainingStrategy`` has no
            # adversarial term either, so nothing computed it (#1669).
            #
            # A silently-dropped loss term is the exact shape non-negotiable 3
            # forbids: the run trains, the logs look like an adversarial run,
            # and the number is wrong. Until the generator-side term is wired to
            # its one owner (``UnifiedGANLossComputer``, which already resolves
            # the adversarial library and its pre-weighting), refuse the
            # configuration instead of honouring half of it.
            if self.adversarial_loss_fn is None:
                # Weight declared, critic attached, and no loss object to score
                # with. Refuse rather than train on the reconstruction term
                # alone: a run that silently drops a declared loss is
                # indistinguishable in its logs from one that honours it, which
                # is exactly what non-negotiable 3 forbids.
                raise ValueError(
                    f"losses.adversarial is declared (~{lambda_adv}) and a "
                    "discriminator is attached, but no adversarial loss object "
                    "could be built from this config, so the generator would "
                    "receive no adversarial gradient (#1669). Check "
                    "`losses.gan.gan_loss_type`, or set the adversarial weight "
                    "to 0 for a non-adversarial diffusion arm."
                )
            # ``critic_cond`` conditions the G step's critic call exactly as the
            # D step conditions its own (#1931). Both steps must score under the
            # same conditioning: a critic trained with labels but queried
            # without them is a different function, and the generator would be
            # chasing a gradient from a critic it never faces.
            fake_pred = discriminator(pred, **(critic_cond or {}))
            if hasattr(self.adversarial_loss_fn, "compute_generator_loss"):
                g_adv = self.adversarial_loss_fn.compute_generator_loss(
                    fake_outputs_d=fake_pred,
                    real_images=target,
                    fake_images=pred,
                )
            else:
                g_adv = self.adversarial_loss_fn(fake_pred, True)
            # ``gan_loss_library.compute_generator_loss`` returns a dict whose
            # entries are ALREADY scaled by their lambdas; re-weighting them
            # would double-count. Take only the adversarial entry -- the
            # reconstruction/perceptual terms in that dict are the GAN
            # generator's, and this generator's reconstruction objective is the
            # diffusion one computed above.
            if isinstance(g_adv, dict):
                adv = g_adv.get("g_adv_loss")
                if adv is None:
                    raise ValueError(
                        "the adversarial loss library returned no `g_adv_loss` "
                        f"(keys: {sorted(g_adv)}); refusing to guess which entry "
                        "is the adversarial term (#1669)."
                    )
                components["adversarial"] = adv
            else:
                components["adversarial"] = g_adv * lambda_adv

        # 4. DYNAMIC COMPONENT LOSSES (from LossBuilder)
        if losses_dict:
            for loss_name, loss_fn in losses_dict.items():
                # Skip if already computed or special keys
                # gradient_penalty and r1 are adversarial regularization losses
                # that require discriminator context — handled by GAN strategy
                if loss_name in components or loss_name in [
                    "diffusion",
                    "reconstruction",
                    "adversarial",
                    "gradient_penalty",
                    "r1",
                    "r1_regularization",
                ]:
                    continue

                try:
                    # CRITICAL DEBUG: Log exact shapes before each loss
                    logger.debug(
                        f"[Loss {loss_name}] BEFORE: pred.shape={pred.shape} (complex={torch.is_complex(pred)}), "
                        f"target.shape={target.shape} (complex={torch.is_complex(target)})"
                    )

                    # TI-CCD High-Fidelity Loss Suite: Apply exclusively to Target channels
                    pred_for_loss = pred
                    target_for_loss = target

                    if loss_name in [
                        "sense_adjoint_phase",
                        "sobolev_frequency",
                        "log_spectral",
                    ]:
                        # Isolate the target channels (the second half, e.g., last 8 channels out of 16)
                        if pred.shape[1] > target.shape[1]:
                            # If pred has more channels (e.g., 16 vs 8), take the last target.shape[1] channels
                            pred_for_loss = pred[:, -target.shape[1] :]
                        elif pred.shape[1] >= 16 and target.shape[1] >= 16:
                            # If both have 16 channels, Target is the second half
                            half_c = pred.shape[1] // 2
                            pred_for_loss = pred[:, half_c:]
                            target_for_loss = target[:, half_c:]

                    elif loss_name == "lpips":
                        # LPIPS expects 3-channel input (RGB). Adapt channels if needed.
                        if pred_for_loss.shape[1] != 3:
                            if pred_for_loss.shape[1] > 3:
                                pred_for_loss = pred_for_loss[:, :3]
                            else:
                                pred_for_loss = pred_for_loss.repeat(1, 3, 1, 1)[:, :3]

                        if target_for_loss.shape[1] != 3:
                            if target_for_loss.shape[1] > 3:
                                target_for_loss = target_for_loss[:, :3]
                            else:
                                target_for_loss = target_for_loss.repeat(1, 3, 1, 1)[:, :3]

                    # Compute loss with filtered kwargs (e.g., passing 'smaps')
                    loss_val = _call_safe_loss(loss_fn, pred_for_loss, target_for_loss, **kwargs)

                    # Log after successful computation
                    logger.debug(
                        f"[Loss {loss_name}] OK: loss_val={loss_val.item() if isinstance(loss_val, torch.Tensor) else loss_val}"
                    )
                    if isinstance(loss_val, torch.Tensor):
                        # Store the RAW component. Weighting is applied exactly
                        # ONCE downstream in ``_stack_components`` (base.py:285-289)
                        # via ``_get_loss_weight``. The previous pre-multiply
                        # here caused a w^2 double-application — invisible only
                        # while every survivor resolved to 1.0. With the
                        # 2026-05-30 DC-blob L1 rebalance
                        # (lambda_log_spectral=0.1, lambda_sobolev_kspace=0.05)
                        # the squaring would re-suppress the very k-space
                        # fidelity terms we are restoring (0.1 -> 0.01). This
                        # edit MUST land with the losses.reconstruction/
                        # losses.physics YAML rebalance; a partial apply is
                        # worse than no change.
                        components[loss_name] = loss_val
                except Exception as _exc:
                    # CRITICAL: Log full traceback and shapes for debugging shape mismatches
                    import traceback

                    logger.error(
                        f"[Loss {loss_name}] FAILED with shape pred={pred.shape}, target={target.shape}. "
                        f"Error: {type(_exc).__name__}: {_exc}"
                    )
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    # Re-raise to fail loudly
                    raise

        # 5. COMPUTE TOTAL LOSS
        total = self._stack_components(components, epoch=epoch, iteration=iteration)

        return LossOutput(total=total, components=components, metrics={})

    def compute_forward_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        epoch: int = 0,
        iteration: int = 0,
        **kwargs: Any,
    ) -> LossOutput:
        """Compute loss for forward diffusion pass (most common).

        Args:
            pred: Model prediction (what noise/score to subtract)
            target: Ground truth (actual noise added)
            timesteps: Diffusion timesteps (B,) with values [0, T]
            epoch: Current epoch
            **kwargs: Additional context

        Returns:
            LossOutput
        """
        device = target.device
        components = {}

        # Main diffusion loss - compute for all timesteps
        if self.diffusion_loss_fn:
            loss = self.diffusion_loss_fn(pred, target)
        else:
            loss = _complex_safe_mse(pred, target)

        components["diffusion_forward"] = loss

        # Optional: weight by timestep (common in diffusion)
        if hasattr(self.config, "loss_schedule"):
            schedule = self.config.loss_schedule
            if schedule == "snr":
                # SNR weighting (from Imagen)
                weights = self._compute_snr_weights(timesteps)
                loss = (loss * weights).mean()
            elif schedule == "timestep":
                # Linear timestep weighting
                max_t = kwargs.get("max_timesteps", 1000)
                weights = (timesteps.float() + 1.0) / max_t
                loss = (loss.mean(dim=(1, 2, 3)) * weights).mean()

        total = loss * self._get_loss_weight("diffusion", epoch, iteration)

        return LossOutput(total=total, components=components, metrics={})

    def _compute_snr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute SNR weighting for diffusion loss.

        Higher SNR at early timesteps -> lower weight
        Lower SNR at later timesteps -> higher weight

        Args:
            timesteps: (B,) timestep tensor

        Returns:
            (B,) weight tensor
        """
        # This would use actual alphas/sigmas from schedule
        # Placeholder implementation
        max_t = 1000
        # Simple linear mapping
        weights = (timesteps.float() + 1.0) / max_t
        return weights

    def validate_config(self) -> None:
        """Validate diffusion config."""
        required = ["num_timesteps"]
        for attr in required:
            if not hasattr(self.config, attr):
                raise ValueError(f"Diffusion config missing: {attr}")


class UnifiedReconstructionLossComputer(BaseLossComputer):
    """Unified reconstruction loss computer (non-diffusion).

    Used for standard reconstruction models without diffusion.
    """

    def __init__(self, config: Any, device: torch.device = torch.device("cpu")):
        """Initialize reconstruction loss computer."""
        self.l1_loss_fn = None
        self.l2_loss_fn = None
        self.perceptual_loss_fn = None

        super().__init__(config, device)

    def _initialize_losses(self) -> None:
        """Initialize loss functions."""
        pass

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        epoch: int = 0,
        iteration: int = 0,
        losses_dict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LossOutput:
        """Compute reconstruction losses.

        Args:
            pred: Reconstructed image
            target: Ground truth image
            epoch: Current epoch
            iteration: Current iteration
            **kwargs: Additional context

        Returns:
            LossOutput with total loss and components
        """
        device = target.device
        components = {}

        # 1. L1 LOSS
        lambda_l1 = self._get_loss_weight("l1", epoch, iteration)
        if lambda_l1 > 0:
            l1_loss = _complex_safe_l1(pred, target)
            components["l1"] = l1_loss

        # 2. L2 LOSS
        lambda_l2 = self._get_loss_weight("l2", epoch, iteration)
        if lambda_l2 > 0:
            l2_loss = _complex_safe_mse(pred, target)
            components["l2"] = l2_loss

        # 3. PERCEPTUAL LOSS
        lambda_percep = self._get_loss_weight("perceptual", epoch, iteration)
        if lambda_percep > 0 and self.perceptual_loss_fn:
            percep_loss = _call_safe_loss(self.perceptual_loss_fn, pred, target, **kwargs)
            components["perceptual"] = percep_loss

        # 4. PINN LOSS
        # Check if PINN loss was computed externally (passed via kwargs or components)
        pinn_loss = kwargs.get("pinn_loss")
        if pinn_loss is not None:
            if isinstance(pinn_loss, torch.Tensor):
                components["pinn"] = pinn_loss
            else:
                # If it's a module, we can't run it here without inputs, assume handled externally
                pass

        # 5. DYNAMIC COMPONENT LOSSES (from LossBuilder)
        if losses_dict:
            for loss_name, loss_fn in losses_dict.items():
                # Skip if already computed
                if loss_name in components:
                    continue

                # Adversarial regularization losses (gradient_penalty, r1*)
                # need discriminator context and are owned by the GAN
                # strategy — they would error if invoked here.  Unconditional
                # skip is correct.
                if loss_name in ("gradient_penalty", "r1", "r1_regularization"):
                    continue

                # L1 / L2 / MSE — the explicit blocks above compute these
                # when ``lambda_l1`` / ``lambda_l2`` are configured non-zero.
                # Skip ONLY when the explicit block actually ran (i.e. the
                # canonical key is already in ``components``).  Without this
                # conditional gate, a YAML that declares its loss exclusively
                # via the declarative list (``image_losses: [{name: mse}]``)
                # would be dropped here while the explicit L2 path was
                # gated off by ``lambda_l2=0.0`` and the L1 path by
                # iteration-warmup — leaving ``components`` empty and the
                # total a disconnected zero leaf with no upstream graph
                # (CLAUDE.md pitfall #9, "silent fallbacks are forbidden").
                if loss_name in ("l1", "l2", "mse"):
                    canonical = "l2" if loss_name == "mse" else loss_name
                    if canonical in components or loss_name in components:
                        continue

                try:
                    # Most losses take (pred, target)
                    # Use _call_safe_loss to pass compatible kwargs like smaps
                    loss_val = _call_safe_loss(loss_fn, pred, target, **kwargs)
                    if isinstance(loss_val, torch.Tensor):
                        components[loss_name] = loss_val
                except ValueError as e:
                    # F-LOSSFAIL / 2026-05-20 — CLAUDE.md pitfall #9 (no
                    # silent fallbacks). A ValueError from a loss is
                    # almost always a shape/channel mismatch, i.e., a
                    # configuration bug that MUST be fixed before
                    # training is meaningful. Previously this branch
                    # only logged a warning, so the smoke run would
                    # complete 10 iters with ``loss=0.0000`` and look
                    # green. Re-raise so the failure is loud.
                    raise ValueError(
                        f"Loss '{loss_name}' raised ValueError during "
                        f"_compute_declarative_losses_with_warmup: "
                        f"{e}. This is a shape / channel / dtype "
                        f"configuration mismatch that must be fixed "
                        f"(CLAUDE.md pitfall #9 forbids silent loss "
                        f"failure)."
                    ) from e
                except Exception as e:
                    logger.warning(
                        f"Failed to compute loss '{loss_name}': {type(e).__name__}: {e!s}"
                    )

        # F-LOSSFAIL / 2026-05-20 — if NO loss component succeeded, the
        # total is a disconnected zero with no upstream graph; training
        # silently runs with ``loss=0`` and gradients are noise. Surface
        # this as a hard error.
        if not components:
            scheduled = getattr(self, "scheduled_weights", None) or {}
            zeroed = sorted(k for k, v in scheduled.items() if float(v) == 0.0)
            hint = (
                f" A loss_schedule rule has set weight 0.0 for {zeroed}, so every "
                f"declared loss is currently gated off (schedule + warmup); disable a "
                f"term only while another stays active."
                if zeroed
                else " Check the loss-block configuration and the model's output "
                "shape against each declared loss."
            )
            raise RuntimeError(
                "All declared losses failed to compute (components is empty). "
                "Training would proceed with a disconnected zero total, which is a "
                "silent failure (CLAUDE.md pitfall #9)." + hint
            )

        # Compute total
        total = self._stack_components(components, epoch=epoch, iteration=iteration)

        return LossOutput(total=total, components=components, metrics={})


__all__ = [
    "UnifiedDiffusionLossComputer",
    "UnifiedReconstructionLossComputer",
]
