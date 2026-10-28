"""Virtual Fiducial Training Strategy — Config-Driven Marker-Guided Reconstruction.

Trains any VF generator (CrossAttentionOracleUNet, HyperMambaUNet, etc.)
using the DigitalTwinSimulator to generate corrupted marker+anatomy pairs.

Core idea: markers are **known ground-truth geometry** embedded before corruption.
The model observes how corruption distorts these markers and transfers the
learned inverse-corruption to the anatomy region.

Pipeline:
    1. Digital Twin: embed markers → apply motion → apply scanner corruptions
    2. Compute marker residual ΔM = corrupted_marker - ideal_marker
    3. Forward: model(corrupted_anatomy, ΔM) → reconstruction
    4. Loss: Natively supports the 40 independent Domain Objectives.
       Utilizes exact `torch.complex64` data mapping without splitting.

All simulator parameters come from ``config.physics.digital_twin`` (SSOT).
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, ClassVar

import torch

from spectramr.core.metrics.complex_input import PhaseBlindInputError
from spectramr.core.module_utils import unwrap_model
from spectramr.infrastructure.physics.spin_history import SpinHistoryStage
from spectramr.infrastructure.training.strategies.vf_phase_stego import PhaseStegoHook
from spectramr.infrastructure.training.strategies.vf_null_space import NullSpaceErasureHook
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.distributed.owned_modules import OWNED_PARAM_GROUP
from spectramr.infrastructure.training.strategies.owned_param_groups import add_owned_param_group
from spectramr.infrastructure.training.strategies.mixins.conditioning_mixin import (
    conditioning_block_of,
    conditioning_enabled,
)
from spectramr.infrastructure.training.strategies.ood_acceleration_readout import (
    ood_acceleration_readout,
    ood_accelerations,
)
from spectramr.infrastructure.training.strategies.simulator_builder import (
    build_simulator_from_config,
    require_epi_profile,
    undersampling_mask_kwargs,
)
from spectramr.infrastructure.training.utils.metric_views import MetricViews
from spectramr.models.conditioning import AdaptiveConditioner, ConditioningContext
from spectramr.models.losses.registry import create_loss

logger = logging.getLogger(__name__)

_INTERMEDIATE_PARAM_CACHE: dict[type, bool] = {}


def _loss_accepts_intermediate(loss_fn: object) -> bool:
    """True if ``loss_fn.forward`` declares an ``intermediate_outputs`` parameter.

    Signature-gates the deformation-field routing so the hyperelastic-Jacobian
    regulariser receives the velocity field while a plain L1/SSIM loss (whose
    forward is ``(pred, target)``) never gets an unexpected kwarg. Cached per
    class — the signature is static."""
    key = type(loss_fn)
    cached = _INTERMEDIATE_PARAM_CACHE.get(key)
    if cached is None:
        try:
            params = inspect.signature(loss_fn.forward).parameters  # type: ignore[attr-defined]
            cached = "intermediate_outputs" in params
        except (TypeError, ValueError):
            cached = False
        _INTERMEDIATE_PARAM_CACHE[key] = cached
    return cached


class ConcreteVirtualFiducialStrategy(BaseTrainingStrategy):
    """Unified training strategy for marker-guided reconstruction.

    Works natively in the `torch.complex64` domain. Does not separate
    real and imaginary components into dual channels during inversion
    to ensure explicit field mapping algorithms (Task 3) function cleanly.

    Attributes:
        simulator: Digital Twin forward physics model (config-driven).
    """

    # parallel_compat: With model.conditioning on, the conditioner is a bare group on opt_g,
    # refused under ddp: its null_token gets no gradient (VF never drops the condition), so a
    # DDP-wrapped conditioner fails the next step's reduction. Without it VF owns nothing.
    parallel_compat = frozenset({"replicated"})

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        on = conditioning_enabled(conditioning_block_of(config))
        own = (("_conditioner", OWNED_PARAM_GROUP),) if on else ()
        return (*super().owned_module_declarations(config), *own)

    synthesises_model_input: ClassVar[bool] = True

    _spin_history: SpinHistoryStage | None = None

    _phase_stego_hook: PhaseStegoHook | None = None
    _null_space_hook: NullSpaceErasureHook | None = None

    #: The twin undersamples at ``physics.digital_twin.acceleration`` when its
    #: ``enable_undersampling`` is true; the top-level ``undersampling:`` block
    #: reaches nothing here (the k-space mixin's generator is never called), so
    #: this strategy does NOT claim ``applies_undersampling`` -- the witness
    #: ``undersampling_block_is_applied`` reports that block on a VF arm (error on
    #: image data, UNVERIFIED on k-space data; VF review 2026-09-03). ``physics.digital_twin.ood_acceleration_range`` is
    #: read by ``validation_step`` (``ood_acceleration_readout``).
    reads_ood_acceleration_range = True

    #: The DigitalTwinSimulator corrupts the input INSIDE the step, so
    #: ``first_steps/input_prepared`` -- captured before the forward pass --
    #: equals ``input_raw``/``target`` and the markers are invisible in it. The
    #: model is fed ``corrupted_norm``; see ``_compute_losses_impl``.
    #:
    #: The code has said so in a comment since the twin snapshot was added
    #: ("the base first_steps auto-snapshot captures the PRE-twin input"), but
    #: the flag stayed at its ``True`` default -- so every VF artifact stamped
    #: ``prepared_equals_model_input: True`` and asserted the opposite of the
    #: comment. A caveat in the source mitigates for whoever edits the file; the
    #: field is what the reader of the artifact actually consumes (pitfall #16).
    snapshot_prepared_is_model_input: bool = False

    #: ``_snapshot_twin_outputs`` already writes this tag, and
    #: ``save_debug_snapshot`` marks the model-input requirement met on the way
    #: through -- so naming it here wires the existing emission to the contract
    #: rather than adding a second copy of the same tensors.
    snapshot_model_input_tag: str | None = "vf_twin"

    def __init__(
        self,
        env: Any,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, device=device, **kwargs)
        # No second ``self._setup_strategy_specific_components()``: the base __init__
        # already runs it, and it is not idempotent once it adds optimizer param groups.

    def _setup_strategy_specific_components(self) -> None:
        """Initialize simulator from config and register losses dynamically."""
        # ── Build simulator from config.physics.digital_twin (SSOT) ──
        self.simulator = build_simulator_from_config(self.config, self.device)
        require_epi_profile(self.simulator)

        # ── Adaptive conditioning (config.model.conditioning) ──
        # Lazily built on first forward so num_features matches the real input
        # channel count; its params are then registered with the generator
        # optimizer (see _ensure_conditioner). Identity at init → disabled or
        # untrained conditioning never changes the reconstruction.
        self._conditioner: AdaptiveConditioner | None = None
        self._conditioning_context: ConditioningContext | None = None

        # ── Build domain-tagged losses from config SSOT ──
        # The declarative v6 loss schema separates losses by the tensor domain
        # they expect (loss.py:1251-1253):
        #   kspace_losses : evaluated on centred k-space   (fft2c of complex)
        #   image_losses  : evaluated on real magnitude     (|complex|)
        #   complex_losses: evaluated on native complex64   (phase-bearing)
        # The previous implementation flattened all three lists and fed every
        # loss the MAGNITUDE image, which silently zeroed phase supervision for
        # the phase-physics arms (B0 / B1 / phase-navigator) whose primary
        # metric is ``val_phase_mse``.  We now route each loss to its declared
        # domain so complex_losses actually receive complex tensors.
        self._loss_specs: list[tuple[str, torch.nn.Module, float, str]] = []
        seen: set[str] = set()
        for domain in ("kspace", "image", "complex"):
            for comp in getattr(self.config.losses, f"{domain}_losses", []) or []:
                if comp.enabled and comp.weight > 0:
                    self._loss_specs.append(
                        (
                            comp.name,
                            create_loss(comp.name).to(self.device),
                            comp.weight,
                            domain,
                        )
                    )
                    seen.add(comp.name)

        # Legacy (non-list) configs: fall back to the flat enabled-loss dict,
        # treating every entry as image-domain (magnitude) as before.
        if not self._loss_specs:
            for name, weight in self.config.losses.get_enabled_losses().items():
                if weight > 0 and name not in seen:
                    self._loss_specs.append(
                        (name, create_loss(name).to(self.device), weight, "image")
                    )

        # ── Phase-domain steganographic marker (training.phase_steganography) ──
        ps = getattr(self.config.training, "phase_steganography", None)
        self._phase_stego_hook = (
            PhaseStegoHook(ps, self.device) if ps is not None and ps.enabled else None
        )
        # ── Null-space marker erasure at validation (training.null_space_erasure) ──
        nse = getattr(self.config.training, "null_space_erasure", None)
        self._null_space_hook: NullSpaceErasureHook | None = None
        if nse is not None and nse.enabled:
            if not nse.apply_at_inference:
                raise ValueError(
                    "training.null_space_erasure.enabled with apply_at_inference=false would "
                    "erase nowhere; set apply_at_inference=true or enabled=false."
                )
            self._null_space_hook = NullSpaceErasureHook(nse.epsilon)

        # ── Spin-history corruption (training.bloch_ode) ──
        bo = getattr(self.config.training, "bloch_ode", None)
        self._spin_history = None
        if bo is not None and bo.enabled:
            self._spin_history = SpinHistoryStage(
                dt=bo.dt,
                slice_thickness=bo.slice_thickness,
                n_steps=bo.n_steps,
                tr=bo.tr,
                t1_ms=bo.t1_ms,
                t2_ms=bo.t2_ms,
                amplitude=bo.amplitude,
                period_ms=bo.period_ms,
            )

        # Marker-anchored loss (always enabled for VF)
        self.loss_marker = create_loss("marker_corruption").to(self.device)
        self._lambda_marker = self.config.losses.reconstruction.lambda_marker

        # No silent L2 fallback (CLAUDE.md #9): a VF arm with zero configured
        # reconstruction losses is a misconfiguration that must fail loudly.
        if not self._loss_specs:
            raise ValueError(
                "[VirtualFiducialStrategy] No enabled reconstruction losses found. "
                "Declare at least one entry under losses.{image,complex,kspace}_losses "
                "(or a legacy losses.reconstruction.lambda_* term). Refusing to fall "
                "back to an implicit L2 — silent fallbacks are forbidden."
            )

        logger.info(
            "[VirtualFiducialStrategy] Active losses (name@domain=weight): %s, λ_marker=%.2f",
            {f"{n}@{d}": f"{w:.2f}" for n, _, w, d in self._loss_specs},
            self._lambda_marker,
        )

    @staticmethod
    def _normalize_to_range(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize complex tensor magnitude for [-1, 1] bounded variation."""
        if not tensor.is_complex():
            # Fallback for real tensors
            t_min = tensor.min()
            t_max = tensor.max()
            t_range = t_max - t_min + 1e-8
            return 2.0 * (tensor - t_min) / t_range - 1.0, t_min, t_max

        # For native complex tensors, divide by the max magnitude
        t_max_mag = tensor.abs().max().clamp(min=1e-8)
        zero = torch.zeros(1, device=tensor.device)
        return tensor / t_max_mag, zero, t_max_mag

    @staticmethod
    def _denormalize_from_range(
        tensor: torch.Tensor, t_min: torch.Tensor, t_max: torch.Tensor
    ) -> torch.Tensor:
        """Reverse complex or real normalization."""
        if tensor.is_complex():
            return tensor * t_max
        t_range = t_max - t_min
        return (tensor + 1.0) / 2.0 * t_range + t_min

    def _to_complex(self, batch: torch.Tensor) -> torch.Tensor:
        """Convert real-stacked tensor [B, 2C, H, W] to complex64 [B, C, H, W]."""
        if torch.is_complex(batch):
            return batch
        C = batch.shape[1]
        if C >= 2 and C % 2 == 0:
            return torch.complex(batch[:, 0::2], batch[:, 1::2])
        return batch.to(torch.complex64)

    @staticmethod
    def _complex_to_real(t: torch.Tensor) -> torch.Tensor:
        """Convert complex [B, C, H, W] → real [B, 2C, H, W] (real/imag stacked)."""
        if not torch.is_complex(t):
            return t
        return torch.cat([t.real, t.imag], dim=1)

    # Condition sources this strategy can actually populate. Sources like
    # ``diffusion_t`` / ``motion_pose`` require the diffusion / meta VF
    # strategies that expose those quantities.
    _SUPPORTED_CONDITION_SOURCES = ("severity_vec",)

    @staticmethod
    def _severity_vector(simulator: Any, batch_size: int, device: torch.device) -> torch.Tensor:
        """Physics-grounded degradation token from the simulator config.

        A constant per-batch ``[B, 5]`` vector
        ``[motion_severity, b0, b1, noise, accel]`` reflecting the configured
        degradation floor — the regime the network is asked to invert. Disabled
        artifacts contribute 0. This is a degradation-aware prompt (PromptIR),
        grounded in the actual twin settings rather than learned from scratch.
        """
        s = float(getattr(simulator, "_motion_severity", 1.0))
        b0 = (
            float(getattr(simulator, "b0_strength", 0.0))
            if getattr(simulator, "enable_b0", False)
            else 0.0
        )
        b1 = (
            float(getattr(simulator, "b1_strength", 0.0))
            if getattr(simulator, "enable_b1", False)
            else 0.0
        )
        snr_lo = float(getattr(simulator, "snr_range", (10.0, 25.0))[0])
        noise = max(0.0, 1.0 - snr_lo / 100.0)
        accel = (
            float(getattr(simulator, "acceleration", 1.0))
            if getattr(simulator, "enable_undersampling", False)
            else 1.0
        )
        accel_n = (accel - 1.0) / 15.0
        vec = torch.tensor([s, b0, b1, noise, accel_n], device=device, dtype=torch.float32)
        return vec.unsqueeze(0).expand(int(batch_size), -1).contiguous()

    @classmethod
    def _build_conditioning_context(
        cls,
        sources: list[str],
        simulator: Any,
        batch_size: int,
        device: torch.device,
    ) -> ConditioningContext | None:
        """Populate a ConditioningContext from the configured sources.

        Only sources this strategy can provide are accepted; an unsupported
        source raises (CLAUDE.md #9 — no silent no-op conditioning).
        """
        if not sources:
            return None
        unsupported = [s for s in sources if s not in cls._SUPPORTED_CONDITION_SOURCES]
        if unsupported:
            raise ValueError(
                f"[VirtualFiducialStrategy] conditioning sources {unsupported} are not "
                f"providable here; this strategy can populate "
                f"{list(cls._SUPPORTED_CONDITION_SOURCES)}. Sources like 'diffusion_t' / "
                f"'motion_pose' require the diffusion / meta VF strategies."
            )
        kwargs: dict[str, torch.Tensor] = {}
        if "severity_vec" in sources:
            kwargs["severity_vec"] = cls._severity_vector(simulator, batch_size, device)
        return ConditioningContext(**kwargs)

    def _ensure_conditioner(self, n_channels: int) -> AdaptiveConditioner | None:
        """Lazily build the conditioner and register its params with opt_g.

        Built on first forward so ``num_features`` matches the real input
        channel count; its parameters are added to the generator optimizer so
        they actually train (mirrors ``XFieldFMStrategy._ensure_hypernet``).
        Returns ``None`` when conditioning is disabled.
        """
        cfg = conditioning_block_of(self.config)
        if not conditioning_enabled(cfg):
            return None
        if self._conditioner is None:
            cond = AdaptiveConditioner.from_config(cfg, num_features=n_channels)
            if cond is not None:
                cond = cond.to(self.device)
                opt = (
                    getattr(self.env, "opt_g", None)
                    or getattr(self.env, "optimizer_g", None)
                    or getattr(self.env, "optimizer", None)
                )
                if opt is not None:
                    add_owned_param_group(
                        self, opt, {"params": list(cond.parameters())}, owner="its conditioner"
                    )
                else:
                    logger.warning(
                        "[VirtualFiducialStrategy] no generator optimizer found; "
                        "conditioner params will NOT be trained."
                    )
            self._conditioner = cond
        return self._conditioner

    @staticmethod
    def _forward_accepts_marker_signal(gen: Any) -> bool:
        """True if the generator's ``forward()`` declares a ``marker_signal`` param.

        Routes a per-sample marker vector to models (NeuralAdvectionGenerator,
        DynamicMRNeRF) that condition on it but expose no ``cross_attention`` /
        ``bridge`` attribute. Signature inspection is model-agnostic — no model
        edits required; a ``**kwargs``-only forward returns False (marker stays
        dropped, preserving prior behaviour for those models).
        """
        try:
            # The inner forward: a DDP/DP wrapper's is ``(*args, **kwargs)``.
            params = inspect.signature(unwrap_model(gen).forward).parameters
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            return False
        return "marker_signal" in params

    @staticmethod
    def _marker_signal_from_residual(marker_residual: torch.Tensor) -> torch.Tensor:
        """Reduce the marker residual to the per-sample scalar a MarkerEncoder wants.

        ``marker_residual`` is the (already-normalised) complex marker corruption
        ΔM = corrupted_marker − ideal_marker, shape ``[B, C, H, W]``. The
        NeuralAdvection / DynamicMRNeRF encoders take ``marker_signal`` of shape
        ``[B, marker_dim]`` (``marker_dim=1`` in these arms), so we summarise ΔM
        by its per-sample mean magnitude — the corruption / motion strength the
        marker encodes — yielding ``[B, 1]``.
        """
        return marker_residual.abs().flatten(1).mean(dim=1, keepdim=True)

    def _forward_model(
        self,
        corrupted_input: torch.Tensor,
        marker_residual: torch.Tensor,
        *,
        marker_prior: torch.Tensor | None = None,
        corrupted_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dispatch forward call to the appropriate model interface.

        Converts complex64 tensors to 2-channel real stacked format before
        passing to the generator (which uses standard nn.Conv2d), then converts
        the output back to complex64.
        """
        gen = self.generator_model

        # ── Convert complex → real channels for nn.Conv2d compatibility ──
        corrupted_real = self._complex_to_real(corrupted_input)
        marker_res_real = self._complex_to_real(marker_residual)

        # ── Adaptive conditioning: FiLM-modulate the model input on the
        # configured condition sources. Identity at init; no-op when disabled
        # or when no context was built for this batch. Model-agnostic — no
        # generator forward signature changes, so no existing model breaks. ──
        conditioner = self._ensure_conditioner(corrupted_real.shape[1])
        if conditioner is not None and self._conditioning_context is not None:
            corrupted_real = conditioner(corrupted_real, self._conditioning_context)

        physics_kwargs: dict[str, Any] = {}
        if marker_prior is not None:
            physics_kwargs["marker_prior"] = self._complex_to_real(marker_prior)
        if corrupted_image is not None:
            physics_kwargs["corrupted_image"] = self._complex_to_real(corrupted_image)
        # Feed the twin's k-space sampling mask to the model's data-consistency
        # layers when undersampling is active (no-op for non-undersampling arms).
        physics_kwargs.update(undersampling_mask_kwargs(self.simulator))

        # Capability probes read the inner module; the call itself stays on the wrapper.
        inner = unwrap_model(gen)
        if hasattr(inner, "cross_attention"):
            out = gen(
                corrupted_real,
                marker_residual=marker_res_real,
                **physics_kwargs,
            )
        elif hasattr(inner, "bridge"):
            fiducial_input = marker_res_real[:, :2]  # 2 real channels
            out = gen(
                corrupted_real,
                corrupted_fiducial=fiducial_input,
                **physics_kwargs,
            )
        elif self._forward_accepts_marker_signal(gen):
            # Generators like NeuralAdvectionGenerator / DynamicMRNeRF condition
            # on a per-sample ``marker_signal`` VECTOR ([B] or [B, marker_dim]),
            # encoded by a small MarkerEncoder/temporal-encoder — NOT the
            # cross-attention 2-channel marker image. Before this branch the
            # dispatch fell to the ``else`` and dropped the marker entirely, so
            # the model ignored the VF anchor and produced time-invariant /
            # identity output (the experiment's mechanism was inert — VF review
            # 2026-06-04, exp_vf_m5m9 / exp_vf_m10). Derive the scalar the
            # MarkerEncoder expects: the per-sample mean marker-residual
            # magnitude, i.e. the corruption/motion strength the marker encodes.
            # (A richer per-marker encoding would need a wider model interface;
            # marker_dim=1 in these arms means a scalar is what the net consumes.)
            marker_signal = self._marker_signal_from_residual(marker_residual)
            out = gen(corrupted_real, marker_signal=marker_signal, **physics_kwargs)
        else:
            out = gen(corrupted_real, **physics_kwargs)

        # ── Convert output back to complex64 ──
        return self._to_complex(out)

    def _twin_input(self, target_complex: torch.Tensor) -> torch.Tensor:
        """Anatomy the twin corrupts: the clean target, carrying the phase marker when enabled."""
        twin_input = target_complex
        if self._phase_stego_hook is not None:
            twin_input = self._phase_stego_hook.inject(twin_input)
        if self._spin_history is not None:
            self._last_spin_gain = self._spin_history.gain(twin_input)
            twin_input = twin_input * self._last_spin_gain
        return twin_input

    def _compute_marker_residual(
        self,
        corrupted_image: torch.Tensor,
        marker_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Compute native complex ΔM in marker region."""
        marker_mask = self.simulator.marker_mask.to(self.device).to(corrupted_image.dtype)
        # We process natively in torch.complex64 without real/imag separation
        return (corrupted_image - marker_prior) * marker_mask

    @staticmethod
    def _apply_domain_losses(
        loss_specs: list[tuple[str, torch.nn.Module, float, str]],
        pred: torch.Tensor,
        target: torch.Tensor,
        device: torch.device,
        intermediate_outputs: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate each loss on the tensor domain it was declared under.

        Args:
            loss_specs: ``(name, loss_fn, weight, domain)`` tuples; ``domain`` is
                one of ``"image"`` (magnitude), ``"complex"`` (native complex64),
                or ``"kspace"`` (centred k-space).
            intermediate_outputs: generator intermediates (e.g. the STN velocity
                /deformation field). Forwarded ONLY to losses whose ``forward``
                declares an ``intermediate_outputs`` parameter (the
                hyperelastic-Jacobian incompressibility regulariser, which must
                act on the deformation field, not the reconstructed image). A
                plain L1/SSIM loss never sees it.
            pred: Model prediction, complex64 ``[B, C, H, W]``.
            target: Clean target, complex64 ``[B, C, H, W]``.
            device: Accumulator device.

        Returns:
            ``(g_total_loss, loss_dict)`` where ``loss_dict`` carries the
            detached per-loss components keyed ``loss_<name>``.
        """
        pred_mag = pred.abs() if pred.is_complex() else pred
        target_mag = target.abs() if target.is_complex() else target
        # Perceptual / SSIM losses require 4D [B, C, H, W].
        if pred_mag.ndim == 3:
            pred_mag = pred_mag.unsqueeze(1)
        if target_mag.ndim == 3:
            target_mag = target_mag.unsqueeze(1)

        g_total_loss = torch.tensor(0.0, device=device)
        loss_dict: dict[str, torch.Tensor] = {}
        kspace_cache: tuple[torch.Tensor, torch.Tensor] | None = None

        for name, loss_fn, weight, domain in loss_specs:
            if domain == "complex":
                p, t = pred, target
            elif domain == "kspace":
                if kspace_cache is None:
                    from spectramr.infrastructure.physics.fft_ops import fft2c

                    kspace_cache = (fft2c(pred), fft2c(target))
                p, t = kspace_cache
            else:  # image / legacy default
                p, t = pred_mag, target_mag

            if intermediate_outputs is not None and _loss_accepts_intermediate(loss_fn):
                loss_val = loss_fn(p, t, intermediate_outputs=intermediate_outputs)
            else:
                loss_val = loss_fn(p, t)
            g_total_loss = g_total_loss + weight * loss_val
            loss_dict[f"loss_{name}"] = loss_val.detach()

        return g_total_loss, loss_dict

    def _snapshot_twin_outputs(
        self,
        *,
        step: int,
        epoch: int,
        clean_norm: torch.Tensor,
        corrupted_norm: torch.Tensor,
        marker_residual: torch.Tensor,
        marker_prior_norm: torch.Tensor,
    ) -> None:
        """Capture the DigitalTwinSimulator outputs for marker-firing checks.

        The base-class ``first_steps`` auto-snapshot (``base.py``) fires UPSTREAM
        of the twin, so its ``input_prepared`` equals ``input_raw``/``target`` and
        cannot prove the marker mechanism ran (pitfall #16 — a facade where the
        VF method silently collapses to a vanilla denoiser). This captures the
        twin's OUTPUTS — the corrupted input the model sees, the marker residual
        ΔM it conditions on, and the ideal marker geometry — plus a scalar
        ``marker_mechanism_fired`` / ``marker_residual_abs_max`` stamped into the
        snapshot ``extra`` so a reviewer can confirm the markers are live without
        opening the PNGs.

        The ``marker_residual`` reduction is a device->host sync, so it is
        DEFERRED: ``extra`` is handed over as a zero-arg callable that
        ``save_debug_snapshot`` invokes only once a write is certain, i.e. after
        both the interval check and the ``max_calls`` budget have passed. The
        warm training loop therefore never pays for it (non-negotiable 9).
        Diagnostics must never break training — any failure is swallowed.
        """

        # No private budget here. `save_debug_snapshot` owns the allowance, and
        # since #706 it is keyed per (run_dir, TAG) -- so `vf_twin` gets its own
        # `max_calls` instead of competing with `first_steps` / `model_output`
        # for one shared counter. A second owner also compared `step > max_calls`,
        # i.e. an ITERATION against a CALL budget, which stops twin snapshots at
        # iteration 8 regardless of how many were actually written.
        #
        # Removing that private budget (#706) is what exposed #1188: it had been
        # the only thing keeping the sync below off the warm path, because it sat
        # in THIS method, upstream of the eager `.item()`. `save_debug_snapshot`'s
        # own budget sits downstream of it and so cannot suppress it. Hence the
        # deferral -- the callable is built here for free and called there, or
        # not at all. Whole-dict form, not per-key: all three values derive from
        # ONE reduction, so per-key closures would sync twice.
        def _twin_extra() -> dict[str, Any]:
            mr_abs_max = float(marker_residual.detach().abs().max().item())
            return {
                "epoch": int(epoch),
                "marker_residual_abs_max": mr_abs_max,
                "marker_mechanism_fired": bool(mr_abs_max > 0.0),
            }

        try:
            self.save_debug_snapshot(
                {
                    "twin_target_clean": clean_norm,
                    "twin_corrupted_input": corrupted_norm,
                    "twin_marker_residual": marker_residual,
                    "twin_marker_prior": marker_prior_norm,
                },
                step=step,
                tag="vf_twin",
                extra=_twin_extra,
                # ``vf_twin`` IS this strategy's ``snapshot_model_input_tag``,
                # so the snapshot must say which of its four tensors the model
                # receives; the other three are the clean twin and the marker
                # halves shown for contrast. It has always been
                # ``corrupted_norm`` (class docstring, line ~74) -- naming it
                # here is what makes that checkable instead of documented.
                # A parameter and not part of ``_twin_extra`` because that dict
                # is deferred: resolving it to read one key would force the very
                # ``.item()`` sync the deferral exists to postpone.
                model_input_key="twin_corrupted_input",
            )
        except Exception as exc:  # diagnostics must never break training
            logger.debug("[VirtualFiducialStrategy] twin snapshot skipped: %s", exc)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        external_b0_field: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute dynamically-configured mathematical objective.

        Evaluates the tensor natively utilizing `torch.complex64` throughout
        the entire autograd pipeline. The magnitude normalization mitigates
        overflows.

        Args:
            input_batch: Unused (we corrupt target_batch internally).
            target_batch: Clean ground truth multi-contrast anatomy.
            epoch: Current epoch.
            **kwargs: Additional kwargs.

        Returns:
            Loss dict with ``g_total_loss`` and individually computed components.
        """
        # Handle 5D TorchIO volumes
        if target_batch.ndim == 5:
            mid = target_batch.shape[-1] // 2
            target_batch = target_batch[..., mid]

        # Ensure native complex formatting
        target_complex = self._to_complex(target_batch)

        # [DOMAIN CORRECTION] Convert k-space to image domain if needed
        # The Digital Twin simulator expects image-domain inputs for motion/marker simulation
        if hasattr(self.config.data, "dataset_type") and self.config.data.dataset_type == "kspace":
            from spectramr.infrastructure.physics.fft_ops import FFTTransformer

            fft = FFTTransformer(device=self.device)
            target_complex = fft.ifft2c(target_complex)

        # 1. Run Digital Twin simulator. When a REAL B0 map is supplied
        # (from the batch, e.g. derived from multi-echo data), the twin applies
        # it verbatim so the field-scoring grades against a real off-resonance
        # pattern instead of a random simulation (VF real-reference seam).
        corrupted_image, marker_prior, joint_clean = self.simulator(
            self._twin_input(target_complex), external_b0_field=self._slice_field(external_b0_field)
        )

        # 2. Compute native complex marker residual ΔM
        marker_residual = self._compute_marker_residual(corrupted_image, marker_prior)

        # 3. Normalize cleanly to complex magnitude [0, 1]
        clean_norm, t_min, t_max = self._normalize_to_range(target_complex)

        # Corrupted input normalizes via the pristine target's bounding magnitude
        if target_complex.is_complex():
            corrupted_norm = corrupted_image / t_max
            marker_residual = marker_residual / t_max
            marker_prior_norm = marker_prior / t_max
        else:
            corrupted_norm = 2.0 * (corrupted_image - t_min) / (t_max - t_min + 1e-8) - 1.0
            marker_residual = marker_residual / (t_max - t_min + 1e-8)
            marker_prior_norm = marker_prior

        # 3a-bis. Marker-mechanism visibility snapshot (pitfall #16 facade guard).
        # The base "first_steps" auto-snapshot captures the PRE-twin input — so
        # input_raw == input_prepared == target and the markers are invisible.
        # Snapshot the simulator's OUTPUTS so the mechanism is verifiable.
        self._snapshot_twin_outputs(
            step=int(kwargs.get("iteration", 0) or 0),
            epoch=epoch,
            clean_norm=clean_norm,
            corrupted_norm=corrupted_norm,
            marker_residual=marker_residual,
            marker_prior_norm=marker_prior_norm,
        )

        # 3b. Build the adaptive-conditioning context for this batch (only the
        # sources declared in config.model.conditioning; consumed inside
        # _forward_model). None when conditioning is disabled.
        self._conditioning_context = self._build_conditioning_context(
            list(getattr(self.config.model.conditioning, "sources", []) or []),
            self.simulator,
            corrupted_norm.shape[0],
            self.device,
        )

        # 4. Forward purely natively as complex64
        pred = self._forward_model(
            corrupted_norm,
            marker_residual,
            marker_prior=marker_prior_norm,
            corrupted_image=corrupted_norm,
        )

        # Enforce complex typing if model outputs real dual-channels (like WienerUNet)
        pred = self._to_complex(pred)

        # 5. Iteratively evaluate the strict domain objectives dictated by SSOT.
        #
        # Each loss is routed to the tensor domain it was declared under:
        #   image   → MAGNITUDE (|complex|): standard L1/L2/SSIM/perceptual are
        #             NOT complex-aware; feeding them complex collapses gradients
        #             on non-marker voxels (network emits marker-only signal).
        #   complex → native complex64: phase-bearing losses
        #             (phase_smoothness_complex, …) MUST see the imaginary part,
        #             otherwise phase supervision is silently zeroed.
        #   kspace  → fft2c(complex): k-space-domain objectives.
        # The complex-aware MarkerCorruptionLoss (step 6) always receives complex.
        # A field-deforming generator (neural_advection) caches its STN velocity
        # field; route it as an intermediate so the hyperelastic-Jacobian loss
        # regularises the DEFORMATION FIELD, not the reconstructed image.
        _deform = getattr(unwrap_model(self.generator_model), "last_deformation_field", None)
        g_total_loss, loss_dict = self._apply_domain_losses(
            self._loss_specs,
            pred,
            clean_norm,
            self.device,
            intermediate_outputs=[_deform] if _deform is not None else None,
        )

        # 6. Default marker anchoring (Transferability constraint component)
        # MarkerCorruptionLoss is complex-aware — receives native complex64.
        marker_mask = self.simulator.marker_mask.to(self.device)
        loss_marker = self.loss_marker(
            pred,
            clean_norm,
            marker_mask=marker_mask,
            ideal_marker=marker_prior_norm,
        )

        g_total_loss = g_total_loss + self._lambda_marker * loss_marker
        loss_dict["loss_marker"] = loss_marker.detach()
        loss_dict["g_total_loss"] = g_total_loss

        return loss_dict

    @staticmethod
    def _to_image_domain(
        pred: torch.Tensor,
        output_domain: str,
        device: torch.device,
    ) -> torch.Tensor:
        """Bring a (complex) prediction into image domain for magnitude/visual.

        VF models may emit k-space (``output_domain == 'kspace'``); the
        validation visual and ``val_psnr`` must be computed in image domain,
        symmetric to the target-side IFFT in :py:meth:`validation_step`. An
        image-domain prediction passes through unchanged (no double-IFFT).

        Regression guard for F1 (``TODO/audit/smoke_audit_20260526.md``): the
        previous code took ``pred.abs()`` directly, rendering raw |k-space| as
        the "fake" image (4-corner-blob / concentric-ring artifact) and a
        cross-domain PSNR for every ``output_domain: kspace`` arm.
        """
        if output_domain == "kspace":
            from spectramr.infrastructure.physics.fft_ops import FFTTransformer

            return FFTTransformer(device=device).ifft2c(pred)
        return pred

    @staticmethod
    def _slice_field(field: torch.Tensor | None) -> torch.Tensor | None:
        """Reduce a reference field to the slice the twin operates on.

        A 5-D ``[B, C, H, W, D]`` reference map is reduced to its middle slice so
        it lines up with the middle-slice target the VF pipeline uses.
        """
        if field is None:
            return None
        if field.ndim == 5:
            field = field[..., field.shape[-1] // 2]
        return field

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        b0_map: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute image-quality validation metrics for the VF reconstruction.

        NOTE: this does NOT fix the simulator noise. ``DigitalTwinSimulator
        .forward`` draws fresh ``torch.randn``/``torch.rand`` internally with no
        generator argument, so val metrics fluctuate per call; truly seeding it
        requires threading a generator into the simulator (digital_twin_
        simulator.py), an other-module change. The earlier "fixed noise for
        reproducibility" claim was inaccurate and has been removed (audit 2026-06).
        """
        val_batch = (input_batch, target_batch)
        batch_idx = kwargs.get("batch_idx", 0)
        if isinstance(val_batch, list | tuple):
            target = val_batch[1]
        elif isinstance(val_batch, dict):
            target = val_batch.get("target", val_batch.get("hr"))
        elif hasattr(val_batch, "target"):
            target = val_batch.target
        else:
            target = val_batch

        target = self._to_device(target)

        # Handle 5D TorchIO volumes [B, C, H, W, D] → select middle slice
        if target.ndim == 5:
            mid = target.shape[-1] // 2
            target = target[..., mid]

        with torch.no_grad():
            # Compute losses via the SAME path as training. A real B0 reference
            # (from the batch) drives the twin so the field-scoring grades the
            # model against a real off-resonance pattern (VF real-reference seam).
            #
            # `snapshot_source` because that shared path is not diagnostics-free:
            # `_compute_losses_impl` emits the `vf_twin` snapshot, so without the
            # declaration the provenance record would attach the TRAIN augmentation
            # chain to a val batch -- the one claim `source` exists to make
            # falsifiable. Reusing the training path is the point here; labelling
            # the artifact as training data is not.
            #
            # #1190: the same direct call also bypasses the ``_compute_losses``
            # wrapper that emits the generic ``model_output`` snapshot, so arm it
            # here too. This is the one bypass site that already emitted SOMETHING
            # (`vf_twin`, from inside the impl) -- the twin is not the model
            # output, so the two are complementary, and `_model_output_snapshot_
            # done` means whichever the impl declares as its own still wins.
            with (
                self.snapshot_source("val"),
                self._capture_model_output(
                    module=self.generator_model,
                    input_batch=target,
                    target_batch=target,
                    step=batch_idx,
                ),
            ):
                losses = self._compute_losses_impl(
                    target, target, epoch=0, external_b0_field=b0_map
                )

            # Reuse the EXACT training normalization for PSNR
            target_complex = self._to_complex(target)

            # [DOMAIN CORRECTION] Convert k-space to image domain if needed
            if (
                hasattr(self.config.data, "dataset_type")
                and self.config.data.dataset_type == "kspace"
            ):
                from spectramr.infrastructure.physics.fft_ops import FFTTransformer

                fft = FFTTransformer(device=self.device)
                target_complex = fft.ifft2c(target_complex)

            scores = self._score_at_current_twin(target_complex, b0_map, cache_visuals=True)

        result = {k: float(v) for k, v in losses.items() if isinstance(v, torch.Tensor)}
        result.update(scores)

        # Field self-consistency (the qMRI-claim test image-only val_psnr can't
        # give). Auto-activates only when the twin applied a B0 displacement
        # field AND the model exposed a shift estimate (e.g. EPI tracker).
        # Scored BEFORE the out-of-distribution rungs so it reads the
        # in-distribution twin fields and the shift estimate of that pass.
        result.update(self._score_field(real_reference=b0_map is not None))

        # Out-of-distribution rungs (physics.digital_twin.ood_acceleration_range):
        # the same corrupt-reconstruct-score pass with the twin held at each rung;
        # ``val_ood_accelerations`` is written on every step (0 when undeclared).
        with torch.no_grad():
            result.update(
                ood_acceleration_readout(
                    self.simulator,
                    ood_accelerations(self.config),
                    lambda: self._score_at_current_twin(
                        target_complex, b0_map, cache_visuals=False
                    ),
                )
            )
        return result

    def _score_at_current_twin(
        self,
        target_complex: torch.Tensor,
        b0_map: torch.Tensor | None,
        *,
        cache_visuals: bool,
    ) -> dict[str, float]:
        """Corrupt ``target_complex`` with the twin as it is set now, reconstruct, score.

        One pass for the in-distribution rung and for every OOD rung: the twin's
        acceleration is whatever the caller set (``at_acceleration`` for OOD),
        and the conditioning context is rebuilt from the twin so a
        severity-conditioned model is told the rate it is scored at. Returns
        ``val_psnr`` plus the configured validation metrics under their own
        keys. ``cache_visuals`` stores the magnitude pair for the pipeline's
        image logging (the in-distribution pass only).
        """
        corrupted_image, marker_prior, joint_clean = self.simulator(
            self._twin_input(target_complex), external_b0_field=self._slice_field(b0_map)
        )
        marker_residual = self._compute_marker_residual(corrupted_image, marker_prior)

        # SAME normalization as _compute_losses_impl
        clean_norm, t_min, t_max = self._normalize_to_range(target_complex)
        if target_complex.is_complex():
            corrupted_norm = corrupted_image / t_max
            marker_residual = marker_residual / t_max
            marker_prior_norm = marker_prior / t_max
        else:
            corrupted_norm = 2.0 * (corrupted_image - t_min) / (t_max - t_min + 1e-8) - 1.0
            marker_residual = marker_residual / (t_max - t_min + 1e-8)
            marker_prior_norm = marker_prior

        self._conditioning_context = self._build_conditioning_context(
            list(getattr(self.config.model.conditioning, "sources", []) or []),
            self.simulator,
            corrupted_norm.shape[0],
            self.device,
        )
        pred = self._forward_model(
            corrupted_norm,
            marker_residual,
            marker_prior=marker_prior_norm,
            corrupted_image=corrupted_norm,
        )
        pred = self._to_complex(pred)

        # [DOMAIN CORRECTION] k-space-output VF models (output_domain='kspace')
        # must be IFFT'd to image domain before magnitude/visual -- symmetric to
        # the target IFFT in validation_step. Without this, the cached "fake"
        # visual and val_psnr are computed on raw |k-space| (4-corner-blob / ring
        # artifact, F1 / TODO/audit/smoke_audit_20260526.md). infer_output_domain
        # is the authoritative SSOT (raises on inconsistency; no silent fallback).
        from spectramr.infrastructure.training.utils.domain_inference import (
            infer_output_domain,
        )

        pred = self._to_image_domain(pred, infer_output_domain(self.config), self.device)

        # Erase the twin's marker from the reconstruction; the unerased PSNR rides along as the
        # arm's baseline so one run carries both sides of the contrast.
        null_space_scores: dict[str, float] = {}
        if self._null_space_hook is not None:
            unerased = pred.abs()
            pred = self._null_space_hook.erase(pred, marker_prior_norm)
            null_space_scores = self._null_space_hook.scores(pred, clean_norm, marker_prior_norm)
            mse_u = torch.nn.functional.mse_loss(unerased, clean_norm.abs())
            rng = clean_norm.abs().max() - clean_norm.abs().min() + 1e-8
            null_space_scores["val_psnr_unerased"] = float(
                10.0 * torch.log10(rng**2 / (mse_u + 1e-10))
            )

        # PSNR on magnitude images (standard MRI metric)
        pred_mag = pred.abs()
        target_mag = clean_norm.abs()
        mse = torch.nn.functional.mse_loss(pred_mag, target_mag)
        data_range = target_mag.max() - target_mag.min() + 1e-8
        psnr = 10.0 * torch.log10(data_range**2 / (mse + 1e-10))

        if cache_visuals:
            # Cache image-domain magnitude visuals for the pipeline's image
            # logging. Without this, train.py falls back to a raw
            # generator(input_batch) call that bypasses the VF simulator
            # pipeline, producing k-space / noise outputs.
            self._last_visual_pred = pred_mag.detach().cpu()
            self._last_visual_target = target_mag.detach().cpu()

        scores: dict[str, float] = {"val_psnr": float(psnr)}
        scores.update(null_space_scores)
        if self._spin_history is not None:
            scores["val_spin_history_gain_std"] = float(self._last_spin_gain.std())
        if self._phase_stego_hook is not None:
            scores.update(self._phase_stego_hook.scores(target_complex))
            displacement = getattr(self.simulator, "last_elastic_displacement_px", None)
            if displacement is not None:
                scores.update(
                    self._phase_stego_hook.displacement_rmse_px(
                        corrupted_image, joint_clean, displacement
                    )
                )
        # Emit the YAML-configured ``validation.metrics`` so early-stopping /
        # best-metric monitors (e.g. ``val_hfen``, ``val_robust_mri_psnr``)
        # resolve. Without this the strategy only returned ``val_psnr`` and
        # every monitor silently never fired -> runs burned full walltime
        # (dispatch 6944227, 2026-05-25). compute() guards each metric
        # individually, so a single bad metric is skipped, not fatal; we still
        # surface failures as a warning rather than swallowing them silently
        # (CLAUDE.md #9/#10).
        computer = getattr(self, "validation_metrics_computer", None)
        if computer is not None:
            # The complex pair goes with the magnitudes so phase metrics are
            # scored on phase (#867). A PhaseBlindInputError is a wiring
            # defect and propagates; the warning below is for metric failures.
            views = MetricViews(
                pred_mag,
                target_mag,
                pred if torch.is_complex(pred) else None,
                clean_norm if torch.is_complex(clean_norm) else None,
            )
            try:
                computed = views.compute(computer)
                scores.update({k: float(v) for k, v in computed.items()})
            except PhaseBlindInputError:
                raise
            except Exception as exc:
                logger.warning(
                    "[VirtualFiducialStrategy] validation_metrics_computer "
                    "failed; only val_psnr is available this step: %s",
                    exc,
                )
        return scores

    def _score_field(self, real_reference: bool = False) -> dict[str, float]:
        """Score the model's estimated displacement against the twin's applied field.

        The DigitalTwinSimulator now exposes the B0 geometric-distortion field it
        applied (``simulator.last_pe_shift_field``, dominant PE pixel-shift), and
        field-tracking models expose their estimate (``generator
        .last_shift_estimate``, the marker-NCC shift in normalised grid units).
        This compares CHARACTERISTIC MAGNITUDES — mean ``|displacement|`` — which
        sidesteps the global-scalar-vs-spatial-field and correction-sign
        ambiguities and answers the falsifiable question the headline qMRI claim
        needs: *did the tracker recover the amount of EPI shift the twin applied?*

        This is a SELF-CONSISTENCY check against the simulator's own field — NOT
        an independent qMRI reference (M4Raw is single-contrast magnitude with no
        real B0). Returns ``{}`` for any arm whose twin applied no displacement
        field or whose model exposes no estimate, so non-EPI arms are unaffected
        (no new YAML knob — it self-gates on the two attributes; pitfall #15).
        """
        twin_shift = getattr(self.simulator, "last_pe_shift_field", None)
        est = getattr(unwrap_model(self.generator_model), "last_shift_estimate", None)
        if twin_shift is None or est is None:
            # Phase-path arms (graph_cut_unwrap / phase_tracking_lstm) have no
            # geometric PE shift; grade their field estimate against the REAL B0's
            # spatial structure instead (unit-safe — see _score_field_structure).
            return self._score_field_structure(real_reference)
        try:
            h_px = int(twin_shift.shape[-2])
            # twin field is in pixels; the marker-NCC estimate is normalised to
            # [-1, 1] grid units (shift_idx-H/2)/H*2 — invert with H/2 to pixels.
            twin_char = twin_shift.abs().flatten(1).mean(dim=1)  # [B] px
            model_char = est.reshape(-1).float().abs() * (h_px / 2.0)  # [B] px
            n = min(int(twin_char.shape[0]), int(model_char.shape[0]))
            diff = model_char[:n] - twin_char[:n]
            out = {
                "val_field_shift_mae": float(diff.abs().mean()),
                "val_field_shift_bias": float(diff.mean()),
                "val_twin_pe_shift_px": float(twin_char[:n].mean()),
                "val_pred_pe_shift_px": float(model_char[:n].mean()),
                # 1.0 when the twin's applied field was a REAL B0 map from the
                # batch (real-reference grading); 0.0 = self-consistency on a
                # synthetic field. Lets a reader tell the two regimes apart.
                "val_field_reference_real": 1.0 if real_reference else 0.0,
            }
            # Full qMRI agreement battery (ICC(3,1) / Bland-Altman LoA / CoV) on
            # the per-sample shift estimates vs the twin's reference shift.
            from spectramr.core.metrics.quantitative.qmri_agreement import (
                qmri_agreement_metrics,
            )

            out.update(
                {
                    f"val_field_{k}": v
                    for k, v in qmri_agreement_metrics(model_char[:n], twin_char[:n]).items()
                }
            )
            return out
        except Exception as exc:  # diagnostics must never break validation
            logger.debug("[VirtualFiducialStrategy] field scoring skipped: %s", exc)
            return {}

    def _score_field_structure(self, real_reference: bool) -> dict[str, float]:
        """Grade a model field estimate against the REAL B0's spatial structure.

        For the phase-domain arms the model emits a field estimate
        (``generator.last_field_estimate``, e.g. unwrapped phase ∝ B0) in units
        that differ from the reference B0 (Hz). Comparing absolute magnitudes
        would be a unit mismatch, so this grades **structure**: the Pearson
        correlation of the spatial patterns and the scale-fit NRMSE (the residual
        after the best least-squares scaling). Self-gates on a REAL reference +
        a model estimate (pitfall #15); returns ``{}`` otherwise.
        """
        b0 = getattr(self.simulator, "last_b0_field", None)
        est = getattr(unwrap_model(self.generator_model), "last_field_estimate", None)
        if not real_reference or b0 is None or est is None:
            return {}
        try:
            from torch.nn.functional import interpolate

            ref = b0.float()
            ref = ref.unsqueeze(1) if ref.dim() == 3 else ref
            e = est.float()
            e = e.unsqueeze(1) if e.dim() == 3 else e
            if e.dim() >= 4 and e.shape[1] > 1:
                e = e.abs().mean(dim=1, keepdim=True)
            if ref.dim() >= 4 and ref.shape[1] > 1:
                ref = ref.abs().mean(dim=1, keepdim=True)
            if e.shape[-2:] != ref.shape[-2:]:
                e = interpolate(e, size=ref.shape[-2:], mode="bilinear", align_corners=False)
            n = min(int(e.shape[0]), int(ref.shape[0]))
            ed = e[:n].reshape(n, -1)
            rd = ref[:n].reshape(n, -1)
            ed = ed - ed.mean(dim=1, keepdim=True)
            rd = rd - rd.mean(dim=1, keepdim=True)
            corr = (ed * rd).sum(1) / (ed.norm(dim=1) * rd.norm(dim=1) + 1e-8)
            scale = (ed * rd).sum(1, keepdim=True) / (ed.pow(2).sum(1, keepdim=True) + 1e-8)
            nrmse = (scale * ed - rd).norm(dim=1) / (rd.norm(dim=1) + 1e-8)
            return {
                "val_field_b0_corr": float(corr.mean()),
                "val_field_b0_nrmse": float(nrmse.mean()),
                "val_field_reference_real": 1.0,
            }
        except Exception as exc:  # diagnostics must never break validation
            logger.debug("[VirtualFiducialStrategy] field-structure scoring skipped: %s", exc)
            return {}


__all__ = ["ConcreteVirtualFiducialStrategy"]
