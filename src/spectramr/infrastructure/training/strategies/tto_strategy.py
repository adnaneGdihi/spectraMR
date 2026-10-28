"""Test-Time Optimization (TTO) Strategy.

Stage 2 of the Virtual Fiducial + HyperMamba motion correction pipeline.
Freezes all network weights and optimizes only the motion trajectory θ̂(t)
to minimize Data Consistency against real M4Raw scanner measurements.

This is a zero-shot approach — no paired ground truth is needed.

Training Loop:
    1. Initialize θ̂ = 0 (no motion assumed)
    2. Generate probe: P_θ̂ = H_θ̂(M)
    3. Forward: x̂ = HyperMambaUNet(y_real, P_θ̂)
    4. DC loss: ||y_real - H_θ̂(x̂)||² + λ_TV ||∇x̂||₁
    5. Backpropagate ∂L/∂θ̂, update only θ̂

Reference:
    Batchelor et al., "Matrix description of general motion correction," MRM 2005.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from spectramr.infrastructure.physics.kinematic_forward import KinematicForwardOperator
from spectramr.infrastructure.physics.virtual_fiducial import (
    MotionTrajectory,
    VirtualFiducial,
)
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

logger = logging.getLogger(__name__)


class ConcreteTTOStrategy(BaseTrainingStrategy):
    """Zero-shot Test-Time Optimization strategy for motion correction.

    Freezes the HyperMambaUNet and optimizes only the motion trajectory θ̂(t)
    to invert real patient motion from raw M4Raw scanner data.

    Pattern: follows ConcretePINNSensitivityStrategy (zero-shot per-subject optimization).

    Attributes:
        kinematic_op: Differentiable motion forward operator.
        fiducial: Virtual Fiducial (frozen).
        motion_traj: Learnable motion trajectory (the ONLY optimized parameters).
    """

    # parallel_compat: Owns the motion trajectory and freezes the wrapped generator.
    parallel_compat = frozenset()

    def __init__(
        self,
        env: Any,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize TTO Strategy.

        Args:
            env: TrainingEnvironment from the builder.
            device: Target device.
            **kwargs: Additional kwargs.
        """
        super().__init__(env=env, device=device, **kwargs)
        # No second ``self._setup_strategy_specific_components()``: the base __init__
        # already runs it, and it is not idempotent once it adds optimizer param groups.

    def _setup_strategy_specific_components(self) -> None:
        """Initialize physics operators and freeze network weights."""
        _ps = self.config.data.sampling.patch_size
        _is = getattr(self.config.data, "image_size", None)
        if _ps and len(_ps) >= 2:
            img_size = (int(_ps[0]), int(_ps[1]))
        elif _is and len(_is) >= 2:
            img_size = (int(_is[0]), int(_is[1]))
        else:
            img_size = (256, 256)
        num_lines = img_size[0]

        # Physics operators
        self.kinematic_op = KinematicForwardOperator(
            im_size=img_size,
        ).to(self.device)

        self.fiducial = VirtualFiducial(
            im_size=img_size,
            grid_spacing=16,
            sigma=2.0,
            learnable=False,
        ).to(self.device)

        # Learnable motion trajectory — the ONLY optimized parameters
        self.motion_traj = MotionTrajectory(
            num_readout_lines=num_lines,
            init_mode="zero",
        ).to(self.device)

        # Freeze ALL network weights — only θ̂ is updated
        gen = self.generator_model
        if hasattr(gen, "freeze_backbone"):
            gen.freeze_backbone()
        for param in gen.parameters():
            param.requires_grad = False

        # TTO optimizes ONLY θ̂ (motion_traj); the generator was just frozen
        # above (requires_grad=False). Register motion_traj on the base loop's
        # opt_g and let it step — the frozen generator params are no-ops. The
        # previous self._tto_optimizer was never stepped by the base loop
        # (dead code), so θ̂ never actually trained.
        opt = getattr(self.env, "opt_g", None)
        if opt is not None:
            existing = {p for g in opt.param_groups for p in g["params"]}
            new_params = [p for p in self.motion_traj.parameters() if p not in existing]
            if new_params:
                tto_lr = 1e-3
                opt.add_param_group({"params": new_params, "lr": tto_lr})
                # opt_g's scheduler was built before this added group; re-sync
                # its base_lrs so scheduler.step() does not raise a zip()
                # length mismatch at iter ~T_0 (smoke audit 2026-06-03, F7d).
                self._sync_schedulers_after_param_group_addition(tto_lr)

        # SSOT: read TTO regularization weights from config.training.tto.
        # The base schema types ``tto`` as ``Any`` (to avoid a
        # circular import with src/config/schemas/training/tto.py),
        # so coerce here to the strongly-typed ``TTOConfig`` before
        # attribute access. Accepts dict-form (from YAML through the
        # base schema's ``extra="allow"`` path), an already-typed
        # ``TTOConfig`` instance, or ``None`` (in which case the
        # strategy falls back to declared defaults).
        from spectramr.config.schemas.training.tto import TTOConfig as _TTOConfig

        _tto_raw = getattr(self.config.training, "tto", None)
        if _tto_raw is None:
            _tto = _TTOConfig()
        elif isinstance(_tto_raw, _TTOConfig):
            _tto = _tto_raw
        elif isinstance(_tto_raw, dict):
            _tto = _TTOConfig(**_tto_raw)
        else:
            # Some other Pydantic model with the right shape — duck-type
            # via model_dump if available; otherwise pass through.
            try:
                _tto = _TTOConfig(**_tto_raw.model_dump())
            except Exception as _exc:
                raise TypeError(
                    f"[TTO Strategy] config.training.tto has unsupported type "
                    f"{type(_tto_raw).__name__}; expected dict, TTOConfig, or "
                    f"None. Underlying error: {_exc}"
                ) from _exc

        self._lambda_tv = _tto.lambda_tv
        self._lambda_dc = _tto.lambda_dc

        logger.info(
            "[TTO Strategy] Setup complete: im_size=%s, lines=%d, λ_dc=%.3f, λ_tv=%.3f",
            img_size,
            num_lines,
            self._lambda_dc,
            self._lambda_tv,
        )

    @staticmethod
    def _normalized_dc_loss(
        predicted_kspace: torch.Tensor, measured_complex: torch.Tensor
    ) -> torch.Tensor:
        r"""Scale-invariant data-consistency residual.

        Raw scanner k-space has arbitrary scale (M4Raw magnitudes are
        ~1e3+), so the unnormalized :math:`\|y - H_{\hat\theta}(\hat x)\|^2`
        is O(1e3-1e4) and the trajectory optimiser diverges (smoke
        2026-05-25: ``loss_dc=7673`` rising → ``gradient_explosion`` despite
        gradient clipping at norm 5.0). Dividing by the measurement energy
        makes the loss O(1) and scale-invariant — the relative residual is
        the quantity the trajectory :math:`\theta` should minimise
        regardless of the raw k-space scale.
        """
        dc_error = predicted_kspace - measured_complex
        measurement_energy = torch.mean(torch.abs(measured_complex) ** 2).clamp(min=1e-8)
        return torch.mean(torch.abs(dc_error) ** 2) / measurement_energy

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute TTO losses (DC + TV) optimizing only θ̂.

        Steps:
            1. Get current motion estimate θ̂
            2. Corrupt fiducial with θ̂ → probe image for bridge
            3. Forward through frozen HyperMambaUNet → x̂ (reconstructed image)
            4. Apply forward model H_θ̂(x̂) → predicted k-space
            5. DC loss: ||y_real - H_θ̂(x̂)||²
            6. TV regularization on x̂

        Args:
            input_batch: Real corrupted k-space from scanner [B, C, H, W].
            target_batch: Same as input_batch (no ground truth in TTO).
            epoch: TTO iteration index.
            **kwargs: Additional kwargs.

        Returns:
            Loss dict with 'g_total_loss'.
        """
        B = input_batch.shape[0]

        # 1. Current motion estimate
        theta_hat = self.motion_traj(batch_size=B)

        # 2. Corrupt fiducial with estimated motion
        fiducial_image = self.fiducial(batch_size=B).to(self.device)
        corrupted_fiducial_kspace = self.kinematic_op(fiducial_image, theta_hat)
        corrupted_fiducial_real = torch.stack(
            [corrupted_fiducial_kspace.real, corrupted_fiducial_kspace.imag],
            dim=1,
        ).squeeze(2)

        # 3. Forward through the (weight-frozen) HyperMambaUNet. The backbone
        # parameters have requires_grad=False, but grads still flow through the
        # theta_hat → fiducial → bridge path, so the forward must NOT be wrapped
        # in torch.no_grad() — otherwise θ̂ would receive no gradient.
        gen = self.generator_model
        pred = gen(
            input_batch,
            corrupted_fiducial=corrupted_fiducial_real,
        )

        # 4. Convert prediction to complex for forward model
        if not torch.is_complex(pred):
            C = pred.shape[1]
            if C >= 2 and C % 2 == 0:
                pred_complex = torch.complex(pred[:, 0::2, :, :], pred[:, 1::2, :, :])
            else:
                pred_complex = pred.to(torch.complex64)
        else:
            pred_complex = pred

        # 5. Apply forward model with estimated motion: H_θ̂(x̂)
        predicted_kspace = self.kinematic_op(pred_complex, theta_hat)

        # Convert input to complex for comparison
        if not torch.is_complex(input_batch):
            C_in = input_batch.shape[1]
            if C_in >= 2 and C_in % 2 == 0:
                measured_complex = torch.complex(
                    input_batch[:, 0::2, :, :], input_batch[:, 1::2, :, :]
                )
            else:
                measured_complex = input_batch.to(torch.complex64)
        else:
            measured_complex = input_batch

        # 6. DC loss: normalized data-consistency residual (see helper).
        loss_dc = self._normalized_dc_loss(predicted_kspace, measured_complex)

        # 7. TV regularization: ||∇x̂||₁
        if pred.shape[-1] > 1 and pred.shape[-2] > 1:
            tv_h = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]).mean()
            tv_w = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]).mean()
            loss_tv = tv_h + tv_w
        else:
            loss_tv = torch.tensor(0.0, device=self.device)

        g_total_loss = self._lambda_dc * loss_dc + self._lambda_tv * loss_tv

        # NOTE: do NOT emit ``max_translation`` / ``max_rotation_deg`` here.
        # Those motion diagnostics read ``MotionTrajectory`` properties that are
        # ``.item()``-backed (D2H GPU host-syncs) — building them every gen step
        # in the hot loop violates the training-loop perf rule (no .item() in
        # the loop). They are already reported once per epoch in
        # ``validation_step`` (max_translation_px / max_rotation_deg), which is
        # outside the hot path.
        return {
            "g_total_loss": g_total_loss,
            "loss_dc": loss_dc.detach(),
            "loss_tv": loss_tv.detach(),
        }

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """TTO validation: report motion parameters and DC-based metrics.

        Args:
            val_batch: Validation batch.
            batch_idx: Batch index.

        Returns:
            Metrics dict including val_psnr proxy.
        """
        val_batch = (input_batch, target_batch)
        batch_idx = kwargs.get("batch_idx", 0)
        # Compute DC loss as a PSNR proxy (lower DC → higher quality)
        if isinstance(val_batch, (list, tuple)):
            input_data = val_batch[0]
        elif isinstance(val_batch, dict):
            input_data = val_batch.get("input", val_batch.get("lr"))
        elif hasattr(val_batch, "input"):
            input_data = val_batch.input
        else:
            input_data = val_batch

        input_data = self._to_device(input_data)

        # Handle 5D TorchIO volumes
        if input_data.ndim == 5:
            mid = input_data.shape[-1] // 2
            input_data = input_data[..., mid]

        dc_psnr = 0.0
        with torch.no_grad():
            # #1190: calling ``_compute_losses_impl`` directly bypasses the
            # ``_compute_losses`` wrapper that emits the ``model_output``
            # snapshot, so arm it here. ``snapshot_source("val")`` is the OUTER
            # manager so the phase is still "val" when the emit runs on exit --
            # without it the record would claim the training data chain.
            with (
                self.snapshot_source("val"),
                self._capture_model_output(
                    module=self.generator_model,
                    input_batch=input_data,
                    target_batch=input_data,
                    step=batch_idx,
                ),
            ):
                losses = self._compute_losses_impl(input_data, input_data, epoch=0)
            # Use negative DC loss as PSNR proxy (monotonically increasing with quality)
            dc_loss = float(losses.get("loss_dc", torch.tensor(1.0)))
            dc_psnr = -10.0 * torch.log10(torch.tensor(dc_loss + 1e-10)).item()

        return {
            "max_translation_px": self.motion_traj.max_translation,
            "max_rotation_deg": self.motion_traj.max_rotation_degrees,
            "val_psnr": dc_psnr,
        }


__all__ = ["ConcreteTTOStrategy"]
