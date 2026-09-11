"""PINN Sensitivity Training Strategy.

This module provides the training strategy for zero-shot Coil Sensitivity Map estimation
using Physics-Informed Neural Networks (PINN). It encapsulates the complex math of
Coordinate Network (SIREN) optimization and adaptive gradient weighting, cleanly
fitting within the framework's BaseTrainingStrategy.
"""

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image

from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.loop_state import resolve_loop_iteration
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.models.generators.siren_pinn import SirenSensNet, get_last_shared_layer
from spectramr.models.losses.registry import create_loss
from spectramr.shared.utils.safe_io import atomic_save_torch

logger = logging.getLogger(__name__)


class ConcretePINNSensitivityStrategy(BaseTrainingStrategy):
    """Strategy for training PINN for CSM estimation.

    This strategy orchestrates:
    - Coordinate grid generation
    - Full-grid inference for Data Consistency (DC), TV, and Normalization
    - Sub-sampled collocation point inference for PDE Loss
    - Adaptive gradient weighting (lambda tracking) mid-step
    - Sensitivity map saving (magnitude/phase PNGs + raw tensors)
    """

    def __init__(
        self,
        env: TrainingEnvironment,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, device=device, **kwargs)

        self._setup_strategy_specific_components()

    def _setup_strategy_specific_components(self) -> None:
        """Setup specialized PINN components securely."""
        self._verify_strategy_config(expected_modes=("pinn",))

        # Load specialized losses via the registry
        self.pde_criterion = create_loss("helmholtz_pde", k_sq=0.0).to(self.device)
        self.norm_criterion = create_loss("unit_norm_coil").to(self.device)
        self.tv_criterion = create_loss("magnitude_tv").to(self.device)

        # Cache config values — SSOT: read directly from physics.pinn (PINNConfig)
        self.pinn_cfg = self.config.physics.pinn

        # Curriculum and EMA settings (typed fields in PINNConfig)
        self.warmup_epochs = self.pinn_cfg.warmup_epochs
        self.alpha_ema = self.pinn_cfg.alpha_ema
        self.ema_grad_pde = None

        # Loss weights — SSOT: read from config.losses.pinn (PINNLossesConfig)
        # Fail loud (CLAUDE.md #9) when the section is missing, rather
        # than letting ``self.pinn_losses_cfg.lambda_unit_norm_coil`` raise
        # the cryptic ``'NoneType' object has no attribute
        # 'lambda_unit_norm_coil'`` observed 2026-05-14 (audit E21).
        self.pinn_losses_cfg = self.config.losses.pinn
        if self.pinn_losses_cfg is None:
            raise ValueError(
                "[PINN Strategy] config.losses.pinn is None — the PINN "
                "strategy requires loss weights (lambda_unit_norm_coil, "
                "lambda_pde, lambda_magnitude_tv, lambda_pinn_dc) under "
                "``losses.pinn`` in the experiment YAML. Add a "
                "``losses:\\n  pinn:\\n    lambda_unit_norm_coil: ...`` "
                "block."
            )
        self.lambda_norm = self.pinn_losses_cfg.lambda_unit_norm_coil
        self.lambda_pde_max = self.pinn_losses_cfg.lambda_pde
        self.lambda_tv = self.pinn_losses_cfg.lambda_magnitude_tv
        self.lambda_dc = self.pinn_losses_cfg.lambda_pinn_dc

        # Extract dimensions from the generator
        self.model = cast(SirenSensNet, self.generator_model)
        self.num_coils = self.model.num_coils

        # Precompute coordinate grid from the canonical ``data.patch_size``.
        # The legacy ``image_size`` key was popped into ``patch_size`` by the
        # data-schema model-validator (data.py:1547-1549), so the old
        # ``hasattr(self.config.data, "image_size")`` probe was permanently
        # False and the precompute silently fell back to (256, 256). Read the
        # canonical, normalized (W, H, D) tuple instead (audit 2026-06).
        # H, W still self-correct from k-space at runtime (see below).
        self.W, self.H = (
            self.config.data.sampling.patch_size[0],
            self.config.data.sampling.patch_size[1],
        )

        y, x = torch.meshgrid(
            torch.linspace(-1, 1, self.H, device=self.device),
            torch.linspace(-1, 1, self.W, device=self.device),
            indexing="ij",
        )
        self.coords_full = torch.stack([x.flatten(), y.flatten()], dim=-1)

        # Sensitivity map output directory
        output_dir = (
            self.config.training.output_dir
            or self.config.loss_logging.output_dir
            or "experiments/results/pinn_csm"
        )
        self._csm_output_dir = Path(output_dir) / "sensitivity_maps"
        self._csm_output_dir.mkdir(parents=True, exist_ok=True)

        # Configurable save interval (typed field in PINNConfig)
        self._csm_save_interval = self.pinn_cfg.map_save_interval

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute PINN losses including adaptive PDE scaling.

        Note: For PINN, input_batch represents the undersampled k-space data y,
        Target is unused.
        """
        batch = kwargs.get("batch", {})
        kspace_y = batch.get("measured_kspace", input_batch)
        mask = batch.get("mask", None)

        # Per CLAUDE.md #2 (no raw torch.fft.* outside fft_ops), centered
        # FFT/iFFT go through fft2c/ifft2c. They handle fftshift, ortho
        # normalization, AMP-fp32 wrapping, and complex coercion in one
        # place. See TODO/audit/06_strategies_specialised.md F10.
        from spectramr.infrastructure.physics.fft_ops import _to_complex, fft2c, ifft2c

        # Convert potentially real-stacked k-space (e.g. [8, 256, 256]) to complex (e.g. [4, 256, 256])
        kspace_y_complex = _to_complex(kspace_y)

        # Collapse the leading batch dim to obtain [Coils, H, W]. k-space is
        # [B, Coils, H, W] (loss_computation_helpers.py:93 contract); for any
        # B >= 1 we select the FIRST sample on the LEADING axis. The previous
        # ``squeeze(0)`` only handled B == 1 and the trailing ``[..., 0]`` loop
        # collapsed W (not the batch dim) for B > 1, corrupting the spatial
        # reshape into ``(num_coils, H, W)`` (audit 2026-06).
        while kspace_y_complex.dim() > 3:
            kspace_y_complex = kspace_y_complex[0]

        if mask is None:
            mask_t = torch.ones_like(kspace_y_complex[0])
        else:
            mask_t = mask.squeeze()
            while mask_t.dim() > 2:
                mask_t = mask_t[0]

        # -----------------------------------------------------------------
        # TOPOLOGICAL FIX: Absolute Data Normalization
        # -----------------------------------------------------------------
        acs_img_raw = ifft2c(kspace_y_complex)
        rsos_raw = torch.sqrt(torch.sum(torch.abs(acs_img_raw) ** 2, dim=0, keepdim=True))
        scale_factor = torch.max(rsos_raw)

        kspace_y_norm = kspace_y_complex / scale_factor
        m_hat_norm = rsos_raw / scale_factor
        m_hat_norm = m_hat_norm.squeeze(0)

        # Ensure coordinate grid matches input spatial dimensions dynamically
        current_H, current_W = kspace_y_complex.shape[-2:]
        if current_H != self.H or current_W != self.W:
            self.H, self.W = current_H, current_W
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, self.H, device=self.device),
                torch.linspace(-1, 1, self.W, device=self.device),
                indexing="ij",
            )
            self.coords_full = torch.stack([x.flatten(), y.flatten()], dim=-1)

        # 1. Compute Data Consistency (DC) Loss (Full Grid, 1st order graph)
        # -----------------------------------------------------------------
        coords_full = self.coords_full.clone().detach()  # No requires_grad for DC
        S_r_full, S_i_full = self.model(coords_full)
        S_c = torch.complex(S_r_full, S_i_full)

        img_predicted = (S_c * m_hat_norm.flatten().view(-1, 1)).T.view(
            self.num_coils, self.H, self.W
        )

        # Centered 2D FFT — fft2c handles shifts and ortho norm.
        k_predicted = fft2c(img_predicted)

        masked_pred = k_predicted * mask_t

        dc_mse = torch.mean(torch.view_as_real(masked_pred - kspace_y_norm) ** 2)
        dc_loss = self.lambda_dc * dc_mse

        # 2. Auxiliary Physics Priors
        # -----------------------------------------------------------------
        norm_loss = self.norm_criterion(S_r_full, S_i_full)

        tv_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_tv > 0:
            tv_loss = self.tv_criterion(S_r_full, S_i_full, shape=(self.H, self.W))

        # 3. Total Loss Composition & Curriculum
        # -----------------------------------------------------------------
        if epoch < self.warmup_epochs:
            # Phase 1: Pure Data Anchoring (Dirichlet Boundary Setup)
            pde_loss = torch.tensor(0.0, device=self.device)
            lambda_pde_clamped = torch.tensor(0.0, device=self.device)
            total_loss = dc_loss + self.lambda_norm * norm_loss + self.lambda_tv * tv_loss
        else:
            # Phase 2: Physics-Informed Extrapolation
            # Randomly sample collocation points to prevent CUDA OOM on the Laplacian
            num_points = self.pinn_cfg.collocation_points
            idx = torch.randperm(self.H * self.W, device=self.device)[:num_points]
            coords_batch = self.coords_full[idx].clone().detach().requires_grad_(True)

            S_r_batch, S_i_batch = self.model(coords_batch)
            pde_loss = self.pde_criterion(S_r_batch, S_i_batch, coords_batch)

            # Adaptive Gradient Weighting (Wang et al. ICML 2021) with EMA
            shared_weights = get_last_shared_layer(self.model)

            grad_dc = torch.autograd.grad(dc_loss, shared_weights, retain_graph=True)[0]
            grad_pde = torch.autograd.grad(pde_loss, shared_weights, retain_graph=True)[0]

            max_grad_dc = torch.max(torch.abs(grad_dc)).detach()
            mean_grad_pde = torch.mean(torch.abs(grad_pde)).detach()

            # EMA Update to stabilize denominator
            if self.ema_grad_pde is None:
                self.ema_grad_pde = mean_grad_pde
            else:
                self.ema_grad_pde = (
                    self.alpha_ema * self.ema_grad_pde + (1 - self.alpha_ema) * mean_grad_pde
                )

            lambda_pde_val = max_grad_dc / (self.ema_grad_pde + 1e-8)
            lambda_pde_clamped = torch.clamp(lambda_pde_val, max=self.lambda_pde_max)

            total_loss = (
                dc_loss
                + lambda_pde_clamped.detach() * pde_loss
                + self.lambda_norm * norm_loss
                + self.lambda_tv * tv_loss
            )

        # Build loss dictionary for logging (detach components)
        losses = {
            "g_total_loss": total_loss,
            "dc_loss": dc_loss,
            "pde_loss": pde_loss,
            "norm_loss": norm_loss,
            "tv_loss": tv_loss,
            "lambda_pde": lambda_pde_clamped.detach(),
        }

        return losses

    # ------------------------------------------------------------------
    # Sensitivity Map Saving
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _save_sensitivity_maps(self, epoch: int, step: int = 0) -> None:
        """Infer the full sensitivity field and log to TensorBoard + save PNGs.

        Produces for each coil ``c``:

        **TensorBoard** (via ``self.logging_service.log_images_batch``):
        - ``csm/magnitude_coil{c}``  — per-coil |S_c(x,y)|  (grayscale)
        - ``csm/phase_coil{c}``      — per-coil ∠S_c(x,y) (HSV colormap → RGB)
        - ``csm/rss_combined``        — root-sum-of-squares overview

        **Filesystem** (via per-coil PNGs + raw ``.pt`` tensor):
        - ``sensitivity_maps/epoch{N}_step{S}/magnitude_coil{c}.png``
        - ``sensitivity_maps/epoch{N}_step{S}/phase_coil{c}.png``
        - ``sensitivity_maps/epoch{N}_step{S}/rss_combined.png``
        - ``sensitivity_maps/sensitivity_maps_epoch{N}_step{S}.pt``

        Args:
            epoch: Current epoch number (used in filenames and TB step).
            step: Current global step (used in filenames and TB step).
        """
        self.model.eval()

        coords = self.coords_full.clone().detach()
        S_r, S_i = self.model(coords)  # (H*W, num_coils) each

        # Reshape to spatial maps: (num_coils, H, W)
        S_r_map = S_r.view(self.H, self.W, self.num_coils).permute(2, 0, 1)
        S_i_map = S_i.view(self.H, self.W, self.num_coils).permute(2, 0, 1)

        # Complex tensor for raw export
        S_complex = torch.complex(S_r_map, S_i_map)  # (num_coils, H, W)

        magnitude = S_complex.abs()  # (num_coils, H, W)
        phase = S_complex.angle()  # (num_coils, H, W)

        # RSS combined magnitude for overview
        rss = torch.sqrt(torch.sum(magnitude**2, dim=0))  # (H, W)

        # --- 1. TensorBoard Logging (experiment 11 pattern) ---
        self._log_maps_to_tensorboard(magnitude, phase, rss, step)

        # --- 2. Save raw tensor ---
        pt_path = self._csm_output_dir / f"sensitivity_maps_epoch{epoch:04d}_step{step:06d}.pt"
        atomic_save_torch(
            {
                "complex": S_complex.cpu(),
                "magnitude": magnitude.cpu(),
                "phase": phase.cpu(),
            },
            pt_path,
        )
        logger.info("[PINN-CSM] Saved raw tensor: %s", pt_path)

        # --- 3. Save per-coil PNGs ---
        epoch_dir = self._csm_output_dir / f"epoch{epoch:04d}_step{step:06d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        for c in range(self.num_coils):
            self._save_map_as_png(
                magnitude[c],
                epoch_dir / f"magnitude_coil{c}.png",
            )
            self._save_map_as_png(
                phase[c],
                epoch_dir / f"phase_coil{c}.png",
                colormap="hsv",
            )

        # RSS overview
        self._save_map_as_png(rss, epoch_dir / "rss_combined.png")

        logger.info("[PINN-CSM] Saved %d coil maps + RSS → %s", self.num_coils, epoch_dir)

        self.model.train()

    def _log_maps_to_tensorboard(
        self,
        magnitude: torch.Tensor,
        phase: torch.Tensor,
        rss: torch.Tensor,
        step: int,
    ) -> None:
        """Log sensitivity maps to TensorBoard via logging_service.

        Follows the same ``log_images_batch({tag: tensor}, step)`` pattern
        used by the diffusion strategy in experiment 11.

        Args:
            magnitude: Per-coil magnitude maps ``(num_coils, H, W)``.
            phase: Per-coil phase maps ``(num_coils, H, W)``  in ``[-π, π]``.
            rss: Root-sum-of-squares combined magnitude ``(H, W)``.
            step: Global training step for TensorBoard x-axis.
        """
        if not hasattr(self, "logging_service") or self.logging_service is None:
            return

        images_dict: dict[str, torch.Tensor] = {}

        # Per-coil magnitude (grayscale → 1-channel, batch dim required)
        for c in range(self.num_coils):
            mag_norm = self._normalize_for_display(magnitude[c])
            # TensorBoard expects (N, C, H, W)
            images_dict[f"csm/magnitude_coil{c}"] = mag_norm.unsqueeze(0).unsqueeze(0)

        # Per-coil phase (HSV colormap → 3-channel RGB)
        for c in range(self.num_coils):
            phase_rgb = self._phase_to_rgb(phase[c])  # (3, H, W)
            images_dict[f"csm/phase_coil{c}"] = phase_rgb.unsqueeze(0)

        # RSS combined (grayscale)
        rss_norm = self._normalize_for_display(rss)
        images_dict["csm/rss_combined"] = rss_norm.unsqueeze(0).unsqueeze(0)

        try:
            self.logging_service.log_images_batch(images_dict, step)
            self.logging_service.log_info(
                f"[PINN-CSM] Logged {len(images_dict)} maps to TensorBoard at step {step}"
            )
        except (OSError, RuntimeError) as e:
            # Narrowed from a blanket ``except Exception`` (audit 2026-06,
            # CLAUDE.md pitfalls #9/#10). Only the logging backend's IO /
            # runtime failures (disk full, TB writer error) are downgraded to
            # a warning; unexpected programming errors (shape/type bugs in the
            # image tensors) now propagate instead of being silently masked.
            self.logging_service.log_warning(f"[PINN-CSM] TensorBoard image logging failed: {e}")

    @staticmethod
    def _normalize_for_display(tensor: torch.Tensor) -> torch.Tensor:
        """Min-max normalize a 2-D tensor to [0, 1] for TensorBoard display.

        Args:
            tensor: 2-D ``(H, W)`` tensor.

        Returns:
            Normalized tensor in ``[0, 1]``.
        """
        t = tensor.detach().float()
        vmin, vmax = t.min(), t.max()
        if (vmax - vmin) > 1e-8:
            return (t - vmin) / (vmax - vmin)
        return torch.zeros_like(t)

    @staticmethod
    def _phase_to_rgb(phase_map: torch.Tensor) -> torch.Tensor:
        """Convert a phase map in [-π, π] to an HSV-colormapped RGB tensor.

        This makes cyclic phase structure visually interpretable in
        TensorBoard (matching the HSV PNGs saved to disk).

        Args:
            phase_map: 2-D ``(H, W)`` tensor in ``[-π, π]``.

        Returns:
            RGB tensor ``(3, H, W)`` in ``[0, 1]``.
        """
        # Normalize [-π, π] → [0, 1]
        normalized = (phase_map.detach().float() + torch.pi) / (2 * torch.pi)
        normalized = normalized.clamp(0.0, 1.0)

        try:
            import matplotlib.cm as cm

            arr = normalized.cpu().numpy()
            rgba = cm.hsv(arr)  # (H, W, 4)
            rgb = torch.from_numpy(rgba[:, :, :3]).permute(2, 0, 1).float()
        except ImportError:
            # Fallback: grayscale
            rgb = normalized.unsqueeze(0).expand(3, -1, -1).cpu()

        return rgb

    @staticmethod
    def _save_map_as_png(
        tensor: torch.Tensor,
        path: Path,
        colormap: str | None = None,
    ) -> None:
        """Save a 2-D tensor as a grayscale or colormapped PNG.

        Args:
            tensor: 2-D (H, W) tensor.
            path: Output file path.
            colormap: If ``"hsv"``, apply matplotlib HSV colormap (useful for
                phase maps).  ``None`` → grayscale.
        """
        arr = tensor.detach().cpu().float().numpy()

        # Per-image min-max → [0, 1]
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax - vmin > 1e-8:
            arr = (arr - vmin) / (vmax - vmin)
        else:
            arr = np.zeros_like(arr)

        if colormap == "hsv":
            try:
                import matplotlib.cm as cm

                mapped = cm.hsv(arr)  # (H, W, 4) RGBA float in [0, 1]
                img = Image.fromarray((mapped[:, :, :3] * 255).astype(np.uint8))
            except ImportError:
                img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        else:
            img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")

        img.save(path)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Save sensitivity maps at the configured interval.

        Called by the training loop at the end of each epoch.

        Args:
            epoch: The epoch that just finished.
            metrics: Dictionary of aggregated metrics for the epoch.
        """
        super().on_epoch_end(epoch, metrics)

        if epoch % self._csm_save_interval == 0 or epoch == 0:
            # Live iteration (loop_state seam): the frozen ``self.env.step``
            # labelled every saved sensitivity map "step 0", so successive
            # epochs wrote indistinguishable filenames (pitfall #16).
            step = resolve_loop_iteration(self)
            self._save_sensitivity_maps(epoch=epoch, step=step)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validate a single batch evaluating the unsupervised physics losses."""
        val_batch = (input_batch, target_batch)
        batch_idx = kwargs.get("batch_idx", 0)
        # The PINN evaluates PDE losses via autograd, so we MUST enable gradients
        # even though the pipeline calls this inside a torch.no_grad() context.
        with torch.enable_grad():
            input_batch, target_batch = self._unpack_batch(val_batch)

            self.env.generator.eval()

            # Formally calculate losses. _compute_losses_impl handles coordinate mesh generation
            # and setting requires_grad=True for the PDE computations.
            #
            # #1190: calling ``_compute_losses_impl`` directly bypasses the
            # ``_compute_losses`` wrapper that emits the ``model_output``
            # snapshot, so arm it here. ``snapshot_source("val")`` is the OUTER
            # manager so the phase is still "val" when the emit runs on exit --
            # without it the record would claim the training data chain.
            with (
                self.snapshot_source("val"),
                self._capture_model_output(
                    module=self.generator_model,
                    input_batch=input_batch,
                    target_batch=target_batch,
                    step=batch_idx,
                ),
            ):
                loss_tensors = self._compute_losses_impl(
                    input_batch=input_batch,
                    target_batch=target_batch,
                    epoch=0,
                    batch=val_batch,
                )

            # Convert tensor dictionary to scalar floats with 'val_' prefix
            metrics = {f"val_{k}": v.item() for k, v in loss_tensors.items()}

            self.env.generator.train()

            return metrics
