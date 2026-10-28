"""Coil Sensitivity Estimation Service

This service provides coil sensitivity map estimation for MRI reconstruction.
Supports ESPIRiT and low-rank methods with optional caching.

Author: spectraMR Research Team
"""

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from spectramr.infrastructure.physics.fft_ops import ifft2c
from spectramr.infrastructure.services import ILoggingService
from spectramr.shared.utils.safe_io import atomic_save_torch, safe_torch_load


@dataclass
class CoilSensitivityResult:
    """Container for coil sensitivity estimation results."""

    sensitivity_maps: torch.Tensor  # Shape: [n_coils, height, width]
    method: str  # 'espirit', 'low_rank', 'fallback'
    snr_estimate: float | None = None
    computation_time: float = 0.0
    cached: bool = False


class CoilSensitivityEstimationService:
    r"""Service for estimating coil sensitivity maps from multi-coil k-space data.

    Supports:
    - ESPIRiT: Eigenvalue-based coil sensitivity estimation
    - Low-rank: SVD-based low-rank approximation
    - Fallback: Simple RSS normalization

    ### Estimation Logic
    ```mermaid
    graph TD
        A[Input Multi-Coil K-Space] --> B{Coil Count?}
        B -- ">= 8" --> C[Low-Rank SVD]
        B -- "4 to 7" --> D[ESPIRiT Calib]
        B -- "< 4" --> E[RSS Fallback]
        C --> F[Normalization]
        D --> F
        E --> F
        F --> G[Sensitivity Maps]
    ```

    ### ESPIRiT Mathematical Basis
    The ESPIRiT method solves for maps $S$ by finding the primary eigenvector of the calibration matrix in the hankel domain:
    $$\mathcal{H}(k) \mathbf{v} = \lambda \mathbf{v}$$
    Where $k$ is the calibration region and $\mathbf{v}$ are the singular vectors.
    """

    def __init__(
        self,
        logging_service: ILoggingService | None,
        cache_dir: str | None = None,
        enable_cache: bool = True,
    ):
        """Initialize the coil sensitivity estimation service.

        Args:
            logging_service: Optional logging service for output
            cache_dir: Directory for caching sensitivity maps
            enable_cache: Whether to enable caching

        """
        # TODO(coil): remove after migrating callers onto estimate_smaps.
        warnings.warn(
            "CoilSensitivityEstimationService (vocab espirit/low_rank/fallback/"
            "auto) is deprecated and scheduled for removal; use the unified "
            "physics.coil_processing.estimation block + "
            "spectramr.infrastructure.physics.coil_sensitivity.estimate_smaps "
            "(vocab none/power_iter/espirit/pinn/rss/file) as the SSOT.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.logging_service = logging_service
        self.enable_cache = enable_cache

        if cache_dir is None:
            cache_dir = Path("cache/coil_sensitivity")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def estimate_sensitivity_maps(
        self,
        kspace_data: torch.Tensor,
        method: str = "auto",
        calibration_size: int = 24,
        threshold: float = 0.05,
        max_rank: int | None = None,
    ) -> CoilSensitivityResult:
        """Estimate coil sensitivity maps from k-space data.

        Args:
            kspace_data: Complex k-space data, shape [n_coils, height, width]
            method: Estimation method ('espirit', 'low_rank', 'fallback', 'auto')
            calibration_size: Size of calibration region for ESPIRiT
            threshold: Threshold for ESPIRiT eigenvalue filtering
            max_rank: Maximum rank for low-rank approximation

        Returns:
            CoilSensitivityResult containing sensitivity maps and metadata

        """
        import time

        start_time = time.time()

        if kspace_data.ndim == 2:
            kspace_data = kspace_data.unsqueeze(0)
        elif kspace_data.ndim < 3:
            raise ValueError(
                "kspace_data must have shape [coils, height, width] or "
                "[height, width] for single-coil acquisitions",
            )

        if kspace_data.ndim > 3:
            height, width = kspace_data.shape[-2:]
            leading = 1
            for dim_size in kspace_data.shape[:-2]:
                leading *= int(dim_size)
            kspace_data = kspace_data.contiguous().view(leading, height, width)

        # Generate cache key
        cache_key = self._generate_cache_key(
            kspace_data,
            method,
            calibration_size,
            threshold,
            max_rank,
        )

        # Check cache first
        if self.enable_cache:
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                cached_result.cached = True
                return cached_result

        # Auto-select method if requested
        if method == "auto":
            method = self._auto_select_method(kspace_data)

        # Estimate sensitivity maps
        try:
            if method == "espirit":
                sensitivity_maps = self._estimate_espirit(
                    kspace_data,
                    calibration_size,
                    threshold,
                )
            elif method == "low_rank":
                sensitivity_maps = self._estimate_low_rank(kspace_data, max_rank)
            else:  # fallback
                sensitivity_maps = self._estimate_fallback(kspace_data)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if self.logging_service:
                    self.logging_service.log_warning(
                        "GPU out of memory during sensitivity estimation; falling back to CPU or simpler method"
                    )
                # Try on CPU
                kspace_cpu = kspace_data.cpu()
                if method == "espirit":
                    sensitivity_maps = self._estimate_espirit(
                        kspace_cpu,
                        calibration_size,
                        threshold,
                    )
                elif method == "low_rank":
                    sensitivity_maps = self._estimate_low_rank(kspace_cpu, max_rank)
                else:
                    sensitivity_maps = self._estimate_fallback(kspace_cpu)
                sensitivity_maps = sensitivity_maps.to(kspace_data.device)
            else:
                raise  # Re-raise other RuntimeErrors

        # Normalize sensitivity maps
        sensitivity_maps = self._normalize_sensitivity_maps(sensitivity_maps)

        # Estimate SNR if possible
        snr_estimate = self._estimate_snr(kspace_data, sensitivity_maps)

        computation_time = time.time() - start_time

        result = CoilSensitivityResult(
            sensitivity_maps=sensitivity_maps,
            method=method,
            snr_estimate=snr_estimate,
            computation_time=computation_time,
            cached=False,
        )

        # Cache result
        if self.enable_cache:
            self._save_to_cache(cache_key, result)

        snr_str = f"{snr_estimate:.2f}" if snr_estimate is not None else "N/A"
        if self.logging_service:
            self.logging_service.log_info(
                f"Estimated coil sensitivity maps using {method} method "
                f"(SNR: {snr_str}, time: {computation_time:.3f}s)",
            )

        return result

    def _auto_select_method(self, kspace_data: torch.Tensor) -> str:
        """Auto-select the best estimation method based on data characteristics."""
        n_coils = kspace_data.shape[0]

        # For high coil counts, prefer low-rank
        if n_coils >= 8:
            return "low_rank"

        # For moderate coil counts, try ESPIRiT
        if n_coils >= 4:
            return "espirit"

        # For low coil counts, use fallback
        return "fallback"

    def _estimate_espirit(
        self,
        kspace_data: torch.Tensor,
        calibration_size: int,
        threshold: float,
    ) -> torch.Tensor:
        """Estimate sensitivity maps using ESPIRiT method.

        Based on Uecker et al., "ESPIRiT—an eigenvalue approach to
        autocalibrating parallel imaging"
        """
        # Try PyTorch version first for GPU acceleration
        use_pytorch = False
        sp_pytorch = None
        sp = None

        try:
            import sigpy.pytorch as sp_pytorch

            # Check if ESPIRiT is available in PyTorch version
            _ = sp_pytorch.mri.app.EspiritCalib
            use_pytorch = True
        except (ImportError, AttributeError):
            # PyTorch version not available or ESPIRiT not supported
            try:
                import sigpy as sp
            except ImportError:
                if self.logging_service:
                    self.logging_service.log_warning(
                        "sigpy not available, falling back to low-rank estimation",
                    )
                return self._estimate_low_rank(kspace_data)

        if not use_pytorch:
            # Ensure we have numpy sigpy for fallback
            if sp is None:
                try:
                    import sigpy as sp
                except ImportError:
                    if self.logging_service:
                        self.logging_service.log_warning(
                            "sigpy not available, falling back to low-rank estimation",
                        )
                    return self._estimate_low_rank(kspace_data)

            try:
                # Try the documented API (sigpy.mri.app.EspiritCalib)
                espirit_class = sp.mri.app.EspiritCalib
            except AttributeError:
                try:
                    # Fallback to older API if available
                    espirit_class = sp.app.EspiritCalib
                except AttributeError:
                    if self.logging_service:
                        self.logging_service.log_warning(
                            "ESPIRiT not available in this sigpy version, "
                            "falling back to low-rank estimation",
                        )
                    return self._estimate_low_rank(kspace_data)
        else:
            # Use PyTorch version
            espirit_class = sp_pytorch.mri.app.EspiritCalib

        if use_pytorch:
            # Use PyTorch tensors directly for GPU acceleration
            sensitivity_maps = espirit_class(
                ksp=kspace_data, calib_width=calibration_size, thresh=threshold
            ).run()
            return sensitivity_maps
        else:
            # Convert to numpy for CPU-based sigpy
            kspace_np = kspace_data.cpu().numpy()

            # Estimate sensitivity maps using ESPIRiT
            # Pass full k-space data and let ESPIRiT extract calibration region
            sensitivity_maps = espirit_class(
                ksp=kspace_np, calib_width=calibration_size, thresh=threshold
            ).run()

            return torch.from_numpy(sensitivity_maps).to(kspace_data.device)

    def _estimate_low_rank(
        self,
        kspace_data: torch.Tensor,
        max_rank: int | None = None,
    ) -> torch.Tensor:
        """Estimate sensitivity maps using low-rank approximation.

        Extracts sensitivity profiles by projecting each coil's image
        onto the dominant spatial pattern obtained via SVD of the
        pixel × coil matrix.

        The rank-1 approximation of the image can be written as:
            I_approx(r, c) = u_1(r) * s_1 * v_1^H(c)
        where u_1 is the dominant spatial pattern and v_1^H gives the
        coil weights. The sensitivity map for coil c is then:
            S(c, r) = I(c, r) / composite(r)
        where composite = sqrt(sum_c |I(c,r)|^2) (RSS denominator).

        For robustness we use rank-k SVD to denoise the coil images
        before computing RSS normalization.
        """
        # Get image space data using physics module for proper centering
        image_data = ifft2c(kspace_data)

        # Reshape for SVD: [n_pixels, n_coils]
        n_coils, height, width = image_data.shape
        image_flat = image_data.permute(1, 2, 0).contiguous().view(-1, n_coils)

        # Perform SVD: image_flat ≈ U @ diag(S) @ Vh
        U, S, Vh = torch.linalg.svd(image_flat, full_matrices=False)

        # Determine rank
        if max_rank is None:
            # Use rank that captures 95% of energy
            total_energy = torch.sum(S**2)
            if total_energy > 0:
                energy = torch.cumsum(S**2, dim=0) / total_energy
                max_rank = int(torch.where(energy >= 0.95)[0][0].item()) + 1
            else:
                max_rank = 1
            max_rank = min(max_rank, n_coils)

        # Low-rank denoised coil images: [n_pixels, n_coils]
        S_trunc = S[:max_rank].to(U.dtype)
        denoised_flat = (U[:, :max_rank] * S_trunc.unsqueeze(0)) @ Vh[:max_rank, :]
        denoised = denoised_flat.view(height, width, n_coils).permute(2, 0, 1)

        # Sensitivity = denoised coil image / RSS composite
        rss = torch.sqrt(torch.sum(torch.abs(denoised) ** 2, dim=0).clamp(min=1e-12))
        sensitivity_maps = denoised / rss.unsqueeze(0)

        return sensitivity_maps

    def _estimate_fallback(self, kspace_data: torch.Tensor) -> torch.Tensor:
        """Fallback estimation using RSS normalization.

        Simple method that works for any coil configuration.
        """
        # Convert to image space using physics module for proper centering
        image_data = ifft2c(kspace_data)

        # Compute RSS
        rss = torch.sqrt(torch.sum(torch.abs(image_data) ** 2, dim=0))

        # Avoid division by zero
        rss = torch.where(rss == 0, torch.ones_like(rss), rss)

        # Normalize each coil by RSS
        sensitivity_maps = image_data / rss.unsqueeze(0)

        return sensitivity_maps

    def _normalize_sensitivity_maps(
        self,
        sensitivity_maps: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize sensitivity maps to have unit norm per pixel."""
        # Compute norm per pixel (sum over coils)
        norms = torch.sqrt(torch.sum(torch.abs(sensitivity_maps) ** 2, dim=0))

        # Avoid division by zero
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)

        # Normalize each pixel
        normalized_maps = sensitivity_maps / norms.unsqueeze(0)

        return normalized_maps

    def _estimate_snr(
        self,
        kspace_data: torch.Tensor,
        sensitivity_maps: torch.Tensor,
    ) -> float | None:
        """Estimate SNR from k-space data using outer-corner noise estimation.

        Uses the four corners of k-space (where signal energy is lowest)
        to estimate noise standard deviation, then computes SNR as:
            SNR = mean(|signal_region|) / std(|noise_region|)
        """
        try:
            # Use 1/8 of each dimension for corner noise ROI
            _, h, w = kspace_data.shape
            ch, cw = max(4, h // 8), max(4, w // 8)

            # Extract four corners and stack
            corners = torch.cat(
                [
                    kspace_data[:, :ch, :cw].reshape(-1),
                    kspace_data[:, :ch, -cw:].reshape(-1),
                    kspace_data[:, -ch:, :cw].reshape(-1),
                    kspace_data[:, -ch:, -cw:].reshape(-1),
                ]
            )

            noise_std = torch.abs(corners).std().item()
            if noise_std < 1e-12:
                return None

            # Signal = center region magnitude mean
            center_h = slice(h // 4, 3 * h // 4)
            center_w = slice(w // 4, 3 * w // 4)
            signal_mean = torch.abs(kspace_data[:, center_h, center_w]).mean().item()

            return signal_mean / noise_std
        except Exception:
            return None

    def _generate_cache_key(
        self,
        kspace_data: torch.Tensor,
        method: str,
        calibration_size: int,
        threshold: float,
        max_rank: int | None,
    ) -> str:
        """Generate a unique cache key for the given parameters."""
        # Create a collision-resistant hash using shape, parameters, and
        # deterministic multi-point sampling across the tensor.
        shape_str = str(kspace_data.shape)
        n = kspace_data.numel()

        # Sample from multiple deterministic locations (not just first 100)
        indices = torch.linspace(0, n - 1, steps=min(256, n)).long()
        sample_values = kspace_data.flatten()[indices].cpu().numpy()

        # Combine parameters
        param_str = f"{method}_{calibration_size}_{threshold}_{max_rank}_{shape_str}"

        # Create hash
        hasher = hashlib.md5(usedforsecurity=False)
        hasher.update(param_str.encode())
        hasher.update(sample_values.tobytes())

        return hasher.hexdigest()

    def _load_from_cache(self, cache_key: str) -> CoilSensitivityResult | None:
        """Load result from cache if available."""
        cache_path = self.cache_dir / f"{cache_key}.pt"

        if cache_path.exists():
            try:
                result = safe_torch_load(cache_path, map_location="cpu", weights_only=False)
                # Validate that we loaded a CoilSensitivityResult
                if not isinstance(result, CoilSensitivityResult):
                    raise ValueError(
                        f"Cache file {cache_path} does not contain a valid CoilSensitivityResult",
                    )
                if self.logging_service:
                    self.logging_service.log_debug(
                        f"Loaded coil sensitivity from cache: {cache_key}",
                    )
                return result
            except Exception as e:
                if self.logging_service:
                    self.logging_service.log_warning(f"Failed to load cache: {e}")

        return None

    def _save_to_cache(self, cache_key: str, result: CoilSensitivityResult) -> None:
        """Save result to cache."""
        cache_path = self.cache_dir / f"{cache_key}.pt"

        # The cache is a pickled result, not a model checkpoint, so it is written
        # beside its reader in safe_io. Only I/O failures degrade to a warning (torch's
        # zip writer reports a full disk as RuntimeError); a programming error must
        # surface rather than silently disable the cache.
        try:
            atomic_save_torch(result, cache_path)
            if self.logging_service:
                self.logging_service.log_debug(
                    f"Saved coil sensitivity to cache: {cache_key}",
                )
        except (OSError, RuntimeError) as e:
            if self.logging_service:
                self.logging_service.log_warning(f"Failed to save cache: {e}")

    def clear_cache(self) -> None:
        """Clear all cached sensitivity maps."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.logging_service:
            self.logging_service.log_info("Cleared coil sensitivity cache")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the cache."""
        if not self.cache_dir.exists():
            return {"total_files": 0, "total_size_mb": 0}

        total_size = 0
        file_count = 0

        for file_path in self.cache_dir.glob("*.pt"):
            file_count += 1
            total_size += file_path.stat().st_size

        return {
            "total_files": file_count,
            "total_size_mb": total_size / (1024 * 1024),
        }
