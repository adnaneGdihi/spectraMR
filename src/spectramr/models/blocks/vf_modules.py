"""Virtual Fiducial Neural Modules for Marker-Anchored Reconstruction.

Reusable building blocks that leverage known marker properties to extract
scan-specific parameters for correction transfer.

Modules
-------
- :class:`NeuralPSFEstimator`    — MLP that estimates the 2D PSF kernel
- :class:`NoiseVarianceEstimator` — estimates σ² from marker uniform region
- :class:`MarkerMetaAdapter`     — MAML-style inner loop on marker residual
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ======================================================================
# 1. Neural PSF Estimator
# ======================================================================


class NeuralPSFEstimator(nn.Module):
    """Estimate the spatial Point Spread Function from marker blur.

    A lightweight MLP maps the blurred marker region to a 2D PSF kernel.
    The sharp edges of the marker create an ideal target for blind
    deconvolution—the PSF is the *only* unknown.

    Architecture::

        flatten(marker_crop) → Linear → ReLU → Linear → ReLU
        → Linear → reshape(kH, kW) → softmax (non-negative, unit sum)

    Args:
        marker_crop_size: Spatial size of the cropped marker patch.
        kernel_size: Output PSF kernel spatial extent.
        hidden_dim: MLP hidden layer width.
    """

    def __init__(
        self,
        marker_crop_size: int = 32,
        kernel_size: int = 15,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        input_dim = marker_crop_size * marker_crop_size

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, kernel_size * kernel_size),
        )

        # Initialise close to delta function
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        center = kernel_size * kernel_size // 2
        with torch.no_grad():
            self.mlp[-1].bias[center] = 5.0  # softmax → sharp peak at centre

    def forward(self, marker_crop: torch.Tensor) -> torch.Tensor:
        """Estimate PSF kernel from blurred marker crop.

        Args:
            marker_crop: Cropped marker region ``[B, 1, crop_H, crop_W]``.

        Returns:
            Estimated PSF ``[B, 1, kH, kW]`` (non-negative, sums to 1).

        forward method for NeuralPSFEstimator.

        Executes PyTorch tensor operations.

        Args:
            marker_crop (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B = marker_crop.shape[0]
        flat = marker_crop.reshape(B, -1)
        raw = self.mlp(flat)  # [B, kH*kW]

        # Softmax → non-negative, unit sum
        psf = F.softmax(raw, dim=-1)
        return psf.reshape(B, 1, self.kernel_size, self.kernel_size)


# ======================================================================
# 2. Noise Variance Estimator
# ======================================================================


class NoiseVarianceEstimator(nn.Module):
    """Estimate per-scan noise variance from marker uniform regions.

    The uniform (flat-signal) regions of the marker provide an ideal
    substrate for noise measurement—any variation is pure thermal noise.

    For complex MRI data, both real and imaginary channels are sampled.

    Args:
        min_voxels: Minimum number of mask voxels required.
    """

    def __init__(self, min_voxels: int = 100) -> None:
        super().__init__()
        self.min_voxels = min_voxels

    @torch.no_grad()
    def forward(
        self,
        image: torch.Tensor,
        marker_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate noise variance from marker region.

        Args:
            image: Acquired image ``[B, C, H, W]``.
            marker_mask: Binary mask ``[1, 1, H, W]`` for uniform marker ROI.

        Returns:
            Noise variance ``[B]`` (one per batch element).

        forward method for NoiseVarianceEstimator.

        Executes PyTorch tensor operations.

        Args:
            image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B = image.shape[0]
        mask_flat = marker_mask.flatten().bool()

        # Boolean-index the flattened spatial axis for ALL batch elements at
        # once: [B, C, N_voxels]. The voxel count comes from the result SHAPE
        # (a Python int) rather than ``mask_flat.sum().item()`` — so the
        # per-step host sync AND the Python ``for b in range(B)`` loop are both
        # removed (this estimator runs every train step for monitoring).
        masked = image.reshape(B, image.shape[1], -1)[..., mask_flat]  # [B, C, N_voxels]
        n_voxels = masked.shape[-1]

        if n_voxels < self.min_voxels:
            logger.warning(
                "Only %d voxels in marker mask (min=%d). Returning zero variance.",
                n_voxels,
                self.min_voxels,
            )
            return torch.zeros(B, device=image.device)

        # Variance across voxels (per channel), averaged over channels -> [B].
        return masked.var(dim=-1).mean(dim=1)


# ======================================================================
# 3. Marker Meta-Adapter (MAML Inner Loop)
# ======================================================================


class MarkerMetaAdapter(nn.Module):
    """MAML-style inner-loop adaptation using marker residual.

    Pre-trained foundational weights are rapidly fine-tuned using the
    corrupted marker as a 1-shot support set.  The network updates its
    weights to target the exact artifact morphology present in *this*
    scan's marker, then applies the correction globally.

    Architecture::

        1. Compute marker residual: ΔM = corrupted_marker - ideal_marker
        2. Inner loop: N gradient steps on L1(denoiser(ΔM), 0)
        3. Apply adapted denoiser to full anatomy

    Args:
        base_denoiser: The pre-trained denoiser module.
        inner_lr: Learning rate for the inner adaptation loop.
        inner_steps: Number of inner-loop gradient steps.
    """

    def __init__(
        self,
        base_denoiser: nn.Module,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
    ) -> None:
        super().__init__()
        self.base_denoiser = base_denoiser
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

    def forward(
        self,
        corrupted_image: torch.Tensor,
        marker_mask: torch.Tensor,
        marker_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Adapt and apply denoiser using marker as 1-shot support.

        Args:
            corrupted_image: Full corrupted image ``[B, C, H, W]``.
            marker_mask: Binary mask ``[1, 1, H, W]``.
            marker_prior: Known ideal marker ``[1, 1, H, W]``.

        Returns:
            Corrected image ``[B, C, H, W]``.

        forward method for MarkerMetaAdapter.

        Executes PyTorch tensor operations.

        Args:
            corrupted_image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_prior (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Extract marker region
        corrupted_marker = corrupted_image * marker_mask

        # Inner adaptation loop (functional, so gradients flow through)
        # Clone parameters for the inner loop (avoid mutating base weights)
        adapted_params = {
            name: param.clone() for name, param in self.base_denoiser.named_parameters()
        }

        for _step in range(self.inner_steps):
            # Forward with current adapted parameters
            denoised_marker = self._functional_forward(corrupted_marker, adapted_params)

            # Loss: denoised marker should match ideal marker
            inner_loss = F.l1_loss(
                denoised_marker * marker_mask,
                marker_prior * marker_mask,
            )

            # Compute inner gradients
            grads = torch.autograd.grad(
                inner_loss,
                adapted_params.values(),
                create_graph=True,
                allow_unused=True,
            )

            # Update adapted parameters
            adapted_params = {
                name: param
                - self.inner_lr * (grad if grad is not None else torch.zeros_like(param))
                for (name, param), grad in zip(adapted_params.items(), grads, strict=False)
            }

        # Apply adapted denoiser to full image
        return self._functional_forward(corrupted_image, adapted_params)

    def _functional_forward(
        self,
        x: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass using externally-provided parameters.

        This enables meta-gradient flow through the inner loop.

        Args:
            x: Input tensor.
            params: Dictionary of named parameters.

        Returns:
            Output tensor.
        """
        # Simple case: if base_denoiser is a Sequential of Linear/Conv layers
        # For complex architectures, this would need model-specific handling
        out = x
        param_iter = iter(params.values())

        for module in self.base_denoiser.modules():
            if isinstance(module, nn.Conv2d):
                weight = next(param_iter)
                bias = next(param_iter) if module.bias is not None else None
                out = F.conv2d(
                    out,
                    weight,
                    bias,
                    stride=module.stride,
                    padding=module.padding,
                    groups=module.groups,
                )
            elif isinstance(module, nn.Linear):
                weight = next(param_iter)
                bias = next(param_iter) if module.bias is not None else None
                out = F.linear(out, weight, bias)
            elif isinstance(module, nn.ReLU):
                out = F.relu(out, inplace=False)
            elif isinstance(module, nn.LeakyReLU):
                out = F.leaky_relu(out, module.negative_slope, inplace=False)

        return out


# ======================================================================
# 4. Marker Dictionary Learner (Sparse Coding)
# ======================================================================


class MarkerDictionaryLearner(nn.Module):
    """K-SVD-style dictionary learning from marker artifact patches.

    Extracts patches from the corrupted marker, builds a dictionary of
    artifact and signal atoms, then reconstructs the patient image using
    only clean signal atoms (removing artifact atoms).

    Pipeline::

        1. Extract patches from corrupted marker and ideal marker
        2. Learn dictionary via alternating sparse coding + dictionary update
        3. Identify artifact atoms (present in corrupted but not ideal)
        4. Reconstruct anatomy using only clean atoms

    Args:
        patch_size: Spatial size of extracted patches.
        num_atoms: Total number of dictionary atoms.
        sparsity: Maximum non-zero coefficients per patch (OMP sparsity).
        num_iters: Number of dictionary learning iterations.
    """

    def __init__(
        self,
        patch_size: int = 8,
        num_atoms: int = 64,
        sparsity: int = 4,
        num_iters: int = 10,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_atoms = num_atoms
        self.sparsity = sparsity
        self.num_iters = num_iters

        # Dictionary: [num_atoms, patch_size²]
        atom_dim = patch_size * patch_size
        self.dictionary = nn.Parameter(
            torch.randn(num_atoms, atom_dim) * 0.01,
        )
        # Normalise atoms
        with torch.no_grad():
            self.dictionary.data = F.normalize(self.dictionary.data, dim=1)

    def _extract_patches(self, image: torch.Tensor) -> torch.Tensor:
        """Extract non-overlapping patches from image.

        Args:
            image: ``[B, 1, H, W]``.

        Returns:
            Patches ``[B*N_patches, patch_size²]``.
        """
        p = self.patch_size
        # Unfold into patches
        patches = image.unfold(2, p, p).unfold(3, p, p)  # [B, 1, nH, nW, p, p]
        B, _, nH, nW, _, _ = patches.shape
        return patches.reshape(B * nH * nW, p * p)

    def _sparse_code(self, patches: torch.Tensor) -> torch.Tensor:
        """Compute sparse codes via greedy matching pursuit (simplified OMP).

        Args:
            patches: ``[N, D]`` where D = patch_size².

        Returns:
            Sparse coefficients ``[N, num_atoms]``.
        """
        D_norm = F.normalize(self.dictionary, dim=1)  # [K, D]
        N = patches.shape[0]
        coeffs = torch.zeros(N, self.num_atoms, device=patches.device)

        residual = patches.clone()
        for _ in range(self.sparsity):
            # Correlation: find best matching atom
            corr = residual @ D_norm.T  # [N, K]
            best_idx = corr.abs().argmax(dim=1)  # [N]

            # Update coefficients and residual for every patch at once.
            picked = best_idx.unsqueeze(1)  # [N, 1]
            c = corr.gather(1, picked)  # [N, 1]
            coeffs.scatter_add_(1, picked, c)
            residual = residual - c * D_norm[best_idx]

        return coeffs

    def identify_artifact_atoms(
        self,
        corrupted_marker: torch.Tensor,
        ideal_marker: torch.Tensor,
        threshold: float = 0.3,
    ) -> torch.Tensor:
        """Identify which dictionary atoms capture artifacts vs. signal.

        Args:
            corrupted_marker: Corrupted marker patch ``[B, 1, H, W]``.
            ideal_marker: Ideal (noise-free) marker ``[B, 1, H, W]``.
            threshold: Energy ratio threshold for artifact classification.

        Returns:
            Boolean mask ``[num_atoms]``: True = artifact atom.
        """
        corrupted_patches = self._extract_patches(corrupted_marker)
        ideal_patches = self._extract_patches(ideal_marker)

        codes_corrupted = self._sparse_code(corrupted_patches)
        codes_ideal = self._sparse_code(ideal_patches)

        # Atoms heavily used by corrupted but not ideal → artifact atoms
        energy_corrupted = codes_corrupted.abs().mean(dim=0)
        energy_ideal = codes_ideal.abs().mean(dim=0)

        ratio = energy_corrupted / (energy_ideal + 1e-6)
        return ratio > (1.0 + threshold)

    def forward(
        self,
        image: torch.Tensor,
        artifact_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct image using only clean dictionary atoms.

        Args:
            image: Input image ``[B, 1, H, W]``.
            artifact_mask: Boolean ``[num_atoms]`` marking artifact atoms.

        Returns:
            Cleaned image ``[B, 1, H, W]``.

        forward method for MarkerDictionaryLearner.

        Executes PyTorch tensor operations.

        Args:
            image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            artifact_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = image.shape
        p = self.patch_size
        nH, nW = H // p, W // p

        patches = self._extract_patches(image)  # [B*nH*nW, D]
        codes = self._sparse_code(patches)  # [B*nH*nW, K]

        # Zero out artifact atom coefficients
        if artifact_mask is not None:
            codes[:, artifact_mask] = 0.0

        # Reconstruct patches from clean codes
        D_norm = F.normalize(self.dictionary, dim=1)
        clean_patches = codes @ D_norm  # [B*nH*nW, D]

        # Fold back into image
        clean_patches = clean_patches.reshape(B, nH, nW, p, p)
        output = torch.zeros_like(image)
        for i in range(nH):
            for j in range(nW):
                output[:, 0, i * p : (i + 1) * p, j * p : (j + 1) * p] = clean_patches[:, i, j]

        return output


# ======================================================================
# 5. Spatiotemporal Marker Attention
# ======================================================================


class SpatiotemporalMarkerAttention(nn.Module):
    """Cross-attention module for temporal noise correlation subtraction.

    In dynamic scans, the stationary marker exhibits temporal fluctuations
    that are purely artifact/noise.  This module uses cross-attention to
    correlate the marker's temporal noise pattern with the patient anatomy
    and subtract the shared noise component.

    Architecture::

        Q = W_q(anatomy_features)     — queries from anatomy
        K = W_k(marker_features)      — keys from marker temporal signal
        V = W_v(marker_features)      — values (noise patterns)

        noise_estimate = softmax(Q·K^T / √d) · V
        cleaned = anatomy - noise_estimate

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        num_timepoints: Number of temporal frames.
    """

    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 4,
        num_timepoints: int = 16,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Project spatial features to embedding dimension
        self.spatial_proj = nn.Conv2d(1, dim, kernel_size=3, padding=1)

        # Temporal projection for marker signal
        self.temporal_proj = nn.Linear(num_timepoints, dim)

        # Multi-head cross attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Final projection back to image space
        self.to_image = nn.Conv2d(dim, 1, kernel_size=1)

        self.scale = (dim // num_heads) ** -0.5

    def forward(
        self,
        anatomy_frame: torch.Tensor,
        marker_temporal: torch.Tensor,
    ) -> torch.Tensor:
        """Subtract temporally-correlated noise using marker attention.

        Args:
            anatomy_frame: Current anatomy frame ``[B, 1, H, W]``.
            marker_temporal: Marker signal over time ``[B, T, H_m, W_m]``
                or flattened ``[B, T]``.

        Returns:
            Noise-subtracted anatomy ``[B, 1, H, W]``.

        forward method for SpatiotemporalMarkerAttention.

        Executes PyTorch tensor operations.

        Args:
            anatomy_frame (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_temporal (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, _, H, W = anatomy_frame.shape

        # Spatial features from anatomy
        spatial_feat = self.spatial_proj(anatomy_frame)  # [B, dim, H, W]
        spatial_flat = spatial_feat.reshape(B, self.dim, -1).permute(0, 2, 1)  # [B, HW, dim]

        # Temporal features from marker
        if marker_temporal.ndim > 2:
            marker_temporal = marker_temporal.reshape(B, marker_temporal.shape[1], -1).mean(dim=-1)
        marker_feat = self.temporal_proj(marker_temporal)  # [B, dim]
        marker_feat = marker_feat.unsqueeze(1)  # [B, 1, dim] — single temporal token

        # Cross attention: anatomy queries, marker keys/values
        Q = self.q_proj(spatial_flat)  # [B, HW, dim]
        K = self.k_proj(marker_feat)  # [B, 1, dim]
        V = self.v_proj(marker_feat)  # [B, 1, dim]

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, HW, 1]
        attn = torch.softmax(attn, dim=-1)

        noise_estimate = torch.matmul(attn, V)  # [B, HW, dim]
        noise_estimate = self.out_proj(noise_estimate)  # [B, HW, dim]

        # Reshape to spatial
        noise_spatial = noise_estimate.permute(0, 2, 1).reshape(B, self.dim, H, W)
        noise_image = self.to_image(noise_spatial)  # [B, 1, H, W]

        return anatomy_frame - noise_image


__all__ = [
    "MarkerDictionaryLearner",
    "MarkerMetaAdapter",
    "NeuralPSFEstimator",
    "NoiseVarianceEstimator",
    "SpatiotemporalMarkerAttention",
]
