"""Fréchet Radiomic Distance (FRD) and Radiomic Feature Stability (RFS).

Medical image quality metrics based on radiomics features rather than
perceptual features from ImageNet-trained networks.

Based on:
- "FRD: A Metric for Medical Image Generation" concept
- PyRadiomics feature extraction library

Follows unit-test.md rules:
- Strict typing with jaxtyping
- NaN/Inf safety checks
- Deterministic operations
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from spectramr.core.metrics.registry import register_metric

try:
    from beartype import beartype
    from jaxtyping import Float, Int, jaxtyped

    JAXTYPING_AVAILABLE = True
except ImportError:
    JAXTYPING_AVAILABLE = False

def _typechecked[F: Callable[..., Any]](fn: F) -> F:
    """``jaxtyped(beartype(fn))`` when both are installed, else ``fn`` unchanged."""
    return jaxtyped(beartype(fn)) if JAXTYPING_AVAILABLE else fn  # type: ignore[no-any-return, unused-ignore]


try:
    import radiomics
    from radiomics import featureextractor

    RADIOMICS_AVAILABLE = True
except ImportError:
    RADIOMICS_AVAILABLE = False
    radiomics = None
    featureextractor = None

try:  # optional in degraded cluster envs (a REQUIRED dep in pyproject)
    import SimpleITK as sitk

    SIMPLEITK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without SimpleITK
    sitk = None  # type: ignore[assignment, unused-ignore]
    SIMPLEITK_AVAILABLE = False

try:  # optional in degraded cluster envs (a REQUIRED dep in pyproject)
    from scipy import linalg

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without scipy
    linalg = None  # type: ignore[assignment, unused-ignore]
    SCIPY_AVAILABLE = False

# Type aliases for clarity
type FeatureVector = NDArray[np.float64]  # [N_features]
type FeatureMatrix = NDArray[np.float64]  # [N_samples, N_features]


@dataclass(frozen=True)
class RadiomicFeatureSet:
    """Container for radiomic features with metadata.

    Immutable to enforce functional programming paradigm (Directive 3.0).
    """

    features: FeatureVector
    feature_names: tuple[str, ...]
    sample_id: str

    def __post_init__(self) -> None:
        """Validate NaN/Inf absence (Directive 4.D.1)."""
        if not np.isfinite(self.features).all():
            raise ValueError(
                f"RadiomicFeatureSet contains NaN/Inf values for sample {self.sample_id}"
            )


class RadiomicFeatureExtractor:
    """Extract 464-dimensional radiomic feature vectors from 3D volumes.

    Features include:
    - First Order: Mean, Variance, Skewness, Kurtosis (18 features)
    - Shape: Volume, Surface Area, Sphericity (14 features)
    - Texture (GLCM): Contrast, Correlation, Homogeneity (24 features)
    - Texture (GLRLM): Run Length features (16 features)
    - Texture (GLSZM): Size Zone features (16 features)
    - Texture (GLDM): Dependence features (14 features)
    - Texture (NGTDM): Neighborhood features (5 features)
    - Wavelet: All above features on LL, LH, HL, HH sub-bands

    Type Safety: All inputs/outputs are explicitly typed (Directive 1.0).
    """

    # Expected feature dimension (may vary with settings)
    EXPECTED_FEATURE_DIM: int = 464

    #: Allowed backend selectors. ``auto`` picks the first available in
    #: ``BACKEND_PRIORITY`` order; the others select explicitly. An
    #: unknown value raises (pitfall #15: every exposed knob must be wired).
    ALLOWED_BACKENDS: tuple[str, ...] = ("pyradiomics", "fastrad", "auto")
    BACKEND_PRIORITY: tuple[str, ...] = ("fastrad", "pyradiomics")

    def __init__(
        self,
        settings: dict[str, object] | None = None,
        normalize: bool = True,
        include_wavelet: bool = True,
        backend: str = "auto",
    ) -> None:
        """
        Args:
            settings: PyRadiomics extractor settings
            normalize: Z-score normalize features
            include_wavelet: Include wavelet-transformed features
            backend: Feature-extraction backend. One of ``"pyradiomics"``
                (CPU, canonical), ``"fastrad"`` (GPU, opt-in), or
                ``"auto"`` (pick the first available in
                :attr:`BACKEND_PRIORITY` order). The resolved backend is
                stamped onto ``self.backend`` so any results JSON can
                record which path produced the features (CLAUDE.md
                pitfall #15 -- self-describing artefacts).

                ``fastrad`` is not yet wired (raises NotImplementedError
                when explicitly selected, falls through under ``auto``).
                Tracked in ``TODO/backlog_fastrad_radiomics_backend.md``.
        """
        if not SIMPLEITK_AVAILABLE:
            raise ImportError(
                "RadiomicFeatureExtractor requires SimpleITK "
                "(pip install SimpleITK) — the pyradiomics extraction path "
                "converts arrays through sitk. It is a REQUIRED dependency in "
                "pyproject; this guard only fires in a degraded cluster env."
            )

        if backend not in self.ALLOWED_BACKENDS:
            raise ValueError(
                f"backend={backend!r} not in {self.ALLOWED_BACKENDS}. "
                "Pick 'pyradiomics' for the canonical CPU path, 'fastrad' "
                "for the GPU path (when wired), or 'auto' to dispatch."
            )

        resolved = self._resolve_backend(backend)
        self.backend: str = resolved

        if resolved == "pyradiomics" and not RADIOMICS_AVAILABLE:
            raise ImportError("PyRadiomics required. Install with: pip install pyradiomics")

        self.normalize = normalize
        self.include_wavelet = include_wavelet

        # Configure extractor
        if settings is None:
            settings = self._default_settings()

        # Ensure normalize setting is applied if not explicit in settings
        if normalize and "normalize" not in settings:
            settings["normalize"] = True
            settings["normalizeScale"] = 100  # Default scale for normalized data

        # Suppress radiomics logging
        radiomics.setVerbosity(40)  # ERROR level only

        self.extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

        # Enable feature classes
        self.extractor.enableFeatureClassByName("firstorder")
        self.extractor.enableFeatureClassByName("shape")
        self.extractor.enableFeatureClassByName("glcm")
        self.extractor.enableFeatureClassByName("glrlm")
        self.extractor.enableFeatureClassByName("glszm")
        self.extractor.enableFeatureClassByName("gldm")
        self.extractor.enableFeatureClassByName("ngtdm")

        if include_wavelet:
            self.extractor.enableImageTypeByName("Wavelet")

        # Cache for feature names (populated on first extraction)
        self._feature_names: tuple[str, ...] | None = None

    @classmethod
    def _resolve_backend(cls, backend: str) -> str:
        """Resolve ``auto`` to a concrete backend; explicit choices fail loud.

        ``fastrad`` is not yet implemented; an explicit selection raises
        NotImplementedError so a misset env var or YAML key cannot
        silently degrade to pyradiomics. Under ``auto`` we fall through
        to pyradiomics (BACKEND_PRIORITY) until the GPU path lands.
        """
        if backend == "pyradiomics":
            return "pyradiomics"
        if backend == "fastrad":
            if not cls._fastrad_available():
                raise NotImplementedError(
                    "backend='fastrad' is not yet wired in this release. "
                    "Use backend='auto' to fall through to pyradiomics, or "
                    "backend='pyradiomics' to opt in explicitly."
                )
            return "fastrad"
        # auto: try priority order, skipping unwired backends.
        for candidate in cls.BACKEND_PRIORITY:
            if candidate == "fastrad" and not cls._fastrad_available():
                continue
            if candidate == "pyradiomics" and not RADIOMICS_AVAILABLE:
                continue
            return candidate
        raise ImportError(
            "No radiomic backend is available. Install pyradiomics "
            "(pip install pyradiomics); the fastrad path is not yet wired."
        )

    @staticmethod
    def _fastrad_available() -> bool:
        """Whether a module named ``fastrad`` is importable on THIS host.

        Not "currently always False", as this said until 2026-08-09: it is a
        live import probe, and it answers True wherever something of that name
        is installed. Three tests read it directly and asserted ``is False``,
        which passed on a dev box and failed on the cluster (job 8004252). A
        test about the fall-through branch should patch this, not observe it.
        """
        try:
            import fastrad  # type: ignore[import-not-found, unused-ignore]  # noqa: F401
        except ImportError:
            return False
        return True

    def _default_settings(self) -> dict[str, object]:
        """Default PyRadiomics settings."""
        return {
            "binCount": 64,  # Safe fixed bin count to prevent segfaults with outliers
            "resampledPixelSpacing": None,  # Keep original spacing
            "interpolator": sitk.sitkBSpline,
            "force2D": False,
            "force2Ddimension": 0,
        }

    @_typechecked
    def extract_from_array(
        self,
        image: NDArray[np.floating],
        mask: NDArray[np.integer] | None = None,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> RadiomicFeatureSet:
        """Extract features from numpy array.

        Args:
            image: [H, W, D] or [H, W] image array
            mask: [H, W, D] or [H, W] binary mask (optional, uses full image if None)
            spacing: Voxel spacing in mm

        Returns:
            RadiomicFeatureSet with extracted features
        """
        # Ensure 3D
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        if mask is None:
            mask = np.ones_like(image, dtype=np.uint8)
        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]

        # Validate shapes match
        if image.shape != mask.shape:
            raise ValueError(f"Shape mismatch: image {image.shape} vs mask {mask.shape}")

        # Convert to SimpleITK
        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
        sitk_image.SetSpacing(spacing)

        sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
        sitk_mask.SetSpacing(spacing)

        # Extract features
        result = self.extractor.execute(sitk_image, sitk_mask)

        # Parse results
        features_dict: dict[str, float] = {}
        for key, value in result.items():
            # Skip diagnostics, keep only features
            if not key.startswith("diagnostics_"):
                try:
                    features_dict[key] = float(value)
                except (TypeError, ValueError):
                    continue

        # Sort for reproducibility
        sorted_names = tuple(sorted(features_dict.keys()))
        features = np.array([features_dict[k] for k in sorted_names], dtype=np.float64)

        # Cache feature names
        if self._feature_names is None:
            self._feature_names = sorted_names

        # NaN safety (Directive 4.D.1)
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)

        return RadiomicFeatureSet(
            features=features,
            feature_names=sorted_names,
            sample_id=f"array_{id(image)}",
        )

    @_typechecked
    def extract_from_tensor(
        self,
        image: Float[Tensor, "batch channels height width"],
        mask: Int[Tensor, "batch 1 height width"] | None = None,
    ) -> list[RadiomicFeatureSet]:
        """Extract features from PyTorch tensor batch.

        Args:
            image: [B, C, H, W] image tensor
            mask: [B, 1, H, W] optional mask tensor

        Returns:
            List of RadiomicFeatureSet, one per batch item
        """
        batch_size = image.shape[0]
        results: list[RadiomicFeatureSet] = []

        for i in range(batch_size):
            # Extract single image, average over channels
            img_np = image[i].mean(dim=0).cpu().numpy()  # [H, W]

            if mask is not None:
                mask_np = mask[i, 0].cpu().numpy().astype(np.uint8)
            else:
                mask_np = None

            feat_set = self.extract_from_array(img_np, mask_np)
            results.append(feat_set)

        return results

    @property
    def feature_names(self) -> tuple[str, ...] | None:
        """Get cached feature names."""
        return self._feature_names

    @property
    def feature_dim(self) -> int:
        """Get feature vector dimension."""
        if self._feature_names is not None:
            return len(self._feature_names)
        return self.EXPECTED_FEATURE_DIM


@register_metric("frd", aliases=["FRD"])
class FrechetRadiomicDistance:
    """Fréchet Radiomic Distance (FRD) metric.

    Measures the distance between two sets of medical images based on
    their radiomic feature distributions, rather than ImageNet features.

    FRD = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2(Σ_real Σ_gen)^0.5)

    Advantages over FID:
    - Stable with small sample sizes (N < 100)
    - Correlates with radiologist judgment
    - Measures clinically relevant biomarkers
    """

    def __init__(
        self,
        extractor: RadiomicFeatureExtractor | None = None,
        eps: float = 1e-6,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Args:
            extractor: Feature extractor (creates default if None)
            eps: Epsilon for numerical stability
            device: Computation device
        """
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "FrechetRadiomicDistance requires scipy (pip install scipy) "
                "for the matrix square root (scipy.linalg.sqrtm) in the "
                "Fréchet covariance term. It is a REQUIRED dependency in "
                "pyproject; this guard only fires in a degraded cluster env."
            )
        self.extractor = extractor or RadiomicFeatureExtractor()
        self.eps = eps
        self.device = device

    @staticmethod
    @_typechecked
    def _compute_statistics(
        features: FeatureMatrix,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute mean and covariance of feature matrix.

        Args:
            features: [N, D] feature matrix

        Returns:
            mu: [D] mean vector
            sigma: [D, D] covariance matrix
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)

        # Handle single sample case
        if features.shape[0] == 1:
            sigma = np.eye(features.shape[1]) * 1e-6

        return mu, sigma

    @staticmethod
    @_typechecked
    def _frechet_distance(
        mu1: NDArray[np.float64],
        sigma1: NDArray[np.float64],
        mu2: NDArray[np.float64],
        sigma2: NDArray[np.float64],
        eps: float = 1e-6,
    ) -> float:
        """Compute Fréchet distance between two Gaussian distributions.

        Args:
            mu1, sigma1: Mean and covariance of first distribution
            mu2, sigma2: Mean and covariance of second distribution
            eps: Epsilon for numerical stability

        Returns:
            Fréchet distance (lower is better)
        """
        # Mean difference term
        diff = mu1 - mu2
        mean_term = np.dot(diff, diff)

        # Covariance term with matrix square root
        # covmean = (sigma1 @ sigma2)^0.5
        covmean = linalg.sqrtm(sigma1.dot(sigma2))  # SciPy >= 1.18 removed ``disp``

        # Handle numerical instability
        if not np.isfinite(covmean).all():
            warnings.warn("FRD: sqrtm produced non-finite values, using eigenvalue correction")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Remove imaginary component (numerical artifact)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        cov_term = np.trace(sigma1 + sigma2 - 2 * covmean)

        frd = float(mean_term + cov_term)

        # NaN safety check (Directive 4.D.1)
        if not np.isfinite(frd):
            raise ValueError(f"FRD computation produced non-finite result: {frd}")

        return frd

    def compute(
        self,
        real_features: FeatureMatrix,
        generated_features: FeatureMatrix,
    ) -> float:
        """Compute FRD between real and generated feature sets.

        Args:
            real_features: [N_real, D] real image features
            generated_features: [N_gen, D] generated image features

        Returns:
            FRD score (lower is better)
        """
        mu1, sigma1 = self._compute_statistics(real_features)
        mu2, sigma2 = self._compute_statistics(generated_features)

        return self._frechet_distance(mu1, sigma1, mu2, sigma2, self.eps)

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        """Registry-compatible call.

        Args:
            prediction: [B, C, H, W] tensor
            target: [B, C, H, W] tensor
            **kwargs: Ignored

        Returns:
            FRD score
        """
        # Convert tensors to lists of numpy arrays (Radiomics expects 2D/3D)
        batch_size = prediction.shape[0]
        # Average over channels if multiple
        pred_np = [prediction[i].mean(dim=0).cpu().numpy() for i in range(batch_size)]
        target_np = [target[i].mean(dim=0).cpu().numpy() for i in range(batch_size)]

        return self.compute_from_images(target_np, pred_np)

    def compute_from_images(
        self,
        real_images: list[NDArray[np.floating]],
        generated_images: list[NDArray[np.floating]],
    ) -> float:
        """Compute FRD from image arrays.

        Args:
            real_images: List of real image arrays
            generated_images: List of generated image arrays

        Returns:
            FRD score
        """
        real_features = np.stack(
            [self.extractor.extract_from_array(img).features for img in real_images]
        )

        gen_features = np.stack(
            [self.extractor.extract_from_array(img).features for img in generated_images]
        )

        return self.compute(real_features, gen_features)


@register_metric("rfs", aliases=["RadiomicStability"])
class RadiomicFeatureStability:
    """Radiomic Feature Stability (RFS) metric.

    Measures how well a reconstruction preserves clinically relevant
    radiomic features compared to ground truth.

    RFS = 1 - (1/N_f) * Σ |R_i(recon) - R_i(GT)| / σ_i

    Higher is better (range: [-inf, 1], with 1 being perfect).
    """

    def __init__(
        self,
        extractor: RadiomicFeatureExtractor | None = None,
        population_std: FeatureVector | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Args:
            extractor: Feature extractor
            population_std: Pre-computed population standard deviation per feature
            device: Computation device
        """
        self.extractor = extractor or RadiomicFeatureExtractor()
        self.population_std = population_std
        self.device = device

    def compute(
        self,
        reconstructed: NDArray[np.floating],
        ground_truth: NDArray[np.floating],
        mask: NDArray[np.integer] | None = None,
    ) -> float:
        """Compute RFS between reconstruction and ground truth.

        Args:
            reconstructed: Reconstructed image array
            ground_truth: Ground truth image array
            mask: Optional ROI mask

        Returns:
            RFS score (higher is better, max 1.0)
        """
        recon_feat = self.extractor.extract_from_array(reconstructed, mask).features
        gt_feat = self.extractor.extract_from_array(ground_truth, mask).features

        # Absolute difference
        diff = np.abs(recon_feat - gt_feat)

        # Normalize by population std when available.
        if self.population_std is not None:
            std = self.population_std
        else:
            # No population std: use each feature's own magnitude as a rough
            # scale, but floor it at a small fraction of the largest feature
            # magnitude. Otherwise a near-zero feature (skewness, correlation,
            # ...) gives std ~ 1e-8 and a tiny difference explodes the relative
            # error, dragging RFS to a meaningless large-negative value.
            feat_scale = float(np.abs(gt_feat).max()) + 1e-8
            std = np.maximum(np.abs(gt_feat), 1e-3 * feat_scale)

        normalized_diff = diff / std

        rfs = float(1.0 - np.mean(normalized_diff))

        return rfs

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        """Registry-compatible call."""
        # Average over batch
        batch_size = prediction.shape[0]
        scores = []
        for i in range(batch_size):
            p = prediction[i].mean(dim=0).cpu().numpy()
            t = target[i].mean(dim=0).cpu().numpy()
            scores.append(self.compute(p, t))

        return float(np.mean(scores))

    def compute_batch(
        self,
        reconstructed_list: list[NDArray[np.floating]],
        ground_truth_list: list[NDArray[np.floating]],
    ) -> tuple[float, float]:
        """Compute mean and std of RFS across a batch.

        Returns:
            (mean_rfs, std_rfs)
        """
        scores = [
            self.compute(recon, gt)
            for recon, gt in zip(reconstructed_list, ground_truth_list, strict=False)
        ]

        return float(np.mean(scores)), float(np.std(scores))
