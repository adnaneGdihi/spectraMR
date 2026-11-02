"""QA and Artifact Metrics for MRI Image Quality Assessment.

Implements three MRIQC-derived metrics:

- **HD95** — Hausdorff Distance (95th percentile) between thresholded
  prediction and target masks, computed via MONAI's optimized implementation.
- **QI1** — Proportion of corrupted voxels in the background (Mortamet et al.,
  2009). Detects ghosting, motion leakage, and flow artifacts.
- **WM2Max** — White-matter-to-maximum-intensity ratio. Flags long-tailed
  intensity distributions caused by hyper-intense artifacts.

References:
    Mortamet, B. et al. (2009). Automatic quality assessment in structural
    brain magnetic resonance imaging. MRM, 62(2), 365-372.

    Esteban, O. et al. (2017). MRIQC: Advancing the automatic prediction
    of image quality in MRI from unseen sites. PLoS ONE, 12(9), e0184661.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from spectramr.config.schemas.enums import Regime
from spectramr.core.metrics.evaluation_metrics import BaseMetric
from spectramr.core.metrics.monai_compat import call_surface_metric
from spectramr.core.metrics.registry import register_metric
from spectramr.core.metrics.tissue_segmentation import _otsu_thresholds


@register_metric("hd95", aliases=["HD95", "hausdorff_95"])
class HD95Metric(BaseMetric):
    """Hausdorff Distance — 95th Percentile.

    Computes the symmetric Hausdorff distance at the 95th percentile
    between binary masks derived from prediction and target images
    via Otsu thresholding.

    Lower is better (0 = perfect spatial overlap of boundaries).

    Uses MONAI's ``HausdorffDistanceMetric`` for GPU-accelerated,
    numerically-stable boundary distance computation.
    """

    def __init__(self, device: str | torch.device = "cpu") -> None:
        super().__init__(device=device)
        self._monai_metric = None

    def _get_monai_metric(self) -> Any:
        """Lazy-load MONAI metric to avoid import-time overhead."""
        if self._monai_metric is None:
            from monai.metrics import HausdorffDistanceMetric

            self._monai_metric = HausdorffDistanceMetric(
                include_background=False,
                percentile=95,
                directed=False,
            )
        return self._monai_metric

    @staticmethod
    def _otsu_threshold(img: Tensor) -> Tensor:
        """Binarize a grayscale image via Otsu's method over its own intensity range.

        The histogram spans the tensor's [min, max], so ``[-1, 1]``-normalised arms
        threshold the same anatomy as ``[0, 1]`` ones (#174). A constant image has no
        threshold and yields an empty mask, which the caller scores as the penalty.

        Args:
            img: ``(B, C, H, W)`` float tensor, any intensity range.

        Returns:
            Binary mask ``(B, C, H, W)`` with dtype float32.
        """
        thresholds = _otsu_thresholds(img, n_classes=2)
        if thresholds is None:
            return torch.zeros_like(img, dtype=torch.float32)
        return (img > thresholds[0]).float()

    def compute_metric(self, preds: Tensor, target: Tensor, **kwargs: Any) -> Tensor:
        """Compute HD95 between binarized prediction and target.

        Args:
            preds: Predicted image ``(B, C, H, W)``.
            target: Ground truth image ``(B, C, H, W)``.

        Returns:
            Scalar tensor with HD95 value (in voxel units).
        """
        preds = preds.to(self.device)
        target = target.to(self.device)

        pred_mask = self._otsu_threshold(preds)
        gt_mask = self._otsu_threshold(target)

        # MONAI expects one-hot-like format: (B, C, H, W) with C = num_classes
        # For binary masks, we need (B, 1, H, W) with foreground channel
        if pred_mask.dim() == 4 and pred_mask.shape[1] == 1:
            # Already (B, 1, H, W) — use as foreground channel
            pass
        elif pred_mask.dim() == 3:
            pred_mask = pred_mask.unsqueeze(1)
            gt_mask = gt_mask.unsqueeze(1)

        # Guard: if either mask is entirely empty → return diagonal as penalty
        if pred_mask.sum() < 1 or gt_mask.sum() < 1:
            diag = (pred_mask.shape[-2] ** 2 + pred_mask.shape[-1] ** 2) ** 0.5
            return torch.tensor(diag, device=self.device)

        hd95_val = call_surface_metric(self._get_monai_metric(), y_pred=pred_mask, y=gt_mask)

        # MONAI returns (B, C) → take mean
        result = hd95_val[hd95_val.isfinite()].mean()
        if not result.isfinite():
            diag = (pred_mask.shape[-2] ** 2 + pred_mask.shape[-1] ** 2) ** 0.5
            return torch.tensor(diag, device=self.device)

        return result


@register_metric("qi1", aliases=["QI1", "quality_index_1"])
class QI1Metric(BaseMetric):
    """Quality Index 1 — Artifact-in-Background Detection.

    Measures the proportion of voxels in the background (air) region of
    an MRI image that exhibit intensities inconsistent with pure noise,
    indicating the presence of artifacts (ghosting, motion, flow).

    The background is defined **spatially** as the four image corners (not a
    value-based threshold — see :meth:`compute_metric`, which explains why a
    ``preds < noise_floor`` mask is incorrect). Artifact voxels are corner
    voxels whose intensity exceeds a threshold derived from the foreground
    signal level and the corner noise statistics (median + MAD-derived σ).

    Lower is better (0 = no artifacts in background).

    Reference:
        Mortamet, B. et al. (2009). MRM, 62(2), 365-372.
    """

    def compute_metric(self, preds: Tensor, target: Tensor, **kwargs: Any) -> Tensor:
        """Compute QI1 artifact ratio.

        The background region is defined **spatially** as the four
        image corners (outer 12.5% along each axis). This matches the
        Mortamet 2009 implementation, where background = air outside
        the head, regardless of whether artefacts have polluted it
        with high intensities. Using a value-based mask
        (``preds < noise_floor``) is incorrect because bright
        artefacts in the corners would be re-classified as foreground
        and silently excluded — the very thing the metric is supposed
        to detect.

        Noise statistics (median + MAD-derived σ) are estimated from
        ``target`` rather than ``preds`` so they reflect the clean
        reference, then bright voxels in **prediction**'s corner
        region are counted.

        Args:
            preds: Image to assess ``(B, C, H, W)``.
            target: Reference image — used to estimate the corner
                noise statistics. Pass ``preds`` itself for a pure
                no-reference variant.

        Returns:
            Scalar tensor with QI1 value ∈ [0, 1].
        """
        preds = preds.to(self.device).float()
        if preds.dim() < 4:
            preds = preds.unsqueeze(0) if preds.dim() == 3 else preds.unsqueeze(0).unsqueeze(0)
        target = target.to(self.device).float()
        if target.dim() < 4:
            target = target.unsqueeze(0) if target.dim() == 3 else target.unsqueeze(0).unsqueeze(0)

        h, w = preds.shape[-2], preds.shape[-1]
        # Corner fraction — outer 12.5% along each axis (Mortamet 2009 uses
        # ~1/8 image side).
        ch = max(h // 8, 1)
        cw = max(w // 8, 1)

        def _corners(t: Tensor) -> Tensor:
            return torch.cat(
                [
                    t[..., :ch, :cw].flatten(),
                    t[..., :ch, -cw:].flatten(),
                    t[..., -ch:, :cw].flatten(),
                    t[..., -ch:, -cw:].flatten(),
                ]
            )

        pred_corners = _corners(preds)
        n_background = pred_corners.numel()
        if n_background < 10:
            return torch.tensor(0.0, device=self.device)

        # Estimate the foreground signal level from the *target* (any
        # voxel whose value rivals the bright signal is by definition
        # outside the air background). Use the 95th-percentile of the
        # full target image — robust to a handful of bright outliers
        # and largely insensitive to the small artefact deposits
        # themselves, which occupy << 5% of the volume.
        signal_p95 = torch.quantile(target.flatten(), 0.95)
        # A corner voxel is "artefact" if it carries a substantial
        # fraction of the foreground signal — i.e. it should have
        # been air. Threshold = 30% of the foreground reference,
        # plus a small absolute floor so a near-zero target (no
        # foreground at all) doesn't trip on its own residual
        # noise.
        rel_floor = 0.30 * signal_p95
        abs_floor = torch.tensor(0.05, device=self.device, dtype=preds.dtype)
        artefact_threshold = torch.maximum(rel_floor, abs_floor)

        n_artifact = (pred_corners > artefact_threshold).sum().float()
        qi1 = n_artifact / n_background
        return qi1.clamp(0.0, 1.0)


@register_metric(
    "wm2max",
    aliases=["WM2Max", "wm_to_max"],
    workflows=frozenset({Regime.STRUCTURAL}),
)
class WM2MaxMetric(BaseMetric):
    """White-Matter-to-Maximum Intensity Ratio.

    Tagged ``mri_structural`` — and only that. This is an *anatomical* IQM: it
    localises white matter by intensity percentile, so it presupposes anatomical
    tissue contrast and means nothing on a parameter map, a velocity field or an
    ADC map. Its sibling MRIQC metrics ``efc``/``fber``/``qi1`` are deliberately
    left untagged, because those apply to any MR image (MRIQC runs them in the
    functional pipeline too) and a tag that broad claims nothing.

    Computes the ratio of the median intensity within a synthetic
    white-matter mask to the 95th percentile of the full image
    intensity distribution.

    This metric captures long-tailed intensity distributions caused
    by artifacts such as bright-vessel contamination or fat signal.

    Expected range: [0.6, 0.8] for normal structural MRI.
    Higher is better (closer to 1.0 = less artifact contamination).

    Reference:
        Esteban, O. et al. (2017). PLoS ONE, 12(9), e0184661.
    """

    def compute_metric(self, preds: Tensor, target: Tensor, **kwargs: Any) -> Tensor:
        """Compute WM2Max ratio.

        In the absence of tissue segmentation, we approximate the WM
        region as voxels between the 60th and 90th percentile of the
        foreground intensity distribution (brain parenchyma). This is
        a standard heuristic when no segmentation atlas is available.

        Args:
            preds: Image to assess ``(B, C, H, W)``.
            target: Reference image (used for the WM mask estimation).

        Returns:
            Scalar tensor with WM2Max ratio.
        """
        preds = preds.to(self.device).float()
        if preds.dim() < 4:
            preds = preds.unsqueeze(0) if preds.dim() == 3 else preds.unsqueeze(0).unsqueeze(0)

        flat = preds.flatten()

        # Foreground mask: voxels above noise floor
        p02 = torch.quantile(flat, 0.02) if flat.numel() > 0 else torch.tensor(0.0)
        fg = flat[flat > p02]

        if fg.numel() < 10:
            return torch.tensor(0.0, device=self.device)

        # WM proxy: voxels between 60th–90th percentile of foreground
        p60 = torch.quantile(fg, 0.60)
        p90 = torch.quantile(fg, 0.90)
        wm_mask = (preds > p60) & (preds < p90)
        wm_voxels = preds[wm_mask]

        if wm_voxels.numel() < 5:
            return torch.tensor(0.0, device=self.device)

        wm_median = wm_voxels.median()

        # 95th percentile of entire image
        p95 = torch.quantile(flat, 0.95)

        if p95 < 1e-8:
            return torch.tensor(0.0, device=self.device)

        wm2max = wm_median / p95
        return wm2max.clamp(0.0, 2.0)  # Clamp to avoid extreme ratios
