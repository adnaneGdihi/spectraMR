r"""Metrics implementing the gaps from `TODO/report_step` Part C.

Bundles 18 previously-missing metrics into one module so the registry
matches the headline Part C 50-metric table:

* DISTS                              (#12)
* MutualInformationMetric            (#17)
* EdgePreservationIndex              (#21)
* RadialKSpaceError                  (#23)
* AverageSurfaceDistance             (#28)
* TargetRegistrationError            (#29)
* FoldingFraction                    (#30)
* DVFMAE                             (#31)
* ThroughPlaneFWHM                   (#32)
* BlandAltmanBias / LimitsOfAgreement (#34, #35)
* ICC3_1                             (#36)
* CoefficientOfVariation             (#37)
* DetectionSensitivity / Specificity (#39)
* AUROC / AUPRC                      (#40)
* ExpectedCalibrationError           (#41)
* NLLBitsPerDim                      (#42)
* Wasserstein1 / SlicedWasserstein   (#45, #46)
* MMDMetric                          (#47)
* ResidualWhiteness                  (#49)
* CohenKappa                         (#50)

Each is registered via ``@register_metric`` and follows the
``BaseMetric`` contract (``compute_metric(preds, target, **kwargs)``
returns a scalar :class:`torch.Tensor`).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

from spectramr.core.metrics.evaluation_metrics import BaseMetric
from spectramr.core.metrics.registry import register_metric

type Image = torch.Tensor


def _to_numpy(t: Tensor) -> np.ndarray:
    if isinstance(t, np.ndarray):
        return t
    return t.detach().cpu().numpy()


def _flatten_pair(p: Tensor, q: Tensor) -> tuple[np.ndarray, np.ndarray]:
    a, b = _to_numpy(p).ravel(), _to_numpy(q).ravel()
    return a, b


# ── #12 DISTS ────────────────────────────────────────────────────────────────


@register_metric("dists", aliases=["DISTS", "structure_texture_similarity"])
class DISTSMetric(BaseMetric):
    """DISTS — Structure & Texture Similarity in deep features.

    Re-uses the existing DISTS *loss* (``dists`` in the loss registry)
    since the underlying computation is identical. Lower is better.
    """

    def __init__(self, device: str | torch.device = "cpu"):
        super().__init__(device=device)
        from spectramr.models.losses import create_loss

        self._impl = create_loss("dists")
        # DISTS loss instantiates VGG19 unconditionally on the
        # default tensor device (CPU) regardless of ``device=`` on
        # the metric wrapper. Move the heavy weights to the
        # requested device explicitly so cluster runs with
        # ``device=cuda`` don't grind through per-call CPU forward
        # passes (audit 2026-05: observed ~7 s/call vs. ~10 ms on GPU).
        # Eval mode + ``requires_grad_(False)`` keep the autograd
        # graph from growing across the precompute loop. Forcing
        # ``.float()`` pins VGG19 to fp32 — bf16 leaks from upstream
        # AMP contexts would otherwise change DISTS numerics
        # silently.
        target_device = torch.device(device)
        self._impl.to(target_device).float()
        self._impl.eval()
        for p in self._impl.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()  # type: ignore[misc, unused-ignore]
    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        # Ensure inputs live on the same device + dtype (fp32) as the
        # metric's internal VGG19. The precompute loop calls this
        # with whatever device the simulator produced; without this
        # guard, a stray CPU input forces an implicit device
        # transfer per call, and a stray bf16 input from an outer
        # autocast block breaks the conv-against-fp32-weights call.
        target_device = next(self._impl.parameters()).device
        if preds.device != target_device:
            preds = preds.to(target_device)
        if target.device != target_device:
            target = target.to(target_device)
        # Defeat any outer autocast — VGG19 is a fp32-only path.
        with (
            torch.amp.autocast("cuda", enabled=False),
            torch.amp.autocast("cpu", enabled=False),
        ):
            return self._impl(preds.float(), target.float()).detach()


# ── Focal Frequency Distance ────────────────────────────────────────────────


@register_metric("focal_frequency", aliases=["FocalFrequency", "ffl"])
class FocalFrequencyMetric(BaseMetric):
    """Focal Frequency distance — a non-saturating spectral-fidelity metric.

    Re-uses the existing ``focal_frequency_radial`` *loss* (weighted squared
    error between the 2-D DFTs of prediction and target, with a radial focal
    weight that up-weights the hard, high-frequency bands). Unlike SSIM/HFEN it
    keeps discriminating once pixel losses have flattened, which is exactly the
    high-frequency saturation regime the MRIxFields2026 arms live in. Lower is
    better. Direction is centrally declared in
    :data:`~spectramr.core.metrics.metric_directions.METRIC_HIGHER_IS_BETTER`.
    """

    def __init__(self, device: str | torch.device = "cpu"):
        super().__init__(device=device)
        from spectramr.models.losses import create_loss

        self._device = torch.device(device)
        # Pure-FFT loss: no VGG backbone, so none of the DISTS fp32-pinning
        # gymnastics are needed. It may still carry a radial-weight buffer,
        # so move it to the requested device.
        self._impl = create_loss("focal_frequency_radial")
        self._impl.to(self._device)
        self._impl.eval()

    @torch.inference_mode()  # type: ignore[misc, unused-ignore]
    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        preds = preds.to(self._device).float()
        target = target.to(self._device).float()
        # Defeat any outer autocast — spectral L2 numerics must stay fp32.
        with (
            torch.amp.autocast("cuda", enabled=False),
            torch.amp.autocast("cpu", enabled=False),
        ):
            return self._impl(preds, target).detach()


# ── #17 Mutual Information ──────────────────────────────────────────────────


@register_metric("mutual_information", aliases=["mi", "MI"])
class MutualInformationMetric(BaseMetric):
    r"""Parzen-window mutual information :math:`I(\hat{x}; x)`.

    Higher is better. Strategies should min-max normalise both inputs
    to [0, 1] (or pass already-normalised tensors).
    """

    def __init__(self, bins: int = 64, device: str | torch.device = "cpu"):
        super().__init__(device=device)
        self.bins = int(bins)

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        a, b = _flatten_pair(preds, target)
        # Normalise to [0, 1] for stable binning
        a = (a - a.min()) / (np.ptp(a) + 1e-12)
        b = (b - b.min()) / (np.ptp(b) + 1e-12)
        h, _, _ = np.histogram2d(a, b, bins=self.bins)
        p = h / max(h.sum(), 1.0)
        px = p.sum(axis=1, keepdims=True)
        py = p.sum(axis=0, keepdims=True)
        nz = p > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(nz, p / (px @ py).clip(min=1e-12), 1.0)
            mi = float((p[nz] * np.log(ratio[nz])).sum())
        return torch.tensor(mi, device=self.device)


# ── #21 Edge-Preservation Index ─────────────────────────────────────────────


@register_metric("edge_preservation_index", aliases=["epi", "EPI"])
class EdgePreservationIndex(BaseMetric):
    r"""Edge-preservation index — Pearson correlation of intensity gradients
    on a Sobel-edge mask. Higher is better, range :math:`[-1, 1]`.
    """

    def __init__(self, edge_threshold: float = 0.1, device: str | torch.device = "cpu"):
        super().__init__(device=device)
        self.edge_threshold = float(edge_threshold)

    def _grad_mag(self, x: np.ndarray) -> np.ndarray:
        gx = np.gradient(x, axis=-1)
        gy = np.gradient(x, axis=-2)
        return np.sqrt(gx * gx + gy * gy)

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        p, t = _to_numpy(preds), _to_numpy(target)
        gp, gt = self._grad_mag(p), self._grad_mag(t)
        edge_mask = gt >= self.edge_threshold * gt.max()
        if not edge_mask.any():
            return torch.tensor(float("nan"), device=self.device)
        a, b = gp[edge_mask], gt[edge_mask]
        if a.std() < 1e-12 or b.std() < 1e-12:
            return torch.tensor(float("nan"), device=self.device)
        rho = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        return torch.tensor(rho, device=self.device)


# ── #23 Radial k-space Error ────────────────────────────────────────────────


@register_metric("radial_k_error", aliases=["radial_kspace_error"])
class RadialKSpaceError(BaseMetric):
    r"""Mean ``|F·x̂ - F·x|`` binned by radial k-space frequency.

    Returns a scalar — the mean over the radial bins. Lower is better.
    Use the plotter ``mri_recon_panel`` (its radial k-error column) for the per-radius curve.
    """

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        p = _to_numpy(preds).astype(np.float64).squeeze()
        t = _to_numpy(target).astype(np.float64).squeeze()
        # Take a 2D slice if a 3D volume was passed
        while p.ndim > 2:
            p = p[p.shape[0] // 2]
            t = t[t.shape[0] // 2]
        fp = np.fft.fftshift(np.fft.fft2(p))
        ft = np.fft.fftshift(np.fft.fft2(t))
        err = np.abs(fp - ft)
        h, w = err.shape
        cy, cx = h // 2, w // 2
        y, x = np.indices(err.shape)
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
        # Divide sums by counts only where a radius is populated. For a
        # non-square (H != W) grid some integer radii have zero count, so the
        # naive ``sums / counts`` produced 0/0 = NaN bins and the subsequent
        # mean() returned NaN even for a perfect reconstruction.
        counts = np.bincount(r.ravel())
        sums = np.bincount(r.ravel(), weights=err.ravel())
        binned = np.divide(
            sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=counts > 0
        )
        return torch.tensor(float(np.nanmean(binned[1:])), device=self.device)


# ── #28 Average Surface Distance ────────────────────────────────────────────


@register_metric("average_surface_distance", aliases=["asd", "ASD"])
class AverageSurfaceDistanceMetric(BaseMetric):
    r"""Mean of all surface-to-surface distances between two binary masks.

    Lower is better. Both inputs are binarised by the supplied threshold
    (or are assumed binary already). Falls back to NaN when either mask
    is empty.
    """

    def __init__(self, threshold: float = 0.5, device: str | torch.device = "cpu"):
        super().__init__(device=device)
        self.threshold = float(threshold)

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        from scipy.ndimage import distance_transform_edt as edt

        a = _to_numpy(preds) > self.threshold
        b = _to_numpy(target) > self.threshold
        if not a.any() or not b.any():
            return torch.tensor(float("nan"), device=self.device)
        d_ab = edt(~b)[a]
        d_ba = edt(~a)[b]
        asd = float(np.concatenate([d_ab, d_ba]).mean())
        return torch.tensor(asd, device=self.device)


# ── #29 Target Registration Error ───────────────────────────────────────────


@register_metric("target_registration_error", aliases=["tre", "TRE"])
class TargetRegistrationErrorMetric(BaseMetric):
    r"""Mean Euclidean distance between paired anatomical landmarks.

    Forward signature is non-standard: ``preds`` and ``target`` here are
    arrays of *landmark coordinates* of shape ``(K, ndim)``. Returns the
    mean distance in voxel units.
    """

    INPUT_SIGNATURE = "landmark_pair"

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        p = _to_numpy(preds).reshape(-1, _to_numpy(preds).shape[-1])
        t = _to_numpy(target).reshape(-1, _to_numpy(target).shape[-1])
        if p.shape != t.shape:
            return torch.tensor(float("nan"), device=self.device)
        d = np.linalg.norm(p - t, axis=-1).mean()
        return torch.tensor(float(d), device=self.device)


# ── #30 Folding Fraction ────────────────────────────────────────────────────


@register_metric("folding_fraction", aliases=["jacobian_negative_fraction"])
class FoldingFraction(BaseMetric):
    r"""Fraction of voxels with ``det(J(φ)) ≤ 0`` in a deformation field.

    Forward signature: ``preds`` is the displacement field
    ``(B, ndim, *spatial)``. ``target`` is ignored. Lower is better
    (must be ≈ 0 to claim a diffeomorphism).
    """

    INPUT_SIGNATURE = "displacement_field"

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        field = _to_numpy(preds)
        if field.ndim < 4:
            return torch.tensor(float("nan"), device=self.device)
        # 2D field shape = (B, 2, H, W); 3D = (B, 3, D, H, W).
        b = field.shape[0]
        ndim = field.shape[1]
        spatial_shape = field.shape[2:]
        # Build per-voxel Jacobian J[b, i, j, *spatial] ≈ ∂φ_i/∂x_j.
        jac = np.empty((b, ndim, ndim) + spatial_shape)
        for i in range(ndim):
            comp = field[:, i]  # shape (B, *spatial); spatial axes start at index 1
            for j in range(ndim):
                jac[:, i, j] = np.gradient(comp, axis=1 + j)
        # Add identity to displacement Jacobian → transformation Jacobian.
        jac += np.eye(ndim).reshape((1, ndim, ndim) + (1,) * len(spatial_shape))
        # Move the (i, j) axes to the end so np.linalg.det operates on the
        # last two dims, leaving (B, *spatial) determinants.
        det = np.linalg.det(np.moveaxis(jac, (1, 2), (-2, -1)))
        return torch.tensor(float((det <= 0).mean()), device=self.device)


# ── #31 DVF MAE ─────────────────────────────────────────────────────────────


@register_metric("dvf_mae", aliases=["DVFMAE", "displacement_field_mae"])
class DVFMAEMetric(BaseMetric):
    r"""Mean abs error between predicted and ground-truth displacement fields.

    Lower is better. ``preds`` and ``target`` are dense fields of the
    same shape ``(B, ndim, *spatial)``.
    """

    INPUT_SIGNATURE = "displacement_field"

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        return torch.tensor(
            float(np.abs(_to_numpy(preds) - _to_numpy(target)).mean()),
            device=self.device,
        )


# ── #32 Through-Plane FWHM ──────────────────────────────────────────────────


@register_metric("through_plane_fwhm", aliases=["tp_fwhm", "z_fwhm"])
class ThroughPlaneFWHM(BaseMetric):
    r"""FWHM of the line-spread function along the through-plane axis.

    Approximates real (not nominal) z-resolution. Smaller is sharper
    (closer to the target slice profile). Forward expects a 3D volume
    ``(D, H, W)`` (or ``(B, C, D, H, W)`` with the through-plane axis
    along ``D``).
    """

    INPUT_SIGNATURE = "volume_3d"

    def __init__(self, axis: int = -3, device: str | torch.device = "cpu"):
        super().__init__(device=device)
        self.axis = axis

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        v = _to_numpy(preds)
        # Reduce to the chosen axis line spread function
        # via central-line average across the orthogonal plane.
        while v.ndim > 3:
            v = v[0]
        # v: (D, H, W). Take the 1-D profile along D at the centre voxel.
        cy = v.shape[1] // 2
        cx = v.shape[2] // 2
        line = v[:, cy, cx].astype(np.float64)
        # ESF → LSF via numerical derivative
        lsf = np.gradient(line)
        lsf = np.abs(lsf - lsf.mean())
        if lsf.max() < 1e-12:
            return torch.tensor(float("nan"), device=self.device)
        half = lsf.max() / 2.0
        idx = np.where(lsf >= half)[0]
        fwhm = float(idx[-1] - idx[0]) if idx.size > 1 else float("nan")
        return torch.tensor(fwhm, device=self.device)


# ── #34/#35 Bland–Altman bias + Limits of Agreement ─────────────────────────


@register_metric("bland_altman_bias", aliases=["ba_bias"])
class BlandAltmanBias(BaseMetric):
    r"""Mean of ``(\hat{x} − x)`` across all elements (target → 0)."""

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        diff = _to_numpy(preds) - _to_numpy(target)
        return torch.tensor(float(diff.mean()), device=self.device)


@register_metric("limits_of_agreement_upper", aliases=["loa_upper"])
class LimitsOfAgreementUpper(BaseMetric):
    r"""Bland-Altman upper limit of agreement: :math:`\bar d + 1.96 s_d`."""

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        diff = _to_numpy(preds) - _to_numpy(target)
        return torch.tensor(
            float(diff.mean() + 1.96 * diff.std(ddof=1) if diff.size > 1 else 0.0),
            device=self.device,
        )


@register_metric("limits_of_agreement_lower", aliases=["loa_lower"])
class LimitsOfAgreementLower(BaseMetric):
    r"""Bland-Altman lower limit of agreement: :math:`\bar d − 1.96 s_d`."""

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        diff = _to_numpy(preds) - _to_numpy(target)
        return torch.tensor(
            float(diff.mean() - 1.96 * diff.std(ddof=1) if diff.size > 1 else 0.0),
            device=self.device,
        )


@register_metric("abs_bland_altman_bias", aliases=["abs_ba_bias"])
class AbsBlandAltmanBias(BaseMetric):
    r"""Size of the Bland-Altman bias, :math:`|\bar d|`: darkening and brightening both raise it."""

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        diff = _to_numpy(preds) - _to_numpy(target)
        return torch.tensor(abs(float(diff.mean())), device=self.device)


@register_metric("limits_of_agreement_halfwidth", aliases=["loa_halfwidth"])
class LimitsOfAgreementHalfwidth(BaseMetric):
    r"""Half-width of the Bland-Altman limits of agreement, :math:`1.96 s_d`."""

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        diff = _to_numpy(preds) - _to_numpy(target)
        return torch.tensor(
            float(1.96 * diff.std(ddof=1) if diff.size > 1 else 0.0), device=self.device
        )


# ── #36 ICC(3,1) ─────────────────────────────────────────────────────────────


@register_metric("icc_3_1", aliases=["icc", "ICC", "icc_two_way"])
class ICC3_1Metric(BaseMetric):
    r"""Intra-class correlation coefficient ICC(3,1) (Shrout & Fleiss 1979).

    Treats ``preds`` and ``target`` as two raters scoring the same
    subjects. Higher is better, range :math:`[-1, 1]`. Used for
    test-retest repeatability claims in qMRI.
    """

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        a = _to_numpy(preds).ravel().astype(np.float64)
        b = _to_numpy(target).ravel().astype(np.float64)
        if a.size != b.size or a.size < 2:
            return torch.tensor(float("nan"), device=self.device)
        n = a.size
        k = 2
        # Two-way mixed-effects, single-rater, absolute-agreement (ICC3,1)
        ratings = np.stack([a, b], axis=1)  # (n, 2)
        mean_subjects = ratings.mean(axis=1)
        mean_raters = ratings.mean(axis=0)
        grand_mean = ratings.mean()
        ms_subjects = k * np.sum((mean_subjects - grand_mean) ** 2) / (n - 1)
        ms_error = np.sum(
            (ratings - mean_subjects[:, None] - mean_raters[None] + grand_mean) ** 2
        ) / ((n - 1) * (k - 1))
        denom = ms_subjects + (k - 1) * ms_error
        if denom < 1e-12:
            return torch.tensor(float("nan"), device=self.device)
        icc = float((ms_subjects - ms_error) / denom)
        return torch.tensor(icc, device=self.device)


# ── #37 Coefficient of Variation ────────────────────────────────────────────


@register_metric("coefficient_of_variation", aliases=["cv", "CV"])
class CoefficientOfVariation(BaseMetric):
    r"""CV = σ / |μ| computed on the prediction tensor.

    ``target`` is ignored — CV is a single-distribution metric used in
    qMRI biomarker repeatability claims. Lower is better.
    """

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        v = _to_numpy(preds).astype(np.float64)
        m = float(np.abs(v.mean()))
        if m < 1e-12:
            return torch.tensor(float("nan"), device=self.device)
        return torch.tensor(float(v.std(ddof=1) / m), device=self.device)


# ── #39 Detection sensitivity / specificity ─────────────────────────────────


@register_metric("detection_sensitivity", aliases=["sensitivity", "recall"])
class DetectionSensitivity(BaseMetric):
    r"""TP / (TP + FN) on a thresholded binary detection task."""

    def __init__(self, threshold: float = 0.5, device: str | torch.device = "cpu"):
        super().__init__(device=device)
        self.threshold = float(threshold)

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        p = (_to_numpy(preds) >= self.threshold).astype(np.int8)
        t = (_to_numpy(target) >= self.threshold).astype(np.int8)
        tp = int(((p == 1) & (t == 1)).sum())
        fn = int(((p == 0) & (t == 1)).sum())
        denom = tp + fn
        if denom == 0:
            return torch.tensor(float("nan"), device=self.device)
        return torch.tensor(tp / denom, device=self.device)


@register_metric("detection_specificity", aliases=["specificity"])
class DetectionSpecificity(BaseMetric):
    r"""TN / (TN + FP) on a thresholded binary detection task."""

    def __init__(self, threshold: float = 0.5, device: str | torch.device = "cpu"):
        super().__init__(device=device)
        self.threshold = float(threshold)

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        p = (_to_numpy(preds) >= self.threshold).astype(np.int8)
        t = (_to_numpy(target) >= self.threshold).astype(np.int8)
        tn = int(((p == 0) & (t == 0)).sum())
        fp = int(((p == 1) & (t == 0)).sum())
        denom = tn + fp
        if denom == 0:
            return torch.tensor(float("nan"), device=self.device)
        return torch.tensor(tn / denom, device=self.device)


# ── #40 AUROC / AUPRC ───────────────────────────────────────────────────────


@register_metric("auroc", aliases=["AUROC", "roc_auc"])
class AUROCMetric(BaseMetric):
    r"""Area under the ROC curve. ``preds`` are continuous scores."""

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        from sklearn.metrics import roc_auc_score

        p = _to_numpy(preds).ravel()
        t = _to_numpy(target).ravel().astype(int)
        try:
            return torch.tensor(float(roc_auc_score(t, p)), device=self.device)
        except ValueError:
            return torch.tensor(float("nan"), device=self.device)


@register_metric("auprc", aliases=["AUPRC", "average_precision"])
class AUPRCMetric(BaseMetric):
    r"""Area under the Precision-Recall curve."""

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        from sklearn.metrics import average_precision_score

        p = _to_numpy(preds).ravel()
        t = _to_numpy(target).ravel().astype(int)
        try:
            return torch.tensor(float(average_precision_score(t, p)), device=self.device)
        except ValueError:
            return torch.tensor(float("nan"), device=self.device)


# ── #41 Expected Calibration Error ──────────────────────────────────────────


@register_metric("expected_calibration_error", aliases=["ece", "ECE"])
class ExpectedCalibrationError(BaseMetric):
    r"""ECE = :math:`\sum_b \tfrac{|B_b|}{N}\,|\mathrm{acc}(B_b) - \mathrm{conf}(B_b)|`.

    Lower is better. ``preds`` are predicted probabilities for the
    positive class (or top-1 confidence in multi-class), ``target`` are
    binary correct/incorrect indicators (or labels matching argmax).
    """

    def __init__(self, n_bins: int = 15, device: str | torch.device = "cpu"):
        super().__init__(device=device)
        self.n_bins = int(n_bins)

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        confs = _to_numpy(preds).ravel().astype(np.float64).clip(0, 1)
        correct = _to_numpy(target).ravel().astype(np.float64)
        if confs.size != correct.size or confs.size == 0:
            return torch.tensor(float("nan"), device=self.device)
        bins = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            in_bin = (confs >= lo) & (confs < hi if hi < 1 else confs <= hi)
            if not in_bin.any():
                continue
            avg_conf = confs[in_bin].mean()
            avg_acc = correct[in_bin].mean()
            ece += (in_bin.sum() / confs.size) * abs(avg_acc - avg_conf)
        return torch.tensor(float(ece), device=self.device)


# ── #42 NLL / bits-per-dim ──────────────────────────────────────────────────


@register_metric("nll_bits_per_dim", aliases=["bits_per_dim", "bpd", "nll"])
class NLLBitsPerDim(BaseMetric):
    r""":math:`-\frac{1}{N}\sum \log p(x) / \log 2` from a per-element NLL tensor.

    Forward signature: ``preds`` is the per-element negative log-likelihood
    in nats. ``target`` is ignored. Standard for likelihood-based
    generative models (flows, autoregressive, diffusion ELBO).
    """

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        nll = _to_numpy(preds).astype(np.float64)
        bpd = float(nll.mean() / math.log(2.0))
        return torch.tensor(bpd, device=self.device)


# ── #45 Wasserstein-1 / Sliced Wasserstein ─────────────────────────────────


@register_metric("wasserstein_1d", aliases=["wasserstein_1", "w1"])
class Wasserstein1D(BaseMetric):
    r"""1-D Wasserstein distance between flattened distributions."""

    # A distribution-vs-distribution metric: it compares two point sets that
    # legitimately differ in cardinality (e.g. N degraded vs M clean images). Free
    # the sample axis only — clearing REQUIRES_MATCHING_SHAPES would also stop
    # checking the trailing dims, and _flatten_pair ravels both tensors, so a
    # [B, 2, H, W] head would score a finite number against a [B, 1, H, W] target.
    ALLOWS_UNEQUAL_SAMPLE_COUNT: bool = True

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        from scipy.stats import wasserstein_distance

        a, b = _flatten_pair(preds, target)
        d = float(wasserstein_distance(a, b))
        return torch.tensor(d, device=self.device)


@register_metric("sliced_wasserstein", aliases=["sw", "swd"])
class SlicedWassersteinMetric(BaseMetric):
    r"""Mean of 1-D Wasserstein distances over random projections.

    Approximates the high-dim Wasserstein distance at O(K·n log n)
    instead of O(n³). Useful for distributional fidelity claims.
    """

    # Distribution-vs-distribution: unequal cardinality is the normal case (#311).
    # Sample axis only; trailing dims stay checked.
    ALLOWS_UNEQUAL_SAMPLE_COUNT: bool = True

    def __init__(self, n_projections: int = 128, device: str | torch.device = "cpu"):
        super().__init__(device=device)
        self.n_projections = int(n_projections)

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        from scipy.stats import wasserstein_distance

        p = _to_numpy(preds).reshape(_to_numpy(preds).shape[0], -1)
        t = _to_numpy(target).reshape(_to_numpy(target).shape[0], -1)
        d = p.shape[1]
        rng = np.random.default_rng(0)
        directions = rng.standard_normal((self.n_projections, d))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(1e-12)
        ws: list[float] = []
        for v in directions:
            ws.append(float(wasserstein_distance((p @ v).ravel(), (t @ v).ravel())))
        return torch.tensor(float(np.mean(ws)), device=self.device)


# ── #47 MMD as metric ──────────────────────────────────────────────────────


@register_metric("mmd_metric", aliases=["mmd_distance"])
class MMDMetric(BaseMetric):
    r"""Multi-bandwidth Gaussian-kernel MMD² as a feature-distribution
    distance. Re-uses the existing MMD loss for parity. Lower is better.
    """

    # Distribution-vs-distribution: unequal cardinality is the normal case (#311).
    # Sample axis only; trailing dims stay checked.
    ALLOWS_UNEQUAL_SAMPLE_COUNT: bool = True

    def __init__(
        self,
        sigmas: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
        device: str | torch.device = "cpu",
    ):
        super().__init__(device=device)
        from spectramr.models.losses import create_loss

        self._impl = create_loss("mmd", sigmas=sigmas)

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        return self._impl(preds, target).detach()


# ── #49 Noise residual whiteness ───────────────────────────────────────────


@register_metric("residual_whiteness", aliases=["whiteness", "noise_whiteness"])
class ResidualWhiteness(BaseMetric):
    r"""Spectral flatness of the noise residual.

    For a denoising prediction, the residual ``y − ĥ(y)`` should ideally
    be white (flat power spectrum) — the metric is the geometric mean of
    the radial PSD divided by its arithmetic mean (Wiener's spectral
    flatness). Higher is better, range :math:`[0, 1]`. Defends against
    over-smoothing.

    Forward signature: ``preds`` = denoiser output, ``target`` = noisy
    input. (Not the clean image.)
    """

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        residual = _to_numpy(target) - _to_numpy(preds)
        residual = residual.astype(np.float64).squeeze()
        while residual.ndim > 2:
            residual = residual[residual.shape[0] // 2]
        f = np.fft.fftshift(np.fft.fft2(residual))
        psd = np.abs(f) ** 2 + 1e-12
        # Spectral flatness = exp(mean(log psd)) / mean(psd)
        sf = float(np.exp(np.mean(np.log(psd))) / np.mean(psd))
        return torch.tensor(sf, device=self.device)


# ── #50 Cohen's κ for inter-rater agreement ─────────────────────────────────


@register_metric("cohen_kappa", aliases=["kappa", "cohens_kappa"])
class CohenKappa(BaseMetric):
    r"""Cohen's κ inter-rater agreement, weighted variant for ordinal Likert.

    Forward signature: ``preds`` and ``target`` are arrays of integer
    class labels (one per item) from two raters. Returns the weighted
    κ when ``weights="quadratic"`` (Likert-friendly) or unweighted when
    ``weights=None``.
    """

    def __init__(self, weights: str | None = "quadratic", device: str | torch.device = "cpu"):
        super().__init__(device=device)
        self.weights = weights

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        from sklearn.metrics import cohen_kappa_score

        a = _to_numpy(preds).ravel().astype(int)
        b = _to_numpy(target).ravel().astype(int)
        try:
            return torch.tensor(
                float(cohen_kappa_score(a, b, weights=self.weights)),
                device=self.device,
            )
        except ValueError:
            return torch.tensor(float("nan"), device=self.device)
