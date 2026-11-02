"""Tissue-segmentation agreement metrics for image synthesis / translation.

The ULF->HF cohort ships **no segmentation labels and no brain masks** (verified
against ``scripts/preprocessing/preprocess_ulf_paired.py``'s record schema, the
manifest key set, and ``UniversalMRIDataset``). So these metrics do not grade a
prediction against a reference *segmentation*; they segment the **prediction and
the target with the same deterministic segmenter** and measure whether the two
label maps agree. That is the standard downstream-task-consistency eval for
synthesis (the SynthSeg-Dice protocol used by SynthSR / LF-SynthSR), and it asks
the question PSNR cannot: *does the synthesized high-field image still support the
same tissue partition as the real one?*

Why not reuse ``synthseg_dice``'s :class:`LabelDiceBackend`? That proxy bins by
intensity **quantile**, so its classes are equal-population *by construction*: it
sees spatial arrangement only, is invariant to any monotone intensity change, and
its per-class volumes can never disagree. :class:`OtsuTissueSegmenter` derives its
thresholds from the histogram instead, so class sizes are data-driven and both
blur and volume error actually register. The two are complementary — run both.

Properties worth knowing before you cite these numbers:

* **Independent thresholds.** Prediction and target are each segmented with their
  *own* Otsu thresholds. The Dice is therefore invariant to a global *affine*
  intensity change (brightness/contrast), by design — it isolates structure, which
  PSNR/MAE already fail to isolate. Non-linear contrast errors and blur do move the
  thresholds and are penalised.
* **Slice-Dice, not volume-Dice.** Under ``data.slice_2d: true`` the validation loop
  hands these metrics one axial slice at a time, so the reported value is a mean of
  per-slice Dice, not a volumetric Dice. Small slices carry equal weight to large
  ones. Aggregate over a *representative* slice set (do not leave
  ``validation.num_validation_batches`` at a small N — the val loader is unshuffled,
  so a small N means "the first N adjacent slices of one volume").
* **"Tissue" is a proxy, and on ADC it is not tissue at all.** The 3-class partition
  approximates CSF/GM/WM on T1w/T2w/FLAIR. On an ADC map it is a generic intensity
  partition — still a valid agreement readout, but do not call those classes tissue.
  The names are deliberately neutral (``tissue_*``, never ``gm_*``/``wm_*``).

Never returns NaN: ``pipelines/train.py`` accumulates validation metrics with a bare
running sum and no non-finite guard, so a single NaN would poison the metric for the
whole evaluation.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from spectramr.core.metrics.evaluation_metrics import BaseMetric
from spectramr.core.metrics.monai_compat import call_surface_metric
from spectramr.core.metrics.registry import register_metric

logger = logging.getLogger(__name__)

#: Tissue classes excluding background (CSF / GM / WM proxy on T1w-like contrasts).
N_TISSUE_CLASSES = 3

#: Slices whose *target* brain mask covers less than this fraction of the FOV carry
#: no anatomy to grade and are dropped from the batch average.
MIN_FOREGROUND_FRACTION = 0.01

_HIST_BINS = 64

#: Upper edge of the Otsu histogram, as a quantile of the values being split.
#:
#: A ``[min, max]`` histogram is destroyed by a single hot voxel. SENSE/ESPIRiT
#: coil combination routinely leaves them near coil boundaries, and the pseudo-GT
#: path normalises by p99, so ``max`` *is* the outlier ratio: the 2026-07-25
#: fastMRI-brain sweep logged max/p99 between 1.5 and 1297, median 101. At 64
#: bins and a ratio of 299 every voxel of anatomy lands in bin 0, Otsu can no
#: longer resolve tissue, the brain mask comes back empty, and all four
#: segmentation metrics report their degenerate constant instead of a
#: measurement. Clipping the top 0.5% costs nothing on well-scaled images
#: (max/p99 there is ~1.6, so the clip is a no-op) and restores the histogram
#: on outlier-dominated ones.
_HIST_CLIP_QUANTILE = 0.995

#: ``torch.quantile`` refuses tensors beyond ~16M elements; above this we take a
#: strided (deterministic — never random) subsample to estimate the clip point.
_QUANTILE_MAX_ELEMS = 1 << 22


def _to_magnitude(x: Tensor) -> Tensor:
    """``[B, C, *spatial] -> [B, *spatial]`` magnitude, single channel."""
    if torch.is_complex(x):
        x = x.abs()
    if x.ndim >= 4 and x.shape[1] > 1:
        # Interleaved real/imag (even C) -> RSS; otherwise take the first channel.
        x = x.pow(2).sum(dim=1, keepdim=True).sqrt() if x.shape[1] % 2 == 0 else x[:, :1]
    return x[:, 0] if x.ndim >= 4 else x


def _robust_max(vals: Tensor) -> float:
    """Upper histogram edge that ignores the extreme tail.

    See :data:`_HIST_CLIP_QUANTILE` for why ``max`` is not usable here.
    """
    n = vals.numel()
    if n > _QUANTILE_MAX_ELEMS:
        vals = vals[:: (n // _QUANTILE_MAX_ELEMS) + 1]
    return float(torch.quantile(vals, _HIST_CLIP_QUANTILE))


def _otsu_thresholds(values: Tensor, n_classes: int) -> Tensor | None:
    """Otsu thresholds splitting *values* into ``n_classes`` bins.

    Range-agnostic: the histogram spans the tensor's own [min, max], so this works
    on ``[-1, 1]``-normalised MRI as well as ``[0, 1]``. ``qa_metrics.hd95``
    binarizes through this function too.

    Returns ``None`` when *values* is degenerate (empty or constant).
    """
    if values.numel() < n_classes:
        return None
    vals = values.detach().float().flatten()
    vmin = float(vals.min())
    vmax = _robust_max(vals)
    if not (vmax > vmin):
        # The top 0.5% *is* the spread (near-binary masks, mostly-constant
        # patches). Fall back to the true max rather than declaring degenerate.
        vmax = float(vals.max())
    if not (vmax > vmin):
        return None

    # Clamp rather than let ``histc`` drop out-of-range values: the hot tail is
    # real signal and must keep its mass, it just must not set the bin width.
    hist = torch.histc(vals.clamp(max=vmax), bins=_HIST_BINS, min=vmin, max=vmax)
    width = (vmax - vmin) / _HIST_BINS
    centers = vmin + width * (torch.arange(_HIST_BINS, device=vals.device) + 0.5)

    p = hist / hist.sum().clamp_min(1.0)
    w = torch.cumsum(p, dim=0)  # w[i] = mass of bins 0..i
    m = torch.cumsum(p * centers, dim=0)  # m[i] = first moment of bins 0..i

    # Maximising the between-class variance is equivalent to maximising
    # sum_k m_k^2 / w_k (the mu_t^2 term is constant), which avoids a division
    # per class mean.
    def _score(masses: list[Tensor], moments: list[Tensor]) -> Tensor:
        total = torch.zeros_like(masses[0])
        for wk, mk in zip(masses, moments, strict=True):
            total = total + mk.pow(2) / wk.clamp_min(1e-12)
        return total

    if n_classes == 2:
        score = _score([w, 1.0 - w], [m, m[-1] - m])
        k = int(score[:-1].argmax())
        return centers[k].reshape(1)

    if n_classes != 3:
        raise ValueError(f"n_classes must be 2 or 3; got {n_classes}")

    # Exhaustive search over threshold pairs (i < j) — 64x64 is trivially cheap.
    idx = torch.arange(_HIST_BINS, device=vals.device)
    i, j = torch.meshgrid(idx, idx, indexing="ij")
    w0, m0 = w[i], m[i]
    w1, m1 = w[j] - w[i], m[j] - m[i]
    w2, m2 = 1.0 - w[j], m[-1] - m[j]

    score = _score([w0, w1, w2], [m0, m1, m2])
    # Only i < j is a valid 3-way split, and j must leave a non-empty top class.
    score = score.masked_fill(i >= j, float("-inf"))
    score = score.masked_fill(j >= _HIST_BINS - 1, float("-inf"))
    if not torch.isfinite(score).any():
        return None

    flat = int(score.argmax())
    bi, bj = flat // _HIST_BINS, flat % _HIST_BINS
    return torch.stack([centers[bi], centers[bj]])


class OtsuTissueSegmenter:
    """Deterministic Otsu brain mask + 3-class tissue partition inside it.

    ``segment(image)`` maps ``[B, C, *spatial]`` to an integer label map
    ``[B, *spatial]`` with ``0 = background`` and ``1..N_TISSUE_CLASSES`` ordered by
    increasing intensity. Pure torch, no learned weights, no new dependencies — so
    the same segmenter runs identically on the prediction and the target, which is
    the whole point (any bias it has cancels in the comparison).
    """

    n_classes: int = N_TISSUE_CLASSES + 1  # background + tissue classes

    def brain_mask(self, image: Tensor) -> Tensor:
        """Otsu foreground (head/brain) mask; ``[B, *spatial]`` bool."""
        mag = _to_magnitude(image).detach().float()
        mask = torch.zeros_like(mag, dtype=torch.bool)
        for b in range(mag.shape[0]):
            thr = _otsu_thresholds(mag[b], n_classes=2)
            if thr is not None:
                mask[b] = mag[b] > thr[0]
        return mask

    def segment(self, image: Tensor) -> Tensor:
        """Label map ``[B, *spatial]``: 0 background, 1..3 tissue by intensity."""
        mag = _to_magnitude(image).detach().float()
        mask = self.brain_mask(image)
        labels = torch.zeros_like(mag, dtype=torch.long)

        for b in range(mag.shape[0]):
            fg = mask[b]
            if not bool(fg.any()):
                continue
            thr = _otsu_thresholds(mag[b][fg], n_classes=N_TISSUE_CLASSES)
            if thr is None:
                # Foreground exists but is constant — one tissue class, not zero.
                labels[b][fg] = 1
                continue
            labels[b][fg] = torch.bucketize(mag[b][fg], thr) + 1
        return labels


def _gradeable(target: Tensor, segmenter: OtsuTissueSegmenter) -> Tensor:
    """Per-image bool: does the TARGET carry enough brain to be worth grading?

    Near-empty slices (the top/bottom of an axial stack) contain no anatomy, so
    scoring them would report the segmenter's behaviour on noise rather than the
    model's. They are dropped from the batch average.
    """
    mask = segmenter.brain_mask(target)
    frac = mask.flatten(1).float().mean(dim=1)
    return frac >= MIN_FOREGROUND_FRACTION


def _warn_degenerate(name: str) -> None:
    logger.warning(
        "[%s] every image in this validation batch has an (almost) empty target "
        "brain mask, so there is no anatomy to grade; reporting the conservative "
        "value. This usually means validation is sampling background slices — with "
        "an unshuffled val loader, a small `validation.num_validation_batches` "
        "grades the first N adjacent slices of one volume.",
        name,
    )


class _SegmentationAgreementMetric(BaseMetric):
    """Shared plumbing: segment pred + target, score only the gradeable images."""

    INPUT_SIGNATURE = "image_pair"

    def __init__(self, device: str | torch.device = "cpu", **_: object) -> None:
        super().__init__(device=device)
        self._segmenter = OtsuTissueSegmenter()

    def _per_image(self, pred: Tensor, target: Tensor) -> Tensor:
        """``[B]`` per-image scores. Implemented by each concrete metric."""
        raise NotImplementedError

    #: Value reported when no image in the batch is gradeable.
    _DEGENERATE_VALUE: float = 0.0

    def compute_metric(self, preds: Tensor, target: Tensor, **_: object) -> float:
        keep = _gradeable(target, self._segmenter)
        if not bool(keep.any()):
            _warn_degenerate(type(self).__name__)
            return self._DEGENERATE_VALUE

        scores = self._per_image(preds, target)[keep]
        value = float(scores.mean())
        # Belt and braces: this metric must never emit NaN (train.py accumulates
        # validation metrics with a bare running sum, so one NaN poisons the eval).
        if not torch.isfinite(torch.tensor(value)):
            _warn_degenerate(type(self).__name__)
            return self._DEGENERATE_VALUE
        return value


def _dice_per_image(seg_a: Tensor, seg_b: Tensor, class_ids: range) -> Tensor:
    """Mean Dice over classes present in either map; ``[B]``."""
    b = seg_a.shape[0]
    out = torch.zeros(b, device=seg_a.device, dtype=torch.float32)
    present = torch.zeros_like(out)
    for c in class_ids:
        a = (seg_a == c).reshape(b, -1).float()
        bb = (seg_b == c).reshape(b, -1).float()
        inter = (a * bb).sum(dim=1)
        denom = a.sum(dim=1) + bb.sum(dim=1)
        # A class absent from BOTH maps contributes nothing — crediting it a free
        # Dice of 1.0 would inflate the score and mask a collapsed prediction.
        out += torch.where(denom > 0, 2.0 * inter / denom, torch.zeros_like(denom))
        present += (denom > 0).float()
    return out / present.clamp_min(1.0)


@register_metric("tissue_dice", aliases=["TissueDice"])
class TissueDiceMetric(_SegmentationAgreementMetric):
    """Mean Dice over the 3 Otsu tissue classes, seg(pred) vs seg(target).

    The headline structural-fidelity number: does the synthesized image still
    partition into the same tissue regions as the real one?
    """

    higher_is_better = True

    def _per_image(self, pred: Tensor, target: Tensor) -> Tensor:
        return _dice_per_image(
            self._segmenter.segment(pred),
            self._segmenter.segment(target),
            range(1, N_TISSUE_CLASSES + 1),
        )


@register_metric("brain_mask_dice", aliases=["BrainMaskDice"])
class BrainMaskDiceMetric(_SegmentationAgreementMetric):
    """Dice of the Otsu brain masks — does the predicted head occupy the same FOV?

    Catches the gross failures (shrunken//bloated anatomy, a DC-blob output whose
    support does not match the measurement) that a tissue-interior Dice can miss.
    """

    higher_is_better = True

    def _per_image(self, pred: Tensor, target: Tensor) -> Tensor:
        mask_p = self._segmenter.brain_mask(pred).long()
        mask_t = self._segmenter.brain_mask(target).long()
        return _dice_per_image(mask_p, mask_t, range(1, 2))


@register_metric("tissue_volume_similarity", aliases=["TissueVolumeSimilarity"])
class TissueVolumeSimilarityMetric(_SegmentationAgreementMetric):
    """Mean per-class volume agreement ``1 - |V_pred - V_target| / V_target``.

    The honest counterpart to ``volume_consistency``, which is degenerate on the
    quantile-proxy backend (its bins are equal-population by construction, so their
    volumes cannot disagree). Otsu class volumes are data-driven, so a prediction
    that over- or under-segments white matter is actually penalised here.

    Clamped to ``[0, 1]``: an unbounded negative score for a wildly over-segmented
    class would otherwise dominate the batch mean.
    """

    higher_is_better = True

    def _per_image(self, pred: Tensor, target: Tensor) -> Tensor:
        seg_p = self._segmenter.segment(pred)
        seg_t = self._segmenter.segment(target)
        b = seg_p.shape[0]
        out = torch.zeros(b, device=seg_p.device, dtype=torch.float32)
        present = torch.zeros_like(out)
        for c in range(1, N_TISSUE_CLASSES + 1):
            vp = (seg_p == c).reshape(b, -1).float().sum(dim=1)
            vt = (seg_t == c).reshape(b, -1).float().sum(dim=1)
            score = (1.0 - (vp - vt).abs() / vt.clamp_min(1e-8)).clamp(0.0, 1.0)
            # Score only classes the TARGET actually has — a class absent from the
            # reference has no volume to agree with.
            out += torch.where(vt > 0, score, torch.zeros_like(score))
            present += (vt > 0).float()
        return out / present.clamp_min(1.0)


@register_metric("tissue_hd95", aliases=["TissueHD95"])
class TissueHD95Metric(_SegmentationAgreementMetric):
    """95th-percentile Hausdorff distance between the tissue boundaries, in pixels.

    Boundary-sensitive where Dice is area-sensitive: a uniformly blurred prediction
    can hold a respectable Dice while its tissue interfaces drift. Reported in
    **pixels** (voxel spacing is not threaded through the validation loop), and
    in-plane only under ``slice_2d``.

    Uses MONAI's ``HausdorffDistanceMetric`` (already a core dependency), lazily
    imported to keep it off the import path — the pattern ``qa_metrics.HD95Metric``
    established.
    """

    higher_is_better = False

    # Two empty masks genuinely have no boundary between them, so a distance of 0
    # is the correct answer rather than a free win.
    _DEGENERATE_VALUE = 0.0

    def __init__(self, device: str | torch.device = "cpu", **_: object) -> None:
        super().__init__(device=device)
        self._monai_metric = None

    def _get_monai_metric(self) -> Any:
        if self._monai_metric is None:
            from monai.metrics import HausdorffDistanceMetric

            self._monai_metric = HausdorffDistanceMetric(
                include_background=False, percentile=95, directed=False
            )
        return self._monai_metric

    @staticmethod
    def _one_hot(seg: Tensor, n_classes: int) -> Tensor:
        """``[B, *spatial] -> [B, n_classes, *spatial]`` float one-hot."""
        oh = torch.nn.functional.one_hot(seg, num_classes=n_classes)
        # one_hot appends the class axis; move it to the channel position.
        return oh.movedim(-1, 1).float()

    def _per_image(self, pred: Tensor, target: Tensor) -> Tensor:
        n = self._segmenter.n_classes
        seg_p = self._segmenter.segment(pred)
        seg_t = self._segmenter.segment(target)
        oh_p = self._one_hot(seg_p, n)
        oh_t = self._one_hot(seg_t, n)

        metric = self._get_monai_metric()
        metric.reset()
        # -> [B, n_classes - 1] (include_background=False drops class 0)
        dist = torch.as_tensor(
            call_surface_metric(metric, y_pred=oh_p, y=oh_t), dtype=torch.float32
        ).reshape(
            seg_p.shape[0], -1
        )

        # MONAI returns nan/inf whenever a class is empty in ONE of the two maps.
        # Averaging over only the finite entries would then hand a COLLAPSED
        # prediction (no tissue classes at all -> every distance nan) a distance of
        # 0.0 — a perfect score for the exact failure this metric exists to catch.
        # A class the target has but the prediction lost is maximal disagreement, so
        # it scores the FOV diagonal (the largest distance the image admits).
        # Only a class absent from BOTH maps carries no boundary and is skipped.
        spatial = seg_p.shape[1:]
        diagonal = float(torch.tensor([float(s) for s in spatial]).pow(2).sum().sqrt())

        b = seg_p.shape[0]
        present_p = torch.stack([(seg_p == c).flatten(1).any(dim=1) for c in range(1, n)], dim=1)
        present_t = torch.stack([(seg_t == c).flatten(1).any(dim=1) for c in range(1, n)], dim=1)
        scored = present_p | present_t  # present in either -> a real boundary to grade

        dist = torch.where(torch.isfinite(dist), dist, torch.full_like(dist, diagonal))
        dist = torch.where(scored, dist, torch.zeros_like(dist))

        count = scored.float().sum(dim=1)
        # No class in either map: no boundary anywhere, so distance 0 is correct
        # (and such images are dropped by the foreground guard upstream anyway).
        return torch.where(
            count > 0,
            dist.sum(dim=1) / count.clamp_min(1.0),
            torch.zeros(b, device=dist.device),
        )


__all__ = [
    "MIN_FOREGROUND_FRACTION",
    "N_TISSUE_CLASSES",
    "BrainMaskDiceMetric",
    "OtsuTissueSegmenter",
    "TissueDiceMetric",
    "TissueHD95Metric",
    "TissueVolumeSimilarityMetric",
]
