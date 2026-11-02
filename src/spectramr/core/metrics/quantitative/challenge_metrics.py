"""MRIxFields2026 leaderboard-parity metrics.

Definitions mirror ``Submission/evaluation-2026/score.py`` of the official
challenge repo (full volume, no brain mask, images in [0,1]).
"""

from __future__ import annotations

import numpy as np
import torch

from spectramr.core.metrics.quantitative.segmentation import LabelDiceBackend
from spectramr.core.metrics.registry import register_metric

_EPS = 1e-10


@register_metric("nrmse_l2", aliases=["NRMSE_L2", "nrmse_official"])
class NRMSEL2Metric:
    """Official nRMSE: ``||pred - target||_2 / ||target||_2`` (0 if target ~0).

    Matches the MRIxFields2026 challenge evaluation script exactly.
    Our existing ``nrmse`` metric normalises by the signal range
    (``max - min``), which does NOT match the leaderboard score; this metric
    does.

    Direction (lower-is-better) is declared centrally in
    ``core/metrics/metric_directions.py::METRIC_HIGHER_IS_BETTER`` and
    injected by ``@register_metric`` at registration time.  Do not add a
    ``higher_is_better`` class attribute here — that would create a duplicate
    declaration and silently shadow the map (see registry.py declares_own logic).
    """

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, **_: object) -> float:
        num = torch.linalg.vector_norm(pred - target)
        den = torch.linalg.vector_norm(target)
        if float(den) < _EPS:
            return 0.0
        return float(num / den)


# ---------------------------------------------------------------------------
# volume_consistency
# ---------------------------------------------------------------------------

#: FreeSurfer ``aseg`` IDs of the 14 deep grey-matter nuclei scored by the
#: MRIxFields2026 challenge: thalamus, caudate, putamen, pallidum,
#: hippocampus, amygdala, accumbens (left/right pairs).
DGM_LABELS_14: tuple[int, ...] = (10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58)


def _label_volume(seg: np.ndarray | torch.Tensor, label: int) -> float:
    """Count voxels equal to *label* in *seg* (works on numpy arrays and torch tensors)."""
    if isinstance(seg, torch.Tensor):
        return float((seg == label).sum().item())
    return float((np.asarray(seg) == label).sum())


@register_metric("volume_consistency", aliases=["volume_similarity", "VolumeConsistency"])
class VolumeConsistencyMetric:
    """Official per-label volume consistency, mean over the 14 DGM nuclei.

    Formula per label ``l``:

    .. code-block:: text

        score(l) = 1 - |V_pred(l) - V_gt(l)| / V_gt(l)

    Edge cases:
    - ``V_gt < 1e-10`` **and** ``V_pred < 1e-10`` → ``1.0`` (both absent).
    - ``V_gt < 1e-10`` (gt absent, pred non-empty) → ``0.0``.

    Volumes are voxel counts; the voxel-volume factor cancels in the ratio.
    The final score is the mean over all ``labels`` (default: :data:`DGM_LABELS_14`).

    Segmentation needs an injected segmenter that emits FreeSurfer aseg IDs (SynthSeg
    2.0 on the cluster).  There is no local stand-in: the intensity-quantile
    ``LabelDiceBackend`` cannot measure DGM volume agreement, so it and a missing
    segmenter both raise (#169).  The backend's ``segment(image)`` method is called
    with no ``labels`` argument.

    Direction (higher-is-better) is declared centrally in
    ``core/metrics/metric_directions.py::METRIC_HIGHER_IS_BETTER`` and injected
    by ``@register_metric``.  Do not add a ``higher_is_better`` class attribute
    here — that would shadow the map (see registry.py ``declares_own`` logic).
    """

    def __init__(
        self,
        segmenter: object = None,
        labels: tuple[int, ...] = DGM_LABELS_14,
    ) -> None:
        self.segmenter = segmenter
        self.labels = labels

    def _compute(
        self,
        pred_map: np.ndarray | torch.Tensor,
        gt_map: np.ndarray | torch.Tensor,
        labels: tuple[int, ...],
    ) -> float:
        """Core formula over two pre-computed label maps."""
        scores: list[float] = []
        for lab in labels:
            vp = _label_volume(pred_map, lab)
            vg = _label_volume(gt_map, lab)
            if vg < _EPS and vp < _EPS:
                scores.append(1.0)
            elif vg < _EPS:
                scores.append(0.0)
            else:
                scores.append(1.0 - abs(vp - vg) / vg)
        return float(np.mean(scores)) if scores else 0.0

    def score_from_segmenters(
        self,
        pred_seg: object,
        gt_seg: object,
        labels: tuple[int, ...] | None = None,
    ) -> float:
        """Compute the score from two pre-built segmenter objects.

        Calls ``pred_seg.segment(None)`` and ``gt_seg.segment(None)``; useful
        for unit-testing the formula with a stub segmenter that ignores its
        input and returns a pre-baked label map.

        The ``segment`` signature matches the real
        :class:`~spectramr.core.metrics.quantitative.segmentation.ISegmenter`
        protocol: ``segment(image)`` with NO ``labels`` argument.

        Args:
            pred_seg: segmenter for the predicted image.
            gt_seg: segmenter for the ground-truth image.
            labels: label IDs to score.  Defaults to ``self.labels`` (SSOT).
        """
        effective_labels = self.labels if labels is None else labels
        pred_map = pred_seg.segment(None)  # type: ignore[attr-defined]
        gt_map = gt_seg.segment(None)  # type: ignore[attr-defined]
        return self._compute(pred_map, gt_map, effective_labels)

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> float:
        seg = kwargs.get("segmenter", self.segmenter)
        if seg is None or isinstance(seg, LabelDiceBackend):
            raise ValueError(
                "volume_consistency needs a SynthSeg-style segmenter that emits FreeSurfer "
                "aseg IDs; none was injected, and the label_dice proxy cannot measure DGM "
                "volume agreement (#169). Omit the metric or inject a real segmenter."
            )
        pred_map = seg.segment(pred)  # type: ignore[attr-defined]
        target_map = seg.segment(target)  # type: ignore[attr-defined]
        return self._compute(pred_map, target_map, self.labels)


# ---------------------------------------------------------------------------
# lpips_alex
# ---------------------------------------------------------------------------


@register_metric("lpips_alex", aliases=["LPIPS_alex"])
class LPIPSAlexMetric:
    """MRIxFields2026 parity LPIPS using the AlexNet backbone.

    The official challenge scorer uses ``lpips.LPIPS(net="alex")`` with
    inputs mapped from ``[0, 1]`` to ``[-1, 1]`` and single-channel
    images repeated to 3 RGB channels.  For 3-D volumes (5-D tensors) the
    score is the mean over spatial slices along the depth axis.

    Direction (lower-is-better) is declared centrally in
    ``core/metrics/metric_directions.py::METRIC_HIGHER_IS_BETTER`` and
    injected by ``@register_metric``.  Do not add a ``higher_is_better``
    class attribute here — that would shadow the map (see registry.py
    ``declares_own`` logic and the regression test
    ``test_lpips_alex_direction_not_a_class_attribute``).
    """

    def __init__(self) -> None:
        import lpips

        self._net = lpips.LPIPS(net="alex")
        self._net.eval()
        for p in self._net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()  # type: ignore[misc]
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **_: object,
    ) -> float:
        """Compute mean AlexNet-LPIPS over all spatial slices."""

        def _prep(x: torch.Tensor) -> torch.Tensor:
            """Map [0, 1] → [-1, 1] and expand to RGB."""
            x = x.float().clamp(0.0, 1.0) * 2.0 - 1.0
            if x.dim() == 3:  # (C, H, W) → (1, C, H, W)
                x = x.unsqueeze(0)
            if x.shape[1] == 1:  # gray → RGB
                x = x.repeat(1, 3, 1, 1)
            return x

        device = pred.device
        self._net = self._net.to(device)

        if pred.dim() == 5:
            # (B, C, D, H, W) — iterate over depth slices
            b, c, d, h, w = pred.shape
            scores: list[float] = []
            for s in range(d):
                p_sl = _prep(pred[:, :, s, :, :])
                t_sl = _prep(target[:, :, s, :, :])
                scores.append(float(self._net(p_sl, t_sl).mean()))
            return float(sum(scores) / len(scores)) if scores else 0.0

        return float(self._net(_prep(pred), _prep(target)).mean())
