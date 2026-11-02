"""Tests for MRIxFields2026 leaderboard-parity metrics (challenge_metrics.py).

Exercises:
  - ``nrmse_l2``: official L2-normalised nRMSE = ‖pred − target‖₂ / ‖target‖₂
  - ``volume_consistency``: official asymmetric per-label volume agreement over
    14 FreeSurfer deep-grey-matter nuclei.
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import pytest
import torch

from spectramr.core.metrics.quantitative import challenge_metrics as cm


def test_nrmse_l2_matches_official_formula():
    t = torch.tensor([[3.0, 4.0]])  # ||t||_2 = 5
    p = torch.tensor([[0.0, 4.0]])  # ||p-t||_2 = 3
    m = cm.NRMSEL2Metric()
    assert abs(m(p, t) - 0.6) < 1e-6  # 3/5
    assert m.higher_is_better is False


def test_nrmse_l2_zero_target_returns_zero():
    z = torch.zeros(1, 4)
    assert cm.NRMSEL2Metric()(z, z) == 0.0


# ---------------------------------------------------------------------------
# Stubs + helpers for volume_consistency tests
# ---------------------------------------------------------------------------


class _FakeSeg:
    """Test stub: segment() ignores the image and returns a pre-baked label map.

    Matches the real ISegmenter protocol: ``segment(image) -> label_map`` with
    NO ``labels`` argument (the real segmenter decides what labels it produces).
    """

    n_classes: int = 14

    def __init__(self, fs_labels: tuple, counts: list[int]) -> None:
        self._labels = fs_labels
        self.counts = counts

    def segment(self, img) -> np.ndarray:  # img ignored — it's a stub
        flat = np.zeros(1000, dtype=np.int64)
        off = 0
        for lab, n in zip(self._labels, self.counts):
            flat[off : off + n] = lab
            off += n
        return flat.reshape(10, 10, 10)


# ---------------------------------------------------------------------------
# volume_consistency tests
# ---------------------------------------------------------------------------


def test_volume_consistency_official_asymmetric_formula():
    """Numeric parity on the official formula with two labels."""
    # label 10: gt=100 vox, pred=90 vox  → 1 - |90-100|/100 = 0.9
    # label 11: gt=50  vox, pred=50 vox  → 1.0
    # mean = 0.95
    pred_seg = _FakeSeg((10, 11), [90, 50])
    gt_seg = _FakeSeg((10, 11), [100, 50])
    m = cm.VolumeConsistencyMetric(labels=(10, 11))
    val = m.score_from_segmenters(pred_seg=pred_seg, gt_seg=gt_seg, labels=(10, 11))
    assert abs(val - 0.95) < 1e-6


# Pattern that matches a class-level attribute assignment at the 4-space indent level.
# Docstrings mention "higher_is_better" in explanatory prose (inside triple-quoted
# strings), which appears at deeper indentation in source text and does NOT match.
# Only an actual class-body assignment (`    higher_is_better = ...` or
# `    higher_is_better: bool = ...`) would match — the regression we guard against.
_CLASS_ATTR_PATTERN = re.compile(r"^\s{4}higher_is_better\s*(?:=|:)", re.MULTILINE)


def test_volume_consistency_direction_higher_is_better():
    """Direction is injected from the map, NOT a class attribute."""
    m = cm.VolumeConsistencyMetric()
    assert m.higher_is_better is True
    # Regression guard: `@register_metric` legitimately injects cls.higher_is_better
    # on the class object after decoration (so __dict__ always contains it post-import).
    # The real regression to prevent is a class *author* hard-coding it in the class body
    # (which sets declares_own=True in the registry and bypasses the direction map).
    # We detect that by checking for a class-level assignment in the raw source.
    src = inspect.getsource(cm.VolumeConsistencyMetric)
    assert not _CLASS_ATTR_PATTERN.search(src), (
        "VolumeConsistencyMetric must not declare higher_is_better in the class body; "
        "direction is owned by the registry map."
    )


def test_nrmse_l2_direction_not_a_class_attribute():
    """Direction for NRMSEL2Metric must come from the registry map, not a class attribute."""
    src = inspect.getsource(cm.NRMSEL2Metric)
    assert not _CLASS_ATTR_PATTERN.search(src), (
        "NRMSEL2Metric must not declare higher_is_better in the class body; "
        "direction is owned by the registry map."
    )


def test_volume_consistency_both_empty_label_scores_one():
    """When both pred and gt have zero voxels for a label → score = 1.0."""
    # Only label 10 present (counts sum < 1000 → rest are 0-class background).
    # Label 99 is absent from both → 1.0.
    pred_seg = _FakeSeg((10,), [50])
    gt_seg = _FakeSeg((10,), [50])
    m = cm.VolumeConsistencyMetric(labels=(10, 99))
    val = m.score_from_segmenters(pred_seg=pred_seg, gt_seg=gt_seg, labels=(10, 99))
    # label 10: equal → 1.0; label 99: both empty → 1.0; mean = 1.0
    assert abs(val - 1.0) < 1e-6


def test_volume_consistency_gt_empty_pred_nonempty_scores_zero():
    """When gt has zero voxels but pred is non-empty → score = 0.0."""
    pred_seg = _FakeSeg((10,), [50])  # label 10 present in pred
    gt_seg = _FakeSeg((), [])  # label 10 absent in gt
    m = cm.VolumeConsistencyMetric(labels=(10,))
    val = m.score_from_segmenters(pred_seg=pred_seg, gt_seg=gt_seg, labels=(10,))
    assert abs(val - 0.0) < 1e-6


def test_volume_consistency_without_segmenter_raises():
    """#169: no injected segmenter must raise, never fall back to the label_dice proxy."""
    img = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError, match="volume_consistency"):
        cm.VolumeConsistencyMetric()(img, img)


def test_volume_consistency_rejects_label_dice_backend():
    """#169: the intensity-quantile proxy cannot measure DGM volume agreement."""
    from spectramr.core.metrics.quantitative.segmentation import LabelDiceBackend

    img = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError, match="label_dice"):
        cm.VolumeConsistencyMetric()(img, img, segmenter=LabelDiceBackend(n_classes=5))


def test_dgm_labels_14_are_the_freesurfer_ids():
    """DGM_LABELS_14 must be exactly the 14 FreeSurfer aseg IDs from the spec."""
    assert cm.DGM_LABELS_14 == (10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58)


# ---------------------------------------------------------------------------
# Task-3: volume_similarity flag → registered metric (un-inert check)
# ---------------------------------------------------------------------------


def test_volume_similarity_flag_resolves_to_registered_metric():
    """The compute_volume_similarity flag must map to a genuinely registered metric.

    Historical pitfall #16: ``volume_similarity`` was NOT registered, so enabling
    the flag produced a silent KeyError / NaN at validation.  Task 2 registered
    ``volume_consistency`` with alias ``volume_similarity``; this test asserts the
    alias resolves and the metric is callable.
    """
    from spectramr.core.metrics.registry import MetricsRegistry, get_metric

    # Alias must be present in the registry (Task-2 guarantee).
    assert MetricsRegistry.is_registered("volume_similarity"), (
        "volume_similarity is not registered; the flag compute_volume_similarity "
        "is still inert (pitfall #16 facade)."
    )
    # get_metric must return a VolumeConsistencyMetric instance (not raise).
    m = get_metric("volume_similarity")
    assert m is not None
    assert isinstance(
        m, cm.VolumeConsistencyMetric
    ), f"Expected VolumeConsistencyMetric, got {type(m)}"


# ---------------------------------------------------------------------------
# Task-4: lpips_alex
# ---------------------------------------------------------------------------


def _lpips_available() -> bool:
    """True if the ``lpips`` package AND AlexNet weights are loadable."""
    try:
        import lpips  # noqa: PLC0415

        lpips.LPIPS(net="alex")
        return True
    except (ImportError, OSError, RuntimeError):
        return False


_LPIPS_REASON = "lpips package not installed or AlexNet weights unavailable offline"


def test_lpips_alex_identical_images_is_zero():
    """LPIPSAlexMetric on identical images should return ~0."""
    if not _lpips_available():
        pytest.skip(_LPIPS_REASON)
    x = torch.rand(1, 1, 32, 32)
    m = cm.LPIPSAlexMetric()
    assert m(x, x) < 1e-4
    assert m.higher_is_better is False


def test_lpips_alex_different_images_positive():
    """LPIPSAlexMetric on different images should return a positive value."""
    if not _lpips_available():
        pytest.skip(_LPIPS_REASON)
    torch.manual_seed(42)
    x = torch.rand(1, 1, 32, 32)
    y = torch.rand(1, 1, 32, 32)
    m = cm.LPIPSAlexMetric()
    assert m(x, y) > 0.0


def test_lpips_alex_direction_not_a_class_attribute():
    """Direction for LPIPSAlexMetric must come from the registry map, not a class attribute."""
    src = inspect.getsource(cm.LPIPSAlexMetric)
    assert not _CLASS_ATTR_PATTERN.search(src), (
        "LPIPSAlexMetric must not declare higher_is_better in the class body; "
        "direction is owned by the registry map."
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda-only")
def test_lpips_alex_net_moves_to_cuda():
    """LPIPSAlexMetric must run without device mismatch when inputs are on CUDA."""
    if not _lpips_available():
        pytest.skip(_LPIPS_REASON)
    x = torch.rand(1, 1, 32, 32, device="cuda")
    m = cm.LPIPSAlexMetric()
    result = m(x, x)
    assert isinstance(result, float)
    assert result < 1e-4  # identical images → near-zero


def test_lpips_alex_is_registered():
    """lpips_alex and LPIPS_alex aliases must be present in the registry."""
    from spectramr.core.metrics.registry import MetricsRegistry

    assert MetricsRegistry.is_registered("lpips_alex"), "lpips_alex is not registered."
    assert MetricsRegistry.is_registered(
        "LPIPS_alex"
    ), "LPIPS_alex alias is not registered."


# ---------------------------------------------------------------------------
# Task-4: ssim parity against skimage (challenge script uses structural_similarity)
# ---------------------------------------------------------------------------


def test_ssim_matches_skimage_slicewise():
    """Our registered 'ssim' must agree with skimage slice-wise SSIM within 5e-2.

    The MRIxFields2026 challenge scorer computes:
        skimage.metrics.structural_similarity(target, pred, data_range=target.max()-target.min())

    If this test fails (diff > 5e-2), a dedicated SSIMChallengeMetric should be
    added (see task-4-brief.md).  The low observed difference (~0.002) means the
    existing SSIMMetric is challenge-faithful and no duplicate is needed.
    """
    skimage_metrics = pytest.importorskip("skimage.metrics")
    sk_ssim = skimage_metrics.structural_similarity
    import numpy as np
    import torch
    from spectramr.core.metrics.registry import get_metric

    rng = np.random.default_rng(0)
    t = rng.random((1, 1, 32, 32)).astype("float32")
    p = (t + 0.05 * rng.standard_normal(t.shape)).astype("float32")

    ours = get_metric("ssim")(torch.tensor(p), torch.tensor(t))
    ref = sk_ssim(t[0, 0], p[0, 0], data_range=float(t.max() - t.min()))

    diff = abs(float(ours) - float(ref))
    assert diff < 5e-2, (
        f"ssim diverges from skimage by {diff:.4f} (> 5e-2); "
        "add SSIMChallengeMetric mirroring skimage slice-wise."
    )
