"""Tests for ``QI1Metric`` and ``WM2MaxMetric``.

Targets ``spectramr.core.metrics.qa_metrics``. These are MRIQC-derived
artifact metrics:

- **QI1**: proportion of background voxels with intensity exceeding 2σ
  of the local noise floor — flags ghosting / motion / flow artefacts.
- **WM2Max**: ratio of (synthetic-WM-region median) to (95th percentile
  of full image) — flags long-tailed intensity distributions caused by
  bright-vessel or fat contamination.

HD95 boundary distances need MONAI (integration tier); its Otsu mask is
tested here.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.qa_metrics import HD95Metric, QI1Metric, WM2MaxMetric

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# QI1 — clean vs artefact contrast
# ---------------------------------------------------------------------------


def test_qi1_clean_image_returns_low_score() -> None:
    """A clean image with no artefacts gives a low QI1 (close to 0)."""
    metric = QI1Metric()
    torch.manual_seed(0)
    # Smooth synthetic phantom: foreground centred, background pure low-noise.
    img = torch.zeros(1, 1, 32, 32)
    img[..., 8:24, 8:24] = 1.0  # bright square = foreground
    img += 0.01 * torch.randn(1, 1, 32, 32)  # mild noise
    out = metric.compute_metric(img.clamp(0, 1), img.clamp(0, 1))
    assert out.item() < 0.1


def test_qi1_artefact_in_background_raises_score() -> None:
    """Injecting bright voxels into the background increases QI1."""
    metric = QI1Metric()
    img_clean = torch.zeros(1, 1, 32, 32)
    img_clean[..., 8:24, 8:24] = 1.0

    img_dirty = img_clean.clone()
    # Plant artefacts in background corners
    img_dirty[..., 0:4, 0:4] = 0.6
    img_dirty[..., 28:32, 28:32] = 0.6

    score_clean = metric.compute_metric(img_clean, img_clean).item()
    score_dirty = metric.compute_metric(img_dirty, img_dirty).item()
    assert score_dirty > score_clean


def test_qi1_returns_in_unit_interval() -> None:
    """QI1 is clamped to ``[0, 1]``."""
    metric = QI1Metric()
    torch.manual_seed(0)
    img = torch.rand(1, 1, 16, 16)
    out = metric.compute_metric(img, img)
    assert 0.0 <= out.item() <= 1.0


def test_qi1_handles_2d_input() -> None:
    """``(H, W)`` 2-D input is expanded to ``(1, 1, H, W)``."""
    metric = QI1Metric()
    img = torch.rand(16, 16)
    out = metric.compute_metric(img, img)
    assert out.dim() == 0


def test_qi1_zero_image_returns_zero() -> None:
    """All-zero image → QI1 = 0 (no artefacts to detect)."""
    metric = QI1Metric()
    img = torch.zeros(1, 1, 16, 16)
    out = metric.compute_metric(img, img)
    assert out.item() == 0.0


# ---------------------------------------------------------------------------
# QI1 — corner-region artefact detection against a clean reference (WS-2)
# ---------------------------------------------------------------------------


def test_qi1_corner_spikes_raise_score_against_clean_reference() -> None:
    """Regression for the corner-region + reference-threshold contract.

    The fixed docstring states QI1's background is the four image
    *corners* and that the artefact threshold is derived from the
    **target** (clean reference) signal level — bright voxels planted in
    the *prediction*'s corners must then be detected.

    Distinct from ``test_qi1_artefact_in_background_raises_score`` (which
    passes the dirty image as its own reference): here the reference is a
    fixed CLEAN target while only ``preds`` changes, so the assertion
    pins the documented corner-detection mechanism rather than a
    self-referential noise estimate.
    """
    metric = QI1Metric()

    # Clean reference: bright central foreground, near-zero corners.
    target = torch.zeros(1, 1, 32, 32)
    target[..., 8:24, 8:24] = 1.0

    # Clean prediction == reference (corners are quiet).
    preds_clean = target.clone()

    # Dirty prediction: bright spikes injected into all four corner
    # regions (outer 12.5% = first/last 4 rows & cols of a 32-px axis).
    preds_dirty = target.clone()
    preds_dirty[..., :4, :4] = 0.8
    preds_dirty[..., :4, -4:] = 0.8
    preds_dirty[..., -4:, :4] = 0.8
    preds_dirty[..., -4:, -4:] = 0.8

    score_clean = metric.compute_metric(preds_clean, target).item()
    score_dirty = metric.compute_metric(preds_dirty, target).item()

    # Clean corners → essentially no detected artefacts.
    assert score_clean < 0.05
    # Corner spikes must drive QI1 strictly higher (lower is better).
    assert score_dirty > score_clean
    # All four corner blocks are above threshold, so the artefact ratio
    # should be substantial, not a marginal bump.
    assert score_dirty > 0.5


def test_qi1_center_artefact_is_not_counted_as_background() -> None:
    """Bright voxels OUTSIDE the corners do not inflate QI1.

    Confirms the background is spatially the corners (not a global
    value-threshold): a bright blob planted in the image *center* leaves
    the corner-region artefact ratio unchanged.
    """
    metric = QI1Metric()

    target = torch.zeros(1, 1, 32, 32)
    target[..., 8:24, 8:24] = 1.0

    preds_center_spike = target.clone()
    preds_center_spike[..., 14:18, 14:18] = 2.0  # bright, but central

    score_clean = metric.compute_metric(target, target).item()
    score_center = metric.compute_metric(preds_center_spike, target).item()

    assert score_center == pytest.approx(score_clean, abs=1e-6)


# ---------------------------------------------------------------------------
# WM2Max
# ---------------------------------------------------------------------------


def test_wm2max_returns_value_in_documented_range() -> None:
    """For a uniform-density image, WM2Max is in ``[0, 2]``."""
    metric = WM2MaxMetric()
    torch.manual_seed(0)
    img = torch.rand(1, 1, 32, 32) * 0.8 + 0.1
    out = metric.compute_metric(img, img)
    assert 0.0 <= out.item() <= 2.0


def test_wm2max_zero_input_returns_zero() -> None:
    """No foreground → WM2Max = 0."""
    metric = WM2MaxMetric()
    img = torch.zeros(1, 1, 16, 16)
    out = metric.compute_metric(img, img)
    assert out.item() == 0.0


def test_wm2max_handles_3d_input() -> None:
    """``(C, H, W)`` 3-D input is unsqueezed."""
    metric = WM2MaxMetric()
    img = torch.rand(1, 16, 16) * 0.5 + 0.2
    out = metric.compute_metric(img, img)
    assert out.dim() == 0


def test_wm2max_long_tail_lowers_ratio() -> None:
    """Adding hyper-intense outliers reduces WM2Max (denominator grows)."""
    metric = WM2MaxMetric()
    torch.manual_seed(0)
    img_normal = torch.rand(1, 1, 32, 32) * 0.6 + 0.2
    img_long_tail = img_normal.clone()
    img_long_tail[..., 0:2, 0:2] = 5.0  # extreme outliers

    score_normal = metric.compute_metric(img_normal, img_normal).item()
    score_long_tail = metric.compute_metric(img_long_tail, img_long_tail).item()
    assert score_long_tail < score_normal


# ---------------------------------------------------------------------------
# Sanity-shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 16, 16), (2, 1, 32, 32), (1, 1, 64, 64)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_qi1_shape_matrix(shape: tuple[int, ...]) -> None:
    """QI1 returns a scalar regardless of input batch / spatial size."""
    metric = QI1Metric()
    img = torch.rand(*shape)
    out = metric.compute_metric(img, img)
    assert out.dim() == 0
    assert torch.isfinite(out)


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 16, 16), (2, 1, 32, 32), (1, 1, 64, 64)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_wm2max_shape_matrix(shape: tuple[int, ...]) -> None:
    """WM2Max returns a scalar regardless of input batch / spatial size."""
    metric = WM2MaxMetric()
    img = torch.rand(*shape) * 0.7 + 0.1
    out = metric.compute_metric(img, img)
    assert out.dim() == 0
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_qa_metrics_registered_under_documented_names() -> None:
    """``qi1`` and ``wm2max`` are both in the metric registry."""
    from spectramr.core.metrics.registry import list_available

    available = list_available()
    assert "qi1" in available
    assert "wm2max" in available


def test_hd95_otsu_mask_is_invariant_to_minus_one_one_normalisation() -> None:
    """#174: a [-1, 1] image must threshold the same anatomy as its [0, 1] original."""
    torch.manual_seed(0)
    img = torch.zeros(1, 1, 32, 32)
    img[..., 8:24, 8:24] = 0.3
    img[..., 12:20, 12:20] = 0.9
    img = (img + 0.02 * torch.rand_like(img)).clamp(0.0, 1.0)
    mask_unit = HD95Metric._otsu_threshold(img)
    mask_signed = HD95Metric._otsu_threshold(img * 2.0 - 1.0)
    assert torch.equal(mask_unit, mask_signed)
    assert 0 < mask_unit.sum() < img.numel()


def test_hd95_otsu_constant_image_gives_empty_mask() -> None:
    assert HD95Metric._otsu_threshold(torch.full((1, 1, 8, 8), -0.5)).sum() == 0
