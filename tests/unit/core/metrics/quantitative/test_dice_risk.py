"""Tests for the SynthSeg-Dice metrics (B-1.7)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from spectramr.core.metrics.quantitative.dice_risk import (
    SynthSegDiceMetric,
    SynthSegDiceRiskMetric,
)


def test_identical_images_dice_one_risk_zero() -> None:
    x = torch.rand(2, 1, 16, 16)
    d = SynthSegDiceMetric()(x, x, segmenter_backend="label_dice", n_classes=5)
    r = SynthSegDiceRiskMetric()(x, x, segmenter_backend="label_dice", n_classes=5)
    assert d > 0.99
    assert r < 0.01
    assert abs((d + r) - 1.0) < 1e-5


def test_different_images_dice_below_one() -> None:
    torch.manual_seed(0)
    d = SynthSegDiceMetric()(
        torch.rand(2, 1, 16, 16), torch.rand(2, 1, 16, 16),
        segmenter_backend="label_dice", n_classes=5,
    )
    assert d < 1.0


def test_synthseg_dice_without_a_segmenter_raises() -> None:
    """Planted violation (#1777 D11): no backend and no injected segmenter must not
    fall back to the intensity-quantile proxy."""
    import pytest

    x = torch.rand(1, 1, 16, 16)
    with pytest.raises(ValueError, match="intensity_quantile_dice_proxy"):
        SynthSegDiceMetric()(x, x)
    with pytest.raises(ValueError, match="intensity_quantile_dice_proxy"):
        SynthSegDiceRiskMetric()(x, x)


def test_proxies_reproduce_the_old_no_segmenter_numbers() -> None:
    """Behaviour preserved for the migrated arms: the proxy equals the label_dice backend
    that synthseg_dice used to fall back to (n_classes 5)."""
    from spectramr.core.metrics.registry import get_metric

    torch.manual_seed(0)
    p, t = torch.rand(2, 1, 16, 16), torch.rand(2, 1, 16, 16)
    old = SynthSegDiceMetric()(p, t, segmenter_backend="label_dice")
    dice = get_metric("intensity_quantile_dice_proxy")
    risk = get_metric("intensity_quantile_dice_risk_proxy")
    assert dice.name == "intensity_quantile_dice_proxy" and dice.higher_is_better
    assert risk.name == "intensity_quantile_dice_risk_proxy" and not risk.higher_is_better
    assert dice(p, t) == pytest.approx(old)
    assert risk(p, t) == pytest.approx(1.0 - old)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_classes": "5"},
        {"n_classes": 1},
        {"segmenter_backend": "synthseg"},
        {"segmenter": MagicMock()},
    ],
)
def test_proxy_refuses_what_it_cannot_honour(kwargs) -> None:
    from spectramr.core.metrics.registry import get_metric

    x = torch.rand(1, 1, 16, 16)
    with pytest.raises(ValueError):
        get_metric("intensity_quantile_dice_proxy")(x, x, **kwargs)


def test_higher_is_better_flags() -> None:
    assert SynthSegDiceMetric().higher_is_better is True
    assert SynthSegDiceRiskMetric().higher_is_better is False


def test_registered_in_metric_registry() -> None:
    from spectramr.core.metrics.registry import get_metric

    assert get_metric("synthseg_dice") is not None
    assert get_metric("synthseg_dice_risk") is not None


def test_constructs_via_computer_path_with_device_kwarg() -> None:
    # The metrics computer instantiates every metric as get(name, device=...). The
    # constructor must accept device (else it errors -> silently stored as NaN at
    # validation). Regression for the review's NaN finding.
    import torch as _t

    from spectramr.core.metrics.registry import get_metric

    m = get_metric("synthseg_dice", device=_t.device("cpu"))
    x = _t.rand(1, 1, 16, 16)
    val = m(x, x, domain="image", data_range=1.0, segmenter_backend="label_dice")
    assert val == val and val > 0.99  # not NaN; identical -> ~1


# ---------------------------------------------------------------------------
# Task-3: 14-label parity for synthseg_dice (segmenter_backend='synthseg')
# ---------------------------------------------------------------------------

def test_synthseg_backend_defaults_to_14_classes() -> None:
    """When segmenter_backend='synthseg', _mean_dice must request n_classes=14.

    Pitfall #16 guard: historically n_classes defaulted to 5 regardless of
    backend.  The challenge grades on 14 DGM nuclei; passing 5 would silently
    score only 4 foreground classes and under-report structural fidelity.
    """
    fake_seg = MagicMock()
    fake_seg.n_classes = 14
    fake_seg.segment.return_value = torch.zeros(1, 16, 16, dtype=torch.long)

    with patch(
        "spectramr.core.metrics.quantitative.dice_risk.get_segmenter",
        return_value=fake_seg,
    ) as mock_gs:
        x = torch.rand(1, 1, 16, 16)
        SynthSegDiceMetric()(x, x, segmenter_backend="synthseg")
        mock_gs.assert_called_once_with("synthseg", n_classes=14)


def test_label_dice_backend_keeps_default_5_classes() -> None:
    """The label_dice proxy must keep its 5-class default (no regression)."""
    fake_seg = MagicMock()
    fake_seg.n_classes = 5
    fake_seg.segment.return_value = torch.zeros(1, 16, 16, dtype=torch.long)

    with patch(
        "spectramr.core.metrics.quantitative.dice_risk.get_segmenter",
        return_value=fake_seg,
    ) as mock_gs:
        x = torch.rand(1, 1, 16, 16)
        SynthSegDiceMetric()(x, x, segmenter_backend="label_dice")
        mock_gs.assert_called_once_with("label_dice", n_classes=5)


def test_explicit_n_classes_overrides_backend_default() -> None:
    """An explicit n_classes kwarg must always win over the backend default."""
    fake_seg = MagicMock()
    fake_seg.n_classes = 7
    fake_seg.segment.return_value = torch.zeros(1, 16, 16, dtype=torch.long)

    with patch(
        "spectramr.core.metrics.quantitative.dice_risk.get_segmenter",
        return_value=fake_seg,
    ) as mock_gs:
        x = torch.rand(1, 1, 16, 16)
        SynthSegDiceMetric()(x, x, segmenter_backend="synthseg", n_classes=7)
        mock_gs.assert_called_once_with("synthseg", n_classes=7)


def test_synthseg_dice_imports_dgm_labels_14() -> None:
    """dice_risk must expose the DGM_LABELS_14 constant imported from challenge_metrics."""
    from spectramr.core.metrics.quantitative import dice_risk
    from spectramr.core.metrics.quantitative.challenge_metrics import DGM_LABELS_14

    assert hasattr(dice_risk, "DGM_LABELS_14"), (
        "dice_risk must re-export DGM_LABELS_14 so callers can inspect the label set"
    )
    assert dice_risk.DGM_LABELS_14 is DGM_LABELS_14


# ---------------------------------------------------------------------------
# I-1: dice_score explicit-labels correctness + label_dice regression
# ---------------------------------------------------------------------------

def test_dice_score_explicit_labels_scores_correct_ids() -> None:
    """dice_score(labels=(10, 49)) must score Dice over label values 10 and 49.

    Regression for the I-1 bug: range(1, 14) would count labels 1..13 and MISS
    the FreeSurfer DGM IDs entirely.  Build tensors that contain 10 and 49 but
    NOT 1 or 2, and verify dice_score detects perfect agreement.
    """
    from spectramr.core.metrics.quantitative.segmentation import dice_score

    seg = torch.zeros(1, 8, 8, dtype=torch.long)
    seg[0, :4, :4] = 10   # top-left quadrant = label 10
    seg[0, 4:, 4:] = 49   # bottom-right quadrant = label 49
    # Identical maps → Dice == 1 for both labels
    scores = dice_score(seg, seg, n_classes=100, labels=(10, 49))
    assert float(scores[0]) > 0.999, f"Expected ~1.0, got {float(scores[0])}"


def test_dice_score_explicit_labels_ignores_contiguous_ids() -> None:
    """dice_score with labels=(10, 49) must ignore label IDs 1 and 2.

    A range(1, n_classes) scorer would count classes 1 and 2 present in seg_a
    but absent from seg_b and lower Dice.  With explicit labels=(10, 49), only
    those two non-contiguous IDs are evaluated — 1 and 2 are invisible.
    """
    from spectramr.core.metrics.quantitative.segmentation import dice_score

    seg_a = torch.zeros(1, 8, 8, dtype=torch.long)
    seg_a[0, :4, :4] = 1    # label 1 present only in seg_a
    seg_a[0, 4:, 4:] = 2    # label 2 present only in seg_a
    seg_b = torch.zeros(1, 8, 8, dtype=torch.long)   # labels 1,2 absent in seg_b
    # With explicit labels=(10, 49): neither map has those IDs → both absent
    # in both → present count stays 0 → clamped to 1 → output = 0/1 = 0.0.
    scores_explicit = dice_score(seg_a, seg_b, n_classes=3, labels=(10, 49))
    assert float(scores_explicit[0]) == 0.0, (
        f"Labels (10,49) absent in both maps should score 0, got {float(scores_explicit[0])}"
    )
    # Contiguous scorer (labels=None, n_classes=3) WOULD count labels 1,2 and get
    # a non-zero present count, scoring low but non-zero.
    scores_range = dice_score(seg_a, seg_b, n_classes=3, labels=None)
    assert float(scores_range[0]) == 0.0, (
        "Labels 1,2 present in seg_a but not seg_b → denom>0 → Dice=0 (not 1)"
    )


def test_dice_score_labels_none_matches_contiguous_behavior() -> None:
    """dice_score(..., labels=None) is byte-identical to the pre-change range behavior.

    Regression guard: the label_dice proxy depends on contiguous range(1, n_classes)
    semantics.  Verifies the default path is untouched.
    """
    from spectramr.core.metrics.quantitative.segmentation import dice_score

    torch.manual_seed(42)
    b = 3
    n = 5
    # Build integer label maps with values in 0..n-1
    seg_a = torch.randint(0, n, (b, 16, 16))
    seg_b = torch.randint(0, n, (b, 16, 16))
    result_none = dice_score(seg_a, seg_b, n_classes=n, labels=None)
    # Compute the same thing manually with range to confirm equivalence
    result_range = dice_score(seg_a, seg_b, n_classes=n, labels=tuple(range(1, n)))
    assert torch.allclose(result_none, result_range), (
        f"labels=None must match labels=range(1, n_classes); diff={result_none - result_range}"
    )


def test_mean_dice_synthseg_passes_dgm_labels_to_dice_score() -> None:
    """_mean_dice with segmenter_backend='synthseg' must pass labels=DGM_LABELS_14.

    Builds a fake segmenter that emits maps with FreeSurfer IDs, then patches
    dice_score to capture the 'labels' argument and assert it equals DGM_LABELS_14.
    """
    from unittest.mock import patch

    from spectramr.core.metrics.quantitative.challenge_metrics import DGM_LABELS_14
    from spectramr.core.metrics.quantitative.dice_risk import _mean_dice

    fake_seg = MagicMock()
    fake_seg.n_classes = 14
    fake_seg.segment.return_value = torch.zeros(1, 8, 8, dtype=torch.long)

    captured = {}

    def _fake_dice_score(seg_a, seg_b, n_classes, labels=None):
        captured["labels"] = labels
        return torch.zeros(1)

    with patch("spectramr.core.metrics.quantitative.dice_risk.get_segmenter", return_value=fake_seg):
        with patch("spectramr.core.metrics.quantitative.dice_risk.dice_score", side_effect=_fake_dice_score):
            x = torch.rand(1, 1, 8, 8)
            _mean_dice(x, x, {"segmenter_backend": "synthseg"})

    assert captured.get("labels") == DGM_LABELS_14, (
        f"Expected labels=DGM_LABELS_14, got {captured.get('labels')}"
    )


def test_mean_dice_label_dice_passes_labels_none() -> None:
    """_mean_dice with segmenter_backend='label_dice' must pass labels=None."""
    from unittest.mock import patch

    from spectramr.core.metrics.quantitative.dice_risk import _mean_dice

    fake_seg = MagicMock()
    fake_seg.n_classes = 5
    fake_seg.segment.return_value = torch.zeros(1, 8, 8, dtype=torch.long)

    captured = {}

    def _fake_dice_score(seg_a, seg_b, n_classes, labels=None):
        captured["labels"] = labels
        return torch.zeros(1)

    with patch("spectramr.core.metrics.quantitative.dice_risk.get_segmenter", return_value=fake_seg):
        with patch("spectramr.core.metrics.quantitative.dice_risk.dice_score", side_effect=_fake_dice_score):
            x = torch.rand(1, 1, 8, 8)
            _mean_dice(x, x, {"segmenter_backend": "label_dice"})

    assert captured.get("labels") is None, (
        f"label_dice proxy must use labels=None (contiguous range), got {captured.get('labels')}"
    )


# ---------------------------------------------------------------------------
# M-1: unknown backend must raise, not default to 5
# ---------------------------------------------------------------------------

def test_unknown_backend_raises_not_defaults() -> None:
    """_mean_dice with an unknown segmenter_backend must raise ValueError (pitfall #9).

    The old code did _DEFAULT_N_CLASSES_BY_BACKEND.get(backend, 5) which silently
    fell through to n_classes=5 for any unknown string.  The fix raises explicitly.
    """
    import pytest

    from spectramr.core.metrics.quantitative.dice_risk import _mean_dice

    x = torch.rand(1, 1, 8, 8)
    with pytest.raises(ValueError, match="Unknown segmenter_backend"):
        _mean_dice(x, x, {"segmenter_backend": "not_a_real_backend"})
