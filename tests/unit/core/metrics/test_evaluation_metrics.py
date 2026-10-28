"""Tests for the gradient-based focus / edge metrics.

Grounds the gradient metric family the sim2rank zero-metric review asked about:

* ``gradient_error`` — Sobel gradient-magnitude L1 (full-reference). Already
  present; pinned here as a regression guard.
* ``gradient_entropy`` — Shannon entropy of the gradient-magnitude histogram.
  This is Atkinson's *Entropy Focus Criterion* family, the validated MRI motion
  autofocus metric (Atkinson et al., IEEE TMI 1997). Already present.
* ``normalized_gradient_squared`` — newly added. The lower-cost autofocus
  counterpart to gradient entropy (McGee et al., JMRI 2000): gradient energy
  normalized by image energy, so it is intensity-scale-invariant and higher for
  sharper images.
"""

from __future__ import annotations

import math
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from spectramr.core.metrics.registry import MetricsRegistry


def _disk(h=64, w=64):
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    return (torch.sqrt(xx**2 + yy**2) < 0.6).float().unsqueeze(0).unsqueeze(0)


def _blur(img, k=5):
    import torch.nn.functional as F

    kernel = torch.ones(1, 1, k, k) / (k * k)
    return F.conv2d(img, kernel, padding=k // 2)


def test_gradient_error_is_zero_on_identity_and_grows():
    metric = MetricsRegistry.get("gradient_error")
    clean = _disk()
    deg = (clean + 0.2 * torch.randn_like(clean)).clamp(0, None)
    assert float(metric(clean, clean)) == pytest.approx(0.0, abs=1e-5)
    assert float(metric(deg, clean)) > 0.0


def test_gradient_entropy_is_finite_and_responds_to_structure():
    metric = MetricsRegistry.get("gradient_entropy")
    sharp = _disk()
    smooth = _blur(sharp)
    v_sharp = float(metric(sharp, sharp))
    v_smooth = float(metric(smooth, smooth))
    assert math.isfinite(v_sharp) and v_sharp >= 0.0
    # Blurring changes the gradient-magnitude distribution → different entropy.
    assert abs(v_sharp - v_smooth) > 1e-3


def test_normalized_gradient_squared_higher_for_sharper():
    """NGS (gradient energy / image energy) must be larger for a sharp image
    than for its blurred version — the autofocus sharpness property."""
    metric = MetricsRegistry.get("normalized_gradient_squared")
    sharp = _disk()
    blurred = _blur(sharp)
    v_sharp = float(metric(sharp, sharp))
    v_blur = float(metric(blurred, blurred))
    assert math.isfinite(v_sharp) and v_sharp >= 0.0
    assert v_sharp > v_blur, f"sharp NGS {v_sharp} should exceed blurred {v_blur}"


# ---------------------------------------------------------------------------
# ClinicalSSIM default-data_range regression (WS-2 core-metrics fix).
#
# Before the fix, ClinicalSSIM.compute_metric resolved its data range with
# ``dr = data_range or self.data_range``. A default-constructed ClinicalSSIM
# (and the registry-built ``clinical_ssim``) has self.data_range=None, so dr
# stayed None and flowed into compute_ssim_map as ``(k1 * None) ** 2`` which
# raised TypeError — the metric was dead-by-default. The fix mirrors the parent
# SSIMMetric 3-way auto-detect (explicit arg → self.data_range → 2.0/1.0 from
# sign of target.min()), so a no-data_range ClinicalSSIM now returns a finite
# scalar in [-1, 1] instead of raising.
# ---------------------------------------------------------------------------


def test_clinical_ssim_default_data_range_finite_no_typeerror():
    """A default-constructed ClinicalSSIM (data_range=None) must not raise
    TypeError and must return a finite scalar in [-1, 1]."""
    from spectramr.core.metrics.evaluation_metrics import ClinicalSSIM

    torch.manual_seed(0)
    pred = torch.rand(2, 1, 32, 32)
    target = torch.rand(2, 1, 32, 32)

    metric = ClinicalSSIM()  # NO data_range -> self.data_range is None
    assert metric.data_range is None

    # Pre-fix this raised TypeError inside compute_ssim_map.
    value = metric(pred, target)
    v = float(value)
    assert math.isfinite(v), f"ClinicalSSIM returned non-finite value {v}"
    assert -1.0 <= v <= 1.0, f"SSIM value {v} outside [-1, 1]"


def test_clinical_ssim_default_data_range_via_registry():
    """The registry-built ``clinical_ssim`` (also data_range=None) is the path
    that was dead-by-default; it must compute a finite scalar in [-1, 1]."""
    torch.manual_seed(0)
    pred = torch.rand(1, 1, 48, 48)
    target = torch.rand(1, 1, 48, 48)

    metric = MetricsRegistry.get("clinical_ssim")
    value = metric(pred, target)
    v = float(value)
    assert math.isfinite(v)
    assert -1.0 <= v <= 1.0


def test_clinical_ssim_default_data_range_identical_is_one():
    """SSIM of an image against itself must be ~1.0 even when data_range is
    auto-detected (the masked-mean of an all-ones SSIM map)."""
    from spectramr.core.metrics.evaluation_metrics import ClinicalSSIM

    torch.manual_seed(1)
    img = torch.rand(1, 1, 32, 32)

    metric = ClinicalSSIM()
    v = float(metric(img, img))
    assert math.isfinite(v)
    assert v == pytest.approx(1.0, abs=1e-3), f"identical SSIM {v} should be ~1.0"


# ---------------------------------------------------------------------------
# AMP-fp16 overflow regression (ms_ssim breakage sweep, 2026-07-04).
#
# compute_ssim_map is the shared chokepoint for the ms_ssim loss (SSIMLoss /
# MSSSIMLoss) and the in-house SSIM metric. The loss is evaluated *inside* an
# active fp16 autocast for image-space arms; ``img1 * img1`` on an off-scale
# prediction overflowed the fp16 range (max ~6.5e4) → ``Inf - Inf = NaN``. The
# fix disables autocast and upcasts to float32 for the whole computation, so a
# fp16 off-scale input yields a finite float32 map (a genuine NaN model output
# still propagates — we upcast the artifact, never a real signal).
# ---------------------------------------------------------------------------


def test_compute_ssim_map_fp16_offscale_finite_and_float32():
    """fp16 off-scale inputs must yield a finite float32 SSIM map."""
    from spectramr.core.metrics.evaluation_metrics import (
        compute_ssim_map,
        gaussian_kernel,
    )

    torch.manual_seed(0)
    scale = 1e3  # (1e3)^2 = 1e6, well past the fp16 ceiling
    img1 = (torch.rand(2, 1, 64, 64) * scale).half()
    img2 = (torch.rand(2, 1, 64, 64) * scale).half()
    window = gaussian_kernel(11, 1.5, torch.device("cpu"))

    out = compute_ssim_map(img1, img2, window, 11, data_range=float(scale))

    assert out.dtype == torch.float32, "SSIM map must upcast off fp16"
    assert torch.isfinite(out).all(), "fp16 off-scale SSIM map went non-finite"


def test_compute_ssim_map_genuine_nan_input_still_propagates():
    """The fp32 upcast must not mask a real NaN in the model output — a NaN
    input must still yield a non-finite map (honest failure, not pitfall #9)."""
    from spectramr.core.metrics.evaluation_metrics import (
        compute_ssim_map,
        gaussian_kernel,
    )

    torch.manual_seed(0)
    img1 = torch.rand(1, 1, 64, 64)
    img1[0, 0, 0, 0] = float("nan")
    img2 = torch.rand(1, 1, 64, 64)
    window = gaussian_kernel(11, 1.5, torch.device("cpu"))

    out = compute_ssim_map(img1, img2, window, 11, data_range=1.0)
    assert not torch.isfinite(out).all(), "a genuine NaN input must surface"


def _phantom_on_background() -> torch.Tensor:
    """A textured square on zero background covering 75 % of the slice (#2274)."""
    g = torch.Generator().manual_seed(0)
    x = torch.zeros(2, 1, 96, 96)
    x[:, :, 24:72, 24:72] = torch.rand(2, 1, 48, 48, generator=g) * 0.8 + 0.1
    return x


class TestSSIMIdentityOnBackground:
    """SSIM(x, x) is exactly 1 on flat background (#2274).

    An extra ``+1e-8`` in the denominator was the size of ``C1*C2`` at
    ``data_range=1`` (8.1e-8), so every flat window scored 0.89 against itself.
    """

    def test_map_is_one_on_flat_windows(self) -> None:
        from spectramr.core.metrics.evaluation_metrics import compute_ssim_map, gaussian_kernel

        x = _phantom_on_background()
        window = gaussian_kernel(11, 1.5, torch.device("cpu"))
        ssim_map = compute_ssim_map(x, x, window, 11, data_range=1.0)
        assert torch.allclose(ssim_map, torch.ones_like(ssim_map), atol=1e-6)

    def test_metric_is_one(self) -> None:
        from spectramr.core.metrics.evaluation_metrics import SSIMMetric

        x = _phantom_on_background()
        assert float(SSIMMetric(data_range=1.0).compute_metric(x, x)) == pytest.approx(1.0, abs=1e-6)

    def test_loss_is_zero(self) -> None:
        from spectramr.models.losses.ssim_loss import MSSSIMLoss, SSIMLoss

        x = _phantom_on_background()
        assert float(SSIMLoss()(x, x)) <= 1e-6
        assert float(MSSSIMLoss()(x, x)) <= 1e-6

    def test_all_zero_pair_is_one_not_nan(self) -> None:
        from spectramr.core.metrics.evaluation_metrics import SSIMMetric
        from spectramr.models.losses.ssim_loss import SSIMLoss

        z = torch.zeros(1, 1, 32, 32)
        assert float(SSIMMetric().compute_metric(z, z)) == 1.0
        assert float(SSIMLoss()(z, z)) == 0.0

    def test_non_positive_data_range_raises(self) -> None:
        from spectramr.core.metrics.evaluation_metrics import compute_ssim_map, gaussian_kernel

        x = _phantom_on_background()
        window = gaussian_kernel(11, 1.5, torch.device("cpu"))
        with pytest.raises(ValueError, match="data_range must be > 0"):
            compute_ssim_map(x, x, window, 11, data_range=0.0)


def test_nrmse_normalizes_by_measured_target_range() -> None:
    """NRMSE must divide RMSE by the target's actual (max - min) range.

    Regression: the previous heuristic assumed a range of 1.0 (or 2.0 for signed
    data), so a target confined to ``[0, 0.5]`` was normalised by 1.0 and the
    score came out 2x too small. Here a constant 0.1 error against a target of
    range 0.5 must give NRMSE = 0.1 / 0.5 = 0.2, not 0.1.
    """
    from spectramr.core.metrics.evaluation_metrics import NRMSE

    target = torch.linspace(0.0, 0.5, steps=64).reshape(1, 1, 8, 8)
    pred = target + 0.1  # RMSE = 0.1 exactly

    nrmse = float(NRMSE()(pred, target))
    assert nrmse == pytest.approx(0.2, rel=1e-4)


# ---------------------------------------------------------------------------
# MS-SSIM / UQI backbone caching (WS-2 6.2): the torchmetrics backbone is
# built ONCE in __init__ and reused (``self._impl``) across compute calls,
# not re-instantiated per validation step. torchmetrics may be unavailable in
# CI, so we inject a fake backbone to assert the object-identity contract
# independently of the real dependency.
# ---------------------------------------------------------------------------


class _FakeBackbone(torch.nn.Module):
    """nn.Module stand-in whose ``.to`` returns self and forward is constant."""

    n_constructed = 0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        type(self).n_constructed += 1

    def forward(self, preds, target):  # noqa: ANN001
        return torch.tensor(0.5)


@pytest.mark.parametrize(
    "cls_name, backbone_attr",
    [
        ("MSSSIM", "MultiScaleStructuralSimilarityIndexMeasure"),
        ("UQI", "UniversalImageQualityIndex"),
    ],
)
def test_torchmetrics_backbone_built_once_and_cached(
    cls_name, backbone_attr, monkeypatch
) -> None:
    import spectramr.core.metrics.evaluation_metrics as em

    _FakeBackbone.n_constructed = 0
    monkeypatch.setattr(em, "TORCHMETRICS_AVAILABLE", True)
    # The backbone is imported lazily from torchmetrics.image at construction, so
    # a stand-in package in sys.modules is what the import resolves to (works
    # whether or not torchmetrics is installed).
    fake_image = types.ModuleType("torchmetrics.image")
    setattr(fake_image, backbone_attr, _FakeBackbone)
    fake_pkg = types.ModuleType("torchmetrics")
    fake_pkg.image = fake_image
    monkeypatch.setitem(sys.modules, "torchmetrics", fake_pkg)
    monkeypatch.setitem(sys.modules, "torchmetrics.image", fake_image)

    metric = getattr(em, cls_name)()
    impl = metric._impl
    assert impl is not None
    # Exactly one backbone was constructed at __init__ time.
    assert _FakeBackbone.n_constructed == 1

    p = torch.zeros(1, 1, 176, 176)  # above ms_ssim's 161 px floor
    t = torch.zeros(1, 1, 176, 176)
    metric.compute_metric(p, t)
    metric.compute_metric(p, t)

    # Same cached object across both calls; no per-call re-instantiation.
    assert metric._impl is impl
    assert _FakeBackbone.n_constructed == 1


# ── ALLOWS_UNEQUAL_SAMPLE_COUNT: the narrow shape opt-out ─────────────────


class TestUnequalSampleCountOptOut:
    """``_check_shapes`` must free ONLY the leading sample axis under the narrow
    opt-out. The wide opt-out (``REQUIRES_MATCHING_SHAPES = False``) returns
    immediately and checks nothing, which is why a distribution metric that needed
    "dim 0 may differ" ended up accepting a channel mismatch too."""

    @staticmethod
    def _metric(**attrs):
        import torch

        from spectramr.core.metrics.evaluation_metrics import BaseMetric

        class _Probe(BaseMetric):
            def compute_metric(self, preds, target, **_):
                return torch.zeros(())

        for k, v in attrs.items():
            setattr(_Probe, k, v)
        return _Probe()

    def test_default_requires_exact_match(self):
        import pytest
        import torch

        m = self._metric()
        with pytest.raises(ValueError):
            m(torch.rand(4, 1, 8, 8), torch.rand(3, 1, 8, 8))

    def test_optout_allows_differing_sample_axis(self):
        import torch

        m = self._metric(ALLOWS_UNEQUAL_SAMPLE_COUNT=True)
        m(torch.rand(4, 1, 8, 8), torch.rand(3, 1, 8, 8))  # must not raise

    def test_optout_still_rejects_trailing_mismatch(self):
        import pytest
        import torch

        m = self._metric(ALLOWS_UNEQUAL_SAMPLE_COUNT=True)
        with pytest.raises(ValueError, match="trailing"):
            m(torch.rand(4, 2, 8, 8), torch.rand(3, 1, 8, 8))
        with pytest.raises(ValueError, match="trailing"):
            m(torch.rand(4, 1, 8, 8), torch.rand(4, 1, 16, 16))

    def test_base_default_is_off(self):
        from spectramr.core.metrics.evaluation_metrics import BaseMetric

        assert BaseMetric.ALLOWS_UNEQUAL_SAMPLE_COUNT is False
        assert BaseMetric.REQUIRES_MATCHING_SHAPES is True


class TestZipperDetectionIsReferenceFree:
    """``zipper_detection`` reads only ``preds``, and the registry must say so.

    The 2026-07-24 NR audit added a guard (``test_no_reference_specs_agree_with
    _registry``) for exactly this drift: a metric typed NO_REFERENCE in
    ``metrics_list`` while registered ``requires_reference=True``. It bit
    ``g_factor`` then and ``zipper_detection`` at the 2026-07-26 reconciliation,
    when the metric was first declared to sim2rank.
    """

    def test_registry_declares_it_reference_free(self):
        from spectramr.core.metrics.registry import MetricsRegistry

        assert MetricsRegistry.requires_reference("zipper_detection") is False

    def test_target_does_not_change_the_score(self):
        """The behavioural claim behind the flag -- not just the flag itself."""
        import torch

        from spectramr.core.metrics.registry import MetricsRegistry

        metric = MetricsRegistry.get("zipper_detection")
        preds = torch.rand(1, 1, 32, 32, generator=torch.Generator().manual_seed(0))
        a = float(metric(preds, torch.zeros_like(preds)))
        b = float(metric(preds, torch.rand_like(preds)))

        assert a == pytest.approx(b)

    def test_directional_corruption_raises_the_score(self):
        """A zipper is anisotropic: striping one axis must beat isotropic noise."""
        import torch

        from spectramr.core.metrics.registry import MetricsRegistry

        metric = MetricsRegistry.get("zipper_detection")
        gen = torch.Generator().manual_seed(0)
        base = torch.rand(1, 1, 64, 64, generator=gen)
        zipped = base.clone()
        zipped[..., ::2, :] += 0.5  # stripe every other row -> one-axis energy

        assert float(metric(zipped, base)) > float(metric(base, base))


# ---------------------------------------------------------------------------
# data_range resolution (issue #180).
#
# `dr = 2.0 if target.min() < 0 else 1.0` appeared at three sites (PSNR, SSIM,
# ClinicalSSIM). The sign test is a *proxy for a normalization contract*, and on
# data that honours no contract it does not fail — it silently returns 1.0. The
# `*_mno` cohort (train_mse = 458_341) then recorded val_psnr pegged at the -30 dB
# clamp floor and SSIM values of -653.8 and -958.3: not bad scores, impossible
# ones, written to CSV as if they were data.
#
# The contract these tests pin: a declared range wins; a *verified* contract is
# used; an unverifiable one is NOT_APPLICABLE. Deliberately NOT `max - min`, which
# would make PSNR incomparable across images and silently restate every number the
# corpus has recorded.
# ---------------------------------------------------------------------------


class TestDataRangeResolution:
    """`resolve_image_data_range` — the single fallback for range-sensitive metrics."""

    @staticmethod
    def _with_peak(peak: float, *, signed: bool = False):
        """A tensor whose extremum is exactly ``peak`` (not a sampled approximation)."""
        t = torch.zeros(1, 1, 4, 4)
        t[0, 0, 0, 0] = peak
        if signed:
            t[0, 0, 1, 1] = -peak
        return t

    def test_normalized_contracts_are_unchanged(self):
        """The whole point: correctly-normalized arms keep their historical values.

        A fix that silently restated every PSNR in the corpus would be worse than
        the bug, so [0,1] must still resolve 1.0 and [-1,1] must still resolve 2.0.
        """
        from spectramr.core.metrics.evaluation_metrics import resolve_image_data_range

        assert resolve_image_data_range(
            self._with_peak(1.0), None, metric_name="psnr"
        ) == pytest.approx(1.0)
        assert resolve_image_data_range(
            self._with_peak(1.0, signed=True), None, metric_name="psnr"
        ) == pytest.approx(2.0)

    def test_declared_range_wins_even_on_unnormalized_data(self):
        """`metrics.data_range` is the config surface for exactly this case.

        A user who states the scale is not second-guessed — declaring it is the
        documented escape hatch the not-applicable message points at.
        """
        from spectramr.core.metrics.evaluation_metrics import resolve_image_data_range

        assert resolve_image_data_range(
            self._with_peak(2479.0), 2479.0, metric_name="psnr"
        ) == pytest.approx(2479.0)

    def test_contract_tolerance_boundary_is_exact(self):
        """2.0 (== 6 dB of PSNR bias) is the documented cut, and it is closed."""
        from spectramr.core.metrics.evaluation_metrics import resolve_image_data_range
        from spectramr.core.metrics.outcome import MetricNotApplicableError

        assert (
            resolve_image_data_range(self._with_peak(2.0), None, metric_name="psnr")
            == 1.0
        )
        with pytest.raises(MetricNotApplicableError):
            resolve_image_data_range(self._with_peak(2.0001), None, metric_name="psnr")

    def test_unresolvable_range_carries_a_machine_readable_reason(self):
        """NOT_APPLICABLE + a declared reason, never a fabricated number."""
        from spectramr.core.metrics.evaluation_metrics import resolve_image_data_range
        from spectramr.core.metrics.outcome import (
            MetricNotApplicableError,
            NotApplicableReason,
        )

        with pytest.raises(MetricNotApplicableError) as exc:
            resolve_image_data_range(self._with_peak(2479.0), None, metric_name="psnr")
        assert exc.value.reason is NotApplicableReason.DATA_RANGE_UNRESOLVED
        # The message must name the escape hatch, or the error is a dead end.
        assert "metrics.data_range" in exc.value.detail


class TestUnnormalizedInputIsNotScored:
    """Issue #180 regression: the `*_mno` signature must not produce a number."""

    @staticmethod
    def _mno_like():
        """An unnormalized pair with the cohort's abs_max ~2479 signature."""
        gen = torch.Generator().manual_seed(0)
        target = torch.rand(2, 1, 32, 32, generator=gen) * 2479.0
        pred = target + 500.0 * torch.randn(2, 1, 32, 32, generator=gen)
        return pred, target

    @pytest.mark.parametrize("name", ["psnr", "ssim", "clinical_ssim"])
    def test_range_sensitive_metrics_report_not_applicable(self, name):
        """Pre-fix these returned -30.0 (clamp floor) and ~-653 (out of codomain)."""
        from spectramr.core.metrics.outcome import MetricNotApplicableError

        metric = MetricsRegistry.get(name)
        pred, target = self._mno_like()

        with pytest.raises(MetricNotApplicableError):
            metric(pred, target)

    def test_psnr_no_longer_saturates_at_the_clamp_floor(self):
        """The specific tell from #179/#180: every checkpoint scoring exactly -30.0.

        A selection metric pegged at a clamp bound carries no information, so
        best-checkpoint selection silently degrades to a tie-break. Refusing to
        score is what makes that visible.
        """
        from spectramr.core.metrics.outcome import MetricNotApplicableError

        psnr = MetricsRegistry.get("psnr")
        pred, target = self._mno_like()
        try:
            value = float(psnr(pred, target))
        except MetricNotApplicableError:
            return  # refused to score — the intended outcome
        pytest.fail(f"psnr returned {value} on unnormalized data (expected refusal)")


class TestDataRangeEscapeHatchesStillWork:
    """The two paths that legitimately bypass the contract check."""

    def test_kspace_domain_uses_its_own_peak(self):
        """k-space has no [0,1] contract to verify — its scale IS the acquisition's."""
        from spectramr.core.metrics.evaluation_metrics import PSNR

        gen = torch.Generator().manual_seed(0)
        target = torch.rand(1, 1, 16, 16, generator=gen) * 5000.0
        pred = target * 1.01

        value = float(PSNR(domain="kspace")(pred, target))
        assert math.isfinite(value)

    def test_use_target_max_opt_in_is_untouched(self):
        """An explicit per-image range is a caller decision, not a guess."""
        from spectramr.core.metrics.evaluation_metrics import PSNR

        gen = torch.Generator().manual_seed(0)
        target = torch.rand(1, 1, 16, 16, generator=gen) * 900.0
        pred = target * 1.01

        value = float(PSNR(use_target_max=True)(pred, target))
        assert math.isfinite(value)


class TestSSIMCodomainGuard:
    """An SSIM outside [-1, 1] is an upstream defect, not a low score."""

    def test_out_of_codomain_value_raises_rather_than_being_reported(self):
        """Backstop for the class of defect, independent of its known cause.

        `resolve_image_data_range` closes the route #180 found. This keeps a
        different route from quietly writing -653 into a CSV again.
        """
        from spectramr.core.metrics.evaluation_metrics import _guard_ssim_codomain

        with pytest.raises(ValueError, match="codomain"):
            _guard_ssim_codomain(torch.tensor(-653.8), metric_name="ssim")

    def test_in_codomain_value_passes_through_unchanged(self):
        from spectramr.core.metrics.evaluation_metrics import _guard_ssim_codomain

        v = torch.tensor(0.87)
        assert _guard_ssim_codomain(v, metric_name="ssim") is v

    def test_nonfinite_value_is_also_refused(self):
        """NaN is not a score either — it is 'the metric did not measure'."""
        from spectramr.core.metrics.evaluation_metrics import _guard_ssim_codomain

        with pytest.raises(ValueError):
            _guard_ssim_codomain(torch.tensor(float("nan")), metric_name="ssim")


# ---------------------------------------------------------------------------
# PSNR is unbounded (issue #179), and the two PSNRs measure different things.
#
# `torch.clamp(psnr, -30.0, 100.0)` sat in both implementations. Nine
# `experiment_vf_*` arms monitored a PSNR and recorded `early_stopping_best_value`
# of exactly -30.0: once every checkpoint scores the bound, "best checkpoint" is
# decided by tie-break order, so those arms had no model selection at all.
# ---------------------------------------------------------------------------


class TestPSNRIsUnbounded:
    @pytest.mark.parametrize("name", ["psnr", "robust_mri_psnr"])
    def test_scores_below_the_old_floor_stay_distinct(self, name):
        """The floor's real cost was ties, not the number: -47 and -112 dB are
        both terrible and are NOT the same terrible."""
        metric = MetricsRegistry.get(name)
        target = torch.rand(1, 1, 32, 32, generator=torch.Generator().manual_seed(0))

        scores = [
            float(metric(target + offset, target)) for offset in (50.0, 200.0, 1000.0)
        ]

        assert all(s < -30.0 for s in scores), f"{name} still bounded below: {scores}"
        assert len(set(scores)) == len(scores), (
            f"{name} returned duplicate scores {scores} — a saturated selection "
            "metric makes best-checkpoint selection a tie-break"
        )
        assert scores == sorted(scores, reverse=True), "worse input must score lower"

    @pytest.mark.parametrize("name", ["psnr", "robust_mri_psnr"])
    def test_ordinary_scores_are_unchanged(self, name):
        """Removing a bound must not move any value that was inside it."""
        metric = MetricsRegistry.get(name)
        gen = torch.Generator().manual_seed(0)
        target = torch.rand(1, 1, 64, 64, generator=gen)
        pred = (target + 0.05 * torch.randn(1, 1, 64, 64, generator=gen)).clamp(0, 1)

        value = float(metric(pred, target))
        assert 15.0 < value < 40.0, f"{name} moved out of its ordinary range: {value}"


class TestExactMatchOutranksEveryImperfectPrediction:
    """The `mse == 0` sentinel did not survive the clamp it was calibrated to.

    `torch.clamp(psnr, max=100.0)` and `if mse == 0: return 100.0` encoded the
    same constant for different reasons. Removing the clamp left the sentinel
    behind, and it inverted the order these metrics exist to impose: an exact
    match scored 100.0 dB while a prediction with 1e-7 noise scored 139.7 dB.
    Both are `higher_is_better`, so best-checkpoint selection preferred the
    imperfect model by ~40 dB, in exactly the near-perfect regime where identity
    collapse and pass-through facades live (pitfall #20).

    Under the clamp the two merely TIED, which is why the sentinel looked
    harmless for as long as the bound was there.
    """

    @pytest.mark.parametrize("name", ["psnr", "robust_mri_psnr"])
    def test_a_perfect_prediction_scores_strictly_highest(self, name):
        metric = MetricsRegistry.get(name)
        gen = torch.Generator().manual_seed(0)
        target = torch.rand(1, 1, 32, 32, generator=gen) + 0.2

        exact = float(metric(target.clone(), target))
        # 1e-7 is the regime the sentinel got wrong; the coarser scales guard
        # against a fix that only reorders the extreme end.
        worse = [
            float(
                metric(
                    target + sigma * torch.randn(target.shape, generator=gen), target
                )
            )
            for sigma in (1e-7, 1e-5, 1e-3)
        ]

        assert all(exact > w for w in worse), (
            f"{name}: exact match scored {exact}, below {worse}. A worse "
            "reconstruction outranking a perfect one inverts model selection."
        )

    @pytest.mark.parametrize("name", ["psnr", "robust_mri_psnr"])
    def test_the_exact_match_score_is_finite(self, name):
        """`inf` would read as 'best' to a human and as 'skip' to the NaN gate.

        `training_loop` gates `save_best` on `math.isfinite(monitor_value)`, so
        returning `inf` for a perfect match would silently refuse to checkpoint
        the one model that earned it.
        """
        metric = MetricsRegistry.get(name)
        target = torch.rand(1, 1, 16, 16, generator=torch.Generator().manual_seed(1))

        value = float(metric(target.clone(), target))

        assert math.isfinite(value), f"{name} returned {value} on an exact match"

    @pytest.mark.parametrize("name", ["psnr", "robust_mri_psnr"])
    def test_scores_are_monotone_in_error(self, name):
        """Monotonicity is the only property selection actually needs."""
        metric = MetricsRegistry.get(name)
        gen = torch.Generator().manual_seed(2)
        target = torch.rand(1, 1, 32, 32, generator=gen) + 0.2

        scores = [
            float(
                metric(
                    target + sigma * torch.randn(target.shape, generator=gen), target
                )
            )
            for sigma in (1e-4, 1e-3, 1e-2, 1e-1)
        ]

        assert scores == sorted(
            scores, reverse=True
        ), f"{name} not monotone in error: {scores}"


class TestTheTwoPSNRsMeasureDifferentThings:
    """`psnr` is canonical whole-image; `robust_mri_psnr` is foreground-restricted.

    They are SUPPOSED to disagree. Pinned so a future "reconciliation" cannot
    quietly collapse one into the other.
    """

    @staticmethod
    def _phantom(h=128, w=128):
        """Mostly-black MRI-like frame with a bright anatomy disk."""
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
        )
        disk = (torch.sqrt(xx**2 + yy**2) < 0.55).float()
        gen = torch.Generator().manual_seed(0)
        amp = 0.6 + 0.4 * torch.rand(h, w, generator=gen)
        return (disk * amp).unsqueeze(0).unsqueeze(0)

    def test_background_only_noise_penalizes_whole_image_psnr_but_not_the_robust_one(
        self,
    ):
        """The defining behaviour: corrupting air must not cost anatomical fidelity."""
        target = self._phantom()
        background = (target < 1e-5).float()
        gen = torch.Generator().manual_seed(1)
        noisy_air = (
            target + 0.2 * torch.randn(target.shape, generator=gen) * background
        ).clamp(0, 1)

        plain = float(MetricsRegistry.get("psnr")(noisy_air, target))
        robust = float(MetricsRegistry.get("robust_mri_psnr")(noisy_air, target))

        assert (
            plain < 30.0
        ), f"whole-image psnr should be penalized by air noise, got {plain}"
        assert robust > plain + 20.0, (
            f"robust_mri_psnr ({robust}) must ignore background corruption that "
            f"whole-image psnr ({plain}) is penalized by — that is the whole point"
        )

    def test_both_degrade_together_when_the_anatomy_itself_is_corrupted(self):
        """Restricting to tissue must not make the metric blind to tissue error."""
        target = self._phantom()
        gen = torch.Generator().manual_seed(2)
        noisy = (target + 0.05 * torch.randn(target.shape, generator=gen)).clamp(0, 1)

        plain = float(MetricsRegistry.get("psnr")(noisy, target))
        robust = float(MetricsRegistry.get("robust_mri_psnr")(noisy, target))

        assert abs(plain - robust) < 6.0, (
            f"under uniform noise the two should track within a few dB "
            f"(psnr={plain}, robust={robust})"
        )


class TestImportWritesNoEnvironment:
    """Importing this module must not mutate the process environment (#1250).

    The module head used to run, at import time::

        os.environ.setdefault("TMPDIR", "/tmp/<username>")
        if torch.cuda.is_available():
            os.environ["TORCH_CUDA_EAGER_CACHE_MANAGER"] = "1"

    Three separate defects, and the third is what made it invisible:

    * ``/tmp/<username>`` hardcoded one developer's username; ``/tmp`` is sticky,
      so on a shared node that directory belongs to whoever created it first.
    * ``TMPDIR`` is *tier 2* of ``env_resolver.resolve_cache_root``. A metrics
      import therefore silently redirected the framework-wide cache root, and
      whether it won depended on import order -- which no caller controls.
    * the assignment sat *below* this module's own ``import torch``. PyTorch
      reads these variables when the CUDA allocator initialises, so the write
      was inert on its own terms. ``main.py`` says exactly this in a comment
      and puts its equivalent line above ``import torch``.

    Checked in a subprocess rather than with ``importlib.reload``: reloading
    re-runs this module's ``@register_metric`` decorators and trips duplicate
    registration, so a reload-based test would fail for a reason unrelated to
    what it claims to measure.
    """

    #: Every variable the deleted block touched, plus the resolver tier it fed.
    FORBIDDEN = ("TMPDIR", "TORCH_CUDA_EAGER_CACHE_MANAGER", "SPECTRAMR_CACHE_ROOT")

    def _import_in_clean_subprocess(self) -> set[str]:
        """Import the module with the forbidden vars unset; report which appear."""
        import json
        import os
        import subprocess
        import sys

        import spectramr

        # Point the child at the SAME tree this test imported from. In a git
        # worktree an inherited PYTHONPATH otherwise resolves to the main
        # checkout, and the child would import a different file than the parent.
        src_root = str(pathlib.Path(spectramr.__file__).resolve().parent.parent)

        env = {k: v for k, v in os.environ.items() if k not in self.FORBIDDEN}
        env["PYTHONPATH"] = src_root

        code = (
            "import os, json, sys\n"
            f"forbidden = {list(self.FORBIDDEN)!r}\n"
            "import spectramr.core.metrics.evaluation_metrics  # noqa: F401\n"
            "json.dump([v for v in forbidden if v in os.environ], sys.stdout)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        assert proc.returncode == 0, f"import failed:\n{proc.stderr[-4000:]}"
        return set(json.loads(proc.stdout))

    def test_import_sets_no_cache_environment_variables(self) -> None:
        written = self._import_in_clean_subprocess()
        assert written == set(), (
            f"importing spectramr.core.metrics.evaluation_metrics set "
            f"{sorted(written)}. main.py owns this bootstrap -- it calls "
            "configure_cache_environment() above `import torch`, and every "
            "entry point reaches it. A write here is both import-order-"
            "dependent and (being below `import torch`) inert."
        )

    def test_module_body_contains_no_environment_write(self) -> None:
        """No import-time ``os.environ`` write, checked structurally.

        Deliberately an AST walk and not a substring search. The module
        docstring quotes the three removed lines -- with the username placeheld,
        the rest byte-for-byte -- so the next reader knows what was there -- a text search cannot tell that
        explanation apart from live code, and would fail on the fixed file.
        Only the parsed module body distinguishes the two.

        Scope is the module body, not the whole file: a write inside a function
        is a different (and legitimate) thing. What was wrong here was that it
        ran on ``import``.
        """
        import ast
        import inspect

        from spectramr.core.metrics import evaluation_metrics

        tree = ast.parse(inspect.getsource(evaluation_metrics))

        def _writes_environ(node: ast.AST) -> bool:
            """True for ``os.environ[...] = ...`` / ``os.environ.setdefault(...)``."""
            for sub in ast.walk(node):
                # os.environ["X"] = ... / os.environ["X"] += ...
                targets: list[ast.AST] = []
                if isinstance(sub, ast.Assign):
                    targets = list(sub.targets)
                elif isinstance(sub, ast.AugAssign):
                    targets = [sub.target]
                for tgt in targets:
                    if (
                        isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.value, ast.Attribute)
                        and tgt.value.attr == "environ"
                    ):
                        return True
                # os.environ.setdefault(...) / .update(...) / .pop(...)
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and isinstance(sub.func.value, ast.Attribute)
                    and sub.func.value.attr == "environ"
                    and sub.func.attr in {"setdefault", "update", "pop", "clear"}
                ):
                    return True
            return False

        offenders = [
            ast.dump(stmt)[:120]
            for stmt in tree.body
            if not isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            and _writes_environ(stmt)
        ]
        assert offenders == [], (
            f"module-level os.environ write(s) reintroduced: {offenders}. "
            "main.py owns environment bootstrap; see this module's docstring."
        )


# ---------------------------------------------------------------------------
# Issue #1347 — PSNR is graded per sample, then averaged.
#
# ``20*log10(DR/sqrt(MSE))`` is concave in MSE, so reducing MSE over a whole
# batch and taking the log ONCE is not the mean of the per-sample scores. The
# published number then depends on how the loader happened to group the images:
# 14.3 dB across batch_size 1 -> 24 on a heterogeneous 24-image set, with the
# predictions held bit-identical.
#
# The effect is driven by heterogeneity ACROSS samples and nearly vanishes on
# uniform synthetic data (0.005 dB), which is why every fixture in this suite
# was blind to it. Every fixture below is therefore deliberately heterogeneous.
# ---------------------------------------------------------------------------


def _heterogeneous_pair(n=24, h=8, w=8, seed=11):
    """``(preds, target)`` whose per-image error spans three decades."""
    g = torch.Generator().manual_seed(seed)
    target = torch.rand(n, 1, h, w, generator=g)
    scales = torch.logspace(-3, -0.5, n).view(n, 1, 1, 1)
    preds = (target + scales * torch.randn(n, 1, h, w, generator=g)).clamp(0, 1)
    return preds, target


def _sample_weighted_epoch(metric, preds, target, batch_size):
    """What ``_run_validation`` now computes: sum(score * n) / sum(n)."""
    total = 0.0
    seen = 0
    for start in range(0, preds.shape[0], batch_size):
        chunk = slice(start, start + batch_size)
        n = preds[chunk].shape[0]
        total += float(metric(preds[chunk], target[chunk])) * n
        seen += n
    return total / seen


class TestPSNRIsGradedPerSample:
    def test_batch_size_does_not_move_the_epoch_score(self):
        """The defect, stated as the property it violates.

        Bit-identical predictions, eight groupings, one number. At the parent
        commit the same loop spans ~14 dB.
        """
        from spectramr.core.metrics.evaluation_metrics import PSNR

        preds, target = _heterogeneous_pair()
        metric = PSNR(data_range=1.0)
        grouping = (1, 2, 3, 4, 6, 8, 12, 24)
        scores = [_sample_weighted_epoch(metric, preds, target, bs) for bs in grouping]
        assert max(scores) - min(scores) < 1e-4, (
            "epoch PSNR moved with batch_size on bit-identical predictions: "
            f"{[round(s, 4) for s in scores]}"
        )

    def test_a_batch_score_is_the_mean_of_its_per_image_scores(self):
        """Jensen, pinned directly: no batch-level MSE reduction survives."""
        from spectramr.core.metrics.evaluation_metrics import PSNR

        preds, target = _heterogeneous_pair()
        metric = PSNR(data_range=1.0)
        per_image = torch.tensor(
            [float(metric(preds[i : i + 1], target[i : i + 1])) for i in range(24)]
        )
        assert float(metric(preds, target)) == pytest.approx(float(per_image.mean()), abs=1e-4)

    def test_a_single_sample_batch_reproduces_the_pre_fix_formula(self):
        """The old formula is the ``N == 1`` case of the new one.

        This is the invariant that covers every per-image caller at once --
        sim2rank grades one image at a time, and its ~137 metrics must not move.
        """
        import torch.nn.functional as F

        from spectramr.core.metrics.evaluation_metrics import PSNR

        preds, target = _heterogeneous_pair(n=1)
        mse = F.mse_loss(preds, target, reduction="mean")
        expected = 20 * torch.log10(torch.tensor(1.0) / (torch.sqrt(mse) + 1e-10))
        assert float(PSNR(data_range=1.0)(preds, target)) == pytest.approx(
            float(expected), abs=1e-6
        )

    def test_a_tensor_with_no_sample_axis_is_one_sample(self):
        """A bare ``(C, H, W)`` must not have its channel axis read as a batch.

        ``reduction="mean"`` was dimension-agnostic; a per-sample reduction is
        not, so a direct caller passing an unbatched image is exactly where the
        fix could re-create the defect one layer down.
        """
        from spectramr.core.metrics.evaluation_metrics import PSNR

        metric = PSNR(data_range=1.0)
        g = torch.Generator().manual_seed(3)
        target = torch.rand(4, 8, 8, generator=g)  # 4 channels, spanning ranges
        target[0] *= 0.01  # one very dark channel -> a batch reading would move
        preds = (target + 0.05 * torch.randn(4, 8, 8, generator=g)).clamp(0, 1)
        assert float(metric(preds, target)) == pytest.approx(
            float(metric(preds.unsqueeze(0), target.unsqueeze(0))), abs=1e-6
        )

    def test_kspace_data_range_is_resolved_per_sample(self):
        """``domain="kspace"`` derives its range from the target's peak.

        Taken over the batch, the loudest spectrum sets the reference for every
        other one -- the same batch-composition dependence, arriving through the
        range instead of the reduction.
        """
        from spectramr.core.metrics.evaluation_metrics import PSNR

        metric = PSNR(domain="kspace")
        g = torch.Generator().manual_seed(5)
        quiet = torch.rand(1, 1, 8, 8, generator=g) * 0.01
        loud = torch.rand(1, 1, 8, 8, generator=g) * 100.0
        target = torch.cat([quiet, loud])
        preds = target * 0.9
        alone = [float(metric(preds[i : i + 1], target[i : i + 1])) for i in range(2)]
        assert float(metric(preds, target)) == pytest.approx(sum(alone) / 2, abs=1e-4)

    def test_use_target_max_range_is_resolved_per_sample(self):
        """``use_target_max`` is documented as a per-IMAGE range. Make it one."""
        from spectramr.core.metrics.evaluation_metrics import PSNR

        metric = PSNR(use_target_max=True)
        g = torch.Generator().manual_seed(6)
        dark = torch.rand(1, 1, 8, 8, generator=g) * 0.2
        bright = torch.rand(1, 1, 8, 8, generator=g)
        target = torch.cat([dark, bright])
        preds = (target + 0.02 * torch.randn(2, 1, 8, 8, generator=g)).clamp(min=0)
        alone = [float(metric(preds[i : i + 1], target[i : i + 1])) for i in range(2)]
        assert float(metric(preds, target)) == pytest.approx(sum(alone) / 2, abs=1e-4)

    def test_the_contract_range_is_deliberately_not_per_sample(self):
        """The default path resolves a CONTRACT, not an extent -- keep it batch-wide.

        A per-sample sign test would read an all-positive sample of a [-1, 1]
        dataset as [0, 1] and halve its range. This pins the decision, not an
        accident: sample 0 is all-positive, sample 1 is signed, and both must be
        graded at DR = 2.
        """
        from spectramr.core.metrics.evaluation_metrics import PSNR

        metric = PSNR()
        g = torch.Generator().manual_seed(7)
        positive = torch.rand(1, 1, 8, 8, generator=g)
        signed = torch.rand(1, 1, 8, 8, generator=g) * 2.0 - 1.0
        target = torch.cat([positive, signed])
        preds = target.clone()
        preds[0, 0, 0, 0] += 0.1
        batched = float(metric(preds, target))
        # Sample 0 alone resolves DR = 1 (it has no negatives), so a per-sample
        # range would score it 20*log10(2) = 6.02 dB lower than the batch does.
        alone = float(metric(preds[:1], target[:1]))
        assert alone < batched
        two = float(metric(preds[:1], target[:1], data_range=2.0))
        assert two - alone == pytest.approx(20 * math.log10(2.0), abs=1e-4)

    def test_an_all_zero_kspace_sample_stays_finite(self):
        """The empty-spectrum fallback survives the vectorised range."""
        from spectramr.core.metrics.evaluation_metrics import PSNR

        target = torch.cat([torch.zeros(1, 1, 8, 8), torch.rand(1, 1, 8, 8)])
        preds = target + 0.01
        assert math.isfinite(float(PSNR(domain="kspace")(preds, target)))

    def test_the_per_image_range_paths_no_longer_sync(self):
        """Both per-image range branches used ``.item()`` on a device tensor.

        That is one GPU sync per range-sensitive metric per validation batch
        (non-negotiable 9). Vectorising the range removed them; pin it by AST so
        a future edit cannot put a sync back while the numbers stay right.
        """
        import ast
        import inspect
        import textwrap

        from spectramr.core.metrics.evaluation_metrics import PSNR

        tree = ast.parse(textwrap.dedent(inspect.getsource(PSNR.compute_metric)))
        syncs = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"item", "tolist", "cpu", "numpy"}
        ]
        assert syncs == [], f"host sync reintroduced into PSNR.compute_metric: {syncs}"


def test_power_spectrum_radial_profile_matches_per_bin_loop() -> None:
    """The scatter-based profile equals the per-bin, per-sample mask loop it replaced."""
    import torch

    from spectramr.core.metrics.evaluation_metrics import PowerSpectrumConsistency
    from spectramr.infrastructure.physics.fft_ops import fft2c

    torch.manual_seed(0)
    m = PowerSpectrumConsistency(num_bins=12)
    img = torch.randn(3, 2, 33, 40)
    power = (torch.abs(fft2c(img)) ** 2).mean(dim=1)
    H, W = img.shape[-2:]
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H) - cy, torch.arange(W) - cx, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2).float()
    edges = torch.linspace(0, min(cy, cx), 13)
    ref = torch.zeros(3, 12)
    for i in range(12):
        mask = (r >= edges[i]) & (r < edges[i + 1])
        for b in range(3):
            if mask.sum() > 0:
                ref[b, i] = power[b][mask].mean()
    torch.testing.assert_close(m.radial_profile(img), ref, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 64, 64), (1, 1, 128, 128), (1, 1, 192, 192, 16), (1, 1, 16, 192, 192)],
)
def test_ms_ssim_declares_images_below_the_scale_floor_not_applicable(shape) -> None:
    from spectramr.core.metrics.outcome import MetricNotApplicableError, NotApplicableReason

    ms_ssim = MetricsRegistry.get("ms_ssim")
    x = torch.rand(*shape)
    with pytest.raises(MetricNotApplicableError) as exc:
        ms_ssim.compute_metric(x, x)
    assert exc.value.reason == NotApplicableReason.IMAGE_TOO_SMALL_FOR_METRIC


def test_ms_ssim_still_scores_a_large_enough_image() -> None:
    ms_ssim = MetricsRegistry.get("ms_ssim")
    x = torch.rand(1, 1, 176, 176)
    assert float(ms_ssim.compute_metric(x, x)) == pytest.approx(1.0, abs=1e-4)
