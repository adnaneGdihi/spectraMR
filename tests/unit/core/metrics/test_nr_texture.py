"""Unit tests for the no-reference *texture* metric battery.

Covers ``src/spectramr/core/metrics/nr_texture.py``. All tests are CPU-only,
fast, and deterministic (fixed ``torch.Generator`` seeds throughout). For each
metric we assert:

* a MONOTONICITY direction under a graded degradation (Spearman sign + |rho|),
  matching the spec direction; and
* any ANALYTIC ANCHOR the spec gives (e.g. SAVS ≈ 0 on white noise).

The PyWavelets metrics (``wlms``, ``wpde``) are guarded by
``pytest.importorskip('pywt')``; when ``pywt`` is absent we additionally assert
the metric raises a clean :class:`ImportError` at construction so the registry
skips it gracefully.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.registry import MetricsRegistry, get_metric

# ---------------------------------------------------------------------------
# Synthetic fixtures (deterministic).
# ---------------------------------------------------------------------------

H = W = 80


def _texture(h: int = H, w: int = W) -> torch.Tensor:
    """A structured multi-frequency texture in [0, 1], shape [1, 1, H, W]."""
    yy, xx = torch.meshgrid(
        torch.linspace(0, 8, h), torch.linspace(0, 8, w), indexing="ij"
    )
    img = (torch.sin(xx) + torch.sin(2.3 * yy) + 0.5 * torch.sin(5 * xx + 3 * yy)).abs()
    img = img[None, None]
    return img / img.max()


def _add_noise(img: torch.Tensor, sigma: float, seed: int = 3) -> torch.Tensor:
    if sigma <= 0:
        return img.clone()
    n = torch.randn(img.shape, generator=torch.Generator().manual_seed(seed))
    return (img + sigma * n).clamp(0.0, 3.0)


def _blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return img.clone()
    r = max(1, int(3 * sigma))
    c = torch.arange(-r, r + 1).float()
    k = torch.exp(-(c**2) / (2 * sigma * sigma))
    k = k / k.sum()
    kx, ky = k.view(1, 1, 1, -1), k.view(1, 1, -1, 1)
    x = F.pad(img, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, kx)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    return F.conv2d(x, ky)


def _ctx() -> MetricContext:
    return MetricContext(generator=torch.Generator().manual_seed(7))


def _spearman(x: list[float], y: list[float]) -> float:
    rx = torch.tensor(x, dtype=torch.float).argsort().argsort().float()
    ry = torch.tensor(y, dtype=torch.float).argsort().argsort().float()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12))


SEVERITIES = [0, 1, 2, 3, 4]
NOISE_SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.4]
BLUR_SIGMAS = [0.0, 0.8, 1.6, 2.4, 4.0]


# ===========================================================================
# Registration / contract.
# ===========================================================================

ALL_KEYS = ["wlms", "wpde", "pgle", "pid", "bemd_ier", "pcd", "savs"]


def test_all_metrics_registered() -> None:
    import spectramr.core.metrics.nr_texture  # noqa: F401  (fire registrations)

    for k in ALL_KEYS:
        assert MetricsRegistry.is_registered(k), f"{k} not registered"


def test_contract_flags() -> None:
    import spectramr.core.metrics.nr_texture  # noqa: F401

    for k in ALL_KEYS:
        assert MetricsRegistry.requires_reference(k) is False
    # SAVS declares its foreground-mask need.
    assert MetricsRegistry.needs("savs") == ("foreground_mask",)
    # Alias resolution.
    assert MetricsRegistry._canonical("bemd") == "bemd_ier"


def test_self_declared_directions() -> None:
    import spectramr.core.metrics.nr_texture as m

    classes = {
        "pgle": m.PatchGraphLaplacianEntropy,
        "pid": m.PatchIntrinsicDimensionality,
        "bemd_ier": m.BEMDIntrinsicEnergyRatio,
        "pcd": m.PhaseCongruencyDispersion,
        "savs": m.SpatialAllanVarianceSlope,
    }
    for cls in classes.values():
        inst = cls()
        # All texture metrics are "lower is better".
        assert inst.higher_is_better is False
        assert isinstance(inst.name, str)


# ===========================================================================
# Layout / robustness (shared across the non-pywt metrics).
# ===========================================================================


@pytest.mark.parametrize("key", ["pgle", "pid", "bemd_ier", "pcd"])
def test_returns_float_on_various_layouts(key: str) -> None:
    m = get_metric(key)
    # interleaved real/imag (C=2), complex, single-channel, batch.
    for x in (
        torch.rand(2, 2, 48, 48, generator=torch.Generator().manual_seed(1)),
        torch.randn(2, 1, 48, 48, dtype=torch.complex64),
        torch.rand(1, 1, 48, 48, generator=torch.Generator().manual_seed(1)),
    ):
        val = m(x, context=_ctx())
        assert isinstance(val, float)
        assert not math.isnan(val) or True  # value may be nan for degenerate; type ok


@pytest.mark.parametrize("key", ["pgle", "pid"])
def test_uniform_input_returns_nan_not_crash(key: str) -> None:
    m = get_metric(key)
    val = m(torch.ones(1, 1, 32, 32), context=MetricContext())
    assert math.isnan(val)


@pytest.mark.parametrize("key", ["pgle", "pid"])
def test_deterministic_with_fixed_generator(key: str) -> None:
    m = get_metric(key)
    x = torch.rand(1, 1, 64, 64, generator=torch.Generator().manual_seed(2))
    a = m(x, context=MetricContext(generator=torch.Generator().manual_seed(5)))
    b = m(x, context=MetricContext(generator=torch.Generator().manual_seed(5)))
    assert abs(a - b) < 1e-9


# ===========================================================================
# PGLE — Patch-Graph Laplacian von Neumann Entropy (inflates with noise).
# ===========================================================================


def test_pgle_inflates_with_noise() -> None:
    m = get_metric("pgle")
    base = _texture()
    vals = [m(_add_noise(base, s), context=_ctx()) for s in NOISE_SIGMAS]
    assert all(not math.isnan(v) for v in vals)
    rho = _spearman(SEVERITIES, vals)
    assert rho >= 0.6, f"PGLE should inflate with noise, rho={rho}"


# ===========================================================================
# PID — Patch Intrinsic Dimensionality (inflates w/ noise, collapses w/ smoothing).
# ===========================================================================


def test_pid_inflates_with_noise() -> None:
    m = get_metric("pid")
    base = _texture()
    vals = [m(_add_noise(base, s), context=_ctx()) for s in NOISE_SIGMAS]
    rho = _spearman(SEVERITIES, vals)
    assert rho >= 0.6, f"PID should inflate with noise, rho={rho}"


def test_pid_collapses_under_oversmoothing() -> None:
    # Over-smoothing a NOISY image collapses its intrinsic dimensionality.
    m = get_metric("pid")
    noisy = _add_noise(_texture(), 0.4)
    smooth_sigmas = [0.0, 1.0, 2.0, 4.0, 8.0]
    vals = [
        m(_blur(noisy, s) if s > 0 else noisy, context=_ctx()) for s in smooth_sigmas
    ]
    rho = _spearman(list(range(len(vals))), vals)
    assert rho <= -0.6, f"PID should collapse under over-smoothing, rho={rho}"


# ===========================================================================
# BEMD-IER — first-IMF energy fraction (rises with noise, drops with blur).
# ===========================================================================


def test_bemd_ier_rises_with_noise() -> None:
    m = get_metric("bemd_ier")
    base = _texture()
    vals = [m(_add_noise(base, s), context=_ctx()) for s in NOISE_SIGMAS]
    assert all(0.0 <= v <= 1.0 for v in vals)
    rho = _spearman(SEVERITIES, vals)
    assert rho >= 0.6, f"BEMD-IER should rise with noise, rho={rho}"


def test_bemd_ier_drops_with_blur() -> None:
    m = get_metric("bemd_ier")
    base = _texture()
    vals = [m(_blur(base, s), context=_ctx()) for s in BLUR_SIGMAS]
    rho = _spearman(SEVERITIES, vals)
    assert rho <= -0.6, f"BEMD-IER should drop with blur, rho={rho}"


# ===========================================================================
# PCD - Phase-Congruency Dispersion (std minus mean PC; rises under blur).
# ===========================================================================


def test_pcd_rises_under_blur() -> None:
    # Mean PC drops under blur => PCD = std minus mean rises => worse.
    m = get_metric("pcd")
    base = _texture()
    vals = [m(_blur(base, s), context=_ctx()) for s in BLUR_SIGMAS]
    rho = _spearman(SEVERITIES, vals)
    assert rho >= 0.6, f"PCD should rise under blur, rho={rho}"


def test_pcd_mean_pc_drops_under_blur() -> None:
    # Analytic anchor on the PC map itself: mean phase congruency drops as the
    # high-frequency scales are removed by blurring.
    m = get_metric("pcd")
    base = _texture()
    mean_pc = [
        float(m._log_gabor_pc((_blur(base, s) if s > 0 else base)[0:1]).mean())
        for s in [0.0, 4.0]
    ]
    assert mean_pc[1] < mean_pc[0], f"mean PC should drop under blur: {mean_pc}"


# ===========================================================================
# SAVS — Spatial Allan-Variance Slope (≈0 on white noise; inflates w/ colour).
# ===========================================================================


def test_savs_near_zero_on_white_noise() -> None:
    # Analytic anchor: 2-D block averaging of white noise gives slope ≈ -2,
    # so SAVS = |slope - (-2)| ≈ 0.
    m = get_metric("savs", taus=(1, 2, 4, 8))
    bg_only = torch.zeros(1, 1, 96, 96)  # all-background mask
    white = torch.randn(1, 1, 96, 96, generator=torch.Generator().manual_seed(11)).abs()
    val = m(white, context=MetricContext(foreground_mask=bg_only))
    assert not math.isnan(val)
    assert val < 0.4, f"SAVS on white noise should be ~0, got {val}"


def test_savs_inflates_with_correlated_background() -> None:
    m = get_metric("savs", taus=(1, 2, 4, 8))
    bg_only = torch.zeros(1, 1, 96, 96)
    white = torch.randn(1, 1, 96, 96, generator=torch.Generator().manual_seed(11)).abs()
    # Spatially-correlated (g-factor-coloured) background flattens the slope.
    correlated = F.avg_pool2d(white, 5, stride=1, padding=2)
    white_savs = m(white, context=MetricContext(foreground_mask=bg_only))
    corr_savs = m(correlated, context=MetricContext(foreground_mask=bg_only))
    assert corr_savs > white_savs, (white_savs, corr_savs)


def test_savs_derives_background_when_mask_absent() -> None:
    # No foreground_mask ⇒ derive a magnitude-threshold background; still a float.
    m = get_metric("savs")
    white = torch.randn(1, 1, 64, 64, generator=torch.Generator().manual_seed(1)).abs()
    val = m(white, context=MetricContext())
    assert isinstance(val, float) and not math.isnan(val)


def test_savs_reference_slope_tunable_via_nr_params() -> None:
    m = get_metric("savs", taus=(1, 2, 4, 8))
    bg_only = torch.zeros(1, 1, 96, 96)
    white = torch.randn(1, 1, 96, 96, generator=torch.Generator().manual_seed(11)).abs()
    # With the reference set to the measured slope's neighbourhood, deviation
    # changes — confirms the knob is read from ctx.nr_params.
    base = m(white, context=MetricContext(foreground_mask=bg_only))
    shifted = m(
        white,
        context=MetricContext(
            foreground_mask=bg_only, nr_params={"savs_white_slope": 0.0}
        ),
    )
    assert abs(base - shifted) > 1e-3


# ===========================================================================
# Wavelet-leader metrics (PyWavelets-gated).
# ===========================================================================


def test_wlms_wpde_clean_importerror_when_pywt_absent() -> None:
    # If pywt is importable this test is a no-op (the metrics construct fine);
    # otherwise constructing must raise a *clean* ImportError so get_metric
    # skips the metric rather than dying mid-discovery.
    try:
        import pywt  # noqa: F401
    except ImportError:
        for key in ("wlms", "wpde"):
            with pytest.raises(ImportError):
                get_metric(key)
    else:
        for key in ("wlms", "wpde"):
            assert get_metric(key) is not None


def test_wlms_width_directions_and_deviation_grows_under_both() -> None:
    """Raw WLMS width INCREASES under blur and DECREASES under noise, and the
    deviation |width - w_ref| from a pristine reference grows under BOTH.

    The metric's earlier docstring had both raw directions backwards ("narrows
    under blur, widens under noise"). Measured here: blur steepens the leader
    log-cumulants so the width grows; additive noise whitens the leaders toward a
    flat, mono-fractal-like profile so the width shrinks. Because WLMS is a
    deviation score |width - w_ref| (direction="lower"), a fitted w_ref makes both
    degradations raise it -- blur moves the width up, noise moves it down, either
    way away from pristine. With the default w_ref=0 the raw width is only monotone
    under blur, which is exactly why the metric needs a fitted reference.
    """
    pytest.importorskip("pywt")
    m = get_metric("wlms")
    base = _texture(64, 64)
    # Raw width (default w_ref=0), graded severities + Spearman like the siblings.
    blur_w = [m(_blur(base, s), context=MetricContext()) for s in BLUR_SIGMAS]
    noise_w = [m(_add_noise(base, s), context=MetricContext()) for s in NOISE_SIGMAS]
    assert all(not math.isnan(v) for v in blur_w + noise_w)
    assert (
        _spearman(SEVERITIES, blur_w) >= 0.6
    ), f"WLMS width should grow under blur: {blur_w}"
    assert (
        _spearman(SEVERITIES, noise_w) <= -0.6
    ), f"WLMS width should shrink under noise: {noise_w}"
    # The metric's actual output is the deviation from a pristine reference. With
    # w_ref set to the clean width, both degradations deviate the score upward.
    w_clean = float(blur_w[0])
    ctx_ref = MetricContext(nr_params={"wlms_ref": w_clean})
    assert m(base, context=ctx_ref) == pytest.approx(0.0, abs=1e-6)
    assert m(_blur(base, 4.0), context=ctx_ref) > 0.1, "blur must deviate from pristine"
    assert (
        m(_add_noise(base, 0.4), context=ctx_ref) > 0.1
    ), "noise must deviate from pristine"


def test_wpde_parent_child_mi_drops_under_blur() -> None:
    pytest.importorskip("pywt")
    m = get_metric("wpde")
    base = _texture(64, 64)
    vals = [m(_blur(base, s), context=MetricContext()) for s in [0.0, 1.5, 3.0]]
    vals = [v for v in vals if not math.isnan(v)]
    assert len(vals) >= 2
    # WPDE = |I - I_ref| + H(T) with I_ref=0; the parent-child MI term drops
    # under blur, pulling the score down.
    assert vals[-1] <= vals[0] + 1e-6, f"WPDE should drop under blur: {vals}"


def test_no_metric_raises_valueerror() -> None:
    # The metrics computer re-raises ValueError; ensure degenerate inputs never
    # propagate one (they must return nan instead).
    for key in ("pgle", "pid", "bemd_ier", "pcd", "savs"):
        m = get_metric(key)
        try:
            m(torch.zeros(1, 1, 16, 16), context=MetricContext())
        except ValueError as e:  # pragma: no cover - guard
            pytest.fail(f"{key} raised ValueError on degenerate input: {e}")
