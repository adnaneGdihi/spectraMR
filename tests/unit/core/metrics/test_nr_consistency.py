"""Unit tests for the ``nr_consistency`` no-reference metric block.

Each metric gets:
  (a) a MONOTONICITY test — a graded degradation swept over 3-5 severities,
      asserting the metric moves in the spec'd direction (Spearman sign);
  (b) the ANALYTIC ANCHOR the spec gives.

All tests are CPU-only, fast and deterministic (fixed ``torch.manual_seed`` /
seeded ``MetricContext.generator``). Context-needing metrics build a synthetic
``MetricContext`` whose ``y_kspace = sense_forward(clean_img, smaps, mask)``.
"""

from __future__ import annotations

import math

import pytest
import torch

# Importing the module registers the five metrics.
import spectramr.core.metrics.nr_consistency as nrc
from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.registry import MetricsRegistry, get_metric
from spectramr.infrastructure.physics.fft_ops import (
    fft2c,
    ifft2c,
    sense_adjoint,
    sense_forward,
)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation (no scipy dependency)."""
    n = len(a)

    def rank(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r

    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)))
    vb = math.sqrt(sum((rb[i] - mb) ** 2 for i in range(n)))
    if va == 0 or vb == 0:
        return 0.0
    return cov / (va * vb)


def _blob(h: int = 32, w: int = 32) -> torch.Tensor:
    """Smooth Gaussian blob image ``[1,1,H,W]`` complex."""
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    img = torch.exp(-(xx**2 + yy**2) / 0.3)
    return img.view(1, 1, h, w).to(torch.complex64)


def _smooth_smaps(c: int = 4, h: int = 32, w: int = 32) -> torch.Tensor:
    """RSS-normalised smooth complex coil maps ``[1,C,H,W]``."""
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    maps = []
    for k in range(c):
        ph = 0.5 * (k + 1) * (xx + yy)
        amp = 0.6 + 0.4 * torch.cos(0.5 * (k + 1) * xx)
        maps.append(amp * torch.exp(1j * ph))
    s = torch.stack(maps, 0).view(1, c, h, w).to(torch.complex64)
    rss = torch.sqrt((s.real**2 + s.imag**2).sum(1, keepdim=True) + 1e-8)
    return s / rss


# --------------------------------------------------------------------------- #
# registration                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_all_registered_with_contract():
    """All five metrics register with the declared NR contract."""
    expected = {
        "ndcr": (False, ("y_kspace", "mask", "coil_maps")),
        "prp_snr": (True, ("y_kspace", "mask", "coil_maps", "reconstructor")),
        "fmrs": (False, ("reconstructor", "mask")),
        "csoe": (False, ("y_kspace", "coil_maps")),
        "aoqv": (False, ("acq_order", "y_kspace", "reconstructor")),
    }
    for key, (hib, needs) in expected.items():
        assert MetricsRegistry.is_registered(key)
        assert MetricsRegistry.requires_reference(key) is False
        assert MetricsRegistry.needs(key) == needs
        assert get_metric(key).higher_is_better is hib


@pytest.mark.unit
def test_metrics_return_nan_without_context():
    """Absent measurement context ⇒ nan (never raise — computer.py re-raises ValueError)."""
    img = _blob()
    for key in ("ndcr", "prp_snr", "fmrs", "csoe", "aoqv"):
        out = get_metric(key)(img, context=MetricContext())
        assert math.isnan(out)


# --------------------------------------------------------------------------- #
# ndcr                                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_ndcr_anchor_pseudoinverse_near_zero():
    """A^+ y on a noiseless synthetic ⇒ NDCR → ~0; zeros ⇒ NDCR = 1."""
    torch.manual_seed(0)
    img, smaps = _blob(), _smooth_smaps()
    mask = torch.ones(32, 32)
    y = sense_forward(img, smaps=smaps, mask=mask)
    ctx = MetricContext(y_kspace=y, mask=mask, coil_maps=smaps)
    xrec = sense_adjoint(y, smaps=smaps, mask=mask)  # A^+ y (normalised maps)

    ndcr = get_metric("ndcr")
    assert ndcr(xrec, context=ctx) < 1e-4
    assert abs(ndcr(torch.zeros(1, 1, 32, 32), context=ctx) - 1.0) < 1e-3


@pytest.mark.unit
def test_ndcr_monotonic_in_kspace_error():
    """NDCR rises monotonically with injected k-space error."""
    torch.manual_seed(0)
    img, smaps = _blob(), _smooth_smaps()
    mask = torch.ones(32, 32)
    y = sense_forward(img, smaps=smaps, mask=mask)
    ctx = MetricContext(y_kspace=y, mask=mask, coil_maps=smaps)
    xrec = sense_adjoint(y, smaps=smaps, mask=mask)
    ndcr = get_metric("ndcr")

    sevs = [0.0, 0.05, 0.1, 0.2, 0.4]
    g = torch.Generator().manual_seed(7)
    noise = torch.randn(1, 1, 32, 32, generator=g).to(torch.complex64)
    vals = [ndcr(xrec + s * noise, context=ctx) for s in sevs]
    assert _spearman(sevs, vals) > 0.9


@pytest.mark.unit
def test_ndcr_mask_none_is_fully_sampled():
    """Regression: ``mask=None`` is a fully-sampled measurement, NOT 'no context'.

    The meta-evaluation engine + offline harness synthesise the measurement with
    ``mask=None`` (the ``sense_forward``/``ifft2c`` fully-sampled convention). A
    prior over-strict guard required a non-None mask, so NDCR — the flagship NR
    metric — silently returned NaN on every standard sweep. NDCR must be finite,
    near-zero on the anchor, and monotone in injected error with no mask supplied.
    """
    torch.manual_seed(0)
    img, smaps = _blob(), _smooth_smaps()
    y = sense_forward(img, smaps=smaps, mask=None)
    ctx_none = MetricContext(y_kspace=y, mask=None, coil_maps=smaps)
    xrec = sense_adjoint(y, smaps=smaps, mask=None)
    ndcr = get_metric("ndcr")

    anchor = ndcr(xrec, context=ctx_none)
    assert anchor == anchor and anchor < 1e-4  # finite (not NaN) and near zero

    sevs = [0.0, 0.05, 0.1, 0.2, 0.4]
    g = torch.Generator().manual_seed(7)
    noise = torch.randn(1, 1, 32, 32, generator=g).to(torch.complex64)
    vals = [ndcr(xrec + s * noise, context=ctx_none) for s in sevs]
    assert all(v == v for v in vals)  # no NaNs
    assert _spearman(sevs, vals) > 0.9

    # mask=None must equal an explicit all-ones mask (same measurement).
    ones = torch.ones(32, 32)
    ctx_ones = MetricContext(y_kspace=y, mask=ones, coil_maps=smaps)
    assert abs(ndcr(xrec + 0.2 * noise, context=ctx_none)
               - ndcr(xrec + 0.2 * noise, context=ctx_ones)) < 1e-5


# --------------------------------------------------------------------------- #
# prp_snr                                                                      #
# --------------------------------------------------------------------------- #
class _PseudoReplicaRecon:
    """Reconstructor that returns ``base + indep complex noise`` per partition.

    Models a real recon whose pseudo-replicas differ only by measurement noise:
    each call draws a fresh realization, so x1 and x2 are independent.
    """

    def __init__(self, base: torch.Tensor, sigma: float, seed: int = 0) -> None:
        self.base = base
        self.sigma = sigma
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self, yk, mk, smaps):
        h, w = self.base.shape[-2], self.base.shape[-1]
        re = torch.randn(1, 1, h, w, generator=self.g)
        im = torch.randn(1, 1, h, w, generator=self.g)
        return self.base + self.sigma * (re + 1j * im) / math.sqrt(2.0)


@pytest.mark.unit
def test_prp_snr_anchor_sigma_matches_injected_variance():
    """sigmâ² = ½E[|d|²] ≈ sigma² on synthetic Gaussian pseudo-replica noise."""
    torch.manual_seed(0)
    h = w = 64
    base = torch.zeros(1, 1, h, w).to(torch.complex64)  # zero signal ⇒ pure noise
    mask_full = torch.ones(h, w)
    sigma = 0.3
    recon = _PseudoReplicaRecon(base, sigma, seed=3)
    metric = nrc.PseudoReplicaPartitionSNR()
    ctx = MetricContext(
        y_kspace=sense_forward(base, mask=mask_full),
        mask=mask_full,
        coil_maps=None,
        reconstructor=recon,
    )
    x1 = metric._recon(ctx, nrc._split_mask_columns(mask_full, parity=0))
    x2 = metric._recon(ctx, nrc._split_mask_columns(mask_full, parity=1))
    d = x1 - x2
    sigma2_hat = float(0.5 * (d.real**2 + d.imag**2).mean())
    assert sigma2_hat == pytest.approx(sigma**2, rel=0.1)


@pytest.mark.unit
def test_prp_snr_monotonic_decreasing_with_noise():
    """PRP-SNR drops monotonically as the pseudo-replica noise grows."""
    torch.manual_seed(0)
    img = _blob(64, 64)
    mask_full = torch.ones(64, 64)
    y = sense_forward(img, mask=mask_full)
    prp = get_metric("prp_snr")

    sevs = [0.05, 0.1, 0.2, 0.4]
    vals = []
    for i, s in enumerate(sevs):
        recon = _PseudoReplicaRecon(img, s, seed=100 + i)
        ctx = MetricContext(y_kspace=y, mask=mask_full, reconstructor=recon)
        vals.append(prp(img, context=ctx))
    # higher_is_better ⇒ rho with severity must be strongly negative.
    assert _spearman(sevs, vals) < -0.9


@pytest.mark.unit
def test_prp_snr_fallback_adjoint_runs():
    """Without a reconstructor, the SENSE-adjoint fallback yields a finite SNR."""
    torch.manual_seed(0)
    img = _blob()
    mask_full = torch.ones(32, 32)
    y = sense_forward(img, mask=mask_full)
    ctx = MetricContext(y_kspace=y, mask=mask_full)
    out = get_metric("prp_snr")(img, context=ctx)
    assert math.isfinite(out) and out >= 0.0


# --------------------------------------------------------------------------- #
# fmrs                                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_fmrs_anchor_projection_recon_near_zero():
    """A projection-like (adjoint) reconstructor ⇒ FMRS ≈ 0."""
    from spectramr.infrastructure.physics.sampling import MaskGenerator

    torch.manual_seed(0)
    img = _blob()
    # Realistic R=2 mask sets the resampling density (a full-density resample is
    # a degenerate identity that the VD generator cannot produce).
    mask = MaskGenerator(seed=2).generate_mask(
        "variable_density", (32, 32), acceleration_factor=2.0
    )

    def proj_recon(yk, mk, smaps):
        return sense_adjoint(yk, smaps=smaps, mask=mk)

    ctx = MetricContext(mask=mask, reconstructor=proj_recon)
    out = get_metric("fmrs")(img, context=ctx)
    assert math.isfinite(out)
    assert out < 0.05


@pytest.mark.unit
def test_fmrs_monotonic_with_recon_instability():
    """FMRS rises as the reconstructor is made progressively less idempotent."""
    from spectramr.infrastructure.physics.sampling import MaskGenerator

    torch.manual_seed(0)
    img = _blob()
    mask = MaskGenerator(seed=5).generate_mask(
        "variable_density", (32, 32), acceleration_factor=2.0
    )
    fmrs = get_metric("fmrs")

    sevs = [0.0, 0.25, 0.5, 1.0]
    vals = []
    for s in sevs:
        # Reconstructor = adjoint blended with an additive perturbation; s scales
        # the deviation from the stable projection.
        def unstable_recon(yk, mk, smaps, _s=s):
            r = sense_adjoint(yk, smaps=smaps, mask=mk)
            g = torch.Generator().manual_seed(int(_s * 1000) + 1)
            pert = torch.randn(*r.shape, generator=g).to(torch.complex64)
            return r + _s * 0.5 * pert
        ctx = MetricContext(mask=mask, reconstructor=unstable_recon)
        vals.append(fmrs(img, context=ctx))
    assert _spearman(sevs, vals) > 0.9


@pytest.mark.unit
def test_fmrs_fallback_roundtrip_runs():
    """Without a reconstructor, the adjoint∘forward round-trip is finite and small."""
    torch.manual_seed(0)
    img = _blob()
    mask = torch.ones(32, 32)
    out = get_metric("fmrs")(img, context=MetricContext(mask=mask))
    assert math.isfinite(out) and out >= 0.0


# --------------------------------------------------------------------------- #
# csoe                                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_csoe_anchor_consistent_coils_near_zero():
    """Coil data = single image x smooth maps ⇒ CSOE ≈ 0."""
    torch.manual_seed(0)
    img, smaps = _blob(), _smooth_smaps()
    y = sense_forward(img, smaps=smaps)  # multi-coil k-space
    ctx = MetricContext(y_kspace=y, coil_maps=smaps)
    out = get_metric("csoe")(img, context=ctx)
    assert out < 1e-6


@pytest.mark.unit
def test_csoe_single_coil_returns_nan():
    """A single-coil cohort makes CSOE undefined ⇒ nan."""
    torch.manual_seed(0)
    img, smaps = _blob(), _smooth_smaps()
    y = sense_forward(img, smaps=smaps)
    ctx = MetricContext(y_kspace=y[:, :1], coil_maps=smaps[:, :1])
    assert math.isnan(get_metric("csoe")(img, context=ctx))


@pytest.mark.unit
def test_csoe_monotonic_with_intercoil_inconsistency():
    """CSOE rises with simulated inter-coil (per-pixel random-phase) inconsistency."""
    torch.manual_seed(0)
    img, smaps = _blob(), _smooth_smaps()
    y0 = sense_forward(img, smaps=smaps)
    csoe = get_metric("csoe")

    sevs = [0.0, 0.25, 0.5, 1.0]
    vals = []
    g = torch.Generator().manual_seed(11)
    phase = torch.randn(32, 32, generator=g)
    for s in sevs:
        y = y0.clone()
        # Corrupt coil 1 with a graded per-pixel random phase (in image domain).
        from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

        coil1_img = ifft2c(y[:, 1:2])
        coil1_img = coil1_img * torch.exp(1j * s * phase)
        y[:, 1:2] = fft2c(coil1_img)
        vals.append(csoe(img, context=MetricContext(y_kspace=y, coil_maps=smaps)))
    assert _spearman(sevs, vals) >= 0.6


@pytest.mark.unit
def test_csoe_estimates_smaps_when_absent():
    """With no coil_maps, CSOE estimates ESPIRiT maps and stays finite & small."""
    torch.manual_seed(0)
    img, smaps = _blob(), _smooth_smaps()
    y = sense_forward(img, smaps=smaps)
    out = get_metric("csoe")(img, context=MetricContext(y_kspace=y))
    assert math.isfinite(out) and out >= 0.0


# --------------------------------------------------------------------------- #
# aoqv                                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_aoqv_anchor_single_jump_adds_j_squared():
    """A single injected jump of magnitude J between breakpoints adds ≈ J²."""
    torch.manual_seed(0)
    h = w = 32
    y = torch.randn(1, 1, h, w) + 1j * torch.randn(1, 1, h, w)
    mask_full = torch.ones(h, w)
    base = torch.zeros(1, 1, h, w).to(torch.complex64)
    jump = torch.zeros(1, 1, h, w).to(torch.complex64)
    jump[0, 0, 0, 0] = 1.0
    j_mag = 3.0
    thresh = w // 2

    class JumpRecon:
        def __call__(self, yk, mk, smaps):
            m = mk.real if torch.is_complex(mk) else mk
            ncols = int((m > 0).any(0).sum().item())
            return base + (j_mag * jump if ncols > thresh else 0.0)

    ctx = MetricContext(y_kspace=y, mask=mask_full, reconstructor=JumpRecon())
    out = nrc.AcquisitionOrderQuadraticVariation(num_breaks=8)(
        torch.zeros(1, 1, h, w), context=ctx
    )
    assert out == pytest.approx(j_mag**2, rel=1e-3)


@pytest.mark.unit
def test_aoqv_stationary_recon_is_zero():
    """A stationary running reconstruction has zero quadratic variation."""
    torch.manual_seed(0)
    h = w = 32
    y = torch.randn(1, 1, h, w) + 1j * torch.randn(1, 1, h, w)
    base = torch.zeros(1, 1, h, w).to(torch.complex64)

    class StatRecon:
        def __call__(self, yk, mk, smaps):
            return base.clone()

    ctx = MetricContext(y_kspace=y, mask=torch.ones(h, w), reconstructor=StatRecon())
    out = nrc.AcquisitionOrderQuadraticVariation(num_breaks=8)(
        torch.zeros(1, 1, h, w), context=ctx
    )
    assert out == pytest.approx(0.0, abs=1e-8)


@pytest.mark.unit
def test_aoqv_monotonic_with_jump_magnitude():
    """AOQV grows monotonically (∝ J²) with the injected jump magnitude."""
    torch.manual_seed(0)
    h = w = 32
    y = torch.randn(1, 1, h, w) + 1j * torch.randn(1, 1, h, w)
    base = torch.zeros(1, 1, h, w).to(torch.complex64)
    jump = torch.zeros(1, 1, h, w).to(torch.complex64)
    jump[0, 0, 0, 0] = 1.0
    thresh = w // 2

    def make_ctx(j_mag: float) -> MetricContext:
        class JumpRecon:
            def __call__(self, yk, mk, smaps):
                m = mk.real if torch.is_complex(mk) else mk
                ncols = int((m > 0).any(0).sum().item())
                return base + (j_mag * jump if ncols > thresh else 0.0)

        return MetricContext(y_kspace=y, mask=torch.ones(h, w), reconstructor=JumpRecon())

    aoqv = nrc.AcquisitionOrderQuadraticVariation(num_breaks=8)
    sevs = [0.0, 1.0, 2.0, 3.0]
    vals = [aoqv(torch.zeros(1, 1, h, w), context=make_ctx(j)) for j in sevs]
    assert _spearman(sevs, vals) > 0.9


@pytest.mark.unit
def test_aoqv_fallback_uses_column_index_order():
    """Without acq_order, AOQV visits sampled columns in index order and is finite."""
    torch.manual_seed(0)
    img, smaps = _blob(), _smooth_smaps()
    y = sense_forward(img, smaps=smaps)
    ctx = MetricContext(y_kspace=y, mask=torch.ones(32, 32), coil_maps=smaps)
    out = get_metric("aoqv")(img, context=ctx)
    assert math.isfinite(out) and out >= 0.0


# --------------------------------------------------------------------------- #
# nse_hall — Null-Space-Energy Hallucination (A-8.2)                           #
# --------------------------------------------------------------------------- #
def _cartesian_mask(h: int = 32, w: int = 32, acquired: int = 12) -> torch.Tensor:
    """Central-band Cartesian PE mask [1,1,H,W] (rows = phase-encode lines)."""
    mask = torch.zeros(1, 1, h, w)
    lo = h // 2 - acquired // 2
    mask[..., lo : lo + acquired, :] = 1.0
    return mask


def _complex_img(seed: int, h: int = 32, w: int = 32) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.complex(
        torch.randn(1, 1, h, w, generator=g), torch.randn(1, 1, h, w, generator=g)
    )


def _band_error(mask: torch.Tensor, seed: int, *, in_null: bool) -> torch.Tensor:
    """A complex image error confined to the null (unmeasured) or range (measured) k-space."""
    e = _complex_img(seed)
    sel = (1.0 - mask) if in_null else mask
    return ifft2c(sel * fft2c(e))


def test_nse_hall_pure_null_space_error_scores_near_one():
    # An error living ENTIRELY in the unmeasured lines is pure hallucination -> ~1.
    mask = _cartesian_mask()
    tgt = _complex_img(1)
    pred = tgt + _band_error(mask, 2, in_null=True)
    out = get_metric("nse_hall")(pred, tgt, context=MetricContext(mask=mask))
    assert out > 0.98


def test_nse_hall_pure_range_error_scores_near_zero():
    # An error confined to the MEASURED lines is data-inconsistency, not hallucination -> ~0.
    mask = _cartesian_mask()
    tgt = _complex_img(1)
    pred = tgt + _band_error(mask, 3, in_null=False)
    out = get_metric("nse_hall")(pred, tgt, context=MetricContext(mask=mask))
    assert out < 0.02


def test_nse_hall_discriminates_null_vs_range_at_fixed_mse():
    # Discrimination (NOT statistical orthogonality): two recons with EQUAL total
    # error energy (equal MSE/PSNR) but different null/range split score very
    # differently — a global MSE cannot tell them apart, this metric can. (The
    # mathematical orthogonality of the decomposition is checked separately by
    # test_nse_hall_partitions_total_error_with_range.)
    mask = _cartesian_mask()
    tgt = _complex_img(1)
    e_null = _band_error(mask, 4, in_null=True)
    e_range = _band_error(mask, 4, in_null=False)
    # match total error energy
    en = (e_null.real**2 + e_null.imag**2).sum().sqrt()
    er = (e_range.real**2 + e_range.imag**2).sum().sqrt()
    e_range = e_range * (en / er)
    mse_null = (e_null.real**2 + e_null.imag**2).sum().item()
    mse_range = (e_range.real**2 + e_range.imag**2).sum().item()
    assert abs(mse_null - mse_range) / mse_null < 1e-4  # equal MSE precondition
    m = get_metric("nse_hall")
    out_null = m(tgt + e_null, tgt, context=MetricContext(mask=mask))
    out_range = m(tgt + e_range, tgt, context=MetricContext(mask=mask))
    assert out_null - out_range > 0.9  # same MSE, opposite hallucination verdict


def test_nse_hall_monotonic_in_null_fraction():
    # Module-convention monotonicity sweep: hold total error energy fixed, sweep the
    # null/range split 0->1, and assert nse_hall rises monotonically. This is the
    # core property (more null-space content -> more hallucination) that the
    # boundary tests (null->1, range->0) only check at the extremes.
    mask = _cartesian_mask()
    tgt = _complex_img(1)
    e_null = _band_error(mask, 4, in_null=True)
    e_range = _band_error(mask, 5, in_null=False)
    # Unit-normalise each component so the mix energy is controlled by alpha alone.
    e_null = e_null / (e_null.real**2 + e_null.imag**2).sum().sqrt()
    e_range = e_range / (e_range.real**2 + e_range.imag**2).sum().sqrt()
    m = get_metric("nse_hall")
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    vals: list[float] = []
    for a in alphas:
        e = a * e_null + (1.0 - a) * e_range  # orthogonal components -> no cross term
        vals.append(m(tgt + e, tgt, context=MetricContext(mask=mask)))
    assert _spearman(alphas, vals) > 0.9
    assert vals[0] < 0.02 and vals[-1] > 0.98  # endpoints match the boundary tests


def test_nse_hall_complements_ndcr():
    # The orthogonal-complement claim, demonstrated with the ACTUAL two metrics and
    # exposing the gaming scenario: a hallucinating recon (null-space error) reads
    # HIGH nse_hall + LOW ndcr; a data-inconsistent recon (range error) reads LOW
    # nse_hall + HIGH ndcr. => nse_hall alone is gameable; it MUST be paired with ndcr.
    mask = _cartesian_mask()
    tgt = _blob()  # real-valued image so sense_forward + ndcr are well-defined
    smaps = _smooth_smaps()
    y = sense_forward(tgt, smaps=smaps, mask=mask)
    nse, ndcr = get_metric("nse_hall"), get_metric("ndcr")

    pred_hallucinate = tgt + _band_error(mask, 8, in_null=True).real
    pred_inconsistent = tgt + _band_error(mask, 9, in_null=False).real

    nse_h = nse(pred_hallucinate, tgt, context=MetricContext(mask=mask))
    ndcr_h = ndcr(pred_hallucinate, context=MetricContext(y_kspace=y, mask=mask, coil_maps=smaps))
    nse_i = nse(pred_inconsistent, tgt, context=MetricContext(mask=mask))
    ndcr_i = ndcr(pred_inconsistent, context=MetricContext(y_kspace=y, mask=mask, coil_maps=smaps))

    assert nse_h > nse_i  # hallucination shows up in nse_hall, not data-consistency
    assert ndcr_i > ndcr_h  # data inconsistency shows up in ndcr, not nse_hall


def test_nse_hall_partitions_total_error_with_range():
    # Correctness of the orthogonal decomposition: null-fraction + range-fraction == 1
    # (Parseval, exact because fft2c is unitary and M is a 0/1 mask).
    mask = _cartesian_mask()
    tgt = _complex_img(1)
    err = _complex_img(7)
    pred = tgt + err
    null_frac = get_metric("nse_hall")(pred, tgt, context=MetricContext(mask=mask))
    k_err = fft2c(err)
    total = (k_err.real**2 + k_err.imag**2).sum()
    range_e = (mask * k_err).abs().pow(2).sum()
    range_frac = float((range_e / total).item())
    assert abs(null_frac + range_frac - 1.0) < 1e-4


def test_nse_hall_requires_mask_and_reference():
    mask = _cartesian_mask()
    tgt = _complex_img(1)
    pred = tgt + _band_error(mask, 2, in_null=True)
    m = get_metric("nse_hall")
    assert math.isnan(m(pred, None, context=MetricContext(mask=mask)))  # no reference
    assert math.isnan(m(pred, tgt, context=MetricContext()))  # no mask


def test_nse_hall_perfect_recon_returns_nan():
    # Pitfall #20: zero error has no decomposition -> undefined, not a spurious 0/0.
    mask = _cartesian_mask()
    tgt = _complex_img(1)
    out = get_metric("nse_hall")(tgt.clone(), tgt, context=MetricContext(mask=mask))
    assert math.isnan(out)


def test_nse_hall_registered_lower_is_better():
    reg = MetricsRegistry()
    assert "nse_hall" in reg.list_available()
    assert get_metric("null_space_energy_hallucination").higher_is_better is False


# --------------------------------------------------------------------------- #
# eta_null — Invented Null Energy (cold-diffusion trust functional T2)         #
# --------------------------------------------------------------------------- #
def _band_image(mask: torch.Tensor, seed: int, *, in_null: bool) -> torch.Tensor:
    """A complex image whose k-space lives entirely in the null or range band."""
    x = _complex_img(seed)
    sel = (1.0 - mask) if in_null else mask
    return ifft2c(sel * fft2c(x))


def test_eta_null_zero_for_range_only_content():
    # A reconstruction whose k-space sits entirely on ACQUIRED lines invented
    # nothing -> eta_null ~ 0.
    mask = _cartesian_mask()
    pred = _band_image(mask, 2, in_null=False)
    out = get_metric("eta_null")(pred, context=MetricContext(mask=mask))
    assert out < 1e-4


def test_eta_null_one_for_pure_null_content():
    # A reconstruction made ENTIRELY of unacquired-line content is pure
    # invention -> eta_null ~ 1.
    mask = _cartesian_mask()
    pred = _band_image(mask, 3, in_null=True)
    out = get_metric("eta_null")(pred, context=MetricContext(mask=mask))
    assert out > 0.999


def test_eta_null_scale_invariant():
    # A relative fraction: multiplying the reconstruction by 10 must not move it.
    mask = _cartesian_mask()
    pred = _complex_img(4)
    m = get_metric("eta_null")
    a = m(pred, context=MetricContext(mask=mask))
    b = m(10.0 * pred, context=MetricContext(mask=mask))
    assert abs(a - b) < 1e-5


def test_eta_null_monotonic_in_null_fraction():
    # Module-convention monotonicity sweep: mixing more null-band content into a
    # fixed-energy image raises eta_null monotonically.
    mask = _cartesian_mask()
    x_range = _band_image(mask, 5, in_null=False)
    x_null = _band_image(mask, 6, in_null=True)
    x_range = x_range / (x_range.real**2 + x_range.imag**2).sum().sqrt()
    x_null = x_null / (x_null.real**2 + x_null.imag**2).sum().sqrt()
    m = get_metric("eta_null")
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    vals = [
        m(a * x_null + (1.0 - a) * x_range, context=MetricContext(mask=mask))
        for a in alphas
    ]
    assert _spearman(alphas, vals) > 0.9
    assert vals[0] < 1e-3 and vals[-1] > 0.999


def test_eta_null_requires_mask():
    # Fully-sampled (mask absent) makes eta_null identically 0 for every input —
    # a measurement-independent constant, so it must be NaN, not 0 (pitfall #20).
    out = get_metric("eta_null")(_complex_img(1), context=MetricContext())
    assert math.isnan(out)


def test_eta_null_no_reference_contract():
    assert MetricsRegistry.is_registered("eta_null")
    assert MetricsRegistry.requires_reference("eta_null") is False
    assert MetricsRegistry.needs("eta_null") == ("mask",)
    assert get_metric("eta_null").higher_is_better is False
    # Paper-facing aliases resolve to the same metric.
    assert get_metric("invented_null_energy").name == "eta_null"


def test_kappa_alias_resolves_to_ndcr():
    # The cold-diffusion trust functional T1 (kappa_T) IS the NDCR metric —
    # aliased, not duplicated.
    assert get_metric("kappa_T").name == "ndcr"
    assert get_metric("t1_inconsistency").name == "ndcr"


# --------------------------------------------------------------------------- #
# fabrication_excess — null-band error energy (cold-diffusion H_T)             #
# --------------------------------------------------------------------------- #
def test_fabrication_excess_zero_at_perfect_recon():
    mask = _cartesian_mask()
    tgt = _complex_img(1)
    out = get_metric("fabrication_excess")(
        tgt.clone(), tgt, context=MetricContext(mask=mask)
    )
    assert out < 1e-6


def test_fabrication_excess_insensitive_to_range_error():
    # An error confined to ACQUIRED lines is data-inconsistency, not fabrication.
    mask = _cartesian_mask()
    tgt = _complex_img(1)
    pred = tgt + _band_error(mask, 3, in_null=False)
    out = get_metric("fabrication_excess")(pred, tgt, context=MetricContext(mask=mask))
    assert out < 1e-6


def test_fabrication_excess_monotonic_in_null_error():
    # More unacquired-line error -> more fabrication, linearly in the amplitude.
    mask = _cartesian_mask()
    tgt = _complex_img(1)
    e_null = _band_error(mask, 4, in_null=True)
    m = get_metric("fabrication_excess")
    amps = [0.1, 0.5, 1.0, 2.0]
    vals = [m(tgt + a * e_null, tgt, context=MetricContext(mask=mask)) for a in amps]
    assert _spearman(amps, vals) > 0.99
    # Linearity in amplitude (fixed normaliser ||F x||): ratios track amplitudes.
    assert abs(vals[3] / vals[1] - 4.0) < 1e-3


def test_fabrication_excess_distinct_from_nse_hall():
    # nse_hall is the null FRACTION of the error; H_T is its MAGNITUDE relative
    # to the truth. A hard-DC recon with a tiny null-band error has nse_hall ~ 1
    # yet tiny fabrication_excess.
    mask = _cartesian_mask()
    tgt = _complex_img(1)
    pred = tgt + 1e-3 * _band_error(mask, 5, in_null=True)
    nse = get_metric("nse_hall")(pred, tgt, context=MetricContext(mask=mask))
    h_t = get_metric("fabrication_excess")(pred, tgt, context=MetricContext(mask=mask))
    assert nse > 0.98  # all error in the null band...
    assert h_t < 1e-2  # ...but almost nothing was actually fabricated


def test_fabrication_excess_requires_mask_and_reference():
    mask = _cartesian_mask()
    tgt = _complex_img(1)
    pred = tgt + _band_error(mask, 2, in_null=True)
    m = get_metric("fabrication_excess")
    assert math.isnan(m(pred, None, context=MetricContext(mask=mask)))
    assert math.isnan(m(pred, tgt, context=MetricContext()))


def test_fabrication_excess_full_reference_contract():
    assert MetricsRegistry.is_registered("fabrication_excess")
    assert MetricsRegistry.requires_reference("fabrication_excess") is True
    assert MetricsRegistry.needs("fabrication_excess") == ("mask",)
    assert get_metric("fabrication_excess").higher_is_better is False
    assert get_metric("H_T").name == "fabrication_excess"


# ---------------------------------------------------------------------------
# null_band_energy_deficit
# ---------------------------------------------------------------------------
def _partially_filled(mask: torch.Tensor, truth: torch.Tensor, fill: float) -> torch.Tensor:
    """Acquired lines exactly the truth; the null band at ``fill`` x its energy.

    This is the shape a hard-DC reconstruction takes: data consistency pins the
    measured lines, so every deviation lives in the band the model had to invent.
    ``fill=0`` is zero-filling, ``fill=1`` is the truth.
    """
    k = fft2c(truth)
    return ifft2c(k * mask + fill * k * (1.0 - mask))


def test_null_band_energy_deficit_anchors():
    # The two ends the metric is defined by: zero-filled leaves the whole
    # unacquired band empty (1.0); the truth fills it exactly (0.0).
    mask = _cartesian_mask()
    tgt = _complex_img(1)
    m = get_metric("null_band_energy_deficit")
    ctx = MetricContext(mask=mask)
    assert m(_partially_filled(mask, tgt, 0.0), tgt, context=ctx) == pytest.approx(1.0, abs=1e-5)
    assert m(tgt, tgt, context=ctx) == pytest.approx(0.0, abs=1e-5)


def test_null_band_energy_deficit_monotone_in_the_fill_fraction():
    mask = _cartesian_mask()
    tgt = _complex_img(1)
    m = get_metric("null_band_energy_deficit")
    fills = [0.0, 0.25, 0.5, 0.75, 1.0]
    vals = [m(_partially_filled(mask, tgt, f), tgt, context=MetricContext(mask=mask)) for f in fills]
    assert _spearman(fills, vals) < -0.99
    # Analytic anchor: the deficit IS 1 - fill while the null band is scaled.
    for f, v in zip(fills, vals, strict=True):
        assert v == pytest.approx(1.0 - f, abs=1e-5)


def test_null_band_energy_deficit_clamps_the_fabrication_direction():
    # Over-filling is a real failure, but it is fabrication_excess's to report
    # (non-negotiable 17). Registering the raw two-sided ratio under a
    # lower-is-better direction would have it read backwards by every consumer
    # that resolves through ``metric_higher_is_better``, so this one clamps.
    mask = _cartesian_mask()
    tgt = _complex_img(1)
    over = _partially_filled(mask, tgt, 1.7)
    assert get_metric("null_band_energy_deficit")(
        over, tgt, context=MetricContext(mask=mask)
    ) == pytest.approx(0.0, abs=1e-6)
    # ...and the owner of that direction does move.
    assert get_metric("fabrication_excess")(over, tgt, context=MetricContext(mask=mask)) > 0.1


def test_null_band_energy_deficit_sees_what_the_rest_of_the_battery_cannot():
    """PLANTED: the experiment_11_attention_none failure shape, R=32.

    A hard-DC reconstruction whose unacquired band is uniformly attenuated. Every
    other trust functional reads healthy or uninformative on it -- which is how a
    reconstruction carrying ~30% of the target's energy passed every gate on a
    70000-iteration run (validation_metrics.csv, 2026-09-16).
    """
    mask = _cartesian_mask()
    tgt = _complex_img(1)
    pred = _partially_filled(mask, tgt, 0.3)
    ctx = MetricContext(mask=mask, y_kspace=fft2c(tgt) * mask)

    # ndcr: data consistency holds exactly, so the residual is at the floor.
    assert get_metric("ndcr")(pred, context=ctx) < 1e-5
    # nse_hall: ~1 whatever the null band's SIZE -- it reports where the error
    # is, not how much was left out, so it is constant across the whole sweep.
    assert get_metric("nse_hall")(pred, tgt, context=ctx) > 0.99
    # eta_null moves, but downward -- it reads like LESS invented content, which
    # is the opposite of an alarm.
    assert get_metric("eta_null")(pred, context=ctx) < get_metric("eta_null")(tgt, context=ctx)
    # Only the deficit names the failure, and names its size.
    assert get_metric("null_band_energy_deficit")(pred, tgt, context=ctx) == pytest.approx(
        0.7, abs=1e-5
    )


def test_null_band_energy_deficit_requires_mask_and_reference():
    mask = _cartesian_mask()
    tgt = _complex_img(1)
    pred = _partially_filled(mask, tgt, 0.3)
    m = get_metric("null_band_energy_deficit")
    assert math.isnan(m(pred, None, context=MetricContext(mask=mask)))
    assert math.isnan(m(pred, tgt, context=MetricContext()))


def test_null_band_energy_deficit_is_nan_when_there_is_nothing_to_fill():
    # A fully-sampled mask has an empty null band, so the ratio is 0/0. Reporting
    # 0.0 there would be a measurement-independent constant dressed as a perfect
    # score (pitfall #9) -- and 1.0 would be a false alarm.
    m = get_metric("null_band_energy_deficit")
    tgt = _complex_img(1)
    full = torch.ones(1, 1, 32, 32)
    assert math.isnan(m(tgt, tgt, context=MetricContext(mask=full)))

    # The shape the guard is actually FOR, and the dangerous one: a null band
    # that is not exactly empty, only negligible. 0/0 is NaN by IEEE and needs no
    # guard, but dividing by ~1e-11 does not raise -- it makes the ratio enormous,
    # so ``1 - ratio`` clamps to 0.0 and the metric reports a PERFECTLY FILLED
    # band for a target that had nothing there to fill. Deleting the guard turns
    # this red; deleting it and testing only the fully-sampled case above does not.
    #
    # The 1e-5 scale is load-bearing: a bandlimited target's null band is not
    # zero but the float32 round-trip floor (~1e-6 relative), so at unit scale it
    # sits ABOVE eps and the guard correctly stays out of the way. Scaling the
    # whole target down moves the absolute null energy under eps while leaving
    # the in-band signal intact.
    mask = _cartesian_mask()
    k = fft2c(_complex_img(1))
    bandlimited = 1e-5 * ifft2c(k * mask)
    pred = bandlimited + _band_error(mask, 7, in_null=True)
    assert math.isnan(m(pred, bandlimited, context=MetricContext(mask=mask)))


def test_null_band_energy_deficit_contract():
    assert MetricsRegistry.is_registered("null_band_energy_deficit")
    assert MetricsRegistry.requires_reference("null_band_energy_deficit") is True
    assert MetricsRegistry.needs("null_band_energy_deficit") == ("mask",)
    assert get_metric("null_band_energy_deficit").higher_is_better is False
    assert get_metric("null_fill_deficit").name == "null_band_energy_deficit"
