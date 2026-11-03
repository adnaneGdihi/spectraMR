import math

import torch

from spectramr.core.metrics.physics_nr_metrics import (
    AsymptoticGFactor,
    NEMACNR,
    NEMASNR,
    GFactor,
)
from spectramr.infrastructure.physics.coil_sensitivity import create_synthetic_csm


def test_nema_snr():
    """Test NEMASNR metric."""
    metric = NEMASNR()
    assert metric.name == "nema_snr"
    assert metric.higher_is_better is True

    # Create synthetic image: noise + bright center
    img = torch.randn(1, 1, 64, 64) * 0.1  # background noise
    img[0, 0, 20:44, 20:44] += 1.0  # signal ROI

    snr = metric(img)
    assert snr > 5.0  # High SNR

    # Too noisy
    noisy_img = torch.randn(1, 1, 64, 64) * 1.0
    noisy_img[0, 0, 20:44, 20:44] += 1.0
    noisy_snr = metric(noisy_img)
    assert noisy_snr < snr  # Worse SNR

    # 0 std edge case
    flat_img = torch.ones(1, 1, 64, 64)
    flat_snr = metric(flat_img)
    assert flat_snr == 0.0


def test_nema_cnr():
    """Test NEMACNR metric."""
    metric = NEMACNR()
    assert metric.name == "nema_cnr"
    assert metric.higher_is_better is True

    # High CNR: left half of center is bright, right half is dark
    img = torch.randn(1, 1, 64, 64) * 0.1  # background noise
    img[0, 0, 20:44, 20:32] += 2.0  # left side bright
    img[0, 0, 20:44, 32:44] += 0.0  # right side dark

    cnr = metric(img)
    assert cnr > 10.0  # High CNR

    # Low CNR: uniform center
    uniform_img = torch.randn(1, 1, 64, 64) * 0.1
    uniform_img[0, 0, 20:44, 20:44] += 1.0
    uniform_cnr = metric(uniform_img)
    assert uniform_cnr < cnr  # Worse CNR

    # 0 std edge case
    flat_img = torch.ones(1, 1, 64, 64)
    flat_cnr = metric(flat_img)
    assert flat_cnr == 0.0


def test_g_factor():
    """Test GFactor metric."""
    metric = GFactor(acceleration=2)
    assert metric.name == "g_factor"
    assert metric.higher_is_better is False

    img = torch.randn(1, 1, 32, 32)

    # Perfect sensitivity maps (no noise amplification)
    smaps = torch.ones(4, 32, 32, dtype=torch.complex64)
    g_perfect = metric(img, sensitivity_maps=smaps)
    # With uniform maps, S^H S is rank deficient, so the inverse acts on the regularizer
    # With our specific regularizer math and clamp, it stays reasonable, but let's just test it runs
    assert isinstance(g_perfect, float)

    # Missing sensitivity maps: the g-factor is UNDEFINED without coil maps.
    # Regression for the review fix (2026-07-01): the old code returned the
    # optimistic 1.0 ("no noise amplification" — the best score), silently
    # certifying a perfect parallel-imaging condition on an unmeasured run.
    # NaN is the module-wide "could not measure" sentinel; 1.0 here was a facade.
    g_missing = metric(img)
    assert math.isnan(g_missing), "missing smaps must be NaN, not the optimistic 1.0"


def test_g_factor_increases_with_acceleration():
    """A proper SENSE g-factor must rise with the acceleration R: more aliased
    pixels fold per readout, so the encoding matrix is more ill-conditioned.
    R=1 means no parallel imaging → g == 1.0."""
    metric = GFactor()
    img = torch.zeros(1, 1, 48, 48)
    smaps = create_synthetic_csm((1, 1, 48, 48), num_coils=8)[0]  # (8, 48, 48)

    g_r1 = metric(img, sensitivity_maps=smaps, acceleration=1)
    g_r2 = metric(img, sensitivity_maps=smaps, acceleration=2)
    g_r4 = metric(img, sensitivity_maps=smaps, acceleration=4)

    assert g_r1 == 1.0, "R=1 (no acceleration) → g-factor 1.0"
    assert g_r4 > g_r2 > 1.0, f"g must grow with R; got R2={g_r2}, R4={g_r4}"


def test_asymptotic_gfactor_accepts_call_time_c():
    """AsymptoticGFactor must accept a call-time aspect ratio c so the engine can
    drive it with c = R / n_coils; larger c → larger worst-case g (1 + sqrt(c))."""
    metric = AsymptoticGFactor(c=0.5)
    img = torch.zeros(1, 1, 16, 16)

    g_default = metric(img)
    g_small = metric(img, c=0.2)
    g_large = metric(img, c=0.7)

    assert g_default > 1.0
    assert g_large > g_small > 1.0, f"g must grow with c; got {g_small}, {g_large}"


def test_g_factor_registered_as_no_reference():
    """g_factor is a pure no-reference geometry metric: its ``__call__`` ignores
    ``target`` and measures the coil/sampling geometry alone. Regression for the
    flag drift (2026-07-24 NR audit) — it was registered with the implicit
    ``requires_reference=True`` default, disagreeing with its NO_REFERENCE type
    in the sim2rank ``METRIC_SPECS`` sweep. The correction is number-neutral
    because the reference target is ignored."""
    from spectramr.core.metrics.registry import MetricsRegistry

    assert MetricsRegistry.requires_reference("g_factor") is False

    metric = GFactor()
    img = torch.zeros(1, 1, 48, 48)
    smaps = create_synthetic_csm((1, 1, 48, 48), num_coils=8)[0]
    ref = torch.randn(1, 1, 48, 48)
    g_no_target = metric(img, sensitivity_maps=smaps, acceleration=2)
    g_with_target = metric(img, ref, sensitivity_maps=smaps, acceleration=2)
    assert g_no_target == g_with_target, "g_factor must ignore the reference target"


def test_g_factor_reads_maps_and_rate_from_the_validation_context():
    """The validation loop threads ``context=``; its maps and R must be used."""
    from spectramr.core.metrics.context import MetricContext

    metric = GFactor()  # default acceleration=2 must not override the context's R
    img = torch.zeros(1, 1, 48, 48)
    smaps = create_synthetic_csm((1, 1, 48, 48), num_coils=8)[0]
    via_ctx = metric(img, context=MetricContext(coil_maps=smaps, acceleration=4.0))
    assert via_ctx == metric(img, sensitivity_maps=smaps, acceleration=4)
    assert via_ctx > metric(img, sensitivity_maps=smaps, acceleration=2) > 1.0
    # A context that carries maps but no rate is not scored at the constructor default.
    assert math.isnan(metric(img, context=MetricContext(coil_maps=smaps)))


def test_g_factor_is_nan_when_r_exceeds_the_coil_count():
    """R > coils leaves SENSE underdetermined: unmeasurable, never the best score 1.0."""
    smaps = create_synthetic_csm((1, 1, 48, 48), num_coils=4)[0]
    assert math.isnan(GFactor()(torch.zeros(1, 1, 48, 48), sensitivity_maps=smaps, acceleration=6))
