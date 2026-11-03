"""Multi-coil Fisher operators, relations and the real Lanczos witness."""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.meta_evaluation.blackwell import multicoil as mc
from spectramr.core.metrics.meta_evaluation.blackwell.coil_catalogue import coil_catalogue
from spectramr.core.metrics.meta_evaluation.blackwell.order import Relation, compare
from spectramr.infrastructure.physics.fft_ops import fft2c, sense_adjoint, sense_forward

pytestmark = pytest.mark.unit
SHAPE = (24, 24)


def _contents() -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return [torch.rand(SHAPE, generator=g) + 0.2 for _ in range(2)]


def _named() -> tuple[dict[str, mc.CoilExperiment], torch.Tensor]:
    exps, smaps, _ = coil_catalogue(_contents())
    return {e.name: e for e in exps}, smaps


def test_sense_operators_are_adjoint_and_maps_normalised() -> None:
    smaps = mc.coil_maps(SHAPE, 4)
    assert torch.allclose((smaps.abs() ** 2).sum(1), torch.ones(SHAPE), atol=1e-5)
    g = torch.Generator().manual_seed(1)
    x = torch.randn((1, 1, *SHAPE), generator=g)
    y = torch.randn((1, 4, *SHAPE), dtype=torch.complex64, generator=g)
    lhs = (sense_forward(x, smaps).conj() * y).sum()
    rhs = (x.to(torch.complex64).conj() * sense_adjoint(y, smaps)).sum()
    assert float((lhs - rhs).abs() / lhs.abs()) < 1e-5


def test_quadratic_form_matches_the_operator_and_full_mask_is_single_coil_noise() -> None:
    named, smaps = _named()
    v = torch.randn(SHAPE, generator=torch.Generator().manual_seed(2))
    exp = named["equispaced_R3"]
    direct = float((v.to(torch.complex64) * mc._apply(v, exp, smaps).conj()).sum().real)
    assert mc.quadratic(v, exp, smaps) == pytest.approx(direct, rel=1e-4)
    full = named["noise_0.1"]  # normalised coils: J = I / sigma^2
    assert mc.quadratic(v, full, smaps) == pytest.approx(float(v.pow(2).sum()) / 0.01, rel=1e-4)


def test_nested_masks_are_ordered_structurally_without_lanczos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named, smaps = _named()

    def _no_witness(*_a: object) -> tuple[bool, bool]:
        raise AssertionError("structural pairs must not reach the witness")

    monkeypatch.setattr(mc, "_signs", _no_witness)
    exps = [named["vd_0.6"], named["vd_0.45"], named["vd_0.3"], named["noise_0.05"]]
    graph = mc.certify_coils(exps, smaps)
    assert set(graph.strict) == {(0, 1), (0, 2), (1, 2), (3, 0), (3, 1), (3, 2)}


def test_uniform_single_coil_matches_compare_on_symmetric_masks() -> None:
    """Planted cross-check: with S = 1, the witness agrees with the verified diagonal order."""
    named, _ = _named()
    one = torch.ones((1, 1, *SHAPE), dtype=torch.complex64)
    a, b = named["equispaced_R2"], named["noise_0.15"]
    assert compare(a.log_weights(), b.log_weights()) is Relation.INCOMPARABLE
    assert mc.certify_coils([a, b], one).incomparable == ((0, 1),)


def test_partial_fourier_against_noise_is_uncertified_over_real_images() -> None:
    """An asymmetric mask never leaves both k and -k unsampled, so no real witness exists."""
    named, smaps = _named()
    graph = mc.certify_coils([named["pf_0.75"], named["noise_0.15"]], smaps)
    assert graph.uncertified == ((0, 1),) and graph.incomparable == ()


def test_identical_experiments_give_no_witness() -> None:
    named, smaps = _named()
    assert mc._signs(named["equispaced_R2"], named["equispaced_R2"], smaps) == (False, False)


def test_noise_free_fully_sampled_display_reproduces_the_content() -> None:
    named, smaps = _named()
    x = _contents()[0]
    noise = torch.zeros((1, 4, *SHAPE), dtype=torch.complex64)
    out = mc.CoilRealiser(smaps)(fft2c(x), named["noise_0.1"], noise, 1.0, torch.ones(1, *SHAPE))
    assert out.shape == (1, 1, *SHAPE)
    assert torch.allclose(out[0, 0], x, atol=1e-4)


def test_catalogue_masks_keep_calibration_and_provenance() -> None:
    exps, smaps, prov = coil_catalogue(_contents(), n_coils=6)
    assert smaps.shape == (1, 6, *SHAPE) and prov["n_coils"] == 6
    centre = SHAPE[0] // 2
    assert all(float(e.mask[centre].min()) == 1.0 for e in exps)
    assert prov["witness"]["space"] == "real"
    with pytest.raises(ValueError, match="2-D"):
        coil_catalogue([torch.zeros(1, *SHAPE)])


def test_random_mask_weights_lines_once_in_the_expected_operator() -> None:
    _, smaps = _named()
    v = torch.randn(SHAPE, generator=torch.Generator().manual_seed(3))
    half = mc.CoilExperiment(
        "half", "bernoulli", 2, torch.full(SHAPE, 0.5, dtype=torch.float64), 0.1, True
    )
    assert mc.quadratic(v, half, smaps) == pytest.approx(
        0.5 * float(v.pow(2).sum()) / 0.01, rel=1e-4
    )


def test_random_masks_are_ordered_by_coupling_without_lanczos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named, smaps = _named()

    def _no_witness(*_a: object) -> tuple[bool, bool]:
        raise AssertionError("coupled pairs must not reach the witness")

    monkeypatch.setattr(mc, "_signs", _no_witness)
    exps = [named["random_vd_0.6"], named["random_vd_0.3"], named["noise_0.05"]]
    assert set(mc.certify_coils(exps, smaps).strict) == {(0, 1), (2, 0), (2, 1)}


def test_equispaced_and_bernoulli_at_one_density_are_incomparable() -> None:
    """Theorem C1.4 dependent design: equal expected density, no coupling either way."""
    named, smaps = _named()
    graph = mc.certify_coils([named["equispaced_R2"], named["bernoulli_R2"]], smaps)
    assert graph.incomparable == ((0, 1),)


def test_shared_uniform_draws_acquire_lines_at_their_probability() -> None:
    _, smaps = _named()
    realiser = mc.CoilRealiser(smaps, n_draws=4000, seed=0)
    for p in (0.25, 0.5):
        assert float((realiser.uniform < p).double().mean()) == pytest.approx(p, abs=0.01)


def test_certain_random_mask_displays_like_full_sampling() -> None:
    _, smaps = _named()
    x = _contents()[0]
    full = mc.CoilExperiment(
        "p1", "bernoulli", 1, torch.ones(SHAPE, dtype=torch.float64), 0.1, True
    )
    noise = torch.zeros((2, 4, *SHAPE), dtype=torch.complex64)
    out = mc.CoilRealiser(smaps, n_draws=2)(fft2c(x), full, noise, 1.0, torch.ones(1, *SHAPE))
    assert out.shape == (1, 2, *SHAPE) and torch.allclose(out[0, 1], x, atol=1e-4)
