"""Registry masks and filters as Blackwell experiments: probe, placement, provenance."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.core.metrics.meta_evaluation.blackwell.order import Relation, certify, compare
from spectramr.infrastructure.physics.degradations import diagonal_catalogue as dc

pytestmark = pytest.mark.unit
SHAPE = (32, 32)


def _contents() -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    out = []
    for _ in range(2):
        img = torch.zeros(SHAPE)
        img[8:24, 6:26] = 1.0
        out.append(img + 0.2 * torch.rand(SHAPE, generator=g))
    return out


def _named(codes: tuple[str, ...], **kw):
    exps, prov = dc.registry_catalogue(_contents(), codes=codes, **kw)
    return {e.name: e for e in exps}, prov


@pytest.mark.parametrize("code", ["cartesian_undersamp", "t2star_blur", "recon_apodization"])
def test_admitted_codes_probe_as_real_diagonal_filters(code: str) -> None:
    h, res = dc.probe_transfer(code, 0.67, 3, SHAPE)
    assert h.shape == SHAPE and h.dtype == torch.float64 and (h >= 0).all()
    assert max(res.values()) <= dc.PROBE_TOL


@pytest.mark.parametrize(
    ("code", "reason"),
    [("translation_drift", "imaginary"), ("adc_clipping", "linearity"), ("rician", "real image")],
)
def test_probe_rejects_phase_nonlinear_and_real_output_operators(code: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        dc.probe_transfer(code, 0.67, 3, SHAPE)


def test_mask_zeros_are_exact_and_a_rebuild_is_equivalent() -> None:
    a, _ = _named(("cartesian_undersamp",), thetas=(0.67,))
    b, _ = _named(("cartesian_undersamp",), thetas=(0.67,))
    exp = a["cartesian_undersamp_0.67"]
    assert torch.isneginf(exp.log_info).any() and torch.isfinite(exp.log_info).any()
    assert compare(exp.log_info, b["cartesian_undersamp_0.67"].log_info) is Relation.EQUIVALENT


def test_two_severities_of_one_filter_are_strictly_ordered() -> None:
    exps, _ = dc.registry_catalogue(_contents(), codes=("t2star_blur",), thetas=(0.33, 1.0))
    names = [e.name for e in exps]
    strict = set(certify(exps).strict)
    assert (names.index("t2star_blur_0.33"), names.index("t2star_blur_1")) in strict


def test_after_noise_filter_is_equivalent_to_noise_alone() -> None:
    exps, _ = dc.registry_catalogue(_contents(), codes=("recon_apodization",), thetas=(0.67,))
    names = [e.name for e in exps]
    pair = tuple(sorted((names.index("noise_0.1"), names.index("recon_apodization_0.67"))))
    assert pair in set(certify(exps).equivalent)


def test_misplacing_apodisation_before_the_noise_changes_the_relation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planted: the placement table decides the relation, so a wrong entry is visible."""
    monkeypatch.setitem(dc.PLACEMENT, "recon_apodization", "before_noise")
    exps, _ = dc.registry_catalogue(_contents(), codes=("recon_apodization",), thetas=(0.67,))
    names = [e.name for e in exps]
    pair = (names.index("noise_0.1"), names.index("recon_apodization_0.67"))
    assert pair in set(certify(exps).strict)


def test_gain_above_one_is_normalised_and_shown_at_unit_dc() -> None:
    named, prov = _named(("tse_t2_blur",), thetas=(1.0,))
    exp = named["tse_t2_blur_1"]
    h, g = exp.transfers()
    centre = (SHAPE[0] // 2, SHAPE[1] // 2)
    assert float(h[centre]) == pytest.approx(1.0)  # brightness kept
    # information is referenced to the peak, so it never exceeds noise alone at sigma
    sigma = exp.noise_std
    assert float(exp.log_info.max()) == pytest.approx(-2 * math.log(sigma), abs=1e-3)
    assert torch.allclose(torch.log(h**2 / (sigma * g) ** 2), exp.log_info, atol=1e-6)
    entry = prov["entries"][0]
    assert entry["max_gain"] > 2.0 and entry["placement"] == "before_noise"
    assert "unit DC gain" in prov["gain_convention"]


def test_content_check_raises_on_a_wrong_transfer() -> None:
    h, _ = dc.probe_transfer("t2star_blur", 0.67, 3, SHAPE)
    with pytest.raises(ValueError, match="misses a content"):
        dc.check_on_contents("t2star_blur", 0.67, 3, torch.ones_like(h), _contents())


def test_unknown_codes_raise() -> None:
    with pytest.raises(ValueError, match="not audited"):
        dc.registry_catalogue(_contents(), codes=("translation_drift",))
