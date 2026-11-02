"""Simulation, envelope and the metric table."""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.meta_evaluation.blackwell.catalogue import build_catalogue
from spectramr.core.metrics.meta_evaluation.blackwell.evaluate import (
    blur_envelope,
    displays,
    metric_table,
    oracle_lsi_control,
)
from spectramr.core.metrics.meta_evaluation.blackwell.order import certify
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

pytestmark = pytest.mark.unit
SHAPE = (32, 32)


def _content(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    img = torch.zeros(SHAPE)
    img[8:24, 8:24] = 1.0
    return img + 0.3 * torch.rand(SHAPE, generator=g)


def _neg_mse(deg: torch.Tensor, clean: torch.Tensor) -> float:
    return -float((deg - clean).pow(2).mean())


def _cat() -> dict:
    return {e.name: e for e in build_catalogue(SHAPE)}


def test_noise_free_identity_display_reproduces_the_content() -> None:
    x = _content(0)
    exp = _cat()["noise_0.05"]
    noise = torch.zeros((1, *SHAPE), dtype=torch.complex64)
    out = displays(fft2c(x), exp, noise, 1.0, blur_envelope(SHAPE, (0.0,), None))
    assert out.shape == (1, 1, *SHAPE)
    assert torch.allclose(out[0, 0], x, atol=1e-5)


def test_unacquired_coefficients_are_zero_filled() -> None:
    x = _content(0)
    exp = _cat()["truncation_0.15"]
    noise = torch.zeros((1, *SHAPE), dtype=torch.complex64)
    out = displays(fft2c(x), exp, noise, 1.0, blur_envelope(SHAPE, (0.0,), None))
    acquired = torch.isfinite(exp.log_info)
    expect = ifft2c(fft2c(x) * acquired).abs()
    assert torch.allclose(out[0, 0], expect, atol=1e-5)


def test_metric_table_shapes_orientation_and_envelope() -> None:
    contents = [_content(0), _content(1)]
    cat = build_catalogue(SHAPE)
    raw, sharp = metric_table(
        {"mse": lambda d, c: float((d - c).pow(2).mean())},
        {"mse": -1.0},
        contents,
        cat,
        n_draws=3,
        envelope_widths=(0.0, 1.0),
        seed=0,
    )
    assert raw["mse"].shape == (2, 25, 3)
    assert sharp["mse"].shape == (2, 25, 1)  # odd draws, scored after choosing on even
    assert (raw["mse"] <= 0).all()  # oriented: higher is better
    assert torch.isfinite(sharp["mse"]).all()


def test_single_width_sharp_is_raw_and_choosing_needs_two_draws() -> None:
    cat = build_catalogue(SHAPE)[:2]
    raw, sharp = metric_table(
        {"q": _neg_mse}, {"q": 1.0}, [_content(0)], cat, n_draws=1, envelope_widths=(0.0,), seed=0
    )
    assert torch.equal(raw["q"], sharp["q"])
    with pytest.raises(ValueError, match="n_draws >= 2"):
        metric_table(
            {"q": _neg_mse},
            {"q": 1.0},
            [_content(0)],
            cat,
            n_draws=1,
            envelope_widths=(0.0, 1.0),
            seed=0,
        )


def test_neg_mse_respects_every_strict_relation_on_average() -> None:
    contents = [_content(s) for s in range(3)]
    cat = build_catalogue(SHAPE)
    _, sharp = metric_table(
        {"q": _neg_mse},
        {"q": 1.0},
        contents,
        cat,
        n_draws=4,
        envelope_widths=(0.0, 0.5, 1.0, 2.0),
        seed=0,
    )
    m = sharp["q"].mean(dim=(0, 2))
    for i, j in certify(cat).strict:
        assert m[i] >= m[j] - 1e-9


def test_metric_nan_is_confined_to_its_cell() -> None:
    cat = build_catalogue(SHAPE)[:2]

    def flaky(d: torch.Tensor, c: torch.Tensor) -> float:
        return float("nan") if float(d.mean()) > 10 else 1.0

    raw, _ = metric_table(
        {"q": flaky},
        {"q": 1.0},
        [_content(0), _content(1) * 100],
        cat,
        n_draws=1,
        envelope_widths=(0.0,),
        seed=0,
    )
    assert torch.isfinite(raw["q"][0]).all() and torch.isnan(raw["q"][1]).all()


def test_metric_table_rejects_bad_inputs() -> None:
    cat = build_catalogue(SHAPE)[:1]
    with pytest.raises(ValueError, match=r"start with 0\.0"):
        metric_table(
            {"q": _neg_mse},
            {"q": 1.0},
            [_content(0)],
            cat,
            n_draws=1,
            envelope_widths=(1.0,),
            seed=0,
        )
    with pytest.raises(ValueError, match="one"):
        metric_table(
            {"q": _neg_mse},
            {"q": 1.0},
            [_content(0), torch.zeros(16, 16)],
            cat,
            n_draws=1,
            envelope_widths=(0.0,),
            seed=0,
        )


def test_oracle_control_is_isotone_in_the_information_order() -> None:
    contents = [_content(0), _content(1)]
    cat = build_catalogue(SHAPE)
    v = oracle_lsi_control(contents, cat)
    for i, j in certify(cat).strict:
        assert (v[:, i] >= v[:, j] - 1e-9).all()


def test_wiener_envelope_has_identity_first_and_undoes_a_post_noise_filter() -> None:
    from spectramr.core.metrics.meta_evaluation.blackwell.evaluate import wiener_envelope

    x = _content(0)
    exp = _cat()["noise_0.1"]
    filters = wiener_envelope((1.0,))(fft2c(x), exp, 1.0)
    assert filters.shape == (2, *SHAPE) and torch.equal(filters[0], torch.ones(SHAPE))
    assert float(filters[1].max()) <= 1.0 + 1e-6  # Wiener shrinks noise-only data
    with pytest.raises(ValueError, match="positive"):
        wiener_envelope((0.0,))
