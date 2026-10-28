"""Any-subset availability and leave-one-contrast-out scoring for the q-map strategy."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.strategies.qmap_availability import (
    draw_availability,
    leave_one_out,
    unroll_protocol,
    validate_contrast_dropout,
)


def test_no_dropout_keeps_every_contrast() -> None:
    assert draw_availability(4, 0.0, torch.device("cpu")) is None


def test_every_sample_keeps_at_least_one_contrast() -> None:
    torch.manual_seed(0)
    avail = draw_availability(4096, 0.95, torch.device("cpu"))
    assert avail.shape == (4096, 3)
    assert set(avail.unique().tolist()) <= {0.0, 1.0}
    assert (avail.sum(1) >= 1).all()


def test_dropout_drops_at_about_its_rate() -> None:
    torch.manual_seed(0)
    avail = draw_availability(20000, 0.3, torch.device("cpu"))
    assert 0.25 < 1 - float(avail.mean()) < 0.32  # the at-least-one rescue adds a little


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_a_dropout_outside_0_1_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="contrast_dropout"):
        validate_contrast_dropout(bad)


def test_the_header_protocol_maps_onto_the_render_arguments() -> None:
    series = [
        {
            "sequence_family": 1.0,
            "acq_tr": 7500.0,
            "acq_te": 98.0,
            "acq_ti": 1655.0,
            "acq_esp": 16.3,
            "acq_etl": 11.0,
            "acq_refocus_deg": 160.0,
        },
    ]
    assert unroll_protocol(series) == [
        {
            "family": 1,
            "tr": 7500.0,
            "te": 98.0,
            "ti": 1655.0,
            "esp": 16.3,
            "etl": 11,
            "refocus_deg": 160.0,
        }
    ]


def test_leave_one_out_holds_out_each_contrast_and_scores_only_it() -> None:
    seen = []

    def render(available):
        seen.append(available.clone())
        return [torch.full((2, 1, 4, 4), float(k)) for k in range(3)]

    def score(k, renders, available):
        assert available[0, k] == 0 and available[0].sum() == 2  # the held-out one only
        return {"val_psnr_loo": float(renders[k].mean()) + 10 * k, "val_other": 1.0}

    out = leave_one_out(render, batch_size=2, device=torch.device("cpu"), score=score)
    assert [a[0].tolist() for a in seen] == [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    assert out == {
        "val_psnr_loo_t1": 0.0,
        "val_psnr_loo_t2": 11.0,
        "val_psnr_loo_flair": 22.0,
        "val_psnr_loo": 11.0,
        "val_other_t1": 1.0,
        "val_other_t2": 1.0,
        "val_other_flair": 1.0,
        "val_other": 1.0,
    }
