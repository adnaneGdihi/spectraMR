"""Tests for the validation-time disentanglement readings of the disentangled strategy.

A tiny generator is factorised by construction (content = the zero-mean structure, style =
the mean intensity), so the swap distances must be 0; two planted breakages -- a decoder
that drops the content, a decoder that drops the style -- must each be read as a failure.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spectramr.core.metrics.disentanglement import DisentanglementAccumulator
from spectramr.infrastructure.training.strategies.disentangled_diagnostics import (
    MAX_SWAP_PAIRS,
    expand_row_names,
    record_validation_rows,
    swap_distances,
)


class _Generator(nn.Module):
    """content = zero-mean structure; style = mean intensity; decode adds them back."""

    use_vae = False

    def __init__(self, drop: str | None = None) -> None:
        super().__init__()
        self.drop = drop
        self.anchor = nn.Parameter(torch.zeros(1))  # gives the module a train/eval flag to check

    def enc_c(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=(1, 2, 3), keepdim=True)

    def enc_s(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(1, 2, 3)).view(-1, 1)

    def condition_style(self, style: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return style

    def decode_tissue(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        content = torch.zeros_like(content) if self.drop == "content" else content
        style = torch.zeros_like(style) if self.drop == "style" else style
        return content + style.view(-1, 1, 1, 1)


def _filled(model: _Generator, per_label: int = 2) -> DisentanglementAccumulator:
    """Rows of three contrasts, each a random structure on a contrast-specific brightness."""
    gen = torch.Generator().manual_seed(0)
    acc = DisentanglementAccumulator(per_label_swap_rows=per_label)
    for label, level in enumerate((0.2, 0.5, 0.9)):
        images = level + 0.05 * torch.randn(4, 1, 8, 8, generator=gen)
        record_validation_rows(
            acc,
            model.enc_c(images),
            model.enc_s(images),
            torch.full((4,), label),
            lambda: {"contrast_type": torch.zeros(4, 1, 1, 1, dtype=torch.long)},
        )
    return acc


def _render(raw: object, code: torch.Tensor) -> torch.Tensor:  # pragma: no cover - not reached
    raise AssertionError("the tiny generator decodes to an image, so nothing is rendered")


def test_a_factorised_generator_swaps_with_zero_distance() -> None:
    out = swap_distances(_Generator(), _render, _filled(_Generator()))
    assert out["val_swap_content_distance"] == pytest.approx(0.0, abs=1e-5)
    assert out["val_swap_style_distance"] == pytest.approx(0.0, abs=1e-5)


def test_without_style_the_swap_reports_the_content_distance_only() -> None:
    out = swap_distances(_Generator(), _render, _filled(_Generator()), include_style=False)
    assert set(out) == {"val_swap_content_distance"}


def test_a_decoder_that_drops_the_content_is_read_as_a_content_failure() -> None:
    broken = _Generator(drop="content")
    out = swap_distances(broken, _render, _filled(broken))
    assert out["val_swap_content_distance"] > 0.9


def test_a_decoder_that_drops_the_style_is_read_as_a_style_failure() -> None:
    broken = _Generator(drop="style")
    out = swap_distances(broken, _render, _filled(broken))
    assert out["val_swap_style_distance"] > 0.9


def test_the_model_is_left_in_the_mode_it_was_found_in() -> None:
    model = _Generator()
    acc = _filled(model)
    model.train()
    swap_distances(model, _render, acc)
    assert model.training
    model.eval()
    swap_distances(model, _render, acc)
    assert not model.training


def test_the_mode_is_restored_when_a_decode_raises() -> None:
    class _Raises(_Generator):
        def decode_tissue(self, content, style):
            raise RuntimeError("decoder failed")

    model = _Raises()
    acc = _filled(model)
    model.train()
    with pytest.raises(RuntimeError, match="decoder failed"):
        swap_distances(model, _render, acc)
    assert model.training


def test_tissue_dicts_are_rendered_with_the_style_sources_sequence_code() -> None:
    seen: list[int] = []

    class _Tissue(_Generator):
        def decode_tissue(self, content, style):
            return {"rho": content + style.view(-1, 1, 1, 1)}

    def render(raw: dict, row: dict) -> torch.Tensor:
        # the style source's family and acquisition both reach the render, one row each
        assert row["acquisition"]["TR"].shape == (1, 1, 1, 1)
        assert float(row["acquisition"]["TR"]) == 100.0 * int(row["contrast_type"])
        seen.append(int(row["contrast_type"]))
        return raw["rho"]

    model = _Tissue()
    acc = DisentanglementAccumulator(per_label_swap_rows=1)
    for label in (0, 1):
        images = torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(label)) + label
        record_validation_rows(
            acc,
            model.enc_c(images),
            model.enc_s(images),
            torch.tensor([label]),
            lambda label=label: {
                "contrast_type": torch.tensor([[[[10 + label]]]]),
                "acquisition": {"TR": torch.tensor([[[[100.0 * (10 + label)]]]])},
            },
        )
    swap_distances(model, render, acc)
    assert sorted(seen) == [10, 11]  # each pair renders with its style source's code


def test_image_decoders_keep_no_sequence_code_and_tissue_decoders_refuse_to_lack_one() -> None:
    """No map means no code to resolve (an image decoder renders nothing); a decoder that
    emits tissue maps with no code kept must fail, not render with a made-up family."""

    class _Tissue(_Generator):
        def decode_tissue(self, content, style):
            return {"rho": content + style.view(-1, 1, 1, 1)}

    for model_cls, raises in ((_Generator, False), (_Tissue, True)):
        model = model_cls()
        acc = DisentanglementAccumulator(per_label_swap_rows=1)
        for label in (0, 1):
            images = torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(label)) + label
            record_validation_rows(
                acc, model.enc_c(images), model.enc_s(images), torch.tensor([label]), None
            )
        assert acc._kept[0][0].batch is None
        if raises:
            with pytest.raises(ValueError, match="bloch_sequence_by_contrast"):
                swap_distances(model, _render, acc)
        else:
            assert swap_distances(model, _render, acc)["val_swap_content_distance"] < 1e-5


def test_no_pair_across_labels_reports_nothing() -> None:
    model = _Generator()
    acc = DisentanglementAccumulator()
    record_validation_rows(
        acc,
        torch.zeros(2, 1, 4, 4),
        torch.zeros(2, 1),
        torch.zeros(2, dtype=torch.long),
        lambda: {"contrast_type": torch.zeros(2, 1, 1, 1)},
    )
    assert swap_distances(model, _render, acc) == {}


def test_the_swap_decode_is_capped() -> None:
    model = _Generator()
    acc = _filled(model)
    assert (
        len(acc.swap_pairs(MAX_SWAP_PAIRS)) == MAX_SWAP_PAIRS
    )  # 3 labels x 2 rows -> 24 ordered pairs


class TestExpandRowNames:
    def test_each_sample_name_covers_its_slices_in_order(self) -> None:
        assert expand_row_names(["S0", "S1"], 6) == ["S0", "S0", "S0", "S1", "S1", "S1"]

    def test_a_single_string_is_one_sample(self) -> None:
        assert expand_row_names("S0", 3) == ["S0"] * 3

    def test_absent_names_stay_absent(self) -> None:
        assert expand_row_names(None, 4) is None

    def test_names_that_cannot_cover_the_rows_raise(self) -> None:
        with pytest.raises(ValueError, match="cannot cover 5 rows"):
            expand_row_names(["S0", "S1"], 5)
