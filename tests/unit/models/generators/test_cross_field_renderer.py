"""Tests for AnatomyFieldRenderer (MICCAI MRIxFields2026 backbone)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.cross_field_renderer import AnatomyFieldRenderer
from spectramr.models.registry import MODEL_REGISTRY


def test_registered() -> None:
    assert "anatomy_field_renderer" in MODEL_REGISTRY
    assert MODEL_REGISTRY["anatomy_field_renderer"]["mode"] == "cross_field_translation"


def test_encode_render_shapes() -> None:
    g = AnatomyFieldRenderer(in_channels=1, out_channels=1, latent_channels=16)
    x = torch.rand(2, 1, 16, 16)
    b = torch.tensor([3.0, 7.0])
    q = g.encode(x, torch.tensor([0.1, 1.5]))
    assert q.shape[0] == 2 and q.shape[2:] == (16, 16)
    y = g.render(q, b)
    assert y.shape == (2, 1, 16, 16)


def test_field_path_is_wired() -> None:
    # FiLM is identity-init, so before training render is field-agnostic. Perturb the
    # FiLM weights to confirm the field embedding actually drives the output (anti-
    # facade: the field argument is not inert).
    torch.manual_seed(0)
    g = AnatomyFieldRenderer(in_channels=1, out_channels=1, latent_channels=16).eval()
    q = g.encode(torch.rand(1, 1, 16, 16), torch.tensor([1.5]))
    for p in g.film.parameters():
        p.data += 0.1
    y_lo = g.render(q, torch.tensor([0.1]))
    y_hi = g.render(q, torch.tensor([7.0]))
    assert not torch.allclose(y_lo, y_hi)


def test_complex_input_raises() -> None:
    g = AnatomyFieldRenderer(in_channels=1, out_channels=1)
    with pytest.raises(ValueError):
        g.encode(torch.rand(1, 1, 8, 8, dtype=torch.cfloat), torch.tensor([1.5]))


def test_non_magnitude_channels_raise() -> None:
    with pytest.raises(ValueError):
        AnatomyFieldRenderer(in_channels=2, out_channels=2)


def test_forward_uses_field_kwarg() -> None:
    g = AnatomyFieldRenderer(in_channels=1, out_channels=1)
    y = g(
        torch.rand(2, 1, 16, 16),
        field_strength=torch.tensor([0.1, 1.5]),
        field_strength_target=torch.tensor([1.5, 3.0]),
    )
    assert y.shape == (2, 1, 16, 16)


def test_encoder_reads_the_source_field() -> None:
    # #197: on Task 1 the target is pinned at 7 T, so only the source field varies per sample.
    # The encoder FiLM must carry it (perturbed off its identity init, distinct sources differ).
    torch.manual_seed(0)
    g = AnatomyFieldRenderer(in_channels=1, out_channels=1, latent_channels=16).eval()
    for p in g.encoder_film.parameters():
        p.data += 0.1
    x = torch.rand(1, 1, 16, 16)
    assert not torch.allclose(g.encode(x, torch.tensor([0.1])), g.encode(x, torch.tensor([5.0])))


def test_forward_requires_both_fields() -> None:
    g = AnatomyFieldRenderer(in_channels=1, out_channels=1)
    with pytest.raises(TypeError):
        g(torch.rand(1, 1, 8, 8), field_strength_target=torch.tensor([7.0]))


def test_contrast_conditioning_runs() -> None:
    g = AnatomyFieldRenderer(in_channels=1, out_channels=1, latent_channels=16)
    q = g.encode(torch.rand(2, 1, 16, 16), torch.tensor([0.1, 1.5]))
    y = g.render(q, torch.tensor([3.0, 7.0]), contrast_id=torch.tensor([0, 2]))
    assert y.shape == (2, 1, 16, 16)
