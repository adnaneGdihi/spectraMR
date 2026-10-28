"""B1 as B2's initialiser: contrast tokens, mask tokens, factorised attention."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.contrast_token_encoder import ContrastTokenEncoder


def _enc() -> ContrastTokenEncoder:
    torch.manual_seed(0)
    return ContrastTokenEncoder(dim=16, depth=2, heads=2, window=4).eval()


def _x() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.rand(2, 3, 32, 32)


def test_the_output_is_one_state_map_per_tissue_parameter_at_full_resolution() -> None:
    with torch.no_grad():
        z = _enc()(_x(), torch.ones(2, 3))
    assert z.shape == (2, 3, 32, 32) and torch.isfinite(z).all()


def test_a_masked_contrast_is_replaced_by_the_mask_token_and_never_read() -> None:
    enc, x = _enc(), _x()
    avail = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    moved = x.clone()
    moved[0, 1] = 50.0 * torch.rand_like(moved[0, 1])
    moved[1, 0] = float("inf")  # even a non-finite hidden image must not leak in
    with torch.no_grad():
        assert torch.equal(enc(x, avail), enc(moved, avail))


def test_an_available_contrast_is_read() -> None:
    enc, x = _enc(), _x()
    moved = x.clone()
    moved[:, 1] += 1.0
    with torch.no_grad():
        assert not torch.allclose(enc(x, torch.ones(2, 3)), enc(moved, torch.ones(2, 3)))


def test_the_mask_token_and_contrast_embeddings_are_trained() -> None:
    enc, x = _enc(), _x()
    enc.train()
    enc(x, torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])).sum().backward()
    assert enc.mask_token.grad is not None and enc.mask_token.grad.abs().sum() > 0
    assert enc.contrast_embedding.grad is not None


def test_a_size_not_divisible_by_the_token_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _enc()(torch.rand(1, 3, 30, 30), torch.ones(1, 3))
