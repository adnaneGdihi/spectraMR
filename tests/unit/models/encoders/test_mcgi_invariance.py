r"""Unit tests for the MCGI encoder (monotone-contrast-group invariance).

Targets ``spectramr.models.encoders.mcgi_encoder``.

MCGI = :math:`\Psi \circ R` (a backbone behind the rank transform), optionally
:math:`\sigma`-symmetrized so the invariance group is :math:`G_+ \rtimes
\mathbb Z_2` (monotone *and* order-reversal intensity maps -- the group that
contains both same-ordering, e.g. T1w-PD, and order-reversing, e.g. T1w-T2w,
contrast relations). The guarantees checked:

- Exact invariance to a strictly-increasing remap (eval / hard rank).
- Exact invariance to order reversal :math:`x \mapsto -x` when symmetrized.
- Order reversal is NOT free without symmetrization (the head matters).
- The model is registered for YAML dispatch.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def _encoder(**kw):
    from spectramr.models.encoders.mcgi_encoder import MCGIEncoder

    torch.manual_seed(0)
    enc = MCGIEncoder(in_channels=1, hidden_channels=8, out_channels=4, depth=2, **kw)
    return enc.eval()


def _mono(t: torch.Tensor) -> torch.Tensor:
    return torch.exp(1.5 * t) + 0.2 * t + 3.0


def test_features_invariant_to_monotone_remap() -> None:
    enc = _encoder(symmetrize_order_reversal=True)
    x = torch.rand(2, 1, 16, 16)
    f_x = enc(x)
    f_gx = enc(_mono(x))
    assert torch.allclose(f_x, f_gx, atol=1e-5), "MCGI must be G_+ invariant (eval)"


def test_symmetrized_features_invariant_to_order_reversal() -> None:
    enc = _encoder(symmetrize_order_reversal=True)
    x = torch.rand(2, 1, 16, 16)
    assert torch.allclose(enc(x), enc(-x), atol=1e-5), "sigma-pool must be Z2 invariant"


def test_order_reversal_not_invariant_without_symmetrization() -> None:
    enc = _encoder(symmetrize_order_reversal=False)
    x = torch.rand(2, 1, 16, 16)
    assert not torch.allclose(enc(x), enc(-x), atol=1e-3)


def test_forward_output_shape() -> None:
    enc = _encoder(symmetrize_order_reversal=True)
    x = torch.rand(3, 1, 12, 12)
    out = enc(x)
    assert out.shape[0] == 3 and out.shape[1] == 4


def test_model_is_registered() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert "mcgi_encoder" in MODEL_REGISTRY


def test_train_mode_backbone_receives_gradient() -> None:
    from spectramr.models.encoders.mcgi_encoder import MCGIEncoder

    enc = MCGIEncoder(in_channels=1, hidden_channels=4, out_channels=1, depth=2).train()
    enc(torch.rand(1, 1, 8, 8)).sum().backward()
    assert all(p.grad is not None for p in enc.backbone.parameters())


def test_hard_rank_eval_false_raises() -> None:
    from spectramr.models.encoders.mcgi_encoder import MCGIEncoder

    with pytest.raises(ValueError, match="2409"):
        MCGIEncoder(hard_rank_eval=False)


def test_retired_soft_rank_knob_is_not_a_parameter() -> None:
    from spectramr.models.encoders.mcgi_encoder import MCGIEncoder

    with pytest.raises(TypeError):
        MCGIEncoder(soft_rank_temperature=0.05)


def test_rank_transform_false_is_the_non_invariant_twin() -> None:
    torch.manual_seed(0)
    x = torch.rand(2, 1, 16, 16)
    enc = _encoder(rank_transform=False)
    assert isinstance(enc.rank, torch.nn.Identity)
    assert not torch.allclose(enc(x), enc(_mono(x)), atol=1e-3)


def test_registered_strict_so_unknown_model_kwargs_fail_the_build() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import is_strict_kwargs

    populate_model_registry()
    assert is_strict_kwargs("mcgi_encoder")
