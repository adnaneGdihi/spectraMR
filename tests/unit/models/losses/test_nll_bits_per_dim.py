"""Unit tests for the ``nll_bits_per_dim`` loss (Phase 3 flows objective)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.losses.nll_bits_per_dim import NLLBitsPerDim  # noqa: E402
from spectramr.models.losses.registry import LossRegistry  # noqa: E402


def test_registered_under_canonical_and_aliases() -> None:
    import spectramr.models.losses  # noqa: F401  triggers registration

    assert "nll_bits_per_dim" in LossRegistry._custom_losses
    for alias in ("bits_per_dim", "bpd"):
        assert alias in LossRegistry._aliases


def test_bpd_from_log_prob_tensor() -> None:
    loss = NLLBitsPerDim(reduction="mean")
    d = 16
    log_prob = torch.full((4,), -d * math.log(2.0))  # exactly 1 bit/dim
    out = loss(log_prob, num_dims=d)
    assert torch.isclose(out, torch.tensor(1.0), atol=1e-5)


def test_bare_log_prob_requires_num_dims() -> None:
    loss = NLLBitsPerDim()
    with pytest.raises(ValueError):
        loss(torch.zeros(4))


def test_bpd_from_z_logdet_tuple_infers_dims() -> None:
    loss = NLLBitsPerDim(reduction="none")
    z = torch.zeros(2, 4)  # d=4, log N(0;0,I) = -0.5*d*log(2pi)
    log_det = torch.zeros(2)
    out = loss((z, log_det))
    expected = (0.5 * 4 * math.log(2 * math.pi)) / (4 * math.log(2.0))
    assert torch.allclose(out, torch.full((2,), expected), atol=1e-5)


def test_dict_input_paths() -> None:
    loss = NLLBitsPerDim()
    z = torch.randn(3, 8)
    log_det = torch.randn(3)
    via_dict = loss({"z": z, "log_det": log_det})
    via_tuple = loss((z, log_det))
    assert torch.isclose(via_dict, via_tuple, atol=1e-6)


def test_invalid_reduction_raises() -> None:
    with pytest.raises(ValueError):
        NLLBitsPerDim(reduction="median")


@pytest.mark.parametrize("bad", [torch.full((4,), 16.0), 16.0, 0, True])
def test_num_dims_must_be_a_positive_int(bad: object) -> None:
    """A ``loss(pred, target)`` call puts the target where ``num_dims`` goes; it
    must raise rather than broadcast into a per-element divisor."""
    with pytest.raises(TypeError, match="num_dims"):
        NLLBitsPerDim()(torch.full((4,), -10.0), num_dims=bad)  # type: ignore[arg-type]
