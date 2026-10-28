"""Tests for the TALLSuperResolution generator (IMPLEMENTATION_SPEC §9.5).

Pins the contract the Phase-3 task-2.3 scaffold signs on:

* Registers under ``tall_sr`` with mode ``reconstruction``.
* Forward + backward succeed on a power-of-two square grid, in both
  train (soft Gumbel-softmax) and eval (hard argmax) regimes.
* Raises (no silent fallback) on a non-square / non-power-of-two grid.
* The TALL traversal genuinely *fires*: the SR loss back-propagates
  into the orientation policy (guards against the inert-facade pitfall
  #16), and the output is input-dependent (guards against the
  degenerate measurement-independent output pitfall #20).
* The returned traversal is a valid permutation and the locality
  regulariser (Spec §9.3) is exposed and finite.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from spectramr.models import generators  # noqa: F401, E402  triggers @register_model
from spectramr.models.generators.tall_sr import TALLSuperResolution  # noqa: E402
from spectramr.models.registry import MODEL_REGISTRY  # noqa: E402
from tests.utils.optional_backends import requires_cuda_for_mamba  # noqa: E402

# order=4 -> 16x16 grid keeps the soft-permutation tensors tiny for CPU CI.
_ORDER = 4
_SIDE = 2 ** _ORDER


def _build(in_ch: int = 1, out_ch: int = 1) -> TALLSuperResolution:
    return TALLSuperResolution(
        in_channels=in_ch,
        out_channels=out_ch,
        order=_ORDER,
        dim=8,
        n_layers=2,
        d_state=8,
    )


def test_model_is_registered_for_reconstruction_mode() -> None:
    assert "tall_sr" in MODEL_REGISTRY
    assert MODEL_REGISTRY["tall_sr"]["mode"] == "reconstruction"
    assert MODEL_REGISTRY["tall_sr"]["class"] is TALLSuperResolution


@requires_cuda_for_mamba
def test_forward_shape_train_and_eval() -> None:
    m = _build()
    x = torch.randn(2, 1, _SIDE, _SIDE)
    m.train()
    y_train = m(x)
    assert y_train.shape == (2, 1, _SIDE, _SIDE)
    assert torch.isfinite(y_train).all()
    m.eval()
    with torch.no_grad():
        y_eval = m(x)
    assert y_eval.shape == (2, 1, _SIDE, _SIDE)


@requires_cuda_for_mamba
def test_non_residual_channel_change_shape() -> None:
    # out_channels != in_channels disables the residual add but still
    # produces the requested channel count.
    m = _build(in_ch=1, out_ch=3)
    y = m(torch.randn(2, 1, _SIDE, _SIDE))
    assert y.shape == (2, 3, _SIDE, _SIDE)


@pytest.mark.parametrize(
    "bad_shape",
    [
        (2, 1, _SIDE, _SIDE + 1),  # non-square
        (2, 1, _SIDE - 1, _SIDE - 1),  # non-power-of-two square
    ],
)
def test_forward_raises_on_incompatible_grid(bad_shape: tuple[int, ...]) -> None:
    m = _build()
    with pytest.raises(ValueError, match="grid"):
        m(torch.randn(*bad_shape))


def test_forward_raises_on_wrong_rank() -> None:
    m = _build()
    with pytest.raises(ValueError, match="4-D"):
        m(torch.randn(2, 1, _SIDE))  # missing a spatial axis


@requires_cuda_for_mamba
def test_traversal_is_valid_permutation() -> None:
    m = _build()
    m.eval()
    with torch.no_grad():
        m(torch.randn(2, 1, _SIDE, _SIDE))
    perm = m._last_permutation
    assert perm is not None
    n = _SIDE * _SIDE
    expected = torch.arange(n).unsqueeze(0).expand(perm.shape[0], -1)
    assert torch.equal(perm.sort(dim=1).values, expected)


@requires_cuda_for_mamba
def test_policy_receives_gradient_mechanism_fires() -> None:
    # Pitfall #16: the traversal must be exercised, not an inert facade.
    # The SR task loss must reach TALL's orientation-policy parameters.
    m = _build()
    m.train()
    y = m(torch.randn(2, 1, _SIDE, _SIDE))
    (y - torch.randn_like(y)).abs().mean().backward()
    policy_grad = m.tall.policy[-1].weight.grad
    assert policy_grad is not None
    assert policy_grad.abs().sum() > 0


@requires_cuda_for_mamba
def test_output_is_input_dependent() -> None:
    # Pitfall #20: reject a degenerate, measurement-independent output.
    m = _build()
    m.eval()
    with torch.no_grad():
        a = m(torch.zeros(1, 1, _SIDE, _SIDE))
        b = m(torch.ones(1, 1, _SIDE, _SIDE))
    assert not torch.allclose(a, b)


@requires_cuda_for_mamba
def test_locality_regulariser_finite_and_guarded() -> None:
    m = _build()
    with pytest.raises(RuntimeError, match="before forward"):
        m.last_locality_loss()
    m.track_locality = True
    m.eval()
    with torch.no_grad():
        m(torch.randn(1, 1, _SIDE, _SIDE))
    loc = m.last_locality_loss()
    assert loc.ndim == 0
    assert torch.isfinite(loc)
    assert loc >= 0
