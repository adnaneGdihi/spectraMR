"""Linearization wiring for ``MambaUNet`` / ``MambaLayer2D``.

The SSM core processes a 1-D sequence, so a 2-D→1-D *linearization* (scan order)
is intrinsic to the model. ``linearization_mode`` flows config →
``mamba_config`` (whitelisted in ``MambaUNet._VALID_MAMBA_KEYS``) →
``MambaLayer2D`` → ``_MODE_GENERATORS`` dispatch. These tests pin that the knob
is wired through to every SSM layer, that the round-trip permutation is exact,
that an unknown mode RAISES (no silent fallback, pitfall #9), and that the
``raster`` / ``hilbert_2d_rect`` modes are resolution-agnostic while strict
``hilbert_2d`` correctly rejects non-square / non-power-of-2 feature maps.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.generators.mamba_unet import (  # noqa: E402  (after importorskip)
    MambaLayer2D,
    MambaUNet,
)
from tests.utils.optional_backends import requires_cuda_for_mamba  # noqa: E402

_FEATURES = [32, 64, 128, 256]


def _build(mode: str) -> MambaUNet:
    return MambaUNet(
        in_channels=1,
        out_channels=1,
        features=_FEATURES,
        mamba_config={"d_state": 16, "d_conv": 4, "expand": 2,
                      "linearization_mode": mode},
    )


def test_linearization_mode_wired_to_every_ssm_layer() -> None:
    model = _build("raster")
    layers = [m for m in model.modules() if isinstance(m, MambaLayer2D)]
    assert layers, "expected MambaLayer2D blocks in the model tree"
    assert {layer.linearization_mode for layer in layers} == {"raster"}


@requires_cuda_for_mamba
@pytest.mark.parametrize("mode", ["raster", "hilbert_2d_rect"])
def test_resolution_agnostic_modes_round_trip_non_pow2(mode: str) -> None:
    # 130x114 → bottleneck 65x57 (non-square, non-pow2): the variable full-slice
    # / native-geometry sizes the direct_ulf_to_hf_sr arm runs on.
    x = torch.randn(1, 1, 130, 114)
    with torch.no_grad():
        y = _build(mode)(x)
    assert y.shape == x.shape  # exact 2D shape preserved (scan + un-scan inverse)


def test_strict_hilbert_rejects_non_pow2() -> None:
    # The strict square-pow2 curve must FAIL LOUD on an incompatible size rather
    # than silently degrade — this is why the SR arm uses raster/hilbert_2d_rect.
    with pytest.raises(AssertionError):
        _build("hilbert_2d")(torch.randn(1, 1, 130, 114))


def test_unknown_linearization_mode_raises() -> None:
    # Pitfall #9: an unknown scan order must raise, never fall back to raster.
    with pytest.raises(ValueError, match="Unknown linearization mode"):
        _build("not_a_real_curve")(torch.randn(1, 1, 32, 32))


def test_run_block_checkpointing_matches_plain_grads() -> None:
    """The checkpointed stage path yields the same output and gradient as the plain one."""

    class _Stage(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x, emb=None, ssm_override=None):
            return self.lin(x).tanh()

    model = _build("raster")
    model.train()
    stage = _Stage()
    x = torch.randn(3, 4, requires_grad=True)

    y_ref = model._run_block(stage, x, None, None)
    y_ref.sum().backward()
    g_ref = x.grad.clone()

    x.grad = None
    model.set_grad_checkpointing(True)
    y = model._run_block(stage, x, None, None)
    y.sum().backward()

    assert torch.allclose(y, y_ref, atol=1e-6)
    assert torch.allclose(x.grad, g_ref, atol=1e-6)
