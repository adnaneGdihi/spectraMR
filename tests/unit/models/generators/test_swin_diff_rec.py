"""Regression tests for ``SwinDiffRec`` -- #1064.

The issue reports two defects. Only one of them is still real, and the other
half of the issue is refuted here rather than implemented:

* **The size-mismatch crash is already fixed** by #1345, which resolves the
  shifted-window mask for the resolution actually in hand instead of the one
  the block was constructed with. The issue asks for a guard at the top of
  ``SwinDiffRec.forward`` rejecting input smaller than ``image_size``; adding
  it now would *forbid* behaviour #1345 made correct. What is pinned instead
  is that the resolution-independence keeps working.
* **The real-valued block path never worked**, and the issue's reason for
  calling it latent ("all arms set ``use_complex_conv: true``") does not hold:
  the knob never reaches this constructor at all -- see
  ``test_production_site_can_only_pass_true``.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest
import torch

from spectramr.models.blocks.complex_blocks import ComplexResBlock
from spectramr.models.blocks.residual import ResidualBlock
from spectramr.models.generators.swin_diff_rec import SwinDiffRec

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_GENERATOR = _REPO_ROOT / "src/spectramr/models/generators/kspace_cold_diffusion_generator.py"


def _tiny(**overrides):
    """A ~0.05M-parameter SwinDiffRec. Deliberately far below any real arm."""
    kwargs = {
        "in_channels": 2,
        "out_channels": 2,
        "image_size": 32,
        "base_channels": 8,
        "channel_mults": (1, 2),
        "num_res_blocks": 1,
        "swin_depth": 2,
        "swin_heads": 2,
        "swin_window_size": 4,
        "physics_emb_dim": 32,
    }
    kwargs.update(overrides)
    return SwinDiffRec(**kwargs)


# ---------------------------------------------------------------------------
# The real-valued path raises instead of silently mis-binding
# ---------------------------------------------------------------------------


def test_default_selects_the_complex_path():
    """The constructor default must be the path that actually works.

    Before #1064 the default was ``False``, i.e. the *default* construction
    took the mis-bound branch. Any caller omitting the knob got it.
    """
    default = inspect.signature(SwinDiffRec).parameters["use_complex_conv"].default
    assert default is True


def test_explicit_false_raises():
    with pytest.raises(ValueError) as exc:
        _tiny(use_complex_conv=False)
    assert "use_complex_conv" in str(exc.value)


def test_raise_names_the_positional_misbinding():
    """The message must explain *why*, not merely refuse.

    A bare "unsupported" would send the reader looking for a missing feature;
    the actual defect is that four arguments bind onto a different contract.
    """
    with pytest.raises(ValueError) as exc:
        _tiny(use_complex_conv=False)
    msg = str(exc.value)
    for token in ("kernel_size", "stride", "ResidualBlock", "ComplexResBlock"):
        assert token in msg, f"raise message does not name {token!r}: {msg}"


def test_raise_fires_before_any_allocation():
    """The guard must precede module construction, not follow it.

    The mis-binding's cost is paid at ``nn.Conv2d`` construction, so a guard
    placed after the encoder loop would OOM before it could refuse.
    """
    src = inspect.getsource(SwinDiffRec.__init__)
    body = src.split("super().__init__()", 1)[1]
    guard_at = body.index("if not use_complex_conv:")
    for allocation in ("nn.Sequential(", "nn.ModuleList()", "nn.Conv2d("):
        if allocation in body:
            assert guard_at < body.index(allocation), (
                f"{allocation} is constructed before the use_complex_conv guard"
            )


def test_the_two_block_contracts_are_genuinely_incompatible():
    """Justifies the raise from primary source rather than from the issue text.

    If these two signatures ever converge, the raise becomes unnecessary and
    this test says so by failing.
    """
    complex_params = list(inspect.signature(ComplexResBlock).parameters)
    real_params = list(inspect.signature(ResidualBlock).parameters)

    assert complex_params[:4] == ["in_channels", "out_channels", "emb_dim", "dropout"]
    assert real_params[:4] == ["channels", "kernel_size", "stride", "padding"]

    # The call site passes 4 positionals shaped for the complex contract, so
    # position 1 (out_channels -> kernel_size) and 2 (emb_dim -> stride) are
    # the ones that detonate.
    assert complex_params[1] != real_params[1]
    assert complex_params[2] != real_params[2]


# ---------------------------------------------------------------------------
# The issue's "latent" claim, refuted from the production call site
# ---------------------------------------------------------------------------


def test_production_site_can_only_pass_true():
    """``use_complex_conv`` reaches this constructor by injection only.

    ``KSpaceColdDiffusionGenerator.__init__`` binds ``use_complex_conv`` as an
    *explicit* parameter and never reads or forwards it, so a YAML value can
    never travel to the backbone. The only writer is the ``force_pure_kspace``
    injection, which sets ``True``. That is why the arm that declared
    ``use_complex_conv: false`` was broken while the ones declaring ``true``
    were fine -- ``force_pure_kspace`` was doing all the work.

    This pins the invariant the default-flip relies on: no production path
    hands this constructor a ``False``.
    """
    tree = ast.parse(_GENERATOR.read_text())

    writes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "use_complex_conv"
            ):
                writes.append(node.value)

    assert writes, "no writer found; the injection site moved"
    for value in writes:
        assert isinstance(value, ast.Constant) and value.value is True, (
            "a production writer now assigns a non-True use_complex_conv; "
            "the SwinDiffRec default-flip no longer covers every arm"
        )


# ---------------------------------------------------------------------------
# #1064 half 1: resolution independence, at model level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fed", [32, 16])
def test_accepts_input_smaller_than_the_constructed_image_size(fed):
    """#1345 made this work; #1064 asked for a guard that would forbid it.

    Built at ``image_size=32`` the bottleneck resolution is 16x16; fed 16x16
    it is 8x8, so the shifted-window mask must be rebuilt for the resolution
    in hand. A guard rejecting ``fed < image_size`` would turn this into a
    crash. ``swin_depth=2`` in the fixture is load-bearing -- at depth 1 every
    block is unshifted, no mask is built, and this test passes even with the
    pre-#1345 behaviour restored (verified by planting it).
    """
    model = _tiny().eval()
    x = torch.randn(1, 2, fed, fed)
    t = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        out = model(x, t)
    assert out.shape[-2:] == (fed, fed)


@pytest.mark.parametrize("scale", ["unit_interval", "integer"])
def test_forward_passes_the_timestep_scale_to_the_sinusoid(scale):
    """A dropped ``timestep_scale`` would keep the dividing form on every arm (#2315)."""
    from unittest import mock

    import spectramr.models.generators.swin_diff_rec as mod

    seen = []
    real = mod.sinusoidal_timestep_embedding

    def spy(t, dim, *, max_timesteps=None, scale="unit_interval"):
        seen.append(scale)
        return real(t, dim, max_timesteps=max_timesteps, scale=scale)

    net = _tiny().eval()
    with mock.patch.object(mod, "sinusoidal_timestep_embedding", spy), torch.no_grad():
        net(torch.randn(1, 2, 32, 32), torch.tensor([3]), max_timesteps=29.0, timestep_scale=scale)
    assert seen == [scale]


class TestContrastEmbeddingBatch:
    """Exactly one contrast row per sample; a truncated batch or a broadcast row raises."""

    def test_contrast_batch_larger_than_the_timestep_batch_raises(self) -> None:
        net = _tiny().eval()
        with pytest.raises(ValueError, match="contrast_emb batch 3"), torch.no_grad():
            net(torch.randn(2, 2, 32, 32), torch.tensor([1, 2]), contrast_emb=torch.randn(3, 32))

    @pytest.mark.parametrize(("rows", "t_rows"), [(1, 2), (2, 3)])
    def test_a_broadcast_row_or_a_larger_timestep_batch_raises(self, rows, t_rows) -> None:
        """A single row used to broadcast onto every sample (#2055)."""
        net = _tiny().eval()
        x = torch.randn(t_rows, 2, 32, 32)
        with pytest.raises(ValueError, match=f"contrast_emb batch {rows}"), torch.no_grad():
            net(x, torch.arange(1, t_rows + 1), contrast_emb=torch.randn(rows, 32))

    @pytest.mark.parametrize("rows", [2])
    def test_per_sample_rows_pass(self, rows) -> None:
        net = _tiny().eval()
        x = torch.randn(2, 2, 32, 32)
        with torch.no_grad():
            out = net(x, torch.tensor([1, 2]), contrast_emb=torch.randn(rows, 32))
        assert out.shape == x.shape
