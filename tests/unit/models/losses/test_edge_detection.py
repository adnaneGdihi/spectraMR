"""Tests for edge-detection losses.

Targets ``spectramr.models.losses.edge_detection``:

- ``SobelOperator`` — 3×3 Sobel gradient module (single + multi-channel)
- ``SobelLoss`` — MSE between Sobel gradients
- ``GradientLoss`` — L1 between x/y gradients (lighter alternative)

- ``EdgePreservationLoss`` — weighted sum of Sobel/Laplacian/gradient terms;
  must return a graph-connected tensor (regression for a former ``.item()``
  gradient-killer).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.edge_detection import (
    GradientLoss,
    SobelLoss,
    SobelOperator,
)


# ---------------------------------------------------------------------------
# SobelOperator — kernel construction
# ---------------------------------------------------------------------------


def test_sobel_operator_kernels_registered() -> None:
    """Sobel kernels are registered as buffers with shape ``(2, 1, 3, 3)``."""
    op = SobelOperator()
    assert op.kernels.shape == (2, 1, 3, 3)


def test_sobel_operator_kernels_are_correct_values() -> None:
    """The kernels match the textbook Sobel definition."""
    op = SobelOperator()
    # X-direction kernel: derivative along W
    expected_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    )
    assert torch.equal(op.kernels[0, 0], expected_x)


def test_sobel_operator_single_channel_output_shape() -> None:
    """Single-channel input ``(N, 1, H, W)`` → ``(N, 1, 2, H, W)``."""
    op = SobelOperator()
    x = torch.zeros(2, 1, 8, 8)
    out = op(x)
    assert out.shape == (2, 1, 2, 8, 8)


def test_sobel_operator_multi_channel_output_shape() -> None:
    """Multi-channel input ``(N, C, H, W)`` → ``(N, C, 2, H, W)``."""
    op = SobelOperator()
    x = torch.zeros(1, 3, 8, 8)
    out = op(x)
    assert out.shape == (1, 3, 2, 8, 8)


def test_sobel_operator_constant_input_zero_gradient() -> None:
    """Constant images have zero Sobel gradient."""
    op = SobelOperator()
    x = torch.full((1, 1, 8, 8), 5.0)
    out = op(x)
    # Interior gradient is zero; boundary may have small non-zero from padding.
    interior = out[..., 1:-1, 1:-1]
    assert torch.allclose(interior, torch.zeros_like(interior), atol=1e-5)


def test_sobel_operator_horizontal_edge_detected() -> None:
    """A vertical edge produces a non-zero x-gradient."""
    op = SobelOperator()
    x = torch.zeros(1, 1, 8, 8)
    x[..., :, 4:] = 1.0  # vertical edge at column 4
    out = op(x)
    grad_x = out[0, 0, 0]  # x-direction gradient
    # Gradient at column 4 should be non-zero in the interior.
    assert grad_x[3:5, 3:5].abs().sum().item() > 0


# ---------------------------------------------------------------------------
# SobelLoss
# ---------------------------------------------------------------------------


def test_sobel_loss_zero_for_identical_inputs() -> None:
    """``loss(x, x) == 0``."""
    loss_fn = SobelLoss()
    x = torch.randn(1, 1, 8, 8)
    out = loss_fn(x, x)
    assert out.item() == 0.0


def test_sobel_loss_grows_with_difference() -> None:
    """Different gradient patterns → strictly positive loss.

    NB: uniform images with different magnitudes still differ under
    Sobel + zero-padding because the *boundary* picks up a phantom
    edge from the implicit zero halo. The ``loss(x, x) == 0`` case
    is covered in :func:`test_sobel_loss_zero_for_identical_inputs`;
    here we focus on the contrast between matched and mismatched
    gradient patterns.
    """
    loss_fn = SobelLoss()
    pred = torch.zeros(1, 1, 8, 8)
    target = torch.zeros(1, 1, 8, 8)
    target[..., :, 4:] = 1.0  # vertical edge in target only
    out = loss_fn(pred, target)
    assert out.item() > 0

    # Sanity: a target with the same edge pattern as the prediction
    # should give a strictly smaller loss than a target with a *different*
    # edge pattern.
    pred_edge = torch.zeros(1, 1, 8, 8)
    pred_edge[..., :, 4:] = 1.0  # same edge as target
    matched = loss_fn(pred_edge, target).item()
    assert matched == 0.0 < out.item()


def test_sobel_loss_reduction_none_returns_per_sample() -> None:
    """``reduction='none'`` returns a per-sample tensor of shape ``(N,)``."""
    loss_fn = SobelLoss(reduction="none")
    pred = torch.randn(2, 1, 8, 8)
    target = torch.randn(2, 1, 8, 8)
    out = loss_fn(pred, target)
    assert out.shape == (2,)


def test_sobel_loss_gradient_flow() -> None:
    """Backward through SobelLoss reaches the prediction tensor."""
    loss_fn = SobelLoss()
    pred = torch.randn(1, 1, 8, 8, requires_grad=True)
    target = torch.zeros(1, 1, 8, 8)
    target[..., :, 4:] = 1.0
    out = loss_fn(pred, target)
    out.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# GradientLoss
# ---------------------------------------------------------------------------


def test_gradient_loss_zero_for_identical() -> None:
    """``loss(x, x) == 0``."""
    loss = GradientLoss()
    x = torch.randn(1, 1, 8, 8)
    assert loss(x, x).item() == 0.0


def test_gradient_loss_multi_channel_uses_grouped_conv() -> None:
    """Multi-channel inputs are processed independently per channel."""
    loss = GradientLoss()
    pred = torch.randn(1, 3, 8, 8)
    target = pred.clone()
    out = loss(pred, target)
    assert out.item() == 0.0


def test_gradient_loss_picks_up_difference() -> None:
    """Different gradient patterns → positive loss."""
    loss = GradientLoss()
    pred = torch.zeros(1, 1, 8, 8)
    target = torch.zeros(1, 1, 8, 8)
    target[..., 4:, :] = 1.0  # horizontal edge
    out = loss(pred, target)
    assert out.item() > 0


# ---------------------------------------------------------------------------
# Sanity-shape matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(1, 1, 8, 8), (2, 1, 16, 16), (1, 3, 32, 32)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_sobel_loss_shape_matrix(shape: tuple[int, ...]) -> None:
    """SobelLoss handles single + multi-channel inputs."""
    loss_fn = SobelLoss()
    pred = torch.randn(*shape)
    out = loss_fn(pred, pred.clone())
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_edge_losses_registered_under_documented_names() -> None:
    """``sobel_edge``, ``edge_consistency``, ``explicit_gradient`` are registered."""
    from spectramr.models.losses.registry import list_available

    available = set(list_available())
    expected = {"sobel_edge", "edge_consistency", "explicit_gradient"}
    assert expected <= available


# ---------------------------------------------------------------------------
# EdgePreservationLoss — gradient flow (regression: previously returned a float)
# ---------------------------------------------------------------------------


def test_edge_preservation_returns_grad_carrying_tensor() -> None:
    """The loss must be a graph-connected tensor, not a Python float.

    Pre-fix the body did ``total_loss = 0.0`` then ``+= (...).item()``, so the
    return value was a gradient-free float and the loss contributed zero
    gradient (pitfall #16 at the autograd layer).
    """
    from spectramr.models.losses.edge_detection import EdgePreservationLoss

    pred = torch.rand(2, 1, 16, 16, requires_grad=True)
    target = torch.rand(2, 1, 16, 16)

    loss = EdgePreservationLoss()(pred, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad and loss.grad_fn is not None
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0


def test_edge_preservation_gradient_is_finite_on_a_flat_background() -> None:
    """A zero background has zero Sobel magnitude, where ``sqrt'(0)`` is infinite."""
    from spectramr.models.losses.edge_detection import EdgePreservationLoss

    target = torch.zeros(2, 1, 16, 16)
    target[..., 4:12, 4:12] = torch.rand(2, 1, 8, 8, generator=torch.Generator().manual_seed(0))
    pred = target.clone().requires_grad_(True)

    EdgePreservationLoss()(pred, target).backward()

    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_edge_preservation_laplacian_matches_per_channel_reference() -> None:
    """The grouped Laplacian equals one 2-D convolution per channel."""
    import torch.nn.functional as F

    from spectramr.models.losses.edge_detection import EdgePreservationLoss

    x = torch.randn(2, 3, 12, 12, generator=torch.Generator().manual_seed(1))
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    reference = torch.cat([F.conv2d(x[:, c : c + 1], kernel, padding=1) for c in range(3)], dim=1)

    torch.testing.assert_close(EdgePreservationLoss()._laplacian_edge_detection(x), reference)


def test_edge_preservation_rejects_an_unknown_weight_key() -> None:
    """A misspelt term must raise, not be dropped while the others still run."""
    from spectramr.models.losses.edge_detection import EdgePreservationLoss

    with pytest.raises(ValueError, match="gradient_magnitude"):
        EdgePreservationLoss(weights={"sobel": 1.0, "laplacian": 0.5, "grad_mag": 0.3})


def test_edge_losses_accept_float64_inputs() -> None:
    """The float32 kernels follow the input dtype instead of raising in conv2d."""
    from spectramr.models.losses.edge_detection import EdgePreservationLoss

    x = torch.rand(1, 2, 8, 8, dtype=torch.float64)
    y = torch.rand(1, 2, 8, 8, dtype=torch.float64)
    assert torch.isfinite(EdgePreservationLoss()(x, y))
    assert torch.isfinite(GradientLoss()(x, y))
