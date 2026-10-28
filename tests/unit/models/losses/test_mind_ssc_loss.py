"""Unit tests for MIND-SSC Loss."""

import pytest
import torch

from spectramr.models.losses.mind_ssc_loss import MINDSSCLoss


class TestMINDSSCLoss:
    """Test MINDSSCLoss functionality."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        loss_fn = MINDSSCLoss()
        assert loss_fn.patch_size == 7
        assert loss_fn.search_radius == 3
        assert loss_fn.gaussian_sigma == 1.0
        assert loss_fn.use_self_similarity is True

    def test_forward_pass_shapes(self):
        """Test forward pass output shape/type."""
        loss_fn = MINDSSCLoss()
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randn(2, 1, 64, 64)
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar

    def test_gradient_stability_constant_input(self):
        """Test gradient stability with constant input (variance=0).

        This tests the fix for SqrtBackward0 NaN error when local variance is 0.
        """
        loss_fn = MINDSSCLoss()
        # Constant image -> local variance is 0
        pred = torch.ones(1, 1, 32, 32, requires_grad=True)
        target = torch.ones(1, 1, 32, 32)

        loss = loss_fn(pred, target)
        loss.backward()

        # Check for NaNs in gradient
        assert not torch.isnan(pred.grad).any(), "Gradient contains NaNs!"
        assert not torch.isinf(pred.grad).any(), "Gradient contains Infs!"

    def test_gradient_stability_random_input(self):
        """Test gradient stability with random input."""
        loss_fn = MINDSSCLoss()
        pred = torch.randn(1, 1, 32, 32, requires_grad=True)
        target = torch.randn(1, 1, 32, 32)

        loss = loss_fn(pred, target)
        loss.backward()

        assert not torch.isnan(pred.grad).any()

    def test_direction_set_is_symmetric(self):
        """The descriptor neighbourhood must be closed under negation.

        Regression: the old ``directions[:6]`` slice kept an asymmetric subset
        (it dropped ``(1, 0)`` and ``(1, 1)`` while keeping their negatives),
        which biased the descriptor by orientation. The full 8-neighbourhood is
        symmetric: every ``(dx, dy)`` has its mirror ``(-dx, -dy)``.
        """
        loss_fn = MINDSSCLoss()
        dirs = set(loss_fn.directions)
        assert len(dirs) == 8
        for dx, dy in dirs:
            assert (-dx, -dy) in dirs, f"missing mirror of {(dx, dy)}"

    def test_descriptor_has_eight_directional_channels(self):
        """Each input channel yields 8 directional MIND features."""
        loss_fn = MINDSSCLoss()
        desc = loss_fn._compute_mind_descriptor(torch.rand(1, 1, 16, 16))
        assert desc.shape[1] == 8


def _reference_mind_descriptor(loss_fn: MINDSSCLoss, img: torch.Tensor) -> torch.Tensor:
    """Per-direction loop with kernels rebuilt per call: the pre-vectorisation form."""
    import torch.nn.functional as F

    c = img.shape[1]
    gaussian = loss_fn._compute_gaussian_kernel(loss_fn.patch_size, loss_fn.gaussian_sigma)
    gaussian = gaussian.view(1, 1, loss_fn.patch_size, loss_fn.patch_size)
    feats = []
    for dx, dy in loss_fn.directions:
        diff = (img - torch.roll(img, shifts=(dy, dx), dims=(-2, -1))) ** 2
        feats.append(F.conv2d(diff, gaussian.repeat(c, 1, 1, 1), padding=loss_fn.padding, groups=c))
    mind = torch.cat(feats, dim=1)
    box = torch.ones(1, 1, loss_fn.patch_size, loss_fn.patch_size) / loss_fn.patch_size**2
    box = box.repeat(mind.shape[1], 1, 1, 1)
    mean = F.conv2d(mind, box, padding=loss_fn.padding, groups=mind.shape[1])
    var = F.conv2d(mind**2, box, padding=loss_fn.padding, groups=mind.shape[1]) - mean**2
    return mind / var.clamp(min=1e-6).sqrt()


def test_vectorised_descriptor_matches_the_per_direction_loop() -> None:
    """One grouped conv over the stacked shifts equals eight per-direction convs."""
    loss_fn = MINDSSCLoss()
    img = torch.rand(2, 3, 20, 20, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(
        loss_fn._compute_mind_descriptor(img), _reference_mind_descriptor(loss_fn, img)
    )


def test_mind_kernels_are_buffers_that_follow_the_module() -> None:
    """The kernels are built once and move with ``.to()`` instead of per forward."""
    loss_fn = MINDSSCLoss(patch_size=5).double()
    assert loss_fn.gaussian_kernel.shape == (1, 1, 5, 5)
    assert loss_fn.gaussian_kernel.dtype == torch.float64
    assert "gaussian_kernel" not in loss_fn.state_dict()


def test_mind_module_left_on_cpu_still_scores_a_cuda_batch() -> None:
    """Callers such as the nr_cross metric never move the module; the kernels follow the input."""
    if not torch.cuda.is_available():
        pytest.skip("needs a second device")
    img = torch.rand(1, 1, 16, 16, device="cuda")
    assert torch.isfinite(MINDSSCLoss()(img, img.flip(-1)))
