import torch
import torchio as tio

from spectramr.data.transforms.realistic_degradations import (
    B0GeometricDistortion,
    RandomB0Distortion,
)


def test_b0_distortion_grid():
    """Test that B0 distortion warps a grid pattern."""
    # Create a grid image [1, 1, 1, 32, 32] -> [B, C, D, H, W]
    H, W = 32, 32
    grid = torch.zeros(1, 1, 1, H, W)
    # Add vertical lines
    grid[..., ::4] = 1.0

    # Create a B0 map (gradient along Y -> shift along Y changes)
    # If B0 is constant, shift is constant (translation).
    # If B0 gradient, shift varies (stretching/compressing).

    # Simple gradient B0 map
    b0_map = torch.linspace(-1, 1, W).view(1, 1, 1, 1, W).expand(1, 1, 1, H, W)
    # This b0 map varies along W (x-axis).
    # shift_dim=4 shifts pixels along W by a W-varying amount, so the lines bend/stretch.

    distorter = B0GeometricDistortion(strength=5.0, shift_dim=4)  # 5 pixels max shift

    out = distorter(grid, b0_map)

    # Check that output is different from input
    assert not torch.allclose(out, grid), "Output should be distorted"

    # Check shape
    assert out.shape == grid.shape


def test_random_b0_distortion_tio():
    """Test TorchIO wrapper."""
    # Create subject
    image_data = torch.rand(1, 32, 32, 1)  # [C, W, H, D] for TIO?
    # TIO standard is (C, W, H, D).
    # Let's use 3D
    image_data = torch.rand(1, 32, 32, 32)
    image = tio.ScalarImage(tensor=image_data)
    subject = tio.Subject(input=image)

    transform = RandomB0Distortion(max_displacement=2.0, p=1.0)

    transformed = transform(subject)

    assert "input" in transformed
    out_data = transformed["input"].data

    assert out_data.shape == image_data.shape
    assert not torch.allclose(
        out_data, image_data
    ), "Output should be distorted with p=1.0"
