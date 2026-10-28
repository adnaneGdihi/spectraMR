"""Per-coil sensitivity-map preview for the debug snapshot (#2488)."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.training.snapshot_coil_maps import (
    COIL_MAP_KEYS,
    render_coil_map_grid,
)

pytestmark = pytest.mark.unit


def _maps(coils: int = 4, size: int = 16) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    maps = torch.stack(
        [
            torch.exp(-((yy - math.cos(a)) ** 2 + (xx - math.sin(a)) ** 2)) * torch.exp(1j * a * yy)
            for a in (2 * math.pi * c / coils for c in range(coils))
        ]
    )[None].to(torch.complex64)
    return maps / torch.sqrt((maps.abs() ** 2).sum(1, keepdim=True))


def test_the_grid_holds_magnitude_and_phase_per_coil_then_the_rss() -> None:
    maps = _maps()
    grid = render_coil_map_grid(maps)
    assert grid.shape == (9, 1, 16, 16)
    assert float(grid.min()) >= 0.0 and float(grid.max()) <= 1.0
    # One shared magnitude scale: the loudest coil reaches 1, and coils differ.
    assert float(grid[:4].amax()) == pytest.approx(1.0)
    assert all(not torch.allclose(grid[0], grid[c]) for c in (1, 2, 3))
    # RSS-normalised maps render a flat RSS, which is why the generic preview showed nothing.
    assert torch.allclose(grid[8], torch.ones_like(grid[8]), atol=1e-5)


def test_phase_is_mapped_from_minus_pi_to_pi() -> None:
    maps = torch.exp(1j * torch.tensor([-math.pi + 1e-4, 0.0, math.pi - 1e-4])).view(1, 3, 1, 1)
    grid = render_coil_map_grid(maps.to(torch.complex64))
    assert grid[3:6].flatten().tolist() == pytest.approx([0.0, 0.5, 1.0], abs=1e-3)


def test_real_interleaved_maps_render_like_complex_ones() -> None:
    maps = _maps()
    stacked = torch.view_as_real(maps).permute(0, 1, 4, 2, 3).reshape(1, 8, 16, 16)
    assert torch.allclose(render_coil_map_grid(stacked), render_coil_map_grid(maps), atol=1e-6)


def test_an_odd_real_channel_count_is_refused() -> None:
    with pytest.raises(ValueError, match="interleaved"):
        render_coil_map_grid(torch.rand(1, 3, 8, 8))


def test_the_snapshot_writes_the_per_coil_grid(tmp_path) -> None:
    """Wiring: ``save_debug_snapshot`` routes ``coil_maps`` through the grid, not the RSS preview."""
    from PIL import Image

    from spectramr.infrastructure.training.debug_snapshot import save_debug_snapshot
    from tests.unit.infrastructure.training.test_debug_snapshot import _make_logging, _reset_budget

    assert "coil_maps" in COIL_MAP_KEYS
    _reset_budget()
    out = save_debug_snapshot(
        run_dir=tmp_path,
        step=0,
        tag="coil",
        tensors={"coil_maps": _maps(size=16)},
        paradigm="Test",
        config_section=_make_logging(),
    )
    with Image.open(out / "coil_maps.png") as png:
        width, height = png.size
    # nrow=4 with torchvision's 2-px padding: 4 tiles wide, 3 rows (mags, phases, rss).
    assert width == 4 * 16 + 5 * 2
    assert height == 3 * 16 + 4 * 2


def test_kspace_conditioning_maps_render_as_their_image_domain_grid() -> None:
    """Cold diffusion feeds ``fft2c`` of the maps; its preview must show the maps, not round-off."""
    from spectramr.infrastructure.physics.fft_ops import fft2c

    maps = _maps()
    stacked_k = torch.view_as_real(0.3 * fft2c(maps)).permute(0, 1, 4, 2, 3).reshape(1, 8, 16, 16)
    assert {"smaps", "model_input__smaps"} <= COIL_MAP_KEYS
    grid = render_coil_map_grid(stacked_k, in_kspace=True)
    assert torch.allclose(grid, render_coil_map_grid(maps), atol=1e-4)
