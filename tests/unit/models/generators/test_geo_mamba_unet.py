"""Unit tests for GeoMambaUNet.

Pin shape correctness, the registry registration, the foreground
re-embedding invariant (background voxels equal input), and that
gradients flow through both the FiLM MLP and the Mamba blocks.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from spectramr.models.blocks.metric_sfc import MetricSFCLinearizer  # noqa: E402
from spectramr.models.generators.geo_mamba_unet import GeoMambaUNet  # noqa: E402
from spectramr.models.registry import MODEL_REGISTRY  # noqa: E402
from tests.utils.optional_backends import requires_cuda_for_mamba  # noqa: E402


def _toy_inputs(B: int = 1, D: int = 4, H: int = 8, W: int = 8):
    rng = np.random.default_rng(0)
    intensity = rng.standard_normal((D, H, W)).astype(np.float32)
    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    cz, cy, cx = (D - 1) / 2.0, (H - 1) / 2.0, (W - 1) / 2.0
    r2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    mask = (r2 <= 2.0**2).astype(np.uint8)
    volume = torch.randn(B, 1, D, H, W)
    mask_t = torch.from_numpy(mask).to(torch.float32)
    return volume, mask_t, torch.from_numpy(intensity)


def test_registered_in_global_registry() -> None:
    assert "geo_mamba_unet" in MODEL_REGISTRY
    entry = MODEL_REGISTRY["geo_mamba_unet"]
    assert entry["mode"] == "geomamba_ulf"
    assert entry["class"].__name__ == "GeoMambaUNet"


def test_package_init_imports_geo_mamba_unet_module() -> None:
    """Registration must fire on the PRODUCTION path — the bare
    ``import spectramr.models.generators`` the model factory performs — not only
    when a unit test imports the submodule directly (which this very file does
    at module load, registering it as a side effect).

    Pinned as a SOURCE assertion because ``@register_model`` is a
    process-wide side effect: once any test has imported
    ``geo_mamba_unet``, a runtime ``in MODEL_REGISTRY`` check passes
    spuriously regardless of whether the package ``__init__`` wires the
    import. Before 2026-06 the line was missing and all 14
    ``GeoMambaULFStrategy`` arms crashed at model-build with
    ``Generator type 'geo_mamba_unet' not registered``.
    """
    import pathlib

    init_src = pathlib.Path(
        "src/spectramr/models/generators/__init__.py"
    ).read_text(encoding="utf-8")
    assert "import spectramr.models.generators.geo_mamba_unet" in init_src


def test_metric_sfc_params_reach_lazy_linearizer() -> None:
    """The metric-SFC knobs (beta / smoothing_sigma / connectivity / cache_dir)
    must reach the lazily-built ``MetricSFCLinearizer``. Before 2026-06 the
    build hard-coded the defaults, so the whole ``MetricSFCConfig`` was inert
    (``abl_no_metric_sfc`` with beta=0 trained identically to ``v0``)."""
    model = GeoMambaUNet(
        in_channels=1,
        out_channels=1,
        d_model=8,
        n_blocks=1,
        sfc_beta=2.5,
        sfc_smoothing_sigma=0.0,
        sfc_connectivity=6,
        sfc_cache_dir=None,
    )
    volume = torch.randn(1, 1, 4, 8, 8)
    mask = (torch.rand(1, 1, 4, 8, 8) > 0.3).to(torch.float32)
    lin = model._resolve_linearizer(volume, mask, None)
    assert lin.beta == 2.5
    assert lin.smoothing_sigma == 0.0
    assert lin.connectivity == 6


def test_metric_sfc_defaults_match_schema() -> None:
    """An unset config is behaviour-preserving: model defaults equal the
    MetricSFCConfig schema defaults (beta=10, sigma=1, conn=26)."""
    model = GeoMambaUNet(in_channels=1, out_channels=1, d_model=8, n_blocks=1)
    assert model.sfc_beta == 10.0
    assert model.sfc_smoothing_sigma == 1.0
    assert model.sfc_connectivity == 26
    assert model.sfc_cache_dir == "cache/metric_sfc"


@requires_cuda_for_mamba
def test_forward_shape_with_injected_linearizer(tmp_path) -> None:
    volume, mask, intensity = _toy_inputs()
    linearizer = MetricSFCLinearizer(
        intensity_volume=intensity, mask=mask, cache_dir=str(tmp_path)
    )
    model = GeoMambaUNet(
        in_channels=1, out_channels=1, d_model=16, n_blocks=2, linearizer=linearizer
    )
    out = model(volume, mask=mask, contrast_id=torch.tensor([0]))
    assert out.shape == volume.shape


@requires_cuda_for_mamba
def test_lazy_linearizer_construction(tmp_path) -> None:
    """Without an injected linearizer, the model builds ONE on the first call
    and reuses it via the early-return path — no per-forward cache-key
    ``.item()`` / ``.cpu()`` GPU sync (the old ``_lazy_cache`` dict)."""
    volume, mask, _ = _toy_inputs()
    model = GeoMambaUNet(in_channels=1, out_channels=1, d_model=16, n_blocks=1)
    assert model._linearizer is None
    out = model(volume, mask=mask)
    assert out.shape == volume.shape
    first = model._linearizer
    assert first is not None
    # A second call reuses the SAME linearizer object (early-return).
    _ = model(volume, mask=mask)
    assert model._linearizer is first
    # The per-forward cache dict (and its GPU-syncing key computation) is gone.
    assert not hasattr(model, "_lazy_cache")


@requires_cuda_for_mamba
def test_lazy_build_is_stationary(tmp_path) -> None:
    """A second forward with a DIFFERENT mask reuses the first linearizer — the
    permutation is a fixed first-batch template (matching MetricSFCLinearizer's
    'coregistered template' design), not a non-stationary per-batch curve."""
    volume, mask, _ = _toy_inputs()
    model = GeoMambaUNet(in_channels=1, out_channels=1, d_model=16, n_blocks=1)
    _ = model(volume, mask=mask)
    first = model._linearizer
    other = (torch.rand_like(mask) > 0.5).to(mask.dtype)
    _ = model(volume, mask=other)
    assert model._linearizer is first


@requires_cuda_for_mamba
def test_lazy_linearizer_state_dict_round_trips(tmp_path) -> None:
    """After the first forward builds the linearizer, its buffers are part of
    the module tree and survive a state_dict save/load round-trip."""
    volume, mask, _ = _toy_inputs()
    model = GeoMambaUNet(in_channels=1, out_channels=1, d_model=16, n_blocks=1)
    _ = model(volume, mask=mask)
    sd = model.state_dict()
    assert any("_linearizer" in k for k in sd)
    fresh = GeoMambaUNet(in_channels=1, out_channels=1, d_model=16, n_blocks=1)
    _ = fresh(volume, mask=mask)  # build the same buffers, then load
    fresh.load_state_dict(sd)


@requires_cuda_for_mamba
def test_background_voxels_unchanged(tmp_path) -> None:
    """Voxels outside the foreground mask must equal the input voxels."""
    volume, mask, intensity = _toy_inputs()
    linearizer = MetricSFCLinearizer(
        intensity_volume=intensity, mask=mask, cache_dir=str(tmp_path)
    )
    model = GeoMambaUNet(
        in_channels=1, out_channels=1, d_model=16, n_blocks=1, linearizer=linearizer
    )
    out = model(volume, mask=mask)
    bg_locations = mask < 0.5
    # Broadcast bg_locations from (D, H, W) to (B, C, D, H, W).
    bg_locations = bg_locations.unsqueeze(0).unsqueeze(0).expand_as(volume)
    assert torch.allclose(out[bg_locations], volume[bg_locations], atol=1e-5)


@requires_cuda_for_mamba
def test_gradient_flows_to_film_and_mamba(tmp_path) -> None:
    volume, mask, intensity = _toy_inputs()
    volume = volume.detach().requires_grad_(True)
    linearizer = MetricSFCLinearizer(
        intensity_volume=intensity, mask=mask, cache_dir=str(tmp_path)
    )
    model = GeoMambaUNet(
        in_channels=1, out_channels=1, d_model=16, n_blocks=2, linearizer=linearizer
    )
    out = model(volume, mask=mask)
    out.sum().backward()
    # Input grads.
    assert volume.grad is not None and torch.any(volume.grad != 0)
    # FiLM MLP weight gradient.
    block0 = model.blocks[0]
    assert block0.film[0].weight.grad is not None
    assert torch.any(block0.film[0].weight.grad != 0)


def test_rejects_wrong_input_dim() -> None:
    model = GeoMambaUNet(in_channels=1, out_channels=1, d_model=16, n_blocks=1)
    bad = torch.zeros(1, 1, 8, 8)  # 4D
    with pytest.raises(ValueError, match="(B, C, D, H, W)"):
        model(bad)


def test_rejects_wrong_channel_count() -> None:
    model = GeoMambaUNet(in_channels=2, out_channels=2, d_model=16, n_blocks=1)
    volume, mask, _ = _toy_inputs()  # 1 channel
    with pytest.raises(ValueError, match="2"):
        model(volume, mask=mask)


def _lazy_curve_model(**kw) -> GeoMambaUNet:
    return GeoMambaUNet(
        in_channels=1,
        out_channels=1,
        d_model=8,
        n_blocks=1,
        sfc_smoothing_sigma=0.0,
        sfc_cache_dir=None,
        **kw,
    )


def test_stored_curve_survives_strict_load_into_fresh_model() -> None:
    """The lazily built curve is persisted, so a fresh model loads it strictly
    and gets the identical permutation without running a forward pass."""
    volume, mask, _ = _toy_inputs()
    trained = _lazy_curve_model()
    trained._resolve_linearizer(volume, mask, None)
    sd = trained.state_dict()
    assert "_linearizer_template" in sd and "_linearizer_mask" in sd
    fresh = _lazy_curve_model()
    assert fresh._linearizer is None
    fresh.load_state_dict(sd)
    assert fresh._linearizer is not None
    assert torch.equal(fresh._linearizer.forward_idx, trained._linearizer.forward_idx)
    assert torch.equal(fresh._linearizer.inverse_idx, trained._linearizer.inverse_idx)


def test_second_foreground_reuses_stored_curve() -> None:
    volume, mask, _ = _toy_inputs()
    model = _lazy_curve_model()
    first = model._resolve_linearizer(volume, mask, None)
    other = (torch.rand_like(mask) > 0.5).to(mask.dtype)
    second = model._resolve_linearizer(volume, other, None)
    assert second is first
    assert torch.equal(model._linearizer_mask, mask[0] if mask.ndim == 4 else mask)


def test_linearizer_rejects_mismatched_spatial_shape() -> None:
    volume, mask, _ = _toy_inputs()
    model = _lazy_curve_model()
    lin = model._resolve_linearizer(volume, mask, None)
    with pytest.raises(ValueError, match="spatial shape mismatch"):
        lin(torch.zeros(1, 1, 4, 8, 4))


def test_checkpoint_without_stored_curve_raises() -> None:
    """Pre-persistence checkpoints carry SFC index buffers but no template; they
    must fail by name rather than load with a silently different curve."""
    volume, mask, _ = _toy_inputs()
    trained = _lazy_curve_model()
    trained._resolve_linearizer(volume, mask, None)
    sd = {
        k: v
        for k, v in trained.state_dict().items()
        if "_linearizer_" not in k or "_linearizer." in k
    }
    fresh = _lazy_curve_model()
    with pytest.raises(ValueError, match="predates the persisted curve"):
        fresh.load_state_dict(sd)


def test_displacement_head_off_exposes_no_displacement() -> None:
    model = GeoMambaUNet(in_channels=1, out_channels=1, d_model=8, n_blocks=1)
    volume, mask, _ = _toy_inputs()
    model(volume, mask=mask)
    assert model.last_displacement is None


def test_zero_displacement_is_the_identity_resample() -> None:
    out = torch.rand(1, 1, 4, 8, 8)
    warped, disp_px = GeoMambaUNet._warp_by_displacement(out, torch.zeros(1, 3, 4, 8, 8), (4, 8, 8))
    torch.testing.assert_close(warped, out)
    assert torch.count_nonzero(disp_px) == 0


def test_head_bias_is_a_planted_shift_in_voxels() -> None:
    """A bias of 0.5 on x is 0.5 * (W-1)/2 voxels of displacement along W."""
    model = GeoMambaUNet(
        in_channels=1, out_channels=1, d_model=8, n_blocks=1, displacement_head=True
    )
    assert model.displacement_head is not None
    with torch.no_grad():
        model.displacement_head.bias[0] = 0.5
    volume, mask, _ = _toy_inputs()
    model(volume, mask=mask)
    disp = model.last_displacement
    assert disp is not None and disp.shape == (1, 3, 4, 8, 8)
    assert torch.allclose(disp[:, 0], torch.full_like(disp[:, 0], 0.5 * (8 - 1) / 2))


def test_displacement_head_receives_gradient() -> None:
    model = GeoMambaUNet(
        in_channels=1, out_channels=1, d_model=8, n_blocks=1, displacement_head=True
    )
    volume, mask, _ = _toy_inputs()
    out = model(volume, mask=mask)
    out.sum().backward()
    assert model.displacement_head.weight.grad is not None
    assert torch.any(model.displacement_head.weight.grad != 0)
