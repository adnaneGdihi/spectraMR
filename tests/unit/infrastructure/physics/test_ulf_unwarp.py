"""Unit tests for UlfUnwarp, the shared train/infer unwarp built from UnwarpConfig."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.config.schemas.training.geomamba_ulf import UnwarpConfig  # noqa: E402
from spectramr.infrastructure.physics.ulf_unwarp import UlfUnwarp  # noqa: E402


def _cfg(**kw) -> UnwarpConfig:
    base = {
        "enable_b0": False,
        "enable_grad_nonlinearity": False,
        "enable_maxwell": True,
        "echo_spacing_s": 5.5e-4,
        "gradient_amplitudes_mt_per_m": [0.0, 0.0, 0.0],
    }
    base.update(kw)
    return UnwarpConfig(**base)


def _vol() -> torch.Tensor:
    return torch.linspace(0.0, 1.0, 4 * 8 * 8).reshape(1, 1, 4, 8, 8)


def test_zero_maxwell_field_is_identity() -> None:
    vol = _vol()
    out = UlfUnwarp(_cfg()).apply(vol, {})
    assert torch.allclose(out, vol, atol=1e-5)


def test_enable_b0_without_map_key_raises() -> None:
    with pytest.raises(ValueError, match="b0_map_key is unset"):
        UlfUnwarp(_cfg(enable_b0=True))


def test_missing_b0_field_in_batch_raises() -> None:
    unwarp = UlfUnwarp(_cfg(enable_b0=True, b0_map_key="b0_hz"))
    with pytest.raises(ValueError, match="b0_hz"):
        unwarp.apply(_vol(), {})


def test_grad_nonlinearity_declared_raises() -> None:
    with pytest.raises(NotImplementedError, match="enable_grad_nonlinearity"):
        UlfUnwarp(_cfg(enable_grad_nonlinearity=True))


def test_gnl_coeff_path_raises_without_reader() -> None:
    with pytest.raises(ValueError, match="gnl_coeff_path"):
        UlfUnwarp(_cfg(gnl_coeff_path="coeffs.npz"))


def test_all_terms_disabled_returns_input_object() -> None:
    vol = _vol()
    unwarp = UlfUnwarp(_cfg(enable_maxwell=False))
    assert unwarp.apply(vol, {}) is vol


def test_maxwell_map_is_built_once_per_shape() -> None:
    unwarp = UlfUnwarp(_cfg(gradient_amplitudes_mt_per_m=[5.0, 5.0, 5.0]))
    unwarp.apply(_vol(), {})
    unwarp.apply(_vol(), {})
    assert len(unwarp._maxwell_cache) == 1


def test_beltrami_diagnostic_is_stamped_and_nonzero_for_a_real_field() -> None:
    unwarp = UlfUnwarp(
        _cfg(
            gradient_amplitudes_mt_per_m=[20.0, 20.0, 20.0],
            field_strength_t=0.064,
            voxel_size_mm=[2.0, 2.0, 2.0],
        )
    )
    unwarp.apply(_vol(), {})
    mu = unwarp.provenance()["beltrami_max"]
    assert mu is not None and mu > 0.0


def test_train_and_inference_paths_agree() -> None:
    """Two UlfUnwarp instances from the same block give the same correction."""
    cfg = _cfg(gradient_amplitudes_mt_per_m=[20.0, 20.0, 20.0], field_strength_t=0.064)
    vol = _vol()
    a = UlfUnwarp(cfg).apply(vol, {})
    b = UlfUnwarp(UnwarpConfig.model_validate(cfg.model_dump())).apply(vol, {})
    assert torch.equal(a, b)


def test_inactive_block_without_echo_spacing_builds_and_is_identity() -> None:
    cfg = UnwarpConfig(
        enable_b0=False, enable_grad_nonlinearity=False, enable_maxwell=False, field_strength_t=0.064
    )
    vol = _vol()
    assert UlfUnwarp(cfg).apply(vol, {}) is vol


@pytest.mark.parametrize("term", ["enable_b0", "enable_maxwell"])
def test_active_epi_block_without_echo_spacing_raises_at_load(term: str) -> None:
    kw = {"enable_b0": False, "enable_grad_nonlinearity": False, "enable_maxwell": False}
    kw[term] = True
    if term == "enable_b0":
        kw["b0_map_key"] = "b0"
    with pytest.raises(ValueError, match="echo_spacing_s is required"):
        UnwarpConfig(**kw)
