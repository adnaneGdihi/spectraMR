"""The GeoMamba-ULF inference strategy reads the training unwarp block and retires the old keys."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.inference.geomamba_ulf_inference_strategy import (  # noqa: E402
    GeoMambaULFInferenceStrategy,
)


def _bare(config: dict) -> GeoMambaULFInferenceStrategy:
    s = GeoMambaULFInferenceStrategy.__new__(GeoMambaULFInferenceStrategy)
    s.config = config
    s.device = torch.device("cpu")
    return s


@pytest.mark.parametrize("key", ["apply_unwarp", "echo_spacing_s"])
def test_retired_keys_raise(key: str) -> None:
    s = _bare({"unwarp": {"echo_spacing_s": 5.5e-4}, key: True})
    with pytest.raises(ValueError, match="retired"):
        s._apply_unwarp(torch.zeros(1, 1, 4, 8, 8), {})


def test_unwarp_block_applies_the_same_correction_as_training() -> None:
    from spectramr.config.schemas.training.geomamba_ulf import UnwarpConfig
    from spectramr.infrastructure.physics.ulf_unwarp import UlfUnwarp

    block = {
        "enable_b0": False,
        "enable_grad_nonlinearity": False,
        "echo_spacing_s": 5.5e-4,
        "gradient_amplitudes_mt_per_m": [20.0, 20.0, 20.0],
        "field_strength_t": 0.064,
    }
    x = torch.linspace(0.0, 1.0, 4 * 8 * 8).reshape(1, 1, 4, 8, 8)
    out = _bare({"unwarp": block})._apply_unwarp(x, {})
    expected = UlfUnwarp(UnwarpConfig(**block)).apply(x, {})
    assert torch.equal(out, expected)
