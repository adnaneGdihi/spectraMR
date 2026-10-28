"""``RectifiedFlowGenerator``: the conditioning modes and the velocity entry point.

Paired with ``src/spectramr/models/generators/rectified_flow_generator.py``.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.rectified_flow_generator import (
    CONDITIONING_MODES,
    RectifiedFlowGenerator,
)

_FEATURES = [8, 16]


def _gen(conditioning: str) -> RectifiedFlowGenerator:
    torch.manual_seed(0)
    return RectifiedFlowGenerator(
        in_channels=1,
        out_channels=1,
        features=_FEATURES,
        context_dim=8,
        flow_steps=2,
        conditioning=conditioning,
    ).eval()


def _pair() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    return torch.randn(2, 1, 16, 16), torch.rand(2), torch.randn(2, 1, 16, 16)


def test_an_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="conditioning='concat'"):
        _gen("concat")


@pytest.mark.parametrize("mode", ["global", "spatial"])
def test_a_conditional_velocity_reads_the_condition(mode: str) -> None:
    gen = _gen(mode)
    z, t, y = _pair()
    with torch.no_grad():
        a = gen.velocity(z, t, y)
        b = gen.velocity(z, t, y * 3.0 + 1.0)
    assert not torch.allclose(a, b)


def test_the_noise_only_prior_ignores_the_condition() -> None:
    gen = _gen("none")
    z, t, y = _pair()
    with torch.no_grad():
        torch.testing.assert_close(gen.velocity(z, t, y), gen.velocity(z, t, y * 3.0 + 1.0))
    assert gen.measurement_encoder is None


def test_a_conditional_mode_without_a_condition_raises() -> None:
    gen = _gen("spatial")
    z, t, _ = _pair()
    with pytest.raises(ValueError, match="needs the condition y"):
        gen.velocity(z, t, None)


def test_forward_with_a_time_is_the_velocity() -> None:
    gen = _gen("spatial")
    z, t, y = _pair()
    with torch.no_grad():
        torch.testing.assert_close(gen(z, t=t, condition=y), gen.velocity(z, t, y))


def test_sampling_is_conditioned_on_the_input() -> None:
    """``generate(y)``, the validation path, integrates from noise under y."""
    gen = _gen("spatial")
    _, _, y = _pair()
    with torch.no_grad():
        torch.manual_seed(5)
        a = gen.generate(y)
        torch.manual_seed(5)
        b = gen.generate(y * 3.0 + 1.0)
    assert a.shape == y.shape and not torch.allclose(a, b)


def test_the_vocabulary_is_closed() -> None:
    assert CONDITIONING_MODES == ("none", "global", "spatial")
