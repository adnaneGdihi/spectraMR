"""Unit tests for the HDSF (Hierarchical Dilated Space-Filling) Mamba encoder.

Focus: the linearizer factory ``HDSFMambaEncoder._get_lin`` must route strict
Hilbert modes through the resolution-agnostic ``*_rect`` sibling so a non-cubic
dynamic sub-grid (the dilated global stream, or a padded non-cubic input) cannot
raise ``AssertionError`` mid-forward. Byte-identical on cubic power-of-2 blocks.

The 3D Mamba blocks need ``mamba_ssm``. Where it is ABSENT the opt-in
Gated-Conv+GRU fallback below keeps ``HDSFMambaEncoder`` constructible, so the
linearizer path — the actual subject of these tests — stays reachable.

That opt-in does NOT make a forward pass CPU-runnable where the kernel is
PRESENT: it is consulted only in ``MambaBlock``'s ``except ImportError`` branch,
and a present kernel never reaches it. The forward test therefore carries
``requires_cuda_for_mamba`` as well; see tests/utils/optional_backends.py.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

from tests.utils.optional_backends import requires_cuda_for_mamba  # noqa: E402


@pytest.fixture(autouse=True)
def _allow_mamba_fallback(monkeypatch):
    """Kernel-absent boxes only; a no-op wherever ``mamba_ssm`` imports."""
    monkeypatch.setenv("SPECTRAMR_ALLOW_MAMBA_FALLBACK", "1")


def _encoder():
    from spectramr.models.generators.hdsf_mamba_generators import HDSFMambaEncoder

    return HDSFMambaEncoder(
        d_model=8, num_layers=1, d_state=4, block_size=4, dilation=2
    )


def test_get_lin_remaps_strict_hilbert_to_rect() -> None:
    enc = _encoder()
    # A non-cubic shape would AssertionError under strict hilbert_3d; the remap
    # must build a hilbert_3d_rect linearizer instead.
    lin = enc._get_lin((6, 10, 12), "hilbert_3d", torch.device("cpu"))
    assert lin.mode == "hilbert_3d_rect"


@requires_cuda_for_mamba
def test_forward_non_cubic_non_pow2() -> None:
    """A non-cubic, non-power-of-2 volume must forward without crashing (the
    dilated global stream produces a non-cubic sub-grid)."""
    enc = _encoder()
    x = torch.randn(1, 8, 6, 10, 12)
    out = enc(x)
    assert out.shape == (1, 8, 6, 10, 12)


class _NoMixing(torch.nn.Module):
    """Stands in for the Mamba encoder so the time code runs on any box."""

    def forward(self, x):
        return torch.zeros_like(x)


def _time_code(model, timesteps):
    seen = []
    model.encoder = _NoMixing()
    model.time_mlp.register_forward_hook(lambda _m, args, _out: seen.append(args[0]))
    model(torch.randn(timesteps.shape[0], 2, 1, 4, 4), timesteps=timesteps)
    return seen[0].squeeze(-1)


@pytest.mark.parametrize("horizon", [29, 1000])
def test_hld_mamba_divides_by_its_horizon(horizon) -> None:
    """PLANTED (#2442): t / 1000 put a 29-step chain in [0, 0.028]."""
    from spectramr.models.generators.hdsf_mamba_generators import HDSFHLDMamba

    model = HDSFHLDMamba(d_model=8, num_layers=1, d_state=4, block_size=4, dilation=2,
                         time_embed_dim=8, num_timesteps=horizon)
    code = _time_code(model, torch.tensor([0, horizon - 1]))
    assert torch.equal(code, torch.tensor([0, horizon - 1]).float() / horizon)


def test_exp_hm_09_resolves_its_declared_horizon() -> None:
    """The only arm on this class declares T=1000, so its code is unchanged."""
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.builders.generator_kwargs import resolve_generator_kwargs
    from spectramr.models.generators.hdsf_mamba_generators import HDSFHLDMamba

    cfg = TrainingSettings.from_yaml("experiments/inprogress/hilbert_mamba/exp_hm_09_hld_mamba.yaml")
    kwargs = resolve_generator_kwargs(config=cfg, model_cls=HDSFHLDMamba, device="cpu")
    assert kwargs["num_timesteps"] == cfg.training.diffusion.timesteps == 1000
