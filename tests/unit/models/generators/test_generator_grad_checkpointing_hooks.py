"""The ten arms that relied on the generic checkpointing fallback still checkpoint (#1333).

The builder no longer wraps leaf layers for a model without ``set_grad_checkpointing``;
it raises. These are the seven models those arms select. Each must expose the hook,
checkpoint at least one segment when it is on, and leave the gradients unchanged.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks import delta_mamba
from spectramr.models.generators import (
    graph_unet_generator,
    hilbert_mamba_generators,
    temporal_coherence_4d,
)


def _ct():
    from spectramr.models.generators.ct_mamba_generator import CTMamba

    return CTMamba(d_model=8, num_layers=2, d_state=4, scan_backend="reference")


def _crm():
    from spectramr.models.generators.crm_mamba_generator import CRMMamba

    return CRMMamba(d_model=8, num_layers=2, d_state=4, scan_backend="reference")


def _hw():
    from spectramr.models.generators.hw_mamba import HWMamba

    return HWMamba(d_model=8, num_layers=2, d_state=4)


def _mm():
    return hilbert_mamba_generators.MMMamba(
        d_model=8, micro_layers=1, macro_layers=1, block_size=64, d_state=4
    )


def _hld():
    from spectramr.models.generators.hdsf_mamba_generators import HDSFHLDMamba

    return HDSFHLDMamba(
        in_channels=1, d_model=8, num_layers=1, d_state=4, time_embed_dim=8, block_size=4
    )


def _gat():
    return graph_unet_generator.GraphUNetGenerator(
        backbone_type="gat_unet", hidden_dim=8, depth=2, heads=2, im_size=(8, 8)
    )


def _tc4d():
    return temporal_coherence_4d.TemporalCoherence4DGenerator(
        in_channels=1, out_channels=1, feature_dim=8, num_3d_layers=2
    )


# (factory, input shape, module whose bound checkpoint function counts the calls, its name)
CASES = {
    "ct_mamba": (_ct, (1, 1, 16, 16), delta_mamba, "checkpoint"),
    "crm_mamba": (_crm, (1, 1, 16, 16), delta_mamba, "checkpoint"),
    "hw_mamba": (_hw, (1, 1, 16, 16), hilbert_mamba_generators, "checkpoint"),
    "mm_mamba": (_mm, (1, 1, 16, 16), hilbert_mamba_generators, "checkpoint"),
    "hdsf_hld_mamba": (_hld, (1, 1, 8, 8, 8), hilbert_mamba_generators, "checkpoint"),
    "graph_unet": (_gat, (1, 2, 8, 8), graph_unet_generator, "checkpoint"),
    "temporal_coherence_4d": (_tc4d, (1, 1, 8, 8), temporal_coherence_4d, "checkpoint_sequential"),
}


def _grads(model: torch.nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    model.zero_grad(set_to_none=True)
    out = model(x)
    out = out.real if torch.is_complex(out) else out
    out.square().sum().backward()
    return [p.grad.clone() for p in model.parameters() if p.grad is not None]


@pytest.mark.parametrize("name", sorted(CASES))
def test_hook_checkpoints_and_keeps_the_gradients(name, monkeypatch) -> None:
    monkeypatch.setenv("SPECTRAMR_ALLOW_MAMBA_FALLBACK", "1")
    factory, shape, module, attr = CASES[name]
    torch.manual_seed(0)
    model = factory().train()
    x = torch.randn(*shape)
    plain = _grads(model, x)

    calls = []
    real = getattr(module, attr)

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, attr, counting)
    model.set_grad_checkpointing(True)
    checkpointed = _grads(model, x)

    assert calls, f"{name}: set_grad_checkpointing(True) checkpointed nothing"
    assert len(plain) == len(checkpointed) > 0
    for a, b in zip(plain, checkpointed, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
