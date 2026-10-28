"""``UnifiedDisentangledLossComputer``: per-step behaviour of ``compute``.

* ``torch.nn.functional`` was imported inside the ``c_a`` PatchNCE branch, so a
  step carrying only the ``c_b`` content codes raised ``UnboundLocalError``.
* ``metrics`` called ``.item()`` on every component every step -- one device
  sync each -- for a dict no training-path caller reads. It now holds detached
  tensors with the same values.
* The stack's seed is not a ``requires_grad`` leaf (see
  ``test_base_stack_seed_and_sync.py``); this override carried its own copy.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.computers import UnifiedDisentangledLossComputer


@pytest.fixture
def computer() -> UnifiedDisentangledLossComputer:
    c = UnifiedDisentangledLossComputer(config=None)
    c.config = type("_Cfg", (), {"losses": None})()
    return c


def _pair() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.rand(2, 1, 8, 8, requires_grad=True), torch.rand(2, 1, 8, 8)


def test_content_codes_for_b_alone_do_not_raise(computer) -> None:
    c_b = torch.rand(2, 4, 4, 4)
    out = computer._compute_latent_losses(c_b=c_b, c_b_recon=c_b + 0.1)
    assert torch.isfinite(out["patch_nce"])


def test_metrics_are_detached_tensors_with_the_component_values(computer, monkeypatch) -> None:
    computer.config = type("_Cfg", (), {"losses": _Losses()})()
    pred, target = _pair()

    def no_item(self: torch.Tensor) -> float:
        raise AssertionError(".item() called on the training path")

    monkeypatch.setattr(torch.Tensor, "item", no_item)
    out = computer.compute(pred, target)
    monkeypatch.undo()

    assert out.metrics.keys() == out.components.keys()
    for name, value in out.metrics.items():
        assert isinstance(value, torch.Tensor) and not value.requires_grad
        assert float(value) == pytest.approx(float(out.components[name].detach()))


def test_a_graph_free_total_has_no_grad_fn(computer) -> None:
    computer.config = type("_Cfg", (), {"losses": _Losses()})()
    with torch.no_grad():
        pred = torch.nn.Linear(8, 8)(torch.rand(2, 1, 8, 8))
    out = computer.compute(pred, torch.rand(2, 1, 8, 8))
    assert out.total.grad_fn is None


class _Section:
    lambda_recon = 1.0


class _Losses:
    reconstruction = _Section()
    latent = None


def test_patch_nce_batched_matches_per_element_loop() -> None:
    """Below the patch cap the batched InfoNCE equals the per-element cross-entropy mean."""
    import torch
    import torch.nn.functional as F

    from spectramr.models.losses.computers.unified_disentangled import _patch_nce

    torch.manual_seed(0)
    real = F.normalize(torch.randn(3, 20, 8), dim=-1)
    fake = F.normalize(torch.randn(3, 20, 8), dim=-1)
    ref = torch.stack(
        [
            F.cross_entropy(fake[i] @ real[i].T / 0.07, torch.arange(20))
            for i in range(3)
        ]
    ).mean()
    torch.testing.assert_close(_patch_nce(real, fake, 0.07), ref)
    # Above the cap each element draws a distinct random subset; the loss stays finite.
    big = _patch_nce(real, fake, 0.07, max_patches=8)
    assert torch.isfinite(big)
