"""Regression: ``rss_coils_to_magnitude`` at the ``pre_model`` hook.

The 2026-05-10 cluster smoke run flagged 16 ``experiments/inprogress``
arms with the runtime error ``[DomainMismatch] Model expects 1 input
channels, but dataset provided 4 channels``. Root cause: the m4raw
dataset's cross-contrast pipeline ([src/data/datasets/m4raw_dataset.py:818-852])
RSS-combines coils per-side, then concats source+target along the
coil dim and interleaves real/imag — producing 4 channels even when
the YAML declares ``coil_processing_mode: rss_image``.

The fix (Option B, 2026-05-11): allow the existing
``rss_coils_to_magnitude`` adapter to run at the ``pre_model`` hook and
wire ``BaseTrainingStrategy.train_step`` to apply the ``pre_model``
chain to both ``input_batch_prepared`` and ``target_batch`` before the
strict DomainMismatch check fires.

This test pins three pieces of that contract:

1. ``pre_model`` is listed in
   ``rss_coils_to_magnitude``'s registered ``insertion_points``.
2. ``AdapterChainBuilder`` accepts a YAML-shaped
   ``adapters.pre_model: [{name: rss_coils_to_magnitude}]`` config
   and produces a runnable chain.
3. Running that chain on a ``[B, 4, H, W]`` tensor collapses it to
   ``[B, 1, H, W]`` via RSS (sum-of-squares over the channel dim).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import spectramr.data.adapters  # noqa: F401, E402  triggers @register_adapter
from spectramr.data.adapters.registry import get_adapter, get_adapter_capabilities  # noqa: E402
from spectramr.infrastructure.builders.leaf.adapter_builders import (  # noqa: E402
    AdapterChainBuilder,
    apply_chain,
)


def test_rss_coils_to_magnitude_advertises_pre_model() -> None:
    caps = get_adapter_capabilities("rss_coils_to_magnitude")
    assert caps is not None, "adapter not registered"
    assert "pre_model" in caps.insertion_points, (
        "rss_coils_to_magnitude must list pre_model as a valid insertion "
        "point so AdapterChainBuilder accepts it under adapters.pre_model "
        f"(got insertion_points={caps.insertion_points})"
    )


def test_adapter_chain_builder_accepts_pre_model_chain() -> None:
    """Mock a v6.0 AdaptersConfigSchema with a single pre_model step."""
    step = SimpleNamespace(name="rss_coils_to_magnitude", enabled=True)
    step.model_dump = lambda: {"name": "rss_coils_to_magnitude", "enabled": True}
    adapters_cfg = SimpleNamespace(
        pre_model=[step],
        post_model=[], pre_loss_target=[], pre_metric=[],
    )
    chains = AdapterChainBuilder(adapters_cfg).build()
    assert "pre_model" in chains
    assert len(chains["pre_model"]) == 1, chains["pre_model"]
    instance = chains["pre_model"][0]
    # The instance is the registered adapter class.
    assert type(instance) is get_adapter("rss_coils_to_magnitude")


def test_pre_model_chain_collapses_four_channel_to_one_channel() -> None:
    """The runtime contract: ``[B, 4, H, W] → [B, 1, H, W]`` via RSS.

    This mirrors what the m4raw dataset produces post-cross-contrast
    real/imag interleave; the chain must collapse it so the model's
    ``in_channels=1`` contract is satisfied before the DomainMismatch
    check at [base.py:633-645].
    """
    step = SimpleNamespace(name="rss_coils_to_magnitude", enabled=True)
    step.model_dump = lambda: {"name": "rss_coils_to_magnitude", "enabled": True}
    adapters_cfg = SimpleNamespace(
        pre_model=[step],
        post_model=[], pre_loss_target=[], pre_metric=[],
    )
    chain = AdapterChainBuilder(adapters_cfg).build()["pre_model"]

    # Synthesise an m4raw-shaped 4-ch real tensor (real/imag interleaved
    # across the 2 cross-contrast "coils").
    x = torch.randn(2, 4, 8, 8)
    y = apply_chain(chain, x)

    assert y.shape == (2, 1, 8, 8), f"expected RSS to produce 1-ch, got {y.shape}"
    # Sanity: RSS of N real channels = sqrt(sum(x^2)) — must be >= max
    # absolute value of any one channel.
    expected_lower_bound = x.abs().max(dim=1, keepdim=True).values
    assert (y >= expected_lower_bound - 1e-5).all(), (
        "RSS magnitude must be >= per-channel max"
    )


def test_pre_model_chain_is_identity_on_single_channel() -> None:
    """Idempotency for the re-application case.

    ``train_step`` applies the pre_model chain early (before the
    DomainMismatch check) and the loss path may re-apply it later.
    For ``rss_coils_to_magnitude`` the second call sees a 1-ch input
    and must return the tensor unchanged — otherwise the loss would
    compare a re-RSS'd tensor against an un-re-RSS'd prediction.
    """
    step = SimpleNamespace(name="rss_coils_to_magnitude", enabled=True)
    step.model_dump = lambda: {"name": "rss_coils_to_magnitude", "enabled": True}
    adapters_cfg = SimpleNamespace(
        pre_model=[step],
        post_model=[], pre_loss_target=[], pre_metric=[],
    )
    chain = AdapterChainBuilder(adapters_cfg).build()["pre_model"]

    x = torch.randn(2, 1, 8, 8)
    y = apply_chain(chain, x)
    assert torch.equal(x, y), "rss_coils_to_magnitude on 1-ch input must be identity"


def test_rss_coils_to_magnitude_lists_its_hooks() -> None:
    """Pin the canonical hook set so future edits surface intent.

    The supported hooks are: ``pre_model``, ``pre_loss_target``,
    ``pre_metric``. ``post_model`` is intentionally NOT listed
    (RSS-then-model-output would lose channel information).
    """
    caps = get_adapter_capabilities("rss_coils_to_magnitude")
    assert caps is not None
    expected = {"pre_model", "pre_loss_target", "pre_metric"}
    assert set(caps.insertion_points) == expected, (
        f"rss_coils_to_magnitude insertion_points drifted: "
        f"got {set(caps.insertion_points)}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# VF-smoke-2026-05-25 — ``magnitude_from_complex`` at ``pre_model``
# ---------------------------------------------------------------------------
#
# Six VF arms (eval_c7, exp_c4, exp_c6, exp_p1, exp_p2, exp_p7) declare
# ``magnitude_from_complex`` under ``adapters.pre_model`` to feed a
# real-valued image model the magnitude of a complex/2C-interleaved
# acquisition. Before the fix ``insertion_points`` omitted ``pre_model``
# so the pipeline failed loud with ``Adapter 'magnitude_from_complex' is
# not allowed at hook 'pre_model'``. The sibling ``rss_coils_to_magnitude``
# (identical magnitude reduction) already allowed ``pre_model``; the
# omission was a configuration-side blocker, not a contract invariant.


def test_magnitude_from_complex_advertises_pre_model() -> None:
    caps = get_adapter_capabilities("magnitude_from_complex")
    assert caps is not None, "adapter not registered"
    assert "pre_model" in caps.insertion_points, (
        "magnitude_from_complex must list pre_model so AdapterChainBuilder "
        "accepts it under adapters.pre_model (VF smoke 2026-05-25). "
        f"Got insertion_points={caps.insertion_points}"
    )


def test_magnitude_from_complex_pre_model_chain_takes_magnitude() -> None:
    """``[B, 2, H, W]`` interleaved real → ``[B, 1, H, W]`` magnitude."""
    step = SimpleNamespace(name="magnitude_from_complex", enabled=True)
    step.model_dump = lambda: {"name": "magnitude_from_complex", "enabled": True}
    adapters_cfg = SimpleNamespace(
        pre_model=[step],
        post_model=[], pre_loss_target=[], pre_metric=[],
    )
    chain = AdapterChainBuilder(adapters_cfg).build()["pre_model"]

    real = torch.tensor([[3.0]]).reshape(1, 1, 1, 1).expand(1, 1, 4, 4)
    imag = torch.tensor([[4.0]]).reshape(1, 1, 1, 1).expand(1, 1, 4, 4)
    x = torch.cat([real, imag], dim=1)  # [1, 2, 4, 4] interleaved (even=real)
    y = apply_chain(chain, x)

    assert y.shape == (1, 1, 4, 4), f"expected 1-ch magnitude, got {y.shape}"
    # sqrt(3^2 + 4^2) == 5
    assert torch.allclose(y, torch.full_like(y, 5.0), atol=1e-5)
