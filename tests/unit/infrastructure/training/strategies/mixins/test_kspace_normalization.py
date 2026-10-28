"""Regression: KspaceMixin.apply_kspace_normalization must NOT silently
swallow a normalization failure (CLAUDE.md #3/#10).

The pre-fix code caught any exception and returned the ORIGINAL un-normalized
tensors, and its only diagnostic was gated on ``self._strategy_logger`` — an
attribute never set anywhere — so failures were fully silent. The fix logs via
the real ``self.logging_service`` and re-raises.
"""

from __future__ import annotations

import inspect
import types
from unittest.mock import MagicMock

import pytest
import torch

from spectramr.infrastructure.training.strategies.mixins.kspace import KspaceMixin


def test_mask_saturation_gate_reads_loop_state_not_frozen_env_step() -> None:
    """WS-3 follow-up: the DEBUG mask-saturation gate in
    ``generate_and_process_mask`` keyed on ``(self.env.step ...) in [0, 1, 2]``
    — but the frozen ``env.step`` was a constant 0, so ``0 in [0, 1, 2]`` was
    always true and the warning fired EVERY step under DEBUG (pitfall #16). It
    now gates on the live ``resolve_loop_iteration(self)`` seam; active code no
    longer reads ``self.env.step``."""
    code = "\n".join(
        ln
        for ln in inspect.getsource(KspaceMixin.generate_and_process_mask).splitlines()
        if not ln.lstrip().startswith("#")
    )
    assert "self.env.step" not in code
    assert "resolve_loop_iteration(self) in [0, 1, 2]" in code


class _RaisingNormalizer:
    def normalize(self, x, *, scale=None, channel_dim=0):
        raise RuntimeError("boom")


class _OkNormalizer:
    """Mirrors ``KSpaceNormalizationSpec.normalize``'s real contract.

    Keyword-only ``scale`` / ``channel_dim``, and it divides (percentile
    divide) rather than returning the input untouched. The mixin has passed
    ``channel_dim=1`` since #572 because strategy tensors are [B, C, H, W]; a
    double whose signature has drifted stops covering the call it claims to.
    """

    def __init__(self):
        self.channel_dims: list[int] = []

    def normalize(self, x, *, scale=None, channel_dim=0):
        self.channel_dims.append(channel_dim)
        s = torch.tensor(2.0) if scale is None else scale
        return x / s, s


def _obj(normalizer):
    o = types.SimpleNamespace()
    o.kspace_normalizer = normalizer
    o.logging_service = MagicMock()
    return o


def test_normalization_failure_raises_and_logs():
    o = _obj(_RaisingNormalizer())
    with pytest.raises(RuntimeError, match="boom"):
        KspaceMixin.apply_kspace_normalization(
            o, torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4), current_step=7
        )
    # Surfaced via the real logging_service (NOT the never-set _strategy_logger).
    o.logging_service.log_error.assert_called_once()


def test_resolver_disagreeing_with_the_caller_raises():
    """A freshly-resolved DISABLED spec means the two readers disagree.

    Reaching this function means the caller read
    ``data.processing.enable_kspace_normalization`` and got True. A spec that
    resolves ``enabled=False`` from the same declaration therefore cannot be
    honoured: ``normalize()`` is a silent no-op when disabled, so it would hand
    back the RAW batch and a unit scale while reporting success. That is how
    ``experiment_11_attention_none`` trained on raw k-space at ``|k|max ~ 2400``
    — the resolver read flat pre-decomposition names no schema still carries.
    """
    from spectramr.config.schemas.data import DataProcessingConfigSchema

    o = types.SimpleNamespace()
    o.logging_service = MagicMock()
    # No `kspace_normalizer` set, so the mixin resolves one -- from a block that
    # declares normalization OFF while the caller decided it was needed.
    o.config = types.SimpleNamespace(
        data=types.SimpleNamespace(
            processing=DataProcessingConfigSchema(enable_kspace_normalization=False)
        )
    )
    with pytest.raises(RuntimeError, match="silent no-op"):
        KspaceMixin.apply_kspace_normalization(o, torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4))


def test_normalization_success_scales_both():
    norm = _OkNormalizer()
    o = _obj(norm)
    inp, tgt, scale = KspaceMixin.apply_kspace_normalization(
        o, torch.ones(1, 2, 4, 4), torch.full((1, 2, 4, 4), 4.0)
    )
    assert float(scale) == 2.0
    # The input's scale is REUSED for the target -- input and target must land
    # in the same units or the loss grades a rescaled reference.
    assert torch.allclose(inp, torch.full((1, 2, 4, 4), 0.5))
    assert torch.allclose(tgt, torch.full((1, 2, 4, 4), 2.0))
    # Strategy tensors are [B, C, H, W]: channel_dim=1, not the spec default 0.
    assert norm.channel_dims == [1, 1]


class TestValidationDoesNotNormalizeATwiceNormalizedBatch:
    """A published ``kspace_scale`` must stop the second division.

    ``_prepare_validation_data`` has two branches and its own comment states the
    rule: when the batch carries a scale, "Dataloader has ALREADY normalized the
    tensors. Do not divide again!". The ``else`` branch recomputes a scale from a
    hardcoded 0.99 quantile of the input and divides input AND target by it.

    Reaching the marker required ``isinstance(batch_data, dict)`` or
    ``hasattr(batch_data, "kspace_scale")``. A :class:`TrainingBatch` fails both,
    so the published scale read as absent and validation always took the divide
    branch -- compressing an already-compressed tensor by a quantity that is not
    even the scale training used (training honours the declared
    ``kspace_percentile``; this branch is pinned at 0.99).

    The compression is ``log1p``-domain, where a division does not commute with
    the ``expm1`` the renderer applies, so the later multiply-back cannot undo it.
    That is the mechanism behind ``experiment_11_attention_none``'s washed-out
    validation renders.
    """

    @staticmethod
    def _strategy():
        o = types.SimpleNamespace()
        o.logging_service = MagicMock()
        # CPU is correct here: the accelerated-run contract (non-negotiable 9b)
        # governs heavy pipelines, not a unit test of one branch decision.
        o.device = torch.device("cpu")
        o.config = types.SimpleNamespace(
            data=types.SimpleNamespace(
                processing=types.SimpleNamespace(enable_kspace_normalization=True)
            ),
            model=types.SimpleNamespace(in_channels=2),
        )
        # The production gate, bound to the stub (#2607): it returns at once
        # because the stub has no generator carrying a kspace_process.
        o._gate_validation_line_axis = types.MethodType(
            KspaceMixin._gate_validation_line_axis, o
        )
        return o

    @staticmethod
    def _batch_publishing(scale):
        from spectramr.data.batch_types import BatchAdapter

        tensor = torch.ones(1, 2, 8, 8)
        return BatchAdapter.from_dict(
            {"input": tensor, "target": tensor.clone(), "kspace_scale": scale}
        )

    def test_a_published_scale_is_honoured_through_a_training_batch(self) -> None:
        """The regression: the tensors must come back UNDIVIDED."""
        published = torch.tensor(224.359)
        inp = torch.full((1, 2, 8, 8), 3.0)
        tgt = torch.full((1, 2, 8, 8), 3.0)

        out_in, out_tgt, scale = KspaceMixin._prepare_validation_data(
            self._strategy(),
            None,
            inp,
            tgt,
            self._batch_publishing(published),
        )

        assert torch.allclose(out_in, inp), (
            "input was divided a second time; the batch published its scale"
        )
        assert torch.allclose(out_tgt, tgt), "target was divided a second time"
        assert float(scale.reshape(-1)[0]) == pytest.approx(224.359), (
            "the returned scale must be the PUBLISHED one, not a recomputed 0.99 "
            "quantile -- downstream de-normalization multiplies by it"
        )

    def test_a_batch_with_no_scale_still_computes_one(self) -> None:
        """Anti-overshoot: absence must keep the compensating branch reachable.

        A loader that genuinely served raw tensors still needs validation to
        normalize them, so the fix must add reach without disabling the fallback.
        """
        from spectramr.data.batch_types import BatchAdapter

        tensor = torch.rand(1, 2, 8, 8) * 100.0
        batch = BatchAdapter.from_dict({"input": tensor, "target": tensor.clone()})

        _, _, scale = KspaceMixin._prepare_validation_data(
            self._strategy(), None, tensor.clone(), tensor.clone(), batch
        )
        assert float(scale.reshape(-1)[0]) != 1.0, (
            "with nothing published, the 0.99-quantile fallback must still run"
        )
