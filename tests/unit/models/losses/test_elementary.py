"""Tests for the registry-routed elementary-loss resolver (WS-6, plan #6).

``resolve_elementary_loss`` replaces the per-file ``if/elif`` chains that
mapped ``loss_type`` strings onto ``torch.nn.functional`` calls inside the
diffusion ``p_losses`` methods and ``vq_reconstruction_loss``. The contract:
registry-resolved, constructed once (shared instances), numerically identical
to the replaced functionals, and raising — never defaulting — on unknown
names (pitfall #9).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from spectramr.models.losses.elementary import (  # noqa: E402
    resolve_elementary_loss,
    shared_loss,
)


def _pair():
    torch.manual_seed(0)
    return torch.randn(2, 2, 8, 8), torch.randn(2, 2, 8, 8)


def test_matches_replaced_functionals() -> None:
    pred, target = _pair()
    assert torch.allclose(resolve_elementary_loss("l1")(pred, target), F.l1_loss(pred, target))
    assert torch.allclose(resolve_elementary_loss("l2")(pred, target), F.mse_loss(pred, target))
    # HuberLoss(delta=1.0) is exactly F.smooth_l1_loss(beta=1.0).
    assert torch.allclose(
        resolve_elementary_loss("huber")(pred, target), F.smooth_l1_loss(pred, target)
    )


def test_instances_are_shared_not_per_call() -> None:
    assert resolve_elementary_loss("l1") is resolve_elementary_loss("l1")
    assert resolve_elementary_loss("huber") is resolve_elementary_loss("huber")


def test_unknown_loss_type_raises_never_defaults() -> None:
    # "ssim" IS a registered loss — but not an elementary one; the resolver
    # must reject anything outside its advertised set.
    with pytest.raises(ValueError, match="Unsupported loss type"):
        resolve_elementary_loss("ssim")
    with pytest.raises(ValueError, match="huber"):
        resolve_elementary_loss("nope")


def test_shared_loss_caches_per_name_and_kwargs() -> None:
    a = shared_loss("log_spectral_phase", phase_weight=0.1)
    b = shared_loss("log_spectral_phase", phase_weight=0.1)
    c = shared_loss("log_spectral_phase", phase_weight=0.2)
    assert a is b
    assert a is not c


def test_shared_instances_are_stateless() -> None:
    """Sharing is only safe with no parameters/buffers — pin that property."""
    for name in ("l1", "l2", "huber"):
        loss = resolve_elementary_loss(name)
        assert not list(loss.parameters())
        assert not list(loss.buffers())


def test_core_metrics_first_import_order_keeps_registry_complete() -> None:
    """Regression (WS-6): importing ``spectramr.core.metrics`` BEFORE
    ``spectramr.models.losses`` used to abort the losses-package init mid-list —
    a metric module imported models.losses, whose init chain re-entered the
    still-partial core.metrics (``get_metric``) and died; the walk-discovery
    swallowed the error, leaving a silently half-populated loss registry
    (151/200, ``l1``/``l2`` unresolvable) for the whole process. The root
    conftest triggers exactly this order via ``tests.fixtures.builders``.
    """
    import os
    import subprocess
    import sys

    code = (
        "import spectramr.core.metrics\n"
        "from spectramr.models.losses.elementary import resolve_elementary_loss\n"
        "for name in ('l1', 'l2', 'huber'):\n"
        "    resolve_elementary_loss(name)\n"
    )
    env = {**os.environ, "SPECTRAMR_SUPPRESS_CLINICAL_WARNING": "1"}
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
