"""The validation loss call forwards the rung mask, so a null-band term can run.

`_compute_losses_impl` forwards `mask` to `UnifiedDiffusionLossComputer.compute`;
`_compute_validation_metrics` built its own `loss_kwargs` and forwarded only
`smaps`. A loss that needs the acquisition support therefore trained for a whole
validation interval and then died at the first validation event -- reachable from
YAML, fatal at runtime.

`null_space_content` is the term that makes this matter: it is the only
registered loss that supervises the *unobserved* bins, which is the band the
kspace_filling cohort reproduces as zero-filled. It raises on `mask=None` by
design (pitfall 9) rather than scoring an empty null space, so the gap was a
crash waiting on whoever enabled it, not a silent zero.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.losses.computers.unified_diffusion_reconstruction import (  # noqa: E402
    _call_safe_loss,
)
from spectramr.models.losses.registry import LossRegistry  # noqa: E402

REPO = Path(__file__).resolve().parents[5]
STRATEGY = REPO / "src/spectramr/infrastructure/training/strategies/diffusion.py"

#: What the reference arm (`experiment_11_attention_none`) enables today. The
#: mask must be inert on every one of them, or this change moves published
#: numbers rather than unblocking a new term.
COHORT_ENABLED = ("complex_l1", "log_spectral", "sobolev_kspace", "complex_spatial_gradient")


def _pair(shape=(2, 16, 16, 16)):
    torch.manual_seed(0)
    return torch.randn(*shape), torch.randn(*shape)


def _mask(shape=(2, 1, 16, 16)):
    torch.manual_seed(1)
    return (torch.rand(*shape) > 0.5).float()


# ── the contract the gap violated ─────────────────────────────────────────────
def test_null_space_content_runs_when_the_mask_is_forwarded():
    loss = LossRegistry.create("null_space_content")
    pred, target = _pair()
    value = _call_safe_loss(loss, pred, target, smaps=None, mask=_mask())
    assert torch.is_tensor(value) and torch.isfinite(value)


def test_null_space_content_raises_when_it_is_not():
    """The pre-fix validation kwargs, verbatim: `smaps` only."""
    loss = LossRegistry.create("null_space_content")
    pred, target = _pair()
    with pytest.raises(ValueError, match="requires the sampling mask"):
        _call_safe_loss(loss, pred, target, smaps=None)


def test_an_absent_rung_mask_still_raises_rather_than_scoring_an_empty_null_space():
    """`_rung_mask` is None on a rung whose generation never reached masking.

    Forwarding it as None must reach the loss's own refusal, not be silently
    dropped into a default that scores the whole grid as unobserved.
    """
    loss = LossRegistry.create("null_space_content")
    pred, target = _pair()
    with pytest.raises(ValueError, match="requires the sampling mask"):
        _call_safe_loss(loss, pred, target, smaps=None, mask=None)


def test_the_null_band_term_only_scores_unobserved_bins():
    """A prediction wrong only on observed bins costs nothing here."""
    loss = LossRegistry.create("null_space_content")
    _, target = _pair()
    mask = _mask()
    pred = target + 10.0 * mask  # error lives entirely inside the support
    assert float(_call_safe_loss(loss, pred, target, mask=mask)) == pytest.approx(0.0, abs=1e-6)


# ── the mask must not move what the cohort already reports ────────────────────
@pytest.mark.parametrize("name", COHORT_ENABLED)
def test_a_signature_filtered_loss_never_sees_the_mask(name):
    """`_call_safe_loss` drops it; these four would raise TypeError if it did not."""
    loss = LossRegistry.create(name)
    pred, target = _pair()
    without = _call_safe_loss(loss, pred, target)
    with_mask = _call_safe_loss(loss, pred, target, mask=_mask())
    assert torch.equal(without.detach(), with_mask.detach())


def test_a_varkwargs_loss_that_ignores_the_mask_is_bit_identical():
    """`hfen` accepts **kwargs, so it RECEIVES the mask -- and must not use it.

    This is the blast radius of adding a key to the shared kwargs dict: 114
    registered losses declare **kwargs. `hfen` is the one this cohort enables,
    and it is also the arm's `best_metric_name`, so a shift here would move
    checkpoint selection.
    """
    loss = LossRegistry.create("hfen")
    pred, target = _pair()
    without = _call_safe_loss(loss, pred, target)
    with_mask = _call_safe_loss(loss, pred, target, mask=_mask())
    assert torch.equal(without.detach(), with_mask.detach())


# ── wiring: the key is in THAT dict, not merely somewhere in the file ─────────
def _loss_kwargs_keys_in(method_name: str) -> set[str]:
    """Keys of the `loss_kwargs` dict literal inside one method.

    Scoped to the method deliberately. A file-wide search passes when the key is
    added to an unrelated dict, which is the decoy #2069's wiring check was
    written to survive.
    """
    tree = ast.parse(STRATEGY.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != method_name:
            continue
        for stmt in ast.walk(node):
            target = None
            if isinstance(stmt, ast.AnnAssign):
                target = stmt.target
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
            if not isinstance(target, ast.Name) or target.id != "loss_kwargs":
                continue
            if isinstance(stmt.value, ast.Dict):
                return {k.value for k in stmt.value.keys if isinstance(k, ast.Constant)}
    raise AssertionError(f"no `loss_kwargs` dict literal found in {method_name}")


def test_the_validation_loss_call_forwards_the_mask():
    assert "mask" in _loss_kwargs_keys_in("_compute_validation_metrics")


def test_the_validation_loss_call_still_forwards_smaps():
    """The kwarg that was already there; the fix adds, never replaces."""
    assert "smaps" in _loss_kwargs_keys_in("_compute_validation_metrics")


def test_the_mask_forwarded_is_this_rungs_declared_support():
    """`_rung_mask`, not a re-derivation from the measurement's non-zeros.

    The measurement-aware metrics a few lines below already read `_rung_mask`
    for exactly this reason: a 195-of-256 readout window makes inferred support
    and declared support different masks.
    """
    tree = ast.parse(STRATEGY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_validation_metrics":
            src = ast.get_source_segment(STRATEGY.read_text(), node) or ""
            assert '"mask": _rung_mask' in src
            return
    raise AssertionError("_compute_validation_metrics not found")


def test_the_training_path_already_forwarded_it():
    """Pins the half that was correct, so the two cannot diverge again silently."""
    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

    src = inspect.getsource(DiffusionTrainingStrategy._compute_losses_impl)
    assert "mask=mask" in src
