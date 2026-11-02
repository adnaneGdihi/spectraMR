"""Autograd-graph contract for ``LossOutput`` and ``_stack_components`` (#1952).

``LossOutput.__post_init__`` used to "ensure gradients" with
``total.detach().requires_grad_(True)``. That does not restore a graph -- it
*manufactures a leaf*: ``requires_grad=True`` with ``grad_fn is None``.
``backward()`` on it succeeds, every ``param.grad`` stays ``None``, and the run
checkpoints having learned nothing. The repair was deleted; these tests pin the
resulting contract from both sides.
"""

import logging

import pytest
import torch

from spectramr.domain.exceptions import ConfigurationError
from spectramr.models.losses.computers.base import BaseLossComputer, LossOutput


class _Probe(BaseLossComputer):
    """Minimal concrete computer, to drive the real ``_stack_components``."""

    def _initialize_losses(self):  # pragma: no cover - abstract satisfaction
        pass

    def compute(self, *args, **kwargs):  # pragma: no cover - abstract satisfaction
        pass

    def _get_loss_weight(self, name, epoch=0, iteration=0, **kwargs):
        return self._weights.get(name, 1.0)


@pytest.fixture
def probe():
    p = _Probe.__new__(_Probe)
    torch.nn.Module.__init__(p)
    p.device = torch.device("cpu")
    p.config = None  # no losses block: warmup_iterations is 0, so every step is post-warm-up
    p._weights = {}
    return p


@pytest.fixture
def param():
    return torch.nn.Parameter(torch.tensor([1.0, 2.0]))


class TestLossOutputDoesNotManufactureALeaf:
    def test_a_grad_free_total_is_left_alone(self):
        """The deleted repair, planted.

        Before the fix this asserted the opposite: ``requires_grad`` came back
        ``True`` with ``grad_fn is None``.
        """
        out = LossOutput(total=torch.tensor(0.0))
        assert out.total.requires_grad is False
        assert out.total.grad_fn is None

    def test_validation_output_constructs_without_raising(self, param):
        """A prediction produced under ``no_grad`` is legitimate at validation.

        This is why the check does NOT live in ``__post_init__``: the *prediction*
        is built under ``no_grad``, which propagates to everything derived from it
        no matter where that arithmetic sits, so a validation loss legitimately
        arrives with ``requires_grad=False``. Modelled on ``guided_sr_strategy``,
        whose generator really does run inside the ``no_grad`` block.
        """
        target = torch.zeros(2)
        with torch.no_grad():
            prediction = param * 2
        loss = ((prediction - target) ** 2).mean()  # built OUTSIDE no_grad
        assert loss.requires_grad is False

        out = LossOutput(total=loss)  # must not raise
        assert out.total.requires_grad is False

    def test_detach_actually_detaches(self, param):
        """``LossOutput.detach()`` did not detach.

        ``__post_init__`` ran on the new instance and re-attached
        ``requires_grad=True`` to the tensor ``detach()`` had just freed, so every
        "detached for logging" copy was a leaf holding a live flag.
        """
        connected = (param * 2).sum()
        out = LossOutput(total=connected, components={"a": connected})
        detached = out.detach()
        assert detached.total.requires_grad is False
        assert detached.total.grad_fn is None

    def test_a_connected_total_is_untouched(self, param):
        connected = (param * 2).sum()
        out = LossOutput(total=connected)
        assert out.total.grad_fn is not None


class TestStackComponentsExitStates:
    """Which exits hand back a leaf -- i.e. what the backward guard will see.

    Characterized by execution rather than by reading: the distinction between
    the NaN-collapse exit (a leaf) and the dead-loss exit (grad-connected) is the
    difference between a guard that fires and one that does not, and both log the
    same "the model is NOT training" ERROR.
    """

    def test_empty_components_returns_a_leaf(self, probe):
        total = probe._stack_components({})
        assert total.grad_fn is None
        assert total.requires_grad is True

    def test_all_components_nan_returns_a_leaf(self, probe, param, caplog):
        """Every component skipped => ``total`` is never reassigned.

        The accumulator itself is the answer, and it is a leaf. This exit also
        launders a NaN into a *finite* 0.0, so the executor's non-finite guard
        never sees it and a ``GradScaler`` never reduces its scale.
        """
        nan = torch.tensor(float("nan")) * param.sum()
        probe._weights = {"a": 1.0, "b": 1.0}
        with caplog.at_level(logging.ERROR):
            total = probe._stack_components({"a": nan, "b": nan})
        assert total.grad_fn is None
        assert torch.isfinite(total).all()  # the NaN was laundered into 0.0
        assert any("SILENT NaN COLLAPSE" in r.getMessage() for r in caplog.records)

    def test_all_weights_zero_after_warmup_raises(self, probe, param):
        """#177: a training step with no weighted term raises, never a zero gradient."""
        connected = (param * 2).sum()
        probe._weights = {"a": 0.0, "b": 0.0}
        with pytest.raises(ConfigurationError, match="objective is zero"):
            probe._stack_components({"a": connected, "b": connected}, iteration=2000)

    def test_all_weights_zero_without_grad_is_logged_not_raised(self, probe, param, caplog):
        """A grad-free call (validation under no_grad) keeps the logged dead-loss exit."""
        connected = (param * 2).sum()
        probe._weights = {"a": 0.0, "b": 0.0}
        with torch.no_grad(), caplog.at_level(logging.ERROR):
            total = probe._stack_components({"a": connected, "b": connected}, iteration=2000)
        assert float(total) == 0.0
        assert any("DEAD LOSS" in r.getMessage() for r in caplog.records)

    def test_one_surviving_component_stays_grad_connected(self, probe, param):
        connected = (param * 2).sum()
        nan = torch.tensor(float("nan")) * param.sum()
        probe._weights = {"a": 1.0, "b": 1.0}
        total = probe._stack_components({"a": nan, "b": connected})
        assert total.grad_fn is not None
