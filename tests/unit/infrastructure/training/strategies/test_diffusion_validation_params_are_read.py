"""Validation helpers declare only parameters they read (#1229).

``_generate_validation_prediction`` required a ``scale_factor`` it never read,
and ``_maybe_hallucination_metrics`` took one only to forward it. A reader
tracing denormalization took it to mean the sampler denormalized its output.
``_compute_validation_metrics`` is the one owner of that inverse. The detector
below reads a function's AST and lists parameters with no ``Name`` load in the
body. It is applied to ``scale_factor`` only here; the other unread parameters
it finds on these helpers are tracked separately.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy


def _unread_parameters(func) -> list[str]:
    node = ast.parse(textwrap.dedent(inspect.getsource(func))).body[0]
    params = [a.arg for a in (*node.args.args, *node.args.kwonlyargs) if a.arg != "self"]
    loads = {
        n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    return [p for p in params if p not in loads]


@pytest.mark.parametrize(
    "name", ["_generate_validation_prediction", "_maybe_hallucination_metrics"]
)
def test_no_scale_factor_is_declared_and_left_unread(name):
    func = getattr(DiffusionTrainingStrategy, name)
    assert "scale_factor" not in inspect.signature(func).parameters
    assert "scale_factor" not in _unread_parameters(func)


def test_the_denormalization_owner_still_takes_the_scale():
    """The parameter moved nowhere: the metrics step reads it, and keeps it."""
    params = inspect.signature(DiffusionTrainingStrategy._compute_validation_metrics).parameters
    assert "scale_factor" in params
    assert "scale_factor" not in _unread_parameters(
        DiffusionTrainingStrategy._compute_validation_metrics
    )


def _planted(self, input_batch, kwargs, scale_factor, *, start_timestep=None):
    return input_batch, kwargs, start_timestep


def _planted_forwarder(self, kwargs, scale_factor):
    """A parameter passed on by keyword is still read; one never named is not."""
    return self._next(kwargs=kwargs)


def test_the_detector_is_red_on_an_unread_positional():
    assert _unread_parameters(_planted) == ["scale_factor"]


def test_the_detector_is_red_on_a_forwarder_that_drops_it():
    assert _unread_parameters(_planted_forwarder) == ["scale_factor"]
