"""There is one dispatcher for "which inference strategy does this config want".

``infrastructure/inference/__init__.py`` exported a second
``create_inference_strategy`` beside ``InferenceStrategyFactory``. It had zero
callers anywhere in the tree, mapped ten training modes to the factory's
fifteen, and detected cold diffusion by a different rule — a drifted copy of a
live dispatcher is worse than no copy, because the next reader cannot tell which
one decides (non-negotiable 17).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_the_dead_dispatcher_is_gone():
    import spectramr.infrastructure.inference as pkg

    assert not hasattr(pkg, "create_inference_strategy")
    assert "create_inference_strategy" not in pkg.__all__


def test_the_live_dispatcher_is_the_one_the_pipeline_calls():
    import inspect

    from spectramr.pipelines import infer

    assert "InferenceStrategyFactory" in inspect.getsource(infer)


def test_the_live_dispatcher_refuses_an_unknown_type():
    """The surviving owner must fail loud, or deleting the other one lost a guard."""
    import torch

    from spectramr.infrastructure.inference.inference_factory import (
        InferenceStrategyFactory,
    )

    with pytest.raises(ValueError, match="Unknown inference strategy_type"):
        InferenceStrategyFactory.create(
            torch.nn.Identity(), torch.device("cpu"), {}, strategy_type="not_a_strategy"
        )
