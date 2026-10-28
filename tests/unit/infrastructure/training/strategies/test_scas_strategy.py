"""Guard tests for ``SCASStrategy`` (finding D11).

SCAS *is* the scout-conditioned acquisition mechanism. ``_prepare_generator_inputs``
used to ``return inputs`` unchanged whenever ``batch['scout']`` was absent, so
the hypernet was never built, never joined ``opt_g``, and neither ``scas_mask``
nor ``scas_logits`` was ever written -- which in turn made the density penalty in
``_compute_losses_impl`` silently no-op. The arm degenerated to plain LOUPE under
the SCAS name (pitfall #16).

Nothing could produce ``scout``: ``ScoutAcquisitionTransform`` had no way to be
constructed until the transform registry landed. The guard now raises and names
the producer.
"""

from __future__ import annotations

import pytest

from spectramr.infrastructure.training.strategies.scas_strategy import SCASStrategy


def _strategy(monkeypatch):
    """A SCASStrategy shell; the parent hook is stubbed to isolate the guard."""
    s = object.__new__(SCASStrategy)
    monkeypatch.setattr(
        "spectramr.infrastructure.training.strategies.scas_strategy.LOUPEStrategy._prepare_generator_inputs",
        lambda self, batch, *a, **k: {"sentinel": True},
        raising=False,
    )
    return s


class TestScoutGuard:
    def test_missing_scout_raises(self, monkeypatch):
        s = _strategy(monkeypatch)
        with pytest.raises(ValueError) as exc:
            s._prepare_generator_inputs({"input": 1, "target": 2})
        assert "scout" in str(exc.value)

    def test_message_names_the_producing_transform(self, monkeypatch):
        """The error must be actionable: it is the only place that says how."""
        s = _strategy(monkeypatch)
        with pytest.raises(ValueError) as exc:
            s._prepare_generator_inputs({"input": 1})
        msg = str(exc.value)
        assert "scout_acquisition" in msg
        assert "data.processing.transforms" in msg or "transforms:" in msg

    def test_message_lists_what_the_batch_actually_supplied(self, monkeypatch):
        s = _strategy(monkeypatch)
        with pytest.raises(ValueError) as exc:
            s._prepare_generator_inputs({"input": 1, "target": 2})
        msg = str(exc.value)
        assert "input" in msg and "target" in msg

    def test_non_dict_batch_also_raises(self, monkeypatch):
        """Previously this returned the parent's inputs untouched."""
        s = _strategy(monkeypatch)
        with pytest.raises(ValueError):
            s._prepare_generator_inputs(object())

    def test_the_silent_passthrough_is_gone(self, monkeypatch):
        """Regression witness: the pre-fix code returned the parent dict.

        If the guard is ever softened back to ``return inputs`` this test is the
        one that fails.
        """
        s = _strategy(monkeypatch)
        with pytest.raises(ValueError):
            result = s._prepare_generator_inputs({"input": 1})
            assert result == {"sentinel": True}  # pragma: no cover


def test_scas_keeps_its_own_ddp_refusal_over_its_migrated_parent():
    """LOUPE's mask logits are an owned module now; SCAS's bare hypernet group is not."""
    from spectramr.infrastructure.distributed.strategy_compat import evaluate_parallel_compat
    from spectramr.infrastructure.training.strategies.loupe_strategy import LOUPEStrategy

    assert not evaluate_parallel_compat(LOUPEStrategy, "ddp").refused
    assert evaluate_parallel_compat(SCASStrategy, "ddp").refused
