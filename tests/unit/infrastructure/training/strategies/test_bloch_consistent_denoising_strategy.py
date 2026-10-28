"""Regression test for the 2026-06 strategy-audit finding SPB-003 on
``bloch_consistent_denoising_strategy``.

Finding SPB-003 [error / silent_fallback]:
    ``__init__`` wrapped the ``BlochSignalSynthesisConsistencyLoss`` import in a
    broad ``except Exception`` that logged a warning and left ``self._bloch =
    None``. ``_compute_losses_impl`` then silently degraded to the plain
    ``ReconstructionTrainingStrategy`` losses — dropping the strategy's
    *defining* physics objective without any error or config change (NN#3 +
    pitfall #10). The fix narrows the catch to ``ImportError`` and re-raises a
    ``RuntimeError`` so the misconfiguration fails fast.

The heavy base ``__init__`` is stubbed so the test isolates the import branch.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from spectramr.infrastructure.training.strategies.bloch_consistent_denoising_strategy import (
    BlochConsistentDenoisingStrategy,
)
from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)


def _stub_base_init(monkeypatch):
    """No-op the heavy base ``__init__`` and wire only ``self.device``."""

    def _fake_init(self, *args, **kwargs):
        self.device = torch.device("cpu")

    monkeypatch.setattr(ReconstructionTrainingStrategy, "__init__", _fake_init, raising=True)


class TestBlochImportFailureRaises:
    def test_import_failure_raises_runtimeerror(self, monkeypatch):
        _stub_base_init(monkeypatch)

        import builtins

        real_import = builtins.__import__
        target = "spectramr.models.losses.bloch_signal_synthesis_consistency_loss"

        def _boom(name, *args, **kwargs):
            if name == target:
                raise ImportError("simulated missing BlochSignalSynthesisConsistencyLoss")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _boom)

        with pytest.raises(RuntimeError, match="BlochConsistentDenoisingStrategy requires"):
            BlochConsistentDenoisingStrategy()

    def test_import_failure_chains_original_importerror(self, monkeypatch):
        _stub_base_init(monkeypatch)

        import builtins

        real_import = builtins.__import__
        target = "spectramr.models.losses.bloch_signal_synthesis_consistency_loss"

        def _boom(name, *args, **kwargs):
            if name == target:
                raise ImportError("simulated missing module")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _boom)

        with pytest.raises(RuntimeError) as exc_info:
            BlochConsistentDenoisingStrategy()
        # The narrowed ImportError must be chained (``from e``) so the root
        # cause is preserved — not swallowed to a warning.
        assert isinstance(exc_info.value.__cause__, ImportError)

    def test_successful_import_sets_bloch(self, monkeypatch):
        # When the import succeeds, ``_bloch`` is populated and no error fires.
        _stub_base_init(monkeypatch)
        strat = BlochConsistentDenoisingStrategy()
        assert strat._bloch is not None


class TestSourceNoSilentFallback:
    def test_init_no_longer_swallows_to_none(self):
        src = inspect.getsource(BlochConsistentDenoisingStrategy.__init__)
        # The broad warn-and-disable fallback is gone.
        assert "except Exception" not in src
        assert "logger.warning" not in src
        # The narrowed catch + re-raise contract is present.
        assert "except ImportError as e" in src
        assert "raise RuntimeError" in src


class TestComputeLossesCallSite:
    """The `_compute_losses_impl` Bloch call was mis-called (2-arg call to a
    5-arg forward), crashing whenever the branch ran. It must now call the loss
    with the real signature, or fail loud with an actionable message."""

    def _strategy_with_gen(self, monkeypatch, gen):
        import types

        _stub_base_init(monkeypatch)
        s = BlochConsistentDenoisingStrategy()
        s.env = types.SimpleNamespace(generator=gen)
        s._resolve_legacy_batch = lambda input_batch, kwargs: kwargs.get("batch", {})
        return s

    def test_absent_bloch_loss_raises_rather_than_running_a_plain_denoiser(self, monkeypatch):
        """``self._bloch is None`` used to drop to the parent objective.

        That trained a plain denoiser under a Bloch-consistent arm name and
        reported success, which is facade #16: the term the strategy exists to
        impose was simply absent.
        """
        gen = lambda x: {"t1": torch.ones(1, 1, 4, 4)}  # noqa: E731
        s = self._strategy_with_gen(monkeypatch, gen)
        s._bloch = None
        with pytest.raises(RuntimeError, match="defining objective"):
            s._compute_losses_impl(batch={"observed_contrasts": torch.randn(1, 2, 4, 4)})

    def test_non_dict_generator_output_fails_loud(self, monkeypatch):
        # A plain image denoiser returns a single tensor -> the forward-synthesis
        # loss cannot consume it -> NotImplementedError (no cryptic TypeError,
        # no silent degrade).
        gen = lambda x: torch.randn(1, 3, 4, 4)  # noqa: E731
        s = self._strategy_with_gen(monkeypatch, gen)
        with pytest.raises(NotImplementedError, match="parameter-map dict"):
            s._compute_losses_impl(batch={"observed_contrasts": torch.randn(1, 3, 4, 4)})

    def test_missing_acquisition_params_fails_loud(self, monkeypatch):
        gen = lambda x: {  # noqa: E731
            "t1": torch.rand(1, 1, 4, 4) * 800 + 200,
            "t2": torch.rand(1, 1, 4, 4) * 80 + 20,
            "pd": torch.rand(1, 1, 4, 4),
        }
        s = self._strategy_with_gen(monkeypatch, gen)
        with pytest.raises(ValueError, match="acquisition_params"):
            s._compute_losses_impl(batch={"observed_contrasts": torch.randn(1, 2, 4, 4)})

    def test_valid_param_maps_compute_finite_bloch_loss(self, monkeypatch):
        gen = lambda x: {  # noqa: E731
            "t1": torch.full((1, 1, 4, 4), 800.0),
            "t2": torch.full((1, 1, 4, 4), 80.0),
            "pd": torch.ones(1, 1, 4, 4),
        }
        s = self._strategy_with_gen(monkeypatch, gen)
        batch = {
            "observed_contrasts": torch.randn(1, 2, 4, 4),
            "acquisition_params": [
                {"TR_ms": 600.0, "TE_ms": 12.0, "FA_deg": 70.0},
                {"TR_ms": 4000.0, "TE_ms": 80.0, "FA_deg": 90.0},
            ],
        }
        out = s._compute_losses_impl(batch=batch)
        assert torch.isfinite(out["loss_total"])
        assert "loss_bloch" in out

    def test_source_calls_loss_with_five_args_not_two(self):
        src = inspect.getsource(BlochConsistentDenoisingStrategy._compute_losses_impl)
        # The old broken 2-arg call is gone; the split helper + full signature is used.
        assert "self._bloch(denoised, observed)" not in src
        assert "split_t1_t2_pd(pred)" in src
        assert "observed_contrasts=observed" in src


class _RecordingGen:
    """Callable spy so a test can assert the generator was never reached."""

    def __init__(self, fn):
        self._fn = fn
        self.calls: list[tuple] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._fn(*args, **kwargs)


def _param_map_gen() -> _RecordingGen:
    return _RecordingGen(
        lambda x: {
            "t1": torch.full((1, 1, 4, 4), 800.0),
            "t2": torch.full((1, 1, 4, 4), 80.0),
            "pd": torch.ones(1, 1, 4, 4),
        }
    )


class TestObservedContrastsGuard:
    """A7: ``observed`` was read and handed straight to ``gen()``.

    Pre-fix, a dict batch missing ``observed_contrasts`` passed ``None`` into
    the generator, and a non-dict batch silently substituted the whole batch
    object. Both downstream errors (``NotImplementedError`` on a non-dict
    prediction, ``ValueError`` on missing acquisition params) only fire *after*
    ``gen`` has already returned, so neither could ever describe the real
    problem. The guard now raises BEFORE the forward pass.

    The load-bearing assertion in each test is ``gen.calls == []`` plus text
    unique to the new guard — the pre-fix code also ended in a ``ValueError``
    mentioning ``observed_contrasts``, so a bare ``pytest.raises(ValueError,
    match="observed_contrasts")`` would have passed against the defect.
    """

    def _strategy_with_gen(self, monkeypatch, gen):
        import types

        _stub_base_init(monkeypatch)
        s = BlochConsistentDenoisingStrategy()
        s.env = types.SimpleNamespace(generator=gen)
        s._resolve_legacy_batch = lambda input_batch, kwargs: kwargs.get("batch", {})
        return s

    def test_dict_batch_missing_key_raises_before_the_generator(self, monkeypatch):
        gen = _param_map_gen()
        s = self._strategy_with_gen(monkeypatch, gen)

        with pytest.raises(ValueError) as exc:
            s._compute_losses_impl(
                batch={
                    "image": torch.randn(1, 2, 4, 4),
                    "target": torch.randn(1, 2, 4, 4),
                }
            )

        assert gen.calls == [], "the generator must never be called with None"
        msg = str(exc.value)
        assert "observed_contrasts" in msg
        # Unique to the new pre-generator guard.
        assert "Supply a qMRI multi-contrast dataset that emits the key" in msg
        # The debugging affordance: what the batch actually supplied. Asserted
        # on the captured string, not via ``match=`` — the list renders as
        # ``['image', 'target']`` and ``[...]`` is a regex character class.
        assert "'image'" in msg
        assert "'target'" in msg

    def test_non_dict_batch_raises_instead_of_substituting_the_batch(self, monkeypatch):
        gen = _param_map_gen()
        s = self._strategy_with_gen(monkeypatch, gen)

        with pytest.raises(ValueError) as exc:
            s._compute_losses_impl(batch=torch.ones(1))

        assert gen.calls == []
        msg = str(exc.value)
        assert "observed_contrasts" in msg
        # Non-dict batches report the type rather than a key list.
        assert "'Tensor'" in msg

    def test_absent_batch_kwarg_reports_an_empty_supplied_list(self, monkeypatch):
        gen = _param_map_gen()
        s = self._strategy_with_gen(monkeypatch, gen)

        with pytest.raises(ValueError) as exc:
            s._compute_losses_impl()

        assert gen.calls == []
        assert "[]" in str(exc.value)

    def test_present_key_reaches_the_generator_unchanged(self, monkeypatch):
        # Happy path: the guard does not fire, the generator receives the real
        # observations object, and the Bloch objective is computed from it.
        gen = _param_map_gen()
        s = self._strategy_with_gen(monkeypatch, gen)
        observed = torch.randn(1, 2, 4, 4)
        batch = {
            "observed_contrasts": observed,
            "acquisition_params": [
                {"TR_ms": 600.0, "TE_ms": 12.0, "FA_deg": 70.0},
                {"TR_ms": 4000.0, "TE_ms": 80.0, "FA_deg": 90.0},
            ],
        }

        out = s._compute_losses_impl(batch=batch)

        assert len(gen.calls) == 1
        assert gen.calls[0][0][0] is observed
        assert torch.isfinite(out["loss_total"])


def test_it_declares_its_own_loss_ownership() -> None:
    """Issue #1918: its hook overrides the parent's without calling super() or the fold.

    Read off ``__dict__``, never the inherited value: this class sits under
    ``ReconstructionTrainingStrategy``, whose ``folds_image_losses = True`` is truthful for ITSELF and
    becomes a lie the moment a subclass replaces ``_compute_losses_impl``. An
    inherited True makes the audit's ``image_losses_reach_the_objective`` witness
    PASS every declared ``losses.image_losses`` entry on this strategy's arms
    while the training step discards them.
    """
    assert BlochConsistentDenoisingStrategy.__dict__["folds_image_losses"] is False
    assert BlochConsistentDenoisingStrategy.__dict__["inline_losses"] == frozenset()
