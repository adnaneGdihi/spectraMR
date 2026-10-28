"""Regression tests for the Bloch-consistency mis-call in
``OneShotMultiParameterStrategy._compute_losses_impl``.

The Bloch signal-synthesis consistency term was called as
``self._bloch_loss(pred, observed, acquisition_params=acq)`` — passing a dict
as ``t1_ms`` — which raised a ``TypeError`` that was swallowed by a broad
``except Exception`` and logged as a warning. The advertised "physics anchor"
therefore never contributed (a facade, pitfall #16; warning-not-OK, #10).

The fix: call the loss with its real ``(t1, t2, pd, observed_contrasts,
acquisition_params)`` signature (via ``split_t1_t2_pd``), and fail loud when
``lambda>0`` but the required multi-contrast data is absent — no swallow.
"""

from __future__ import annotations

import inspect
import types

import pytest
import torch

from spectramr.infrastructure.training.strategies.multi_param_mapping_strategy import (
    OneShotMultiParameterStrategy,
)


def _make_strategy(gen, *, bloch_lambda: float, parameters=("t1", "t2", "pd")):
    s = object.__new__(OneShotMultiParameterStrategy)
    s.device = torch.device("cpu")
    s._log_sigmas = None
    s._mp_cfg = types.SimpleNamespace(
        parameters=list(parameters),
        bloch_consistency_lambda=bloch_lambda,
    )
    s.env = types.SimpleNamespace(generator=gen)
    return s


def _param_gen():
    return lambda x: {
        "t1": torch.full((1, 1, 4, 4), 800.0),
        "t2": torch.full((1, 1, 4, 4), 80.0),
        "pd": torch.ones(1, 1, 4, 4),
    }


class _RecordingGen:
    """Callable spy so a test can assert the generator was never reached."""

    def __init__(self, fn):
        self._fn = fn
        self.calls: list[tuple] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._fn(*args, **kwargs)


def test_no_swallow_in_source() -> None:
    src = inspect.getsource(OneShotMultiParameterStrategy._compute_losses_impl)
    # The broad swallow-to-warning is gone; the correct call is present.
    assert "except Exception" not in src
    assert 'logger.warning("Bloch consistency loss failed' not in src
    assert "split_t1_t2_pd(pred)" in src
    assert "observed_contrasts=observed" in src


def test_lambda_positive_without_data_fails_loud() -> None:
    # lambda>0 but no acquisition_params / observed_contrasts -> raise, not warn.
    s = _make_strategy(_param_gen(), bloch_lambda=0.5)
    with pytest.raises(ValueError, match="acquisition_params"):
        s._compute_losses_impl(
            torch.randn(1, 1, 4, 4),
            torch.randn(1, 1, 4, 4),
            epoch=0,
            batch={"observed_contrasts": torch.randn(1, 2, 4, 4)},  # no acq params
        )


def test_lambda_zero_skips_bloch_cleanly() -> None:
    # With lambda=0 the Bloch branch is skipped; loss_total is the (empty) task sum.
    s = _make_strategy(_param_gen(), bloch_lambda=0.0)
    out = s._compute_losses_impl(
        torch.randn(1, 1, 4, 4),
        torch.randn(1, 1, 4, 4),
        epoch=0,
        batch={"observed_contrasts": torch.randn(1, 2, 4, 4)},
    )
    assert "loss_bloch" not in out
    assert torch.isfinite(out["loss_total"])


def test_bloch_term_contributes_when_data_present() -> None:
    s = _make_strategy(_param_gen(), bloch_lambda=1.0)
    batch = {
        "observed_contrasts": torch.randn(1, 2, 4, 4),
        "acquisition_params": [
            {"TR_ms": 600.0, "TE_ms": 12.0, "FA_deg": 70.0},
            {"TR_ms": 4000.0, "TE_ms": 80.0, "FA_deg": 90.0},
        ],
    }
    out = s._compute_losses_impl(
        torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4), epoch=0, batch=batch
    )
    assert "loss_bloch" in out
    assert out["loss_bloch"].item() > 0.0
    assert torch.isfinite(out["loss_total"])


# ---------------------------------------------------------------------------
# A7: ``observed_contrasts`` is read then handed STRAIGHT to ``gen()``.
#
# Pre-fix, ``observed = batch.get("observed_contrasts") if isinstance(batch,
# dict) else input_batch`` meant:
#   * a dict batch missing the key passed ``None`` into the generator (an
#     opaque arity crash naming nothing), and
#   * a non-dict batch silently substituted a DIFFERENT tensor, fitting
#     T1/T2/PD against the wrong observations.
#
# The actionable ``ValueError`` further down is gated on
# ``bloch_consistency_lambda > 0``, so a lambda-0 arm could never reach it —
# it just ran ``gen(None)`` and returned a zero loss with no error at all.
# Every test below therefore uses ``bloch_lambda=0.0``, which is the case that
# genuinely discriminates the fix, and spies on the generator: the guard now
# raises BEFORE the forward pass.
# ---------------------------------------------------------------------------


def test_missing_observed_contrasts_raises_before_the_generator() -> None:
    gen = _RecordingGen(_param_gen())
    s = _make_strategy(gen, bloch_lambda=0.0)

    with pytest.raises(ValueError) as exc:
        s._compute_losses_impl(
            torch.randn(1, 1, 4, 4),
            torch.randn(1, 1, 4, 4),
            epoch=0,
            batch={"image": torch.randn(1, 1, 4, 4), "target": torch.randn(1, 1, 4, 4)},
        )

    assert gen.calls == [], "the generator must never be called with None"
    msg = str(exc.value)
    assert "observed_contrasts" in msg
    # Text unique to the new pre-generator guard, so this cannot be satisfied
    # by the older lambda-gated message (which also says "observed_contrasts").
    assert "substituting the input tensor" in msg
    # The debugging affordance: what the batch actually supplied. Asserted on
    # the captured string, not via ``match=`` — the list renders as
    # ``['image', 'target']`` and ``[...]`` is a regex character class.
    assert "'image'" in msg
    assert "'target'" in msg


def test_non_dict_batch_raises_instead_of_substituting_input_batch() -> None:
    # A one-element tensor: ``batch = kwargs.get("batch") or {}`` calls
    # ``bool()``, which a multi-element tensor would reject before the guard.
    gen = _RecordingGen(_param_gen())
    s = _make_strategy(gen, bloch_lambda=0.0)

    with pytest.raises(ValueError) as exc:
        s._compute_losses_impl(
            torch.randn(1, 1, 4, 4),
            torch.randn(1, 1, 4, 4),
            epoch=0,
            batch=torch.ones(1),
        )

    assert gen.calls == []
    msg = str(exc.value)
    assert "observed_contrasts" in msg
    # Non-dict batches report the type rather than a key list.
    assert "'Tensor'" in msg


def test_absent_batch_kwarg_reports_an_empty_supplied_list() -> None:
    gen = _RecordingGen(_param_gen())
    s = _make_strategy(gen, bloch_lambda=0.0)

    with pytest.raises(ValueError) as exc:
        s._compute_losses_impl(
            torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4), epoch=0
        )

    assert gen.calls == []
    assert "[]" in str(exc.value)


def test_present_observed_contrasts_reaches_the_generator_unchanged() -> None:
    # Happy path: the guard does not fire and the generator receives the real
    # observations object, not a substitute.
    gen = _RecordingGen(_param_gen())
    s = _make_strategy(gen, bloch_lambda=0.0)
    observed = torch.randn(1, 2, 4, 4)

    out = s._compute_losses_impl(
        torch.randn(1, 1, 4, 4),
        torch.randn(1, 1, 4, 4),
        epoch=0,
        batch={"observed_contrasts": observed},
    )

    assert len(gen.calls) == 1
    assert gen.calls[0][0][0] is observed
    assert torch.isfinite(out["loss_total"])


def test_the_input_batch_substitution_is_gone_from_source() -> None:
    """The other half of A7, and a DIFFERENT property from "a missing key raises".

    ``observed = batch.get(...) if isinstance(batch, dict) else input_batch``
    silently estimated the parameter maps from a single contrast when the batch
    was not a dict -- a different objective under the same arm name, with no
    error to notice. A raise-based test cannot see a substitution that was
    removed; only the source can.
    """
    import inspect

    src = inspect.getsource(OneShotMultiParameterStrategy._compute_losses_impl)
    assert "else input_batch" not in src


def _built_via_init(monkeypatch, cls=OneShotMultiParameterStrategy):
    """The real ``__init__`` body over a stub reconstruction parent, uncertainty weighting on."""
    from spectramr.config.schemas.training.base import MultiParameterTrainingConfigSchema
    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    def _stub_init(self, *a, **k):
        self.config = types.SimpleNamespace(
            parallel=None,
            training=types.SimpleNamespace(
                spin_sde=None,
                multi_parameter=MultiParameterTrainingConfigSchema(
                    parameters=["t1", "t2"],
                    uncertainty_weighting=True,
                    bloch_consistency_lambda=0.0,
                ),
            ),
        )
        self.device = torch.device("cpu")
        self.env = types.SimpleNamespace(
            opt_g=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
            generator=lambda x: {"t1": x[:, :1] * 2.0, "t2": x[:, 1:] * 3.0},
        )
        self._owned_modules = {}
        self._owned_wrappers = {}

    monkeypatch.setattr(ReconstructionTrainingStrategy, "__init__", _stub_init)
    return cls()


def test_log_sigmas_are_an_owned_module_with_an_unchanged_checkpoint_key(monkeypatch) -> None:
    """#2629: the holder is registered for DDP but is not an attribute, so the
    checkpoint still carries ``params._log_sigmas`` and an old checkpoint loads."""
    s = _built_via_init(monkeypatch)
    assert s.owned_modules()["_log_sigmas"].value is s._log_sigmas
    state = s.strategy_state_dict()
    assert sorted(state["params"]) == ["_log_sigmas"] and state["modules"] == {}

    x = torch.randn(1, 2, 4, 4)
    batch = {"observed_contrasts": x, "t1": torch.zeros(1, 1, 4, 4), "t2": torch.ones(1, 1, 4, 4)}
    s._compute_losses_impl(x, x, 0, batch=batch)["loss_total"].backward()
    assert s._log_sigmas.grad is not None and s._log_sigmas.grad.abs().sum() > 0

    fresh = _built_via_init(monkeypatch)
    fresh.load_strategy_state_dict({"params": {"_log_sigmas": torch.tensor([0.5, -0.5])}})
    assert torch.equal(fresh._log_sigmas.detach(), torch.tensor([0.5, -0.5]))
    assert fresh.owned_modules()["_log_sigmas"]() is fresh._log_sigmas
