"""Regression tests: FederatedDPConformalULFStrategy actually wires DP-SGD.

Bugs these tests pin (see the strategy module docstring):

1. **Unwired privacy schema (pitfall #15).** ``TrainingSettings`` carried NO
   ``privacy`` field, so the strategy's ``getattr(config, "privacy", None)``
   always returned ``None`` and every DP knob collapsed to a hardcoded
   literal (``noise_multiplier=1.1`` ...). A YAML-set
   ``privacy.noise_multiplier`` had zero effect. The fix adds a typed
   :class:`PrivacyConfig` mounted on ``TrainingSettings``; these tests build a
   REAL ``PrivacyConfig`` and assert the configured value reaches
   ``self._noise_multiplier`` (would fail pre-fix: the literal 1.1 was used).
2. **Swallowed DP-config failure (pitfall #10).** ``configure_dp``'s
   ``ValueError`` on an out-of-range knob was caught and downgraded to a
   warning, so a DP strategy silently trained with NO privacy. The fix lets
   it propagate; an absent privacy block now raises. Tested via
   ``pytest.raises``.
3. **Honest sample rate (pitfall #15 / finding #4).** The sample rate was
   computed from a hardcoded ``dataset_size=1000``. The fix derives it from
   ``len(self.env.train_loader.dataset)``; tested against a real loader.
4. **Mis-named helper.** ``_per_sample_grad_norms`` returned per-PARAMETER
   norms, not per-sample. Renamed to ``_param_grad_norms`` with an honest
   contract; tested for length == #params and that the old name is gone.

Methods are exercised on a strategy built via ``object.__new__`` so no DI
container / CUDA / loss computer is required.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from spectramr.config.schemas.data import DataConfigSchema
from spectramr.config.schemas.training.privacy import PrivacyConfig
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.training.strategies.federated_dp_conformal_strategy import (
    FederatedDPConformalULFStrategy,
)
from tests.utils.config_block_stub import block_stub


def _bare_strategy() -> FederatedDPConformalULFStrategy:
    """Construct without running BaseTrainingStrategy.__init__ (no DI/CUDA)."""
    s = FederatedDPConformalULFStrategy.__new__(FederatedDPConformalULFStrategy)
    s.device = torch.device("cpu")
    return s


def _loader_env(num_samples: int) -> SimpleNamespace:
    """A minimal ``env`` exposing a real ``train_loader.dataset`` of length N."""
    ds = torch.utils.data.TensorDataset(torch.zeros(num_samples, 1))
    loader = torch.utils.data.DataLoader(ds, batch_size=1)
    return SimpleNamespace(train_loader=loader, generator=None)


def _data(batch_size: int = 8) -> DataConfigSchema:
    """A minimal valid DataConfigSchema (only batch_size matters here)."""
    return DataConfigSchema(batch_size=batch_size)


def _real_config(
    *,
    noise_multiplier: float = 0.5,
    max_grad_norm: float = 2.0,
    target_delta: float = 1e-6,
    batch_size: int = 8,
    privacy_dataset_size: int | None = None,
) -> SimpleNamespace:
    """A config whose ``privacy`` is a REAL ``PrivacyConfig`` (not a dict)."""
    return SimpleNamespace(
        privacy=PrivacyConfig(
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            target_delta=target_delta,
            dataset_size=privacy_dataset_size,
        ),
        data=_data(batch_size),
    )


# ---------------------------------------------------------------------------
# Finding #1: a YAML-set privacy.noise_multiplier reaches the strategy.
# ---------------------------------------------------------------------------


def test_yaml_privacy_knobs_reach_strategy_via_typed_schema() -> None:
    """The configured PrivacyConfig values flow through (NOT the 1.1 default).

    Pre-fix this failed because ``privacy`` was unwired (no field) and the
    strategy fell back to the literal ``noise_multiplier=1.1``.
    """
    s = _bare_strategy()
    s.env = _loader_env(num_samples=400)
    s.config = _real_config(
        noise_multiplier=0.7, max_grad_norm=3.5, target_delta=2e-6, batch_size=16
    )
    s._configure_from_settings()

    assert s.dp_enabled is True
    assert s._noise_multiplier == 0.7
    assert s._max_grad_norm == 3.5
    assert s._target_delta == 2e-6
    assert s._dp_batch_size == 16


def test_untyped_data_block_raises_rather_than_guessing_the_batch_size() -> None:
    """The DP sample rate must never be computed from a substituted default.

    ``bs`` feeds ``sample_rate = batch_size / dataset_size``, which feeds the
    accountant -- so a wrong batch size silently misreports epsilon. The old
    read was ``getattr(data, "batch_size", 4)``, which meant that once the key
    folded to ``data.loader.batch_size`` EVERY arm accounted at a batch size of
    4 regardless of config, with no symptom at all.

    This file already coerces a raw mapping for ``privacy``, so an untyped
    ``data`` is a shape that genuinely reaches here. Refusing is the only safe
    answer: there is no default that is honest about a privacy guarantee.
    """
    s = _bare_strategy()
    s.env = _loader_env(num_samples=400)
    s.config = SimpleNamespace(
        privacy=PrivacyConfig(
            noise_multiplier=0.5, max_grad_norm=2.0, target_delta=1e-6
        ),
        data={"batch_size": 8},  # a dict, not a DataConfigSchema
    )
    with pytest.raises(ConfigurationError, match=re.escape("data.loader.batch_size")):
        s._configure_from_settings()


def test_batch_size_is_read_from_the_canonical_loader_path() -> None:
    """A real config's declared batch size reaches the accountant."""
    s = _bare_strategy()
    s.env = _loader_env(num_samples=400)
    s.config = _real_config(batch_size=16)
    assert s.config.data.loader.batch_size == 16, "fixture drifted"
    s._configure_from_settings()
    assert s._dp_batch_size == 16


def test_dict_privacy_block_is_coerced_to_typed_schema() -> None:
    """A raw mapping for ``privacy`` is validated into PrivacyConfig, not ignored."""
    s = _bare_strategy()
    s.env = _loader_env(num_samples=200)
    s.config = SimpleNamespace(
        privacy={"noise_multiplier": 0.9, "max_grad_norm": 4.0, "target_delta": 1e-5},
        data=_data(4),
    )
    s._configure_from_settings()
    # The dict's values are honoured (pre-fix getattr would have used 1.1/1.0).
    assert s._noise_multiplier == 0.9
    assert s._max_grad_norm == 4.0


# ---------------------------------------------------------------------------
# Finding #2: configure_dp's ValueError must propagate (no silent downgrade).
# ---------------------------------------------------------------------------


def test_absent_privacy_block_raises_not_silent_defaults() -> None:
    """A DP strategy with no privacy block must FAIL LOUDLY, not default.

    Pre-fix the missing block fell back to hardcoded 1.1/1.0/1e-5 and trained
    anyway. The fix raises so a DP run can never silently lose its guarantee.
    """
    s = _bare_strategy()
    s.env = _loader_env(num_samples=100)
    s.config = SimpleNamespace(privacy=None, data=_data(4))
    with pytest.raises(ConfigurationError):
        s._configure_from_settings()


def test_out_of_range_knob_propagates_not_swallowed() -> None:
    """An out-of-range knob raises instead of disabling DP via a warning.

    ``target_delta == 1.0`` is rejected by PrivacyConfig (lt=1.0); bypassing
    the schema with ``model_construct`` to reach configure_dp's own guard, its
    ValueError must still propagate rather than being caught + downgraded.
    """
    s = _bare_strategy()
    s.env = _loader_env(num_samples=100)
    bad_priv = PrivacyConfig.model_construct(
        noise_multiplier=0.5, max_grad_norm=1.0, target_delta=1.0, dataset_size=None
    )
    s.config = SimpleNamespace(privacy=bad_priv, data=_data(4))
    with pytest.raises(ValueError):
        s._configure_from_settings()
    # DP must NOT have been silently enabled with a meaningless budget.
    assert getattr(s, "_dp_enabled", False) is False


def test_schema_rejects_out_of_range_at_construction() -> None:
    """The typed schema itself rejects a non-positive noise multiplier."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        PrivacyConfig(noise_multiplier=0.0, max_grad_norm=1.0, target_delta=1e-5)


# ---------------------------------------------------------------------------
# Finding #4: sample rate reflects the REAL dataset length, not 1000.
# ---------------------------------------------------------------------------


def test_sample_rate_uses_real_dataloader_length_not_1000() -> None:
    """q = batch_size / len(train_loader.dataset), NOT batch_size / 1000."""
    s = _bare_strategy()
    s.env = _loader_env(num_samples=250)  # real dataset of 250 samples
    s.config = _real_config(batch_size=10)
    s._configure_from_settings()

    # Honest: 10 / 250 = 0.04. The buggy literal would give 10 / 1000 = 0.01.
    assert s._dp_dataset_size == 250
    assert abs(s._dp_sample_rate - (10 / 250)) < 1e-9
    assert abs(s._sample_rate - 0.04) < 1e-9
    assert abs(s._sample_rate - 0.01) > 1e-6  # definitely not the 1000-literal rate


def test_sample_rate_falls_back_to_privacy_dataset_size_without_loader() -> None:
    """With no live loader, dataset_size comes from config, never a literal 1000."""
    s = _bare_strategy()
    s.env = SimpleNamespace(train_loader=None, generator=None)
    s.config = _real_config(batch_size=8, privacy_dataset_size=640)
    s._configure_from_settings()
    assert s._dp_dataset_size == 640
    assert abs(s._dp_sample_rate - (8 / 640)) < 1e-9


def test_unresolvable_dataset_size_raises() -> None:
    """No loader AND no configured dataset_size -> raise (no magic 1000)."""
    s = _bare_strategy()
    s.env = SimpleNamespace(train_loader=None, generator=None)
    # privacy.dataset_size None AND data.dataset_size None -> nothing to derive q.
    s.config = _real_config(batch_size=8, privacy_dataset_size=None)
    with pytest.raises(ConfigurationError):
        s._configure_from_settings()


# ---------------------------------------------------------------------------
# Finding #3: helper renamed and its per-PARAMETER contract is honest.
# ---------------------------------------------------------------------------


def test_param_grad_norms_returns_one_value_per_parameter() -> None:
    """Renamed helper returns #params-with-grad values (per-parameter, not per-sample)."""
    s = _bare_strategy()
    gen = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    for p in gen.parameters():
        p.grad = torch.ones_like(p)
    s.env = SimpleNamespace(generator=gen, train_loader=None)

    norms = s._param_grad_norms()
    n_params_with_grad = sum(1 for p in gen.parameters() if p.grad is not None)
    assert norms.numel() == n_params_with_grad  # one norm PER PARAMETER tensor
    # The old name no longer exists (rename, not alias).
    assert not hasattr(s, "_per_sample_grad_norms")


# ---------------------------------------------------------------------------
# Property: the DP transform still clips + noises a known gradient in-place.
# ---------------------------------------------------------------------------


def test_apply_dp_adds_calibrated_noise_without_reclipping() -> None:
    s = _bare_strategy()
    s.env = _loader_env(num_samples=400)
    s.config = _real_config(noise_multiplier=0.5, max_grad_norm=1.0)
    s._configure_from_settings()
    s._dp_step_examples = 2
    assert s._dp_enabled and s._max_grad_norm == 1.0

    model = torch.nn.Linear(4096, 1, bias=False)
    model.weight.grad = torch.zeros_like(model.weight)

    torch.manual_seed(0)
    s._apply_dp_to_grads(model)

    # Per-sample clipping happens upstream; the hook only adds N(0, (sigma*C/B)^2).
    std = float(model.weight.grad.std())
    assert abs(std - 0.5 * 1.0 / 2) < 0.02


def test_apply_dp_noop_when_disabled() -> None:
    s = _bare_strategy()
    s._dp_enabled = False
    s._dp_batch_size = 4
    model = torch.nn.Linear(4, 1, bias=False)
    g0 = torch.full_like(model.weight, 3.0)
    model.weight.grad = g0.clone()
    s._apply_dp_to_grads(model)
    assert torch.equal(model.weight.grad, g0)  # untouched when DP off


# ---------------------------------------------------------------------------
# Property: the Trainer-facing hook privatises grads AND advances accountant.
# ---------------------------------------------------------------------------


def test_clip_and_log_hook_applies_dp_and_records_step() -> None:
    s = _bare_strategy()
    s.env = _loader_env(num_samples=400)
    s.config = SimpleNamespace(
        privacy=PrivacyConfig(
            noise_multiplier=0.5, max_grad_norm=1.0, target_delta=1e-5
        ),
        data=_data(4),
        # Nested shape: phase 8 moved clipping to
        # `optimization.gradient.clip.{value,enabled}`. A stand-in still
        # carrying the flat spelling raises in `_resolve_clipping`.
        optimization=SimpleNamespace(
            gradient=SimpleNamespace(
                clip=SimpleNamespace(value=1.0, enabled=True),
            )
        ),
        # `log_interval` folded to `intervals.log`; strategies/base.py reads
        # `logging.intervals.log`, so the flat double left it unreachable.
        logging=block_stub("logging", log_gradients=False, log_interval=50),
    )
    s._configure_from_settings()

    gen = torch.nn.Linear(4, 1, bias=False)
    opt_g = torch.optim.SGD(gen.parameters(), lr=0.1)
    # Re-bind env to expose generator + opt_g for the hook (keep the loader).
    s.env = SimpleNamespace(
        generator=gen, opt_g=opt_g, train_loader=s.env.train_loader
    )

    big = torch.zeros_like(gen.weight)
    big[0, 0] = 50.0
    gen.weight.grad = big.clone()

    steps_before = s.accountant.steps
    with pytest.raises(RuntimeError, match="per-sample clipping path"):
        s._clip_and_log_gradients(gen, epoch=0, step=0, model_name="generator")
    assert s.accountant.steps == steps_before

    s._dp_per_sample_clipped = True
    torch.manual_seed(1)
    total_norm = s._clip_and_log_gradients(gen, epoch=0, step=0, model_name="generator")

    assert s.accountant.steps == steps_before + 1
    assert not s._dp_per_sample_clipped  # consumed: the next step must clip again
    g = gen.weight.grad
    assert torch.isfinite(g).all()
    assert g[0, 1] != 0.0
    assert isinstance(total_norm, float)


def test_post_step_returns_privacy_and_provenance_without_double_accounting() -> None:
    s = _bare_strategy()
    s.env = _loader_env(num_samples=500)
    s.config = _real_config(batch_size=10)
    s._configure_from_settings()

    steps_before = s.accountant.steps
    metrics = s.post_step()
    assert s.accountant.steps == steps_before  # snapshot does not advance accountant
    assert "dp/epsilon" in metrics
    assert metrics["dp/max_grad_norm"] == s._max_grad_norm
    # Provenance stamped: the honest sample rate / dataset size are reported.
    assert metrics["dp/dataset_size"] == 500.0
    assert abs(metrics["dp/sample_rate"] - (10 / 500)) < 1e-9
