"""``validation.scoring.compute`` is validated against registry-or-emitted (cohort review T0.5)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin
from spectramr.models.capabilities import StrategyCapabilities


def _host(emitted: frozenset[str]):
    class _Host(MetricsMixin):
        capabilities = StrategyCapabilities(emitted_metrics=emitted)

    return _Host()


def _overriding_host(emitted: frozenset[str]):
    """A strategy whose own ``validation_step`` writes ``val_loss``, as ``ModelValidationMixin`` does."""

    class _Host(MetricsMixin):
        capabilities = StrategyCapabilities(emitted_metrics=emitted)

        def validation_step(self, input_batch, target_batch, **kwargs):
            return {"val_loss": 0.0}

    return _Host()


def _registered(name: str) -> bool:
    return name in {"psnr", "ssim", "hfen"}


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    import spectramr.core.metrics.registry as reg

    monkeypatch.setattr(reg.MetricsRegistry, "is_registered", staticmethod(_registered))


def test_emitted_names_are_accepted_and_removed_from_the_computer_list() -> None:
    host = _host(frozenset({"val_field_mse", "val_field_bias"}))
    assert host._registry_backed_validation_metrics(["psnr", "field_mse", "field_bias"]) == ["psnr"]


def test_declaring_strategy_raises_on_an_ownerless_name() -> None:
    """The planted violation: a typo that used to become a missing column."""
    host = _host(frozenset({"val_field_mse"}))
    with pytest.raises(ValueError, match="neither registered nor emitted"):
        host._registry_backed_validation_metrics(["psnr", "field_typo"])


def test_silent_strategy_logs_a_census_line_and_keeps_the_registry_part(caplog) -> None:
    host = _host(frozenset())
    with caplog.at_level(logging.INFO):
        kept = host._registry_backed_validation_metrics(["psnr", "crlb_objective"])
    assert kept == ["psnr"]
    assert any("declares no emitted_metrics" in r.getMessage() for r in caplog.records)


def test_val_loss_is_owned_by_a_validation_step_that_writes_it() -> None:
    host = _overriding_host(frozenset({"val_field_mse"}))
    assert host._registry_backed_validation_metrics(["val_loss", "ssim"]) == ["ssim"]


def test_val_loss_is_unowned_on_the_metrics_mixin_path() -> None:
    """PLANTED (#2838): ``MetricsMixin.validation_step`` returns only registry keys.

    ``val_loss`` was exempt here too, so a declaring MRF-style strategy accepted
    a selector column that never arrives.
    """
    host = _host(frozenset({"val_field_mse"}))
    with pytest.raises(ValueError, match=r"\['val_loss'\]"):
        host._registry_backed_validation_metrics(["val_loss", "ssim"])


def test_universal_keys_follow_the_resolved_validation_step() -> None:
    from spectramr.core.metrics.validation_names import UNIVERSAL_VALIDATION_KEYS
    from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
        universal_validation_keys,
    )
    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    assert universal_validation_keys(type(_host(frozenset()))) == frozenset()
    assert universal_validation_keys(type(_overriding_host(frozenset()))) == (
        UNIVERSAL_VALIDATION_KEYS
    )
    assert universal_validation_keys(ReconstructionTrainingStrategy) == UNIVERSAL_VALIDATION_KEYS


def test_the_mrf_strategy_named_in_2838_earns_no_val_loss() -> None:
    from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
        universal_validation_keys,
    )
    from spectramr.infrastructure.training.strategies.mrf_acquisition_strategies import (
        ConformalMRFDictlessReconStrategy,
    )

    assert universal_validation_keys(ConformalMRFDictlessReconStrategy) == frozenset()


def test_the_rule_is_read_from_the_one_owner_not_a_local_copy() -> None:
    """#2819: the mixin consumes the core predicate the audit also reads.

    The audit once kept its own registry-only copy of this rule and rejected
    ``val_loss``, which every run writes; a local copy here would start the same drift.
    """
    import inspect

    src = inspect.getsource(MetricsMixin._registry_backed_validation_metrics)
    assert "partition_validation_compute(" in src
    assert "frozenset({\"loss\", \"val_loss\"})" not in src


# ---------------------------------------------------------------------------
# #2839: an all-emitted compute list builds an empty computer, not the default set
# ---------------------------------------------------------------------------


def _validating_host(emitted: frozenset[str], compute: list[str] | None):
    """A host whose config carries only a real ``validation`` block; no model is built."""
    from spectramr.config.schemas.validation import ValidationConfigSchema

    host = _host(emitted)
    host.device = torch.device("cpu")
    host.config = SimpleNamespace(
        validation=ValidationConfigSchema(scoring={"compute": compute, "primary": "val_loss"})
    )
    return host


def test_all_emitted_compute_list_builds_a_computer_with_no_metrics() -> None:
    """The planted case: PSNR/SSIM/MSE used to be substituted for the empty registry list."""
    host = _validating_host(frozenset({"crlb_objective"}), ["crlb_objective"])
    computer = host._get_validation_metrics_computer(host.config)
    assert computer.get_metric_names() == []
    # The selector still resolves its direction through the (empty) computer.
    assert computer.config.primary_metric == "val_loss"


def test_explicitly_empty_compute_list_builds_a_computer_with_no_metrics() -> None:
    host = _validating_host(frozenset(), [])
    assert host._get_validation_metrics_computer(host.config).get_metric_names() == []


def test_undeclared_compute_list_inherits_training_metrics() -> None:
    """The documented default: ``compute: None`` inherits ``training.metrics``."""
    host = _validating_host(frozenset(), None)
    host.config.metrics = SimpleNamespace(compute=["ssim"])
    assert host._get_validation_metrics_computer(host.config).get_metric_names() == ["ssim"]


def test_undeclared_compute_list_without_a_metrics_block_keeps_psnr_ssim_mse() -> None:
    """The factory default, reached only by a config with no ``metrics`` block."""
    host = _validating_host(frozenset(), None)
    names = host._get_validation_metrics_computer(host.config).get_metric_names()
    assert set(names) == {"psnr", "ssim", "mse"}


def test_validation_step_skips_the_forward_when_no_registry_metric_is_requested() -> None:
    """The strategy's output need not be an image; nothing may score it unasked."""

    class _Forbidden:
        def eval(self):
            raise AssertionError("generator ran for a computer with no metrics")

    host = _validating_host(frozenset({"crlb_objective"}), ["crlb_objective"])
    host.env = SimpleNamespace(generator=_Forbidden())
    assert host.validation_step(torch.rand(2, 6, 2), torch.rand(2, 5)) == {}
