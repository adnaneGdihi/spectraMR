"""Unit tests for the equivariance-conformal calibration strategy (B-1 / Idea 1).

Direct-import tests (no factory / DI). The heavyweight ``BaseTrainingStrategy``
``__init__`` is bypassed via ``object.__new__`` so we exercise the calibration
logic on tiny tensors; the module-level ``run_equivariance_calibration`` and
helpers are tested as plain functions.
"""

from __future__ import annotations

import json
import types

import torch

from spectramr.config.schemas.training.equivariance_conformal import (
    EquivarianceConformalConfig,
)
from spectramr.infrastructure.training.strategies.equivariance_conformal_strategy import (
    EquivarianceConformalCalibrationStrategy,
    run_equivariance_calibration,
    _extract_measurement,
    _two_sample_ks,
)


def _identity(y, csm=None):
    return y


def _tensor_loader(n_batches=4, b=2, c=1, hw=24):
    return [torch.randn(b, c, hw, hw) for _ in range(n_batches)]


def test_extract_measurement_variants():
    t = torch.randn(2, 1, 8, 8)
    assert _extract_measurement(t) is t
    assert _extract_measurement((t, torch.zeros(1))) is t
    assert _extract_measurement([t]) is t
    assert torch.equal(_extract_measurement({"input": t}), t)
    assert torch.equal(_extract_measurement({"kspace": t}), t)


def test_extract_measurement_bad_batch_raises():
    try:
        _extract_measurement({"nope": torch.randn(1)})
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_run_calibration_loop_identity_quantile_zero():
    """Identity recon => all scores 0 => quantile 0."""
    report = run_equivariance_calibration(
        _identity,
        _tensor_loader(),
        alpha=0.1,
        num_rotations=4,
        group="SO2",
        seed=0,
        device="cpu",
    )
    assert report["score_fn"] == "equivariance_defect"
    assert report["n_calibration"] == 8
    assert report["quantile"] >= 0.0
    assert abs(report["quantile"]) < 1e-5
    assert abs(report["mean_score"]) < 1e-5


def test_run_calibration_non_equivariant_positive_quantile():
    ramp = torch.linspace(-1.0, 1.0, 24).view(1, 1, 1, 24)
    op = lambda y, csm=None: y * ramp  # noqa: E731
    report = run_equivariance_calibration(
        op, _tensor_loader(), alpha=0.2, num_rotations=6, group="SO2", seed=1, device="cpu"
    )
    assert report["quantile"] > 0.0
    assert report["mean_score"] > 0.0
    assert "exchangeability" in report
    assert report["exchangeability"]["test"] == "ks_test"


def test_run_calibration_emits_coverage_at_alpha():
    """The equivariance-conformal arm grades on ``coverage_at_alpha`` — the
    report must PRODUCE it (held-out split estimate), not leave the metric
    absent (#18, metric<->claim mismatch). For identity recon all scores are 0,
    so the held-out half is fully covered (coverage == 1.0)."""
    report = run_equivariance_calibration(
        _identity,
        _tensor_loader(),
        alpha=0.1,
        num_rotations=4,
        group="SO2",
        seed=0,
        device="cpu",
    )
    assert "coverage_at_alpha" in report
    cov = report["coverage_at_alpha"]
    assert cov is not None and 0.0 <= cov <= 1.0
    assert report["empirical_coverage"] == cov


def test_run_calibration_empty_loader_raises():
    try:
        run_equivariance_calibration(_identity, [], device="cpu")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for empty loader")


def test_run_calibration_handles_unbatched_3d_sample():
    loader = [torch.randn(1, 24, 24)]  # [C, H, W] -> auto-unsqueezed
    report = run_equivariance_calibration(
        _identity, loader, num_rotations=2, group="SO2", device="cpu"
    )
    assert report["n_calibration"] == 1


def test_two_sample_ks_identical_distributions():
    a = torch.linspace(0, 1, 100)
    b = torch.linspace(0, 1, 100)
    stat, p = _two_sample_ks(a, b)
    assert stat < 0.05
    assert p > 0.05  # not flagged


def test_two_sample_ks_disjoint_distributions():
    a = torch.zeros(100)
    b = torch.ones(100)
    stat, p = _two_sample_ks(a, b)
    assert stat > 0.9
    assert p < 0.05  # flagged as non-exchangeable


def _make_bare_strategy(cfg, device="cpu"):
    """Build a strategy instance without running the heavyweight __init__."""
    strat = object.__new__(EquivarianceConformalCalibrationStrategy)
    strat._ec_cfg = cfg
    strat.device = torch.device(device)
    # _frozen_reconstructor reads self.context.generator then generator_model.
    strat.context = types.SimpleNamespace(generator=torch.nn.Identity())
    return strat


def test_strategy_run_calibration_writes_artefact(tmp_path):
    out = tmp_path / "q.json"
    cfg = EquivarianceConformalConfig(
        num_rotations_K=4,
        group="SO2",
        alpha=0.1,
        output_artefact=str(out),
    )
    strat = _make_bare_strategy(cfg)
    report = strat.run_calibration(_tensor_loader())
    assert report["quantile"] >= 0.0
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["score_fn"] == "equivariance_defect"
    assert loaded["n_calibration"] == 8


def test_strategy_run_calibration_requires_block():
    strat = _make_bare_strategy(None)
    try:
        strat.run_calibration(_tensor_loader())
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError without equivariance_conformal block")


def test_strategy_train_step_is_noop():
    cfg = EquivarianceConformalConfig()
    strat = _make_bare_strategy(cfg)
    out = strat.train_step(batch=torch.randn(1, 1, 8, 8))
    # train_step returns an EMPTY step-config list (no optimizer steps); the
    # old dict return hit KeyError('optimizer') in Trainer.execute_step (see
    # the strategy's train_step docstring). The zero loss is asserted via
    # _compute_losses_impl in test_strategy_compute_losses_zero.
    assert out == []


def test_strategy_compute_losses_zero():
    cfg = EquivarianceConformalConfig()
    strat = _make_bare_strategy(cfg)
    x = torch.randn(1, 1, 8, 8)
    losses = strat._compute_losses_impl(x, x, epoch=0)
    assert float(losses["g_total_loss"]) == 0.0


def test_frozen_reconstructor_detaches_and_freezes():
    cfg = EquivarianceConformalConfig()
    strat = _make_bare_strategy(cfg)
    lin = torch.nn.Conv2d(1, 1, 3, padding=1)
    strat.context = types.SimpleNamespace(generator=lin)
    fn = strat._frozen_reconstructor()
    y = torch.randn(1, 1, 8, 8)
    out = fn(y)
    assert not out.requires_grad
    assert all(not p.requires_grad for p in lin.parameters())


def test_setup_coerces_raw_dict_block_to_typed_config():
    """Regression: the mode-dispatch leaves ``training.equivariance_conformal``
    as a raw dict (config.training loads as the generic schema), so the strategy
    must coerce it to EquivarianceConformalConfig before attribute access —
    otherwise ``cfg.alpha`` raised "'dict' object has no attribute 'alpha'"
    (cluster smoke 20260605, job 7095209, reachable once the upstream
    'conditioning' kwarg leak was fixed)."""
    strat = object.__new__(EquivarianceConformalCalibrationStrategy)
    raw = {
        "alpha": 0.1,
        "num_rotations_K": 16,
        "group": "SO2",
        "output_artefact": "out.json",
    }
    strat.config = types.SimpleNamespace(
        training=types.SimpleNamespace(equivariance_conformal=raw)
    )
    strat._setup_strategy_specific_components()
    assert isinstance(strat._ec_cfg, EquivarianceConformalConfig)
    assert strat._ec_cfg.alpha == 0.1
    assert strat._ec_cfg.num_rotations_K == 16
