"""The strategy is what routes a step into the authors' code — observed, not assumed.

The load-bearing test here is :func:`test_the_old_strategy_leaves_the_upstream_silent`.
Without it, a spy that fires under the new strategy proves only that the method exists;
it does not prove the new strategy is what reaches it. The planted violation is the
previous arrangement: the same adapter under ``GraphColdDiffusionStrategy``, where the
upstream process must never run (non-negotiable 15).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.training.strategies.upstream_process_strategy import (
    UpstreamProcessStrategy,
)

_SIZE = 32


class _SpyAdapter(torch.nn.Module):
    """Stands in for a BaselineAdapter, counting the calls that matter."""

    UPSTREAM_LOSS_FAMILY = "l1"

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.training_loss_calls: list[tuple] = []
        self.forward_calls = 0
        self.validation_sample_calls: list[tuple] = []
        self.raise_on_validation_sample = False

    def training_loss(self, x_0, batch=None):
        self.training_loss_calls.append((x_0.shape, batch))
        return (self.weight * x_0).abs().mean()

    def forward(self, x, *args, **kwargs):
        self.forward_calls += 1
        return x

    def validation_sample(self, input_batch, target_batch, batch=None):
        self.validation_sample_calls.append((input_batch.shape, target_batch.shape, batch))
        if self.raise_on_validation_sample:
            raise NotImplementedError("no reverse process wired for this spy")
        return target_batch


def _entry(name: str, weight: float = 1.0, enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(name=name, weight=weight, enabled=enabled)


def _config(loss_names: list[str], output_dir: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        losses=SimpleNamespace(
            image_losses=[_entry(n) for n in loss_names],
            kspace_losses=[],
            complex_losses=[],
            latent_losses=[],
        ),
        training=SimpleNamespace(output_dir=output_dir),
    )


def _strategy(adapter, config) -> UpstreamProcessStrategy:
    """A strategy with its environment bound, bypassing the full director."""
    strategy = UpstreamProcessStrategy.__new__(UpstreamProcessStrategy)
    strategy.config = config
    strategy.env = SimpleNamespace(generator=adapter)
    return strategy


def test_the_step_runs_the_adapters_training_loss() -> None:
    """The whole objective comes from the adapter, on the CLEAN target."""
    adapter = _SpyAdapter()
    strategy = _strategy(adapter, _config(["l1"]))
    target = torch.randn(2, 2, _SIZE, _SIZE)

    out = strategy._compute_losses_impl(torch.zeros_like(target), target, epoch=0)

    assert adapter.training_loss_calls, "the authors' training step never ran"
    assert adapter.training_loss_calls[0][0] == target.shape, (
        "the upstream must degrade the clean target itself; handing it the loader's "
        "already-undersampled input would degrade twice"
    )
    assert set(out) == {"g_total_loss"}
    assert out["g_total_loss"].grad_fn is not None


def test_the_old_strategy_leaves_the_upstream_silent() -> None:
    """PLANTED VIOLATION: the previous arrangement must not reach the upstream.

    ``GraphColdDiffusionStrategy`` builds its own mask ladder and calls the model's
    ``forward``. If this ever starts calling ``training_loss``, the two strategies have
    converged and the distinction this PR rests on is gone.
    """
    from spectramr.infrastructure.training.strategies.graph_cold_diffusion_strategy import (
        GraphColdDiffusionStrategy,
    )

    source = GraphColdDiffusionStrategy._compute_losses_impl.__doc__ or ""
    body = GraphColdDiffusionStrategy._compute_losses_impl.__code__.co_names
    assert "training_loss" not in body, (
        "GraphColdDiffusionStrategy now calls training_loss; the upstream process would "
        f"run under BOTH strategies and the arms could not be told apart. {source[:80]}"
    )
    assert "_apply_degradation" in body, (
        "GraphColdDiffusionStrategy no longer applies its own degradation — the planted "
        "violation has stopped describing the arrangement it guards against"
    )


def test_a_model_without_an_upstream_process_is_refused() -> None:
    """A plain generator on this strategy would train nothing; say so at setup."""
    strategy = _strategy(torch.nn.Conv2d(2, 2, 1), _config(["l1"]))
    with pytest.raises(TypeError, match="must be a BaselineAdapter"):
        strategy._adapter()


def test_a_declared_objective_the_upstream_will_not_compute_is_refused() -> None:
    """PLANTED VIOLATION: FDB is MSE, so an arm declaring l1 must not start."""
    adapter = _SpyAdapter()
    adapter.UPSTREAM_LOSS_FAMILY = "l2"
    strategy = _strategy(adapter, _config(["l1"]))
    with pytest.raises(ValueError, match="minimises 'l2'"):
        strategy._reject_objective_mismatch(adapter)


def test_the_agreeing_declaration_is_accepted() -> None:
    """The check must not be vacuous — the matching case has to pass."""
    adapter = _SpyAdapter()
    strategy = _strategy(adapter, _config(["l1"]))
    strategy._reject_objective_mismatch(adapter)


def test_a_disabled_entry_does_not_count_against_the_objective() -> None:
    """The arms carry a disabled ``ssim`` entry; it declares nothing to minimise."""
    adapter = _SpyAdapter()
    config = _config(["l1"])
    config.losses.image_losses.append(_entry("ssim", weight=0.1, enabled=False))
    strategy = _strategy(adapter, config)
    strategy._reject_objective_mismatch(adapter)


def test_calibration_is_pointed_at_the_run_output(tmp_path) -> None:
    """An upstream that writes state to disk gets a directory under the run."""
    seen: list = []
    adapter = _SpyAdapter()
    adapter.set_calibration_dir = seen.append
    strategy = _strategy(adapter, _config(["l1"], output_dir=str(tmp_path / "run")))

    strategy._point_calibration_at_the_run(adapter)

    assert seen == [tmp_path / "run" / "upstream_calibration"]


def test_an_adapter_without_the_hook_is_left_alone() -> None:
    """Only adapters that declare the hook are touched — no attribute is invented."""
    adapter = _SpyAdapter()
    strategy = _strategy(adapter, _config(["l1"], output_dir="/tmp/run"))
    strategy._point_calibration_at_the_run(adapter)
    assert not hasattr(adapter, "_calibration_dir")


def test_the_strategy_declares_its_loss_ownership() -> None:
    """Undeclared ownership is what leaves an arm's image_losses UNVERIFIED (#1918)."""
    assert UpstreamProcessStrategy.folds_image_losses is False
    assert UpstreamProcessStrategy.inline_losses == frozenset({"l1", "l2"})


# ---------------------------------------------------------------------------
# Finding 21: validation must drive the adapter's OWN reverse process, never
# fall through to a bare forward() at whatever timestep is left to default.
# ---------------------------------------------------------------------------


def test_validation_step_is_not_the_bare_t0_fallback() -> None:
    """PLANTED VIOLATION: an undefined `validation_step` resolves to the t=0 fallback.

    Without an override, `UpstreamProcessStrategy.validation_step` resolves to
    `MetricsMixin.validation_step` -- one `generator(input_batch)` call with no
    timestep. This is the exact shape finding 21 reports; pinning the identity
    check is what would catch a future refactor that quietly deletes the override.
    """
    from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
        MetricsMixin,
    )

    assert UpstreamProcessStrategy.validation_step is not MetricsMixin.validation_step


def test_run_reverse_process_calls_validation_sample_not_forward() -> None:
    """The reverse-process call reaches `validation_sample`, and never `forward`."""
    adapter = _SpyAdapter()
    strategy = _strategy(adapter, _config(["l1"]))
    target = torch.randn(2, 2, _SIZE, _SIZE)
    input_batch = torch.zeros_like(target)

    out = strategy._run_reverse_process(adapter, input_batch, target, batch_data=None)

    assert adapter.validation_sample_calls, "the adapter's own reverse process never ran"
    assert adapter.forward_calls == 0, "validation must not fall back to a bare forward()"
    assert torch.equal(out, target)


def test_run_reverse_process_translates_refusal_to_a_named_failure() -> None:
    """PLANTED VIOLATION: an adapter that cannot sample must not be silently accepted.

    `validation_sample` raising `NotImplementedError` is the honest state (finding 21);
    swallowing it and falling back to `forward()` would be the exact defect this
    strategy exists to close, so the refusal must surface, not disappear.
    """
    adapter = _SpyAdapter()
    adapter.raise_on_validation_sample = True
    strategy = _strategy(adapter, _config(["l1"]))
    target = torch.randn(2, 2, _SIZE, _SIZE)
    input_batch = torch.zeros_like(target)

    with pytest.raises(RuntimeError, match="no reverse-sampling process"):
        strategy._run_reverse_process(adapter, input_batch, target, batch_data=None)
    assert adapter.forward_calls == 0, "the refusal must not fall through to forward()"


class _StubMetricsComputer:
    """Records what it was asked to score, without a real metrics pipeline."""

    def __init__(self) -> None:
        self.compute_calls: list[tuple] = []

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        self.compute_calls.append((pred, target))
        return {"val_psnr": 42.0}


def test_validation_step_drives_the_reverse_process_and_scores_it(monkeypatch) -> None:
    """End-to-end: `validation_step` runs the reverse process, then scores its output.

    Only the metrics-computer PROPERTY (a data descriptor `MetricsMixin` owns, so it
    cannot be shadowed by a plain instance attribute) and the metric-transform hook are
    stubbed; the reverse-process call and the eval/train toggling are real strategy code.
    """
    adapter = _SpyAdapter()
    config = _config(["l1"])
    config.validation = SimpleNamespace()
    strategy = _strategy(adapter, config)
    strategy._apply_metric_transforms = lambda pred, target, cfg: (pred, target)
    stub_computer = _StubMetricsComputer()
    monkeypatch.setattr(
        type(strategy), "validation_metrics_computer", property(lambda self: stub_computer)
    )

    target = torch.randn(2, 2, _SIZE, _SIZE)
    input_batch = torch.zeros_like(target)
    metrics = strategy.validation_step(input_batch, target, batch_data={"mask": None})

    assert metrics == {"val_psnr": 42.0}
    assert adapter.validation_sample_calls, "the reverse process never ran"
    assert adapter.forward_calls == 0, "validation must not fall back to a bare forward()"
    assert stub_computer.compute_calls and torch.equal(stub_computer.compute_calls[0][0], target)
    assert adapter.training, "the generator must be left in train() mode afterward"


def test_validation_step_leaves_train_mode_even_when_the_reverse_process_raises(
    monkeypatch,
) -> None:
    """The eval()/train() toggle is a `finally` -- a raised refusal must not strand it."""
    adapter = _SpyAdapter()
    adapter.raise_on_validation_sample = True
    config = _config(["l1"])
    config.validation = SimpleNamespace()
    strategy = _strategy(adapter, config)
    strategy._apply_metric_transforms = lambda pred, target, cfg: (pred, target)
    monkeypatch.setattr(
        type(strategy),
        "validation_metrics_computer",
        property(lambda self: _StubMetricsComputer()),
    )

    target = torch.randn(2, 2, _SIZE, _SIZE)
    input_batch = torch.zeros_like(target)
    with pytest.raises(RuntimeError, match="no reverse-sampling process"):
        strategy.validation_step(input_batch, target)

    assert adapter.training, "a raised refusal must not strand the generator in eval()"
