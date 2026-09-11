"""BaseTrainingStrategy: the per-step metrics accessor must not sync (#707)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.training.strategies.base import (  # noqa: E402
    BaseTrainingStrategy,
)


def test_get_last_metrics_does_not_sync_the_gpu(no_gpu_sync):
    no_gpu_sync(BaseTrainingStrategy.get_last_metrics)


def test_get_last_metrics_on_a_strategy_that_published_nothing(empty_metrics_is_safe):
    empty_metrics_is_safe(BaseTrainingStrategy.get_last_metrics)


def _capture_snapshot_kwargs(monkeypatch, tmp_path, tensors, **call_kwargs):
    """Drive `BaseTrainingStrategy.save_debug_snapshot` and return what it forwarded.

    Everything the method touches is stubbed onto a bare instance: it resolves
    `run_dir`/`config` off `self` and imports its collaborators at call time, so
    patching the module attributes is enough — no strategy construction needed.
    """
    from types import SimpleNamespace

    from spectramr.infrastructure.training import debug_snapshot as ds
    from spectramr.infrastructure.training.utils import domain_inference as di

    seen: dict = {}

    def _fake_save(**kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(ds, "save_debug_snapshot", _fake_save)
    # The config SSOT declares this arm's tensors to be k-space; that declaration
    # is what turns on the authoritative (veto-bypassing) key set.
    monkeypatch.setattr(di, "needs_ifft_for_visualization", lambda _cfg: (True, True))

    strategy = BaseTrainingStrategy.__new__(BaseTrainingStrategy)
    strategy.env = SimpleNamespace(run_output_dir=str(tmp_path))
    strategy.config = SimpleNamespace(
        training=SimpleNamespace(output_dir=str(tmp_path)),
        logging=None,
        data=SimpleNamespace(processing=SimpleNamespace(enable_log_scaling=False)),
    )
    strategy._model_output_snapshot_done = False
    BaseTrainingStrategy.save_debug_snapshot(strategy, tensors, **call_kwargs)
    return seen


def test_cold_diffusion_degraded_keys_bypass_the_spectrum_veto(monkeypatch, tmp_path):
    """The tensor the model is actually fed must render as an image, not speckle.

    `q_sample` is `x_0 * mask`, so `noisy_kspace` is k-space by the same config
    declaration that covers `input` and `target`. Left off the authoritative set
    it fell through to the spectrum veto — the one the 2026-06-27 audit records
    as false-negating on normalized multicoil M4Raw — and the accelerated,
    zero-filled input rendered as raw, off-centre k-space.
    """
    seen = _capture_snapshot_kwargs(
        monkeypatch,
        tmp_path,
        {
            "noisy_kspace": torch.zeros(1, 2, 4, 4),
            "noisy_images": torch.zeros(1, 2, 4, 4),
            "mask": torch.ones(1, 1, 4, 4),
        },
        step=1,
        tag="diffusion_step",
        in_kspace_keys={"noisy_kspace"},
    )
    assert "noisy_kspace" in seen["authoritative_kspace_keys"]
    # `mask` is a 0/1 sampling pattern, NOT k-space: marking it would IFFT the
    # very picture that shows which lines were kept.
    assert "mask" not in seen["authoritative_kspace_keys"]
    # `noisy_images` must NOT be here. This set bypasses the veto
    # unconditionally, and the name is bound to a LATENT in the
    # latent-diffusion branch (`q_sample(latent_z0, ...)`), so membership would
    # IFFT a latent with nothing left to catch it. No emitter writes the key
    # today — the diffusion capture names it `noisy_kspace` — which is what
    # kept the hazard invisible: only a synthetic tensor dict like this one
    # puts it in scope at all.
    assert "noisy_images" not in seen["authoritative_kspace_keys"]


def test_extra_context_reaches_the_snapshot_writer(monkeypatch, tmp_path):
    """`extra` is how a snapshot says what its tensors were derived from.

    The cold path stamps `degradation_source` here; without forwarding, reading
    the PNG would still require cross-referencing `resolved_config.json`.
    """
    seen = _capture_snapshot_kwargs(
        monkeypatch,
        tmp_path,
        {"noisy_kspace": torch.zeros(1, 2, 4, 4)},
        step=1,
        tag="diffusion_step",
        extra={"degradation_source": "input"},
    )
    assert seen["extra"] == {"degradation_source": "input"}


def _bare_strategy(tmp_path, **env_kwargs):
    """A `BaseTrainingStrategy` with only what the snapshot helpers read."""
    from types import SimpleNamespace

    strategy = BaseTrainingStrategy.__new__(BaseTrainingStrategy)
    strategy.env = SimpleNamespace(run_output_dir=str(tmp_path), **env_kwargs)
    strategy.config = SimpleNamespace(data=None)
    return strategy


class TestCanonicalKeyContract:
    """`first_steps/input_prepared` must never silently claim to be the model
    input. It is captured BEFORE the forward pass, so for a strategy that
    degrades further inside the step it is the clean tensor — the exact reading
    that made an accelerated cold-diffusion arm look fully sampled."""

    def test_the_default_strategy_asserts_prepared_is_the_model_input(self, tmp_path) -> None:
        strategy = _bare_strategy(tmp_path)
        extra = BaseTrainingStrategy._first_steps_extra(strategy, epoch=3)
        assert extra == {"epoch": 3, "prepared_equals_model_input": True}

    def test_a_deferring_strategy_names_where_the_real_input_lives(self, tmp_path) -> None:
        """ "This is not the model input" alone sends the reader back to the
        code; the artifact must say where to look instead."""
        strategy = _bare_strategy(tmp_path)
        strategy.snapshot_prepared_is_model_input = False
        strategy.snapshot_model_input_tag = "diffusion_step"
        extra = BaseTrainingStrategy._first_steps_extra(strategy, epoch=0)
        assert extra["prepared_equals_model_input"] is False
        assert extra["model_input_snapshot_tag"] == "diffusion_step"

    def test_the_diffusion_family_declares_the_carve_out(self) -> None:
        """Pins the contract at the class that needs it: `q_sample` degrades
        inside the step for every diffusion arm, cold or noise."""
        from spectramr.infrastructure.training.strategies.diffusion import (
            DiffusionTrainingStrategy,
        )

        assert DiffusionTrainingStrategy.snapshot_prepared_is_model_input is False
        assert DiffusionTrainingStrategy.snapshot_model_input_tag == "diffusion_step"
        assert BaseTrainingStrategy.snapshot_prepared_is_model_input is True


class TestProvenanceIsBuiltOncePerStrategy:
    def test_the_dataset_walk_is_not_paid_per_step(self, monkeypatch, tmp_path) -> None:
        """Non-negotiable 9: snapshots fire inside the training step, and the
        walk touches the dataset wrapper chain and the whole module tree.
        Nothing it reads changes after the environment is built."""
        from spectramr.infrastructure.training import snapshot_provenance as sp

        calls = {"n": 0}

        def _counting(*_args, **_kwargs):
            calls["n"] += 1
            return {"source": "train", "declared": {}, "incomplete": []}

        monkeypatch.setattr(sp, "build_snapshot_provenance", _counting)
        strategy = _bare_strategy(tmp_path, data_loaders={}, generator=None)

        first = BaseTrainingStrategy._snapshot_provenance(strategy)
        for _ in range(5):
            BaseTrainingStrategy._snapshot_provenance(strategy)
        assert calls["n"] == 1
        assert first is BaseTrainingStrategy._snapshot_provenance(strategy)

    def test_a_failure_to_introspect_never_reaches_the_caller(self, monkeypatch, tmp_path) -> None:
        """A diagnostic must never be the reason a training run dies."""
        from spectramr.infrastructure.training import snapshot_provenance as sp

        def _boom(*_args, **_kwargs):
            raise RuntimeError("dataset exploded")

        monkeypatch.setattr(sp, "build_snapshot_provenance", _boom)
        strategy = _bare_strategy(tmp_path, data_loaders={}, generator=None)
        assert BaseTrainingStrategy._snapshot_provenance(strategy) is None

    def test_a_failed_build_is_cached_too(self, monkeypatch, tmp_path) -> None:
        """`None` must mean "built, and it failed" — not "not built yet".

        Under `getattr(..., None)` the two were indistinguishable, so a strategy
        whose introspection raises re-ran the entire dataset/model walk on EVERY
        subsequent snapshot: the exact per-step cost the cache exists to avoid
        (non-negotiable 9), paid hardest by the runs already in trouble.
        """
        from spectramr.infrastructure.training import snapshot_provenance as sp

        calls = {"n": 0}

        def _boom(*_args, **_kwargs):
            calls["n"] += 1
            raise RuntimeError("dataset exploded")

        monkeypatch.setattr(sp, "build_snapshot_provenance", _boom)
        strategy = _bare_strategy(tmp_path, data_loaders={}, generator=None)
        for _ in range(5):
            assert BaseTrainingStrategy._snapshot_provenance(strategy) is None
        assert calls["n"] == 1


class TestProvenanceNamesTheSplitItCameFrom:
    """`source` exists so a val snapshot cannot claim the train augmentation
    chain — they are built by *different* `tio.Compose` objects. A single
    train-built record reused everywhere asserts exactly that falsehood."""

    def test_each_source_gets_its_own_record_and_its_own_loader(
        self, monkeypatch, tmp_path
    ) -> None:
        from types import SimpleNamespace

        from spectramr.infrastructure.training import snapshot_provenance as sp

        seen = []

        def _record(_config, *, dataset=None, model=None, source="train"):
            seen.append((source, dataset))
            return {"source": source}

        monkeypatch.setattr(sp, "build_snapshot_provenance", _record)
        loaders = {
            "train": SimpleNamespace(dataset="TRAIN_DS"),
            "val": SimpleNamespace(dataset="VAL_DS"),
        }
        strategy = _bare_strategy(tmp_path, data_loaders=loaders, generator=None)

        train = BaseTrainingStrategy._snapshot_provenance(strategy, "train")
        val = BaseTrainingStrategy._snapshot_provenance(strategy, "val")

        assert train == {"source": "train"}
        assert val == {"source": "val"}
        # The val record must be built from the VAL loader, or `source` is a
        # label on the train chain rather than a claim about the val one.
        assert seen == [("train", "TRAIN_DS"), ("val", "VAL_DS")]

        # Still cached per source — one walk each, not one per snapshot.
        BaseTrainingStrategy._snapshot_provenance(strategy, "train")
        BaseTrainingStrategy._snapshot_provenance(strategy, "val")
        assert len(seen) == 2

    def test_the_phase_declaration_nests_and_restores(self, tmp_path) -> None:
        strategy = _bare_strategy(tmp_path)
        assert strategy._snapshot_phase == "train"
        with BaseTrainingStrategy.snapshot_source(strategy, "val"):
            assert strategy._snapshot_phase == "val"
            with BaseTrainingStrategy.snapshot_source(strategy, "test"):
                assert strategy._snapshot_phase == "test"
            # Restores the PREVIOUS value, not the class default — resetting to
            # "train" here would relabel the rest of an outer val block.
            assert strategy._snapshot_phase == "val"
        assert strategy._snapshot_phase == "train"

    def test_the_phase_is_restored_even_when_the_block_raises(self, tmp_path) -> None:
        strategy = _bare_strategy(tmp_path)
        with (
            pytest.raises(RuntimeError),
            BaseTrainingStrategy.snapshot_source(strategy, "val"),
        ):
            raise RuntimeError("validation blew up")
        assert strategy._snapshot_phase == "train"

    def test_the_virtual_fiducial_val_path_declares_itself(self) -> None:
        """The one live emitter. `validation_step` calls `_compute_losses_impl`
        directly, and that impl emits `vf_twin` — so the declaration has to be
        at the call, not on the emitter."""
        import inspect

        from spectramr.infrastructure.training.strategies import (
            virtual_fiducial_strategy as vf,
        )

        src = inspect.getsource(vf.ConcreteVirtualFiducialStrategy.validation_step)
        assert 'self.snapshot_source("val")' in src
        assert "self._compute_losses_impl(" in src


class _SyncCountingTensor:
    """Counts ``.detach()``, the first hop of every device->host reduction."""

    def __init__(self, real) -> None:
        self._real = real
        self.syncs = 0

    def detach(self):
        self.syncs += 1
        return self._real

    def __getattr__(self, name: str):
        return getattr(self._real, name)


class TestModelOutputScaleContextIsDeferred:
    """`_snapshot_model_output` must not price its own diagnostic (#1188).

    This emitter is the widest one in the codebase: every strategy routed
    through `_compute_losses` reaches it once per training step. Its `extra`
    is `_model_output_scale_context`, which does TWO `float(tensor)` device
    syncs -- so building it eagerly cost two syncs per step for the whole run
    while `max_calls` suppressed the write after the eighth. Same shape as the
    `vf_twin` defect #1188 names, in the path that carries far more traffic.
    """

    @staticmethod
    def _strategy(tmp_path, max_calls: int):
        from types import SimpleNamespace

        from tests.utils.block_config_stub import LoggingConfigStub

        strategy = BaseTrainingStrategy.__new__(BaseTrainingStrategy)
        strategy.env = SimpleNamespace(run_output_dir=str(tmp_path))
        strategy.config = SimpleNamespace(
            training=SimpleNamespace(output_dir=str(tmp_path)),
            logging=LoggingConfigStub(snapshots={"max_calls": max_calls}),
            data=None,
        )
        strategy._model_output_snapshot_done = False
        return strategy

    def test_the_scale_reduction_is_not_paid_once_the_budget_is_spent(self, tmp_path):
        import torch

        strategy = self._strategy(tmp_path, max_calls=1)
        target = torch.zeros(1, 1, 4, 4)

        # Step 0 writes, burning the single allowance.
        strategy._last_generator_output = torch.ones(1, 1, 4, 4)
        strategy._snapshot_model_output(input_batch=target, target_batch=target, step=0)
        assert (tmp_path / "debug_snapshots" / "model_output_step_000000").is_dir()

        probes = []
        for step in (1, 2, 3):
            probe = _SyncCountingTensor(torch.ones(1, 1, 4, 4))
            probes.append(probe)
            # `_compute_losses` clears this per step; the emitter's own early
            # return on it would otherwise mask which gate did the suppressing.
            strategy._model_output_snapshot_done = False
            strategy._last_generator_output = probe
            strategy._snapshot_model_output(input_batch=target, target_batch=target, step=step)
        assert [p.syncs for p in probes] == [0, 0, 0]

    def test_the_scale_context_still_reaches_a_written_snapshot(self, tmp_path):
        """Deferring must not cost the diagnostic — #587's whole point."""
        import json

        import torch

        strategy = self._strategy(tmp_path, max_calls=8)
        strategy._last_generator_output = torch.full((1, 1, 4, 4), 3.0)
        strategy._snapshot_model_output(
            input_batch=torch.zeros(1, 1, 4, 4),
            target_batch=torch.full((1, 1, 4, 4), 12.0),
            step=0,
        )
        snap = tmp_path / "debug_snapshots" / "model_output_step_000000"
        extra = json.loads((snap / "snapshot.json").read_text())["extra"]
        assert extra["abs_max_model_output"] == 3.0
        assert extra["abs_max_target"] == 12.0


class TestDeclaredMetricKeys:
    """#1682 -- the producer-declared CSV column hook.

    Base must declare NOTHING. The hook exists so a strategy that renames
    between knob and metric can bridge that rename itself; a base that declared
    anything would promise columns for strategies that never stamp them, which
    is the always-empty-column direction the header builder already rejected.
    """

    def test_base_declares_nothing(self):
        from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

        assert BaseTrainingStrategy.declared_metric_keys(object()) == frozenset()

    def test_hook_returns_a_set_not_a_list(self):
        from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

        result = BaseTrainingStrategy.declared_metric_keys(object())
        assert isinstance(result, frozenset), (
            "the header builder does `expected_loss_keys.update(...)`; a str "
            "return would silently splat into single characters"
        )


# ---------------------------------------------------------------------------
# The declared-loss contract (#1918, #1919)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SENSE_BRIDGE_ARM = (
    _REPO_ROOT
    / "experiments"
    / "inprogress"
    / "kspace_filling"
    / "experiment_11_sense_bridge_critic.yaml"
)


def _strategy_for(arm_path):
    """A real strategy carrying a real config and a recording logging service."""
    from types import SimpleNamespace

    from spectramr.config.settings import TrainingSettings

    assert arm_path.exists(), f"missing experiment config: {arm_path}"
    strategy = BaseTrainingStrategy.__new__(BaseTrainingStrategy)
    strategy.config = TrainingSettings.from_yaml(str(arm_path))
    strategy.logged = []
    strategy.logging_service = SimpleNamespace(
        log_info=lambda m: strategy.logged.append(("INFO", m)),
        log_warning=lambda m: strategy.logged.append(("WARN", m)),
    )
    return strategy


@pytest.fixture
def sense_bridge_strategy():
    return _strategy_for(_SENSE_BRIDGE_ARM)


class TestDeclaredLossKeysAreDerived:
    """The set the strategy must guarantee comes from the weight SSOT, and the
    builder's own skip set decides what it is NOT responsible for."""

    def test_the_domain_lists_are_seen(self, sense_bridge_strategy) -> None:
        """The lists this replaced read `losses.reconstruction` only, so this
        arm — which declares its objective in `kspace_losses`/`image_losses` —
        produced an EMPTY set and guaranteed no CSV column at all."""
        keys = sense_bridge_strategy._declared_loss_keys
        assert {"complex_l1", "log_spectral", "sense_adjoint_l1", "hfen"} <= keys

    def test_builder_owned_losses_are_not_this_strategys_job(self, sense_bridge_strategy) -> None:
        """Subtracting the builder skip set is what stops a per-step phantom
        warning: `pre_dc_kspace` is declared and active, but the builder owns
        it and it reaches the CSV under a renamed key."""
        from spectramr.infrastructure.training.builders.loss_builder import (
            STRATEGY_MANAGED_LOSSES,
        )
        from spectramr.models.losses.reporting import active_loss_names

        keys = sense_bridge_strategy._declared_loss_keys
        assert not (keys & STRATEGY_MANAGED_LOSSES)
        declared = active_loss_names(sense_bridge_strategy._weight_table)
        assert "pre_dc_kspace" in declared, "fixture drift: arm no longer declares it"
        assert "pre_dc_kspace" not in keys

    def test_the_sets_are_cached_not_rebuilt_per_step(self, sense_bridge_strategy) -> None:
        """Non-negotiable 9 — this is read on the per-step path."""
        first = sense_bridge_strategy._declared_loss_keys
        assert sense_bridge_strategy._declared_loss_keys is first
        gated = sense_bridge_strategy._warmup_gated_loss_keys
        assert sense_bridge_strategy._warmup_gated_loss_keys is gated

    def test_they_survive_a_subclass_that_skips_setup(self) -> None:
        """Lazy `getattr` caching, not attributes assigned in setup: ten
        subclasses override `_setup_strategy_specific_components` without
        calling `super()`, and a missing attribute here is an AttributeError
        on the first training step."""
        strategy = _strategy_for(_SENSE_BRIDGE_ARM)
        assert not hasattr(strategy, "_declared_loss_key_cache")
        assert strategy._declared_loss_keys  # must build on first touch


class TestZeroFillAndReporting:
    """Fill silently, warn only on an absence the warm-up gate cannot explain."""

    @staticmethod
    def _produced(strategy, *, drop=(), add=()):
        keys = (set(strategy._declared_loss_keys) - set(drop)) | set(add)
        return {k: torch.zeros(()) for k in keys} | {"g_total_loss": torch.zeros(())}

    def test_a_gated_loss_absent_during_warmup_is_filled_without_warning(
        self, sense_bridge_strategy
    ) -> None:
        """`adversarial` is declared active but its branch is skipped while the
        gate holds. Warning here would emit one line per step for the whole
        warm-up — 1000 per run, on 58 arms, every one of them false."""
        gated = sense_bridge_strategy._warmup_gated_loss_keys
        assert "adversarial" in gated, "fixture drift: adversarial is not gated"

        losses = self._produced(sense_bridge_strategy, drop=gated)
        filled = sense_bridge_strategy._ensure_declared_losses_present(losses, iteration=0)

        assert filled == gated
        assert all(k in losses for k in gated), "the CSV column must exist from step 0"
        assert not [m for lvl, m in sense_bridge_strategy.logged if lvl == "WARN"]

    def test_the_same_absence_after_warmup_is_reported(self, sense_bridge_strategy) -> None:
        """Past the gate there is no excuse left, so it becomes a real finding."""
        gated = sense_bridge_strategy._warmup_gated_loss_keys
        losses = self._produced(sense_bridge_strategy, drop=gated)
        warmup = sense_bridge_strategy._weight_table.warmup_iterations

        sense_bridge_strategy._ensure_declared_losses_present(losses, iteration=warmup)
        warnings_ = [m for lvl, m in sense_bridge_strategy.logged if lvl == "WARN"]
        assert len(warnings_) == 1
        assert "adversarial" in warnings_[0]

    def test_an_unexplained_drop_is_reported_once_not_per_step(self, sense_bridge_strategy) -> None:
        """Non-negotiable 9: the training loop runs this every step."""
        for step in range(5):
            losses = self._produced(sense_bridge_strategy, drop={"complex_l1"})
            sense_bridge_strategy._ensure_declared_losses_present(losses, iteration=step)
        warnings_ = [m for lvl, m in sense_bridge_strategy.logged if lvl == "WARN"]
        assert len(warnings_) == 1, f"one warning per step: {warnings_}"
        assert "complex_l1" in warnings_[0]

    def test_a_new_defect_is_reported_even_after_an_earlier_one(
        self, sense_bridge_strategy
    ) -> None:
        """Warn-once must key on the SET, or the second drop is swallowed."""
        for drop in ({"complex_l1"}, {"complex_l1", "hfen"}):
            losses = self._produced(sense_bridge_strategy, drop=drop)
            sense_bridge_strategy._ensure_declared_losses_present(losses, iteration=5000)
        warnings_ = [m for lvl, m in sense_bridge_strategy.logged if lvl == "WARN"]
        assert len(warnings_) == 2
        assert "hfen" in warnings_[1]

    def test_nothing_is_filled_when_every_declared_loss_arrived(
        self, sense_bridge_strategy
    ) -> None:
        losses = self._produced(sense_bridge_strategy)
        before = dict(losses)
        filled = sense_bridge_strategy._ensure_declared_losses_present(losses, iteration=0)
        assert filled == frozenset()
        assert losses.keys() == before.keys()
        assert not sense_bridge_strategy.logged

    def test_presence_is_tested_on_the_canonical_name(self, sense_bridge_strategy) -> None:
        """An arm declaring `mse` resolves to `l2` in the table while a computer
        may key either. Two `kspace_filling` arms do exactly this; a raw-string
        compare fills a duplicate key and reports a phantom miss every step."""
        from spectramr.models.losses.weights import canonical_loss_name

        assert canonical_loss_name("mse") == "l2"
        strategy = sense_bridge_strategy
        strategy._declared_loss_key_cache = frozenset({"l2"})
        strategy._warmup_gated_loss_key_cache = frozenset()

        losses = {"mse": torch.zeros(())}
        filled = strategy._ensure_declared_losses_present(losses, iteration=0)
        assert filled == frozenset(), "the alias was already present under `mse`"
        assert "l2" not in losses
        assert not strategy.logged

    def test_the_zero_matches_the_device_of_the_losses_it_joins(
        self, sense_bridge_strategy
    ) -> None:
        """A CPU scalar landing in a CUDA loss dict forces a transfer on the
        per-step logging path (non-negotiable 9).

        The anchor is on ``meta`` deliberately.  A cpu anchor cannot fail this
        assertion: the fill's own default IS cpu, so ``torch.as_tensor(0.0)``
        -- the defect -- and ``torch.zeros((), device=device)`` -- the fix --
        both land there and the test passes either way.  ``meta`` is a device
        the defect has no way to reach, which is what makes the check real.
        """
        strategy = sense_bridge_strategy
        strategy._declared_loss_key_cache = frozenset({"complex_l1"})
        strategy._warmup_gated_loss_key_cache = frozenset()

        anchor = torch.zeros((), device="meta")
        assert anchor.device != torch.zeros(()).device, (
            "meta must differ from the cpu default, or this test is vacuous"
        )
        losses = {"g_total_loss": anchor}
        strategy._ensure_declared_losses_present(losses, iteration=9999)
        assert losses["complex_l1"].device == anchor.device

    def test_an_arm_declaring_nothing_fills_nothing(self, sense_bridge_strategy) -> None:
        strategy = sense_bridge_strategy
        strategy._declared_loss_key_cache = frozenset()
        losses = {"g_total_loss": torch.zeros(())}
        assert strategy._ensure_declared_losses_present(losses, iteration=0) == frozenset()
        assert list(losses) == ["g_total_loss"]
