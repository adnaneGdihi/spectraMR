"""Tests for the ``run_training_pipeline`` env/strategy injection seam (WS-A).

When a caller (e.g. ``spectramr.api.fit``) supplies a pre-built
:class:`TrainingEnvironment`, the pipeline must skip the config-driven
``TrainingEnvironmentDirector`` and drive the SAME loop on the injected env.
The loop itself is stubbed here so these stay fast unit tests; what they pin is
the injection branch + the dict-config normalization (which previously crashed
because ``config.seed`` was read before the dict→settings cast).
"""

from __future__ import annotations

from types import SimpleNamespace

import json
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import spectramr.pipelines.train as train_mod
import spectramr.pipelines.training_loop as training_loop_mod
from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment


def _config_dict(tmp_path) -> dict:
    return {
        "model": {"model_type": "unet"},
        "data": {"dataset_type": "synthetic"},
        "optimization": {},
        "logging": {},
        "checkpoint": {},
        "losses": {
            "output_domain": "image",
            "image_losses": [{"name": "l1", "weight": 1.0}],
        },
        "training": {
            "strategy_class": "reconstruction",
            "output_dir": str(tmp_path),
            "epochs": 1,
        },
    }


def _env(cfg) -> TrainingEnvironment:
    model = nn.Conv2d(2, 2, 3, padding=1)
    loader = DataLoader(TensorDataset(torch.randn(2, 2, 8, 8)), batch_size=1)
    return TrainingEnvironment.from_components(
        config=cfg,
        models={"generator": model},
        optimizers={"opt_g": torch.optim.Adam(model.parameters())},
        losses={"l1": nn.L1Loss()},
        data_loaders={"train": loader, "val": loader},
        device="cpu",
    )


def test_injected_env_skips_director(monkeypatch, tmp_path):
    cfg = TrainingSettings.settings_from_dict(_config_dict(tmp_path))
    env = _env(cfg)

    def _boom(*a, **k):
        raise AssertionError("TrainingEnvironmentDirector must not run when env is injected")

    monkeypatch.setattr(train_mod, "TrainingEnvironmentDirector", _boom)
    monkeypatch.setattr(
        training_loop_mod,
        "_execute_training_loop",
        lambda *a, **k: {"success": True, "iterations_completed": 0},
    )

    result = train_mod.run_training_pipeline(cfg, env=env, device="cpu")
    assert result["success"] is True


def test_dict_config_is_normalized(monkeypatch, tmp_path):
    """A dict config must be normalized via settings_from_dict at the top (so
    ``config.seed`` no longer crashes) and reach the loop."""
    monkeypatch.setattr(train_mod, "TrainingEnvironmentDirector", lambda *a, **k: _RaiseOnBuild())
    monkeypatch.setattr(
        training_loop_mod,
        "_execute_training_loop",
        lambda *a, **k: {"success": True, "iterations_completed": 0},
    )
    cfg_dict = _config_dict(tmp_path)
    cfg = TrainingSettings.settings_from_dict(cfg_dict)
    env = _env(cfg)
    # Pass the raw DICT (not the settings object) — exercises the normalization.
    result = train_mod.run_training_pipeline(cfg_dict, env=env, device="cpu")
    assert result["success"] is True


def test_scheduler_cadence_reads_step_executor_not_dead_trainer():
    """The LR-scheduler-cadence gradient-accumulation read goes through
    ``strategy.step_executor`` (L1 post-merge fix). #131 renamed the attribute
    ``self.trainer`` → ``self.step_executor``; a stale
    ``getattr(strategy, "trainer", None)`` would always resolve to ``None`` — an
    inert dead read (pitfall #16) that silently falls back to the config value.
    """
    import inspect

    # The loop body moved to pipelines/training_loop.py (WS-3 PR-2), and the
    # read then moved AGAIN out of ``_execute_training_loop`` and into the
    # dedicated ``resolve_scheduler_cadence`` -- because that site now reads two
    # attributes off the executor, preferring the CONFIGURED
    # ``requested_gradient_accumulation_steps`` over the negotiated one that
    # DeepSpeed forces to 1. So this follows the read to its new home rather
    # than pinning the caller: what the docstring above is actually about is
    # that the receiver is ``step_executor`` and never the renamed-away
    # ``trainer``, and that is a property of the resolver.
    src = inspect.getsource(training_loop_mod.resolve_scheduler_cadence)
    assert 'getattr(strategy, "step_executor", None)' in src
    assert 'getattr(strategy, "trainer", None)' not in src

    # ...and the resolver is not orphaned. A correct read that nobody calls is
    # the same dead-read pitfall (#16) in a different place, which is exactly
    # what this test exists to catch.
    caller = inspect.getsource(training_loop_mod._execute_training_loop)
    assert "resolve_scheduler_cadence(strategy, config)" in caller
    assert 'getattr(strategy, "trainer", None)' not in caller


def test_startup_summary_is_emitted_before_the_logging_level_is_configured(monkeypatch, tmp_path):
    """The parallelism/knobs summary must precede ``bootstrap.build_container``.

    ``build_container`` runs ``LoggingService.setup``, which pushes
    ``logging.sinks.level`` onto the root logger, every existing logger AND
    every handler. Any INFO emitted after it is discarded on an arm setting
    ``level: warning`` -- which is what erased every configuration line from
    the attention_shootout runs. Ordering is the entire mechanism here, so it
    is what this pins; moving the call below the container silently restores
    the bug with no test failing on content alone.
    """
    order: list[str] = []
    real_build = train_mod.bootstrap.build_container

    def _record_summary(config, logger=None):
        order.append("summary")

    def _record_container(*args, **kwargs):
        order.append("container")
        return real_build(*args, **kwargs)

    monkeypatch.setattr(
        "spectramr.infrastructure.logging.provenance.log_startup_summary",
        _record_summary,
    )
    monkeypatch.setattr(train_mod.bootstrap, "build_container", _record_container)
    monkeypatch.setattr(
        training_loop_mod,
        "_execute_training_loop",
        lambda *a, **k: {"success": True, "iterations_completed": 0},
    )

    cfg = TrainingSettings.settings_from_dict(_config_dict(tmp_path))
    train_mod.run_training_pipeline(cfg, env=_env(cfg), device="cpu")

    assert order[:2] == ["summary", "container"], (
        f"startup summary must precede the container build, got {order}"
    )


def test_startup_summary_reports_parallelism_and_cost_knobs(monkeypatch, tmp_path, caplog):
    """Content check with the real renderer, complementing the ordering test.

    Both lines have to survive to the console: the parallelism line is the one
    the user asked for, and the knobs line is what makes an applied ``-O``
    checkable against what the arm declared.
    """
    monkeypatch.setattr(
        training_loop_mod,
        "_execute_training_loop",
        lambda *a, **k: {"success": True, "iterations_completed": 0},
    )
    cfg = TrainingSettings.settings_from_dict(_config_dict(tmp_path))
    with caplog.at_level("INFO", logger=train_mod.__name__):
        train_mod.run_training_pipeline(cfg, env=_env(cfg), device="cpu")

    messages = [r.getMessage() for r in caplog.records]
    parallel = [m for m in messages if m.startswith("parallel   :")]
    knobs = [m for m in messages if m.startswith("knobs      :")]
    assert parallel, f"no parallelism line in {messages[:12]}"
    assert knobs, f"no knobs line in {messages[:12]}"
    assert "world=" in parallel[0]
    assert "grad_ckpt=" in knobs[0]
    assert "amp=" in knobs[0]


class _RaiseOnBuild:
    def build_environment(self):  # pragma: no cover - should never be called
        raise AssertionError("director build must be skipped when env is injected")


# ---------------------------------------------------------------------------
# DDP validation-metric reduction
# ---------------------------------------------------------------------------
# Under DDP the val loader is wrapped in a DistributedSampler (shards + pads),
# so each rank's val_accum / val_count cover only its shard. _run_validation
# finalised ``v_sum / val_count`` on the LOCAL shard and never all-reduced, so
# rank-0 reported (and early-stopped on) one padded shard's metric. The fix
# sums per-rank metric sums and counts before dividing.


def test_all_reduce_val_metrics_noop_without_process_group():
    """On the default single-process path (no process group) the reducer is a
    no-op: returns the accumulators and count unchanged."""
    accum = {"psnr": 20.0, "l1": 4.0}
    out_accum, out_count = train_mod._all_reduce_val_metrics(accum, 7, torch.device("cpu"))
    assert out_count == 7
    assert out_accum == {"psnr": 20.0, "l1": 4.0}


def test_all_reduce_val_metrics_single_rank_identity():
    """With a real (world_size=1) process group the reducer runs the dist
    branch; all-reduce-SUM is the identity for a single rank, and tensor
    accumulators are coerced to float scalars."""
    import os
    import socket

    import torch.distributed as dist

    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    try:
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"could not init gloo process group: {exc}")
    try:
        accum = {"psnr": 20.0, "l1": torch.tensor(4.0)}
        out_accum, out_count = train_mod._all_reduce_val_metrics(accum, 8, torch.device("cpu"))
        assert out_count == 8
        assert out_accum["psnr"] == pytest.approx(20.0)
        assert out_accum["l1"] == pytest.approx(4.0)  # tensor coerced to float
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Validation 5D→4D preprocessing gated on spatial_dims (2026-07 ldm slab arms)
# ---------------------------------------------------------------------------
# The depth-into-batch flatten "for 2D networks" was unconditional, so the
# volumetric SLAT slab→volume VAE (spatial_dims=3) had its 5D [B,C,H,W,D] slab
# collapsed to [B*D,C,H,W]; the generator then re-inflated depth via d_out,
# desyncing the 5D pred from the flattened 4D target and tripping vae.py:340.


def _cfg(spatial_dims: int) -> SimpleNamespace:
    # patch_size H,W == input H,W so the square-pad step never fires here.
    return SimpleNamespace(
        model=SimpleNamespace(spatial_dims=spatial_dims),
        data=SimpleNamespace(patch_size=[32, 32, 3]),
    )


def test_preprocess_validation_tensor_keeps_5d_for_volumetric():
    """spatial_dims=3 (volumetric): the 5D slab is preserved so pred matches
    the 5D target — no depth-into-batch flatten."""
    t = torch.randn(1, 1, 32, 32, 3)
    out = train_mod._preprocess_validation_tensor(t, _cfg(spatial_dims=3))
    assert out.shape == (1, 1, 32, 32, 3)


def test_preprocess_validation_tensor_keeps_5d_depth1_for_volumetric():
    """The sparse_vae_slab arm has depth==1 (d_in=1). B*D==1 makes the flatten
    LOOK like a batch no-op, but it silently drops the trailing depth dim, so
    the generator re-inflates it (d_out=1) → 5D pred vs 4D target. Preserving
    the 5D slab keeps pred and target both [1,1,H,W,1]."""
    t = torch.randn(1, 1, 32, 32, 1)
    out = train_mod._preprocess_validation_tensor(t, _cfg(spatial_dims=3))
    assert out.shape == (1, 1, 32, 32, 1)


def test_preprocess_validation_tensor_flattens_5d_for_2d_net():
    """spatial_dims=2 (2D net): depth folds into batch, historical behaviour."""
    t = torch.randn(1, 1, 32, 32, 3)
    out = train_mod._preprocess_validation_tensor(t, _cfg(spatial_dims=2))
    assert out.shape == (3, 1, 32, 32)


def test_model_schema_defaults_spatial_dims_to_2_for_undeclared_2d_arms():
    """A 2D arm that never declares spatial_dims still flattens.

    ``_preprocess_validation_tensor`` now reads ``config.model.spatial_dims``
    directly (SSOT #368) instead of a getattr(..., 2) fallback -- the default
    ``2`` lives in the schema, not the pipeline. So an arm that omits the key
    still gets 2 from ``ModelConfigSchema`` and its 5D slab folds into the batch
    (that flatten is covered by test_..._flattens_5d_for_2d_net above). This pins
    the guarantee that makes the direct read safe."""
    from spectramr.config.schemas.model import ModelConfigSchema

    assert ModelConfigSchema().spatial_dims == 2


class TestVisualCaptureMissEscalation:
    """A run that saves no validation image must eventually say so at ERROR level.

    `mrixfields_field_cocycle_ablate_cocycle` logged the per-pass WARNING 49 times
    across an 8-hour run, saved zero images, and still reported ``status: OK`` — the
    crash detectors key on the training loss, so an image-less run is pitfall #10 at
    the artefact layer. The contact sheet then showed a PREVIOUS run's pictures.
    """

    def test_limit_is_small_enough_to_catch_a_broken_seam(self) -> None:
        from spectramr.pipelines.train import _VISUAL_CAPTURE_MISS_LIMIT

        # One miss can be a fluke (an odd batch); a handful in a row cannot.
        assert 2 <= _VISUAL_CAPTURE_MISS_LIMIT <= 5

    def test_the_counter_survives_a_frozen_carrier(self) -> None:
        """The escalation must work against the objects production actually has.

        The sibling test below proves the ARITHMETIC on a ``SimpleNamespace``,
        which accepts any attribute. Production does not: the counters were
        being set on ``pipeline``, the ``TrainingEnvironment``, which is a
        ``@dataclass(frozen=True)``. Every assignment raised

            FrozenInstanceError: cannot assign to field '_visual_capture_misses'

        and since the assignment sits INSIDE the "we wanted images and got none"
        branch, the instrumentation meant to surface a silent skip crashed the
        run at the exact moment it was supposed to report one. Seven of the
        eleven paradigms in ``tests/smoke/test_fit_paradigms_smoke.py`` died on
        it, while the arithmetic test stayed green -- a mutable stand-in cannot
        fail the way the real carrier does.

        So this exercises ``_visual_capture_state`` against a frozen dataclass
        AND asserts the state persists across calls, which is what makes a
        *consecutive* count meaningful.
        """
        import dataclasses

        from spectramr.pipelines.train import (
            _VISUAL_CAPTURE_MISS_LIMIT,
            _visual_capture_state,
        )

        @dataclasses.dataclass(frozen=True)
        class FrozenCarrier:
            name: str = "env"

        carrier = FrozenCarrier()
        # The old code did `carrier._visual_capture_misses = 1` here and raised.
        with pytest.raises(dataclasses.FrozenInstanceError):
            carrier._visual_capture_misses = 1

        for expected in range(1, _VISUAL_CAPTURE_MISS_LIMIT + 1):
            state = _visual_capture_state(carrier)
            state["misses"] = state.get("misses", 0) + 1
            assert state["misses"] == expected, "state must persist across calls"

        _visual_capture_state(carrier)["misses"] = 0
        assert _visual_capture_state(carrier)["misses"] == 0

        # Two carriers must not share a counter, or one arm's misses would
        # escalate against another's.
        other = FrozenCarrier(name="other")
        _visual_capture_state(other)["misses"] = 7
        assert _visual_capture_state(carrier)["misses"] == 0

    def test_consecutive_misses_reach_the_limit_and_a_hit_resets(self) -> None:
        """Pins the counter contract the escalation branch in _run_validation uses."""
        from types import SimpleNamespace

        from spectramr.pipelines.train import _VISUAL_CAPTURE_MISS_LIMIT

        pipeline = SimpleNamespace()
        escalations = 0
        # Three misses, one successful capture, then three more misses.
        for captured in (
            [False] * _VISUAL_CAPTURE_MISS_LIMIT + [True] + [False] * _VISUAL_CAPTURE_MISS_LIMIT
        ):
            if not captured:
                misses = getattr(pipeline, "_visual_capture_misses", 0) + 1
                pipeline._visual_capture_misses = misses
                if misses == _VISUAL_CAPTURE_MISS_LIMIT:
                    escalations += 1
            else:
                pipeline._visual_capture_misses = 0
        # Escalates once per unbroken run of misses, not once per miss.
        assert escalations == 2


class TestOutputSanityEscalation:
    """The validation-output degeneracy guard wired into ``_run_validation``.

    Companion to :class:`TestVisualCaptureMissEscalation`: that one catches a run
    that saves NO images, this one catches a run that saves images of nothing.
    Eight arms of the 2026-07 MRIxFields2026 cohort shipped ``success: True``
    with unusable pictures — full-frame speckle, white-out, or a dead black
    frame — because SSIM/PSNR grade agreement with the reference and never ask
    whether the prediction renders air as air (pitfall #20).
    """

    def test_limit_matches_the_visual_capture_convention(self) -> None:
        from spectramr.pipelines.train import (
            _OUTPUT_SANITY_MISS_LIMIT,
            _VISUAL_CAPTURE_MISS_LIMIT,
        )

        # One degenerate pass can catch a model mid-transient; a run of them cannot.
        assert 2 <= _OUTPUT_SANITY_MISS_LIMIT <= 5
        assert _OUTPUT_SANITY_MISS_LIMIT == _VISUAL_CAPTURE_MISS_LIMIT

    def test_consecutive_degenerate_passes_escalate_and_a_clean_pass_resets(
        self,
    ) -> None:
        """Pins the counter contract the escalation branch uses."""
        from types import SimpleNamespace

        from spectramr.pipelines.train import _OUTPUT_SANITY_MISS_LIMIT

        pipeline = SimpleNamespace()
        escalations = 0
        pattern = [True] * _OUTPUT_SANITY_MISS_LIMIT + [False] + [True] * _OUTPUT_SANITY_MISS_LIMIT
        for degenerate in pattern:
            if degenerate:
                bad = getattr(pipeline, "_output_sanity_misses", 0) + 1
                pipeline._output_sanity_misses = bad
                if bad == _OUTPUT_SANITY_MISS_LIMIT:
                    escalations += 1
            else:
                pipeline._output_sanity_misses = 0
        assert escalations == 2

    def test_the_guard_is_importable_from_the_pipeline_module(self) -> None:
        """The check must be reachable where validation runs, not only in tests —
        a guard that lives only in ``core/metrics`` protects nothing."""
        import spectramr.pipelines.train as train_mod

        assert hasattr(train_mod, "measure_output_sanity")


# ---------------------------------------------------------------------------
# Provenance stamps the RESOLVED device (accelerated-run contract, 9b)
# ---------------------------------------------------------------------------
# ``collect_run_provenance`` is called before the environment exists, so the
# only device it can see is the REQUESTED one — the CLI's ``--device``, which is
# None unless the caller passed it. The dispatcher never does, so every cluster
# run of the ldm_two_stage_ulf_to_hf cohort wrote ``"device": null`` while
# training on a Tesla V100 (SLURM 7796517, 2026-07-25). Provenance exists to
# answer "did this run on an accelerator?"; null cannot answer it.


def _provenance_from_run(tmp_path, monkeypatch, *, requested_device):
    """Run the pipeline with an injected cpu env and return its provenance.json.

    ``device: cpu`` is declared explicitly rather than left to the schema default
    (``"cuda"``). Without it, the pipeline asks the resolver for cuda and, on any
    CPU-only host, ``AcceleratorRequiredError`` fires — the accelerated-run
    contract (9b) working exactly as designed, reported as a test failure. The
    test read ambient GPU availability instead of pinning it, so it passed on a
    GPU dev box and failed on every cluster node (#630). Declaring the documented
    CPU opt-in also puts the ``cpu_opt_in`` branch under test, which marking the
    test ``gpu`` would merely have skipped.
    """
    import json

    # ``run.device``, not a root-level ``device``: the bare scalar was renamed on
    # 2026-07-31 (``renames.py:509``) and now RAISES rather than folding, so this
    # helper -- and both device tests below it -- had been failing on a
    # ValidationError that never reached the behaviour under test.
    cfg = TrainingSettings.settings_from_dict({**_config_dict(tmp_path), "run": {"device": "cpu"}})
    env = _env(cfg)  # TrainingEnvironment.from_components(..., device="cpu")
    monkeypatch.setattr(train_mod, "TrainingEnvironmentDirector", lambda *a, **k: _RaiseOnBuild())
    monkeypatch.setattr(
        training_loop_mod,
        "_execute_training_loop",
        lambda *a, **k: {"success": True, "iterations_completed": 0},
    )
    train_mod.run_training_pipeline(cfg, env=env, device=requested_device)

    stamps = list(tmp_path.rglob("provenance.json"))
    assert stamps, f"no provenance.json written under {tmp_path}"
    return json.loads(stamps[0].read_text())


def test_provenance_stamps_resolved_device(tmp_path, monkeypatch):
    """The regression: ``--device`` unset must NOT leave ``device: null``.

    The injected environment resolves to cpu, so that is what the record must
    carry — not the None the caller passed in.
    """
    prov = _provenance_from_run(tmp_path, monkeypatch, requested_device=None)
    assert prov["device"] is not None, "provenance still records the requested device"
    assert "cpu" in prov["device"]


def test_provenance_device_follows_env_not_request(tmp_path, monkeypatch):
    """Even a MISMATCHED request loses to the resolved device.

    Stamping the request would let a run claim ``cuda`` while executing on cpu —
    exactly the silent-degradation the contract forbids.
    """
    prov = _provenance_from_run(tmp_path, monkeypatch, requested_device="cuda")
    assert "cpu" in prov["device"], "record must follow the environment, not the ask"


class TestProvenanceDataCounts:
    """A count without a unit is not an answer.

    The record said ``data: {train: 768}``. The user compared it to a folder of
    1024 files and reported 25 % of the data missing. Nothing was missing:
    ``1024 files -> 384 (patient, contrast) groups -> x4 samples_per_volume =
    1536 patches -> /2 batch_size (drop_last) = 768 batches``. Four units, one
    unlabelled number.
    """

    def test_the_record_carries_units_not_a_bare_int(self, tmp_path, monkeypatch):
        """End-to-end: the artifact a run leaves behind must be self-describing."""
        prov = _provenance_from_run(tmp_path, monkeypatch, requested_device="cpu")
        data = prov["data"]
        assert set(data) == {"train", "val"}
        for split, counts in data.items():
            assert isinstance(counts, dict), f"{split} is still a bare {type(counts).__name__}"
            # the injected env is 2 samples at batch_size=1
            assert counts["batches"] == 2, counts
            assert counts["samples"] == 2, counts

    def test_data_counts_are_not_gated_on_a_model_being_built(self):
        """The parameter count needs a generator; the data counts do not.

        Both sat behind ``if provenance and generator is not None``, so the
        in-process ``env=`` entry point -- the one path that reaches here without
        building a model -- lost its data counts and the banner with them.
        "Fixed for all strategies" has to mean every entry point, so this is
        asserted structurally rather than left to the CLI path that happens to
        always have a generator.
        """
        import ast
        import inspect

        from spectramr.pipelines import train

        tree = ast.parse(inspect.getsource(train))

        def _guards(target: str) -> list[list[str]]:
            """One entry per assignment to *target*: its enclosing ``if`` tests.

            A list-of-lists rather than a flat list, because an assignment under
            no guard at all yields an EMPTY guard list -- and that is precisely
            the state being asserted, so it must stay distinguishable from
            "no such assignment exists".
            """
            found: list[list[str]] = []

            def walk(node, guards):
                if isinstance(node, ast.Assign) and target in ast.unparse(node):
                    found.append(list(guards))
                if isinstance(node, ast.If):
                    inner = [*guards, ast.unparse(node.test)]
                    for stmt in node.body:
                        walk(stmt, inner)
                    for stmt in node.orelse:
                        walk(stmt, guards)
                    return  # the test expression itself holds no assignments
                for child in ast.iter_child_nodes(node):
                    walk(child, guards)

            walk(tree, [])
            return found

        data_guards = _guards("provenance['data']")
        model_guards = _guards("provenance['model']")
        assert data_guards, "no provenance['data'] assignment found at all"
        assert not any("generator" in g for gs in data_guards for g in gs), (
            f"data counts are still gated on a model: {data_guards}"
        )
        assert any("generator" in g for gs in model_guards for g in gs), (
            f"the parameter count must still require a generator: {model_guards}"
        )


# ── validation-image windowing ────────────────────────────────────────────
def test_percentile_window_renders_a_sparse_image() -> None:
    """A mostly-zero image must render, not silently come out black.

    Regression for the 2026-07 exp_vf_01 run. The saved validation panels are
    masked to the anatomical object support, so most of the frame is exactly
    zero. The windowing helper took the 0.5th and 99.5th percentiles and, when
    that span collapsed, returned ``zeros`` — so ``val/predictions`` and
    ``val/targets`` were written as pure black PNGs while the run reported
    success. The run log caught it in the act::

        pred_mag range before norm:   [0.0000, 14.8116] -> after: [0.0000, 0.0000]
        target_mag range before norm: [0.0000, 92.7404] -> after: [0.0000, 0.0000]

    Emitting a black frame for a non-constant tensor is a silent fallback
    (CLAUDE.md #9): it looks like a rendered image and hides the signal.
    """
    img = torch.zeros(2, 1, 32, 32)
    img[:, :, 15:17, 15:17] = 5.0  # 0.4% of the frame carries all the signal

    out = train_mod._percentile_window(img)

    assert out.shape == img.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert float(out.max()) > 0.0, "a non-constant image must not render black"
    # the signal region is the bright one, the masked-out background stays dark
    assert float(out[:, :, 15:17, 15:17].min()) > float(out[:, :, 0, 0].max())


def test_percentile_window_keeps_black_for_a_constant_image() -> None:
    """A genuinely constant frame has no window to stretch — black is correct."""
    assert float(train_mod._percentile_window(torch.full((1, 1, 8, 8), 3.0)).max()) == 0.0


def test_percentile_window_still_clips_outliers_on_a_dense_image() -> None:
    """The percentile window is retained where it works (DC spikes stay clipped)."""
    img = torch.rand(1, 1, 32, 32)
    img[:, :, 0, 0] = 1000.0  # single hot voxel
    out = train_mod._percentile_window(img)
    # the bulk must still use most of the dynamic range, not be squashed to ~0
    assert float(out.median()) > 0.1


# ---------------------------------------------------------------------------
# resolved_config.json carries the DELTA, not just the final state (_ledger)
# ---------------------------------------------------------------------------
# The artifact always recorded what the run resolved TO. It never recorded what
# the declaration became on the way there, so a key the schema dropped left no
# trace anywhere: `min_center_fraction` went missing under extra="ignore" and the
# declared 32x rung realised 12.2x with every artifact reporting success
# (#534/#550).


def _resolved_config_from_run(tmp_path, monkeypatch, *, extra_config=None):
    """Run the real pipeline and return the parsed resolved_config.json."""
    import json

    from spectramr.core.execution_ledger import ExecutionLedger

    raw = _config_dict(tmp_path)
    if extra_config:
        raw.update(extra_config)

    ExecutionLedger.begin_run(source="test")
    cfg = TrainingSettings.settings_from_dict(raw)
    env = _env(cfg)
    monkeypatch.setattr(train_mod, "TrainingEnvironmentDirector", lambda *a, **k: _RaiseOnBuild())
    monkeypatch.setattr(
        training_loop_mod,
        "_execute_training_loop",
        lambda *a, **k: {"success": True, "iterations_completed": 0},
    )
    train_mod.run_training_pipeline(cfg, env=env, device="cpu")

    stamps = list(tmp_path.rglob("resolved_config.json"))
    assert stamps, f"no resolved_config.json written under {tmp_path}"
    return json.loads(stamps[0].read_text())


def test_resolved_config_carries_a_ledger_block(tmp_path, monkeypatch):
    """The run must declare that it recorded, so silence is provable."""
    payload = _resolved_config_from_run(tmp_path, monkeypatch)

    assert "_ledger" in payload, "the run left no record of its substitutions"
    ledger = payload["_ledger"]
    assert ledger["write_status"] == "ok"
    assert ledger["schema_version"] == 2, "v2 added the per-key `defaults` list"
    assert "substitutions" in ledger
    assert ledger["defaults_injected"] > 0, "a real config resolves defaults"
    # The count must stay derived from the list, not accumulated beside it.
    assert ledger["defaults_injected"] == len(ledger["defaults"])


def test_ledger_block_records_a_dropped_key_at_its_dotted_path(tmp_path, monkeypatch):
    """DEFECT: the #550 mechanism must be visible in the run's own artifact."""
    payload = _resolved_config_from_run(
        tmp_path,
        monkeypatch,
        extra_config={"acceleration": {"center_fraction": 0.08, "min_centre_fraction": 0.02}},
    )

    dropped = [
        s for s in payload["_ledger"]["substitutions"] if s["class_id"] == "extra_ignore_dropped"
    ]
    assert [s["path"] for s in dropped] == ["acceleration.min_centre_fraction"]
    assert dropped[0]["requested"] == 0.02
    assert dropped[0]["severity"] == "error"


def test_clean_config_records_no_dropped_key(tmp_path, monkeypatch):
    """CONTROL: proves the artifact is not simply always reporting drops."""
    payload = _resolved_config_from_run(tmp_path, monkeypatch)

    dropped = [
        s for s in payload["_ledger"]["substitutions"] if s["class_id"] == "extra_ignore_dropped"
    ]
    assert dropped == [], f"clean config reported drops: {[s['path'] for s in dropped]}"


def test_ledger_persists_the_otherwise_discarded_health_report(tmp_path, monkeypatch):
    """``validate_config_health`` runs every run and its report was thrown away.

    It already classifies findings under ``category="silent_fallback"``, so
    dropping it meant a run's own fallback findings vanished as it started.
    """
    payload = _resolved_config_from_run(tmp_path, monkeypatch)

    assert payload["_ledger"]["health_report"] is not None, (
        "the health report is computed on every run and must now survive it"
    )
    assert "passed" in payload["_ledger"]["health_report"]


def test_existing_readers_still_parse_the_enhanced_artifact(tmp_path, monkeypatch):
    """``_ledger`` is additive: cohort_ablation reads metadata.* as raw scalars.

    An inline per-key annotation would have broken every existing consumer.
    """
    from spectramr.infrastructure.reporting.cohort_ablation import _read_resolved_config

    payload = _resolved_config_from_run(tmp_path, monkeypatch)
    stamps = list(tmp_path.rglob("resolved_config.json"))
    reread = _read_resolved_config(stamps[0].parent)

    assert reread, "the canonical reader could not parse the enhanced artifact"
    for key in payload:
        if key != "_ledger":
            assert key in reread, f"top-level key {key!r} was lost"


class TestParallelProvenanceStamp:
    """The config says what was ASKED for; only the runtime knows what was BUILT.

    Every plugin already computed this record during ``adopt`` and it was
    thrown away, so an FSDP run and a single-process run produced byte-identical
    provenance -- and the one question a multi-GPU run's record has to answer
    ("did this actually shard?") had no answer on disk.
    """

    @staticmethod
    def _source():
        import inspect

        from spectramr.pipelines import train

        return inspect.getsource(train)

    def test_the_resolved_runtime_is_stamped(self):
        source = self._source()
        assert 'provenance["parallel"]' in source

    def test_it_is_stamped_beside_the_resolved_device(self):
        """Same reason, same place: both are facts only the built pipeline has,
        and the record is assembled before the pipeline exists."""
        source = self._source()
        assert source.index('provenance["device"]') < source.index('provenance["parallel"]')

    def test_it_falls_back_to_the_strategy_name(self):
        """A plugin returning an empty provenance dict must still record WHICH
        strategy ran, or the stamp is indistinguishable from absent."""
        assert 'getattr(parallel_runtime, "strategy", None)' in self._source()


def test_train_resolves_target_size_from_patch_size_only() -> None:
    """`_resize_to_target`'s `elif hasattr(data_cfg, "image_size")` branch was
    doubly dead: the field is undeclared on an extra="ignore" block, and
    `migrate_legacy_sizes` already folds img_size/target_size/image_size into
    `patch_size`. Re-declaring it would resurrect the spelling the migration
    exists to retire."""
    import ast
    import inspect

    from spectramr.config.schemas.data import DataConfigSchema
    from spectramr.pipelines import train

    assert "image_size" not in DataConfigSchema.model_fields
    # The canonical key the migration folds into is still there.
    assert "patch_size" in DataConfigSchema.model_fields

    # AST, not a substring: the removal left an explanatory comment that quotes
    # the old `hasattr(data_cfg, "image_size")` probe verbatim, and a grep-based
    # assertion matches its own documentation.
    tree = ast.parse(inspect.getsource(train))
    probes = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == "image_size"]
    assert not probes, (
        "`image_size` is referenced as a live string in pipelines/train.py at "
        f"lines {[n.lineno for n in probes]} — the branch was supposed to go, "
        "and re-adding the key would resurrect the spelling migrate_legacy_sizes "
        "exists to retire."
    )


# ---------------------------------------------------------------------------
# final_metrics.json `best` block: one direction resolver, no fallback guess.
#
# `_column_higher_is_better` used to re-implement the SSOT lookup by scanning
# METRIC_HIGHER_IS_BETTER for the longest entry appearing as a raw character
# SUBSTRING — so `mad` matched inside `made`. Below it sat a 6-substring fallback
# (`psnr`/`ssim`/`accuracy`/`dice`/`f1`/`iou`) that fired exactly where the SSOT
# had declined to resolve, i.e. where nothing justified a flip to maximize.
# ---------------------------------------------------------------------------


class TestBestColumnDirectionUsesTheSSOT:
    @pytest.mark.parametrize(
        ("column", "expected"),
        [
            ("val_psnr", True),
            ("val_lpips", False),
            ("val_robust_mri_psnr_2x", True),
            ("g_total_loss", False),
            ("ssim_loss", False),  # a loss, not the SSIM metric
        ],
    )
    def test_declared_columns_resolve(self, column, expected):
        from spectramr.pipelines.train import _column_higher_is_better

        assert _column_higher_is_better(column) is expected

    def test_raw_substring_false_positive_is_gone(self):
        """`train_made_up` used to resolve via `mad` inside `made`.

        The SSOT matches whole underscore-delimited token runs, so an invented
        column now returns None instead of borrowing MAD's direction.
        """
        from spectramr.pipelines.train import _column_higher_is_better

        assert _column_higher_is_better("train_made_up") is None

    @pytest.mark.parametrize("column", ["lr", "grad_norm"])
    def test_non_metric_columns_are_undeclared(self, column):
        from spectramr.pipelines.train import _column_higher_is_better

        assert _column_higher_is_better(column) is None


class TestSummariseBestMetricsFromCsv:
    @staticmethod
    def _write(tmp_path, rows):
        import csv

        path = tmp_path / "training_metrics.csv"
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return path

    def test_metric_takes_max_and_loss_takes_min(self, tmp_path):
        from spectramr.pipelines.train import _summarise_best_metrics_from_csv

        path = self._write(
            tmp_path,
            [
                {"iteration": 10, "epoch": 1, "val_psnr": 20.0, "g_total_loss": 5.0},
                {"iteration": 20, "epoch": 1, "val_psnr": 30.0, "g_total_loss": 2.0},
            ],
        )
        best = _summarise_best_metrics_from_csv(str(path), final_iteration=20)
        assert best["val_psnr_best"] == pytest.approx(30.0)
        assert best["g_total_loss_best"] == pytest.approx(2.0)

    def test_a_loss_named_after_a_metric_is_still_minimized(self, tmp_path):
        """The regression: `ssim_loss_best` reported the WORST (max) loss."""
        from spectramr.pipelines.train import _summarise_best_metrics_from_csv

        path = self._write(
            tmp_path,
            [
                {"iteration": 10, "epoch": 1, "ssim_loss": 0.9},
                {"iteration": 20, "epoch": 1, "ssim_loss": 0.1},
            ],
        )
        best = _summarise_best_metrics_from_csv(str(path), final_iteration=20)
        assert best["ssim_loss_best"] == pytest.approx(0.1)

    def test_undeclared_columns_get_no_best_rather_than_a_guess(self, tmp_path):
        """A 'best grad_norm' is not a quantity; omitting it beats inventing it."""
        from spectramr.pipelines.train import _summarise_best_metrics_from_csv

        path = self._write(
            tmp_path,
            [
                {"iteration": 10, "epoch": 1, "val_psnr": 20.0, "grad_norm": 3.0},
                {"iteration": 20, "epoch": 1, "val_psnr": 30.0, "grad_norm": 1.0},
            ],
        )
        best = _summarise_best_metrics_from_csv(str(path), final_iteration=20)
        assert "val_psnr_best" in best
        assert "grad_norm_best" not in best


# ---------------------------------------------------------------------------
# #481: validation numbers never reached the surfaces a triager reads.
#
# `training_metrics.csv` DECLARED val_* columns and left every one empty; the real
# numbers sat in a separate `validation_metrics.csv` that nothing aggregated. So
# `final_metrics.best` carried only train_* keys and `run_summary.best_metrics`
# was null — on runs where validation ran and produced val_psnr ~6 dB against
# train_psnr ~30 dB. The metric was computed; the surface never showed it.
# ---------------------------------------------------------------------------


class TestValidationCsvDerivation:
    def test_name_swap_keeps_a_suffixed_pair_together(self):
        from spectramr.pipelines.train import validation_csv_for

        got = validation_csv_for("/runs/a/logs/training_metrics_gan.csv")
        assert got.name == "validation_metrics_gan.csv"
        assert got.parent.as_posix() == "/runs/a/logs"

    def test_falls_back_to_a_sibling_when_the_name_does_not_match(self):
        from spectramr.pipelines.train import validation_csv_for

        got = validation_csv_for("/runs/a/logs/curve.csv")
        assert got.as_posix() == "/runs/a/logs/validation_metrics.csv"


class TestBestFoldsBothCsvs:
    @staticmethod
    def _write(path, fieldnames, rows):
        import csv

        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    def _make_pair(self, tmp_path):
        """Reproduce #481's shape: declared-but-empty val_* + a real val CSV."""
        from spectramr.pipelines.train import validation_csv_for

        train = tmp_path / "training_metrics.csv"
        self._write(
            train,
            ["iteration", "epoch", "g_total_loss", "train_psnr", "val_psnr"],
            [
                {
                    "iteration": 100,
                    "epoch": 1,
                    "g_total_loss": 0.5,
                    "train_psnr": 28.0,
                    "val_psnr": "",
                },
                {
                    "iteration": 200,
                    "epoch": 1,
                    "g_total_loss": 0.3,
                    "train_psnr": 30.1,
                    "val_psnr": "",
                },
            ],
        )
        self._write(
            validation_csv_for(train),
            ["iteration", "epoch", "val_psnr", "val_ssim"],
            [
                {"iteration": 100, "epoch": 1, "val_psnr": 6.2, "val_ssim": 0.01},
                {"iteration": 200, "epoch": 1, "val_psnr": 6.8, "val_ssim": 0.03},
            ],
        )
        return train

    def test_val_metrics_reach_the_best_block(self, tmp_path):
        from spectramr.pipelines.train import _summarise_best_metrics_from_csv

        best = _summarise_best_metrics_from_csv(str(self._make_pair(tmp_path)), final_iteration=200)
        # The alarming number a triager needs to see, previously invisible.
        assert best["val_psnr_best"] == pytest.approx(6.8)
        assert best["val_ssim_best"] == pytest.approx(0.03)
        # ...alongside the training numbers, which must not regress.
        assert best["train_psnr_best"] == pytest.approx(30.1)
        assert best["g_total_loss_best"] == pytest.approx(0.3)

    def test_empty_declared_val_columns_do_not_shadow_the_real_ones(self, tmp_path):
        """The training CSV's `val_psnr` column is all empty strings.

        If those were folded as values the key would be absent or wrong; the real
        value has to come from the validation CSV.
        """
        from spectramr.pipelines.train import _summarise_best_metrics_from_csv

        best = _summarise_best_metrics_from_csv(str(self._make_pair(tmp_path)), final_iteration=200)
        assert best["val_psnr_best"] == pytest.approx(6.8)

    def test_a_missing_validation_csv_is_not_an_error(self, tmp_path):
        """Validation-disabled arms must still get their training bests."""
        from spectramr.pipelines.train import _summarise_best_metrics_from_csv

        train = tmp_path / "training_metrics.csv"
        self._write(
            train,
            ["iteration", "epoch", "g_total_loss"],
            [{"iteration": 10, "epoch": 1, "g_total_loss": 0.9}],
        )
        best = _summarise_best_metrics_from_csv(str(train), final_iteration=10)
        assert best == {"g_total_loss_best": pytest.approx(0.9)}

    def test_the_run_window_is_applied_to_the_validation_csv_too(self, tmp_path):
        """#586's window must not be bypassed by the second file."""
        from spectramr.pipelines.train import (
            _summarise_best_metrics_from_csv,
            validation_csv_for,
        )

        train = tmp_path / "training_metrics.csv"
        self._write(
            train,
            ["iteration", "epoch", "train_psnr"],
            [{"iteration": 100, "epoch": 1, "train_psnr": 20.0}],
        )
        self._write(
            validation_csv_for(train),
            ["iteration", "epoch", "val_psnr"],
            [
                {"iteration": 100, "epoch": 1, "val_psnr": 25.0},
                # A later run's row appended to the same file.
                {"iteration": 900, "epoch": 9, "val_psnr": 99.0},
            ],
        )
        best = _summarise_best_metrics_from_csv(str(train), final_iteration=100)
        assert best["val_psnr_best"] == pytest.approx(25.0), (
            "a row beyond this run's final iteration leaked into best"
        )


class TestReportingHasOneEntryPoint:
    """``generate_report`` is the only end-of-training report path now.

    It used to be two, arranged backwards: ``MetricsReportGenerator`` ran from
    ``training_loop`` on EVERY run with no config gate at all, while the
    canonical ``generate_report`` sat behind ``reporting.enabled``, which
    defaults False. So the legacy generator always ran and the SSOT pipeline
    almost never did.
    """

    @staticmethod
    def _logger():
        from types import SimpleNamespace

        seen: list[tuple[str, tuple]] = []
        return seen, SimpleNamespace(
            info=lambda *a: seen.append(("info", a)),
            warning=lambda *a: seen.append(("warning", a)),
            debug=lambda *a: seen.append(("debug", a)),
        )

    def test_an_unconfigured_run_gets_the_tables_only_floor(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from spectramr.pipelines import train as train_mod

        calls: list[dict] = []
        monkeypatch.setattr(
            "spectramr.infrastructure.reporting.generate_report",
            lambda run_dir, **kw: (
                calls.append({"run_dir": run_dir, **kw})
                or {"tables": {"tab_run_summary": {"csv": tmp_path / "x.csv"}}}
            ),
        )
        _, logger_ = self._logger()
        train_mod._maybe_run_reporting(
            SimpleNamespace(reporting=None), run_dir=tmp_path, logger_=logger_
        )

        assert len(calls) == 1, "the floor did not run for an unconfigured arm"
        kw = calls[0]
        assert kw["figures"] == [], "the floor must emit no figures"
        assert kw["tables_"] == ["tab_run_summary"]
        assert kw["html_report"] is False and kw["interactive"] is False

    def test_reporting_disabled_still_gets_the_floor(self, tmp_path, monkeypatch):
        """`enabled: false` means "no figures", not "no artifacts at all"."""
        from types import SimpleNamespace

        from spectramr.pipelines import train as train_mod

        calls: list[dict] = []
        monkeypatch.setattr(
            "spectramr.infrastructure.reporting.generate_report",
            lambda run_dir, **kw: calls.append(kw) or {"tables": {}},
        )
        _, logger_ = self._logger()
        train_mod._maybe_run_reporting(
            SimpleNamespace(reporting=SimpleNamespace(enabled=False)),
            run_dir=tmp_path,
            logger_=logger_,
        )
        assert len(calls) == 1 and calls[0]["figures"] == []

    def test_the_floor_never_raises_into_the_training_wrapup(self, tmp_path, monkeypatch):
        """A report failure must not fail a finished run."""
        from types import SimpleNamespace

        from spectramr.pipelines import train as train_mod

        def _boom(*a, **k):
            raise RuntimeError("plotter exploded")

        monkeypatch.setattr("spectramr.infrastructure.reporting.generate_report", _boom)
        seen, logger_ = self._logger()
        train_mod._maybe_run_reporting(
            SimpleNamespace(reporting=None), run_dir=tmp_path, logger_=logger_
        )
        assert any(lvl == "warning" for lvl, _ in seen)

    def test_training_loop_no_longer_generates_a_report(self):
        """The ungated legacy call site is gone. The CLASS stays —
        ``scripts/render_full_reporting_pipeline.py`` still builds from it — so
        this asserts the CALL SITE, not the module's existence."""
        import inspect

        from spectramr.pipelines import training_loop

        src = inspect.getsource(training_loop)
        code = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
        assert "MetricsReportGenerator(" not in code, (
            "training_loop instantiates MetricsReportGenerator again — reporting "
            "has two entry points and the ungated one wins on every run."
        )


class TestAnAllNaNValidationRowFailsTheRun:
    """#181, the half `val_count == 0` cannot catch.

    That guard covers "every batch raised". This covers the case that gets
    through it: batches SUCCEED, so `val_count > 0` and nothing looks wrong,
    but every metric they produced is NaN.

    The state is unrecoverable rather than merely uninformative — a NaN monitor
    disarms `save_best` (gated on `math.isfinite`) and freezes `wait_count` in
    `EarlyStoppingService.update`, so the run trains to its full budget and
    exits 0 with no `checkpoint_best.pt`. That is the #178 outcome reached
    through a value.
    """

    @staticmethod
    def _guard_source() -> str:
        import inspect

        from spectramr.pipelines import train

        src = inspect.getsource(train)
        start = src.index('EVERY "\n                f"metric is non-finite')
        return src[max(0, start - 1600) : start + 900]

    def test_the_guard_raises(self):
        assert "raise RuntimeError" in self._guard_source()

    def test_it_fires_only_when_every_metric_is_non_finite(self):
        """One finite metric is enough to keep the run alive — individual NaNs
        are a per-metric matter the outcome contract already handles."""
        block = self._guard_source()
        assert "not any(" in block and "math.isfinite" in block

    def test_it_ignores_non_numeric_entries(self):
        """`avg_metrics` can carry non-numeric values; `math.isfinite` would
        raise on them and turn a diagnostic into a crash inside a crash."""
        block = self._guard_source()
        assert "isinstance(v, (int, float))" in block

    def test_an_empty_metrics_dict_does_not_raise(self):
        """`not any(...)` over an empty dict is True. Guarding on `_numeric`
        being non-empty keeps "measured nothing" distinct from "measured only
        NaN" — a strategy that emits no scalar metrics is not this defect."""
        block = self._guard_source()
        assert "if _numeric and not any(" in block

    def test_the_message_points_at_the_likely_cause(self):
        """An operator needs the next action, not just the verdict."""
        block = self._guard_source()
        assert "data_range" in block
        assert "NOT APPLICABLE" in block


class TestRunSummaryRecordsWhereCheckpointsWent:
    """#503 part 1: an empty `checkpoints/` cannot say WHY it is empty.

    exp_vf_01 retrieved a present-but-empty directory while its log named two
    files it had written. Both were true — the arm's `checkpoint_dir` pointed
    outside the collected run directory, so the files existed on the cluster and
    the bundle never contained them. "Written elsewhere" and "never written"
    need opposite responses (fix the sync vs fix the run), so the footer has to
    tell them apart.
    """

    @staticmethod
    def _summary(tmp_path, ckpt_dir):
        from types import SimpleNamespace

        from spectramr.pipelines.train import _checkpoint_summary

        cfg = SimpleNamespace(checkpoint=SimpleNamespace(checkpoint_dir=str(ckpt_dir)))
        return _checkpoint_summary(cfg, tmp_path)

    def test_a_dir_inside_the_run_is_flagged_inside(self, tmp_path):
        ck = tmp_path / "checkpoints"
        ck.mkdir()
        assert self._summary(tmp_path, ck)["inside_run_dir"] is True

    def test_a_dir_outside_the_run_is_flagged_outside(self, tmp_path):
        """The exp_vf_01 shape, and the one that matters."""
        outside = tmp_path.parent / "elsewhere_checkpoints"
        outside.mkdir(exist_ok=True)
        assert self._summary(tmp_path, outside)["inside_run_dir"] is False

    def test_files_present_are_listed_with_sizes(self, tmp_path):
        ck = tmp_path / "checkpoints"
        ck.mkdir()
        (ck / "checkpoint_best.pt").write_bytes(b"x" * 11)
        (ck / "notes.txt").write_text("not a checkpoint")

        out = self._summary(tmp_path, ck)

        assert [f["name"] for f in out["files"]] == ["checkpoint_best.pt"]
        assert out["total_bytes"] == 11

    def test_an_empty_dir_is_distinguishable_from_an_absent_one(self, tmp_path):
        """Empty-with-a-path and no-path-at-all are different diagnoses."""
        ck = tmp_path / "checkpoints"
        ck.mkdir()
        empty = self._summary(tmp_path, ck)
        assert empty["dir"] is not None and empty["files"] == []

        from types import SimpleNamespace

        from spectramr.pipelines.train import _checkpoint_summary

        absent = _checkpoint_summary(SimpleNamespace(checkpoint=None), tmp_path)
        assert absent["dir"] is None

    def test_it_never_raises(self, tmp_path):
        """A footer diagnostic must not cost a finished run its summary."""
        from types import SimpleNamespace

        from spectramr.pipelines.train import _checkpoint_summary

        class _Explodes:
            @property
            def checkpoint_dir(self):
                raise OSError("network filesystem stat failed")

        out = _checkpoint_summary(SimpleNamespace(checkpoint=_Explodes()), tmp_path)
        assert out["dir"] is None

    def test_the_summary_carries_the_block(self):
        """Anti-vacuity: the helper is useless if nothing stamps it."""
        import inspect

        from spectramr.pipelines import train

        src = inspect.getsource(train._emit_run_summary)
        assert '"checkpoints": _checkpoint_summary(config, run_dir)' in src


# ---------------------------------------------------------------------------
# _write_hparams (2026-08-12)
#
# The helper read ``optimization.precision.use_amp`` -- a field retired by the
# optimization-block decomposition (``RENAMES``: ``optimization.use_amp`` ->
# ``optimization.precision.enabled``). Nothing exercised it against a resolved
# config, so the AttributeError only surfaced on the cluster, where it killed
# the run twice: the success-path call raised, the handler logged "Critical
# Training Error", and the handler's own call raised the same error uncaught.
#
# These resolve a REAL ``TrainingSettings`` rather than a stub namespace --
# a stub answers to any attribute name and would have passed on the broken
# spelling.
# ---------------------------------------------------------------------------


class _RecordingWriter:
    """Stands in for ``TensorBoardWriter`` -- records the stamped dict."""

    def __init__(self) -> None:
        self.captured: dict | None = None

    def hparams(self, hparams: dict) -> None:
        self.captured = hparams


class TestWriteHparams:
    @staticmethod
    def _config(tmp_path, **overrides):
        cfg = _config_dict(tmp_path)
        for key, value in overrides.items():
            cfg[key] = {**cfg.get(key, {}), **value} if isinstance(value, dict) else value
        return TrainingSettings.settings_from_dict(cfg)

    def test_it_stamps_the_ablation_axes_off_a_real_config(self, tmp_path):
        """The regression: every path here must resolve on a real settings object."""
        writer = _RecordingWriter()
        train_mod._write_hparams(writer, self._config(tmp_path))

        assert writer.captured is not None, (
            "nothing was stamped -- a read raised and was swallowed by the "
            "soft-fail guard, which is the `precision.use_amp` bug returning"
        )
        assert set(writer.captured) >= {
            "model_type",
            "training_mode",
            "learning_rate",
            "optimizer",
            "batch_size",
            "amp",
            "grad_accum",
        }
        assert writer.captured["model_type"] == "unet"

    def test_float32_dtype_records_amp_as_off(self, tmp_path):
        """``enabled: true`` + ``dtype: float32`` is AMP OFF -- the third state.

        Reading ``precision.enabled`` alone would record True and mislabel the
        run in the HParams view, which exists to make confounds visible.
        """
        writer = _RecordingWriter()
        cfg = self._config(
            tmp_path,
            optimization={"precision": {"enabled": True, "dtype": "float32"}},
        )
        train_mod._write_hparams(writer, cfg)

        assert writer.captured["amp"] is False

    def test_bfloat16_records_amp_as_on(self, tmp_path):
        writer = _RecordingWriter()
        cfg = self._config(
            tmp_path,
            optimization={"precision": {"enabled": True, "dtype": "bfloat16"}},
        )
        train_mod._write_hparams(writer, cfg)

        assert writer.captured["amp"] is True

    def test_baseline_metadata_reaches_the_dashboard(self, tmp_path):
        """``metadata`` is a dict, so the old ``getattr`` always returned None."""
        writer = _RecordingWriter()
        cfg = self._config(tmp_path, metadata={"baseline": "exp_11_control"})
        train_mod._write_hparams(writer, cfg)

        assert writer.captured["baseline"] == "exp_11_control"

    def test_it_never_raises(self, tmp_path, caplog):
        """A diagnostics stamp must not end a run, nor clobber a real error.

        It is called from ``run_training_pipeline``'s ``except`` handler as
        well as from its success path.
        """
        writer = _RecordingWriter()

        class _Explodes:
            @property
            def optimization(self):
                raise RuntimeError("resolved config went away")

        with caplog.at_level("WARNING"):
            train_mod._write_hparams(writer, _Explodes())

        assert writer.captured is None
        assert "HParams stamp failed" in caplog.text

    def test_both_call_sites_are_guarded(self):
        """Anti-vacuity: the soft-fail only matters if these are the callers."""
        import inspect

        src = inspect.getsource(train_mod.run_training_pipeline)
        assert src.count("_write_hparams(tb_writer, config)") == 2


class TestFatalHealthChecksAbortBeforeBootstrap:
    """A terminal pre-flight failure must not build the training environment.

    ``check_deepspeed_extra_installed`` reported, verbatim, that the run "would
    fail after building the whole training environment" -- and the pipeline then
    built it, because only the literal name ``domain_alignment`` aborted and
    every other error-severity check fell through to a one-line warning. On a
    cluster node that meant data validation, device resolution and environment
    construction all ran before the import error the checker had already found.
    """

    @staticmethod
    def _report(check_name: str, severity: str = "error"):
        from spectramr.infrastructure.validation.config_health_checker import (
            HealthCheckReport,
            HealthCheckResult,
        )

        return HealthCheckReport(
            results=[
                HealthCheckResult(
                    False,
                    check_name,
                    f"{check_name} failed",
                    severity,
                    fix_hint="pip install -e '.[deepspeed]'",
                )
            ]
        )

    def _run(self, monkeypatch, tmp_path, check_name: str, severity: str = "error"):
        """Drive the pipeline to the health gate, recording whether it got past it."""
        reached = []
        monkeypatch.setattr(
            train_mod, "validate_config_health", lambda *a, **k: self._report(check_name, severity)
        )
        monkeypatch.setattr(
            train_mod.bootstrap,
            "build_container",
            lambda *a, **k: reached.append(True) or _stop(),
        )
        cfg = TrainingSettings.settings_from_dict(_config_dict(tmp_path))
        try:
            result = train_mod.run_training_pipeline(cfg, device="cpu")
        except _ReachedBootstrapError:
            result = {"success": None, "error": ""}
        return result, reached

    def test_deepspeed_extra_missing_aborts_before_the_container_is_built(
        self, monkeypatch, tmp_path
    ):
        result, reached = self._run(monkeypatch, tmp_path, "deepspeed_extra_installed")

        assert reached == [], (
            "the pipeline built the training environment after a fatal pre-flight "
            "failure -- the exact waste the check exists to prevent"
        )
        assert result["success"] is False

    def test_the_abort_carries_the_fix_hint(self, monkeypatch, tmp_path):
        """A cluster user reads the returned error, not only the log.

        Without the hint the abort says what is wrong and not what to type,
        which is the difference between a 10-second fix and a support round-trip.
        """
        result, _ = self._run(monkeypatch, tmp_path, "deepspeed_extra_installed")

        assert "pip install -e '.[deepspeed]'" in result["error"]

    def test_domain_alignment_still_aborts(self, monkeypatch, tmp_path):
        """Regression guard: the pre-existing fail-fast must survive the move
        from a hardcoded name to the declared set."""
        result, reached = self._run(monkeypatch, tmp_path, "domain_alignment")

        assert reached == []
        assert result["success"] is False
        assert "domain_alignment" in result["error"]

    def test_a_non_fatal_error_still_warns_and_continues(self, monkeypatch, tmp_path):
        """The other ~150 error checks are NOT promoted.

        This is the whole safety argument for the change: only checks whose
        failure makes the run impossible abort, so nothing that trains today
        stops training.
        """
        _result, reached = self._run(monkeypatch, tmp_path, "deepspeed_topology_coherent")

        assert reached == [True], (
            "a non-fatal error-severity check aborted the run -- the gate has "
            "become a blanket refusal"
        )

    def test_every_declared_fatal_check_is_actually_emitted(self):
        """Anti-vacuity: a name nothing emits is a check that never fires.

        ``train.py`` filters on the **emitted** ``HealthCheckResult.check_name``,
        not on method names. The two are independent values and already diverge
        for 14 checks -- ``check_required_sections`` emits ``required_section``,
        singular -- so the older ``hasattr(checker, f"check_{name}")`` form passed
        for names no runtime result can ever carry (#1355).

        The emitted set comes from the gate that owns this invariant,
        ``scripts/ci/check_health_check_names.py``; re-deriving it here would be a
        second owner of the same rule (CLAUDE.md #17).
        """
        import importlib.util
        import sys
        from pathlib import Path

        from spectramr.infrastructure.validation.config_health_checker import (
            FATAL_HEALTH_CHECKS,
        )

        script = (
            Path(__file__).resolve().parents[3] / "scripts" / "ci" / "check_health_check_names.py"
        )
        spec = importlib.util.spec_from_file_location("_check_health_check_names", script)
        assert spec is not None and spec.loader is not None
        gate = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = gate
        spec.loader.exec_module(gate)

        mapping, unresolved = gate.collect(gate.SOURCE.read_text(encoding="utf-8"))
        assert unresolved == [], (
            f"the collector could not statically read {len(unresolved)} emission "
            f"site(s), so the set below is a floor, not the set: {unresolved[:3]}"
        )
        emitted = {name for names in mapping.values() for name in names}

        # Anti-vacuity for the collector itself. An over-broad collector -- one
        # that fell back to method suffixes -- would satisfy the assertion below
        # exactly as the old hasattr form did. ``required_sections`` is the
        # sharpest known divergence, so pinning its absence witnesses that
        # ``emitted`` really is the emitted set. This breaks deliberately on the
        # day someone fixes that plural/singular slip.
        assert "check_required_sections" in mapping
        assert "required_sections" not in emitted, (
            "check_required_sections emits 'required_section' (singular); seeing "
            "the plural means the collector fell back to method names and every "
            "assertion in this test is vacuous"
        )

        missing = sorted(name for name in FATAL_HEALTH_CHECKS if name not in emitted)
        assert not missing, (
            f"FATAL_HEALTH_CHECKS names {missing}, which no check emits -- a method "
            "called check_<name> is not enough, train.py matches the emitted string"
        )


class _ReachedBootstrapError(Exception):
    """Sentinel: the pipeline reached bootstrap, which these tests treat as the
    observable 'did not fail fast'. Raised so the test never builds a container."""


def _stop():
    raise _ReachedBootstrapError


# ---------------------------------------------------------------------------
# Issue #1124: the two seeding calls, and why their ORDER is the whole fix.
#
# Stage 0 seeds every rank identically so weight init agrees (FSDP shards from
# rank 0's parameters; strategy-owned aux modules built at stage 7 are never
# broadcast). The rank-offset re-seed must therefore land after construction —
# but before stage 9, because checkpoint resume restores saved per-rank RNG
# state and a later re-seed would discard the resumed stream. The window is
# closed on both sides, so a move in either direction is a real regression and
# neither end can be exercised without a live process group.
# ---------------------------------------------------------------------------


class TestSeedRankOffsetOrdering:
    @staticmethod
    def _source() -> str:
        import inspect

        from spectramr.pipelines import train

        return inspect.getsource(train)

    def test_the_step_zero_seeding_passes_no_rank(self) -> None:
        """Rank-offsetting before model construction diverges initial weights
        across ranks — the failure this ordering exists to prevent."""
        import ast
        import inspect

        from spectramr.pipelines import train

        tree = ast.parse(inspect.getsource(train))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "set_global_seed"
        ]
        assert len(calls) == 2, (
            f"expected exactly 2 set_global_seed calls, found {len(calls)} at "
            f"lines {[c.lineno for c in calls]} — the shared-weights/per-rank-data "
            "split depends on there being exactly one of each."
        )

        first, second = sorted(calls, key=lambda c: c.lineno)
        assert not any(k.arg == "rank" for k in first.keywords), (
            f"the stage-0 seeding at line {first.lineno} passes `rank`; that runs "
            "before the model is built and would diverge weight init across ranks."
        )
        assert any(k.arg == "rank" for k in second.keywords), (
            f"the post-construction seeding at line {second.lineno} must pass "
            "`rank` — without it every rank draws identical augmentations."
        )

    def test_the_rank_offset_seeding_sits_between_construction_and_resume(
        self,
    ) -> None:
        src = self._source()
        offset_call = src.index("_data_rank = resolve_data_rank(config)")
        strategy_setup = src.index("# 7. Strategy setup")
        resume = src.index("# 9. Resume from Checkpoint")

        assert strategy_setup < offset_call, (
            "the rank-offset re-seed moved BEFORE stage 7 — strategy-owned "
            "modules built there are not broadcast, so their weights would "
            "diverge across ranks."
        )
        assert offset_call < resume, (
            "the rank-offset re-seed moved AFTER stage 9 — resume restores "
            "the saved RNG state (core.rng_state, applied by "
            "CheckpointDirector.load_from) and a "
            "later re-seed throws the resumed stream away."
        )

    def test_both_seeding_calls_use_the_same_determinism_resolver(self) -> None:
        """Two different determinism values would let the cuDNN flags flap
        mid-build."""
        src = self._source()
        assert src.count("_determinism_from_config(config)") == 2


# ---------------------------------------------------------------------------
# W6: the runtime parallel record must be MERGED, the per-rank collective must
# stay outside every rank-divergent guard, and the write must be rank-gated.
#
# These are source-level because `run_training_pipeline` builds a whole training
# environment; but they are not mirrors of the implementation -- the merge test
# EXECUTES the real merge expression, and the placement test walks the real AST
# rather than matching a comment.
# ---------------------------------------------------------------------------


def _train_ast():
    """Parse `run_training_pipeline`'s source."""
    import ast
    import inspect
    import textwrap

    from spectramr.pipelines import train

    return ast.parse(textwrap.dedent(inspect.getsource(train.run_training_pipeline)))


def _ancestors(tree, target):
    """Every AST node on the path from `tree` to `target`, root-first."""
    import ast

    path = []

    def walk(node, trail):
        if node is target:
            path.extend(trail)
            return True
        return any(walk(child, [*trail, node]) for child in ast.iter_child_nodes(node))

    walk(tree, [])
    return path


class TestTheRuntimeParallelRecordIsMergedNotReplaced:
    """`train.py` overwrote `provenance["parallel"]` with the plugin's thin
    record, discarding the `rank`, `local_rank`, `launcher`, `initialized`,
    `backend` and declared device/node counts that `parallel_provenance` had
    already resolved -- i.e. exactly the fields that answer "why does this say
    1 GPU when I asked for 4".
    """

    @staticmethod
    def _merge_expression():
        """The real merge, lifted from source by AST."""
        import ast

        tree = _train_ast()
        hits = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            # `ast.unparse` normalises string literals to single quotes, so
            # the source's `provenance["parallel"]` round-trips as this.
            and ast.unparse(node.targets[0]) == "provenance['parallel']"
        ]
        assert len(hits) == 1, (
            f"expected one assignment to provenance['parallel'], found {len(hits)}"
        )
        return ast.unparse(hits[0].value)

    def test_the_base_record_survives_the_merge(self):
        """Evaluate the shipped expression: base-only keys must still be there."""
        expr = self._merge_expression()
        namespace = {
            "provenance": {
                "parallel": {
                    "rank": 3,
                    "local_rank": 1,
                    "launcher": "torchrun",
                    "initialized": True,
                    "backend": "nccl",
                    "declared_num_devices": 4,
                    "strategy": "ddp",
                }
            },
            "_plugin_record": {"strategy": "deepspeed", "zero_stage": 2},
        }
        merged = eval(expr, namespace)
        for key in (
            "rank",
            "local_rank",
            "launcher",
            "initialized",
            "backend",
            "declared_num_devices",
        ):
            assert key in merged, f"{key} was discarded by the merge"

    def test_the_plugin_wins_a_collision(self):
        """Where both speak, the RUNTIME is the authority: the plugin observed
        what was actually built, the base record only what the env implied."""
        expr = self._merge_expression()
        namespace = {
            "provenance": {"parallel": {"strategy": "ddp", "backend": "nccl"}},
            "_plugin_record": {"strategy": "deepspeed"},
        }
        merged = eval(expr, namespace)
        assert merged["strategy"] == "deepspeed"

    def test_it_is_not_vacuously_a_passthrough(self):
        """Anti-vacuity: a merge that ignored the plugin would satisfy the first
        test trivially."""
        expr = self._merge_expression()
        namespace = {
            "provenance": {"parallel": {"strategy": "ddp"}},
            "_plugin_record": {"zero_stage": 3},
        }
        merged = eval(expr, namespace)
        assert merged.get("zero_stage") == 3


class TestThePerRankGatherCannotDeadlock:
    """`all_gather_object` is a collective: every rank must reach it or the job
    hangs forever. That makes its PLACEMENT load-bearing, and placement is
    exactly what a future edit can silently break -- so it is pinned here.
    """

    @staticmethod
    def _gather_call():
        import ast

        tree = _train_ast()
        hits = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "rank_device_inventory"
        ]
        assert len(hits) == 1, f"expected one gather call, found {len(hits)}"
        return tree, hits[0]

    def test_it_is_not_nested_in_a_rank_conditional(self):
        """Gating a collective on rank 0 is a guaranteed hang: ranks 1..N-1 never
        arrive and rank 0 waits for them."""
        import ast

        tree, call = self._gather_call()
        for node in _ancestors(tree, call):
            if isinstance(node, ast.If):
                test_src = ast.unparse(node.test)
                assert "rank" not in test_src.lower(), (
                    "the per-rank device gather is nested inside a rank-dependent "
                    f"conditional (`if {test_src}`). It is a COLLECTIVE -- every "
                    "rank must reach it or the job deadlocks."
                )

    def test_it_is_not_nested_in_a_provenance_truthiness_check(self):
        """`collect_run_provenance` fail-opens per rank, so `provenance` can be
        `{}` on one rank and populated on another. Guarding the collective on it
        makes reachability rank-dependent by accident."""
        import ast

        tree, call = self._gather_call()
        for node in _ancestors(tree, call):
            if isinstance(node, ast.If):
                assert ast.unparse(node.test).strip() != "provenance", (
                    "the gather is guarded on `if provenance:`, which is not "
                    "uniform across ranks (provenance capture fail-opens)"
                )

    def test_it_runs_before_the_pipeline_build(self):
        """The build's `except` returns early on a failing rank, so anything
        after it is unreachable on that rank while the others block."""
        import inspect

        from spectramr.pipelines import train

        source = inspect.getsource(train.run_training_pipeline)
        assert source.index("rank_device_inventory") < source.index(
            "Training pipeline build failed"
        ), (
            "the gather moved past the build's failure path; a build error on "
            "one rank would deadlock the rest"
        )


class TestOnlyRankZeroWritesTheCanonicalArtifacts:
    """`run_dir` is rank-INVARIANT -- it comes from `config.training.output_dir`
    on a frozen config, not a per-rank timestamp. So under DDP every rank wrote
    the same `provenance.json` and `resolved_config.json`: last-writer-wins, and
    the surviving record is an arbitrary rank's view of the run.
    """

    @staticmethod
    def _source():
        import inspect

        from spectramr.pipelines import train

        return inspect.getsource(train.run_training_pipeline)

    def test_the_resolved_config_write_is_rank_gated(self):
        source = self._source()
        assert 'if hasattr(config, "model_dump") and _is_rank_zero:' in source

    def test_non_zero_ranks_write_their_own_provenance_file(self):
        """Not silently dropped: the per-rank `local_rank`/`device` facts exist
        nowhere else, and discarding them would make the new inventory
        unverifiable."""
        source = self._source()
        assert 'f"provenance_rank{_resolved_rank}.json"' in source

    def test_the_naming_rank_comes_from_the_shared_resolver(self):
        """A provenance file named for a different rank than the one that sharded
        the data would be worse than no per-rank file at all."""
        source = self._source()
        assert "_resolved_rank = resolve_data_rank(config)" in source

    def test_the_id_named_copy_is_rank_gated_too(self):
        """#1299's ``provenance_run_<run_id>.json`` needs the same gate.

        Ungated it has two failure modes and no good one. Ranks that AGREE on
        ``run_id`` all write one path -- last-writer-wins, under a name that
        reads as the run's own record, which is the very thing this class
        exists to prevent. Ranks that DISAGREE are worse: ``run_id`` is
        ``<slug>-<YYYYmmdd_HHMMSS>-<sha>`` built from each rank's own
        ``datetime.now()``, and ``collect_run_provenance`` fail-opens per rank,
        so two ranks straddling a second boundary mint different ids and one
        run scatters N files each claiming to be its record.

        Asserted on the AST, not the source text: the check is about which
        NAME the write uses being chosen by rank, and that must survive a
        reformat of the conditional.
        """
        import ast

        tree = _train_ast()
        gated = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "_id_name" for t in node.targets):
                continue
            # The value must CHOOSE by rank, not be a single f-string.
            assert isinstance(node.value, ast.IfExp), (
                "_id_name is not a conditional -- the id-named provenance copy "
                "writes one path on every rank"
            )
            names = {n.id for n in ast.walk(node.value.test) if isinstance(n, ast.Name)}
            gated.append(names)

        assert gated, "no _id_name assignment found -- the id-named copy moved or vanished"
        assert any("_is_rank_zero" in names for names in gated), gated

    def test_the_two_provenance_names_cannot_collide_across_ranks(self):
        """Both writes must differentiate by rank, or one overwrites the other.

        Pins the pair rather than each alone: the canonical name was already
        gated when the id-named one was added ungated beside it, so a test that
        only ever checked one of them would have stayed green through exactly
        that regression.
        """
        source = self._source()
        assert 'f"provenance_rank{_resolved_rank}.json"' in source

        _head, sep, tail = source.partition("_id_name")
        assert sep, "no _id_name in the pipeline -- the id-named copy moved or vanished"
        assert "_resolved_rank" in tail[:400], (
            "the id-named copy does not mention _resolved_rank anywhere near "
            "its construction -- non-zero ranks cannot be writing distinct paths"
        )


class TestTheLogDestinationIsStamped:
    """W10. `logging.sinks.dir` wins over the run dir (correct per 3b), so the
    log routinely sits outside the run it describes. provenance.json is the one
    artifact that can bridge the two, and it recorded nothing.

    Asserted on the AST rather than by running the pipeline: the stamp sits in
    `run_training_pipeline`, which needs a built container, a dataset and a
    strategy before it reaches this line.
    """

    def _stamp(self):
        import ast

        tree = _train_ast()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "provenance"
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == "logging"
            ):
                return tree, node
        raise AssertionError("train.py never assigns provenance['logging']")

    def _record_keys(self, tree) -> set[str]:
        """Constant keys reaching the log record, from the dict literal and from
        any later `record[...] = ...` assignment."""
        import ast

        keys: set[str] = set()
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign | ast.Assign) and isinstance(
                getattr(node, "value", None), ast.Dict
            ):
                targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
                for tgt in targets:
                    if isinstance(tgt, ast.Name) and "log" in tgt.id.lower():
                        names.add(tgt.id)
                        keys |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if (
                        isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id in names
                        and isinstance(tgt.slice, ast.Constant)
                    ):
                        keys.add(tgt.slice.value)
        return keys

    def test_provenance_carries_the_log_record(self):
        _, stamp = self._stamp()
        assert stamp is not None

    def test_it_records_the_resolved_path_beside_the_declared_dir(self):
        """Declared beside applied, per non-negotiable 14: the divergence between
        `sinks.dir` and where the file actually opened IS the finding, and a
        record holding only one half cannot express it."""
        tree, _ = self._stamp()
        keys = self._record_keys(tree)
        assert "resolved_path" in keys, f"no resolved path in the record: {keys}"
        assert "declared_sinks_dir" in keys, (
            f"the declared logging.sinks.dir is not recorded beside it: {keys}"
        )

    def test_a_relocation_is_recorded_as_incomplete_not_omitted(self):
        """Pitfall #16: what cannot be delivered is declared, never dropped. A
        log moved to a temp dir is gone after teardown, so a bare path would read
        as a log that exists."""
        tree, _ = self._stamp()
        keys = self._record_keys(tree)
        assert "relocated_from" in keys, f"a relocation is not recorded: {keys}"
        assert "incomplete" in keys, f"a relocated log is not flagged incomplete: {keys}"

    def test_it_reads_the_service_attributes_rather_than_guessing_a_path(self):
        """Reconstructing `sinks.dir + name` would report the path the service
        INTENDED, which is exactly the value that survives relocation unchanged
        -- the failure this stamp exists to make visible."""
        import ast

        tree, _ = self._stamp()
        attrs = {
            node.args[1].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
        }
        assert "resolved_log_path" in attrs
        assert "log_dir_relocated_from" in attrs

    def test_the_stamp_is_not_rank_gated(self):
        """Every rank's log matters when only rank 0's is named, so the stamp
        sits outside the `_is_rank_zero` write gate -- a per-rank provenance file
        then names that rank's own log."""
        import ast

        tree, stamp = self._stamp()
        for ancestor in _ancestors(tree, stamp):
            if isinstance(ancestor, ast.If):
                test_src = ast.unparse(ancestor.test)
                assert "rank" not in test_src.lower(), (
                    "the log stamp is inside a rank-gated branch "
                    f"(`if {test_src}`), so non-zero ranks record no log path"
                )


# ---------------------------------------------------------------------------
# validation-interval reachability (`validation_can_fire` / `epoch_validation_can_fire`)
# ---------------------------------------------------------------------------
#
# `validation.schedule.interval_steps` was the only interval knob in the loop
# with NEITHER a budget guard NOR an unconditional first/last-iteration force:
# `logging.intervals.log` and `metrics.train_metric_interval` have both. So an
# interval above the run's budget produced ZERO validation events in silence --
# no early stopping, no `checkpoint_best.pt`, exit 0 reporting success.
#
# The oracle below is deliberately a re-implementation of the loop's own two
# gate expressions rather than a table of expected answers: what has to hold is
# that the predicate agrees with the gate it claims to model, on every shape.


def _simulate_validation_events(
    *,
    max_iterations: int,
    eval_interval: int,
    eval_on_epoch: bool = False,
    eval_interval_epochs: int = 1,
    train_loader_len: int = 1,
    has_train_loader: bool = False,
) -> list[int]:
    """Replay the loop's real validation gates and return the firing iterations.

    Transcribed from ``_execute_training_loop``: the loop runs
    ``range(first_iteration, max_iterations + 1)``, sets
    ``epoch = iteration // train_loader_len``, fires on
    ``iteration % eval_interval == 0``, and additively on
    ``_is_epoch_boundary(...) and epoch % eval_interval_epochs == 0``.
    """
    events = []
    for iteration in range(1, max_iterations + 1):
        epoch = iteration // train_loader_len
        fires = iteration % eval_interval == 0
        is_epoch_end = has_train_loader and iteration % train_loader_len == 0
        if eval_on_epoch and is_epoch_end and epoch % eval_interval_epochs == 0:
            fires = True
        if fires:
            events.append(iteration)
    return events


class TestValidationCanFire:
    """The predicate must agree with the loop's own gates, not with a table."""

    @pytest.mark.parametrize(
        ("max_iterations", "eval_interval"),
        [
            (30000, 5000),  # the arm as declared -- 6 events
            (5000, 5000),  # the override that was actually run -- 1 event, last iteration
            (5000, 1),
            (10, 10),
            (10, 3),
        ],
    )
    def test_reachable_intervals_are_reported_reachable(self, max_iterations, eval_interval):
        assert _simulate_validation_events(
            max_iterations=max_iterations, eval_interval=eval_interval
        )
        assert train_mod.validation_can_fire(
            eval_interval=eval_interval, max_iterations=max_iterations
        )

    @pytest.mark.parametrize(
        ("max_iterations", "eval_interval"),
        [
            (4000, 5000),  # the silent regime: one notch below the observed override
            (40, 5000),  # the budget the sibling guards' comments were measured at
            (1, 2),
            (999, 1000),
        ],
    )
    def test_unreachable_intervals_are_reported_unreachable(self, max_iterations, eval_interval):
        assert (
            _simulate_validation_events(max_iterations=max_iterations, eval_interval=eval_interval)
            == []
        )
        assert not train_mod.validation_can_fire(
            eval_interval=eval_interval, max_iterations=max_iterations
        )

    def test_the_boundary_is_interval_equals_budget_not_interval_exceeds_it(self):
        """A `<` predicate would have missed the run this was found on.

        The two siblings test `max_iterations < interval`. For validation that
        is off by one case: `interval == max_iterations` still fires (once, on
        the final iteration), and `interval == max_iterations + 1` is the first
        value that never fires. Pinning both sides stops a future edit from
        sliding the comparison.
        """
        assert train_mod.validation_can_fire(eval_interval=5000, max_iterations=5000)
        assert not train_mod.validation_can_fire(eval_interval=5001, max_iterations=5000)

    def test_epoch_mode_rescues_an_unreachable_step_interval(self):
        """`on_epoch` adds events, so it can make an over-budget interval moot."""
        shape = {
            "max_iterations": 100,
            "eval_interval": 5000,
            "eval_on_epoch": True,
            "eval_interval_epochs": 1,
            "train_loader_len": 10,
            "has_train_loader": True,
        }
        assert _simulate_validation_events(**shape) == [10 * k for k in range(1, 11)]
        assert train_mod.validation_can_fire(**shape)

    def test_epoch_mode_does_not_rescue_when_one_epoch_exceeds_the_budget(self):
        """A train loader longer than the whole budget has no boundary in it."""
        shape = {
            "max_iterations": 100,
            "eval_interval": 5000,
            "eval_on_epoch": True,
            "eval_interval_epochs": 1,
            "train_loader_len": 512,
            "has_train_loader": True,
        }
        assert _simulate_validation_events(**shape) == []
        assert not train_mod.validation_can_fire(**shape)

    def test_interval_epochs_pushes_the_first_epoch_event_out(self):
        """`interval_epochs=N` means the first epoch-gated event is at N*len.

        `epoch % interval_epochs == 0` is satisfied first at epoch N (epoch 0
        is iteration 0, which the loop never visits), so a budget under
        `N * train_loader_len` sees nothing.
        """
        shape = {
            "max_iterations": 25,
            "eval_interval": 5000,
            "eval_on_epoch": True,
            "eval_interval_epochs": 3,
            "train_loader_len": 10,
            "has_train_loader": True,
        }
        assert _simulate_validation_events(**shape) == []
        assert not train_mod.validation_can_fire(**shape)

        wider = {**shape, "max_iterations": 30}
        assert _simulate_validation_events(**wider) == [30]
        assert train_mod.validation_can_fire(**wider)

    def test_a_missing_train_loader_is_not_a_perpetual_epoch_boundary(self):
        """Same trap `_is_epoch_boundary` documents: `train_loader_len` falls
        back to 1 when the loader is absent, and `iteration % 1 == 0` is always
        True -- so without the `has_train_loader` guard every step would read as
        an epoch boundary and the guard would never fire."""
        assert not train_mod.validation_can_fire(
            eval_interval=5000,
            max_iterations=100,
            eval_on_epoch=True,
            train_loader_len=1,
            has_train_loader=False,
        )

    def test_agreement_with_the_simulated_gate_across_a_grid(self):
        """Exhaustive agreement over small shapes -- the real anti-regression net."""
        for max_iterations in range(1, 18):
            for eval_interval in range(1, 20):
                for train_loader_len in (1, 3, 5, 16):
                    for eval_interval_epochs in (1, 2, 3):
                        for eval_on_epoch in (False, True):
                            for has_train_loader in (False, True):
                                shape = {
                                    "max_iterations": max_iterations,
                                    "eval_interval": eval_interval,
                                    "eval_on_epoch": eval_on_epoch,
                                    "eval_interval_epochs": eval_interval_epochs,
                                    "train_loader_len": train_loader_len,
                                    "has_train_loader": has_train_loader,
                                }
                                assert train_mod.validation_can_fire(**shape) == bool(
                                    _simulate_validation_events(**shape)
                                ), shape


class TestEpochValidationCanFire:
    """The epoch half, split out because the caller needs it on its own."""

    def test_it_is_false_whenever_on_epoch_is_off(self):
        assert not train_mod.epoch_validation_can_fire(
            max_iterations=10_000,
            eval_on_epoch=False,
            train_loader_len=1,
            has_train_loader=True,
        )

    def test_it_matches_the_epoch_half_of_the_simulated_gate(self):
        for max_iterations in (1, 7, 20, 33):
            for train_loader_len in (1, 4, 10, 64):
                for eval_interval_epochs in (1, 2, 5):
                    for has_train_loader in (False, True):
                        shape = {
                            "max_iterations": max_iterations,
                            "eval_on_epoch": True,
                            "eval_interval_epochs": eval_interval_epochs,
                            "train_loader_len": train_loader_len,
                            "has_train_loader": has_train_loader,
                        }
                        # `eval_interval` above the budget isolates the epoch gate.
                        simulated = _simulate_validation_events(
                            eval_interval=max_iterations + 1, **shape
                        )
                        assert train_mod.epoch_validation_can_fire(**shape) == bool(simulated), (
                            shape
                        )


class TestTheResumeDirectorCarriesTheParallelRuntime:
    """The resume site built its `CheckpointDirector` without
    `with_parallel_runtime`, so it resolved `DefaultCheckpointAdapter` no matter
    what strategy actually wrote the files.

    That has two failure modes, and the quiet one is the worse one. A sharded
    best checkpoint carries no generic payload at all, so resuming from one
    raised. A *periodic* checkpoint does carry the generic envelope, so the
    weights restored and the resume reported success -- while the ZeRO optimizer
    partitions in the tag directory beside it went unread, silently continuing a
    resumed run on a zeroed optimizer.

    Walks the real AST rather than matching a comment, and rejects a literal
    `None` argument: `with_parallel_runtime(None)` would satisfy a name-only
    check while resolving the very adapter this exists to avoid.
    """

    @staticmethod
    def _calls_on(tree, receiver):
        """Every `receiver.<m>(...)` Call node in `tree`, by method name."""
        import ast

        found = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            # Walk to the base of the chain: `d.a().b().c()` bottoms out at
            # `d`. Call and Attribute ALTERNATE, so this has to loop until
            # neither applies -- two sequential `while`s stop one link short
            # and silently miss every method past the second.
            base = func.value
            while isinstance(base, ast.Call | ast.Attribute):
                base = base.func if isinstance(base, ast.Call) else base.value
            if isinstance(base, ast.Name) and base.id == receiver:
                found.setdefault(func.attr, []).append(node)
        return found

    def test_the_resume_director_is_given_the_parallel_runtime(self):
        calls = self._calls_on(_train_ast(), "resume_director")

        # Non-vacuity: if the resume block is ever renamed or restructured this
        # test must fail loudly rather than pass by finding nothing.
        assert "load_from" in calls, (
            "no `resume_director.load_from(...)` found in run_training_pipeline "
            f"-- the resume block moved; found methods: {sorted(calls)}"
        )
        assert "with_parallel_runtime" in calls, (
            "the resume director is built without with_parallel_runtime, so it "
            "resolves DefaultCheckpointAdapter and cannot read a sharded "
            f"strategy's checkpoint; found methods: {sorted(calls)}"
        )

    def test_the_runtime_argument_is_resolved_not_hardcoded_none(self):
        import ast

        calls = self._calls_on(_train_ast(), "resume_director")
        node = calls["with_parallel_runtime"][0]

        assert len(node.args) == 1, (
            "with_parallel_runtime takes the runtime positionally; "
            f"got {len(node.args)} positional args"
        )
        arg = node.args[0]
        assert not (isinstance(arg, ast.Constant) and arg.value is None), (
            "with_parallel_runtime(None) resolves DefaultCheckpointAdapter -- "
            "the exact adapter this wiring exists to avoid"
        )
        src = ast.unparse(arg)
        assert "parallel" in src, (
            "the argument should read the runtime off the built environment "
            f"(the way training_loop.py does); got `{src}`"
        )


# ---------------------------------------------------------------------------
# Issue #1347 — the epoch metric is a SAMPLE-weighted mean.
#
# ``val_accum`` was divided by ``val_count``, a count of BATCHES. With
# ``drop_last=False`` a short final batch then weighed exactly as much as a full
# one, and every metric that is a per-sample mean came out at the wrong value
# whenever the val set did not divide evenly by the batch size.
#
# These drive the real ``_run_validation``; the strategy returns a metric that
# differs per batch so an unweighted mean is distinguishable from a weighted one.
# ---------------------------------------------------------------------------


def _val_pipeline(batches):
    from types import SimpleNamespace as NS

    from tests.utils.config_block_stub import block_stub
    from tests.utils.data_config_stub import DataConfigStub

    logging_cfg = block_stub(
        "logging",
        log_validation_images=False,
        save_validation_images=False,
        validation_image_interval=1,
        max_images_per_batch=4,
        log_input_images=False,
        log_difference_images=True,
    )
    cfg = NS(
        validation=block_stub(
            "validation",
            num_validation_batches=len(batches),
            num_samples=None,
            effective_val_batch_size=1,
        ),
        logging=logging_cfg,
        data=DataConfigStub(patch_size=(8, 8)),
        model=NS(spatial_dims=2),
    )
    return NS(
        ema=None,
        device=torch.device("cpu"),
        config=cfg,
        data_loaders={"val": batches},
        generator=nn.Identity(),
        models={"generator": nn.Identity()},
    ), logging_cfg


class _PerBatchValueStrategy:
    """Returns ``score`` = the batch's own sample count, so the epoch mean is
    readable: a batch-weighted mean of [5, 5, 5, 5, 4] is 4.8, a sample-weighted
    one is (4*25 + 16)/24 = 4.833..."""

    def __init__(self, logging_cfg):
        from types import SimpleNamespace as NS

        self.config = NS(logging=logging_cfg)

    def validation_step(self, input_batch, target_batch, batch_idx=0):
        return {"score": float(input_batch.shape[0])}

    def finalize_validation(self) -> dict:
        return {}


def test_validation_epoch_mean_weights_a_short_final_batch_by_its_size():
    """24 samples as 4x5 + 1x4. Batch-weighted gives 4.8; sample-weighted 4.8333."""
    sizes = [5, 5, 5, 5, 4]
    batches = [{"input": torch.randn(n, 1, 8, 8), "target": torch.randn(n, 1, 8, 8)} for n in sizes]
    pipeline, logging_cfg = _val_pipeline(batches)
    metrics = train_mod._run_validation(
        pipeline,
        _PerBatchValueStrategy(logging_cfg),
        iteration=1,
        epoch=0,
        logging_service=None,
    )
    expected = sum(n * n for n in sizes) / sum(sizes)
    assert metrics["score"] == pytest.approx(expected, abs=1e-6)
    assert metrics["score"] != pytest.approx(sum(sizes) / len(sizes), abs=1e-6)


def test_validation_epoch_mean_is_unchanged_when_every_batch_is_full():
    """The weighting must be a no-op on an evenly-divided val set -- otherwise
    the fix would restate numbers it has no business restating."""
    batches = [
        {"input": torch.randn(4, 1, 8, 8), "target": torch.randn(4, 1, 8, 8)} for _ in range(3)
    ]
    pipeline, logging_cfg = _val_pipeline(batches)
    metrics = train_mod._run_validation(
        pipeline,
        _PerBatchValueStrategy(logging_cfg),
        iteration=1,
        epoch=0,
        logging_service=None,
    )
    assert metrics["score"] == pytest.approx(4.0, abs=1e-6)


def test_validation_batch_sample_count_reads_the_target_first():
    """The target is the tensor every full-reference metric graded against."""
    from spectramr.data.batch_types import TrainingBatch

    assert (
        train_mod._validation_batch_sample_count(
            None, torch.randn(2, 1, 4, 4), torch.randn(7, 1, 4, 4)
        )
        == 7
    )
    # Falls back to the input when there is no target...
    assert train_mod._validation_batch_sample_count(None, torch.randn(3, 1, 4, 4), None) == 3
    # ...then to the container.
    batch = TrainingBatch(input=torch.randn(5, 1, 4, 4), target=torch.randn(5, 1, 4, 4))
    assert train_mod._validation_batch_sample_count(batch, None, None) == 5
    six = {"input": torch.randn(6, 1, 4, 4)}
    assert train_mod._validation_batch_sample_count(six, None, None) == 6


def test_validation_batch_sample_count_reports_rather_than_guesses():
    """An unsizeable batch returns None. The caller weights it as one sample and
    warns once -- a silent mix of weight-1 and weight-N batches would re-create
    the defect the weighting exists to remove."""
    assert train_mod._validation_batch_sample_count(object(), None, None) is None
    assert train_mod._validation_batch_sample_count({}, None, None) is None


def test_an_unsizeable_batch_is_reported_not_silently_weighted(caplog):
    """Drive the real loop with a batch carrying no tensor and read the warning."""
    import logging as _logging

    class _NoTensorStrategy(_PerBatchValueStrategy):
        def validation_step(self, batch, batch_idx=0):
            return {"score": 1.0}

    # NOT a dict and not a TrainingBatch, so ``BatchAdapter`` never sees it and
    # neither ``_val_input`` nor ``_val_target`` is ever bound -- the only shape
    # in which a batch reaches the accumulator with nothing to size it by.
    from types import SimpleNamespace as NS

    batches = [NS(payload="no tensors here")]
    pipeline, logging_cfg = _val_pipeline(batches)
    with caplog.at_level(_logging.WARNING, logger="spectramr.pipelines.train"):
        train_mod._run_validation(
            pipeline,
            _NoTensorStrategy(logging_cfg),
            iteration=1,
            epoch=0,
            logging_service=None,
        )
    messages = [r.getMessage() for r in caplog.records]
    assert any("weighted as a single sample" in m for m in messages), messages


class TestReportArtifactWritesAreRankGuardedAndOrdered:
    """#1685 -- four ranks raced on ``report_cases/``; three lost.

    ``ReportCaseRecorder.write`` and the per-case sink ran on EVERY rank
    against one shared run dir, and the reporting hook that reads them ran
    unsynchronised. Both are ordering facts about a ~700-line function, not
    values it returns, so they are pinned structurally: the AST is the only
    thing a comment cannot satisfy, and ``inspect.getsource`` substring checks
    on this file would score green on the prose that documents the fix.
    """

    @staticmethod
    def _pipeline_body():
        import ast
        import inspect

        from spectramr.pipelines import train as train_mod

        with open(inspect.getfile(train_mod), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        fns = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "run_training_pipeline"
        ]
        assert len(fns) == 1
        return fns[0]

    @staticmethod
    def _calls(node):
        import ast

        out = []
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Attribute):
                    out.append((n, f.attr))
                elif isinstance(f, ast.Name):
                    out.append((n, f.id))
        return out

    def test_both_artifact_writers_sit_under_the_rank_zero_guard(self):
        import ast

        fn = self._pipeline_body()
        guarded = set()
        for node in ast.walk(fn):
            if not (isinstance(node, ast.If) and isinstance(node.test, ast.Name)):
                continue
            if node.test.id != "_is_rank_zero":
                continue
            guarded.update(attr for _, attr in self._calls(node) if attr == "write")
        assert guarded == {"write"}, (
            "``_report_recorder.write`` / ``_per_case_sink.write`` must run under "
            "``if _is_rank_zero:`` -- unguarded, every rank writes the same files"
        )

    def test_no_write_call_escapes_the_guard(self):
        """The set-difference form: a THIRD writer added later must be caught.

        A test that only proves the two known calls are inside the guard stays
        green when a new sink is appended outside it.
        """
        import ast

        fn = self._pipeline_body()
        writers = {
            id(n)
            for n, attr in self._calls(fn)
            if attr == "write"
            and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id in {"_report_recorder", "_per_case_sink"}
        }
        assert len(writers) == 2, f"expected two artifact writers, found {len(writers)}"
        inside = set()
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "_is_rank_zero"
            ):
                inside.update(id(n) for n, _ in self._calls(node))
        assert writers <= inside, "an artifact writer sits outside the rank-zero guard"

    def test_a_barrier_separates_the_writes_from_the_reporting_hook(self):
        """The hook READS what the guarded block WRITES.

        Without the barrier a straggler enters reporting against a run dir only
        rank 0 has finished populating -- the same ``Bad CRC-32`` symptom from
        the other direction, and one that a rank guard alone cannot fix.
        """
        fn = self._pipeline_body()
        barriers = [n.lineno for n, attr in self._calls(fn) if attr == "barrier"]
        hooks = [n.lineno for n, attr in self._calls(fn) if attr == "_maybe_run_reporting"]
        assert barriers, "no ``RankUtility.barrier()`` in the training pipeline"
        assert hooks, "the reporting hook call site moved or was renamed"
        assert min(barriers) < min(hooks), (
            f"barrier at line {min(barriers)} must precede the reporting hook at line {min(hooks)}"
        )

    def test_the_barrier_is_the_existing_owner_not_a_second_one(self):
        """Non-negotiable 17. ``RankUtility.barrier`` already owns this.

        A fresh ``dist.barrier()`` here would need its own is-initialised guard
        and would be a second, differently-conditioned owner of one primitive.
        """
        import ast

        fn = self._pipeline_body()
        for node, attr in self._calls(fn):
            if attr != "barrier":
                continue
            assert isinstance(node.func, ast.Attribute)
            assert isinstance(node.func.value, ast.Name)
            assert node.func.value.id == "RankUtility", ast.dump(node.func)

    def test_rank_utility_barrier_is_a_noop_without_a_process_group(self):
        """Behavioural half: the guard must not break single-process training."""
        from spectramr.infrastructure.distributed.distributed_training import RankUtility

        RankUtility.barrier()  # must not raise


# ---------------------------------------------------------------------------
# `metadata.author` reaches the run record (2026-09-03).
#
# The field was declared on ExperimentMetadataSchema and read nowhere, which is
# what `TestReachabilityAwareConsumption::test_no_new_unreachable_reads` reports
# once the typed metadata mount makes the key visible: a key the YAMLs set whose
# only reads cannot execute. A run record that names the run but not who declared
# it answers half of the same provenance question, so the summary stamps it.
# ---------------------------------------------------------------------------


def test_the_run_summary_carries_the_declared_author(tmp_path) -> None:
    """The fires-test: an arm's `metadata.author` reaches `run_summary.json`."""
    import json
    import logging
    from types import SimpleNamespace

    from spectramr.pipelines import train as train_mod

    config = SimpleNamespace(
        metadata=SimpleNamespace(name="an_arm", author="Adnane Gdihi"),
        logging=SimpleNamespace(identity=SimpleNamespace(run=None)),
        training=SimpleNamespace(strategy_class="S", training_mode="reconstruction"),
        model=SimpleNamespace(model_type="unet"),
    )
    train_mod._emit_run_summary(
        config,
        run_dir=tmp_path,
        result={"success": True},
        seed=42,
        logger_=logging.getLogger(__name__),
        provenance={"run_id": "r1"},
    )
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["author"] == "Adnane Gdihi"
    assert summary["run_name"] == "an_arm"


def test_the_run_summary_author_is_none_when_the_arm_declares_none(tmp_path) -> None:
    """Absent is a state to report, not one to invent."""
    import json
    import logging
    from types import SimpleNamespace

    from spectramr.pipelines import train as train_mod

    config = SimpleNamespace(
        metadata=SimpleNamespace(name="an_arm", author=None),
        logging=SimpleNamespace(identity=SimpleNamespace(run=None)),
        training=SimpleNamespace(strategy_class="S", training_mode="reconstruction"),
        model=SimpleNamespace(model_type="unet"),
    )
    train_mod._emit_run_summary(
        config,
        run_dir=tmp_path,
        result={"success": True},
        seed=42,
        logger_=logging.getLogger(__name__),
        provenance=None,
    )
    assert json.loads((tmp_path / "run_summary.json").read_text())["author"] is None


# ---------------------------------------------------------------------------
# `--resume`: two discovery spellings, and the difference is load-bearing.
#
# `auto` raises when the checkpoint directory is empty. That is right when you
# meant to continue a specific run, and fatal for a WALL-CLOCK CHAIN, where
# every link runs the identical command and link 1 has nothing to resume from.
# Softening `auto` to cover both would have deleted the only spelling that can
# still tell you your checkpoints went missing, so `if-present` is a second
# value rather than a change to the first.
# ---------------------------------------------------------------------------


class TestResumeDiscoveryModes:
    @staticmethod
    def _source() -> str:
        import inspect

        from spectramr.pipelines import train

        return inspect.getsource(train.run_training_pipeline)

    def test_both_spellings_are_registered(self) -> None:
        from spectramr.pipelines.train import _RESUME_DISCOVERY_MODES

        assert set(_RESUME_DISCOVERY_MODES) == {"auto", "if-present"}

    def test_auto_still_raises_on_an_empty_directory(self) -> None:
        """The regression that would make this feature cost more than it gave:
        a silent fresh start when you asked to continue a run."""
        src = self._source()
        raise_at = src.index("No checkpoint found in {checkpoint_dir} for auto-resume")
        guard = src[max(0, raise_at - 300) : raise_at]
        assert '_resume_mode == "auto"' in guard, (
            "the strict spelling lost its raise -- `auto` on an empty directory "
            "now starts fresh, which is indistinguishable from resuming"
        )

    def test_an_unknown_spelling_raises_rather_than_being_read_as_a_path(self) -> None:
        """`--resume if_present` (underscore) must not be treated as a filename
        that happens not to exist yet (non-negotiable 3)."""
        src = self._source()
        assert "neither an existing checkpoint" in src
        assert "elif not Path(_resume_mode).exists():" in src

    def test_the_decision_is_stamped(self) -> None:
        """provenance.json is written at stage 8, before this decision exists,
        so the resume gets its own record (non-negotiable 8)."""
        src = self._source()
        assert "_stamp_resume_record(run_dir, _resume_record" in src
        for outcome in ('"outcome": "fresh"', '"outcome": "restored"'):
            assert outcome in src

    def test_the_record_appends_so_a_chain_is_readable(self, tmp_path) -> None:
        """One entry per link; the sequence of start_iteration values is what
        shows the chain advanced rather than restarting."""
        from spectramr.pipelines.train import _stamp_resume_record

        _stamp_resume_record(tmp_path, {"outcome": "fresh", "start_iteration": 0})
        _stamp_resume_record(tmp_path, {"outcome": "restored", "start_iteration": 5000})
        _stamp_resume_record(tmp_path, {"outcome": "restored", "start_iteration": 9500})

        history = json.loads((tmp_path / "resume_history.json").read_text())
        assert [e["start_iteration"] for e in history] == [0, 5000, 9500]
        assert all(e["recorded_at"] for e in history)

    def test_stamping_never_takes_down_a_run(self, tmp_path) -> None:
        """A provenance hiccup must not cost a job that is ready to train."""
        from spectramr.pipelines.train import _stamp_resume_record

        _stamp_resume_record(tmp_path / "does" / "not" / "exist", {"outcome": "fresh"})


class TestAYieldedRunDoesNotClaimCompletion:
    """A yielded run is mid-training, and the artifacts a finished run emits
    would describe a model that does not exist yet. Worse, they run INSIDE the
    save margin: a report long enough to reach the wall takes the launcher down
    with it, before the requeue is ever issued."""

    @staticmethod
    def _source() -> str:
        import inspect

        from spectramr.pipelines import train

        return inspect.getsource(train.run_training_pipeline)

    def test_reporting_is_skipped(self) -> None:
        src = self._source()
        assert 'if result.get("wall_clock_yield"):' in src
        gate = src.index('if result.get("wall_clock_yield"):')
        assert "_maybe_run_reporting" in src[gate : gate + 900]

    def test_the_marker_is_written_before_the_epilogue(self) -> None:
        """If Slurm kills the job during the epilogue, the marker must already
        be on disk — otherwise the run yielded and nothing ever requeues it."""
        src = self._source()
        assert src.index("_write_yield_marker(run_dir, result") < src.index("_emit_run_summary(")


# ---------------------------------------------------------------------------
# Async CUDA faults must say that their own traceback is misdirection
#
# One case per shape the classifier has to separate: each async fault it claims
# to recognise, a synchronous CUDA error it must NOT claim, and a plain Python
# error (non-negotiable 15).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "CUDA error: misaligned address",
        "CUDA error: an illegal memory access was encountered",
        "CUDA error: unspecified launch failure",
        "CUDA error: device-side assert triggered",
    ],
)
def test_async_cuda_faults_name_the_serialising_flag(message: str) -> None:
    """The operator gets CUDA_LAUNCH_BLOCKING=1 at the point of failure."""
    hint = train_mod._async_cuda_fault_hint(RuntimeError(message))
    assert hint is not None
    assert "CUDA_LAUNCH_BLOCKING=1" in hint


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("Expected all tensors to be on the same device"),
        ValueError("mask channel count 3 does not divide 8"),
    ],
    ids=["oom", "device-mismatch", "plain-python"],
)
def test_synchronous_failures_get_no_async_hint(exc: Exception) -> None:
    """A failure whose frame IS the culprit must not be labelled misdirection."""
    assert train_mod._async_cuda_fault_hint(exc) is None
