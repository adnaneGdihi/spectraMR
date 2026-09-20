"""Tests for the top-level ``spectramr`` CLI parser (:mod:`spectramr.cli.app`).

``build_parser`` was extracted from ``main`` (PR #130 N1) so the full
subcommand parser can be constructed and introspected WITHOUT executing a
command — the seam the ``launch`` verb-compatibility tests rely on.
"""

from __future__ import annotations

import argparse
import re

import pytest
import torch as _torch
import yaml as _yaml

from spectramr.cli.app import build_parser, main, predict
from spectramr.config.overrides import apply_overrides
from spectramr.config.settings import TrainingSettings
from spectramr.core.execution_ledger import ExecutionLedger
from spectramr.models.registry import register_model as _register_model
from tests.unit.config.test_settings import _minimal_config


@pytest.fixture(autouse=True)
def _disarm():
    """Keep the ledger out of neighbouring tests.

    ``_ACTIVE`` is a module-level ContextVar, so a test that arms the ledger
    leaves it armed for everything that runs after it in the same process --
    turning any later ``current() is None`` assertion into an order-dependent
    failure. Mirrors ``tests/unit/models/factories/test_model_factory.py``.
    """
    ExecutionLedger.reset()
    yield
    ExecutionLedger.reset()


def test_build_parser_returns_a_parser_with_subcommands():
    parser = build_parser()
    subact = next(
        a for a in parser._actions if getattr(a, "choices", None) and "train" in a.choices
    )
    # A representative spread of the registered subcommands is present.
    for verb in ("train", "audit", "campaign", "launch", "infer", "report"):
        assert verb in subact.choices


def test_parser_requires_a_subcommand():
    # ``required=True`` on the subparsers: a bare invocation must error out.
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_main_dispatches_via_build_parser(monkeypatch):
    # main() builds its parser through build_parser() and dispatches the
    # resolved handler — patch the parser so no real command runs.
    import argparse

    import spectramr.cli.app as app

    ns = argparse.Namespace(command="x", func=lambda a: 7, verbose=False)
    monkeypatch.setattr(app, "build_parser", lambda: _StubParser(ns))
    assert main(["anything"]) == 7


class _StubParser:
    def __init__(self, ns):
        self._ns = ns

    def parse_args(self, argv=None):
        return self._ns


def test_build_parser_is_pure_and_repeatable():
    # No shared mutable state: building twice yields independent, working parsers.
    p1, p2 = build_parser(), build_parser()
    assert p1 is not p2
    assert p1.parse_args(["train", "--config", "a.yaml"]).command == "train"
    assert p2.parse_args(["train", "--config", "b.yaml"]).config.name == "b.yaml"


# ---------------------------------------------------------------------------
# predict() — rewired onto the SSOT run_inference_pipeline (WS-A A5).
#
# predict previously used the deprecated, config-less InferencePipeline. It now
# delegates to the SAME entry point as ``infer`` and fails loud without a config
# instead of silently falling back to the removed deprecated path (pitfall #9).
# ---------------------------------------------------------------------------
import argparse as _argparse  # noqa: E402
from pathlib import Path as _Path  # noqa: E402


def test_predict_without_config_fails_loud(monkeypatch, caplog):
    """Config-less predict must return a non-zero code and NOT touch the
    deprecated InferencePipeline (no silent fallback)."""
    import spectramr.cli.app as app

    # If predict ever imports the deprecated pipeline again, blow up the test.
    def _boom(*a, **k):  # pragma: no cover - only fires on regression
        raise AssertionError("predict must not import the deprecated InferencePipeline")

    monkeypatch.setattr("spectramr.pipelines.infer.run_inference_pipeline", _boom, raising=True)

    ns = _argparse.Namespace(
        model=_Path("best.pt"),
        input=_Path("data/"),
        output=_Path("out/"),
        config=None,
        device="cpu",
    )
    rc = app.predict(ns)
    assert rc == 2
    assert any("requires --config" in r.message for r in caplog.records)


def test_predict_without_config_uses_the_artifact_beside_the_model(monkeypatch, tmp_path):
    """No --config, but the run directory holds resolved_config.json: predict rebuilds
    the settings from its declared block (#1379) and hands them to the pipeline."""
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.validation.resolved_config_artifact import write_resolved_config

    captured = {}

    def _fake_pipeline(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(
        "spectramr.pipelines.infer.run_inference_pipeline", _fake_pipeline, raising=True
    )
    run = tmp_path / "run"
    run.mkdir()
    write_resolved_config(run, TrainingSettings.from_yaml(str(_minimal_config_path())), run_id="t")
    model = run / "best.pt"
    model.write_bytes(b"")
    ns = _argparse.Namespace(
        model=model, input=_Path("data/"), output=_Path("out/"), config=None, device="cpu"
    )
    assert predict(ns) == 0
    assert captured["config_path"] is None
    assert captured["from_yaml"] is False
    assert isinstance(captured["settings"], TrainingSettings)


def test_predict_from_yaml_refuses_to_run_without_a_config(monkeypatch, tmp_path, caplog):
    """--from-yaml with no --config fails loud even when the artifact exists."""
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.validation.resolved_config_artifact import write_resolved_config

    monkeypatch.setattr(
        "spectramr.pipelines.infer.run_inference_pipeline",
        lambda **k: (_ for _ in ()).throw(AssertionError("must not run")),
        raising=True,
    )
    run = tmp_path / "run"
    run.mkdir()
    write_resolved_config(run, TrainingSettings.from_yaml(str(_minimal_config_path())), run_id="t")
    model = run / "best.pt"
    model.write_bytes(b"")
    ns = _argparse.Namespace(
        model=model,
        input=_Path("data/"),
        output=_Path("out/"),
        config=None,
        device="cpu",
        from_yaml=True,
    )
    assert predict(ns) == 2


def _minimal_config_path() -> _Path:
    """A real, loadable TrainingSettings YAML.

    ``predict`` now *loads* the config (``main.begin_inference_run``) instead of
    forwarding the path untouched, so a placeholder filename no longer suffices.
    That this test previously passed with a nonexistent ``run.yaml`` was itself
    the evidence: it asserted "predict forwards the path" and so could not
    detect that predict never read the SSOT it claimed to be driven by.
    """
    return _Path(__file__).resolve().parents[2] / "_fixtures" / "minimal_settings" / "gan.yaml"


def test_predict_with_config_delegates_to_ssot(monkeypatch):
    """With --config, predict forwards to run_inference_pipeline with the
    checkpoint (``--model``) mapped through as ``checkpoint_path``."""
    captured = {}

    def _fake_pipeline(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(
        "spectramr.pipelines.infer.run_inference_pipeline", _fake_pipeline, raising=True
    )

    cfg = _minimal_config_path()
    ns = _argparse.Namespace(
        model=_Path("best.pt"),
        input=_Path("data/"),
        output=_Path("out/"),
        config=cfg,
        device="cpu",
    )
    rc = predict(ns)
    assert rc == 0
    assert captured["config_path"] == cfg
    assert captured["checkpoint_path"] == _Path("best.pt")
    assert captured["input_path"] == _Path("data/")
    assert captured["output_path"] == _Path("out/")
    assert captured["device"] == "cpu"


def test_predict_runs_the_inference_preamble(monkeypatch):
    """``predict``'s docstring claimed parity with ``infer``; it had none of it.

    No execution ledger, no seed, no determinism policy, no ``torch.no_grad()``,
    and a hardcoded ``or "cuda"`` that bypassed the accelerated-run contract --
    even though ``"predict"`` is a registered member of ``HEAVY_PIPELINES``
    (non-negotiables 8 and 9b). Every ``predict`` run was non-reproducible while
    the docstring said otherwise (pitfall #16).
    """
    import torch

    seen = {}

    def _fake_pipeline(**kwargs):
        # Sampled *inside* the pipeline call, which is where they matter.
        seen["grad_enabled"] = torch.is_grad_enabled()
        ledger = ExecutionLedger.current()
        seen["ledger_source"] = None if ledger is None else ledger.source
        return {"status": "ok"}

    monkeypatch.setattr(
        "spectramr.pipelines.infer.run_inference_pipeline", _fake_pipeline, raising=True
    )

    cfg = _minimal_config_path()
    ns = _argparse.Namespace(
        model=_Path("best.pt"),
        input=_Path("data/"),
        output=_Path("out/"),
        config=cfg,
        device="cpu",
    )
    assert predict(ns) == 0
    assert seen["grad_enabled"] is False, "predict must run under torch.no_grad()"
    assert seen["ledger_source"] == str(cfg), (
        "predict must arm the execution ledger against the config it ran, so a "
        "knob the schema drops is recorded on this surface too"
    )


def test_predict_has_no_hardcoded_cuda_fallback():
    """9b: the device comes from the resolver, never from ``or "cuda"``."""
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[3] / "src" / "spectramr" / "cli" / "app.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "predict")
    literals = {
        n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert "cuda" not in literals, (
        'predict still hardcodes a "cuda" device; resolution belongs to '
        "spectramr.core.compute_device (non-negotiable 9b)"
    )
    called = {
        n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "begin_inference_run" in called, (
        "predict must share infer's preamble, not re-implement a subset"
    )


# ---------------------------------------------------------------------------
# export() ONNX branch — rewired onto the ONNXExporter SSOT (WS-C C1).
#
# The CLI previously inlined torch.onnx.export (opset 14, hardcoded axes). It now
# dispatches through ONNXExporter, passing the REAL get_sample_batch tensor as
# input_sample (satisfying the no-dummy guard).
# ---------------------------------------------------------------------------
def test_export_onnx_branch_dispatches_through_onnxexporter(monkeypatch, tmp_path):
    import torch

    import spectramr.cli.app as app

    real_sample = torch.zeros(1, 1, 8, 8)
    model = torch.nn.Conv2d(1, 1, 3, padding=1)

    monkeypatch.setattr("torch.load", lambda *a, **k: {"model": model}, raising=True)
    monkeypatch.setattr(
        "spectramr.config.settings.TrainingSettings.from_yaml",
        lambda *a, **k: object(),
        raising=True,
    )
    monkeypatch.setattr(
        "spectramr.shared.utils.data_utils.get_sample_batch",
        lambda *a, **k: real_sample,
        raising=True,
    )

    calls = {}

    class _SpyExporter:
        def __init__(self, opset_version=17):
            calls["opset"] = opset_version

        def export(self, m, path, input_sample=None, **kw):
            calls["model"] = m
            calls["input_sample"] = input_sample
            calls["path"] = path
            return path

    monkeypatch.setattr("spectramr.exports.onnx.ONNXExporter", _SpyExporter, raising=True)

    ns = _argparse.Namespace(model=tmp_path / "best.pt", format="onnx", config=tmp_path / "c.yaml")
    rc = app.export(ns)

    assert rc == 0
    assert calls["model"] is model
    # The REAL sample (not a fabricated dummy) is what gets traced.
    assert calls["input_sample"] is real_sample
    assert calls["opset"] == 17


def test_load_clean_volumes_from_h5_populates_real_assets(monkeypatch, tmp_path):
    """The .h5 loader prepares a real-asset MetricContext per content_id.

    Regression for WS1: the multi-coil reference (coil images + smaps) that the
    pseudo-GT produces must reach the metric context as REAL assets rather than
    being discarded to magnitude.
    """
    import torch

    import spectramr.cli.app as app
    from spectramr.core.metrics.context import MetricContext

    c, h, w = 4, 8, 8
    smaps = torch.ones(1, c, h, w, dtype=torch.complex64) / (c**0.5)
    coil_images = torch.ones(1, c, h, w, dtype=torch.complex64)
    x_gt_mag = torch.ones(1, 1, h, w)
    p99 = torch.tensor(1.0)

    monkeypatch.setattr(
        "spectramr.data.datasets.m4raw_repetition_groups.discover_repetition_groups",
        lambda *a, **k: [[_Path("2022091411_T201.h5")]],
        raising=True,
    )
    monkeypatch.setattr(
        "spectramr.infrastructure.physics.m4raw_pseudo_gt.synthesize_pseudo_gt",
        lambda *a, **k: (coil_images, smaps, x_gt_mag, p99),
        raising=True,
    )

    assets: dict[str, object] = {}
    volumes = app._load_clean_volumes_from_h5(tmp_path, max_subjects=1, assets_out=assets)

    assert volumes and volumes[0][0] == "2022091411_T2"
    ctx = assets["2022091411_T2"]
    assert isinstance(ctx, MetricContext)
    assert ctx.coil_maps is smaps  # real maps threaded, not synthesised
    assert ctx.y_kspace is not None
    assert ctx.reconstructor is not None


class TestAuditEmitsResolvedConfig:
    """The audit resolves the same config training will, so it must record it.

    Before this, the pre-flight surface — the one whose entire job is to catch
    problems before GPU time — was the only one leaving no artifact: its report
    goes to stdout and under ``--json`` to a pipe. A config's dropped knobs were
    therefore discoverable only after committing to the run.
    """

    CONFIG = """config_version: '1.0'
model:
  model_type: unet
data:
  batch_size: 2
training:
  max_iterations: 10
optimization:
  learning_rate: 1.0e-4
logging: {}
"""

    def _run(self, tmp_path, extra_yaml="", args=()):
        import os
        import subprocess
        import sys

        cfg = tmp_path / "arm.yaml"
        cfg.write_text(self.CONFIG + extra_yaml)
        # Inherit the environment (PYTHONPATH included) rather than replacing it:
        # a stripped env makes spectramr unimportable in the child, and the test
        # would then "fail" for a reason unrelated to what it asserts.
        env = {**os.environ, "SPECTRAMR_SUPPRESS_CLINICAL_WARNING": "1"}
        return subprocess.run(
            [sys.executable, "-m", "spectramr.cli", "audit", str(cfg), *args],
            capture_output=True,
            text=True,
            env=env,
        ), cfg

    def test_write_resolved_config_flag_emits_the_artifact(self, tmp_path):
        out = tmp_path / "out"
        result, _ = self._run(tmp_path, args=("--write-resolved-config", str(out)))
        assert result.returncode in (0, 1, 2), result.stderr[-2000:]

        import json

        path = out / "resolved_config.json"
        assert path.exists(), f"audit wrote no artifact. stderr:\n{result.stderr[-2000:]}"
        ledger = json.loads(path.read_text())["_ledger"]
        assert ledger["write_status"] == "ok"
        assert ledger["source"].endswith("arm.yaml")

    def test_the_audit_records_a_dropped_knob_before_any_gpu_time(self, tmp_path):
        """DEFECT: the #550 class, caught pre-flight instead of at hour eight."""
        import json

        out = tmp_path / "out"
        result, _ = self._run(
            tmp_path,
            extra_yaml="acceleration:\n  center_fraction: 0.08\n  min_centre_fraction: 0.02\n",
            args=("--write-resolved-config", str(out)),
        )
        path = out / "resolved_config.json"
        assert path.exists(), result.stderr[-2000:]

        dropped = [
            s
            for s in json.loads(path.read_text())["_ledger"]["substitutions"]
            if s["class_id"] == "extra_ignore_dropped"
        ]
        assert [s["path"] for s in dropped] == ["acceleration.min_centre_fraction"]
        assert dropped[0]["severity"] == "error"

    def test_clean_config_records_no_drop(self, tmp_path):
        """CONTROL: proves the audit is not simply always reporting drops."""
        import json

        out = tmp_path / "out"
        self._run(tmp_path, args=("--write-resolved-config", str(out)))
        ledger = json.loads((out / "resolved_config.json").read_text())["_ledger"]
        dropped = [s for s in ledger["substitutions"] if s["class_id"] == "extra_ignore_dropped"]
        assert dropped == []

    def test_no_flag_means_no_artifact(self, tmp_path):
        """Opt-in: the audit must not litter directories by default."""
        _result, cfg = self._run(tmp_path)
        assert not (cfg.parent / "resolved_config.json").exists()


# ---------------------------------------------------------------------------
# export() model construction — routed onto ModelBuilder + resolve_state_dict.
#
# `export` used to read `checkpoint.get("model", checkpoint)` and bail with
#
#     "Error: Checkpoint contains state_dict, not model. Cannot export."
#
# on anything that was not a pickled nn.Module. That is every ordinary training
# checkpoint this repo writes (`checkpoint_director` emits `{"generator": ...}`),
# so the verb did not work on the artifacts it exists to consume. `--config` is
# `required=True`, so the architecture was available the whole time.
#
# These tests must FAIL on the pre-fix code: it never reaches an exporter.
# ---------------------------------------------------------------------------
@_register_model("_export_witness", "reconstruction")
class _ExportWitness(_torch.nn.Module):
    """A registered generator whose width is readable off the built module."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()
        self.conv = _torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        return self.conv(x)


class TestExportBuildsTheModelTheConfigDeclares:
    """`export` reconstructs the architecture rather than refusing the file."""

    @staticmethod
    def _arm(tmp_path):
        cfg = {
            "config_version": "1.0",
            "model": {
                "model_type": "_export_witness",
                "in_channels": 3,
                "out_channels": 3,
            },
            "data": {
                "sampling": {"patch_size": [16, 16]},
                "loader": {"batch_size": 1},
            },
            "optimization": {},
            "logging": {},
        }
        path = tmp_path / "arm.yaml"
        path.write_text(_yaml.safe_dump(cfg))
        return path

    @staticmethod
    def _spy_exporter(monkeypatch, calls):
        class _Spy:
            def __init__(self, opset_version=17):
                pass

            def export(self, m, path, input_sample=None, **kw):
                calls["model"] = m
                return path

        monkeypatch.setattr("spectramr.exports.onnx.ONNXExporter", _Spy, raising=True)
        monkeypatch.setattr(
            "spectramr.shared.utils.data_utils.get_sample_batch",
            lambda *a, **k: _torch.zeros(1, 3, 16, 16),
            raising=True,
        )

    def _run(self, monkeypatch, tmp_path, payload):
        import argparse as _ap

        import spectramr.cli.app as app

        calls: dict = {}
        self._spy_exporter(monkeypatch, calls)
        ckpt = tmp_path / "checkpoint_best.pt"
        _torch.save(payload, ckpt)
        rc = app.export(_ap.Namespace(model=ckpt, format="onnx", config=self._arm(tmp_path)))
        return rc, calls

    def test_a_generator_envelope_exports(self, monkeypatch, tmp_path):
        """The envelope `checkpoint_director` actually writes (#1310)."""
        ref = _ExportWitness(in_channels=3, out_channels=3)
        rc, calls = self._run(monkeypatch, tmp_path, {"generator": ref.state_dict()})

        assert rc == 0, "a real training checkpoint must be exportable"
        assert "model" in calls, "the exporter was never reached"

    def test_the_exported_weights_are_the_trained_ones(self, monkeypatch, tmp_path):
        """Anti-facade: building the architecture is not the same as loading it.

        A rebuild that skipped the load would still return 0 and still hand the
        exporter an nn.Module — and would ship randomly-initialised weights
        inside a file that looks authoritative.
        """
        ref = _ExportWitness(in_channels=3, out_channels=3)
        _torch.nn.init.constant_(ref.conv.weight, 0.375)
        rc, calls = self._run(monkeypatch, tmp_path, {"generator": ref.state_dict()})

        assert rc == 0
        exported = calls["model"].state_dict()
        for k, v in ref.state_dict().items():
            assert _torch.equal(exported[k], v), f"{k} did not load"

    def test_the_declared_width_is_not_defaulted(self, monkeypatch, tmp_path):
        """The arm declares 3 channels; the schema default is 1."""
        ref = _ExportWitness(in_channels=3, out_channels=3)
        rc, calls = self._run(monkeypatch, tmp_path, {"generator": ref.state_dict()})

        assert rc == 0
        assert calls["model"].conv.in_channels == 3

    def test_a_partial_state_dict_is_refused(self, monkeypatch, tmp_path, caplog):
        """strict=True: an export is a frozen artifact handed to another runtime.

        Loading what matches and shipping the rest random is the failure mode
        `strict` exists to stop — and the reason this reader is stricter than
        the warm-start one.
        """
        import logging

        caplog.set_level(logging.ERROR, logger="spectramr.cli.app")
        ref = _ExportWitness(in_channels=3, out_channels=3)
        partial = {k: v for k, v in ref.state_dict().items() if "bias" not in k}
        rc, calls = self._run(monkeypatch, tmp_path, {"generator": partial})

        assert rc == 1
        assert "model" not in calls, "a partially-loaded model reached the exporter"
        # The REASON matters: the pre-fix code also returned 1 here, but only
        # because it refused every non-pickled checkpoint. Naming the missing
        # key is what proves the load was attempted and rejected on its merits.
        assert any("conv.bias" in r.getMessage() for r in caplog.records), (
            f"the refusal did not name the missing key: {[r.getMessage() for r in caplog.records]}"
        )

    def test_an_unrecognised_envelope_is_refused(self, monkeypatch, tmp_path, caplog):
        """Zero overlap raises in `resolve_state_dict` — no fifth reader guesses."""
        import logging

        caplog.set_level(logging.ERROR, logger="spectramr.cli.app")
        rc, calls = self._run(monkeypatch, tmp_path, {"totally": {"unrelated": 1}})

        assert rc == 1
        assert "model" not in calls
        # Same discrimination as above: the shared reader's own words, not the
        # blanket "Cannot export" the pre-fix code emitted for everything.
        assert any("shares zero parameter names" in r.getMessage() for r in caplog.records), (
            f"the refusal came from somewhere else: {[r.getMessage() for r in caplog.records]}"
        )

    def test_a_pickled_module_is_used_as_is(self, monkeypatch, tmp_path):
        """Branch order: older exports hold a module, and must not be rebuilt."""
        import argparse as _ap

        import spectramr.cli.app as app

        calls: dict = {}
        self._spy_exporter(monkeypatch, calls)

        built = {"n": 0}

        def _no_build(*a, **k):
            built["n"] += 1
            raise AssertionError("a pickled module must not be rebuilt")

        monkeypatch.setattr(
            "spectramr.infrastructure.training.builders.model_builder.ModelBuilder.__init__",
            _no_build,
            raising=True,
        )
        pickled = _ExportWitness(in_channels=3, out_channels=3)
        _torch.nn.init.constant_(pickled.conv.weight, 0.125)
        ckpt = tmp_path / "legacy.pt"
        _torch.save({"model": pickled}, ckpt)

        rc = app.export(_ap.Namespace(model=ckpt, format="onnx", config=self._arm(tmp_path)))

        assert built["n"] == 0, "the pickled module was rebuilt from the config"
        assert rc == 0
        # `torch.load` returns a copy, so identity cannot hold -- what must hold
        # is that the exported module IS the checkpoint's, weights and all.
        assert type(calls["model"]) is _ExportWitness
        assert _torch.equal(calls["model"].conv.weight, pickled.conv.weight)


# --------------------------------------------------------------------------
# The ``-O`` examples printed by ``--help`` must be paths the config accepts.
#
# ``TrainingSettings`` is ``extra="forbid"``, so an override naming a path that
# is neither canonical nor a folding legacy spelling does not degrade -- it
# raises out of ``apply_overrides`` before training starts. A help string is
# read at exactly the moment an operator is composing an override, so a wrong
# example there is a facade (pitfall #16): it presents a working example that
# is not one. #1314 was two such strings (``validation.val_interval``).
#
# This walks the WHOLE parser tree rather than the two known sites, so a new
# subcommand that copies the example inherits the gate.
# --------------------------------------------------------------------------

_OVERRIDE_EXAMPLE = re.compile(r"(?:-O|--override)[=\s]+([A-Za-z_][\w.]*=[^\s,)]+)")


def _iter_parsers(parser):
    """Yield ``parser`` and every parser reachable through its subparsers."""
    yield parser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                yield from _iter_parsers(sub)


def _override_examples() -> list[tuple[str, str]]:
    """Every ``KEY=VALUE`` an ``-O``/``--override`` example tells a user to type.

    Scans option ``help`` strings plus each parser's own ``description`` and
    ``epilog`` -- all three render into ``--help`` output.
    """
    found: list[tuple[str, str]] = []
    for parser in _iter_parsers(build_parser()):
        texts = [parser.description or "", parser.epilog or ""]
        texts += [a.help or "" for a in parser._actions]
        for text in texts:
            found += [(parser.prog, m.group(1)) for m in _OVERRIDE_EXAMPLE.finditer(text)]
    return found


def test_every_override_example_in_cli_help_is_a_path_the_config_accepts():
    examples = _override_examples()

    # Non-vacuity in both directions: the walk must reach the nested
    # subcommand parsers, and it must actually find examples there. Without
    # this the test passes trivially if the walk or the regex ever breaks.
    progs = {prog for prog, _ in examples}
    assert examples, "walk found no -O examples at all -- the regex or the walk is broken"
    assert any(p.endswith("train") for p in progs), progs
    assert any(p.endswith("sanity_check") for p in progs), progs

    base = TrainingSettings(**_minimal_config())
    broken = []
    for prog, example in examples:
        try:
            apply_overrides(base, [example])
        except Exception as exc:
            broken.append(f"{prog}: -O {example} -> {type(exc).__name__}: {exc!s:.140}")
    assert not broken, "CLI --help advertises override paths the config rejects:\n" + "\n".join(
        broken
    )


def test_the_override_example_walk_would_catch_a_bad_path():
    """CONTROL: proves the check above fails on a path that does not exist.

    Without this, a regex that silently matches nothing would make the test
    green for the wrong reason.
    """
    base = TrainingSettings(**_minimal_config())
    with pytest.raises(Exception):  # noqa: B017 -- any refusal is the point
        apply_overrides(base, ["validation.val_interval=100"])


def test_bulk_counts_use_the_reports_definition_of_a_warning() -> None:
    """Planted violation: a passed advisory carrying severity "warning" is not a warning.

    Counting severities made three ``vf_01`` arms ERROR(strict) in the bulk run
    while the single-arm audit exited 0 (2026-09-03); the report owns the rule.
    """
    from types import SimpleNamespace

    from spectramr.cli.app import _bulk_counts

    failed_warning = SimpleNamespace(passed=False, severity="warning")
    report = SimpleNamespace(errors=[], warnings=[failed_warning])
    assert _bulk_counts(report) == (0, 1)
    advisory_only = SimpleNamespace(errors=[], warnings=[])
    assert _bulk_counts(advisory_only) == (0, 0)


def test_bulk_audit_builds_the_same_report_as_the_single_arm_audit(tmp_path, monkeypatch) -> None:
    """Planted violation: a bulk loop that calls validate_config_health directly
    never reaches this spy. Both surfaces go through ``_audit_report`` (2026-09-03)."""
    import argparse
    from types import SimpleNamespace

    from spectramr.cli import app

    arm = tmp_path / "arm.yaml"
    arm.write_text("config_version: '1.0'\n")
    seen: list[str] = []

    def _spy(config_path, config):
        seen.append(config_path)
        return SimpleNamespace(errors=[], warnings=[], results=[], passed=True)

    monkeypatch.setattr(app, "_audit_report", _spy)
    monkeypatch.setattr(
        "spectramr.config.settings.TrainingSettings.from_yaml",
        staticmethod(lambda p: SimpleNamespace()),
    )
    args = argparse.Namespace(config=tmp_path, json=False, strict=True, exclude=None, probe=False)
    app._audit_bulk(args)
    assert seen == [str(arm)]


# ------------------------------------------------------------- --val-batches


class TestValBatchesFlag:
    """``--val-batches`` on the two verbs that run a training-time validation.

    The flag is sugar for an override, and the lowering happens once in
    ``_dispatch_training`` -- so these assert the *observable* result (what lands
    in ``args.override``), not that a helper was called.
    """

    @pytest.mark.parametrize("verb", ["train", "sanity_check"])
    def test_both_training_verbs_accept_it(self, verb):
        args = build_parser().parse_args(
            [verb, "--config", "c.yaml", "--val-batches", "2"]
        )
        assert args.val_batches == 2

    @pytest.mark.parametrize("verb", ["train", "sanity_check"])
    def test_it_defaults_to_none_so_a_plain_run_is_unchanged(self, verb):
        args = build_parser().parse_args([verb, "--config", "c.yaml"])
        assert args.val_batches is None

    def _dispatch(self, monkeypatch, argv):
        """Parse ``argv`` and run the dispatcher with the bootstrap stubbed out."""
        import spectramr.main as _main

        seen: dict[str, object] = {}
        monkeypatch.setattr(
            _main, "train_command", lambda a: seen.update(override=a.override)
        )
        monkeypatch.setattr(
            _main, "sanity_check_command", lambda a: seen.update(override=a.override)
        )
        args = build_parser().parse_args(argv)
        args.func(args)
        return seen["override"]

    @pytest.mark.parametrize(
        ("verb", "flag"), [("train", "--val-batches"), ("sanity_check", "--val-batches")]
    )
    def test_dispatch_lowers_it_onto_the_key_the_framework_reads(
        self, monkeypatch, verb, flag
    ):
        overrides = self._dispatch(
            monkeypatch, [verb, "--config", "c.yaml", flag, "3"]
        )
        assert overrides == ["validation.loader.num_batches=3"]

    def test_it_composes_with_an_unrelated_override(self, monkeypatch):
        overrides = self._dispatch(
            monkeypatch,
            ["train", "--config", "c.yaml", "-O", "training.max_iterations=5",
             "--val-batches", "1"],
        )
        assert overrides == [
            "training.max_iterations=5",
            "validation.loader.num_batches=1",
        ]

    def test_without_the_flag_no_override_is_invented(self, monkeypatch):
        assert self._dispatch(monkeypatch, ["train", "--config", "c.yaml"]) is None

    def test_conflicting_with_an_explicit_override_raises(self, monkeypatch):
        with pytest.raises(ValueError, match=r"conflicts with the override"):
            self._dispatch(
                monkeypatch,
                ["train", "--config", "c.yaml",
                 "-O", "validation.loader.num_batches=9", "--val-batches", "1"],
            )

    def test_the_override_it_emits_survives_the_real_loader(self):
        """The lowering is only worth anything if `apply_overrides` accepts it."""
        settings = TrainingSettings(**_minimal_config())
        updated = apply_overrides(settings, ["validation.loader.num_batches=2"])
        assert updated.validation.loader.num_batches == 2
