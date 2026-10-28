"""The CLI flags every launcher forwards to ``run_training_pipeline``.

``main.train_command``, ``main.experiment_command`` and the distributed
launcher (``pipelines.distributed.run_distributed_training``, the one the
sharded n2n arms use) each call ``run_training_pipeline`` once; a flag a parser
accepts but a call does not forward is advertised and inert on that launcher
(non-negotiable 8). The calls are pinned by reading the source, because driving
any of them builds a container and an accelerator, which a unit test must not.
"""

from __future__ import annotations

import ast
import inspect

import spectramr.main as main_mod
import spectramr.pipelines.distributed as distributed_mod


def _pipeline_calls(module) -> list[ast.Call]:
    tree = ast.parse(inspect.getsource(module))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_training_pipeline"
    ]


def test_every_pipeline_call_forwards_validate_only_and_resume() -> None:
    main_calls = _pipeline_calls(main_mod)
    assert len(main_calls) >= 2, "train_command and experiment_command each call the pipeline"
    distributed_calls = _pipeline_calls(distributed_mod)
    assert len(distributed_calls) == 1, "one launcher call per torchrun worker"
    for call in [*main_calls, *distributed_calls]:
        keywords = {kw.arg for kw in call.keywords}
        assert {"resume_path", "validate_only"} <= keywords, sorted(keywords)
        forwarded = next(kw.value for kw in call.keywords if kw.arg == "validate_only")
        assert "validate_only" in ast.unparse(forwarded)


def _live_train_exit_code(monkeypatch, result) -> int | None:
    """Drive ``__common_train_setup`` with the pipeline stubbed to return ``result``."""
    import argparse

    from spectramr.config.settings import TrainingSettings
    from spectramr.core.execution_ledger import ExecutionLedger

    settings = TrainingSettings(model={}, data={}, optimization={}, logging={})
    monkeypatch.setattr(
        main_mod.TrainingSettings, "from_yaml", classmethod(lambda _cls, _p: settings)
    )
    monkeypatch.setattr(ExecutionLedger, "begin_run", classmethod(lambda _cls, **_k: None))
    monkeypatch.setattr(main_mod, "initialize_accelerator", lambda *a, **k: "cpu")
    monkeypatch.setattr("spectramr.pipelines.run_training_pipeline", lambda *a, **k: result)
    args = argparse.Namespace(
        config="unused.yaml", seed=None, device=None, override=None, dry_run=False, resume=None
    )
    try:
        getattr(main_mod, "__common_train_setup")(args)
    except SystemExit as exc:
        return exc.code
    return None


def test_a_result_without_a_success_flag_exits_nonzero(monkeypatch) -> None:
    """#181: the CLI defaulted an absent flag to True and exited 0."""
    assert _live_train_exit_code(monkeypatch, {"iterations_completed": 3}) == 1


def test_an_explicit_success_exits_zero(monkeypatch) -> None:
    assert _live_train_exit_code(monkeypatch, {"success": True}) is None


def test_a_reported_failure_exits_nonzero(monkeypatch) -> None:
    assert _live_train_exit_code(monkeypatch, {"success": False, "error": "x"}) == 1
