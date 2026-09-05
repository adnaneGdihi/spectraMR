"""Regression test for BB8.

``infer_command`` previously called ``run_inference_pipeline`` with
three kwargs the function never accepted (``model_path``,
``override_config``, ``override_model_type``). Every ``infer``
invocation crashed with ``TypeError``. The post-fix code passes only
the kwargs from the canonical signature in ``src/pipelines/infer.py``.
"""

from __future__ import annotations

import inspect

from spectramr.pipelines.infer import run_inference_pipeline
from tests.utils.repo_scripts import require_repo_file


def test_infer_command_only_uses_supported_kwargs() -> None:
    """Pin the signature contract so future drift fails this test."""
    sig_params = set(inspect.signature(run_inference_pipeline).parameters)
    expected = {
        "config_path",
        "checkpoint_path",
        "input_path",
        "output_path",
        "device",
        "batch_size",
    }
    assert expected <= sig_params, (
        f"run_inference_pipeline lost one of the expected kwargs: "
        f"{expected - sig_params}. infer_command depends on these."
    )

    # The legacy bad kwargs must NOT silently re-appear as accepted
    # parameter names (which would mask the regression).
    banned = {"model_path", "override_config", "override_model_type"}
    leaked = banned & sig_params
    assert not leaked, (
        f"run_inference_pipeline gained legacy kwarg(s) {leaked}. "
        "If a real config-override path is needed, surface it explicitly "
        "via TrainingSettings.from_yaml at the call site, not through a "
        "second parallel kwarg."
    )


def test_main_infer_command_does_not_pass_banned_kwargs() -> None:
    """Source-level guard against the bad call site re-appearing."""
    from pathlib import Path

    # Derive the source path from the imported module — the old hardcoded
    # ``src/main.py`` went stale in the src→spectramr refactor and the guard
    # silently failed on FileNotFoundError ever since.
    import spectramr.main as main_mod

    main_src = Path(main_mod.__file__).read_text()
    # Find the body of infer_command — between its `def` and the next
    # `def ` at column 0. Crude but enough to scope the check.
    start = main_src.index("def infer_command(args:")
    end = main_src.index("\ndef ", start + 1)
    body = main_src[start:end]

    for bad in ("model_path=", "override_config=", "override_model_type="):
        assert bad not in body, (
            f"infer_command re-introduced kwarg `{bad}` — "
            "run_inference_pipeline doesn't accept it (TypeError at runtime). "
            "See TODO/audit/00_implementation_tracker.md BB8."
        )


# ── Determinism knob wiring (2026-06 determinism SSOT) ─────────────────


def test_resolved_determinism_reads_training_knob() -> None:
    """``main._resolved_determinism`` must honour config.training.deterministic."""
    import types

    from spectramr.main import _resolved_determinism

    on = types.SimpleNamespace(training=types.SimpleNamespace(deterministic=True))
    off = types.SimpleNamespace(training=types.SimpleNamespace(deterministic=False))
    missing = types.SimpleNamespace(training=types.SimpleNamespace())
    no_training = types.SimpleNamespace(training=None)

    assert _resolved_determinism(on) is True
    assert _resolved_determinism(off) is False
    # Reproducible-by-default: absent knob means deterministic.
    assert _resolved_determinism(missing) is True
    assert _resolved_determinism(no_training) is True


def test_infer_command_resolves_seed_and_determinism_from_config(
    monkeypatch, tmp_path
) -> None:
    """Pitfall-#15 regression (2026-07-01): ``infer`` hardcoded
    ``initialize_accelerator(args.device, 42)``, silently ignoring
    ``run.seed`` and ``training.deterministic`` from the required
    ``--config`` YAML. Both are now resolved from the config.

    2026-07-14 (issue #285): ``fake_init`` was a **stale double** — it lacked the
    ``pipeline`` kwarg ``main.py`` now forwards, so every run of this test died
    with ``TypeError`` instead of checking anything. It has been red on ``dev``
    ever since, invisible because Actions is disabled on this repo. The fake now
    mirrors the real keyword-only signature, and ``pipeline`` is asserted rather
    than merely absorbed: it defaults to ``"train"`` in
    ``accelerator.initialize_accelerator``, so an ``infer`` call that stopped
    passing it would be graded against the wrong pipeline class by the
    accelerated-run contract (``HEAVY_PIPELINES``) — silently, and this test is
    the only thing watching.
    """
    import argparse

    import yaml

    import spectramr.main as main_mod

    with open(require_repo_file("experiments/inprogress/dummy/dummy_gan.yaml"), encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw.setdefault("training", {})["deterministic"] = False
    # `seed` is a fact about the RUN. Both `training.seed` and the top-level `seed` were
    # retired to `run.seed` on 2026-07-31 with posture=raise, so a fixture declaring
    # either fails to construct.
    raw.setdefault("run", {})["seed"] = 777
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")

    captured = {}

    def fake_init(device, seed=None, rank=0, deterministic=True, *, pipeline="train"):
        captured["init"] = (device, seed, deterministic)
        captured["pipeline"] = pipeline

    monkeypatch.setattr(main_mod, "initialize_accelerator", fake_init)
    monkeypatch.setattr(
        "spectramr.pipelines.run_inference_pipeline",
        lambda **kw: {"success": True},
    )
    args = argparse.Namespace(
        config=cfg,
        checkpoint=tmp_path / "ckpt.pt",
        input=tmp_path / "in",
        output=tmp_path / "out",
        device="cpu",
    )
    main_mod.infer_command(args)

    assert captured["init"] == ("cpu", 777, False), (
        "infer_command must resolve seed/determinism from --config, "
        "not hardcode (42, deterministic=True)"
    )
    assert captured["pipeline"] == "infer", (
        "infer_command must tag the accelerator init with pipeline='infer'; it "
        "defaults to 'train', so dropping it would silently grade the inference "
        "path against the wrong HEAVY_PIPELINES entry (accelerated-run contract)."
    )


def test_main_no_longer_hardcodes_cudnn_override() -> None:
    """The '[AUDIT FIX] Enforce Full Determinism' blocks must be gone.

    They overrode the cudnn policy AFTER ``initialize_accelerator`` on every
    command, making ``config.training.deterministic`` unreadable-by-design.
    The policy now lives in ``initialize_accelerator(deterministic=...)``.
    """
    import inspect

    import spectramr.main as main_mod

    src = inspect.getsource(main_mod)
    assert "cudnn.benchmark = False" not in src
    assert "cudnn.deterministic = True" not in src
