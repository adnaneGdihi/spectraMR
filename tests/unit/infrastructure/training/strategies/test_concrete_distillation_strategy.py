"""Regression pins for ConcreteDistillationStrategy (smoke audit 2026-06-03).

F7a — ``validation_step`` used to call ``_compute_losses_impl(target, target)``
(target compared to itself), so it returned only loss scalars and never an
actual reconstruction metric. Early stopping monitors ``val_robust_mri_psnr``,
which was therefore never emitted ("Early Stopping monitor metric ... not found
in validation metrics", smoke run 20260603_182243). It now runs the blind
student forward on a corrupted target and computes ``val_``-prefixed metrics via
the SSOT validation-metrics computer.

F7b — ``_load_teacher`` logged a WARNING when no teacher checkpoint was
configured, even though no-teacher is a supported, graceful mode (the
distillation loss simply becomes 0.0). A WARNING here trips the smoke audit
(CLAUDE.md #10) on every distillation arm run without a Method-A checkpoint, so
it is now INFO.

These are grep-pins (source-text assertions) — instantiating the strategy needs
a full TrainingEnvironment + generator + simulator, which is out of scope for a
unit test. The behavioural contract is pinned so a future refactor cannot
silently regress it.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest
import torch

# Anchored to this file, not to the CWD. A bare ``pathlib.Path("src/...")``
# resolves against the process working directory, and this read happens at
# MODULE level -- so launching pytest from anywhere but the repo root raises
# FileNotFoundError during collection, and a collection error is not a test
# failure: pytest discards the whole session. Running from ``tests/`` was
# measured to abort the run with "Interrupted: 2 errors during collection",
# taking every unrelated test with it.
_SRC_PATH = (
    pathlib.Path(__file__).resolve().parents[5]
    / "src/spectramr/infrastructure/training/strategies/distillation_strategy.py"
)
_SRC = _SRC_PATH.read_text(encoding="utf-8")


def _validation_step_body() -> str:
    """Return the source of ``validation_step`` (up to the next method or EOF)."""
    start = _SRC.index("def validation_step(")
    rest = _SRC[start + 1 :]
    nxt = rest.find("\n    def ")  # validation_step may be the last method
    end = len(rest) if nxt == -1 else nxt
    return _SRC[start : start + 1 + end]


def test_validation_step_does_not_compare_target_to_itself() -> None:
    body = _validation_step_body()
    assert "_compute_losses_impl(target, target" not in body, (
        "validation_step must not grade the target against itself — that "
        "returns losses, never val_robust_mri_psnr (smoke audit 2026-06-03, F7a)"
    )


def test_validation_step_emits_val_metrics_from_forward_pass() -> None:
    body = _validation_step_body()
    # Runs the student forward and routes through the SSOT metrics computer,
    # returning val_-prefixed metrics so early stopping resolves its monitor.
    assert "_get_validation_metrics_computer" in body
    assert "self.generator_model" in body
    assert 'f"val_{k}"' in body


def test_validation_step_caches_image_domain_visuals() -> None:
    """F2 (smoke 2026-06-16): validation_step must cache ``_last_visual_pred`` /
    ``_last_visual_target`` (magnitude images) so train.py's override seam logs
    real reconstructions instead of falling back to a raw generator forward that
    raises (1-ch input → 2-ch conv) and yields "visual_samples never captured".
    """
    body = _validation_step_body()
    assert "self._last_visual_pred" in body
    assert "self._last_visual_target" in body
    # image-domain magnitude (the train.py seam treats these as image-domain)
    assert ".abs()" in body


def test_target_routed_through_image_domain_seam_both_paths() -> None:
    """F-kspace-real (smoke 2026-06-13): the svd-coil method_c arm delivers a
    k-space target; both ``_compute_losses_impl`` and ``validation_step`` must
    route it through ``_ensure_image_domain_target`` before the Digital Twin /
    loss / cached visuals, or the clean target / REAL reference is raw |k-space|
    and the FAKE collapses to black. The seam is a no-op for the rss_image
    distillation arms (eval_c2/c3/c7, exp_c4) via the SSOT domain decision.
    """
    # Both target-prep sites convert before the simulator is called.
    assert _SRC.count("_ensure_image_domain_target(target_complex)") >= 2, (
        "both _compute_losses_impl and validation_step must IFFT a k-space target"
    )
    # Each conversion precedes the digital-twin simulator call in its block.
    body = _validation_step_body()
    conv = body.index("_ensure_image_domain_target(target_complex)")
    sim = body.index("self.simulator(target_complex)")
    assert conv < sim, "validation_step must convert the target BEFORE the twin"


def test_no_teacher_branch_logs_info_not_warning() -> None:
    anchor = "No teacher checkpoint configured"
    assert anchor in _SRC
    idx = _SRC.index(anchor)
    preceding = _SRC[max(0, idx - 300) : idx]
    assert "logger.info(" in preceding, (
        "no-teacher is a supported mode — it must log at INFO, not WARNING "
        "(CLAUDE.md #10; smoke audit 2026-06-03, F7b)"
    )
    assert "logger.warning(" not in preceding


def _bare_strategy_with_teacher(teacher_checkpoint):
    from spectramr.infrastructure.training.strategies.distillation_strategy import (
        ConcreteDistillationStrategy,
    )

    s = ConcreteDistillationStrategy.__new__(ConcreteDistillationStrategy)
    s.device = "cpu"
    s._teacher = None
    s.config = SimpleNamespace(
        training=SimpleNamespace(
            distillation=SimpleNamespace(teacher_checkpoint=teacher_checkpoint)
        ),
        model=SimpleNamespace(in_channels=1, out_channels=1, model_kwargs=None),
    )
    return s


def test_configured_teacher_load_failure_raises(monkeypatch) -> None:
    """When ``training.distillation.teacher_checkpoint`` IS set but the load fails,
    the strategy must RAISE — a run that explicitly requested distillation must not
    silently train as plain reconstruction (pitfall #16). Contrast with the
    unconfigured graceful mode above."""
    import spectramr.infrastructure.training.strategies.distillation_strategy as ds

    # A Linear stands in for the oracle UNet: the failure under test is the load.
    monkeypatch.setattr(ds, "_build_teacher", lambda config: torch.nn.Linear(1, 1))
    s = _bare_strategy_with_teacher("/nonexistent/teacher_checkpoint.pt")
    with pytest.raises(RuntimeError, match="[Tt]eacher"):
        s._load_teacher()


def test_unconfigured_teacher_is_graceful() -> None:
    """No ``training.distillation.teacher_checkpoint`` -> supported no-teacher mode,
    no raise, teacher None."""
    s = _bare_strategy_with_teacher(None)
    s._load_teacher()
    assert s._teacher is None


def test_no_distillation_block_is_the_no_teacher_mode() -> None:
    """``training.distillation`` defaults to ``None``: no teacher, no raise."""
    s = _bare_strategy_with_teacher(None)
    s.config.training.distillation = None
    s._load_teacher()
    assert s._teacher is None


def test_a_configured_teacher_loads_through_weights(tmp_path, monkeypatch) -> None:
    import spectramr.infrastructure.training.strategies.distillation_strategy as ds

    seen: dict = {}
    monkeypatch.setattr(ds, "weights", lambda ref, part, into, *, policy: seen.update(
        ref=ref, part=part, policy=policy))
    monkeypatch.setattr(ds, "_build_teacher", lambda config: torch.nn.Linear(1, 1))
    s = _bare_strategy_with_teacher(str(tmp_path / "best.safetensors"))
    s._load_teacher()
    assert seen == {"ref": str(tmp_path / "best.safetensors"), "part": "generator", "policy": "overlap"}
    assert all(not p.requires_grad for p in s._teacher.parameters())


def test_the_student_latent_probe_reads_through_a_wrapper() -> None:
    """A DDP/DP wrapper's forward takes only ``*inputs, **kwargs``; probing it dropped the
    distillation term silently (#2629)."""
    from spectramr.infrastructure.training.strategies.distillation_strategy import (
        ConcreteDistillationStrategy,
    )

    class _Student(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, 1)

        def forward(self, x, return_latent=False):
            return self.conv(x)

    wrapped = torch.nn.DataParallel(_Student())
    assert ConcreteDistillationStrategy._student_returns_latent(wrapped)
    assert not ConcreteDistillationStrategy._student_returns_latent(torch.nn.Conv2d(1, 1, 1))
