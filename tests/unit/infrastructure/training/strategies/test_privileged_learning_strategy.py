"""The domain-adversarial term uses real per-sample labels and never fabricates them.

The strategy once read ``batch.get("domain_label", 1)``, so every sample carried
domain 1 and the gradient-reversal discriminator was trained to separate one class
from itself (the A6 census's single "substitutes" case). The label now comes from
``DomainLabelledBalancer`` (``data.multi_domain.balancing: domain_labelled``); with
``delta > 0`` and no label the strategy raises, and with labels the recon and
distill terms are restricted to the synthetic samples (spec §4.3).
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from spectramr.data.builders.site_balancer import build_balancer
from spectramr.infrastructure.training.strategies.privileged_learning_strategy import (
    PrivilegedLearningStrategy,
)


class _TinyStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def write_teacher_checkpoint(path: Path, teacher: nn.Module) -> Path:
    """A real best checkpoint whose generator part is ``teacher``'s state dict."""
    from spectramr.infrastructure.checkpointing import codecs
    from tests.unit.infrastructure.checkpointing.conftest import make_run_state

    state = make_run_state("safetensors", role="best")
    state["models"], state["ema"] = {"generator": teacher.state_dict()}, None
    path.parent.mkdir(parents=True, exist_ok=True)
    codecs.write(path, state)
    return path


def _strategy(delta: float = 0.1) -> PrivilegedLearningStrategy:
    s = object.__new__(PrivilegedLearningStrategy)
    s._owned_modules, s._owned_wrappers = {}, {}
    s.device = torch.device("cpu")
    dummy = nn.Parameter(torch.zeros(1))
    s.env = types.SimpleNamespace(
        generator=_TinyStudent(),
        opt_g=torch.optim.SGD([dummy], lr=0.1),
    )
    s.config = types.SimpleNamespace(model=types.SimpleNamespace(out_channels=1))
    s._disc_hidden_dim = 16
    s._teacher = None
    s._domain_discriminator = None
    # beta 0: these tests are about the domain term, which needs no teacher.
    s._alpha, s._beta, s._delta = 1.0, 0.0, delta
    s._tau0, s._n_min, s._n_max = 5.0, 0.0, 1.0
    s._stage, s._teacher_checkpoint, s._declared_knobs = "student", None, frozenset()
    s._setup_strategy_specific_components()
    return s


def _call(s, batch, batch_size: int = 2):
    return s._compute_losses_impl(
        input_batch=torch.randn(batch_size, 1, 8, 8),
        target_batch=torch.randn(batch_size, 1, 8, 8),
        epoch=0,
        batch={"marker": torch.randn(batch_size, 1, 8, 8), **batch},
    )


class TestDomainLabelIsRequired:
    def test_an_absent_label_raises_instead_of_fabricating_one(self) -> None:
        """The regression. Pre-fix this returned a finite, meaningless loss."""
        with pytest.raises(ValueError) as excinfo:
            _call(_strategy(), {})

        message = str(excinfo.value)
        assert "domain_label" in message
        # The message must name BOTH exits, or it just blocks the run.
        assert "delta: 0.0" in message, "the opt-out is not stated"
        assert "balancing: domain_labelled" in message, "the producer is not named"

    def test_the_term_is_optional_and_delta_zero_is_the_way_out(self) -> None:
        """Opting out must not require inventing labels.

        ``delta: 0.0`` is the declared "I am not running the domain term" path,
        so a labelless batch has to stay legal there -- otherwise the fix would
        make the strategy unusable rather than honest.
        """
        losses = _call(_strategy(delta=0.0), {})

        assert "g_total_loss" in losses
        assert "g_priv_domain" not in losses, (
            "the domain term contributed despite delta=0"
        )

    def test_real_labels_are_used_per_sample(self) -> None:
        """Anti-vacuity: the fix must pass the labels through, not just gate.

        Two batches identical except for their domain labels must produce
        different domain losses. Under the old constant-``1`` default they were
        necessarily equal, which is precisely the defect.
        """
        torch.manual_seed(0)
        s = _strategy()
        x = torch.randn(4, 1, 8, 8)
        target = torch.randn(4, 1, 8, 8)
        marker = torch.randn(4, 1, 8, 8)

        def domain_loss(labels):
            out = s._compute_losses_impl(
                input_batch=x,
                target_batch=target,
                epoch=0,
                batch={"marker": marker, "domain_label": labels},
            )
            return float(out["g_priv_domain"])

        all_ones = domain_loss([1.0, 1.0, 1.0, 1.0])
        mixed = domain_loss([0.0, 1.0, 0.0, 1.0])
        assert all_ones != pytest.approx(mixed), (
            "the domain loss ignores the labels -- the substitution survives"
        )

    def test_a_scalar_label_is_broadcast_but_only_when_explicit(self) -> None:
        """A single-domain run is legitimate -- as a DECLARATION, not a default."""
        losses = _call(_strategy(), {"domain_label": 1.0})
        assert "g_priv_domain" in losses

    def test_a_length_mismatch_raises_rather_than_broadcasting(self) -> None:
        """Silent broadcast would mislabel the batch, the same failure again."""
        with pytest.raises(ValueError, match="per-SAMPLE"):
            _call(_strategy(), {"domain_label": [0.0, 1.0, 0.0]}, batch_size=2)


class TestMixedBatchFromTheBalancer:
    """Reachability: ``build_balancer('domain_labelled')`` -> strategy, delta > 0."""

    @staticmethod
    def _loader(n: int, target_value: float) -> DataLoader:
        class _Domain(Dataset):
            def __len__(self) -> int:
                return n

            def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
                g = torch.Generator().manual_seed(i)
                return {
                    "input": torch.randn(1, 8, 8, generator=g),
                    "target": torch.full((1, 8, 8), target_value),
                    "marker": torch.randn(1, 8, 8, generator=g),
                }

        return DataLoader(_Domain(), batch_size=2)

    def _losses(self, real_target: float) -> dict[str, torch.Tensor]:
        torch.manual_seed(0)
        s = _strategy(delta=0.1)
        balancer = build_balancer(
            {"synthetic": self._loader(4, 0.0), "real": self._loader(4, real_target)},
            strategy="domain_labelled",
        )
        batch = next(iter(balancer))
        return s._compute_losses_impl(
            input_batch=batch["input"], target_batch=batch["target"], epoch=0, batch=batch
        )

    def test_the_domain_term_fires_on_a_balancer_batch(self) -> None:
        losses = self._losses(real_target=0.0)
        assert torch.isfinite(losses["g_priv_domain"])

    def test_real_targets_do_not_supervise_the_student(self) -> None:
        """Spec §4.3: L_S is defined on the synthetic twin only."""
        clean, planted = self._losses(0.0), self._losses(1e3)
        assert float(planted["g_priv_recon"]) == pytest.approx(float(clean["g_priv_recon"]))
        assert float(planted["g_priv_domain"]) == pytest.approx(float(clean["g_priv_domain"]))


def _settings(privileged: dict):
    from spectramr.config.settings import TrainingSettings

    return TrainingSettings(
        data={"train_path": "/tmp/t", "val_path": "/tmp/v", "batch_size": 2},
        model={"model_type": "unet", "in_channels": 1, "out_channels": 1},
        optimization={"learning_rate": 1e-2},
        logging={},
        training={"training_mode": "privileged_learning", "privileged": privileged},
    )


def _build(privileged: dict) -> tuple[PrivilegedLearningStrategy, nn.Module]:
    """The strategy as production resolves it: training_mode -> factory -> real __init__."""
    import tempfile

    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
    from tests.utils.logging_service import make_logging_service
    from tests.utils.mock_environment import create_mock_training_env

    student = _TinyStudent()
    env = create_mock_training_env(
        config=_settings(privileged),
        device="cpu",
        generator=student,
        opt_g=torch.optim.SGD(student.parameters(), lr=0.1),
    )
    log = make_logging_service(tempfile.gettempdir())
    return TrainingStrategyFactory().create_strategy(env, logging_service=log), student


class TestTeacherCheckpoint:
    """The student's teacher is the stage-1 checkpoint, never a copy of the untrained student."""

    def test_teacher_weights_come_from_the_checkpoint(self, tmp_path: Path) -> None:
        torch.manual_seed(1)
        trained = _TinyStudent()
        ckpt = write_teacher_checkpoint(tmp_path / "checkpoints" / "best.safetensors", trained)

        s, student = _build({"beta": 0.5, "delta": 0.0, "teacher_checkpoint": str(ckpt)})

        assert isinstance(s, PrivilegedLearningStrategy)
        for name, value in s._teacher.state_dict().items():
            assert torch.equal(value, trained.state_dict()[name]), name
            assert not torch.equal(value, student.state_dict()[name]), name
        assert all(not p.requires_grad for p in s._teacher.parameters())

        marker = torch.ones(1, 1, 8, 8)
        out = s._compute_losses_impl(
            input_batch=torch.zeros(1, 1, 8, 8),
            target_batch=torch.zeros(1, 1, 8, 8),
            epoch=0,
            batch={"marker": marker},
        )
        with torch.no_grad():
            expected = trained(torch.zeros(1, 1, 8, 8) + marker)
            student_pred = student(torch.zeros(1, 1, 8, 8))
        assert float(out["g_priv_distill"]) == pytest.approx(
            float((student_pred - expected).pow(2).mean())
        )

    def test_a_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _build({"beta": 0.5, "teacher_checkpoint": str(tmp_path / "absent.safetensors")})

    def test_distillation_without_a_checkpoint_raises(self) -> None:
        with pytest.raises(ValueError, match="teacher_checkpoint is unset"):
            _build({"beta": 0.5, "delta": 0.0})

    def test_a_checkpoint_with_beta_zero_raises(self, tmp_path: Path) -> None:
        ckpt = write_teacher_checkpoint(tmp_path / "best.safetensors", _TinyStudent())
        with pytest.raises(ValueError, match="never enter the loss"):
            _build({"beta": 0.0, "delta": 0.0, "teacher_checkpoint": str(ckpt)})

    def test_an_unknown_stage_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="stage"):
            _settings({"stage": "oracle"})

    def test_no_teacher_when_beta_is_zero(self) -> None:
        s, _ = _build({"beta": 0.0, "delta": 0.0})
        assert s._teacher is None


class TestTeacherStage:
    """Stage 1 trains f(input + marker) and validates on the same input."""

    def test_the_teacher_stage_fits_input_plus_marker(self) -> None:
        s, student = _build({"stage": "teacher", "beta": 0.0, "delta": 0.0})
        assert s._teacher is None
        x, target, marker = torch.zeros(2, 1, 8, 8), torch.randn(2, 1, 8, 8), torch.ones(2, 1, 8, 8)

        out = s._compute_losses_impl(
            input_batch=x, target_batch=target, epoch=0, batch={"marker": marker}
        )

        with torch.no_grad():
            expected = (student(x + marker) - target).pow(2).mean()
        assert float(out["g_total_loss"]) == pytest.approx(float(expected))
        assert "g_priv_distill" not in out

    @pytest.mark.parametrize(
        "bad",
        [{"beta": 0.5}, {"delta": 0.1}, {"curriculum_tau0": 3.0}, {"teacher_checkpoint": "x"}],
    )
    def test_knobs_inert_in_the_teacher_stage_raise(self, bad: dict) -> None:
        with pytest.raises(ValueError, match="stage=teacher"):
            _build({"stage": "teacher", "beta": 0.0, "delta": 0.0, **bad})

    def test_teacher_validation_adds_the_marker(self, monkeypatch) -> None:
        s, _ = _build({"stage": "teacher", "beta": 0.0, "delta": 0.0})
        seen = {}

        def _score(self, input_batch, target_batch, **kwargs):
            seen["input"] = input_batch
            return {}

        from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin

        monkeypatch.setattr(MetricsMixin, "validation_step", _score)
        x, marker = torch.zeros(1, 1, 8, 8), torch.ones(1, 1, 8, 8)
        s.validation_step(x, x, batch_data={"marker": marker})
        assert torch.equal(seen["input"], x + marker)

        with pytest.raises(ValueError, match="no 'marker'"):
            s.validation_step(x, x, batch_data={})
