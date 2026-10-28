"""A6: the domain-adversarial term must not fabricate its own labels.

``privileged_learning_strategy.py`` read the per-sample domain label as
``batch.get("domain_label", 1)``. No dataset under ``spectramr.data`` emits that
key -- the A6 census found it among 60 batch keys with no data-layer producer --
so the default always won and every sample in every batch was labelled domain
``1``. The gradient-reversal discriminator was therefore trained to separate one
class from itself.

This is the census's single **substitutes** case, and it is worse than the 33
keys whose absence merely skips a term: a skipped term is absent, but here the
mechanism is present, running, and wrong. The BCE loss is well-defined and
finite, so it plots a plausible curve while the domain signal it exists to supply
is identically uninformative. ``training.privileged.delta`` defaults to ``0.1``,
so the branch was ON by default, and both corpus arms that select this strategy
were training the confounded objective.

See ``TODO/inprogress/a6_orphan_batch_key_triage_2026_08_06.md``.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.privileged_learning_strategy import (
    PrivilegedLearningStrategy,
)


class _TinyStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


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
    s._alpha, s._beta, s._delta = 1.0, 0.5, delta
    s._tau0, s._n_min, s._n_max = 5.0, 0.0, 1.0
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
        assert "no dataset under spectramr.data emits it" in message

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


class TestNoProducerEmitsTheKey:
    def test_the_data_layer_still_does_not_emit_domain_label(self) -> None:
        """Pins the premise. If a loader ever starts emitting the key, the
        raise above becomes reachable-but-wrong and this test says so first."""
        from pathlib import Path

        data_root = Path(__file__).resolve().parents[5] / "src" / "spectramr" / "data"
        assert data_root.is_dir(), f"data layer not at {data_root}"
        emitters = [
            path.relative_to(data_root)
            for path in data_root.rglob("*.py")
            if "domain_label" in path.read_text(encoding="utf-8", errors="ignore")
        ]
        assert not emitters, (
            f"a data-layer module now mentions domain_label ({emitters}); "
            "re-run scripts/ci/a6_batch_key_census.py and revisit the raise"
        )
