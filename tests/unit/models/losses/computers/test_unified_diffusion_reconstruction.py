"""Loss-weight resolution + single-application tests for the unified
diffusion loss computer.

These lock in the Experiment-11 DC-blob L1 fix:

1. The effective k-space fidelity weights must resolve from the *owning*
   schema sections — ``lambda_complex_l1`` / ``lambda_log_spectral`` /
   ``lambda_sobolev_kspace`` / ``lambda_sense_adjoint_l1`` from
   ``losses.reconstruction`` and ``lambda_complex_spatial_gradient`` from
   ``losses.physics``. Putting the physics term under ``reconstruction``
   is silently dropped (``extra='ignore'``) and regresses to 1.0 — the
   wrong-section test pins this so a future edit cannot re-break it.

2. Each *dynamic* loss component (from ``losses_dict``) must be weighted
   exactly ONCE. The pre-fix code pre-multiplied the component by its
   weight inside ``compute`` and then ``_stack_components`` re-applied the
   same weight (a ``w**2`` double-application). The single-application
   test reproduces that bug on a unit-weight loss.
"""

from types import SimpleNamespace

import pytest
import torch

from spectramr.config.schemas.loss_schedule import LossScheduleConfigSchema
from spectramr.config.schemas.loss import (
    LossConfigSchema,
    PhysicsLossesConfig,
    ReconstructionLossesConfig,
)
from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
    UnifiedDiffusionLossComputer,
    UnifiedReconstructionLossComputer,
)


def _exp11_config() -> SimpleNamespace:
    """The corrected exp11 loss layout: recon terms + the physics term."""
    losses = LossConfigSchema(
        output_domain="kspace",
        reconstruction=ReconstructionLossesConfig(
            lambda_complex_l1=1.0,
            lambda_log_spectral=0.1,
            lambda_sobolev_kspace=0.05,
            lambda_sense_adjoint_l1=0.3,
        ),
        physics=PhysicsLossesConfig(
            lambda_complex_spatial_gradient=0.2,
        ),
    )
    return SimpleNamespace(losses=losses)


def _wrong_section_config() -> SimpleNamespace:
    """complex_spatial_gradient placed (wrongly) under reconstruction.

    ``ReconstructionLossesConfig`` has no such field and (pydantic default)
    extra='ignore', so the key is silently dropped and ``_get_loss_weight``
    falls back to the PhysicsLossesConfig default of 1.0.
    """
    losses = LossConfigSchema(
        output_domain="kspace",
        reconstruction=ReconstructionLossesConfig(
            # this key does not exist on ReconstructionLossesConfig -> dropped
            lambda_complex_l1=1.0,
        ),
        # physics left at default (None) -> no override
    )
    return SimpleNamespace(losses=losses)


class TestLambdaResolution:
    """Effective weights must resolve from the section that OWNS each."""

    def test_exp11_lambdas_resolve_from_owning_sections(self):
        comp = UnifiedDiffusionLossComputer(
            config=_exp11_config(), device=torch.device("cpu")
        )
        assert comp._get_loss_weight("complex_l1", iteration=5000) == 1.0
        assert comp._get_loss_weight("log_spectral", iteration=5000) == 0.1
        assert comp._get_loss_weight("sobolev_kspace", iteration=5000) == 0.05
        assert comp._get_loss_weight("sense_adjoint_l1", iteration=5000) == 0.3
        # physics-section term — regression for skeptic #3: must NOT be 1.0
        assert comp._get_loss_weight("complex_spatial_gradient", iteration=5000) == 0.2

    def test_complex_spatial_gradient_under_reconstruction_is_ignored(self):
        """Wrong-section placement silently regresses to the 1.0 default."""
        comp = UnifiedDiffusionLossComputer(
            config=_wrong_section_config(), device=torch.device("cpu")
        )
        assert comp._get_loss_weight("complex_spatial_gradient", iteration=5000) == 1.0


class _ConstantLoss(torch.nn.Module):
    """A loss that always returns 1.0 (raw, unweighted)."""

    def forward(self, pred, target, **kwargs):
        return torch.tensor(1.0, requires_grad=True)


class TestSingleApplication:
    """Dynamic loss components must be weighted exactly once."""

    def test_dynamic_loss_weight_applied_exactly_once(self):
        config = SimpleNamespace(
            losses=LossConfigSchema(
                output_domain="kspace",
                reconstruction=ReconstructionLossesConfig(
                    lambda_log_spectral=0.1,
                ),
            )
        )
        comp = UnifiedDiffusionLossComputer(config=config, device=torch.device("cpu"))
        pred = torch.zeros(1, 2, 4, 4, requires_grad=True)
        target = torch.zeros(1, 2, 4, 4)

        out = comp.compute(
            pred,
            target,
            losses_dict={"log_spectral": _ConstantLoss()},
        )

        # The stored component is RAW (1.0), not pre-multiplied.
        assert out.components["log_spectral"].item() == pytest.approx(1.0)
        # The total applies the 0.1 weight exactly once (not 0.01 = w**2).
        # diffusion MSE on zeros contributes 0, so the total is the weighted
        # log_spectral component alone.
        assert out.total.item() == pytest.approx(0.1, abs=1e-6)


class TestScheduledWeightOverride:
    """The LossScheduleController seam: ``computer.scheduled_weights`` overrides
    a term's effective weight, UNCACHED, winning over both static config and the
    spatial-loss warmup gate (the third leg of the loss-curriculum feature)."""

    def test_override_wins_over_static_config(self):
        """Base-computer path (UnifiedDiffusionLossComputer has no own gate)."""
        comp = UnifiedDiffusionLossComputer(
            config=_exp11_config(), device=torch.device("cpu")
        )
        # static config says complex_l1 == 1.0
        assert comp._get_loss_weight("complex_l1", iteration=5000) == 1.0
        comp.scheduled_weights = {"complex_l1": 9.0}
        assert comp._get_loss_weight("complex_l1", iteration=5000) == 9.0

    def test_empty_override_preserves_static_config(self):
        comp = UnifiedDiffusionLossComputer(
            config=_exp11_config(), device=torch.device("cpu")
        )
        comp.scheduled_weights = {}  # no scheduling -> static behavior
        assert comp._get_loss_weight("log_spectral", iteration=5000) == 0.1

    def test_override_bypasses_warmup_gate(self):
        """Recon computer: a spatial term is gated to 0.0 during warmup, but an
        explicit schedule override must enable it early (uncached)."""
        config = SimpleNamespace(
            losses=LossConfigSchema(
                reconstruction=ReconstructionLossesConfig(
                    lambda_l1=1.0, warmup_iterations=1000
                ),
            )
        )
        comp = UnifiedReconstructionLossComputer(
            config=config, device=torch.device("cpu")
        )
        # default: l1 is a SPATIAL_LOSS, gated to 0.0 while iteration < warmup
        assert comp._get_loss_weight("l1", iteration=0) == 0.0
        # override enables it early, beating the warmup gate
        comp.scheduled_weights = {"l1": 0.7}
        assert comp._get_loss_weight("l1", iteration=0) == 0.7

    def test_override_can_disable_a_term(self):
        comp = UnifiedReconstructionLossComputer(
            config=SimpleNamespace(
                losses=LossConfigSchema(
                    reconstruction=ReconstructionLossesConfig(lambda_l1=1.0)
                )
            ),
            device=torch.device("cpu"),
        )
        # past warmup: l1 resolves to its config 1.0
        assert comp._get_loss_weight("l1", iteration=5000) == 1.0
        comp.scheduled_weights = {"l1": 0.0}  # disable
        assert comp._get_loss_weight("l1", iteration=5000) == 0.0

    def test_warmup_gate_intact_without_override(self):
        """Regression: with no override the warmup gate still fires (no facade)."""
        comp = UnifiedReconstructionLossComputer(
            config=SimpleNamespace(
                losses=LossConfigSchema(
                    reconstruction=ReconstructionLossesConfig(
                        lambda_l1=1.0, warmup_iterations=1000
                    )
                )
            ),
            device=torch.device("cpu"),
        )
        assert comp._get_loss_weight("l1", iteration=0) == 0.0  # gated
        assert comp._get_loss_weight("l1", iteration=5000) == 1.0  # released


class TestWeightResolutionIsSingleSourced:
    """``resolve_static_loss_weight`` and its 18-entry default table are GONE.

    They were one of eight resolvers. The property that replaces them: every consumer —
    the computer, the schedule controller, the fold seam — resolves a term to the SAME
    number, because they all read one table (``models/losses/weights.py``).
    """

    def test_the_legacy_resolver_and_its_default_table_are_deleted(self):
        import spectramr.models.losses.computers.unified_diffusion_reconstruction as mod

        assert not hasattr(mod, "resolve_static_loss_weight")
        assert not hasattr(mod, "_DEFAULT_LOSS_WEIGHTS")

    def test_computer_and_schedule_controller_resolve_identically(self):
        from spectramr.infrastructure.training.loss_schedule_controller import (
            LossScheduleController,
        )

        cfg = SimpleNamespace(
            losses=LossConfigSchema(
                reconstruction=ReconstructionLossesConfig(
                    lambda_l2=3.0, warmup_iterations=0
                )
            )
        )
        comp = UnifiedReconstructionLossComputer(config=cfg, device=torch.device("cpu"))
        ctrl = LossScheduleController(
            LossScheduleConfigSchema(enabled=False, rules=[]), loss_config=cfg
        )
        assert comp._get_loss_weight("l2", iteration=5000) == pytest.approx(3.0)
        assert ctrl._resolve_static_weight("l2") == pytest.approx(3.0)

    def test_an_alias_resolves_to_the_same_knob(self):
        """`mse` and `l2` are one loss; the legacy resolvers keyed on the raw string."""
        cfg = SimpleNamespace(
            losses=LossConfigSchema(
                reconstruction=ReconstructionLossesConfig(
                    lambda_l2=3.0, warmup_iterations=0
                )
            )
        )
        comp = UnifiedReconstructionLossComputer(config=cfg, device=torch.device("cpu"))
        assert comp._get_loss_weight("mse", iteration=5000) == pytest.approx(3.0)


class TestReconstructionFallbackDedup:
    """The auto-wired ``reconstruction`` fallback must not double-count a term
    the LossBuilder already supplies via ``losses_dict``.

    Regression for the 2026-07-11 exp_11 finding: ``_initialize_losses``
    auto-wires ``complex_l1`` into the ``reconstruction`` slot whenever
    ``lambda_complex_l1 > 0``, and with T=28 the ``timesteps < 50`` gate is
    always open — so ``complex_l1`` ran at an effective 2.0 (slot at the 1.0
    ``lambda_reconstruction`` default + dynamic component at its own lambda).
    """

    def _comp(self) -> UnifiedDiffusionLossComputer:
        return UnifiedDiffusionLossComputer(
            config=_exp11_config(), device=torch.device("cpu")
        )

    def test_fallback_skipped_when_losses_dict_carries_same_term(self):
        comp = self._comp()
        # lambda_complex_l1=1.0 auto-wired the fallback slot
        assert comp._recon_fallback_name == "complex_l1"
        pred = torch.zeros(1, 2, 4, 4, requires_grad=True)
        target = torch.zeros(1, 2, 4, 4)
        out = comp.compute(
            pred,
            target,
            losses_dict={"complex_l1": _ConstantLoss()},
            timesteps=torch.zeros(1, dtype=torch.long),  # < 50: recon gate open
        )
        assert "complex_l1" in out.components
        # the regression: the slot must NOT also compute the same term
        assert "reconstruction" not in out.components
        # total = 1.0 * lambda_complex_l1 (diffusion MSE on zeros contributes 0)
        assert out.total.item() == pytest.approx(1.0, abs=1e-6)

    def test_fallback_still_fires_without_losses_dict(self):
        comp = self._comp()
        pred = torch.zeros(1, 2, 4, 4, requires_grad=True)
        target = torch.zeros(1, 2, 4, 4)
        out = comp.compute(pred, target, timesteps=torch.zeros(1, dtype=torch.long))
        # no builder losses -> the fallback path is still the safety net
        assert "reconstruction" in out.components

    def test_injected_reconstruction_fn_is_not_deduped(self):
        cfg = SimpleNamespace(
            losses=LossConfigSchema(
                output_domain="kspace",
                reconstruction=ReconstructionLossesConfig(
                    lambda_complex_l1=0.0, lambda_l1=0.0
                ),
            )
        )
        comp = UnifiedDiffusionLossComputer(config=cfg, device=torch.device("cpu"))
        assert comp._recon_fallback_name is None
        comp.reconstruction_loss_fn = _ConstantLoss()  # external injection
        out = comp.compute(
            torch.zeros(1, 2, 4, 4, requires_grad=True),
            torch.zeros(1, 2, 4, 4),
            losses_dict={"complex_l1": _ConstantLoss()},
            timesteps=torch.zeros(1, dtype=torch.long),
        )
        # an externally-injected reconstruction fn is not the fallback: keep it
        assert "reconstruction" in out.components


class TestDiffusionWeightWiring:
    """``training.diffusion.lambda_mse`` must gate the main diffusion MSE.

    Regression for the exp_11 hidden-term finding: the term ran at the 1.0
    default on arms that declared ``lambda_mse: 0.0`` (pitfall #15 —
    advertised-but-unread knob).
    """

    def test_training_lambda_mse_zero_disables_diffusion_term(self):
        cfg = SimpleNamespace(
            losses=LossConfigSchema(
                output_domain="kspace",
                reconstruction=ReconstructionLossesConfig(lambda_complex_l1=1.0),
            ),
            training=SimpleNamespace(diffusion=SimpleNamespace(lambda_mse=0.0)),
        )
        comp = UnifiedDiffusionLossComputer(config=cfg, device=torch.device("cpu"))
        assert comp._get_loss_weight("diffusion") == 0.0
        out = comp.compute(
            torch.ones(1, 2, 4, 4, requires_grad=True),
            torch.zeros(1, 2, 4, 4),
            losses_dict={"complex_l1": _ConstantLoss()},
        )
        assert "diffusion" not in out.components

    def test_unset_knob_keeps_legacy_default(self):
        comp = UnifiedDiffusionLossComputer(
            config=_exp11_config(), device=torch.device("cpu")
        )
        # no training block, no lambda_diffusion anywhere -> historical 1.0
        assert comp._get_loss_weight("diffusion") == 1.0

    def test_losses_diffusion_lambda_mse_beats_training_knob(self):
        from spectramr.config.schemas.loss import DiffusionLossesConfig

        cfg = SimpleNamespace(
            losses=LossConfigSchema(
                output_domain="kspace",
                diffusion=DiffusionLossesConfig(lambda_mse=0.5),
            ),
            training=SimpleNamespace(diffusion=SimpleNamespace(lambda_mse=0.0)),
        )
        comp = UnifiedDiffusionLossComputer(config=cfg, device=torch.device("cpu"))
        assert comp._get_loss_weight("diffusion") == pytest.approx(0.5)

    def test_explicit_lambda_diffusion_wins_over_lambda_mse(self):
        cfg = SimpleNamespace(
            losses=SimpleNamespace(
                reconstruction=SimpleNamespace(lambda_diffusion=0.25)
            ),
            training=SimpleNamespace(diffusion=SimpleNamespace(lambda_mse=0.0)),
        )
        comp = UnifiedDiffusionLossComputer(config=cfg, device=torch.device("cpu"))
        assert comp._get_loss_weight("diffusion") == pytest.approx(0.25)

    def test_scheduled_override_wins(self):
        comp = UnifiedDiffusionLossComputer(
            config=_exp11_config(), device=torch.device("cpu")
        )
        comp.scheduled_weights = {"diffusion": 0.7}
        assert comp._get_loss_weight("diffusion") == pytest.approx(0.7)


def test_empty_components_error_names_the_schedule_when_a_term_was_zeroed():
    """L4: disabling the sole active term raises an error that names the schedule,
    not a misleading shape/config message."""
    comp = UnifiedReconstructionLossComputer(
        config=SimpleNamespace(
            losses=LossConfigSchema(
                reconstruction=ReconstructionLossesConfig(
                    lambda_l1=1.0, warmup_iterations=0
                )
            )
        ),
        device=torch.device("cpu"),
    )
    comp.scheduled_weights = {"l1": 0.0}  # schedule disables the only loss
    pred = torch.zeros(1, 1, 4, 4, requires_grad=True)
    target = torch.zeros(1, 1, 4, 4)
    with pytest.raises(RuntimeError, match="loss_schedule"):
        comp.compute(pred=pred, target=target, iteration=5000)


# ── no device->host sync on a discarded log message (non-negotiable 9) ───────
def test_the_debug_log_does_not_sync_when_debug_is_off():
    """An f-string evaluates ``.item()`` before the logger decides to discard it.

    The message was built eagerly for every dynamic loss component on every
    training step, so each one paid a device->host synchronisation at any log
    level. The sync is now behind ``isEnabledFor``, which is what makes it cost
    nothing when DEBUG is off.
    """
    import inspect

    from spectramr.models.losses.computers import unified_diffusion_reconstruction as mod

    src = inspect.getsource(mod.UnifiedDiffusionLossComputer.compute)
    assert "logger.isEnabledFor(logging.DEBUG)" in src, "the sync is unguarded again"
    assert 'f"[Loss {loss_name}] OK: loss_val={loss_val.item()' not in src, (
        "the eager f-string is back; it syncs before the logger discards it"
    )


def test_the_guarded_log_still_reports_when_debug_is_on():
    """Guards the check above from passing because the log was simply deleted."""
    import inspect

    from spectramr.models.losses.computers import unified_diffusion_reconstruction as mod

    src = inspect.getsource(mod.UnifiedDiffusionLossComputer.compute)
    assert '"[Loss %s] OK: loss_val=%s"' in src, "the debug message was dropped, not deferred"
