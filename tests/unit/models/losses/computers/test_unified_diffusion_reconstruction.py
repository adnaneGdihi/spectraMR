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
    _call_safe_loss,
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


_KSPACE_FILLING_ARM = (
    "experiments/inprogress/kspace_filling/attention_shootout/experiment_11_attention_none.yaml"
)


@pytest.fixture(scope="module")
def built_coil_barrier():
    """The ``coil_subspace_residual`` module the builder registers for a shipped arm."""
    from pathlib import Path

    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.training.builders.loss_builder import LossBuilder

    if not Path(_KSPACE_FILLING_ARM).exists():
        pytest.skip("kspace_filling cohort not present")
    cfg = TrainingSettings.from_yaml(_KSPACE_FILLING_ARM)
    losses = LossBuilder(cfg, torch.device("cpu")).build_reconstruction_losses().build()
    return losses["coil_subspace_residual"]


class TestTheBuiltBarrierReceivesTheStrategysMaps:
    """``_call_safe_loss`` on what the builder actually registers, not the bare loss.

    The earlier reachability test called ``_call_safe_loss`` on a bare
    ``CoilSubspaceResidualLoss`` and passed, while production hands it the
    builder's ``_BridgedLoss``: a ``**kwargs`` wrapper whose inner bridge did its
    own filtering without the coil-map reconciliation, so every kspace_filling arm
    raised ``requires coil_sensitivities`` at iteration 1.
    """

    @staticmethod
    def _kspace(images: torch.Tensor) -> torch.Tensor:
        """k-space as the arm's data path serves it: ``log1p``-compressed (#2338)."""
        from spectramr.data.transforms.normalization import compress_kspace_log
        from spectramr.infrastructure.physics.fft_ops import fft2c

        k = compress_kspace_log(fft2c(images))
        return torch.stack([k.real, k.imag], dim=2).flatten(1, 2)

    def test_the_arm_serves_compressed_kspace(self) -> None:
        """The premise of ``_kspace``: if the arm stops log-scaling, re-aim it."""
        from spectramr.config.settings import TrainingSettings

        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        cfg = TrainingSettings.from_yaml(_KSPACE_FILLING_ARM)
        assert KSpaceNormalizationSpec.from_data_config(cfg.data).log_compressed is True

    @staticmethod
    def _phantom() -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        m = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
        maps = torch.randn(2, 4, 16, 16, dtype=torch.complex64)
        maps = maps / maps.abs().pow(2).sum(1, keepdim=True).sqrt()
        return maps * m, maps

    def test_the_arm_bridges_it(self, built_coil_barrier) -> None:
        """Pins the shape this class exists for; if it stops bridging, re-aim the test."""
        assert type(built_coil_barrier).__name__ == "_BridgedLoss"

    def test_a_realisable_image_costs_nothing(self, built_coil_barrier) -> None:
        images, maps = self._phantom()
        k = self._kspace(images)
        assert float(_call_safe_loss(built_coil_barrier, k, k, smaps=maps, mask=None)) < 1e-6

    def test_energy_off_the_manifold_is_penalised(self, built_coil_barrier) -> None:
        images, maps = self._phantom()
        images[:, 0] += 0.5
        k = self._kspace(images)
        assert float(_call_safe_loss(built_coil_barrier, k, k, smaps=maps, mask=None)) > 1e-2


# ---------------------------------------------------------------------------
# #2546: the non-finite substitution must not read the loss back to the host.
# ---------------------------------------------------------------------------

from spectramr.models.losses.computers.unified_diffusion_reconstruction import (  # noqa: E402
    _complex_safe_l1,
    _complex_safe_mse,
)

_SAFE_LOSSES = pytest.mark.parametrize("safe_loss", [_complex_safe_mse, _complex_safe_l1])


def _forbid_host_reads(monkeypatch) -> None:
    """Make any device->host scalar read raise, so a hidden sync fails the test."""

    def _raise(*_a, **_k):
        raise AssertionError("host read of a tensor value")

    for name in ("__bool__", "__float__", "item"):
        monkeypatch.setattr(torch.Tensor, name, _raise)


@_SAFE_LOSSES
@pytest.mark.parametrize(
    ("poison", "expected"),
    [(float("nan"), 0.0), (float("inf"), 1e4)],
)
def test_a_non_finite_loss_keeps_its_old_substitute(safe_loss, poison, expected) -> None:
    pred = torch.zeros(1, 1, 4, 4)
    pred[0, 0, 0, 0] = poison
    loss = safe_loss(pred, torch.zeros_like(pred))
    assert loss.item() == expected


@_SAFE_LOSSES
def test_a_finite_loss_is_unchanged_and_keeps_its_gradient(safe_loss) -> None:
    pred = torch.randn(2, 1, 4, 4, requires_grad=True)
    target = torch.randn(2, 1, 4, 4)
    loss = safe_loss(pred, target)
    reference = (
        torch.nn.functional.mse_loss(pred, target)
        if safe_loss is _complex_safe_mse
        else torch.nn.functional.l1_loss(pred, target)
    )
    assert torch.equal(loss, reference)
    (grad,) = torch.autograd.grad(loss, pred)
    (ref_grad,) = torch.autograd.grad(reference, pred)
    assert torch.equal(grad, ref_grad)


@_SAFE_LOSSES
def test_a_nan_loss_still_gives_a_zero_gradient(safe_loss) -> None:
    pred = torch.zeros(1, 1, 4, 4)
    pred[0, 0, 0, 0] = float("nan")
    pred.requires_grad_(True)
    (grad,) = torch.autograd.grad(safe_loss(pred, torch.zeros(1, 1, 4, 4)), pred)
    assert torch.equal(grad[0, 0, 1:], torch.zeros_like(grad[0, 0, 1:]))


@_SAFE_LOSSES
@pytest.mark.parametrize("poison", [None, float("nan")])
def test_the_loss_reads_nothing_back_to_the_host(safe_loss, poison, monkeypatch) -> None:
    pred = torch.randn(1, 1, 4, 4)
    if poison is not None:
        pred[0, 0, 0, 0] = poison
    target = torch.randn(1, 1, 4, 4)
    _forbid_host_reads(monkeypatch)
    safe_loss(pred, target)


def test_the_detector_catches_the_old_guard(monkeypatch) -> None:
    """Planted violation: the pre-#2546 ``if isnan or isinf`` guard trips it."""
    loss = torch.tensor(1.0)
    _forbid_host_reads(monkeypatch)
    with pytest.raises(AssertionError, match="host read"):
        if torch.isnan(loss) or torch.isinf(loss):
            pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sync debug mode needs CUDA")
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype")
@_SAFE_LOSSES
def test_no_synchronizing_call_on_cuda(safe_loss) -> None:
    pred = torch.randn(1, 1, 4, 4, device="cuda")
    target = torch.randn(1, 1, 4, 4, device="cuda")
    torch.cuda.set_sync_debug_mode("error")
    try:
        safe_loss(pred, target)
    finally:
        torch.cuda.set_sync_debug_mode("default")


# ---------------------------------------------------------------------------
# #2546: the substitution stays visible as a device-side run count.
# ---------------------------------------------------------------------------

from spectramr.models.losses.computers.unified_diffusion_reconstruction import (  # noqa: E402
    NONFINITE_SUBSTITUTIONS_KEY,
    NonFiniteSubstitutionCounter,
)


def _nan_pred() -> torch.Tensor:
    pred = torch.zeros(1, 1, 4, 4)
    pred[0, 0, 0, 0] = float("nan")
    return pred.requires_grad_(True)


def _assert_counts_one_substitution(safe_loss) -> None:
    counter = NonFiniteSubstitutionCounter()
    safe_loss(_nan_pred(), torch.zeros(1, 1, 4, 4), counter=counter)
    assert counter.total is not None and counter.total.item() == 1.0


@_SAFE_LOSSES
def test_a_nan_loss_increments_the_counter(safe_loss) -> None:
    _assert_counts_one_substitution(safe_loss)


@_SAFE_LOSSES
def test_finite_and_grad_free_losses_are_not_counted(safe_loss) -> None:
    counter = NonFiniteSubstitutionCounter()
    safe_loss(torch.randn(1, 1, 4, 4, requires_grad=True), torch.zeros(1, 1, 4, 4), counter=counter)
    safe_loss(_nan_pred().detach(), torch.zeros(1, 1, 4, 4), counter=counter)
    assert counter.total is not None and counter.total.item() == 0.0


@_SAFE_LOSSES
def test_the_counter_is_cumulative(safe_loss) -> None:
    counter = NonFiniteSubstitutionCounter()
    for _ in range(3):
        safe_loss(_nan_pred(), torch.zeros(1, 1, 4, 4), counter=counter)
    assert counter.total.item() == 3.0


@_SAFE_LOSSES
def test_the_detector_catches_a_dropped_counter(safe_loss, monkeypatch) -> None:
    """Planted violation: a counter that never observes leaves the run silent."""
    monkeypatch.setattr(NonFiniteSubstitutionCounter, "observe", lambda self, loss: None)
    with pytest.raises(AssertionError):
        _assert_counts_one_substitution(safe_loss)


def _l2_computer() -> UnifiedReconstructionLossComputer:
    cfg = SimpleNamespace(
        losses=LossConfigSchema(
            reconstruction=ReconstructionLossesConfig(
                lambda_l1=0.0, lambda_l2=1.0, warmup_iterations=0
            )
        )
    )
    return UnifiedReconstructionLossComputer(config=cfg, device=torch.device("cpu"))


def _assert_compute_reports_the_count(comp) -> None:
    out = comp.compute(pred=_nan_pred(), target=torch.zeros(1, 1, 4, 4), iteration=5000)
    assert out.components[NONFINITE_SUBSTITUTIONS_KEY].item() == 1.0
    # Reported after stacking: the count is logged, never summed into the loss.
    assert out.total.item() == 0.0


class TestTheComputerReportsTheCount:
    def test_a_nan_step_shows_up_in_the_components(self) -> None:
        _assert_compute_reports_the_count(_l2_computer())

    def test_a_clean_step_stamps_a_zero(self) -> None:
        """Stamped every call, so a declared CSV column is never left empty."""
        out = _l2_computer().compute(
            pred=torch.randn(1, 1, 4, 4, requires_grad=True),
            target=torch.randn(1, 1, 4, 4),
            iteration=5000,
        )
        assert out.components[NONFINITE_SUBSTITUTIONS_KEY].item() == 0.0

    def test_the_step_reads_nothing_back_to_the_host(self, monkeypatch) -> None:
        comp = _l2_computer()
        pred, target = _nan_pred(), torch.zeros(1, 1, 4, 4)
        _forbid_host_reads(monkeypatch)
        out = comp.compute(pred=pred, target=target, iteration=5000)
        monkeypatch.undo()
        assert out.components[NONFINITE_SUBSTITUTIONS_KEY].item() == 1.0

    def test_the_detector_catches_a_computer_that_drops_the_count(self, monkeypatch) -> None:
        """Planted violation: without ``report_into`` the count never reaches the log."""
        monkeypatch.setattr(
            NonFiniteSubstitutionCounter, "report_into", lambda self, components, like: None
        )
        with pytest.raises(KeyError):
            _assert_compute_reports_the_count(_l2_computer())

    def test_both_computers_declare_the_key(self) -> None:
        assert NONFINITE_SUBSTITUTIONS_KEY in UnifiedReconstructionLossComputer.stamped_component_keys
        assert NONFINITE_SUBSTITUTIONS_KEY in UnifiedDiffusionLossComputer.stamped_component_keys



def test_a_declared_loss_that_fails_is_never_dropped_in_silence():
    """A non-ValueError inside a list loss used to be logged and skipped, so a coil-map
    barrier that errored left its arm training as the control arm."""

    def broken(pred, target):
        raise RuntimeError("planted: coil maps missing")

    comp = UnifiedReconstructionLossComputer(
        config=SimpleNamespace(
            losses=LossConfigSchema(
                reconstruction=ReconstructionLossesConfig(lambda_l1=1.0, warmup_iterations=0)
            )
        ),
        device=torch.device("cpu"),
    )
    pred = torch.zeros(1, 1, 4, 4, requires_grad=True)
    target = torch.zeros(1, 1, 4, 4)
    with pytest.raises(RuntimeError, match="planted: coil maps missing"):
        comp.compute(pred=pred, target=target, iteration=5000, losses_dict={"barrier": broken})
