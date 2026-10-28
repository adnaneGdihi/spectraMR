"""Tests for ReconstructionTrainingStrategy's loss-folding seam.

Scoped deliberately: the strategy's forward/backward path is exercised end-to-end by
the Tier-2 audit probe and by the per-paradigm strategy tests. What is pinned here is
the ``inline_losses`` declaration — the Open-Closed hook a subclass uses to declare
which registered losses it computes inline, so that declaring those losses on
``losses.image_losses`` (the only surface ``LossScheduleController`` can resolve a
curriculum rule's base weight from) does not silently double-count them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.training.strategies.loss_folding import (
    _INLINE_MANAGED,
    declared_inline_losses,
    inline_managed_with,
)
from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)


def test_base_strategy_owns_nothing_inline_and_folds_everything() -> None:
    """The base recon strategy computes every declared entry through the builder."""
    assert ReconstructionTrainingStrategy.inline_losses == frozenset()
    assert ReconstructionTrainingStrategy.folds_image_losses is True


def test_an_undeclared_subclass_keeps_the_universal_skip_set() -> None:
    """The base's empty declaration must not reach a subclass that has not declared:
    that subclass's inline l1 would be folded a second time (the fold passes
    ``below=ReconstructionTrainingStrategy``)."""

    class _Undeclared(ReconstructionTrainingStrategy):
        pass

    assert declared_inline_losses(_Undeclared, below=ReconstructionTrainingStrategy) is None
    assert inline_managed_with() == _INLINE_MANAGED == frozenset({"l1", "l2"})
    assert declared_inline_losses(_Undeclared) == frozenset()  # the witness's view


def test_a_declared_subclass_replaces_the_skip_set_with_its_own_terms() -> None:
    """Bloch synthesis computes l1 and segmentation_dice inline and declares exactly those."""
    from spectramr.infrastructure.training.strategies.bloch_synth_strategy import (
        BlochSynthesisStrategy,
    )

    assert issubclass(BlochSynthesisStrategy, ReconstructionTrainingStrategy)
    declared = declared_inline_losses(BlochSynthesisStrategy, below=ReconstructionTrainingStrategy)
    assert declared == frozenset({"l1", "segmentation_dice"})
    assert BlochSynthesisStrategy.folds_image_losses is True


def test_recoverability_vib_owns_its_l1_and_folds_nothing() -> None:
    """A subclass whose ``_compute_losses_impl`` neither calls the parent nor the fold."""
    from spectramr.infrastructure.training.strategies.recoverability_vib_strategy import (
        RecoverabilityVIBStrategy,
    )

    assert RecoverabilityVIBStrategy.inline_losses == frozenset({"l1"})
    assert RecoverabilityVIBStrategy.folds_image_losses is False


def test_the_inline_l1_weight_comes_from_the_loss_weight_table() -> None:
    """One owner (mrixfields review 2026-09-03): ``losses.image_losses[l1].weight`` through
    the table; an arm without an l1 entry cannot weight a term it never declared."""
    from types import SimpleNamespace

    import pytest

    s = ReconstructionTrainingStrategy.__new__(ReconstructionTrainingStrategy)
    s.config = SimpleNamespace(
        losses=SimpleNamespace(
            image_losses=[{"name": "l1", "weight": 0.25}], kspace_losses=[], complex_losses=[]
        )
    )
    assert s._declared_inline_l1_weight() == 0.25
    s.config = SimpleNamespace(
        losses=SimpleNamespace(
            image_losses=[{"name": "hfen", "weight": 1.0}], kspace_losses=[], complex_losses=[]
        )
    )
    with pytest.raises(ValueError, match="losses.image_losses"):
        s._declared_inline_l1_weight()


def test_multislice_flag_is_read_without_a_hasattr_fallback() -> None:
    """`multislice_enabled` was undeclared, so the `hasattr` guard was always
    False and `is_multislice` could never be True — the flag both docstrings
    advertise as replacing shape heuristics never fired (pitfall #16)."""
    import inspect

    from spectramr.config.schemas.data import DataConfigSchema
    from spectramr.infrastructure.training.strategies import reconstruction

    assert DataConfigSchema().multislice_enabled is False
    assert DataConfigSchema(multislice_enabled=True).multislice_enabled is True
    src = inspect.getsource(reconstruction)
    assert 'hasattr(config.data, "multislice_enabled")' not in src
    assert "multislice_enabled = config.data.multislice_enabled" in src


class TestCoilMapsReachTheLosses:
    """The maps reached the MODEL's forward and stopped there.

    ``_prepare_generator_inputs`` passes ``batch_context['coil_sensitivities']``
    to the model as ``sensitivity_maps``, but the declarative loss terms are
    invoked through ``UnifiedReconstructionLossComputer.compute``, whose
    ``_call_safe_loss`` filters kwargs by signature -- so a term needing the
    coil geometry was called as ``(pred, target)`` and raised. The maps are a
    loss kwarg too now.
    """

    def test_the_strategy_threads_the_maps_into_the_loss_computer(self) -> None:
        """Source-level, because constructing the strategy needs a built model."""
        import inspect

        from spectramr.infrastructure.training.strategies.reconstruction import (
            ReconstructionTrainingStrategy,
        )

        source = inspect.getsource(ReconstructionTrainingStrategy._compute_losses_impl)
        assert 'batch_context["coil_sensitivities"]' in source, (
            "the strategy no longer threads the coil maps into the loss computer, so "
            "a term like coil_subspace_residual is invoked as (pred, target) and raises"
        )
        assert "**loss_context" in source, "the threaded kwargs no longer reach compute()"

    def test_it_is_conditional_so_armless_runs_are_untouched(self) -> None:
        """An arm with no maps must not gain a `coil_sensitivities=None` kwarg."""
        import inspect

        from spectramr.infrastructure.training.strategies.reconstruction import (
            ReconstructionTrainingStrategy,
        )

        source = inspect.getsource(ReconstructionTrainingStrategy._compute_losses_impl)
        assert 'if batch_context.get("coil_sensitivities") is not None:' in source


class TestTheCoNEXLossMaskReachesTheLoss:
    """``loss_mask`` travels subject -> collate -> TrainingBatch -> loss context.

    Driven through the real transform, collation, ``BatchAdapter`` and
    ``_compute_losses_impl`` on the committed CoNEX arm's config, with a spy
    model and a spy loss, because the defect this guards is a key that every
    stage accepts and none forwards: the loss would then raise at the first step
    (``masked_kspace_l2`` refuses to score without the mask).
    """

    _ARM = "experiments/inprogress/self_supervised/conex_r2_independent_m4raw.yaml"

    @staticmethod
    def _batch(with_mask: bool = True):
        import torch
        import torchio as tio

        from spectramr.data.batch_types import BatchAdapter
        from spectramr.data.collation.strategies import ImageCollateStrategy
        from spectramr.data.transforms.complementary_rep_masking import (
            ComplementaryRepMasking,
        )

        subjects = []
        for i in range(2):
            data = torch.randn(8, 16, 32, 1, generator=torch.Generator().manual_seed(i))
            data[:, :, :4] = 0.0
            subject = tio.Subject(
                input=tio.ScalarImage(tensor=data.clone()),
                target=tio.ScalarImage(tensor=data + 0.1),
            )
            subject["file_id"] = f"scan{i}"
            subjects.append(ComplementaryRepMasking(2.0, 0.125)(subject))
        collated = ImageCollateStrategy(squeeze_depth_dim=True).collate(subjects)
        if not with_mask:
            collated.pop("loss_mask")
        return collated, BatchAdapter.from_dict(collated)

    def _strategy(self, loss):
        from unittest.mock import MagicMock

        import torch

        from spectramr.config.settings import TrainingSettings
        from tests.utils.mock_environment import create_mock_training_env

        class _Spy(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gain = torch.nn.Parameter(torch.ones(()))

            def forward(self, x, **_):
                return x * self.gain

        cfg = TrainingSettings.from_yaml(self._ARM)
        env = create_mock_training_env(config=cfg, device="cpu", generator=_Spy())
        env.losses = {"masked_kspace_l2": loss}
        return ReconstructionTrainingStrategy(
            env=env,
            logging_service=MagicMock(),
            metrics_service=MagicMock(),
            checkpoint_service=MagicMock(),
        )

    @staticmethod
    def _spy_loss():
        from spectramr.models.losses.masked_kspace_l2_loss import MaskedKSpaceL2Loss

        class _SpyLoss(MaskedKSpaceL2Loss):
            def __init__(self) -> None:
                super().__init__()
                self.seen: list = []

            def forward(self, pred, target, loss_mask=None):
                self.seen.append(loss_mask)
                return super().forward(pred, target, loss_mask=loss_mask)

        return _SpyLoss()

    def test_the_mask_the_transform_drew_is_the_one_the_loss_receives(self) -> None:
        import torch

        collated, batch = self._batch()
        loss = self._spy_loss()
        out = self._strategy(loss)._compute_losses_impl(
            batch.input, batch.target, 0, batch=batch, iteration=1
        )
        assert len(loss.seen) == 1
        assert torch.equal(loss.seen[0], collated["loss_mask"])
        assert loss.seen[0].shape == (2, 1, 16, 32)
        # The CoNEX objective is the masked term and nothing else: an undeclared
        # l1 resolves to its schema default 10.0 over the WHOLE residual unless the
        # arm pins it off, which would make every mask the full-target bound.
        assert set(out) & {"l1", "l2"} == set()
        assert torch.allclose(out["g_total_loss"], out["masked_kspace_l2"])

    def test_a_batch_without_the_mask_fails_loudly(self) -> None:
        """Planted violation: drop the key and the step must raise, not score all lines."""
        import pytest

        _, batch = self._batch(with_mask=False)
        strategy = self._strategy(self._spy_loss())
        with pytest.raises(ValueError, match="loss_mask"):
            strategy._compute_losses_impl(batch.input, batch.target, 0, batch=batch, iteration=1)

    def test_val_loss_is_the_masked_objective_not_a_whole_kspace_l1(self) -> None:
        """val_loss was an L1 over every line, flat at ~10x the masked training loss."""
        collated, batch = self._batch()
        loss = self._spy_loss()
        strategy = self._strategy(loss)
        pred = batch.input * 0.5
        val_loss = strategy._validation_loss(pred, batch.target, {}, batch)
        assert torch.equal(loss.seen[-1], collated["loss_mask"])
        expected = loss(pred, batch.target, loss_mask=collated["loss_mask"])
        assert torch.allclose(val_loss, expected)

    def test_val_loss_defers_to_the_mixin_without_a_loss_mask(self) -> None:
        _, batch = self._batch(with_mask=False)
        strategy = self._strategy(self._spy_loss())
        assert strategy._validation_loss(batch.input, batch.target, {}, batch) is None


class TestArmFValidationReadsTheCoilManifold:
    """Arm F's validation, end to end: ESPIRiT on the input, collate, adapter, strategy.

    An identity spy model on a synthetic 4-coil scene. Arm F scores the output
    projected onto the coil manifold, so the estimator graded is the projected
    identity, whose NEX-equivalence is about ``C`` (one coil's noise survives
    the projection), not the identity's 1 (#2616). Its true value is computed
    here from the noise-free scene. Every stage is the production one except
    the model.
    """

    _ARM = "experiments/inprogress/n2n/n2n_f_projection_m4raw.yaml"
    _C, _S = 4, 64

    _NOISE = 0.05  # per real/imaginary part, per coil pixel

    def _scene(self, seed: int):
        """``(maps, anatomy, gen)``; ``gen`` continues into the noise draws."""
        import torch

        s = self._S
        gen = torch.Generator().manual_seed(seed)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s), indexing="ij")
        anatomy = ((xx**2 + yy**2) < 0.6) * (1 + 0.3 * torch.randn(s, s, generator=gen))
        centres = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
        maps = torch.stack([torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) for a, b in centres])
        maps = (maps / maps.pow(2).sum(0, keepdim=True).sqrt()).to(torch.complex64)
        return maps, anatomy, gen

    def _subject(self, seed: int):
        import torch
        import torchio as tio

        from spectramr.data.transforms.espirit_sensitivity import ESPIRiTSensitivityTransform
        from spectramr.infrastructure.physics.fft_ops import fft2c

        c, s = self._C, self._S
        maps, anatomy, gen = self._scene(seed)

        def rep():
            noise = torch.complex(torch.randn(c, s, s, generator=gen), torch.randn(c, s, s, generator=gen))
            return fft2c(maps * anatomy + self._NOISE * noise)

        def image(k):
            out = torch.zeros(2 * c, s, s, 1)
            out[0::2, ..., 0], out[1::2, ..., 0] = k.real, k.imag
            return tio.ScalarImage(tensor=out)

        reps = [rep() for _ in range(4)]
        reference = (reps[2] + reps[3]) / 2
        subject = tio.Subject(input=image(reps[0]), target=image(reference),
                              kspace=image(reference), input_alt=image(reps[1]))
        return ESPIRiTSensitivityTransform(acs_size=24, keys=["input"])(subject)

    def test_every_declared_metric_fires_with_its_closed_form(self) -> None:
        from unittest.mock import MagicMock

        import torch

        from spectramr.config.settings import TrainingSettings
        from spectramr.data.batch_types import BatchAdapter
        from spectramr.data.collation.strategies import ImageCollateStrategy
        from tests.utils.mock_environment import create_mock_training_env

        class _Identity(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gain = torch.nn.Parameter(torch.ones(()))
                self.calls = 0

            def forward(self, x, **_):
                self.calls += 1
                return x * self.gain

        collated = ImageCollateStrategy(squeeze_depth_dim=True).collate(
            [self._subject(0), self._subject(1)]
        )
        batch = BatchAdapter.from_dict(collated)
        # The closed forms below are for the bare model; the arm's Monte-Carlo
        # estimator is driven separately in the next test.
        cfg = _without_recorrupted_inference(TrainingSettings.from_yaml(self._ARM))
        model = _Identity()
        env = create_mock_training_env(config=cfg, device="cpu", generator=model)
        strategy = ReconstructionTrainingStrategy(
            env=env, logging_service=MagicMock(), metrics_service=MagicMock(),
            checkpoint_service=MagicMock(),
        )
        with torch.no_grad():
            out = strategy.validation_step(batch.input, batch.target, batch_data=batch)

        for name in cfg.validation.scoring.compute:
            assert f"val_{name}" in out, f"val_{name} was declared and not reported"
        assert model.calls == 2  # the input, then input_alt for rep_variance / bias_squared
        # True N_eq of the projected identity: one excitation's noise energy over
        # its error to the noise-free scene, with the maps the batch carries.
        from spectramr.core.metrics.coil_manifold import coil_images
        from spectramr.infrastructure.physics.fft_ops import fft2c
        from spectramr.infrastructure.training.strategies.mixins.validation_context import (
            project_prediction,
        )

        projected, _ = project_prediction(batch.input, batch, "kspace")
        truth = torch.stack([fft2c(m * a) for m, a, _ in (self._scene(0), self._scene(1))])
        error = (coil_images(projected.squeeze(-1)) - coil_images(truth)).abs().square().sum()
        floor = 2 * self._C * self._S**2 * 2 * self._NOISE**2
        oracle = float(floor / error)
        assert oracle > 0.9 * self._C  # the projection, not the identity (which reads 1)
        # ESPIRiT maps estimated from one noisy repetition leak signal into the
        # null space (PAPER 3.4), which reads as reference noise and inflates
        # N_eq; 5.48 against 4.87 on this scene, and within 1 % with the true maps.
        assert abs(out["val_nex_equivalent"] / oracle - 1.0) < 0.2
        assert out["val_rep_variance"] > 0
        # The projected prediction is what PSNR grades; the unprojected input is not.
        assert abs(out["val_psnr"] - out["val_zf_psnr"]) > 0.1

    def test_the_arms_own_estimator_runs_j_forwards_per_input(self) -> None:
        """Arm F as declared: every metric fires on the mean over its J recorruptions."""
        from unittest.mock import MagicMock

        import torch

        from spectramr.config.settings import TrainingSettings
        from spectramr.data.batch_types import BatchAdapter
        from spectramr.data.collation.strategies import ImageCollateStrategy
        from tests.utils.mock_environment import create_mock_training_env

        class _Counting(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gain = torch.nn.Parameter(torch.ones(()))
                self.calls = 0

            def forward(self, x, **_):
                self.calls += 1
                return x * self.gain

        subjects = [self._subject(0), self._subject(1)]
        for subject in subjects:
            # What the normalization transform publishes on every m4raw subject.
            subject["kspace_scale"] = torch.tensor(80.0)
        batch = BatchAdapter.from_dict(ImageCollateStrategy(squeeze_depth_dim=True).collate(subjects))
        cfg = TrainingSettings.from_yaml(self._ARM)
        samples = cfg.validation.scoring.recorrupted_inference.samples
        model = _Counting()
        env = create_mock_training_env(config=cfg, device="cpu", generator=model)
        strategy = ReconstructionTrainingStrategy(
            env=env, logging_service=MagicMock(), metrics_service=MagicMock(),
            checkpoint_service=MagicMock(),
        )
        with torch.no_grad():
            out = strategy.validation_step(batch.input, batch.target, batch_data=batch)

        for name in cfg.validation.scoring.compute:
            assert f"val_{name}" in out, f"val_{name} was declared and not reported"
        assert model.calls == 2 * samples  # J on the input, J on input_alt


def _without_recorrupted_inference(cfg):
    """``cfg`` with ``validation.scoring.recorrupted_inference`` switched off."""
    scoring = cfg.validation.scoring.model_copy(update={"recorrupted_inference": None})
    validation = cfg.validation.model_copy(update={"scoring": scoring})
    return cfg.model_copy(update={"validation": validation})


class TestValidationBatchContext:
    """``validation_batch_context``: the one builder the metric pass and the image
    capture share (#2431)."""

    @staticmethod
    def _stub(enabled: bool) -> tuple[SimpleNamespace, list]:
        calls: list = []

        def prepare(input_batch, target_batch, **kwargs):
            calls.append(kwargs.get("batch"))
            return {"mask": torch.ones(2, 1, 4, 4), "use_dc": False}

        stub = SimpleNamespace(validates_on_batch_context=enabled, _prepare_batch_context=prepare)
        return stub, calls

    def test_the_base_strategy_keeps_the_bare_context(self) -> None:
        assert ReconstructionTrainingStrategy.validates_on_batch_context is False

    def test_off_returns_none_without_preparing_anything(self) -> None:
        stub, calls = self._stub(enabled=False)
        x = torch.zeros(2, 2, 4, 4)
        assert ReconstructionTrainingStrategy.validation_batch_context(stub, x, x, {"k": 1}) is None
        assert calls == []

    def test_on_without_a_batch_returns_none(self) -> None:
        stub, calls = self._stub(enabled=True)
        x = torch.zeros(2, 2, 4, 4)
        assert ReconstructionTrainingStrategy.validation_batch_context(stub, x, x, None) is None
        assert calls == []

    def test_on_prepares_from_the_batch_and_aligns_to_the_input(self) -> None:
        stub, calls = self._stub(enabled=True)
        x = torch.zeros(2, 2, 4, 4)
        batch = {"input": x}
        context = ReconstructionTrainingStrategy.validation_batch_context(stub, x, x, batch)
        assert calls == [batch]
        assert context["mask"].shape == (2, 1, 4, 4) and context["use_dc"] is False

    def test_a_misaligned_batch_field_raises(self) -> None:
        stub, _ = self._stub(enabled=True)
        x = torch.zeros(3, 2, 4, 4)
        with pytest.raises(ValueError, match="mask"):
            ReconstructionTrainingStrategy.validation_batch_context(stub, x, x, {"input": x})


# ---------------------------------------------------------------------------
# #2759: the generic forward is not a diffusion process. A backbone that names
# ``timesteps`` (complex_unet, nafnet, ...) used to receive torch.rand(B) in
# training and zeros at validation, and any ``**kwargs`` forward received the
# random draw too. Time-conditioned subclasses draw their own t.
# ---------------------------------------------------------------------------


class _TimeSpy(torch.nn.Module):
    """Records the ``timesteps`` each forward received."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[torch.Tensor | None] = []

    def forward(self, x, timesteps=None):
        self.seen.append(timesteps)
        return x


class _KwargsSpy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[dict] = []

    def forward(self, x, **kwargs):
        self.seen.append(dict(kwargs))
        return x


class _DDPLike(torch.nn.Module):
    """A training wrapper: ``**kwargs`` forward, the real model under ``.module``."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.module = inner

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _AliasLike(torch.nn.Module):
    """A model-level pass-through (``_BackboneAlias``): not a training wrapper."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = inner

    def forward(self, x, *args, **kwargs):
        return self.backbone(x, *args, **kwargs)


def _time_host(generator: torch.nn.Module) -> ReconstructionTrainingStrategy:
    host = ReconstructionTrainingStrategy.__new__(ReconstructionTrainingStrategy)
    config = SimpleNamespace(
        model=SimpleNamespace(model_type="complex_unet", input_type="image"),
        data=SimpleNamespace(multislice_enabled=False),
        training=SimpleNamespace(),
        validation=None,
    )
    host.env = SimpleNamespace(generator=generator, config=config)
    host.state = SimpleNamespace(config=config)
    host.config = config
    host.logging_service = None
    host.dc_layer = None
    return host


def _time_ctx() -> dict:
    return {"use_dc": False, "multimodal_inputs": None, "measured_kspace": None}


class TestNoDiffusionTimestepsOnTheGenericForward:
    def test_train_and_validation_condition_on_the_same_fixed_t(self) -> None:
        """Planted: the old code drew a fresh t per training call and 0 at validation."""
        spy = _TimeSpy()
        host = _time_host(spy)
        x = torch.randn(3, 2, 8, 8)
        for _ in range(2):  # the two forwards of one Rep-EI step
            _lr, kw = host._prepare_generator_inputs(_time_ctx(), x, validation=False)
            spy(x, **kw)
        host._validation_forward(x, _time_ctx())
        assert len(spy.seen) == 3
        for t in spy.seen:
            assert t is not None and torch.equal(t, torch.zeros(3))

    def test_a_kwargs_forward_sees_the_same_fixed_t_on_both_splits(self) -> None:
        """Planted: a ``**kwargs`` pass-through (``_BackboneAlias``) drew random t too."""
        spy = _KwargsSpy()
        host = _time_host(spy)
        x = torch.randn(2, 2, 8, 8)
        for _ in range(2):
            _lr, kw = host._prepare_generator_inputs(_time_ctx(), x, validation=False)
            spy(x, **kw)
        host._validation_forward(x, _time_ctx())
        for seen in spy.seen:
            assert torch.equal(seen["timesteps"], torch.zeros(2))

    @pytest.mark.parametrize("wrapper_cls", [_DDPLike, _AliasLike])
    def test_a_wrapped_backbone_keeps_its_validation_t(self, wrapper_cls: type) -> None:
        """Validation/predict must not change for a checkpoint behind a pass-through
        wrapper, whether or not ``unwrap_model`` peels it."""
        spy = _TimeSpy()
        host = _time_host(wrapper_cls(spy))
        x = torch.randn(2, 2, 8, 8)
        host._validation_forward(x, _time_ctx())
        assert torch.equal(spy.seen[-1], torch.zeros(2))

    def test_the_time_conditioned_subclasses_draw_their_own_t(self) -> None:
        """They own both forwards, so the generic seam never feeds their t."""
        from spectramr.infrastructure.training.strategies.doob_bridge_strategy import (
            DoobBridgeStrategy,
        )
        from spectramr.infrastructure.training.strategies.field_guided_diffusion_strategy import (
            FieldGuidedDiffusionStrategy,
        )
        from spectramr.infrastructure.training.strategies.generative_refiner_strategy import (
            GenerativeRefinerStrategy,
        )
        from spectramr.infrastructure.training.strategies.ulf_dps_strategy import (
            UlfDpsStrategy,
        )

        for cls in (
            DoobBridgeStrategy,
            FieldGuidedDiffusionStrategy,
            GenerativeRefinerStrategy,
            UlfDpsStrategy,
        ):
            for hook in ("_compute_losses_impl", "_validation_forward"):
                owner = getattr(cls, hook).__qualname__.split(".")[0]
                assert owner == cls.__name__, f"{cls.__name__}.{hook} is inherited"


def test_the_step_lives_on_a_baseless_mixin_so_composition_adds_no_depth() -> None:
    """NoiseToNoiseStrategy reuses the step by composing the same bases, not by
    subclassing, so no strategy's inheritance depth rises (non-negotiable 20)."""
    from spectramr.infrastructure.training.strategies.n2n_strategy import NoiseToNoiseStrategy
    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionStepMixin,
        ReconstructionTrainingStrategy,
    )

    assert ReconstructionStepMixin.__bases__ == (object,)
    assert "_compute_losses_impl" in ReconstructionStepMixin.__dict__
    assert ReconstructionTrainingStrategy not in NoiseToNoiseStrategy.__mro__
    assert NoiseToNoiseStrategy.__bases__[1:] == ReconstructionTrainingStrategy.__bases__[1:]



class TestTheNIGTensorReachesTheEvidentialLoss:
    """exp_09: the structural losses see gamma; the evidential loss sees all four channels."""

    _ARMS = "experiments/inprogress/10_paradigms/exp_09_{}.yaml"

    @staticmethod
    def _spy():
        import torch

        class _NIG(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(4))

            def forward(self, x, **_):
                f = torch.nn.functional.softplus
                s = self.scale
                return torch.cat(
                    [x * s[0], f(x * s[1]) + 1e-4, f(x * s[2]) + 1.0001, f(x * s[3]) + 1e-4], dim=1
                )

        return _NIG()

    def _build(self, arm, losses):
        from unittest.mock import MagicMock

        from spectramr.config.settings import TrainingSettings
        from tests.utils.mock_environment import create_mock_training_env

        cfg = TrainingSettings.from_yaml(self._ARMS.format(arm))
        env = create_mock_training_env(config=cfg, device="cpu", generator=self._spy())
        env.losses = losses
        return ReconstructionTrainingStrategy(
            env=env,
            logging_service=MagicMock(),
            metrics_service=MagicMock(),
            checkpoint_service=MagicMock(),
        )

    def _run(self, arm, losses):
        from unittest.mock import MagicMock

        import torch

        from spectramr.config.settings import TrainingSettings
        from spectramr.data.batch_types import BatchAdapter
        from tests.utils.mock_environment import create_mock_training_env

        model = self._spy()
        cfg = TrainingSettings.from_yaml(self._ARMS.format(arm))
        env = create_mock_training_env(config=cfg, device="cpu", generator=model)
        env.losses = losses
        strategy = ReconstructionTrainingStrategy(
            env=env,
            logging_service=MagicMock(),
            metrics_service=MagicMock(),
            checkpoint_service=MagicMock(),
        )
        g = torch.Generator().manual_seed(0)
        batch = BatchAdapter.from_dict(
            {"input": torch.rand(2, 1, 16, 16, generator=g), "target": torch.rand(2, 1, 16, 16, generator=g)}
        )
        out = strategy._compute_losses_impl(batch.input, batch.target, 0, batch=batch, iteration=1)
        out["g_total_loss"].backward()
        return out, model.scale.grad

    def test_the_evidential_loss_trains_all_four_channels(self) -> None:
        from spectramr.models.losses.evidential_loss import EvidentialLoss

        out, grad = self._run("evidential_learning", {"evidential": EvidentialLoss(coeff=0.01)})
        assert torch.all(grad.abs() > 0), grad
        assert torch.isfinite(out["nig_epistemic_var"]) and out["nig_epistemic_var"] > 0

    def test_the_epistemic_column_is_declared_for_the_csv(self) -> None:
        """The CSV header is built from declared keys; an undeclared stamp is dropped."""
        from spectramr.pipelines.training_loop import resolve_producer_declared_columns

        strategy = self._build("evidential_learning", {})
        assert "nig_epistemic_var" in resolve_producer_declared_columns(strategy)

    def test_mse_on_the_mean_leaves_the_evidence_untrained(self) -> None:
        """The control: only gamma's scale moves."""
        _, grad = self._run("control_mse", {"mse": torch.nn.MSELoss()})
        assert grad[0].abs() > 0
        assert torch.all(grad[1:] == 0), grad


class TestTheLossComputerComponentsReachTheCsv:
    """``loss_nonfinite_substitutions`` was discarded on every CoNEX step (header had no column)."""

    def test_the_base_step_forwards_its_computers_components(self) -> None:
        assert ReconstructionTrainingStrategy.forwards_loss_components is True

    def test_a_subclass_that_replaces_the_step_does_not_promise_them(self) -> None:
        from spectramr.infrastructure.training.strategies.equivariant_imaging_strategy import (
            EquivariantImagingStrategy,
        )
        from spectramr.infrastructure.training.strategies.ssdu_strategy import (
            SSDUReconstructionStrategy,
        )

        class _KeepsTheStep(ReconstructionTrainingStrategy):
            pass

        class _ReplacesTheStep(ReconstructionTrainingStrategy):
            def _compute_losses_impl(self, *a, **kw):
                return {}

        class _Declares(_ReplacesTheStep):
            forwards_loss_components = True

        assert _KeepsTheStep.forwards_loss_components is True
        assert _ReplacesTheStep.forwards_loss_components is False
        assert _Declares.forwards_loss_components is True
        assert EquivariantImagingStrategy.forwards_loss_components is False
        assert SSDUReconstructionStrategy.forwards_loss_components is False
