"""Tests for the shared generator-kwarg resolution SSOT.

These pin the behaviour that used to live as two copies in two builders and a
third hand-rolled copy in the audit probe. The contract-gating is the point:
an SSOT value is injected only when the constructor can accept it, and a
``model_kwargs`` entry that contradicts an SSOT block is a hard error rather
than a silent divergence (CLAUDE.md pitfall #9).
"""

from types import SimpleNamespace

import pydantic
import pytest

from spectramr.infrastructure.builders.generator_kwargs import (
    _SKIP_MODEL_FIELDS,
    ResolvedGeneratorKwargs,
    apply_gradient_checkpointing,
    apply_model_field_sweep,
    resolve_contract,
    resolve_full_generator_kwargs,
    resolve_generator_kwargs,
)


class _Explicit:
    """Names every SSOT kwarg explicitly."""

    def __init__(
        self,
        acceleration_config=None,
        kspace_log_scaled=None,
        use_dc=None,
        dc_method=None,
        dc_weight=None,
    ):
        pass


class _VarKwargsOnly:
    """Names nothing; consumes via ``**kwargs`` (the KSpaceColdDiffusion shape)."""

    def __init__(self, **kwargs):
        pass


class _Bare:
    """Accepts nothing beyond channels."""

    def __init__(self, in_channels=1, out_channels=1):
        pass


def _config(model_kwargs=None, **blocks):
    """A duck-typed config carrying only the blocks a test needs."""
    model = SimpleNamespace(model_type="stub", model_kwargs=dict(model_kwargs or {}))
    return SimpleNamespace(model=model, **blocks)


class TestResolveContract:
    def test_reads_explicit_parameters(self) -> None:
        contract = resolve_contract(model_cls=_Explicit)
        assert "acceleration_config" in contract.accepted
        assert not contract.accepts_var_kwargs

    def test_detects_var_kwargs(self) -> None:
        contract = resolve_contract(model_cls=_VarKwargsOnly)
        assert contract.accepts_var_kwargs
        assert "dc_method" not in contract.accepted

    def test_unresolvable_type_returns_empty_and_does_not_raise(self) -> None:
        """Tolerance is the contract: the audit probe must never crash here."""
        contract = resolve_contract(model_type="definitely_not_registered_xyzzy")
        assert contract.accepted == frozenset()
        assert contract.accepts_var_kwargs is False

    def test_no_identifiers_at_all_returns_empty(self) -> None:
        assert resolve_contract().accepted == frozenset()


class TestAccelerationConfig:
    def test_injected_when_constructor_names_it(self) -> None:
        sentinel = object()
        kwargs = resolve_generator_kwargs(_config(undersampling=sentinel), model_cls=_Explicit)
        assert kwargs["acceleration_config"] is sentinel

    def test_not_injected_when_constructor_cannot_accept_it(self) -> None:
        kwargs = resolve_generator_kwargs(_config(undersampling=object()), model_cls=_Bare)
        assert "acceleration_config" not in kwargs

    def test_var_kwargs_alone_does_not_trigger_injection(self) -> None:
        """Matches the pre-extraction rule: explicit naming only.

        Unlike the data-consistency keys below, acceleration is injected only
        for a constructor that names it. Widening this would hand a kwarg to
        every ``**kwargs`` generator in the registry.
        """
        kwargs = resolve_generator_kwargs(_config(undersampling=object()), model_cls=_VarKwargsOnly)
        assert "acceleration_config" not in kwargs

    def test_absent_block_injects_nothing(self) -> None:
        assert "acceleration_config" not in resolve_generator_kwargs(_config(), model_cls=_Explicit)


class _AccelAndKwargs:
    """``acceleration_config`` and ``phase_encode_axis`` named (the cold-diffusion shape)."""

    def __init__(self, acceleration_config=None, phase_encode_axis=None, **kwargs):
        pass


class TestPhaseEncodeAxis:
    """``data.acquisition.phase_encode_axis`` reaches the generator beside its block (#2160)."""

    @staticmethod
    def _cfg(direction="phase", axis=-1, model_kwargs=None):
        acquisition = None if axis is None else SimpleNamespace(phase_encode_axis=axis)
        return _config(
            model_kwargs,
            undersampling=SimpleNamespace(mask_direction=direction),
            data=SimpleNamespace(acquisition=acquisition, processing=None),
        )

    def test_injected_for_a_line_mask_arm(self) -> None:
        kwargs = resolve_generator_kwargs(self._cfg(), model_cls=_AccelAndKwargs)
        assert kwargs["phase_encode_axis"] == -1

    @pytest.mark.parametrize(
        ("cfg", "cls"),
        [({"axis": None}, _AccelAndKwargs), ({}, _Explicit), ({}, _VarKwargsOnly)],
        ids=["undeclared", "not_named", "var_kwargs_only"],
    )
    def test_not_injected_where_nothing_reads_it(self, cfg, cls) -> None:
        assert "phase_encode_axis" not in resolve_generator_kwargs(self._cfg(**cfg), model_cls=cls)

    def test_the_cold_diffusion_generator_names_it(self) -> None:
        """The production consumer, resolved without building it."""
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            KSpaceColdDiffusionGenerator,
        )

        kwargs = resolve_generator_kwargs(self._cfg(), model_cls=KSpaceColdDiffusionGenerator)
        assert kwargs["phase_encode_axis"] == -1

    def test_a_contradicting_model_kwargs_copy_raises(self) -> None:
        with pytest.raises(ValueError, match="phase_encode_axis"):
            resolve_generator_kwargs(
                self._cfg(model_kwargs={"phase_encode_axis": -2}), model_cls=_AccelAndKwargs
            )


class _NamesHorizon:
    """Names ``num_timesteps`` (the KSpaceColdDiffusion shape) and nothing else."""

    def __init__(self, num_timesteps=None, **kwargs):
        pass


class TestDiffusionHorizon:
    """``training.diffusion.timesteps`` is the horizon the strategy samples t from."""

    def _cfg(self, horizon, model_kwargs=None):
        diffusion = SimpleNamespace(timesteps=horizon, sampler=None)
        return _config(model_kwargs, training=SimpleNamespace(diffusion=diffusion))

    def test_injected_when_model_kwargs_omit_it(self) -> None:
        """PLANTED: experiment_40 declares only training.diffusion.timesteps=28 and
        resolved no horizon, so its generator fell back to 1000."""
        kwargs = resolve_generator_kwargs(self._cfg(28), model_cls=_NamesHorizon)
        assert kwargs["num_timesteps"] == 28

    @pytest.mark.parametrize("key", ["timesteps", "num_timesteps"])
    def test_an_agreeing_copy_reaches_the_named_horizon(self, key: str) -> None:
        """PLANTED (#2440): x_diffusion's ``timesteps`` copy fell into **kwargs and
        left its horizon unset."""
        kwargs = resolve_generator_kwargs(self._cfg(29, {key: 29}), model_cls=_NamesHorizon)
        assert kwargs["num_timesteps"] == 29
        assert "timesteps" not in kwargs

    @pytest.mark.parametrize("key", ["timesteps", "num_timesteps"])
    def test_a_disagreeing_copy_raises(self, key: str) -> None:
        with pytest.raises(ValueError, match=f"model_kwargs.{key}=1000 disagrees"):
            resolve_generator_kwargs(self._cfg(29, {key: 1000}), model_cls=_NamesHorizon)

    def test_not_injected_when_constructor_does_not_name_it(self) -> None:
        kwargs = resolve_generator_kwargs(self._cfg(28), model_cls=_VarKwargsOnly)
        assert "num_timesteps" not in kwargs


class _NamesSteps:
    """Names ``sampling_steps`` as KSpaceColdDiffusionGenerator does."""

    def __init__(self, sampling_steps=None, **kwargs):
        pass


class TestSamplingSteps:
    """``training.diffusion.sampling_steps`` is the reverse budget validation resolves (#566)."""

    def _cfg(self, steps, model_kwargs=None):
        diffusion = SimpleNamespace(timesteps=None, sampler=None, sampling_steps=steps)
        return _config(model_kwargs, training=SimpleNamespace(diffusion=diffusion))

    def test_injected_when_model_kwargs_omit_it(self) -> None:
        """PLANTED: experiment_11_attention_none declared 20 and the generator ran its literal 50."""
        assert resolve_generator_kwargs(self._cfg(20), model_cls=_NamesSteps)["sampling_steps"] == 20

    def test_a_disagreeing_copy_raises(self) -> None:
        with pytest.raises(ValueError, match=r"model_kwargs\.sampling_steps=50 disagrees"):
            resolve_generator_kwargs(self._cfg(20, {"sampling_steps": 50}), model_cls=_NamesSteps)

    def test_an_undeclared_block_value_injects_nothing(self) -> None:
        kwargs = resolve_generator_kwargs(self._cfg(None, {"sampling_steps": 7}), model_cls=_NamesSteps)
        assert kwargs["sampling_steps"] == 7

    def test_not_injected_when_constructor_does_not_name_it(self) -> None:
        kwargs = resolve_generator_kwargs(self._cfg(20), model_cls=_VarKwargsOnly)
        assert "sampling_steps" not in kwargs


class TestKspaceLogScaled:
    def _cfg(self, enable, model_kwargs=None, *, normalize=True):
        from spectramr.config.schemas.data import DataProcessingConfigSchema

        processing = DataProcessingConfigSchema(
            enable_kspace_normalization=normalize, enable_log_scaling=enable
        )
        return _config(model_kwargs, data=SimpleNamespace(processing=processing))

    def test_piped_from_the_data_block(self) -> None:
        kwargs = resolve_generator_kwargs(self._cfg(True), model_cls=_Explicit)
        assert kwargs["kspace_log_scaled"] is True

    def test_agreeing_declaration_is_accepted(self) -> None:
        kwargs = resolve_generator_kwargs(
            self._cfg(True, {"kspace_log_scaled": True}), model_cls=_Explicit
        )
        assert kwargs["kspace_log_scaled"] is True

    def test_contradicting_declaration_raises(self) -> None:
        """#1281: a declared reverse_clip_ratio of 1.3 realised 29.8x."""
        with pytest.raises(ValueError, match="same fact"):
            resolve_generator_kwargs(
                self._cfg(False, {"kspace_log_scaled": True}), model_cls=_Explicit
            )

    def test_log_scaling_without_normalization_pipes_false(self) -> None:
        """#2353: without the k-space normalization nothing is compressed, so the
        magnitude ceilings must bound physical, not log, units."""
        kwargs = resolve_generator_kwargs(self._cfg(True, normalize=False), model_cls=_Explicit)
        assert kwargs["kspace_log_scaled"] is False

    def test_declaration_trusting_the_bare_flag_raises(self) -> None:
        with pytest.raises(ValueError, match="same fact"):
            resolve_generator_kwargs(
                self._cfg(True, {"kspace_log_scaled": True}, normalize=False),
                model_cls=_Explicit,
            )

    def test_missing_data_block_leaves_kwarg_unset(self) -> None:
        """Never a silent default — the generator raises at point of use."""
        kwargs = resolve_generator_kwargs(_config(), model_cls=_Explicit)
        assert "kspace_log_scaled" not in kwargs

    def test_not_injected_when_constructor_cannot_accept_it(self) -> None:
        kwargs = resolve_generator_kwargs(self._cfg(True), model_cls=_Bare)
        assert "kspace_log_scaled" not in kwargs


class TestDataConsistencyReconciliation:
    def _cfg(self, model_kwargs=None, method="hard"):
        return _config(
            model_kwargs,
            physics=SimpleNamespace(
                data_consistency=SimpleNamespace(enabled=True, method=method, weight=0.5)
            ),
        )

    def test_injects_all_three_keys(self) -> None:
        kwargs = resolve_generator_kwargs(self._cfg(), model_cls=_Explicit)
        assert kwargs["use_dc"] is True
        assert kwargs["dc_method"] == "hard"
        assert kwargs["dc_weight"] == 0.5

    def test_var_kwargs_constructor_is_reconciled(self) -> None:
        """The ``**kwargs`` branch: without it, generators reading DC keys via
        ``kwargs.get(...)`` silently bypass the SSOT (pitfall #9)."""
        kwargs = resolve_generator_kwargs(self._cfg(), model_cls=_VarKwargsOnly)
        assert kwargs["dc_method"] == "hard"

    def test_conflicting_declaration_raises(self) -> None:
        """Silent divergence here disabled experiment-32a's adversarial training."""
        with pytest.raises(ValueError, match="Data-consistency configuration conflict"):
            resolve_generator_kwargs(self._cfg({"dc_method": "soft"}), model_cls=_Explicit)

    def test_bare_constructor_is_left_alone(self) -> None:
        kwargs = resolve_generator_kwargs(self._cfg(), model_cls=_Bare)
        assert "dc_method" not in kwargs


class TestDiffusionWrapperFields:
    def test_forwarded_when_present(self) -> None:
        cfg = _config()
        cfg.model.denoising_model = {"model_type": "unet"}
        cfg.model.base_diffusion_config = {"timesteps": 10}
        kwargs = resolve_generator_kwargs(cfg, model_cls=_VarKwargsOnly)
        assert kwargs["denoising_model"] == {"model_type": "unet"}
        assert kwargs["base_diffusion_config"] == {"timesteps": 10}

    def test_model_kwargs_entry_wins(self) -> None:
        cfg = _config({"denoising_model": {"model_type": "explicit"}})
        cfg.model.denoising_model = {"model_type": "top_level"}
        kwargs = resolve_generator_kwargs(cfg, model_cls=_VarKwargsOnly)
        assert kwargs["denoising_model"] == {"model_type": "explicit"}


class _ModelSchema(pydantic.BaseModel):
    model_type: str = "stub"
    model_kwargs: dict = {}
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    spatial_dims: int = 3
    target_domain: str = "image"
    checkpoint_path: str = "/warm/start.pt"


class TestModelFieldSweep:
    def _cfg(self):
        return SimpleNamespace(model=_ModelSchema())

    def test_forwards_top_level_fields(self) -> None:
        """#560/#878: `model.spatial_dims: 3` was dropped before reaching MONAI."""
        resolved = apply_model_field_sweep({}, self._cfg())
        assert resolved.kwargs["spatial_dims"] == 3
        assert resolved.kwargs["base_channels"] == 32

    def test_skips_the_skip_set(self) -> None:
        resolved = apply_model_field_sweep({}, self._cfg())
        for field in ("target_domain", "checkpoint_path", "model_type", "model_kwargs"):
            assert field in _SKIP_MODEL_FIELDS
            assert field not in resolved.kwargs

    def test_strips_explicit_channel_args(self) -> None:
        """The factory passes these positionally; leaving them duplicates them."""
        resolved = apply_model_field_sweep(
            {"in_channels": 7, "out_channels": 9, "depth": 4}, self._cfg()
        )
        assert "in_channels" not in resolved.kwargs
        assert "out_channels" not in resolved.kwargs
        assert resolved.kwargs["depth"] == 4

    def test_declared_keys_snapshot_excludes_swept_fields(self) -> None:
        """The factory needs the two apart — a swept field landing nowhere is
        expected, a declared one landing nowhere is the #560 failure."""
        resolved = apply_model_field_sweep({"depth": 4}, self._cfg())
        assert resolved.declared_keys == frozenset({"depth"})
        assert "spatial_dims" in resolved.kwargs

    def test_existing_value_is_not_overwritten(self) -> None:
        resolved = apply_model_field_sweep({"base_channels": 64}, self._cfg())
        assert resolved.kwargs["base_channels"] == 64

    def test_duck_typed_model_is_swept_over_its_attributes(self) -> None:
        """The probe passes stand-ins; forwarding must not depend on the type.

        ``spatial_dims`` reaching the constructor is the ULF stage-1 fix
        (#560/#878); ``input_type`` NOT reaching it is the UNetConfig kwarg
        leak. Both must hold for a duck-typed config too.
        """
        stub = SimpleNamespace(
            model=SimpleNamespace(
                model_type="stub",
                model_kwargs={},
                spatial_dims=3,
                input_type="kspace",
            )
        )
        resolved = apply_model_field_sweep({"depth": 4}, stub)
        assert resolved.kwargs["spatial_dims"] == 3
        assert resolved.kwargs["depth"] == 4
        assert "input_type" not in resolved.kwargs
        assert "model_type" not in resolved.kwargs
        assert resolved.declared_keys == frozenset({"depth"})

    def test_input_mapping_is_not_mutated(self) -> None:
        original = {"depth": 4}
        apply_model_field_sweep(original, self._cfg())
        assert original == {"depth": 4}


class TestFullComposition:
    def test_applies_both_halves_in_pipeline_order(self) -> None:
        class _Cfg(pydantic.BaseModel):
            model: _ModelSchema = _ModelSchema()
            undersampling: int = 4

        resolved = resolve_full_generator_kwargs(_Cfg(), model_cls=_Explicit)
        assert isinstance(resolved, ResolvedGeneratorKwargs)
        # half 1 (SSOT injection) ...
        assert resolved.kwargs["acceleration_config"] == 4
        # ... snapshot sits between them ...
        assert "acceleration_config" in resolved.declared_keys
        assert "spatial_dims" not in resolved.declared_keys
        # ... then half 2 (sweep).
        assert resolved.kwargs["spatial_dims"] == 3

    def test_model_type_override_selects_the_contract(self) -> None:
        """pipeline_strategy/diffusion build stage_cfg.model_type, not the
        arm's headline model; the contract must describe what is built."""
        kwargs = resolve_generator_kwargs(
            _config(undersampling=object()), model_type="definitely_not_registered"
        )
        assert "acceleration_config" not in kwargs


class _NativeCkpt:
    def __init__(self):
        self.called = None

    def set_grad_checkpointing(self, flag):
        self.called = flag


class TestApplyGradientCheckpointing:
    def _cfg(self, enabled):
        return SimpleNamespace(
            optimization=SimpleNamespace(gradient=SimpleNamespace(enable_checkpointing=enabled))
        )

    def test_uses_native_hook_when_available(self) -> None:
        model = _NativeCkpt()
        apply_gradient_checkpointing(model, self._cfg(True))
        assert model.called is True

    def test_noop_when_disabled(self) -> None:
        model = _NativeCkpt()
        apply_gradient_checkpointing(model, self._cfg(False))
        assert model.called is None

    def test_noop_when_config_lacks_the_block(self) -> None:
        model = _NativeCkpt()
        apply_gradient_checkpointing(model, SimpleNamespace())
        assert model.called is None

    def test_model_without_hook_raises(self) -> None:
        """#1333: the request used to lapse to a warning and train uncheckpointed."""
        with pytest.raises(NotImplementedError, match="set_grad_checkpointing"):
            apply_gradient_checkpointing(object(), self._cfg(True))

    def test_model_without_hook_is_fine_when_disabled(self) -> None:
        apply_gradient_checkpointing(object(), self._cfg(False))


class _DeviceAware:
    """Names ``device`` explicitly -- the ``kspace_cold_diffusion`` shape."""

    def __init__(self, device=None, **kwargs):
        pass


class TestDeviceInjection:
    """Step 3d: the run's device is inherited from the configuration (#1508).

    The generator's mask table has to be built on the device it will be served
    from, and the constructor cannot learn that from its own parameters (it has
    none yet) -- so it is injected here rather than sniffed from a tensor at call
    time, which non-negotiable 9b forbids.
    """

    def test_injected_when_constructor_names_it(self) -> None:
        kwargs = resolve_generator_kwargs(_config(), model_cls=_DeviceAware, device="cuda")
        assert kwargs["device"] == "cuda"

    def test_not_injected_into_a_var_kwargs_only_constructor(self) -> None:
        """The blast-radius guard, and the reason 3d does not use ``accepts()``.

        ``_VarKwargsOnly`` is the shape of nearly every registered generator, and
        several forward ``**kwargs`` straight into strict sub-configs that raise
        on an unexpected key -- the failure ``SKIP_MODEL_FIELDS`` exists to stop.
        Widening the gate to the tolerant :func:`accepts` (as step 3c uses)
        turns this red, which is the point of pinning it.
        """
        kwargs = resolve_generator_kwargs(_config(), model_cls=_VarKwargsOnly, device="cuda")
        assert "device" not in kwargs

    def test_not_injected_when_constructor_cannot_accept_it(self) -> None:
        kwargs = resolve_generator_kwargs(_config(), model_cls=_Bare, device="cuda")
        assert "device" not in kwargs

    def test_no_device_means_no_injection(self) -> None:
        """``None`` is "the caller resolved nothing", never "resolve one here"."""
        kwargs = resolve_generator_kwargs(_config(), model_cls=_DeviceAware)
        assert "device" not in kwargs

    def test_model_kwargs_contradicting_the_run_device_raises(self) -> None:
        with pytest.raises(ValueError, match="Device configuration conflict"):
            resolve_generator_kwargs(
                _config(model_kwargs={"device": "cpu"}), model_cls=_DeviceAware, device="cuda"
            )

    def test_model_kwargs_agreeing_with_the_run_device_is_accepted(self) -> None:
        kwargs = resolve_generator_kwargs(
            _config(model_kwargs={"device": "cuda"}), model_cls=_DeviceAware, device="cuda"
        )
        assert kwargs["device"] == "cuda"

    def test_full_resolution_forwards_the_device(self) -> None:
        """The probe calls the composed form; it must inject what training does."""
        resolved = resolve_full_generator_kwargs(
            _config(), model_cls=_DeviceAware, device="cuda"
        )
        assert resolved.kwargs["device"] == "cuda"


# ---------------------------------------------------------------------------
# physics.data_consistency noise keys reach the generator (#1525)
# ---------------------------------------------------------------------------


class TestDCNoiseKeysAreForwarded:
    """These were validated, documented schema fields with no consumer.

    ``generator_kwargs`` step 3c forwarded exactly ``use_dc`` / ``dc_method`` /
    ``dc_weight``, so every DC layer fell back to its own hard-coded 0.01/0.005
    and a declared value was discarded in silence (non-negotiable 8).
    """

    @staticmethod
    def _config(**dc_overrides):
        from types import SimpleNamespace

        dc = {
            "enabled": True,
            "method": "hard",
            "weight": 1.0,
            "train_noise_level": 0.01,
            "eval_noise_level": 0.005,
            "noise_type": "gaussian",
        }
        dc.update(dc_overrides)
        return SimpleNamespace(
            model=SimpleNamespace(model_type="kspace_cold_diffusion", model_kwargs={}),
            physics=SimpleNamespace(data_consistency=SimpleNamespace(**dc)),
        )

    def test_declared_noise_levels_reach_the_kwargs(self) -> None:
        from spectramr.infrastructure.builders.generator_kwargs import (
            resolve_generator_kwargs,
        )

        kwargs = resolve_generator_kwargs(
            self._config(train_noise_level=0.077, eval_noise_level=0.033),
            model_type="kspace_cold_diffusion",
        )
        assert kwargs["train_noise_level"] == 0.077
        assert kwargs["eval_noise_level"] == 0.033
        assert kwargs["noise_type"] == "gaussian"

    def test_the_forwarded_set_comes_from_the_shared_table(self) -> None:
        """One owner (NN17): a key added to DC_SSOT_KEYS must be forwarded."""
        from spectramr.infrastructure.builders.generator_kwargs import (
            resolve_generator_kwargs,
        )
        from spectramr.infrastructure.physics.dc_settings import DC_SSOT_KEYS

        kwargs = resolve_generator_kwargs(
            self._config(), model_type="kspace_cold_diffusion"
        )
        for kwarg, _field in DC_SSOT_KEYS:
            assert kwarg in kwargs, f"{kwarg} is in DC_SSOT_KEYS but was not forwarded"
