"""Tests for :mod:`spectramr.infrastructure.training.strategies.mixins.kspace`.

Focused on :meth:`KspaceMixin._prepare_model_input` domain classification.

Regression anchor (2026-06-28): the entire ``experiment_11`` k-space cohort
delivers **8-channel real-stacked** multi-coil k-space (4 coils x real/imag;
``cross_contrast`` = 16-ch). The classifier previously treated k-space input as
k-space only for complex-typed OR exactly 2-channel real tensors, so the 8/16-ch
arms were misclassified as *image* and an extra ``fft2c`` was applied. A second
forward FFT reflects the signal (``F{F{img}} = img(-x)``) -> the model's
"prepared k-space" became a 180-deg-rotated image (the "doubled brain"). These
tests pin the passthrough so no double-FFT can recur.
"""

import pytest
import torch

from spectramr.infrastructure.training.strategies.mixins.kspace import KspaceMixin


class _StubConfigModel:
    def __init__(self, model_domain: str, input_type: str) -> None:
        self.model_domain = model_domain
        self.input_type = input_type


class _StubConfig:
    def __init__(self, model_domain: str, input_type: str) -> None:
        self.model = _StubConfigModel(model_domain, input_type)


class _Harness(KspaceMixin):
    """Minimal carrier exposing only what ``_prepare_model_input`` reads."""

    def __init__(self, model_domain: str, input_type: str) -> None:
        self.config = _StubConfig(model_domain, input_type)
        self.device = torch.device("cpu")


def test_eight_channel_real_kspace_is_passthrough_no_double_fft():
    """8-ch real-stacked k-space with input_type=kspace must NOT be FFT'd."""
    h = _Harness(model_domain="kspace", input_type="kspace")
    x = torch.randn(1, 8, 16, 16)  # 4 coils x (real, imag)
    out = h._prepare_model_input(x)
    # Passthrough: identical object/values, no extra fft2c applied.
    assert out.shape == x.shape
    assert torch.equal(out, x)


def test_sixteen_channel_real_kspace_is_passthrough():
    """cross_contrast 16-ch real-stacked k-space must also pass through."""
    h = _Harness(model_domain="kspace", input_type="kspace")
    x = torch.randn(1, 16, 16, 16)
    out = h._prepare_model_input(x)
    assert torch.equal(out, x)


def test_two_channel_real_kspace_still_passthrough():
    """Regression guard: the original 2-ch behaviour is preserved."""
    h = _Harness(model_domain="kspace", input_type="kspace")
    x = torch.randn(1, 2, 16, 16)
    out = h._prepare_model_input(x)
    assert torch.equal(out, x)


def test_image_input_to_kspace_model_still_transforms():
    """input_type=image into a k-space model must still get fft2c'd.

    Guards against over-broadening the passthrough: a genuine image-domain
    real/imag tensor declared ``input_type: image`` must NOT be treated as
    k-space, so the legitimate ``image_to_kspace`` transform still fires.
    """
    h = _Harness(model_domain="kspace", input_type="image")
    x = torch.randn(1, 8, 16, 16)
    out = h._prepare_model_input(x)
    # Transform applied -> not the identity tensor.
    assert not (out.shape == x.shape and torch.equal(out, x))


# ---------------------------------------------------------------------------
# setup_kspace_components: the accelerator kwargs it hands the mask generator
#
# Regression anchor (2026-08-16): a 1-rank probe of
# ``experiment_11_attention_none`` trained fine and then raised at the first
# validation step, when the strategy's generator lazily built its accelerator:
#
#     TypeError: ['adaptive', 'enable_dynamic_mask', ..., 'mask_seed', ...,
#     'use_gradient_checkpointing'] is not read by any registered k-space
#     accelerator, so DensityNestedKSpaceAccelerator would silently discard it.
#
# The mixin dumped the whole frozen ``AccelerationConfigSchema`` and removed
# exactly one key, so every schema default rode along and ``mask_seed`` was
# never translated to the accelerator's ``seed``. Training never noticed:
# masks arrive with the batch there, so the accelerator is not constructed
# until validation asks for one.
# ---------------------------------------------------------------------------


class _StubEnv:
    def __init__(self, generator) -> None:
        self.generator = generator


class _AccelHarness(KspaceMixin):
    """Carrier exposing only what ``setup_kspace_components`` reads."""

    def __init__(self, generator) -> None:
        from types import SimpleNamespace

        self.config = SimpleNamespace(
            physics=None,
            model=SimpleNamespace(model_type="kspace_cold_diffusion"),
        )
        self.env = _StubEnv(generator)
        self.device = torch.device("cpu")

    def _is_cold_diffusion(self) -> bool:
        return True


def _exp11_acceleration():
    from spectramr.config.schemas.acceleration import AccelerationConfigSchema

    return AccelerationConfigSchema(
        acceleration_type="density_nested",
        base_acceleration=2.0,
        max_acceleration=32.0,
        center_fraction=0.08,
        min_center_fraction=0.02,
        acceleration_range=[2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0],
        mask_direction="phase",
        schedule_type="step",
        mask_seed=42,
        enforce_nested=True,
        enable_dynamic_mask=True,
    )


def _model_with_process(undersampling=None, *, timesteps: int = 28):
    """A stand-in for ``KSpaceColdDiffusionGenerator``, built the same way.

    The constructor resolves ``undersampling:`` through
    ``resolve_undersampling_kwargs`` and hands the result to
    ``KSpaceUndersamplingProcess``; reproducing exactly that here is what makes
    the borrow assertions below statements about the production handoff rather
    than about a mock.
    """
    from types import SimpleNamespace

    from spectramr.models.diffusion.kspace_process import (
        KSpaceUndersamplingProcess,
        resolve_undersampling_kwargs,
    )

    process = KSpaceUndersamplingProcess(
        num_timesteps=timesteps,
        **resolve_undersampling_kwargs(undersampling or {}, {}),
    )
    return SimpleNamespace(dc_layer=None, kspace_process=process)


def test_the_strategy_borrows_the_models_generator_rather_than_building_one():
    """THE pin for #2056. Two instances is the defect, not two log lines.

    They agreed on every accelerator kwarg -- both sides translate the same
    block through the same allowlist -- so the duplication was invisible except
    as ``Creating Accelerator`` appearing twice. Identity is the only assertion
    that a value comparison cannot pass by coincidence.
    """
    model = _model_with_process(_exp11_acceleration())
    h = _AccelHarness(model)
    h.setup_kspace_components()
    assert h.mask_generator is model.kspace_process.mask_generator


def test_one_accelerator_is_constructed_however_many_sides_ask_for_it(caplog):
    """The duplicate log line the cluster run showed, pinned as a count.

    ``_get_accelerator`` is lazy and memoises per pattern, so one shared
    generator logs once no matter how many callers materialise it; two
    generators log twice with byte-identical params, which is exactly what
    job 8592576 printed.
    """
    import logging

    model = _model_with_process(_exp11_acceleration())
    h = _AccelHarness(model)
    h.setup_kspace_components()

    with caplog.at_level(logging.INFO, logger="spectramr.infrastructure.physics.sampling"):
        h.mask_generator._get_accelerator(None)
        model.kspace_process.mask_generator._get_accelerator(None)

    created = [r for r in caplog.records if "Creating Accelerator" in r.getMessage()]
    assert len(created) == 1, f"expected one accelerator, got {len(created)}"


def test_a_ddp_wrapped_generator_is_unwrapped_before_the_lookup():
    """``kspace_process`` hangs off the module, not off the DDP wrapper.

    Without the unwrap the attribute lookup misses under distributed training
    and every multi-GPU cold-diffusion arm would hit the raise below.
    """
    from types import SimpleNamespace

    model = _model_with_process(_exp11_acceleration())
    wrapped = SimpleNamespace(module=model, dc_layer=None)
    h = _AccelHarness(wrapped)
    h.setup_kspace_components()
    assert h.mask_generator is model.kspace_process.mask_generator


def test_a_generator_without_a_process_raises_rather_than_building_a_second():
    """Non-negotiable 3. Falling back is what put two owners here."""
    from types import SimpleNamespace

    h = _AccelHarness(SimpleNamespace(dc_layer=None))
    with pytest.raises(ValueError, match="kspace_process"):
        h.setup_kspace_components()


def test_validation_accelerator_constructs_from_a_real_arm_config():
    """The exact construction that raised on the cluster must still succeed.

    ``_get_accelerator`` is where the kwargs are finally splatted, so calling it
    is the assertion -- the vocabulary gate raises on any unread name.
    """
    h = _AccelHarness(_model_with_process(_exp11_acceleration()))
    h.setup_kspace_components()
    assert h.mask_generator._get_accelerator(None) is not None


def test_mask_seed_reaches_the_accelerator_as_seed():
    """``seed=None`` would send masking to the global RNG (issue #1059).

    The cascade then re-draws a fresh permutation per call instead of
    truncating one fixed ranking, so ``M_{t+1} subset-of M_t`` no longer holds
    -- which cold diffusion's forward process assumes.
    """
    h = _AccelHarness(_model_with_process(_exp11_acceleration()))
    h.setup_kspace_components()
    assert h.mask_generator._accelerator_kwargs["seed"] == 42
    assert h.mask_generator._get_accelerator(None).seed == 42


def test_unread_schema_defaults_are_not_forwarded():
    """Anti-vacuity for the test above: a dump-and-filter would carry these."""
    h = _AccelHarness(_model_with_process(_exp11_acceleration()))
    h.setup_kspace_components()
    kwargs = h.mask_generator._accelerator_kwargs
    for junk in (
        "mixed_precision",
        "use_compile",
        "use_distributed",
        "gradient_accumulation_steps",
        "ground_truth_folder",
        "schedule_steps",
        "enable_dynamic_mask",
        "acceleration_type",
        "mask_seed",
    ):
        assert junk not in kwargs, f"{junk} would reach the accelerator"


def test_declared_values_survive_the_handoff():
    """Borrowing must not cost the arm its ladder.

    Filtering alone could pass the two tests above while dropping real values,
    and a borrow could pass the identity test while the MODEL's generator was
    the one built wrong.
    """
    h = _AccelHarness(_model_with_process(_exp11_acceleration()))
    h.setup_kspace_components()
    kwargs = h.mask_generator._accelerator_kwargs
    assert h.mask_generator.num_timesteps == 28, "the arm's schedule length must survive"
    assert h.mask_generator.default_pattern == "density_nested"
    assert kwargs["max_acceleration"] == 32.0
    assert kwargs["base_acceleration"] == 2.0
    assert kwargs["min_center_fraction"] == 0.02
    assert kwargs["acceleration_schedule"] == "step"
    assert kwargs["mask_direction"] == "phase"
    assert kwargs["enforce_nested"] is True
    assert kwargs["acceleration_range"] == [2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0]


def test_an_absent_undersampling_block_still_yields_one_owner():
    """The two sides agree even where there is nothing declared to agree on.

    This replaces a pin on the strategy's own "linear, no kwargs" default.
    That default existed to stop a strategy-side resolver from inventing a 32x
    ladder for an arm that declared none; with no strategy-side resolver left,
    the model's process is the only thing that can answer, and agreeing with it
    is the property worth holding.
    """
    model = _model_with_process(None)
    h = _AccelHarness(model)
    h.setup_kspace_components()
    assert h.mask_generator is model.kspace_process.mask_generator


# ---------------------------------------------------------------------------
# _prepare_validation_data: the published scale must match the tensor it scales
#
# Regression anchor (2026-08-19): a 40-iteration cluster relaunch of
# ``experiment_11_attention_none`` trained fine and died at the first
# validation step with ``RuntimeError: The size of tensor a (36) must match
# the size of tensor b (2) at non-singleton dimension 0``.
#
# ``36 = 2 subjects x 18 slices``. ``train.py._preprocess_validation_tensor``
# flattens depth into the batch axis for ``val_batch.input``/``.target`` and
# leaves per-sample batch fields alone, so ``kspace_scale`` stayed length 2.
# This method sized ``scale_factor`` correctly from ``input_batch.size(0)`` and
# then REPLACED it with the length-2 field via ``view(-1, 1, 1, 1)``.
#
# The method's own 5D branch cannot compensate: the tensor arrives already 4D,
# so ``input_batch.dim() == 5`` is False and no expansion runs.
# ---------------------------------------------------------------------------


class _ValidationHarness(KspaceMixin):
    """Carrier exposing only what ``_prepare_validation_data`` reads."""

    def __init__(self, *, enable_kspace_normalization: bool = True) -> None:
        from types import SimpleNamespace

        self.config = SimpleNamespace(
            model=SimpleNamespace(in_channels=1),
            data=SimpleNamespace(
                processing=SimpleNamespace(enable_kspace_normalization=enable_kspace_normalization)
            ),
        )
        self.device = torch.device("cpu")


def test_per_subject_scale_expands_to_a_pre_flattened_batch():
    """The exact cluster shapes: a length-2 scale must become length 36."""
    h = _ValidationHarness()
    # Already flattened by train.py: 2 subjects x 18 slices.
    input_batch = torch.ones(36, 1, 8, 8)
    target_batch = torch.ones(36, 1, 8, 8)
    batch_data = {"kspace_scale": torch.tensor([224.36, 198.15])}

    _, _, scale_factor = h._prepare_validation_data(None, input_batch, target_batch, batch_data)

    assert scale_factor.shape == (36, 1, 1, 1)
    # The multiply that raised on the cluster.
    assert (input_batch * scale_factor).shape == (36, 1, 8, 8)


def test_the_expansion_is_subject_major_not_interleaved():
    """Anti-vacuity for the shape check above.

    ``repeat`` yields the identical shape and applies subject 0's scale to
    subject 1's slices -- silently wrong metrics rather than a crash. Pinned by
    value, with D=3 so the two orderings differ.
    """
    h = _ValidationHarness()
    input_batch = torch.ones(6, 1, 4, 4)
    batch_data = {"kspace_scale": torch.tensor([10.0, 20.0])}

    _, _, scale_factor = h._prepare_validation_data(
        None, input_batch, input_batch.clone(), batch_data
    )

    assert scale_factor.flatten().tolist() == [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]


def test_a_published_scale_is_not_recomputed():
    """Guard the branch boundary: the ``else`` arm divides a second time.

    Reaching the quantile fallback with a scale already published is the defect
    that ``read_batch_field`` fixed upstream; this pins that the aligned path is
    still the one taken.
    """
    h = _ValidationHarness()
    input_batch = torch.full((4, 1, 4, 4), 3.0)
    batch_data = {"kspace_scale": torch.tensor([2.0, 5.0])}

    out_input, _, scale_factor = h._prepare_validation_data(
        None, input_batch, input_batch.clone(), batch_data
    )

    # Published path: tensors pass through undivided ("Do not divide again!").
    assert torch.equal(out_input, input_batch)
    assert scale_factor.flatten().tolist() == [2.0, 2.0, 5.0, 5.0]


def test_a_scalar_published_scale_still_covers_the_batch():
    """The one arm of the old ndim ladder that was already correct."""
    h = _ValidationHarness()
    input_batch = torch.ones(4, 1, 4, 4)
    batch_data = {"kspace_scale": torch.tensor(9.0)}

    _, _, scale_factor = h._prepare_validation_data(
        None, input_batch, input_batch.clone(), batch_data
    )

    assert scale_factor.shape == (4, 1, 1, 1)
    assert scale_factor.flatten().tolist() == [9.0] * 4


def test_an_unalignable_scale_raises_instead_of_reaching_the_multiply():
    """A length that does not divide has no benign reading (non-negotiable 3).

    Before this change the mismatch surfaced ~40 frames downstream as a bare
    ``RuntimeError`` at ``hr_fakes * denom_scale`` with no field named.
    """
    import pytest

    h = _ValidationHarness()
    input_batch = torch.ones(36, 1, 8, 8)
    batch_data = {"kspace_scale": torch.ones(5)}

    with pytest.raises(ValueError, match="kspace_scale"):
        h._prepare_validation_data(None, input_batch, input_batch.clone(), batch_data)


# ---------------------------------------------------------------------------
# #1917 -- the asymmetric-degradation write in ``generate_and_process_mask``
#
# ``expand_mask_to_channels`` widens [B, 1, H, W] -> [B, C, H, W] with
# ``Tensor.expand``, i.e. a stride-0 broadcast view: all C channels alias one
# row of memory. Ten of its eleven callers only multiply by the mask, so the
# cheap view is the right contract and must stay. The eleventh writes into it,
# and therefore owns the copy. These pin both halves of that split.
# ---------------------------------------------------------------------------


class _AsymMaskHarness(KspaceMixin):
    """Carrier exposing only what ``generate_and_process_mask`` reads."""

    def __init__(self, target_channels: int | None, out_channels: int) -> None:
        from types import SimpleNamespace

        from spectramr.infrastructure.training.utils.kspace_masks import (
            KSpaceMaskGenerator,
        )

        self.device = torch.device("cpu")
        self.config = SimpleNamespace(
            data=SimpleNamespace(domain=SimpleNamespace(target_channels=target_channels)),
            model=SimpleNamespace(out_channels=out_channels),
        )
        self.mask_generator = KSpaceMaskGenerator()


def _run_mask(h: "_AsymMaskHarness", mask: torch.Tensor, c_total: int) -> torch.Tensor:
    b, _, height, width = mask.shape
    return h.generate_and_process_mask(
        batch_size=b,
        timesteps=torch.zeros(b, dtype=torch.long),
        target_shape=(b, c_total, height, width),
        current_step=0,
        batch_data={"mask": mask},
    )


def test_asymmetric_write_survives_the_expanded_broadcast_view():
    """Shape 1: a [B, 1, H, W] mask widened to C_total is a stride-0 view.

    Before the fix this raised ``RuntimeError: ... more than one element of the
    written-to tensor refers to a single memory location``. The TI-CCD split
    (C_total=16, target_channels=8) is the shape the k-space cohort delivers.
    """
    h = _AsymMaskHarness(target_channels=8, out_channels=8)
    mask = torch.zeros(2, 1, 8, 8)
    mask[:, :, :, ::2] = 1.0

    out = _run_mask(h, mask, c_total=16)

    assert out.shape == (2, 16, 8, 8)
    # Source half forced fully sampled, target half left at the sampled pattern.
    assert bool(out[:, :8].eq(1.0).all())
    assert torch.equal(out[:, 8:], mask.expand(-1, 8, -1, -1))


def test_asymmetric_write_does_not_mutate_the_caller_s_batch():
    """Shape 2: a mask that ALREADY has C_total channels is returned unchanged.

    ``.to(device).float()`` is a no-op for an already-float tensor on the same
    device, so without the copy the write reached through into
    ``batch_data["mask"]`` and pinned the source channels to 1.0 for every later
    consumer of that batch. That failure is silent -- no exception, wrong data.
    """
    h = _AsymMaskHarness(target_channels=8, out_channels=8)
    mask = torch.zeros(2, 16, 8, 8)
    before = mask.clone()

    out = _run_mask(h, mask, c_total=16)

    assert bool(out[:, :8].eq(1.0).all())
    assert torch.equal(mask, before), "generate_and_process_mask mutated its input"


def test_asymmetric_write_survives_on_the_legacy_equal_split_branch():
    """Shape 3: ``target_channels=None`` falls to the C_total // 2 branch.

    That branch carries the same in-place write, so it needs the same copy; a
    fix applied to only the first branch would leave it crashing.
    """
    h = _AsymMaskHarness(target_channels=None, out_channels=4)
    mask = torch.zeros(2, 1, 8, 8)
    mask[:, :, :, ::2] = 1.0

    out = _run_mask(h, mask, c_total=16)

    assert out.shape == (2, 16, 8, 8)
    assert bool(out[:, :8].eq(1.0).all())


def test_expand_mask_to_channels_still_returns_a_cheap_broadcast_view():
    """The producer must NOT be "fixed" by making its result contiguous.

    Ten of the eleven callers only multiply by the mask, inside the training
    loop. Materialising C copies there to spare the single mutating caller a
    ``clone()`` would be a needless per-step allocation (non-negotiable 9).
    This pins the cheap contract so that trade-off has to be made deliberately.
    """
    from spectramr.infrastructure.training.utils.kspace_masks import (
        KSpaceMaskGenerator,
    )

    widened = KSpaceMaskGenerator().expand_mask_to_channels(torch.zeros(2, 1, 8, 8), 16)

    assert widened.shape == (2, 16, 8, 8)
    assert widened.stride()[1] == 0, "channel dim must still alias, not be copied"
