"""Field-conditioning injection on the LIVE reconstruction forward (regressions).

A field-conditioned generator (``AnatomyFieldRenderer`` / ``CrossFieldRenderer`` &
friends) takes ``field_strength`` as a *keyword-only, required* argument. When such
a model is routed through the GENERIC reconstruction validation path — e.g. the
calibration strategy (``ConformalCalibrationStrategy``), which inherits
``_validation_forward`` from ``ReconstructionTrainingStrategy`` and has no
field-aware override — the forward kwargs MUST carry ``field_strength`` or the
generator raises ``missing 1 required keyword-only argument: 'field_strength'`` and
the run produces zero successful validation batches (CLAUDE.md #10; the mrixfields
``b17_dice_risk_calibration`` crash, re-reproduced 2026-06-24).

The original fix landed in an uncalled mixin copy of the generator-input
preparation, so a green test guarded dead code while b17 kept crashing. That copy
was deleted (#2787); these tests exercise the live
``ReconstructionTrainingStrategy._prepare_generator_inputs`` and the real
``_validation_forward`` seam.

The injection is signature-gated (``_callable_accepts_kwarg``) so plain models are
untouched, prefers the TARGET field over the source, and is never defaulted (a
default field would be a silent fallback, pitfall #9).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)


class _FieldGen(nn.Module):
    """Cross-field renderer stub: field_strength is keyword-only + required."""

    def forward(self, x, *, field_strength, contrast_id=None):  # noqa: D401
        # Echo the field so a test can assert it actually flowed through.
        self.last_field_strength = field_strength
        self.last_contrast_id = contrast_id
        return x


class _PlainGen(nn.Module):
    def forward(self, x):  # noqa: D401
        return x


def _host(generator: nn.Module) -> ReconstructionTrainingStrategy:
    """Minimal LIVE strategy host (no full DI build) for the forward-prep seam."""
    host = ReconstructionTrainingStrategy.__new__(ReconstructionTrainingStrategy)
    host.env = SimpleNamespace(
        generator=generator,
        config=SimpleNamespace(
            model=SimpleNamespace(model_type="anatomy_field_renderer", input_type="image"),
            # `multislice_enabled` is a declared DataConfigSchema field with a
            # default, so a real config always carries it. The stub omitted it,
            # which only passed while the reader used a hasattr fallback.
            data=SimpleNamespace(multislice_enabled=False),
            training=SimpleNamespace(),
            validation=None,
        ),
    )
    host.state = SimpleNamespace(config=host.env.config)
    host.config = host.env.config
    host.logging_service = None
    host.dc_layer = None
    return host


def _ctx(**extra):
    base = {"use_dc": False, "multimodal_inputs": None, "measured_kspace": None}
    base.update(extra)
    return base


def test_field_strength_injected_when_generator_requires_it():
    host = _host(_FieldGen())
    x = torch.randn(1, 1, 8, 8)
    ctx = _ctx(field_strength=torch.tensor([0.1]), field_strength_target=torch.tensor([3.0]))
    _lr, fwd_kwargs = host._prepare_generator_inputs(ctx, x, validation=True)
    assert "field_strength" in fwd_kwargs
    # prefers the TARGET field (render at HF), not the 0.1 T source
    assert torch.allclose(fwd_kwargs["field_strength"], torch.tensor([3.0]))


def test_field_strength_falls_back_to_source_when_no_target():
    host = _host(_FieldGen())
    x = torch.randn(1, 1, 8, 8)
    ctx = _ctx(field_strength=torch.tensor([0.064]))
    _lr, fwd_kwargs = host._prepare_generator_inputs(ctx, x, validation=True)
    assert torch.allclose(fwd_kwargs["field_strength"], torch.tensor([0.064]))


def test_field_strength_not_injected_for_plain_generator():
    host = _host(_PlainGen())
    x = torch.randn(1, 1, 8, 8)
    ctx = _ctx(field_strength_target=torch.tensor([3.0]))
    _lr, fwd_kwargs = host._prepare_generator_inputs(ctx, x, validation=True)
    assert "field_strength" not in fwd_kwargs


def test_contrast_id_injected_when_generator_accepts_it():
    host = _host(_FieldGen())
    x = torch.randn(1, 1, 8, 8)
    ctx = _ctx(field_strength_target=torch.tensor([3.0]), contrast_id=torch.tensor([1]))
    _lr, fwd_kwargs = host._prepare_generator_inputs(ctx, x, validation=True)
    assert "contrast_id" in fwd_kwargs
    assert torch.equal(fwd_kwargs["contrast_id"], torch.tensor([1]))


def test_validation_forward_does_not_raise_for_field_renderer():
    """The real crash seam: ``_validation_forward`` -> generator(**kwargs).

    Before the fix this raised ``missing 1 required keyword-only argument:
    'field_strength'`` (b17). Now it must complete and the field must reach the
    generator's forward.
    """
    gen = _FieldGen()
    host = _host(gen)
    x = torch.randn(1, 1, 8, 8)
    ctx = _ctx(field_strength_target=torch.tensor([3.0]))
    out = host._validation_forward(x, ctx)
    assert out.shape == x.shape
    assert torch.allclose(gen.last_field_strength, torch.tensor([3.0]))


# ---------------------------------------------------------------------------
# #2324: `measured_kspace` is the measurement, never the fully sampled reference.
#
# Every k-space route keeps the reference under `kspace`: the FastMRI subject
# builder stores the unmasked primary there and `PhysicsInformedMasking` leaves
# it, and the M4Raw dataset sets it to the target. The alias table read `kspace`
# FIRST, so EI built x_hat = f(A^H y_full) and trained supervised with D4
# augmentation. Measured on the real path, before the fix, for ei_unet,
# ei_control_mc_only and idea_1_hermitian_fno_recon_4x: measured_kspace == target.
# ---------------------------------------------------------------------------


def _measured_context(arm: str, mapping_override=None, monkeypatch=None):
    from spectramr.infrastructure.training.strategies.mixins import reconstruction as mixin
    from tests.fixtures.self_supervised_batches import load_arm, real_batches

    if mapping_override is not None:
        monkeypatch.setitem(mixin.AUX_KEY_MAPPING, "measured_kspace", mapping_override)
    config = load_arm(arm)
    batch, _ = real_batches(config)

    class _Host(mixin.ReconstructionMixin):
        env = None

    host = _Host()
    host.state = type("S", (), {"config": config})()
    context = host._prepare_batch_context_reconstruction(batch.input, batch.target, batch=batch)
    return batch, context


def _measurement_error(batch, context) -> str | None:
    """Why ``context['measured_kspace']`` is not the masked input, or ``None``."""
    measured = context.get("measured_kspace")
    if measured is None:
        return "measured_kspace is absent"
    if torch.equal(measured, batch.target):
        return "measured_kspace is the fully sampled target"
    if not torch.equal(measured, batch.input):
        return "measured_kspace is not the masked input"
    if (measured * (1 - context["mask"])).abs().max() != 0:
        return "measured_kspace carries energy off the acquisition mask"
    return None


@pytest.mark.parametrize("arm", ["ei_unet_m4raw_r4", "ei_control_mc_only_m4raw_r4"])
def test_the_ei_measurement_is_the_masked_input_through_the_real_path(arm: str) -> None:
    """Including the alpha=0 control, which must see undersampled input at training."""
    batch, context = _measured_context(arm)
    assert _measurement_error(batch, context) is None


def test_the_old_alias_order_is_red(monkeypatch) -> None:
    """The planted violation: `kspace` first resolves to the reference again."""
    batch, context = _measured_context(
        "ei_unet_m4raw_r4", mapping_override=("kspace", "measured_kspace"), monkeypatch=monkeypatch
    )
    assert _measurement_error(batch, context) == "measured_kspace is the fully sampled target"


def test_the_explicit_key_outranks_the_reference() -> None:
    from spectramr.infrastructure.training.strategies.mixins.reconstruction import AUX_KEY_MAPPING

    assert AUX_KEY_MAPPING["measured_kspace"][0] == "measured_kspace"


def test_generator_input_preparation_has_one_owner() -> None:
    """#2787: the mixin once carried an uncalled copy of the generator-input
    preparation, and fixes (field_strength, #2759's timesteps) kept landing in it.
    The shared step is the single owner; no mixin copy may come back, and the
    strategies that run the step inherit it rather than redefine it."""
    from spectramr.infrastructure.training.strategies.mixins import reconstruction
    from spectramr.infrastructure.training.strategies.n2n_strategy import NoiseToNoiseStrategy
    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionStepMixin,
    )

    mixin = reconstruction.ReconstructionMixin
    copies = [name for name in dir(mixin) if name.startswith("_prepare_generator_inputs")]
    assert copies == []
    assert "_prepare_generator_inputs" in vars(ReconstructionStepMixin)
    for strategy in (ReconstructionTrainingStrategy, NoiseToNoiseStrategy):
        assert "_prepare_generator_inputs" not in vars(strategy)
