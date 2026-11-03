"""SSOT for MAE/pretraining config access (issue #368).

The MAE strategy reads the typed ``training.mae`` block. That read used to be a
``getattr(self.config.training, "mae", None)`` fallback and is now guarded direct
access -- the schema declares ``training.mae`` (a first-class field), so the
getattr default was redundant SSOT. The legacy ``task_settings`` dict reads are
intentionally left as getattr: ``task_settings`` is NOT a schema field, so direct
access would raise; they are the two known-remaining entries the ssot ratchet
already accounts for. These tests pin both facts so a new config getattr-fallback
cannot creep in unnoticed (the global test_ssot_compliance guard is CI-advisory
and let pretraining.py drift past its allowance once).
"""

from __future__ import annotations

import inspect
import re


def _source() -> str:
    import spectramr.infrastructure.training.strategies.pretraining as pre

    return inspect.getsource(pre)


def test_mae_is_read_via_direct_attribute_access() -> None:
    src = _source()
    assert 'getattr(self.config.training, "mae"' not in src, (
        "training.mae is a first-class schema field; read it directly "
        "(self.config.training.mae), not via a getattr fallback."
    )
    assert re.search(r"self\.config\.training\.mae\b", src), (
        "expected a direct self.config.training.mae read after the conversion"
    )


def test_config_getattr_fallbacks_do_not_exceed_the_known_ratchet() -> None:
    # Same flat pattern the ssot ratchet uses. Only the two documented legacy
    # task_settings reads may remain; no new config getattr-fallbacks.
    src = _source()
    flat = re.findall(r"getattr\s*\(\s*[a-zA-Z_.]*config\.[a-zA-Z_.]+\s*,", src)
    assert len(flat) <= 2, f"pretraining.py config getattr-fallbacks grew: {flat}"


def test_kspace_masks_differ_per_sample_and_repeat_per_seed() -> None:
    # #2393: MaskGenerator(seed=epoch) gave every sample of an epoch one mask.
    import torch

    from spectramr.infrastructure.training.strategies.pretraining import kspace_masked_input

    x = torch.rand(3, 1, 32, 32)
    a = kspace_masked_input(x, [1, 2, 3], 0.75)
    assert a.shape == x.shape and not torch.is_complex(a)
    # One image under three seeds: each sample's corruption must differ.
    same = x[:1].expand(3, -1, -1, -1).contiguous()
    b = kspace_masked_input(same, [1, 2, 3], 0.75)
    assert not torch.allclose(b[0], b[1]) and not torch.allclose(b[1], b[2])
    # The same seeds reproduce the same masks (the fixed validation set).
    assert torch.equal(kspace_masked_input(same, [1, 2, 3], 0.75), b)


def test_kspace_masked_input_rejects_a_seed_count_mismatch() -> None:
    import pytest
    import torch

    from spectramr.infrastructure.training.strategies.pretraining import kspace_masked_input

    with pytest.raises(ValueError, match="one mask seed per sample"):
        kspace_masked_input(torch.rand(2, 1, 16, 16), [0], 0.5)


def test_training_mask_stream_is_seeded_and_checkpointed() -> None:
    from spectramr.infrastructure.training.strategies.pretraining import MAEPretrainingStrategy

    assert MAEPretrainingStrategy.checkpointed_streams == ("_mask_rng",)
    src = _source()
    assert "MaskGenerator(seed=epoch)" not in src
    assert "rank_seeded_generator(int(self.config.run.seed))" in src


# ---------------------------------------------------------------------------
# Image-domain patch masking (mask_domain: image)
# ---------------------------------------------------------------------------

_IMAGE_TWIN = "experiments/inprogress/pillars/exp_pillar_06_control_image_mae.yaml"


class _Recorder:
    """Identity generator that records what it is fed."""

    def __init__(self, owns_spatial_masking: bool = False) -> None:
        self.owns_spatial_masking = owns_spatial_masking
        self.calls: list[tuple[object, dict]] = []

    def __call__(self, x, **kwargs):
        self.calls.append((x.clone(), dict(kwargs)))
        return x

    def eval(self):
        return self


def _strategy_from_twin(generator: _Recorder):
    """Resolve the strategy the image twin declares, through the production factory."""
    import types
    from pathlib import Path

    import torch

    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
    from spectramr.models.losses.computers.base import LossOutput

    root = Path(__file__).resolve().parents[5]
    config = TrainingSettings.from_yaml(str(root / _IMAGE_TWIN))
    cls = TrainingStrategyFactory().get_strategy_class(config)
    s = object.__new__(cls)
    s.config = config
    s.device = torch.device("cpu")
    s._loss_dict_reuse = {}
    s._mask_rng = torch.Generator().manual_seed(0)
    s.env = types.SimpleNamespace(generator=generator, losses={})
    s._resolve_legacy_batch = lambda ib, kw: ib
    s.targets = []

    def compute(**kw):
        s.targets.append(kw["target"])
        return LossOutput(total=torch.zeros((), requires_grad=True), components={}, metrics={})

    s.loss_computer = types.SimpleNamespace(compute=compute)
    return s


def test_image_twin_declares_image_patch_masking() -> None:
    from spectramr.infrastructure.training.strategies.pretraining import MAEPretrainingStrategy

    s = _strategy_from_twin(_Recorder())
    assert isinstance(s, MAEPretrainingStrategy)
    assert s._mae_masking() == ("image", 0.75, 16)


def test_image_route_masks_patches_and_targets_the_clean_image() -> None:
    import torch

    gen = _Recorder()
    s = _strategy_from_twin(gen)
    clean = torch.rand(2, 1, 64, 64) + 0.5  # strictly positive, so zeroed pixels are visible
    s._compute_losses_impl(clean, clean, epoch=0)

    fed, kwargs = gen.calls[0]
    zeroed = fed == 0
    assert "disable_spatial_masking" not in kwargs
    # 75% of the 16 patches (12) zeroed per sample; the rest is the clean image untouched.
    for b in range(2):
        assert int(zeroed[b].sum()) == 12 * 16 * 16
    assert torch.equal(fed[~zeroed], clean[~zeroed])
    assert torch.equal(s.targets[0], clean)


def test_image_route_validation_mask_is_fixed_per_batch_index() -> None:
    import torch

    gen = _Recorder()
    s = _strategy_from_twin(gen)
    s._unpack_batch = lambda batch: (None, batch)
    s._get_validation_metrics_computer = lambda cfg: type(
        "C", (), {"compute": staticmethod(lambda pred, tgt: {"mse": ((pred - tgt) ** 2).mean()})}
    )()
    clean = torch.rand(2, 1, 64, 64) + 0.5
    m0 = s.validation_step(clean, batch_idx=3)
    s.validation_step(clean, batch_idx=3)
    s.validation_step(clean, batch_idx=4)
    a, b, c = (call[0] for call in gen.calls)
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert int((a == 0).sum()) == 2 * 12 * 16 * 16
    assert float(m0["val_mse"]) > 0.0  # graded against the clean image, not the masked input


def test_generator_that_owns_masking_receives_the_clean_image() -> None:
    import torch

    gen = _Recorder(owns_spatial_masking=True)
    s = _strategy_from_twin(gen)
    clean = torch.rand(1, 1, 64, 64) + 0.5
    s._compute_losses_impl(clean, clean, epoch=0)
    fed, kwargs = gen.calls[0]
    assert torch.equal(fed, clean)
    assert "disable_spatial_masking" not in kwargs


def test_mae_patch_size_is_validated() -> None:
    import pytest
    from pydantic import ValidationError

    from spectramr.config.schemas.training.strategy_knobs_2026_06 import (
        MAEPretrainingTrainingConfigSchema,
    )

    assert MAEPretrainingTrainingConfigSchema().patch_size == 16
    with pytest.raises(ValidationError):
        MAEPretrainingTrainingConfigSchema(patch_size=0)
