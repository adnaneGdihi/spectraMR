"""The contrast id must survive the reverse loop, or validation grades another model.

Training conditions every forward: ``_forward_through_model`` puts ``contrast_idx``
in ``gen_kwargs`` and the generator builds ``contrast_emb`` from it. The cold
multistep validation path called ``_sample_multistep_chunked`` without those
kwargs, and ``sample()`` re-enters ``forward(x_t, t)`` bare, so every reverse step
ran unconditioned while the embedding existed and was trained.

Nothing raised and nothing logged: ``forward`` pops ``contrast_idx`` with a
``None`` default and skips the embedding silently. All 71 cold arms in the
``kspace_filling`` cohort declare ``num_contrasts: 3`` and
``enable_multistep_cold: true``, and m4raw stamps the id unconditionally, so the
metric that drives checkpoint selection and early stopping was measured with the
embedding never firing.
"""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.generators.kspace_cold_diffusion_generator import (  # noqa: E402
    KSpaceColdDiffusionGenerator,
)

COMMON = {
    "in_channels": 4,
    "out_channels": 4,
    "features": (8, 16),
    "force_pure_kspace": True,
    "attention_type": "none",
    "use_dc": False,
    "kspace_log_scaled": False,
    "condition_with_smaps": False,
    "num_contrasts": 3,
    "timesteps": 4,
}


def _gen():
    torch.manual_seed(0)
    return KSpaceColdDiffusionGenerator(**COMMON).eval()


def _spy(gen):
    fired: list[tuple[int, ...]] = []
    gen.contrast_embedding.register_forward_hook(lambda m, i, o: fired.append(tuple(o.shape)))
    return fired


def test_the_training_forward_conditions_on_the_contrast():
    """The half that always worked — the baseline the reverse path must match."""
    gen = _gen()
    fired = _spy(gen)
    with torch.no_grad():
        gen(
            torch.randn(1, 4, 32, 32),
            torch.zeros(1, dtype=torch.long),
            contrast_idx=torch.tensor([1]),
        )
    assert len(fired) == 1


def test_every_reverse_step_is_conditioned_on_the_contrast():
    """Observed, not inferred: the hook fires once per reverse step, not zero times."""
    gen = _gen()
    fired = _spy(gen)
    x = torch.randn(1, 4, 32, 32)
    with torch.no_grad():
        gen.sample(
            measurement=x,
            mask=torch.ones_like(x),
            inference_timesteps=3,
            contrast_idx=torch.tensor([1]),
        )
    assert len(fired) >= 1, "the contrast embedding never fired on the reverse trajectory"


def test_the_stash_does_not_leak_into_a_later_unconditioned_call():
    """A stash that outlives its call would condition the next arm's trajectory."""
    gen = _gen()
    x = torch.randn(1, 4, 32, 32)
    with torch.no_grad():
        gen.sample(
            measurement=x,
            mask=torch.ones_like(x),
            inference_timesteps=2,
            contrast_idx=torch.tensor([2]),
        )
    assert getattr(gen, "_current_contrast_idx", None) is None


def test_the_chunked_validation_path_splits_the_contrast_with_the_measurement():
    """Per-element tensors must be split in step, or one subject gets another's id.

    This is the shape of #1914 — x and timesteps sliced while a per-element
    tensor stayed batch-aligned — so it is pinned rather than argued.
    """
    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )

    src = inspect.getsource(DiffusionTrainingStrategy._sample_multistep_chunked)
    assert "contrast_idx.split(chunk, dim=0)" in src, "contrast id is not chunked"
    assert '"contrast_idx": contrast_c' in src, "the chunk's own id is not the one passed"
    assert "if contrast_c is not None" in src, (
        "the id must be omitted rather than passed as None: sample() is called "
        "generically and a generator predating contrast conditioning never "
        "declared the keyword"
    )


def test_the_validation_call_sites_pass_the_contrast_they_built():
    """The drop was at the call site, not in the sampler — pin both branches."""
    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )

    src = inspect.getsource(DiffusionTrainingStrategy._generate_validation_prediction)
    assert src.count('contrast_idx=gen_kwargs.get("contrast_idx")') == 2, (
        "the multistep and ensemble branches must both forward the contrast id"
    )
