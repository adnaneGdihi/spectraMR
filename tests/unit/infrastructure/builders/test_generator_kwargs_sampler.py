"""``training.diffusion.sampler`` must reach the generator, or it selects nothing.

The generator resolved its sampler from ``model_kwargs.sampler`` /
``model_kwargs.inference_sampler`` and fell through to the literal ``cold_mri``.
Nothing on the production path wrote either key, so the schema field was inert:
an arm declaring ``dps_posterior`` trained, validated and reported success while
running the cold sampler.

It was latent rather than live here -- all 71 cohort arms declare ``cold_mri``,
which is also the literal -- so wiring it moves no current run. What changes is
that a different declared value now takes effect, and an unregistered one raises
instead of degrading (non-negotiable 3).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.builders.generator_kwargs import (
    resolve_generator_kwargs,
)
from spectramr.models.generators.kspace_cold_diffusion_generator import (
    KSpaceColdDiffusionGenerator,
)

ARM = "experiments/inprogress/kspace_filling/experiment_11_kfn_none.yaml"


def _config_with_sampler(value):
    base = TrainingSettings.from_yaml(ARM)
    diffusion = base.training.diffusion.model_copy(update={"sampler": value})
    training = base.training.model_copy(update={"diffusion": diffusion})
    return base.model_copy(update={"training": training})


def _resolve(cfg):
    return resolve_generator_kwargs(
        config=cfg,
        model_cls=KSpaceColdDiffusionGenerator,
        model_type="kspace_cold_diffusion",
        device="cpu",
    )


def test_the_declared_sampler_reaches_the_generator_kwargs():
    """The planted case: a value that is NOT the literal fallback."""
    assert _resolve(_config_with_sampler("dps_posterior")).get("sampler") == "dps_posterior"


def test_the_cohort_value_is_unchanged():
    """71 arms declare cold_mri, so wiring the knob must move no current run."""
    assert _resolve(_config_with_sampler("cold_mri")).get("sampler") == "cold_mri"


def test_an_unregistered_sampler_raises_at_the_module_that_looks_it_up():
    """The consumer validates, not the resolver.

    The three prior-method baselines declare ``ddpm`` -- a value the schema's
    own Literal advertises and the runtime registry does not carry. They never
    reach a reverse loop, so refusing them in the resolver would fail three arms
    for a name nothing would have looked up. The generator that calls
    ``get_sampler`` is the one owner that can judge it.
    """
    with pytest.raises(ValueError, match="Unknown sampler"):
        KSpaceColdDiffusionGenerator(
            in_channels=4,
            out_channels=4,
            features=(8, 16),
            force_pure_kspace=True,
            attention_type="none",
            use_dc=False,
            kspace_log_scaled=False,
            condition_with_smaps=False,
            sampler="not_a_sampler",
        )


def test_a_baseline_arm_keeps_building_with_a_sampler_its_model_never_reads():
    """Regression: validating in the resolver broke all three baseline arms."""
    for arm in ("baseline_cdiffmr", "baseline_fdb", "baseline_shen2024"):
        cfg = TrainingSettings.from_yaml(f"experiments/inprogress/kspace_filling/{arm}.yaml")
        assert cfg.training.diffusion.sampler == "ddpm"
        _resolve(cfg)  # must not raise


def test_the_arm_on_disk_still_resolves_to_its_declared_sampler():
    """Guards the three above from passing on a config the corpus does not have."""
    cfg = TrainingSettings.from_yaml(ARM)
    assert cfg.training.diffusion.sampler == "cold_mri"
    assert _resolve(cfg).get("sampler") == "cold_mri"
