"""The disentangled M4Raw arms train the objective their knobs advertise.

First sabine run: experiments 50 and 51 declared a PatchGAN discriminator and
`enable_pcgrad` but no `losses.gan`, so the adversarial term was never computed, the
discriminator never trained and PCGrad never fired; all three arms declared `use_vae` with
a KL weight of 0, so the posterior had no prior term.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectramr.config.settings import TrainingSettings

ROOT = Path(__file__).resolve().parents[3] / "experiments/inprogress"
ARMS = [
    "vae_latent/experiment_32a_vae_hf_multicontrast.yaml",
    "physics_driven/experiment_50_strict_physical_disentangled.yaml",
    "physics_driven/experiment_51_fno_epg_multicontrast.yaml",
    "vae_latent/experiment_50_disentangled_anatomy_contrast.yaml",
]


def _load(arm: str) -> TrainingSettings:
    return TrainingSettings.from_yaml(str(ROOT / arm))


@pytest.mark.parametrize("arm", ARMS)
def test_a_declared_discriminator_is_trained_adversarially(arm: str) -> None:
    settings = _load(arm)
    kwargs = settings.model.model_kwargs or {}
    declares_gan = "discriminator_component" in kwargs or settings.model.model_dump().get(
        "discriminator_component"
    )
    if declares_gan or kwargs.get("enable_pcgrad"):
        assert settings.losses.gan.enable_adversarial, (
            "a discriminator or enable_pcgrad is declared but losses.gan.enable_adversarial is "
            "off: neither would ever act"
        )


@pytest.mark.parametrize("arm", ARMS)
def test_a_vae_arm_has_a_prior_term(arm: str) -> None:
    settings = _load(arm)
    if (settings.model.model_kwargs or {}).get("use_vae"):
        assert settings.losses.latent.lambda_kl > 0


@pytest.mark.parametrize("arm", ARMS)
def test_the_discriminator_is_declared_where_it_is_built(arm: str) -> None:
    # Under model_kwargs it is swallowed by the model and no critic is built (job 17885803).
    settings = _load(arm)
    assert settings.model.discriminator_component is not None
    assert "discriminator_component" not in (settings.model.model_kwargs or {})


@pytest.mark.parametrize("arm", ARMS)
def test_the_declared_loss_is_the_only_reconstruction_term(arm: str) -> None:
    # The GAN computer's own L1 defaults to weight 1.0 and was added on top of the
    # declared MSE once the discriminator was built (job 17886489, g_adv_loss 0.75 while
    # the adversarial weight was still warm-up gated to 0).
    import torch

    from spectramr.models.losses.computers.unified_gan import UnifiedGANLossComputer

    computer = UnifiedGANLossComputer(_load(arm), torch.device("cpu"))
    assert computer._get_loss_weight("reconstruction") == 0.0


@pytest.mark.parametrize("arm", ARMS)
def test_the_critic_scores_images_not_kspace(arm: str) -> None:
    # Magnitude images min-max scaled to [-1, 1] reach the critic unconverted; a k-space
    # critic would face a dynamic range of several orders of magnitude.
    import torch

    from spectramr.models.losses.computers.unified_gan import UnifiedGANLossComputer

    settings = _load(arm)
    computer = UnifiedGANLossComputer(settings, torch.device("cpu"))
    assert settings.model.input_type == "image"
    assert settings.physics.coil_processing.output.domain == "image"
    assert computer._critic_from_domain == "image"
    assert computer._critic_conversion is None


@pytest.mark.parametrize("arm", ARMS)
def test_the_critic_is_a_sparse_wasserstein_critic(arm: str) -> None:
    gan = _load(arm).losses.gan
    assert gan.gan_loss_type == "wgan-gp" and gan.enable_gradient_penalty
    assert gan.disc_update_interval == 20
    assert gan.lambda_adv == pytest.approx(0.1)
    # Critic terms arrive scaled by lambda_adv, the penalty is not: keep the WGAN-GP ratio 10.
    assert gan.lambda_gp == pytest.approx(10 * gan.lambda_adv)
    disc = _load(arm).model.discriminator_component
    assert disc.kwargs.get("spectral_norm") is False  # the penalty is the Lipschitz constraint


@pytest.mark.parametrize("arm", ARMS)
def test_the_critic_is_balanced_against_the_generator(arm: str) -> None:
    # At lambda_adv 1.0 with one learning rate the critic dominated arm 51 (d_real/d_fake
    # 0.0002/0.0) and cost 2.5-3.7 dB of val PSNR once it switched on (job 17886987).
    settings = _load(arm)
    opt = settings.optimization.optimizer
    assert opt.discriminator_learning_rate == pytest.approx(4 * opt.learning_rate)  # TTUR
    assert settings.losses.gan.lambda_adv <= 0.1


@pytest.mark.parametrize("arm", ARMS)
def test_patch_nce_weight_is_stated(arm: str) -> None:
    raw = (ROOT / arm).read_text()
    assert "lambda_patch_nce" in raw
