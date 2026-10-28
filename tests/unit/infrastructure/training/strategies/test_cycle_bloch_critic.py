"""CycleBloch critic resolution: losses.gan -> adversarial loss spec."""

import pytest


def test_resolve_critic_absent_block_is_lsgan():
    from spectramr.infrastructure.training.strategies.cycle_bloch_critic import resolve_critic

    spec = resolve_critic(None)
    assert (spec.name, spec.loss_name, spec.d_scale, spec.uses_gp) == (
        "lsgan",
        "gan_lsgan",
        0.5,
        False,
    )


def test_resolve_critic_reads_the_gradient_penalty_for_wgan_gp():
    from spectramr.config.schemas.loss import GANLossesConfig
    from spectramr.infrastructure.training.strategies.cycle_bloch_critic import resolve_critic

    spec = resolve_critic(
        GANLossesConfig(gan_loss_type="wgan-gp", enable_gradient_penalty=True, lambda_gp=5.0)
    )
    assert (spec.loss_name, spec.uses_gp, spec.lambda_gp) == ("gan_wgan", True, 5.0)
    with pytest.raises(ValueError, match="not implemented"):
        resolve_critic(GANLossesConfig(gan_loss_type="hinge"))
