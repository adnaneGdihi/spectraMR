"""Training-CSV columns of :class:`DisentangledTrainingStrategy`.

The header is built once from the config, and a stamped key it does not name is measured
and discarded. The 2026-10-10 sabine smoke (job 17885505) discarded ``total_loss``,
``recon``, ``patch_nce`` and ``style_consistency`` on all three disentangled arms, and on
the GAN arm the discriminator terms under a doubled prefix (``d_d_loss_real``) while the
header's ``d_real_loss``/``d_fake_loss`` stayed empty.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.training.strategies.disentangled_metric_keys import (
    declared_step_keys,
    discriminator_column,
    generator_column,
)
from spectramr.models.losses.computers import UnifiedDisentangledLossComputer

#: What the smoke discarded, less the executor aliases the loop already explains.
DISCARDED_ON_THE_SMOKE = {"total_loss", "recon", "patch_nce", "style_consistency"}


def test_the_keys_the_smoke_discarded_are_declared() -> None:
    declared = declared_step_keys("mse", adversarial=False)
    assert declared >= DISCARDED_ON_THE_SMOKE


def test_the_declared_reconstruction_alias_is_declared() -> None:
    assert "mse" in declared_step_keys("mse", adversarial=False)
    assert "l1" in declared_step_keys("l1", adversarial=False)


def test_gan_terms_are_declared_only_with_the_adversarial_step() -> None:
    gan = {"g_adv_loss", "d_total_loss", "d_real_loss", "d_fake_loss"}
    assert gan <= declared_step_keys("mse", adversarial=True)
    assert not gan & declared_step_keys("mse", adversarial=False)


def test_the_gan_reconstruction_term_is_declared_only_when_it_is_weighted() -> None:
    assert "g_reconstruction" in declared_step_keys("mse", adversarial=True, gan_recon=True)
    assert "g_reconstruction" not in declared_step_keys("mse", adversarial=True)


def test_the_gradient_penalty_is_declared_only_when_it_runs() -> None:
    assert discriminator_column("gp_loss") == "d_gp_loss"
    assert "d_gp_loss" in declared_step_keys("mse", adversarial=True, gradient_penalty=True)
    assert "d_gp_loss" not in declared_step_keys("mse", adversarial=True)


def test_no_generator_column_is_prefixed_twice() -> None:
    # The 2,000-iteration run (job 17886743) discarded g_g_adv_loss and g_g_total_loss.
    for name in ("g_adv_loss", "g_total_loss", "l1_loss", "reconstruction", "adversarial"):
        column = generator_column(name)
        assert column is None or not column.startswith("g_g_")


def test_the_library_totals_do_not_overwrite_the_step_total() -> None:
    # The closure stamps g_adv_loss from the whole adversarial total itself.
    assert generator_column("g_adv_loss") is None
    assert generator_column("g_total_loss") is None
    assert generator_column("adversarial") is None


def test_every_generator_term_of_the_real_gan_computer_is_declared() -> None:
    """Run the GAN computer an arm builds, past warm-up, and map what it emits."""
    from pathlib import Path

    from spectramr.config.settings import TrainingSettings
    from spectramr.models.losses.computers.unified_gan import UnifiedGANLossComputer

    arm = (
        Path(__file__).resolve().parents[5]
        / "experiments/inprogress/vae_latent/experiment_32a_vae_hf_multicontrast.yaml"
    )
    computer = UnifiedGANLossComputer(TrainingSettings.from_yaml(str(arm)), torch.device("cpu"))
    critic = torch.nn.Conv2d(1, 1, 3, padding=1)
    x = torch.rand(2, 1, 16, 16)
    out = computer.compute_generator_loss(x, x.flip(-1), discriminator=critic, iteration=10**6)
    declared = declared_step_keys("mse", adversarial=True)
    columns = {generator_column(k) for k in out.components} - {None}
    assert columns, "past warm-up the adversarial step must stamp something"
    assert columns <= declared, columns - declared


def test_every_critic_term_of_the_real_gan_computer_is_declared() -> None:
    """The arm's WGAN-GP critic step: every component lands in a declared column."""
    from pathlib import Path

    from spectramr.config.settings import TrainingSettings
    from spectramr.models.losses.computers.unified_gan import UnifiedGANLossComputer

    arm = (
        Path(__file__).resolve().parents[5]
        / "experiments/inprogress/vae_latent/experiment_32a_vae_hf_multicontrast.yaml"
    )
    settings = TrainingSettings.from_yaml(str(arm))
    computer = UnifiedGANLossComputer(settings, torch.device("cpu"))
    critic = torch.nn.Conv2d(1, 1, 3, padding=1)
    x = torch.rand(2, 1, 16, 16)
    out = computer.compute_discriminator_loss(x, x.flip(-1), critic, iteration=10**6)
    declared = declared_step_keys(
        "mse",
        adversarial=True,
        gradient_penalty=settings.losses.gan.enable_gradient_penalty,
    )
    columns = {discriminator_column(k) for k in out.components} - {None}
    assert columns <= declared, columns - declared


def test_library_discriminator_terms_land_in_the_promised_columns() -> None:
    assert discriminator_column("d_loss_real") == "d_real_loss"
    assert discriminator_column("d_loss_fake") == "d_fake_loss"
    assert discriminator_column("d_real") == "d_real_loss"
    assert discriminator_column("d_fake") == "d_fake_loss"


def test_no_discriminator_column_is_prefixed_twice() -> None:
    for name in ("d_loss_real", "d_loss_fake", "d_gp", "r1_penalty", "discriminator"):
        column = discriminator_column(name)
        assert column is None or not column.startswith("d_d_")


def test_the_library_total_does_not_overwrite_the_step_total() -> None:
    # The closure stamps d_total_loss from the full total, R1 included; the library's own
    # d_total_loss omits R1 and must not replace it.
    assert discriminator_column("d_total_loss") is None


def test_every_component_of_a_paired_step_is_declared() -> None:
    computer = UnifiedDisentangledLossComputer(config=None)
    computer.config = type("_Cfg", (), {"losses": _Losses()})()
    computer.losses = {"mse": torch.nn.MSELoss()}
    x_a, x_b = torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
    c, s = torch.rand(2, 4, 8, 8), torch.rand(2, 3)
    out = computer.compute(
        x_a,
        x_a,
        is_paired=True,
        x_a=x_a,
        x_b=x_b,
        x_ab=x_b,
        c_a=c,
        c_a_recon=c + 0.1,
        s_b=s,
        s_b_recon=s + 0.1,
    )
    assert set(out.to_dict()) <= declared_step_keys("mse", adversarial=False) | {
        "recon_a",
        "recon_b",
    }


class _Section:
    loss_type = "mse"
    lambda_recon = 1.0
    image_losses = None


class _Losses:
    reconstruction = _Section()
    latent = None
    image_losses = None
