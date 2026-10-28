"""Training-CSV columns stamped by :class:`DisentangledTrainingStrategy`.

The training loop builds its CSV header once, from the config, and discards any stamped key
the header does not name (``training_loop.resolve_producer_declared_columns``). These keys
are the ones no config walk can name, so the strategy declares them from here.
"""

from __future__ import annotations

__all__ = ["declared_step_keys", "discriminator_column", "generator_column"]

#: Every ``LossOutput.to_dict()`` of the disentangled computer carries ``total_loss`` and the
#: ``recon`` alias. ``style_consistency`` and ``patch_nce`` come from the paired path, which
#: every training batch takes because its ``x_b`` is the batch target.
_ALWAYS = frozenset({"total_loss", "recon", "style_consistency", "patch_nce"})

#: Stamped by the generator and discriminator closures when the adversarial step runs. The
#: ``g_*_loss`` sub-terms are the GAN library's pre-weighted auxiliaries, which it emits on
#: every generator call (zero when their lambda is 0).
_ADVERSARIAL = frozenset(
    {
        "g_adv_loss",
        "d_total_loss",
        "d_real_loss",
        "d_fake_loss",
        "g_l1_loss",
        "g_perceptual_loss",
        "g_feat_match_loss",
        "g_ssim_loss",
        "g_ms_ssim_loss",
        "g_lpips_loss",
    }
)

#: Library and computer totals: the closure stamps ``g_adv_loss`` from the whole adversarial
#: total, so these would duplicate or overwrite it.
_GENERATOR_TOTALS = frozenset({"g_adv_loss", "g_total_loss", "adversarial"})

#: The GAN library's names for the discriminator sub-terms, mapped onto the columns the
#: header promises for an adversarial arm.
_DISCRIMINATOR_COLUMNS = {
    "d_loss_real": "d_real_loss",
    "d_loss_fake": "d_fake_loss",
    "d_real": "d_real_loss",
    "d_fake": "d_fake_loss",
}


def declared_step_keys(
    loss_type: str, *, adversarial: bool, gan_recon: bool = False, gradient_penalty: bool = False
) -> frozenset[str]:
    """The keys one training step stamps that the header cannot derive.

    Args:
        loss_type: ``losses.reconstruction.loss_type``; the computer stamps the
            reconstruction term under this name as well as ``recon``.
        adversarial: whether the discriminator step runs.
        gan_recon: whether the GAN computer's own reconstruction term has a positive
            weight, in which case the generator closure stamps ``g_reconstruction``.
        gradient_penalty: whether ``losses.gan.enable_gradient_penalty`` is on, in
            which case the critic step stamps ``d_gp_loss``.
    """
    keys = set(_ALWAYS) | {loss_type}
    if adversarial:
        keys |= _ADVERSARIAL
        if gan_recon:
            keys.add("g_reconstruction")
        if gradient_penalty:
            keys.add("d_gp_loss")
    return frozenset(keys)


def generator_column(component: str) -> str | None:
    """The CSV column for a generator-side GAN component, or ``None`` to skip it."""
    if component in _GENERATOR_TOTALS:
        return None
    return component if component.startswith("g_") else f"g_{component}"


def discriminator_column(component: str) -> str | None:
    """The CSV column for a discriminator loss component, or ``None`` to skip it.

    The library's ``d_total_loss`` is skipped: the closure stamps that column from the full
    discriminator total, which includes R1, and the library value would overwrite it.
    """
    if component == "d_total_loss":
        return None
    if component in _DISCRIMINATOR_COLUMNS:
        return _DISCRIMINATOR_COLUMNS[component]
    return component if component.startswith("d_") else f"d_{component}"
