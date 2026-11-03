"""Single-application (w**2) tests for the unified VAE / VQ-VAE loss computers.

Both computers pre-multiplied each *dynamic* loss component by its weight inside
``compute`` and then handed it to ``_stack_components``, which re-applies the same
weight (``base.py`` ``total += weight * loss_val``) — a ``w**2`` double-application.

It was invisible for as long as every dynamic weight resolved to exactly 1.0 (1**2
== 1) or 0.0 (0**2 == 0), which is what every VAE arm in the repo happened to
declare. It becomes visible the moment a real weight is set: the
``ldm_two_stage_ulf_to_hf`` stage-1 arms enable SSIM at ``lambda_ssim: 0.5``, which
the pre-fix code would have silently computed at 0.25.

The equivalent fix already landed in ``UnifiedDiffusionLossComputer``; see
``test_unified_diffusion_reconstruction.py::TestSingleApplication``. These tests
pin the VAE side so it cannot regress independently.
"""

from types import SimpleNamespace

import pytest
import torch

from spectramr.config.schemas.loss import (
    LossComponentConfig,
    LossConfigSchema,
    ReconstructionLossesConfig,
)
from spectramr.models.losses.computers.unified_vae import (
    UnifiedVAELossComputer,
    UnifiedVQVAELossComputer,
)


class _ConstantLoss(torch.nn.Module):
    """A (pred, target) loss that always returns 1.0, so any weighting is legible."""

    def forward(self, pred, target, **kwargs):
        return torch.tensor(1.0, requires_grad=True)


def _config(**recon_kwargs) -> SimpleNamespace:
    """A minimal frozen-config stand-in carrying only the loss block."""
    return SimpleNamespace(
        losses=LossConfigSchema(
            output_domain="image",
            reconstruction=ReconstructionLossesConfig(lambda_l2=1.0, **recon_kwargs),
        )
    )


class TestVAESingleApplication:
    """Dynamic loss components must be weighted exactly once, not squared."""

    def test_dynamic_loss_weight_applied_exactly_once(self):
        # warmup_iterations=0 so the spatial-loss warmup gate does not zero ssim.
        config = _config(lambda_ssim=0.5, warmup_iterations=0)
        comp = UnifiedVAELossComputer(config=config, device=torch.device("cpu"))

        pred = torch.zeros(1, 1, 4, 4, requires_grad=True)
        target = torch.zeros(1, 1, 4, 4)

        out = comp.compute(pred, target, losses_dict={"ssim": _ConstantLoss()})

        # The stored component is RAW (1.0), never pre-multiplied.
        assert out.components["ssim"].item() == pytest.approx(1.0)

        # reconstruction (L1 on zeros) contributes 0 and there is no posterior,
        # so the KL term is absent -> the total IS the weighted ssim component.
        # Exactly once => 0.5. The pre-fix w**2 bug would give 0.25.
        assert out.total.item() == pytest.approx(0.5, abs=1e-6)
        assert out.total.item() != pytest.approx(0.25, abs=1e-6)

    def test_unit_weight_is_unchanged_by_the_fix(self):
        """The regression that hid the bug: at w == 1.0, w and w**2 agree."""
        config = _config(lambda_ssim=1.0, warmup_iterations=0)
        comp = UnifiedVAELossComputer(config=config, device=torch.device("cpu"))

        out = comp.compute(
            torch.zeros(1, 1, 4, 4, requires_grad=True),
            torch.zeros(1, 1, 4, 4),
            losses_dict={"ssim": _ConstantLoss()},
        )
        assert out.total.item() == pytest.approx(1.0, abs=1e-6)


class TestVQVAESingleApplication:
    """The VQ-VAE computer carried the identical pre-multiply."""

    def test_dynamic_loss_weight_applied_exactly_once(self):
        config = _config(lambda_ssim=0.5, warmup_iterations=0)
        comp = UnifiedVQVAELossComputer(config=config, device=torch.device("cpu"))

        out = comp.compute(
            torch.zeros(1, 1, 4, 4, requires_grad=True),
            torch.zeros(1, 1, 4, 4),
            losses_dict={"ssim": _ConstantLoss()},
        )

        assert out.components["ssim"].item() == pytest.approx(1.0)
        assert out.total.item() == pytest.approx(0.5, abs=1e-6)


class TestKLAnnealNamingSchism:
    """The KL warm-up never fired: config spelling != resolver spelling.

    Every VAE arm in ``experiments/`` was authored with ``enable_kl_annealing``
    / ``kl_anneal_start`` / ``kl_anneal_end`` under ``training.vae``, while
    ``_get_loss_weight`` read ``anneal_kl_beta`` / ``kl_beta_start`` /
    ``kl_beta_end``. Two independent failures stacked:

    1. ``training.vae`` was never DECLARED on the training schema, so
       ``extra='allow'`` admitted it as a raw ``dict`` and every ``getattr``
       against it returned ``None``.
    2. Even reached as a mapping, the key NAMES did not match.

    Net effect: ``anneal_kl_beta`` resolved False and beta was pinned at
    ``kl_beta_end`` (1.0) from step 0 — full KL pressure on an untrained
    encoder, which is the posterior-collapse setup warm-up exists to prevent.
    """

    @staticmethod
    def _comp(training) -> UnifiedVAELossComputer:
        config = SimpleNamespace(
            losses=LossConfigSchema(
                output_domain="image",
                reconstruction=ReconstructionLossesConfig(lambda_l2=1.0),
            ),
            training=training,
        )
        return UnifiedVAELossComputer(config=config, device=torch.device("cpu"))

    def test_yaml_spelling_on_a_raw_dict_still_anneals(self):
        """The exact shape the 26 arms ship: YAML names, un-validated dict."""
        comp = self._comp(
            SimpleNamespace(
                vae={
                    "enable_kl_annealing": True,
                    "kl_anneal_start": 0.0,
                    "kl_anneal_end": 1.0,
                    "kl_anneal_steps": 10_000,
                },
                latent=None,
            )
        )
        # Pre-fix this returned 1.0 at EVERY iteration (no ramp at all).
        assert comp._get_loss_weight("kl", iteration=0) == pytest.approx(0.0)
        assert comp._get_loss_weight("kl", iteration=5_000) == pytest.approx(0.5)
        assert comp._get_loss_weight("kl", iteration=10_000) == pytest.approx(1.0)
        # Past the schedule it saturates, never overshoots.
        assert comp._get_loss_weight("kl", iteration=99_000) == pytest.approx(1.0)

    def test_validated_schema_block_anneals_identically(self):
        """Declared `training.vae` + AliasChoices must agree with the dict path."""
        from spectramr.config.schemas.training.base import (
            TrainingStrategyConfigSchema,
        )

        training = TrainingStrategyConfigSchema(
            training_mode="vae",
            vae={
                "enable_kl_annealing": True,
                "kl_anneal_start": 0.0,
                "kl_anneal_end": 1.0,
                "kl_anneal_steps": 10_000,
            },
        )
        # The block validates into a model, not a dict -- that IS the fix.
        assert not isinstance(training.vae, dict)
        assert training.vae.anneal_kl_beta is True
        assert training.vae.kl_beta_end == pytest.approx(1.0)

        comp = self._comp(training)
        assert comp._get_loss_weight("kl", iteration=5_000) == pytest.approx(0.5)

    def test_canonical_spelling_is_not_broken_by_the_aliases(self):
        """Configs written against the resolver's names keep working."""
        comp = self._comp(
            SimpleNamespace(
                vae={
                    "anneal_kl_beta": True,
                    "kl_beta_start": 0.0,
                    "kl_beta_end": 0.5,
                    "kl_anneal_steps": 1_000,
                },
                latent=None,
            )
        )
        assert comp._get_loss_weight("kl", iteration=500) == pytest.approx(0.25)

    def test_annealing_disabled_pins_beta_at_the_end_value(self):
        """21 of the 26 arms set enable_kl_annealing: false -- unchanged."""
        comp = self._comp(
            SimpleNamespace(
                vae={"enable_kl_annealing": False, "kl_anneal_end": 1.0},
                latent=None,
            )
        )
        assert comp._get_loss_weight("kl", iteration=0) == pytest.approx(1.0)
        assert comp._get_loss_weight("kl", iteration=50_000) == pytest.approx(1.0)

    def test_vae_block_takes_precedence_over_legacy_latent(self):
        comp = self._comp(
            SimpleNamespace(
                vae={"kl_anneal_end": 0.25},
                latent=SimpleNamespace(kl_beta_end=0.75, anneal_kl_beta=None),
            )
        )
        assert comp._get_loss_weight("kl", iteration=0) == pytest.approx(0.25)

    def test_vae_bounds_fall_through_to_latent_for_unset_keys(self):
        """Declaring a schema must not let its DEFAULTS shadow the next block.

        The ldm_two_stage stage-1 arms split the schedule across both blocks:
        the beta bounds sit under ``training.vae`` and the anneal flag/steps
        under ``training.latent``. Once ``vae`` became a declared schema it
        gained defaults for the keys it does NOT set (``anneal_kl_beta=False``,
        ``kl_anneal_steps=10000``) -- and a resolver that treats "attribute
        exists" as "was configured" reads those defaults instead of falling
        through, silently switching annealing back off. Only an explicitly-set
        field may win. See ``test_vae_kl_weight_wiring.py`` for the arm-level
        guard this protects.
        """
        from spectramr.config.schemas.training.base import (
            TrainingStrategyConfigSchema,
        )

        training = TrainingStrategyConfigSchema(
            training_mode="vae",
            # bounds only -- deliberately silent on anneal flag + steps
            vae={"kl_beta_start": 1.0e-6, "kl_beta_end": 1.0e-4},
            latent={"anneal_kl_beta": True, "kl_anneal_steps": 5_000},
        )
        comp = self._comp(training)

        assert comp._get_loss_weight("kl", iteration=0) == pytest.approx(1e-6)
        assert comp._get_loss_weight("kl", iteration=5_000) == pytest.approx(1e-4)
        # Midpoint proves the ramp advances rather than sitting at either bound.
        mid = comp._get_loss_weight("kl", iteration=2_500)
        assert 1e-6 < mid < 1e-4


def test_vae_reconstruction_is_the_declared_mse_not_an_l1() -> None:
    """The declared reconstruction term is the one trained, at its declared weight (#2280)."""
    import torch
    from torch import nn

    from spectramr.config.schemas.loss import LossComponentConfig, LossConfigSchema
    from spectramr.models.losses.computers.unified_vae import UnifiedVAELossComputer
    from spectramr.models.losses.weights import build_loss_weight_table

    cfg = LossConfigSchema(
        output_domain="image",
        image_losses=[LossComponentConfig(name="mse", weight=2.0)],
    )
    comp = UnifiedVAELossComputer(config=SimpleNamespace(losses=cfg))
    comp.reconstruction_name, comp.reconstruction_loss_fn = "l2", nn.MSELoss()
    comp._loss_weight_table = build_loss_weight_table(cfg)
    out = comp.compute(torch.zeros(2, 4), torch.ones(2, 4))
    assert out.components["reconstruction"].item() == pytest.approx(1.0)
    assert out.total.item() == pytest.approx(2.0)


def test_vae_refuses_a_declared_reconstruction_weight_with_no_declared_loss() -> None:
    from types import SimpleNamespace

    from spectramr.config.schemas.loss import LossConfigSchema, ReconstructionLossesConfig
    from spectramr.domain.exceptions import ConfigurationError
    from spectramr.models.losses.computers.unified_vae import (
        _refuse_undeclared_reconstruction,
    )

    cfg = LossConfigSchema(
        reconstruction=ReconstructionLossesConfig(lambda_reconstruction=1.0)
    )
    with pytest.raises(ConfigurationError, match="no reconstruction loss is declared"):
        _refuse_undeclared_reconstruction(SimpleNamespace(losses=cfg))


def test_vq_loss_is_included_at_its_declared_weight() -> None:
    import torch

    from spectramr.config.schemas.loss import LatentLossesConfig, LossConfigSchema
    from spectramr.models.losses.computers.unified_vae import UnifiedVQVAELossComputer
    from spectramr.models.losses.weights import build_loss_weight_table

    cfg = LossConfigSchema(
        output_domain="image",
        latent=LatentLossesConfig(lambda_vq=0.5),
        image_losses=[LossComponentConfig(name="mse", weight=1.0)],
    )
    comp = UnifiedVQVAELossComputer(config=SimpleNamespace(losses=cfg))
    comp._loss_weight_table = build_loss_weight_table(cfg)
    out = comp.compute(torch.zeros(2, 4), torch.zeros(2, 4), vq_loss=torch.tensor(2.0))
    assert "vq_loss" in out.components
    assert out.total.item() == pytest.approx(1.0)


def _arm(rel: str):
    from pathlib import Path

    from spectramr.config.settings import TrainingSettings

    root = Path(__file__).resolve().parents[5]
    return TrainingSettings.from_yaml(str(root / "experiments/inprogress" / rel))


@pytest.mark.parametrize(
    "rel",
    [
        "ldm_two_stage_ulf_to_hf/stage1_vae_hf.yaml",
        "ldm_two_stage_ulf_to_hf/stage1_sparse_vae_slab.yaml",
        "ldm_two_stage_ulf_to_hf/stage1_slat_slab_to_volume.yaml",
    ],
)
def test_stage1_vae_arm_reconstructs_with_its_declared_l1(rel: str) -> None:
    comp = UnifiedVAELossComputer(config=_arm(rel), device=torch.device("cpu"))
    assert comp.reconstruction_name == "l1"


@pytest.mark.parametrize(
    "rel",
    [
        "vae_latent/vqvae2_brain.yaml",
        "vae_latent/experiment_54_hierarchical_vq_vae.yaml",
    ],
)
def test_declared_vq_term_fires_on_a_real_vq_arm(rel: str) -> None:
    comp = UnifiedVQVAELossComputer(config=_arm(rel), device=torch.device("cpu"))
    out = comp.compute(torch.zeros(2, 4), torch.zeros(2, 4), vq_loss=torch.tensor(2.0))
    assert out.components["vq_loss"].item() == pytest.approx(2.0)


@pytest.mark.parametrize("cls", [UnifiedVAELossComputer, UnifiedVQVAELossComputer])
def test_no_declared_reconstruction_raises(cls) -> None:
    """Decision 16: a VAE/VQ-VAE arm with no reconstruction term is refused, not latent-only."""
    from spectramr.domain.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="no reconstruction loss is declared"):
        cls(config=None, device=torch.device("cpu"))
    bare = SimpleNamespace(losses=LossConfigSchema(output_domain="image"))
    with pytest.raises(ConfigurationError, match="no reconstruction loss is declared"):
        cls(config=bare, device=torch.device("cpu"))
    off = SimpleNamespace(
        losses=LossConfigSchema(
            output_domain="image",
            image_losses=[LossComponentConfig(name="mse", weight=0.0)],
        )
    )
    with pytest.raises(ConfigurationError, match="no reconstruction loss is declared"):
        cls(config=off, device=torch.device("cpu"))
