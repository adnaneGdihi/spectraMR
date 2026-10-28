"""``gan_composite`` takes its four reconstruction weights from the table (#1949).

The sibling of ``test_composite_gan_perceptual_weight_2026_09`` (#1923), and the
last of the same shape. ``LossBuilder._build_composite_gan`` read four more fields
raw — ``recon_config.lambda_l1`` / ``lambda_ssim`` / ``lambda_ms_ssim`` /
``lambda_lpips`` — straight into the ctor, where the weight table could not see them.

``lambda_l1`` is the damaging one. Its schema default is **10.0**, nothing gates it,
and unlike the six other terms in ``CompositeGANLoss.compute_generator_loss`` the
``l1_loss`` term carries no ``if self.lambda_l1 > 0`` guard — so it is added to the
generator objective on every step whatever the weight. Every GAN arm that never
mentioned an L1 therefore trained one at 10.0.

Measured over the 660 ``inprogress`` arms: 25 build the composite, **22 change**.
Twenty go ``10.0 -> 0.0``; two (``mrixfields2026/task3/*``) go ``10.0 -> 1.0`` and
pick up the ``ms_ssim``/``lpips`` they declared and never trained. ``lambda_ssim``
moves on **zero** arms — the three that declare ``ssim`` all set ``enabled: false``,
so the table already answers 0.0 and the raw read agreed by coincidence. That
coincidence is why :class:`TestDeclaredWeightsReachTheConstructor` probes with
**3.7**: a value that is not the schema default of any lambda in the file.

Why the DECLARED weight and not ``resolve_loss_weight``: that helper applies the
warm-up gate against an ``iteration``, and this composite is constructed **once**
with a scalar. ``l1`` is in ``LEGACY_WARMUP_LOSSES``, so resolving it at iteration 0
returns 0.0 — a permanent silent zero, worse than the defect being fixed. That is
not left as prose here; it is asserted, in
:meth:`TestNotResolveLossWeight.test_resolve_loss_weight_would_bake_in_a_warmup_zero`.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

_GAN = {"enable_adversarial": True, "lambda_adv": 1.0, "gan_loss_type": "lsgan"}
_BASE: dict = {
    "image_losses": [{"name": "mse", "weight": 1.0, "enabled": True}],
    "kspace_losses": [],
    "complex_losses": [],
    "policy": {"output_domain": "image"},
    "gan": _GAN,
}


def _declaring(name: str, weight: float, *, enabled: bool = True) -> dict:
    """A GAN arm declaring exactly one extra loss, through a domain list."""
    losses = dict(_BASE)
    losses["image_losses"] = [
        *_BASE["image_losses"],
        {"name": name, "weight": weight, "enabled": enabled},
    ]
    return losses


@pytest.fixture
def build(monkeypatch):
    """Drive the real builder chain; stub only the modules the ctor would download.

    ``CompositeGANLoss.__init__`` constructs ``SSIMLoss`` / ``MSSSIMLoss`` /
    ``LPIPSLoss(net="vgg")`` itself, behind deferred imports, whenever the matching
    lambda is positive — LPIPS pulls torchvision's pretrained VGG. The stubs keep the
    test hermetic and, because they are installed on the modules the ctor imports
    *from*, they also make "was it built at all?" observable.
    """
    import spectramr.infrastructure.training.builders.loss_builder as loss_builder
    import spectramr.models.losses.lpips_loss as lpips_mod
    import spectramr.models.losses.ssim_loss as ssim_mod
    from spectramr.config.schemas.data import DataConfigSchema
    from spectramr.config.schemas.logging import LoggingConfigSchema
    from spectramr.config.schemas.loss import LossConfigSchema
    from spectramr.config.schemas.metrics import MetricsConfigSchema
    from spectramr.config.schemas.model import ModelConfigSchema
    from spectramr.config.schemas.optimization import OptimizationConfigSchema
    from spectramr.config.settings import TrainingSettings

    built: list[str] = []

    def _stub(tag: str):
        def factory(*_a, **_kw):
            built.append(tag)
            return nn.Identity()

        return factory

    monkeypatch.setattr(ssim_mod, "SSIMLoss", _stub("ssim"))
    monkeypatch.setattr(ssim_mod, "MSSSIMLoss", _stub("ms_ssim"))
    monkeypatch.setattr(lpips_mod, "LPIPSLoss", _stub("lpips"))

    def _build(losses_block: dict):
        settings = TrainingSettings(
            model=ModelConfigSchema(),
            data=DataConfigSchema(),
            optimization=OptimizationConfigSchema(),
            logging=LoggingConfigSchema(),
            metrics=MetricsConfigSchema(),
            losses=LossConfigSchema(**losses_block),
        )
        losses = (
            loss_builder.LossBuilder(settings, device=torch.device("cpu"))
            .build_reconstruction_losses()
            .build_adversarial_losses()
            .build_physics_losses()
            .build_regularization_losses()
            .build_diffusion_losses()
            .build_latent_losses()
            .build_ssl_losses()
            .build()
        )
        return losses["adversarial"], built

    return _build


class TestUndeclaredWeightsAreZeroNotTheSchemaDefault:
    """The planted violation: today's defect on 20 corpus arms, committed as a test."""

    def test_a_gan_arm_that_never_mentions_l1_gets_weight_zero(self, build):
        """``10.0`` here is not a neutral default — it is the largest weight in the table."""
        adversarial, _ = build(dict(_BASE))
        assert adversarial.lambda_l1 == 0.0, (
            "an undeclared L1 must not inherit the 10.0 schema default"
        )

    @pytest.mark.parametrize("attr", ["lambda_ssim", "lambda_ms_ssim", "lambda_lpips"])
    def test_the_three_siblings_are_zero_too(self, build, attr):
        adversarial, _ = build(dict(_BASE))
        assert getattr(adversarial, attr) == 0.0

    def test_and_no_ssim_or_lpips_module_is_constructed(self, build):
        """Weighted zero is the weaker claim; never built is the one that matters.

        The one place the construction spy is unambiguous: an arm declaring none of
        the three must construct none of them *anywhere* in the chain, so an empty
        list cannot be satisfied by a bystander. Where a module IS expected the
        assertion moves to the ctor's own slot — see
        :meth:`TestDeclaredWeightsReachTheConstructor.test_a_declared_weight_also_builds_its_module`.
        """
        adversarial, constructed = build(dict(_BASE))
        assert constructed == [], constructed
        assert (adversarial.ssim_loss, adversarial.ms_ssim_loss, adversarial.lpips_loss) == (
            None,
            None,
            None,
        )

    def test_an_arm_with_no_reconstruction_block_still_gets_zero(self, build):
        """The old code's other branch: ``recon_config.lambda_l1 if recon_config else 10.0``.

        With no ``losses.reconstruction`` block the raw read fell through to a
        hard-coded ``10.0`` — the same wrong answer by a second route, which is why
        deleting the raw read had to cover both. ``_BASE`` declares no
        ``reconstruction`` block, so this is that branch.
        """
        adversarial, _ = build(dict(_BASE))
        assert adversarial.lambda_l1 == 0.0


class TestDeclaredWeightsReachTheConstructor:
    """The other half: a declared weight must arrive intact, not be zeroed."""

    @pytest.mark.parametrize(
        ("name", "attr"),
        [
            ("l1", "lambda_l1"),
            ("ssim", "lambda_ssim"),
            ("ms_ssim", "lambda_ms_ssim"),
            ("lpips", "lambda_lpips"),
        ],
    )
    def test_a_declared_weight_arrives_at_the_ctor(self, build, name, attr):
        adversarial, _ = build(_declaring(name, 3.7))
        assert getattr(adversarial, attr) == 3.7, (
            f"{name} declared at 3.7 must reach the ctor at 3.7"
        )

    @pytest.mark.parametrize(
        ("name", "slot"),
        [("ssim", "ssim_loss"), ("ms_ssim", "ms_ssim_loss"), ("lpips", "lpips_loss")],
    )
    def test_a_declared_weight_also_builds_its_module(self, build, name, slot):
        """The two ``task3`` arms do this for the first time after #1949.

        They declare ``ms_ssim: 0.1`` and ``lpips: 0.1`` and were handed 0.0, so the
        ``> 0`` guards in ``CompositeGANLoss.__init__`` skipped both modules and the
        declared losses never trained. Fixing the weight makes the ctor construct
        them — which is a *new* failure surface, since both construct-or-raise.

        Asserted on the ctor's own slot, NOT on the fixture's construction spy: that
        spy sees the whole builder chain, and ``create_loss("ms_ssim")`` builds an
        ``SSIMLoss`` of its own on the way (measured — the spy records
        ``['ssim', 'ms_ssim']`` for an arm declaring only ``ms_ssim``). A membership
        assertion over it would therefore pass for a module the composite never
        built. The spy stays useful in exactly one place: the all-negative case
        below, where an empty list is unambiguous.
        """
        adversarial, _ = build(_declaring(name, 3.7))
        assert getattr(adversarial, slot) is not None, (
            f"{name} declared at 3.7 must make the ctor build its module"
        )

    @pytest.mark.parametrize(
        ("name", "slot"),
        [("ssim", "ssim_loss"), ("ms_ssim", "ms_ssim_loss"), ("lpips", "lpips_loss")],
    )
    def test_a_disabled_declaration_builds_no_module_either(self, build, name, slot):
        """The weight and the module must move together, or the fix is half-applied."""
        adversarial, _ = build(_declaring(name, 3.7, enabled=False))
        assert getattr(adversarial, slot) is None

    @pytest.mark.parametrize(
        ("name", "attr"),
        [
            ("l1", "lambda_l1"),
            ("ssim", "lambda_ssim"),
            ("ms_ssim", "lambda_ms_ssim"),
            ("lpips", "lambda_lpips"),
        ],
    )
    def test_a_disabled_declaration_is_zero(self, build, name, attr):
        """``enabled: false`` is a declaration that resolves to 0.0, not an absence.

        This is the state all three ``ssim``-declaring corpus arms are in, and it is
        why ``lambda_ssim`` changes on zero arms: the table and the raw read agreed
        by coincidence. A test that only covered the undeclared case would not
        distinguish a fix from a builder that ignored the table's ``enabled`` flag.
        """
        adversarial, _ = build(_declaring(name, 3.7, enabled=False))
        assert getattr(adversarial, attr) == 0.0


class TestNotResolveLossWeight:
    """Why the DECLARED weight, and not the warm-up-gated resolver."""

    def test_resolve_loss_weight_would_bake_in_a_warmup_zero(self):
        """The concrete reason ``_declared_weight`` exists as a separate helper.

        ``l1`` is in ``LEGACY_WARMUP_LOSSES``, and the composite is constructed once,
        before step 0. Asking ``resolve_loss_weight`` at iteration 0 — the only
        iteration available at construction time — returns 0.0 and freezes it there
        for the whole run: a declared 3.7 silently never trains.

        Asserted rather than argued, because the two functions agree on every
        non-warm-up loss and a reviewer checking ``ssim`` alone would see no
        difference at all.
        """
        from spectramr.config.schemas.loss import LossConfigSchema
        from spectramr.models.losses.weights import build_loss_weight_table, resolve_loss_weight

        table = build_loss_weight_table(LossConfigSchema(**_declaring("l1", 3.7)))
        assert table.get("l1").warmup_gated, "premise: l1 is warm-up gated"

        assert resolve_loss_weight(table, "l1", iteration=0) == 0.0
        assert resolve_loss_weight(table, "l1", iteration=10_000) == 3.7
        assert table.get("l1").weight == 3.7, "the DECLARED weight is iteration-free"

    def test_the_helper_answers_the_declared_weight(self, build):
        """End to end: the gated loss still reaches the ctor at its declared value."""
        adversarial, _ = build(_declaring("l1", 3.7))
        assert adversarial.lambda_l1 == 3.7
