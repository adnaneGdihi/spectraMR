"""StarGAN v2 multi-domain FIELD-translation training strategy (MRIxFields2026 baseline).

StarGAN v2 (Choi et al., "StarGAN v2: Diverse Image Synthesis for Multiple
Domains", CVPR 2020) performs *any-to-any* domain translation with a single
generator conditioned on a per-domain **style** vector. For MRIxFields Task 3 the
domains are the five discrete field levels ``{0.1, 1.5, 3, 5, 7} T``; a continuous
Tesla value is snapped to the nearest level via :meth:`_field_to_domain`.

Four networks are trained (Tasks 8-9 supply all four):

* :class:`~spectramr.models.generators.stargan_v2.StarGANv2Generator` ``G(x, s)`` —
  translates image ``x`` under style ``s``.
* :class:`~spectramr.models.generators.stargan_v2.MappingNetwork` ``F(z, y)`` — maps
  a latent ``z`` + target domain ``y`` to a style vector (the *diverse* branch).
* :class:`~spectramr.models.generators.stargan_v2.StyleEncoder` ``E(x, y)`` —
  extracts the style of a reference image (the *reference-guided* branch + style
  reconstruction + cycle style).
* :class:`~spectramr.models.discriminators.stargan_v2_discriminator.StarGANv2Discriminator`
  ``D(x, y)`` — one real/fake logit per sample for its OWN domain head.

Optimiser wiring (anti-facade, CLAUDE.md pitfall #16):
    StarGAN v2 trains ``{G, F, E}`` on ONE optimiser (the "generator side") and
    ``D`` on a second. We fold the strategy-owned mapping-network + style-encoder
    parameters into the environment's ``opt_g`` (mirroring how CUT folds its NCE
    projection MLP), so *all three* generator-side nets learn together. An
    unoptimised mapping net would silently collapse StarGAN v2 to a plain AdaIN
    denoiser — the exact facade the mechanism-fires + gradient tests guard against.

Loss design (bespoke ``train_step``, mirroring the fixed Task-6/7 patterns):
    * **G-step** (``_compute_losses_impl``): adversarial (``gan_lsgan`` on
      ``D(G(x, s1), y_trg)``) + style-reconstruction (``||E(G(x, s1), y_trg) -
      s1||_1``) + style-diversification (``-||G(x, s1) - G(x, s2)||_1``, MAXIMISED
      hence the negative sign) + cycle (``||G(G(x, s1), E(x, y_src)) - x||_1``).
      There is **no** paired pixel-L1 between the fake and the real target — the
      only reconstruction anchor is the CYCLE back to the source.
    * **D-step** (in the bespoke ``train_step``): ``gan_lsgan`` on the real
      ``D(target, y_trg)`` + the detached fake ``D(G(x, s1).detach(), y_trg)``,
      plus an ``r1`` gradient penalty on the real image (grad flows through ``D``).

The five fixed integration patterns from Tasks 6-7 are honoured: (I1) ``.train()``
on every net at the top of ``train_step``; (I2) AMP-config every strategy-owned net
built after ``base.__init__``'s AMP pass; (I3) route detached losses through
``_handle_anomalies``; (I4) expose PLAIN metric names; (I5) no ``.item()`` /
``.cpu()`` in the loss math.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.training.step_io import accepts_step_io
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.mixins.adversarial import (
    AdversarialMixin,
)

# Import the concrete model / loss classes so their ``@register_*`` decorators run
# (registry lookups elsewhere then resolve independent of two-phase populate
# ordering). MappingNetwork / StyleEncoder are plain importable classes (their
# ``forward(*, y)`` is not the registered-generator contract), so we import them
# directly rather than via ``get_model_class``.
from spectramr.models.discriminators.stargan_v2_discriminator import (
    StarGANv2Discriminator,
)
from spectramr.models.generators.stargan_v2 import (
    MappingNetwork,
    StarGANv2Generator,
    StyleEncoder,
)
from spectramr.models.losses.gan_loss_library import LSGANLoss, R1RegularizationLoss

logger = logging.getLogger(__name__)


class _DomainBoundDiscriminator(nn.Module):
    """Bind a fixed domain label ``y`` to a multi-domain discriminator so it can be
    called single-argument by the library ``r1`` loss.

    :class:`R1RegularizationLoss` calls ``discriminator(real_images)`` and requires
    an :class:`~torch.nn.Module` (an ``isinstance`` gate that a bare lambda would
    fail, silently zeroing R1 — a facade). This wrapper adds **no NEW parameters of
    its own** and is **never added to any optimizer**: the wrapped discriminator's
    parameters already live in ``opt_d``. The wrapper is ephemeral — rebuilt per
    D-step purely to bind ``y_trg`` so the single-argument R1 library call
    ``D(x)`` retains the full gradient path through ``disc``'s weights.
    """

    def __init__(self, disc: nn.Module, y: torch.Tensor) -> None:
        super().__init__()
        self.disc = disc
        self.y = y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.disc(x, self.y)


class StarGANv2TrainingStrategy(BaseTrainingStrategy, AdversarialMixin):
    """Multi-domain (five FIELD levels) StarGAN v2 translation.

    Resolves the primary ``stargan_v2_generator`` / ``stargan_v2_discriminator``
    from the training environment when present (the canonical pipeline output),
    always builds the (unregistered) mapping-network + style-encoder itself, and
    folds their parameters into ``opt_g`` so ``{G, F, E}`` all train.
    """

    #: The five discrete MRIxFields FIELD levels (Tesla). A continuous
    #: ``field_strength`` is snapped to the nearest of these → domain index 0..4.
    _FIELD_LEVELS: tuple[float, ...] = (0.1, 1.5, 3.0, 5.0, 7.0)
    #: Latent-code width fed to the mapping network (StarGAN v2 default 16).
    _latent_dim: int = 16
    #: Style-vector width (StarGAN v2 default 64); overridden if a reused
    #: ``env.generator`` declares a different ``style_dim``.
    _style_dim: int = 64

    def _setup_strategy_specific_components(self) -> None:
        """Build/resolve the four networks (called from the base ``__init__``)."""
        self.setup_models()

    # ------------------------------------------------------------------ #
    # Model construction (reuse env stargan_v2 nets; always build F / E)
    # ------------------------------------------------------------------ #
    def setup_models(self) -> None:
        """Build ``generator`` / ``mapping`` / ``style_encoder`` / ``discriminator``
        (idempotent).

        ``generator`` / ``discriminator`` reuse the env-built primary nets when
        present; the mapping-network + style-encoder are always strategy-owned (they
        are not registered models). Reused-generator ``style_dim`` / ``img_channels``
        are honoured so ``F`` / ``E`` emit a compatible style vector.
        """
        model_cfg = getattr(self.config, "model", None)
        img_channels = int(getattr(model_cfg, "in_channels", 1) or 1)
        num_domains = len(self._FIELD_LEVELS)

        env = getattr(self, "env", None)
        primary_gen = getattr(env, "generator", None) if env is not None else None
        primary_disc = getattr(env, "discriminator", None) if env is not None else None

        if isinstance(primary_gen, StarGANv2Generator):
            self.generator: nn.Module = primary_gen
            style_dim = int(primary_gen.style_dim)
            img_channels = int(primary_gen.img_channels)
        else:
            style_dim = int(self._style_dim)
            self.generator = StarGANv2Generator(img_channels=img_channels, style_dim=style_dim)

        self.style_dim = style_dim
        self.latent_dim = int(self._latent_dim)
        self.num_domains = num_domains

        self.mapping: nn.Module = MappingNetwork(
            latent_dim=self.latent_dim, style_dim=style_dim, num_domains=num_domains
        )
        self.style_encoder: nn.Module = StyleEncoder(
            img_channels=img_channels, style_dim=style_dim, num_domains=num_domains
        )

        if isinstance(primary_disc, StarGANv2Discriminator):
            self.discriminator: nn.Module = primary_disc
        else:
            self.discriminator = StarGANv2Discriminator(
                img_channels=img_channels, num_domains=num_domains
            )

        # LSGAN adversarial op (parameter-free) and the R1 gradient penalty. R1's
        # weight is the config SSOT ``losses.gan.lambda_r1`` (validated ge=0). The
        # gate ``losses.gan.enable_r1`` must be True for R1 to fire — an unwired
        # ``enable_r1: false`` with a non-zero ``lambda_r1`` would otherwise silently
        # apply R1 (CLAUDE.md pitfall #15). Default True via getattr for forward
        # compat with configs that predate the field.
        _gan_cfg_setup = self.config.losses.gan if self.config.losses is not None else None
        self._enable_r1: bool = bool(
            getattr(_gan_cfg_setup, "enable_r1", True) if _gan_cfg_setup is not None else True
        )
        self._adv = LSGANLoss()
        self._r1 = R1RegularizationLoss(weight=self._lambda_r1())

        device = getattr(self, "device", torch.device("cpu"))
        for module in (
            self.generator,
            self.mapping,
            self.style_encoder,
            self.discriminator,
            self._adv,
            self._r1,
        ):
            module.to(device)

        # I2: ``base.__init__`` AMP-configures ONLY ``env.generator`` /
        # ``env.discriminator`` (base.py:434-436) BEFORE this hook runs
        # (base.py:492). The strategy-owned mapping / style-encoder (and any
        # strategy-built generator / discriminator) must be AMP-configured here too,
        # or their grads won't scale under a real AMP GradScaler. Guarded for the
        # ``object.__new__`` unit-test path (no ``amp_helper``).
        if hasattr(self, "amp_helper") and self.amp_helper is not None:
            if self.generator is not primary_gen:
                self.amp_helper.configure_model_for_amp(self.generator)
            self.amp_helper.configure_model_for_amp(self.mapping)
            self.amp_helper.configure_model_for_amp(self.style_encoder)
            if self.discriminator is not primary_disc:
                self.amp_helper.configure_model_for_amp(self.discriminator)

        self._register_owned_params()

    def _register_owned_params(self) -> None:
        """Fold the generator-side owned params (mapping + style-encoder, plus the
        generator/discriminator when the strategy built them itself) into the env
        optimizers so ``{G, F, E}`` train on ``opt_g`` and ``D`` on ``opt_d``.

        No-op when the optimizers are absent (unit-test / scripting path). Idempotent:
        a param already present in a group is not re-added (PyTorch forbids a param in
        two groups). Uses :meth:`_sync_schedulers_after_param_group_addition` so a
        scheduler built before this addition does not raise on its next ``step()``.
        """
        env = getattr(self, "env", None)
        if env is None:
            return
        opt_cfg = getattr(self.config, "optimization", None)
        lr = float(getattr(getattr(opt_cfg, "optimizer", None), "learning_rate", 2e-4) or 2e-4)

        opt_g = getattr(env, "opt_g", None)
        if opt_g is not None:
            # Mapping + style-encoder are ALWAYS strategy-owned → always folded in.
            g_side = list(self.mapping.parameters()) + list(self.style_encoder.parameters())
            # A strategy-built generator (env.generator was absent / non-stargan) is
            # not yet in opt_g — prepend it so G trains at all.
            if self.generator is not getattr(env, "generator", None):
                g_side = list(self.generator.parameters()) + g_side
            new_params = self._params_not_in_optimizer(opt_g, g_side)
            if new_params:
                opt_g.add_param_group({"params": new_params})
                self._sync_schedulers_after_param_group_addition(lr)

        opt_d = getattr(env, "opt_d", None)
        if opt_d is not None and self.discriminator is not getattr(env, "discriminator", None):
            disc_params = list(self.discriminator.parameters())
            new_disc = self._params_not_in_optimizer(opt_d, disc_params)
            if new_disc:
                opt_d.add_param_group({"params": new_disc})
                self._sync_schedulers_after_param_group_addition(lr)

    @staticmethod
    def _params_not_in_optimizer(
        optimizer: torch.optim.Optimizer, params: list[nn.Parameter]
    ) -> list[nn.Parameter]:
        """Return the subset of ``params`` not already registered on ``optimizer``."""
        existing = {id(p) for group in optimizer.param_groups for p in group["params"]}
        return [p for p in params if id(p) not in existing]

    # ------------------------------------------------------------------ #
    # Field → domain snapping and batch / weight resolution
    # ------------------------------------------------------------------ #
    def _field_to_domain(self, field_strength: torch.Tensor) -> torch.Tensor:
        """Snap continuous Tesla values to the nearest discrete FIELD domain index.

        Vectorised nearest-neighbour against :data:`_FIELD_LEVELS`; returns a
        ``long`` tensor of shape ``[B]`` with values in ``{0, .., num_domains-1}``.
        Used for both the source (``field_strength``) and target
        (``field_strength_target``) domains.
        """
        field = (
            field_strength if torch.is_tensor(field_strength) else torch.as_tensor(field_strength)
        )
        field = field.float().reshape(-1)
        levels = torch.tensor(self._FIELD_LEVELS, device=field.device, dtype=torch.float32)
        return (field[:, None] - levels[None, :]).abs().argmin(dim=1)

    def _resolve_batch(self, *candidates: Any) -> dict[str, Any]:
        """Return the first mapping-like batch among ``candidates`` (dict /
        TrainingBatch). Raises (no silent fallback, #9) when none is present."""
        for candidate in candidates:
            if (
                candidate is not None
                and not torch.is_tensor(candidate)
                and hasattr(candidate, "get")
            ):
                return candidate
        raise ValueError(
            "StarGANv2TrainingStrategy requires a mapping batch with keys "
            "'input', 'target', 'field_strength', 'field_strength_target'. "
            f"Got candidates of types {[type(c).__name__ for c in candidates]}."
        )

    def _weights(self) -> tuple[float, float, float]:
        """Read ``(lambda_style, lambda_diversity, lambda_cycle)`` from the config
        SSOT ``losses.gan``. Raises when the block is absent."""
        gan_cfg = self.config.losses.gan if self.config.losses is not None else None
        if gan_cfg is None:
            raise ValueError(
                "StarGANv2TrainingStrategy requires a `losses.gan:` block (the home "
                "of lambda_style / lambda_diversity / lambda_cycle / lambda_r1)."
            )
        return (
            float(gan_cfg.lambda_style),
            float(gan_cfg.lambda_diversity),
            float(gan_cfg.lambda_cycle),
        )

    def _lambda_r1(self) -> float:
        """Read the R1 gradient-penalty weight ``losses.gan.lambda_r1`` (SSOT)."""
        gan_cfg = self.config.losses.gan if self.config.losses is not None else None
        if gan_cfg is None:
            raise ValueError(
                "StarGANv2TrainingStrategy requires a `losses.gan:` block (the home of lambda_r1)."
            )
        return float(gan_cfg.lambda_r1)

    def _sample_latents(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample a ``[B, latent_dim]`` standard-normal latent for the mapping net."""
        return torch.randn(batch_size, self.latent_dim, device=device)

    # ------------------------------------------------------------------ #
    # Loss computation (generator side — the required extension point)
    # ------------------------------------------------------------------ #
    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute the StarGAN-v2 generator-side losses for one batch.

        Returns ``{g_total_loss, adv_g, style, diversity, cycle}`` where ``style`` /
        ``diversity`` / ``cycle`` are already lambda-weighted (so
        ``g_total = adv_g + style + diversity + cycle``). The discriminator loss is
        owned by the D-closure, not returned here. No paired pixel-L1 to the target.
        """
        batch = self._resolve_batch(kwargs.get("batch"), input_batch)
        lambda_style, lambda_diversity, lambda_cycle = self._weights()

        device = getattr(self, "device", torch.device("cpu"))
        x = batch["input"].to(device)
        y_trg = self._field_to_domain(batch["field_strength_target"]).to(device)
        y_src = self._field_to_domain(batch["field_strength"]).to(device)
        batch_size = x.shape[0]

        # Two latents → two styles → two fakes (the diverse branch).
        z1 = self._sample_latents(batch_size, device)
        z2 = self._sample_latents(batch_size, device)
        s1 = self.mapping(z1, y_trg)
        s2 = self.mapping(z2, y_trg)
        x_fake = self.generator(x, s1)
        x_fake2 = self.generator(x, s2)

        # Adversarial: push D's score on the fake (at the TARGET domain head) → real.
        adv_g = self._adv.compute_generator_loss(self.discriminator(x_fake, y_trg))

        # Style reconstruction: the style encoder must recover the injected style.
        style = F.l1_loss(self.style_encoder(x_fake, y_trg), s1) * lambda_style

        # Style diversification: MAXIMISE the L1 gap between the two fakes (negative
        # sign → contributes negatively to the minimised total). A constant
        # lambda_diversity is used for this baseline; the paper decays it over
        # training — that schedule is intentionally NOT built here.
        diversity = -(x_fake - x_fake2).abs().mean() * lambda_diversity

        # Cycle: re-encode the SOURCE style from the real source and translate the
        # fake back — the only reconstruction anchor (no paired L1 to target).
        s_src = self.style_encoder(x, y_src)
        x_cycle = self.generator(x_fake, s_src)
        cycle = F.l1_loss(x_cycle, x) * lambda_cycle

        g_total = adv_g + style + diversity + cycle
        return {
            "g_total_loss": g_total,
            "adv_g": adv_g,
            "style": style,
            "diversity": diversity,
            "cycle": cycle,
        }

    # ------------------------------------------------------------------ #
    # Two-optimiser alternating step (D on discriminator, G on G+F+E)
    # ------------------------------------------------------------------ #
    @accepts_step_io
    def train_step(
        self,
        batch: Any,
        epoch: int,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        iteration: int = 0,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return the discriminator + generator step closures for the Trainer."""
        # I1: restore train mode on ALL four nets before stepping. This bespoke
        # ``train_step`` overrides the base one; ``_validation_forward`` leaves the
        # generator in ``eval()``, so without this BatchNorm/Dropout-style layers
        # would run in inference mode during training after the first validation.
        for _m in (
            self.generator,
            self.mapping,
            self.style_encoder,
            self.discriminator,
        ):
            _m.train()

        resolved = self._resolve_batch(kwargs.get("batch"), input_batch, batch)
        device = getattr(self, "device", torch.device("cpu"))
        x = resolved["input"].to(device)
        real_trg = resolved["target"].to(device)
        y_trg = self._field_to_domain(resolved["field_strength_target"]).to(device)

        self._last_step_metrics: dict[str, torch.Tensor] = {}
        step_configs: list[dict[str, Any]] = []
        env = self.env

        def d_closure() -> torch.Tensor:
            self.discriminator.requires_grad_(True)
            # Fresh fake for the D-step (detached — no generator gradient path).
            with torch.no_grad():
                z = self._sample_latents(x.shape[0], device)
                s = self.mapping(z, y_trg)
                x_fake = self.generator(x, s)
            real_logits = self.discriminator(real_trg, y_trg)
            fake_logits = self.discriminator(x_fake, y_trg)
            real_loss, fake_loss = self._adv.compute_discriminator_loss(real_logits, fake_logits)
            d_adv = real_loss + fake_loss
            # R1 gradient penalty on the REAL target image at its target-domain head
            # (grad flows through D via the domain-bound wrapper). Gated on
            # ``config.losses.gan.enable_r1`` (pitfall #15 — unwired knob guard).
            if self._enable_r1:
                r1 = self._r1(_DomainBoundDiscriminator(self.discriminator, y_trg), real_trg)
            else:
                r1 = torch.zeros((), device=device)
            d_total = d_adv + r1
            with torch.no_grad():
                self._last_step_metrics["d_total_loss"] = d_total.detach()
                self._last_step_metrics["d_adv"] = d_adv.detach()
                self._last_step_metrics["r1"] = r1.detach()
            return d_total

        opt_d = getattr(env, "opt_d", None)
        if opt_d is not None:
            step_configs.append(
                {
                    "optimizer": opt_d,
                    "closure": d_closure,
                    "model": self.discriminator,
                    "name": "discriminator",
                }
            )

        def g_closure() -> torch.Tensor:
            # Freeze the discriminator for the generator step (gradients still flow
            # to G/F/E THROUGH the frozen discriminator activations).
            self.discriminator.requires_grad_(False)
            # #1190: direct ``_compute_losses_impl`` call, so the wrapper that
            # emits the ``model_output`` snapshot is bypassed -- arm it here.
            #
            # ``self.generator``, NOT ``env.generator``: ``_build_models`` reuses
            # the env primary only when it is a ``StarGANv2Generator`` and builds
            # its own otherwise, so hooking the env module would install the hook
            # on something this strategy never forwards.
            #
            # The impl takes the whole ``resolved`` batch, so the snapshot's
            # input/target rows are named from it -- ``x`` is the source image and
            # ``real_trg`` the target-domain reference, the same two tensors
            # ``d_closure`` scores.
            #
            # Armed TIGHTLY: ``d_closure`` above runs ``self.generator(x, s)``
            # under ``no_grad`` to make its fake, which would otherwise overwrite
            # the stash.
            with self._capture_model_output(
                module=self.generator,
                input_batch=x,
                target_batch=real_trg,
                step=iteration,
            ):
                losses = self._compute_losses_impl(batch=resolved, epoch=epoch)
            g_total = losses["g_total_loss"]
            # I3 + I4: route the detached G-losses through the base NaN/Inf anomaly
            # guard and expose PLAIN metric names (style / diversity / cycle / adv_g /
            # g_total_loss). ``update`` preserves ``d_total_loss`` from the D-closure.
            detached = {k: v.detach() for k, v in losses.items()}
            self._last_step_metrics.update(self._handle_anomalies(detached, {}, epoch))
            return g_total

        step_configs.append(
            {
                "optimizer": getattr(env, "opt_g", None),
                "closure": g_closure,
                "model": self.generator,
                "name": "generator",
            }
        )
        return step_configs

    def _backward_and_step(
        self, losses: dict[str, torch.Tensor], epoch: int, step: int = 0
    ) -> None:
        """No-op: the custom ``train_step`` owns backward/step (mirrors GAN)."""

    # ------------------------------------------------------------------ #
    # Validation — render at the sample's target field
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _validation_forward(
        self,
        input_batch: torch.Tensor,
        *args: Any,
        reference: torch.Tensor | None = None,
        batch: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Translate ``input`` at its target FIELD.

        Reference-guided when a ``reference`` image (the real target-field sample) is
        available: ``G(x, E(reference, y_trg))``. Otherwise latent-guided with a
        deterministic zero latent: ``G(x, F(0, y_trg))``. The target domain is read
        from the batch's ``field_strength_target`` when present, else defaults to the
        highest field (7 T HF target). ``*args`` absorbs a positional ``batch_context``
        from the base ``validation_step`` contract.
        """
        self.generator.eval()
        x = input_batch
        device = x.device
        batch_size = x.shape[0]

        if batch is None and args:
            # Base ModelValidationMixin passes batch_context positionally; only a
            # mapping with the field key is useful here.
            first = args[0]
            if first is not None and not torch.is_tensor(first) and hasattr(first, "get"):
                batch = first

        fst = None
        if batch is not None and hasattr(batch, "get"):
            fst = batch.get("field_strength_target")
            if reference is None:
                reference = batch.get("target")

        if fst is not None:
            y_trg = self._field_to_domain(fst).to(device)
        else:
            y_trg = torch.full((batch_size,), self.num_domains - 1, dtype=torch.long, device=device)

        if reference is not None:
            style = self.style_encoder(reference.to(device), y_trg)
        else:
            z = torch.zeros(batch_size, self.latent_dim, device=device)
            style = self.mapping(z, y_trg)
        return self.generator(x, style)

    @torch.no_grad()
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Translate ``input`` at its target field (reference-guided by ``target``
        when available) and score against ``target`` when a metrics computer is
        present (evaluation only — NOT a training pixel loss)."""
        pred = self._validation_forward(input_batch, reference=target_batch, **kwargs)
        computer = getattr(self, "validation_metrics_computer", None)
        if computer is not None and target_batch is not None:
            try:
                return dict(computer.compute(pred, target_batch))
            except Exception as exc:  # pragma: no cover - defensive; never break val
                logger.warning("validation metric computation failed: %s", exc)
                return {}
        return {}
