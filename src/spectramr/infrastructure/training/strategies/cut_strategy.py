"""Contrastive Unpaired Translation (CUT) training strategy (MRIxFields2026 baseline).

CUT (Park et al., ECCV 2020) is the *single-generator* answer to CycleGAN: ONE
generator ``G`` + ONE PatchGAN discriminator ``D`` + a patch-wise InfoNCE
(``cut_patch_nce``) that ties each translated patch to its co-located source patch
in the generator's own encoder feature space. There is **no** second generator, **no**
cycle-consistency, and **no** pixel-L1 between ``G(input)`` and ``target`` (that would
silently turn the arm into a paired denoiser — CLAUDE.md pitfall #16). The batch
``target`` is used ONLY as a real B-domain sample for ``D``.

Design (why a bespoke two-optimiser ``train_step``):
    Neither reusable base loop can wire CUT honestly. ``BaseTrainingStrategy.train_step``
    emits only a single generator step (no discriminator update), so the adversarial
    term would train against a frozen, never-updated ``D``. ``GANTrainingStrategy.train_step``
    *does* alternate D/G, but its generator closure computes the loss through
    ``loss_computer.compute_generator_loss`` and **never calls ``_compute_losses_impl``**,
    so the PatchNCE term would be advertised-but-inert (pitfall #16). We therefore mirror
    the sibling :class:`CycleGANTrainingStrategy` (Task 6) fixed patterns: (I1) restore
    ``.train()`` on every net at the top of ``train_step``; (I2) AMP-config the
    strategy-owned nets/loss built after ``base.__init__``'s AMP pass; (I3) route the
    detached generator losses through the base ``_handle_anomalies`` guard and expose
    plain metric names.

Two subtleties this strategy handles:
    1. **Encoder-feature hook** — :meth:`_encode_features` registers forward hooks on the
       encoder half of ``cyclegan_generator`` (the ReLUs after the input+downsampling
       convs, plus the first ResnetBlocks), captures their outputs, and returns them as an
       equal-length list. ``feat_source = _encode_features(x)`` and
       ``feat_translated = _encode_features(G(x))`` feed ``cut_patch_nce`` via its required
       ``context`` dict (it RAISES on a length/shape mismatch).
    2. **PatchNCE MLP is trained** — ``cut_patch_nce``'s ``_PatchSampleMLP`` builds its
       per-channel projection heads *lazily* (on first observation of each channel count),
       so at construction its ``parameters()`` are EMPTY. :meth:`_materialize_nce_heads`
       runs one no-grad warm-up forward to instantiate the heads, then
       :meth:`_register_owned_params` folds them into the generator optimizer (CUT's
       ``optimizer_F`` merged into ``opt_g``) so they learn alongside ``G``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from spectramr.core.module_utils import unwrap_model
from spectramr.infrastructure.distributed.owned_modules import OWNED_MODULE
from spectramr.infrastructure.training.step_io import accepts_step_io
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

# Reuse the sibling CycleGAN baseline's pure, ``.item()``-free LSGAN helpers — the
# adversarial math (``gan_lsgan``) is identical and already unit-tested there, so we
# don't duplicate it (DRY; same layer, same directory).
from spectramr.infrastructure.training.strategies.cyclegan_strategy import (
    _lsgan_d_loss,
    _lsgan_g_loss,
)
from spectramr.infrastructure.training.strategies.loss_folding import unpaired_source_terms
from spectramr.infrastructure.training.strategies.mixins.adversarial import (
    AdversarialMixin,
)

# Import the concrete model / loss classes so their ``@register_*`` decorators run
# (the registry lookup below then resolves independent of two-phase populate
# ordering). ``ResnetBlock`` is needed to identify the encoder resblocks to hook.
from spectramr.models.discriminators.patchgan_discriminator import (  # noqa: F401
    PatchGANDiscriminator,
)
from spectramr.models.generators.cycle_gan import (  # noqa: F401
    ResnetBlock,
    ResNetGenerator,
)
from spectramr.models.losses.cut_contrastive_loss import CUTPatchNCELoss
from spectramr.models.registry import get_model_class

logger = logging.getLogger(__name__)


class CUTTrainingStrategy(BaseTrainingStrategy, AdversarialMixin):
    """Single-generator Contrastive Unpaired Translation (CUT).

    Resolves the primary ``cyclegan_generator`` / ``patch_gan`` from the training
    environment (falling back to building them for the unit-test / scripting path),
    owns a ``cut_patch_nce`` loss module, and registers the loss's projection-head
    params on the generator optimizer so the PatchNCE head trains with ``G``.
    """

    # parallel_compat: The NCE loss (its projection heads) is an owned module, built and wrapped
    # in __init__ on every rank, and called through forward_of.
    parallel_compat = frozenset({"replicated"})

    #: How many encoder ResnetBlocks contribute a NCE feature scale (the "early
    #: resblocks" of the CUT encoder half — the deeper blocks are decoder-adjacent).
    _max_encoder_resblocks: int = 2

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        return (*super().owned_module_declarations(config), ("nce_loss", OWNED_MODULE))

    def _setup_strategy_specific_components(self) -> None:
        """Build/resolve the generator, discriminator and NCE loss (called from
        the base ``__init__``)."""
        self.setup_models()

    # ------------------------------------------------------------------ #
    # Model / loss construction (registry-resolved; reuse cyclegan_generator/patch_gan)
    # ------------------------------------------------------------------ #
    def setup_models(self) -> None:
        """Build ``generator`` / ``discriminator`` / ``nce_loss`` (idempotent).

        ``generator`` / ``discriminator`` reuse the env-built primary generator /
        discriminator when present (the canonical single-generator pipeline
        output); otherwise they are built here. The ``cut_patch_nce`` loss module
        is always owned by the strategy.
        """
        model_cfg = getattr(self.config, "model", None)
        in_ch = int(getattr(model_cfg, "in_channels", 1) or 1)
        out_ch = int(getattr(model_cfg, "out_channels", 1) or 1)

        gen_cls = get_model_class("cyclegan_generator")
        disc_cls = get_model_class("patch_gan")

        env = getattr(self, "env", None)
        primary_gen = getattr(env, "generator", None) if env is not None else None
        primary_disc = getattr(env, "discriminator", None) if env is not None else None

        self.generator: nn.Module = (
            primary_gen if isinstance(primary_gen, nn.Module) else gen_cls(in_ch, out_ch)
        )
        self.discriminator: nn.Module = (
            primary_disc if isinstance(primary_disc, nn.Module) else disc_cls(in_channels=out_ch)
        )
        # The only trainable state of the NCE term is its per-channel projection
        # head (``sampler``); the loss itself is otherwise a fixed contrastive op.
        self.nce_loss: CUTPatchNCELoss = CUTPatchNCELoss()

        device = getattr(self, "device", torch.device("cpu"))
        for module in (self.generator, self.discriminator, self.nce_loss):
            module.to(device)

        # I2: ``base.__init__`` AMP-configures ONLY ``env.generator`` /
        # ``env.discriminator`` (base.py:434-436), BEFORE this hook runs
        # (base.py:492). Any net/loss the strategy builds itself must be
        # AMP-configured here too, or its grads won't scale under a real AMP
        # GradScaler. Guarded for the ``object.__new__`` unit-test path (no
        # ``amp_helper``).
        if hasattr(self, "amp_helper") and self.amp_helper is not None:
            if self.generator is not primary_gen:
                self.amp_helper.configure_model_for_amp(self.generator)
            if self.discriminator is not primary_disc:
                self.amp_helper.configure_model_for_amp(self.discriminator)
            self.amp_helper.configure_model_for_amp(self.nce_loss)

        # Eagerly instantiate the NCE sampler's lazy heads so their params exist
        # for optimizer registration NOW (see method docstring — CLAUDE.md #16).
        self._materialize_nce_heads(in_ch)
        # The env models are wrapped by the parallel backend; everything built here is owned.
        # The heads must all exist now: DDP reduces only the parameters it saw at the wrap.
        if self.generator is not primary_gen:
            self.register_owned_module("generator", self.generator)
        if self.discriminator is not primary_disc:
            self.register_owned_module("discriminator", self.discriminator)
        self.register_owned_module("nce_loss", self.nce_loss)
        self._nce_head_count = len(self.nce_loss.sampler.heads)
        self._register_owned_params()

    def _materialize_nce_heads(self, in_ch: int) -> None:
        """Force the NCE sampler's lazy per-channel projection heads into existence.

        ``_PatchSampleMLP`` creates a Linear-ReLU-Linear head the first time it sees
        a feature map of a given channel count, so at construction
        ``self.nce_loss.sampler.parameters()`` is EMPTY and registering it on the
        optimizer would train nothing — a PatchNCE that "fires" yet never learns its
        projection (pitfall #16). One no-grad warm-up forward through the encoder +
        the loss instantiates a head per encoder channel scale.
        """
        device = getattr(self, "device", torch.device("cpu"))
        was_training = self.generator.training
        self.generator.eval()
        try:
            with torch.no_grad():
                dummy = torch.randn(1, in_ch, 32, 32, device=device)
                feats = self._encode_features(dummy)
                # Forward through the loss to trigger lazy head creation per scale.
                self.nce_loss(
                    dummy,
                    dummy,
                    context={"feat_source": feats, "feat_translated": feats},
                )
        finally:
            if was_training:
                self.generator.train()

    def _register_owned_params(self) -> None:
        """Fold the strategy-owned params (NCE sampler-MLP, and the generator /
        discriminator when the strategy built them itself) into the env optimizers.

        No-op when the optimizers are absent (unit-test / scripting path). Uses
        :meth:`_sync_schedulers_after_param_group_addition` so a scheduler built
        before this addition does not raise on its next ``step()``. Idempotent: a
        param already present in a group is not re-added (PyTorch forbids a param in
        two groups).
        """
        env = getattr(self, "env", None)
        if env is None:
            return
        opt_cfg = getattr(self.config, "optimization", None)
        lr = float(getattr(getattr(opt_cfg, "optimizer", None), "learning_rate", 2e-4) or 2e-4)

        opt_g = getattr(env, "opt_g", None)
        if opt_g is not None:
            # CUT trains its projection head (``optimizer_F``) alongside G; we fold
            # it into ``opt_g`` rather than owning a second optimizer.
            sampler_params = list(self.nce_loss.sampler.parameters())
            if not sampler_params:
                raise RuntimeError(
                    "NCE sampler heads were not materialized before _register_owned_params — "
                    "_materialize_nce_heads must run first (CUT PatchNCE projection would never train)."
                )
            if not self._params_in_optimizer(opt_g, sampler_params):
                opt_g.add_param_group({"params": sampler_params})
                self._sync_schedulers_after_param_group_addition(lr)
            # A strategy-built generator (env.generator was absent) is not yet in
            # opt_g — register it so G trains at all.
            if self.generator is not getattr(env, "generator", None):
                gen_params = list(self.generator.parameters())
                if not self._params_in_optimizer(opt_g, gen_params):
                    opt_g.add_param_group({"params": gen_params})
                    self._sync_schedulers_after_param_group_addition(lr)

        opt_d = getattr(env, "opt_d", None)
        if opt_d is not None and self.discriminator is not getattr(env, "discriminator", None):
            disc_params = list(self.discriminator.parameters())
            if not self._params_in_optimizer(opt_d, disc_params):
                opt_d.add_param_group({"params": disc_params})
                self._sync_schedulers_after_param_group_addition(lr)

    @staticmethod
    def _params_in_optimizer(optimizer: torch.optim.Optimizer, params: list[nn.Parameter]) -> bool:
        """True if ANY of ``params`` is already registered on ``optimizer``."""
        existing = {id(p) for group in optimizer.param_groups for p in group["params"]}
        return any(id(p) in existing for p in params)

    # ------------------------------------------------------------------ #
    # Encoder-feature hook (subtlety #1)
    # ------------------------------------------------------------------ #
    def _encoder_submodules(self) -> list[nn.Module]:
        """Encoder-half submodules of ``cyclegan_generator`` to hook for NCE feats.

        The ``ResNetGenerator`` body is a single ``nn.Sequential``; the encoder half
        is everything before the first upsampling ``ConvTranspose2d``. We hook the
        post-conv ``ReLU``s (feature maps at the 64/128/256-channel scales) plus the
        first :data:`_max_encoder_resblocks` ResnetBlocks. Raises (no silent
        fallback, CLAUDE.md #9) if the generator is not a ``.model``-Sequential
        ResNet — CUT's content-preservation hypothesis is defined on that encoder.
        """
        model = getattr(unwrap_model(self.generator), "model", None)
        if not isinstance(model, nn.Sequential):
            raise TypeError(
                "CUTTrainingStrategy encoder-feature hooks require a "
                "cyclegan_generator (ResNetGenerator with a `.model` Sequential); "
                f"got {type(self.generator).__name__}."
            )
        assert isinstance(
            model, nn.Sequential
        )  # pyright narrowing; runtime no-op after the raise above
        max_resblocks = getattr(self, "_max_encoder_resblocks", 2)
        hooked: list[nn.Module] = []
        resblocks_seen = 0
        for layer in model:
            if isinstance(layer, nn.ConvTranspose2d):
                break  # reached the decoder / upsampling half
            if isinstance(layer, nn.ReLU):
                hooked.append(layer)
            elif isinstance(layer, ResnetBlock) and resblocks_seen < max_resblocks:
                hooked.append(layer)
                resblocks_seen += 1
        if not hooked:
            raise RuntimeError("No encoder submodules found to hook for CUT PatchNCE features.")
        return hooked

    def _forward_with_feats(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run ``generator(x)`` while capturing encoder feature maps via forward
        hooks. Returns ``(output, feats)``; the hooks are always removed."""
        feats: list[torch.Tensor] = []
        handles = [
            submod.register_forward_hook(lambda _m, _i, o: feats.append(o))
            for submod in self._encoder_submodules()
        ]
        try:
            out = self.forward_of("generator")(x)
        finally:
            for handle in handles:
                handle.remove()
        return out, feats

    def _encode_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Encoder feature maps captured while forwarding ``x`` through ``G`` — the
        ``feat_source`` / ``feat_translated`` lists consumed by ``cut_patch_nce``."""
        _out, feats = self._forward_with_feats(x)
        return feats

    # ------------------------------------------------------------------ #
    # Batch / weight resolution
    # ------------------------------------------------------------------ #
    def _resolve_domains(
        self, input_batch: Any, target_batch: Any, kwargs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(real_a, real_b)`` from either a mapping batch or tensors."""
        candidate = kwargs.get("batch")
        if candidate is None:
            candidate = input_batch
        if candidate is not None and not torch.is_tensor(candidate) and hasattr(candidate, "get"):
            real_a = candidate.get("input")
            real_b = candidate.get("target")
        else:
            real_a = input_batch
            real_b = target_batch
        if real_a is None or real_b is None:
            raise ValueError(
                "CUTTrainingStrategy requires BOTH an A-domain 'input' and a "
                "B-domain 'target' sample (unpaired). Got "
                f"input={type(real_a)!r}, target={type(real_b)!r}."
            )
        device = getattr(self, "device", real_a.device)
        return real_a.to(device), real_b.to(device)

    def _nce_weight(self) -> float:
        """Read ``lambda_nce`` from the config SSOT (``losses.gan``)."""
        gan_cfg = self.config.losses.gan if self.config.losses is not None else None
        if gan_cfg is None:
            raise ValueError(
                "CUTTrainingStrategy requires a `losses.gan:` block (the home of lambda_nce)."
            )
        return float(gan_cfg.lambda_nce)

    # ------------------------------------------------------------------ #
    # Loss computation (the required extension point + mechanism-fires probe)
    # ------------------------------------------------------------------ #
    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute the CUT generator losses for one unpaired batch.

        ``y = G(x)``; ``adv_g`` (LSGAN) pushes ``D(y)`` toward real; ``nce`` is the
        PatchNCE between the encoder features of the source ``x`` and of the output
        ``y`` (both run through the SAME encoder). The total is
        ``adv_g + lambda_nce * nce`` — there is deliberately NO pixel-L1 to
        ``target`` (the unpaired contract). The discriminator loss is owned by the
        D-closure, not returned here.
        """
        # Only the A-domain source is needed here; the B-domain real sample is the
        # D-closure's business (the unpaired contract — no pixel-L1 to ``target``).
        real_a, _real_b = self._resolve_domains(input_batch, target_batch, kwargs)
        lambda_nce = self._nce_weight()

        # feat_source is captured on the SAME forward that produces the translation
        # ``y`` (two forwards total: G(x) and G(y)); both feat lists carry gradients
        # so PatchNCE trains the encoder end-to-end.
        y, feat_source = self._forward_with_feats(real_a)
        feat_translated = self._encode_features(y)

        adv_g = _lsgan_g_loss(self.forward_of("discriminator")(y))
        nce = self.forward_of("nce_loss")(
            y,
            real_a,
            context={
                "feat_source": feat_source,
                "feat_translated": feat_translated,
            },
        )
        if len(self.nce_loss.sampler.heads) != self._nce_head_count:
            raise RuntimeError(
                "CUT PatchNCE built a projection head after construction; it is in neither "
                "opt_g nor the DDP wrapper, so it would never train. Its encoder feature "
                "widths must match the warm-up in _materialize_nce_heads."
            )

        g_total = adv_g + lambda_nce * nce
        # Declared ``role: source`` image losses (e.g. MIND-SSC) lock structure to the input.
        extra, components = unpaired_source_terms(self, y, real_a)
        if extra is not None:
            g_total = g_total + extra
        return {"g_total_loss": g_total, "adv_g": adv_g, "nce": nce, **components}

    # ------------------------------------------------------------------ #
    # Two-optimizer alternating step (D on discriminator, G on generator + NCE head)
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
        # I1: restore train mode on both nets before stepping. This bespoke
        # ``train_step`` overrides the base one; ``_validation_forward`` leaves the
        # generator in ``eval()``, so without this BatchNorm/Dropout-style layers
        # would run in inference mode during training after the first validation.
        self.generator.train()
        self.discriminator.train()
        self.nce_loss.train()

        first = input_batch if input_batch is not None else batch
        real_a, real_b = self._resolve_domains(first, target_batch, kwargs)
        self._last_step_metrics: dict[str, torch.Tensor] = {}
        step_configs: list[dict[str, Any]] = []

        env = self.env

        def d_closure() -> torch.Tensor:
            self.discriminator.requires_grad_(True)
            disc = self.forward_of("discriminator")
            with torch.no_grad():
                y = self.forward_of("generator")(real_a)
            d_loss = _lsgan_d_loss(disc(real_b), disc(y))
            with torch.no_grad():
                self._last_step_metrics["d_total_loss"] = d_loss.detach()
            return d_loss

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
            # to G THROUGH the frozen discriminator activations).
            self.discriminator.requires_grad_(False)
            # #1190: this closure calls ``_compute_losses_impl`` DIRECTLY, so it
            # bypasses the ``_compute_losses`` wrapper that normally emits the
            # ``model_output`` debug snapshot. Arm it here instead.
            #
            # ``self.generator``, NOT ``env.generator``: ``_build_models`` reuses
            # the env primary only when it is a ``cyclegan_generator`` and builds
            # its own otherwise, so hooking the env module would install the hook
            # on something this strategy never forwards.
            #
            # Armed TIGHTLY around the forward: ``d_closure`` above also runs
            # ``self.generator(real_a)`` (under ``no_grad``) to make its fake, and
            # a flag left armed across the whole step would let that overwrite the
            # stash so the snapshot showed D's fake instead of G's output.
            with self._capture_model_output(
                module=self.generator,
                input_batch=real_a,
                target_batch=real_b,
                step=iteration,
            ):
                losses = self._compute_losses_impl(
                    input_batch=real_a, target_batch=real_b, epoch=epoch
                )
            g_total = losses["g_total_loss"]
            # I3: route the detached G-losses through the base NaN/Inf anomaly guard
            # and expose PLAIN metric names (nce / adv_g / g_total_loss). ``update``
            # preserves ``d_total_loss`` written by the D-closure.
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
    # Validation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _validation_forward(
        self, input_batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Validation prediction is the A->B translation ``generator(input)``."""
        self.generator.eval()
        return self.generator(input_batch)

    @torch.no_grad()
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Translate ``input`` A->B and score against ``target`` when a metrics
        computer is available (evaluation only — NOT a training pixel loss)."""
        pred = self._validation_forward(input_batch)
        computer = getattr(self, "validation_metrics_computer", None)
        if computer is not None and target_batch is not None:
            try:
                return dict(computer.compute(pred, target_batch))
            except Exception as exc:  # pragma: no cover - defensive; never break val
                logger.warning("validation metric computation failed: %s", exc)
                return {}
        return {}
