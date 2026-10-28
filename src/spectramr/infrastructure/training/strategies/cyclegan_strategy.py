"""Unpaired CycleGAN training strategy (MRIxFields2026 baseline).

Two generators (``gen_ab`` A->B, ``gen_ba`` B->A) and two PatchGAN
discriminators (``disc_a``, ``disc_b``) trained WITHOUT paired supervision.
The batch ``target`` is used ONLY as a real B-domain sample for ``disc_b`` (and
for B-side cycle / identity self-consistency) -- there is **no** pixel-L1 between
``gen_ab(input)`` and ``target`` (that would silently turn the arm into a paired
denoiser; CLAUDE.md pitfall #16).

The ``.item()``-free loss math is lifted from
:class:`spectramr.models.generators.cycle_gan.CycleGAN.forward` (LSGAN adversarial +
cycle + identity), but the cycle / identity **weights are read from the config
SSOT** ``config.losses.gan.lambda_cycle`` / ``.lambda_identity`` (Task 5) instead of
the container's hard-coded ``10.0`` / ``0.5``. The registered ``cyclegan_generator``
(ResNet) and ``patch_gan`` discriminator components are reused verbatim.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.core.module_utils import unwrap_model
from spectramr.infrastructure.distributed.owned_modules import (
    OWNED_CALL,
    OWNED_MODULE,
    refuse_outside_forward_call,
)
from spectramr.infrastructure.training.step_io import accepts_step_io
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.loss_folding import unpaired_source_terms
from spectramr.infrastructure.training.strategies.mixins.adversarial import (
    AdversarialMixin,
)

# Import the concrete model classes so their ``@register_model`` decorators run
# (the registry lookup below then resolves independent of two-phase populate
# ordering); the classes stay reachable via the registry -- the SSOT.
from spectramr.models.discriminators.patchgan_discriminator import (  # noqa: F401
    PatchGANDiscriminator,
)
from spectramr.models.generators.b0_hypernetwork import B0ConditionedGenerator
from spectramr.models.generators.cycle_gan import ResNetGenerator  # noqa: F401
from spectramr.models.losses.domain_adversarial_grl import DomainAdversarialLoss
from spectramr.models.registry import get_model_class

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- #
# Pure, testable loss math (no ``.item()`` / ``.cpu()`` / host syncs).
# ----------------------------------------------------------------------------- #


def _lsgan_g_loss(pred_fake: torch.Tensor) -> torch.Tensor:
    """LSGAN generator term: push the discriminator score on fakes toward 1."""
    return torch.mean((pred_fake - 1.0) ** 2)


def _lsgan_d_loss(pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
    """LSGAN discriminator term (real->1, fake->0), averaged (the ``* 0.5``)."""
    return 0.5 * (torch.mean((pred_real - 1.0) ** 2) + torch.mean(pred_fake**2))


def _call_gen(gen: nn.Module, x: torch.Tensor, field: torch.Tensor | None) -> torch.Tensor:
    """Run a generator, passing its output-domain B0 only when the batch carries one.

    A B0-conditioned generator raises on ``field=None``, so a missing stamp fails loudly.
    """
    return gen(x) if field is None else gen(x, field_strength=field)


def _call_gen_anat(
    gen: nn.Module, x: torch.Tensor, field: torch.Tensor | None, want: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``_call_gen`` that also returns Z_anat when ``want`` (one pass, no extra forward)."""
    if not want:
        return _call_gen(gen, x, field), None
    # A non-forward call: the inner module under dp; ddp and sharded backends are refused.
    inner = unwrap_model(gen)
    if field is None:
        return inner.forward_with_anatomy(x)
    return inner.forward_with_anatomy(x, field_strength=field)


def compute_cyclegan_losses(
    gen_ab: nn.Module,
    gen_ba: nn.Module,
    disc_a: nn.Module,
    disc_b: nn.Module,
    real_a: torch.Tensor,
    real_b: torch.Tensor,
    *,
    lambda_cycle: float = 10.0,
    lambda_identity: float = 0.5,
    source_fold: Callable[[torch.Tensor, torch.Tensor], tuple[Any, dict[str, Any]]] | None = None,
    field_ab: torch.Tensor | None = None,
    field_ba: torch.Tensor | None = None,
    domain_fold: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, dict[str, Any]]]
    | None = None,
) -> dict[str, torch.Tensor]:
    """CycleGAN losses for one unpaired batch.

    Runs A->B->A and B->A->B, returning the generator total plus its raw
    (un-weighted) cycle / identity components and the separate discriminator loss
    on **detached** fakes. Only same-domain L1 terms appear (cycle: ``rec`` vs its
    own source; identity: ``gen(real)`` vs the same ``real``); there is deliberately
    no ``L1(fake_b, real_b)`` paired term.

    Args:
        gen_ab / gen_ba: the two generators (A->B, B->A).
        disc_a / disc_b: PatchGAN discriminators for the A- and B-domains.
        real_a: A-domain samples (the batch ``input``).
        real_b: B-domain samples (the batch ``target``) -- used ONLY as a real
            B-domain example, never as a pixel target for ``fake_b``.
        lambda_cycle / lambda_identity: weights from ``config.losses.gan``.
        field_ab / field_ba: output-domain B0 of ``gen_ab`` (domain B) and ``gen_ba``
            (domain A), forwarded only to B0-conditioned generators.
        domain_fold: optional ``(z_a, z_b) -> (weighted term, components)`` over the
            bottleneck of ``gen_ab`` on the A and B samples; reuses the forwards below.

    Returns:
        ``{"g_total_loss", "adv_g", "adv_d", "cycle", "identity"}`` (all scalar
        tensors; ``cycle`` / ``identity`` are the raw sums, weighted into the
        total).
    """
    want_anat = domain_fold is not None
    fake_b, z_a = _call_gen_anat(gen_ab, real_a, field_ab, want_anat)
    rec_a = _call_gen(gen_ba, fake_b, field_ba)
    fake_a = _call_gen(gen_ba, real_b, field_ba)
    rec_b = _call_gen(gen_ab, fake_a, field_ab)

    # Adversarial: generators try to fool the opposite-domain discriminator.
    adv_g = _lsgan_g_loss(disc_b(fake_b)) + _lsgan_g_loss(disc_a(fake_a))

    # Cycle consistency (raw L1; weighted by lambda_cycle in the total).
    cycle = F.l1_loss(rec_a, real_a) + F.l1_loss(rec_b, real_b)

    # Identity (raw L1; weighted by lambda_identity). Requires in==out channels
    # (the M4Raw 1-ch magnitude case), matching the container's id_A / id_B.
    id_a = _call_gen(gen_ba, real_a, field_ba)
    id_b, z_b = _call_gen_anat(gen_ab, real_b, field_ab, want_anat)
    identity = F.l1_loss(id_a, real_a) + F.l1_loss(id_b, real_b)

    # Discriminator loss on DETACHED fakes (no generator gradient path).
    adv_d = _lsgan_d_loss(disc_a(real_a), disc_a(fake_a.detach())) + _lsgan_d_loss(
        disc_b(real_b), disc_b(fake_b.detach())
    )

    g_total = adv_g + lambda_cycle * cycle + lambda_identity * identity
    out = {
        "g_total_loss": g_total,
        "adv_g": adv_g,
        "adv_d": adv_d,
        "cycle": cycle,
        "identity": identity,
    }
    if source_fold is not None:
        # Declared ``role: source`` image losses compare the translation with the input.
        extra, components = source_fold(fake_b, real_a)
        if extra is not None:
            out["g_total_loss"] = g_total + extra
            out.update(components)
    if domain_fold is not None:
        assert z_a is not None and z_b is not None  # set by _call_gen_anat when want_anat
        term, components = domain_fold(z_a, z_b)
        out["g_total_loss"] = out["g_total_loss"] + term
        out.update(components)
    return out


def _domain_grl_enabled(config: Any) -> bool:
    gan = getattr(getattr(config, "losses", None), "gan", None)
    return bool(getattr(getattr(gan, "domain_grl", None), "enabled", False))


class CycleGANTrainingStrategy(BaseTrainingStrategy, AdversarialMixin):
    """Unpaired two-generator CycleGAN for cross-field / cross-contrast translation.

    Resolves the primary ``cyclegan_generator`` / ``patch_gan`` from the training
    environment as ``gen_ab`` / ``disc_b`` and builds the reverse ``gen_ba`` and the
    second discriminator ``disc_a`` itself, registering the owned modules on the
    generator / discriminator optimizers so all four networks train.
    """

    # parallel_compat: gen_ba / disc_a (and the GRL head) are owned modules, DDP-wrapped and
    # called through forward_of; the GRL bottleneck tap is a non-forward call, refused.
    parallel_compat = frozenset({"replicated"})

    #: ``(field_strength_t, field_strength_source_t)`` of the current training step.
    _step_fields: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None)

    #: Gradient-reversal domain head on gen_ab's bottleneck; built only when domain_grl is on.
    domain_head: DomainAdversarialLoss | None = None

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        own: tuple[tuple[str, str], ...] = (("gen_ba", OWNED_MODULE), ("disc_a", OWNED_MODULE))
        if _domain_grl_enabled(config):
            own += (("domain_head", OWNED_MODULE), ("gen_ab.forward_with_anatomy", OWNED_CALL))
        return (*super().owned_module_declarations(config), *own)

    def _setup_strategy_specific_components(self) -> None:
        """Build/resolve the four networks (called from the base ``__init__``)."""
        self.setup_models()

    # ------------------------------------------------------------------ #
    # Model construction (registry-resolved; reuse cyclegan_generator/patch_gan)
    # ------------------------------------------------------------------ #
    def setup_models(self) -> None:
        """Build ``gen_ab``, ``gen_ba``, ``disc_a``, ``disc_b`` (idempotent).

        ``gen_ab`` / ``disc_b`` reuse the env-built primary generator /
        discriminator when present (the canonical single-generator pipeline
        output); the reverse ``gen_ba`` and second ``disc_a`` are always built
        here and registered on the env optimizers.
        """
        model_cfg = getattr(self.config, "model", None)
        in_ch = int(getattr(model_cfg, "in_channels", 1) or 1)
        out_ch = int(getattr(model_cfg, "out_channels", 1) or 1)

        gen_cls = get_model_class("cyclegan_generator")
        disc_cls = get_model_class("patch_gan")

        env = getattr(self, "env", None)
        primary_gen = getattr(env, "generator", None) if env is not None else None
        primary_disc = getattr(env, "discriminator", None) if env is not None else None

        self.gen_ab: nn.Module = (
            primary_gen if isinstance(primary_gen, nn.Module) else gen_cls(in_ch, out_ch)
        )
        # The env models arrive wrapped under dp/ddp, so their class and taps are read inside.
        inner_ab = unwrap_model(self.gen_ab)
        # A B0-conditioned primary carries its reverse twin, so each direction gets its own
        # output-domain B0 (domain B for gen_ab, domain A for gen_ba).
        if isinstance(inner_ab, B0ConditionedGenerator):
            self.gen_ba: nn.Module = inner_ab.reverse(out_ch, in_ch)
        else:
            self.gen_ba = gen_cls(out_ch, in_ch)
        self.disc_b: nn.Module = (
            primary_disc if isinstance(primary_disc, nn.Module) else disc_cls(in_channels=out_ch)
        )
        self.disc_a: nn.Module = disc_cls(in_channels=in_ch)

        gan_cfg = getattr(getattr(self.config, "losses", None), "gan", None)
        self.domain_head = None
        if gan_cfg is not None and gan_cfg.domain_grl.enabled:
            refuse_outside_forward_call(
                getattr(self.config, "parallel", None),
                owner=type(self).__name__,
                name="gen_ab.forward_with_anatomy",
            )
            if not hasattr(inner_ab, "forward_with_anatomy"):
                raise ValueError(
                    "losses.gan.domain_grl.enabled needs a generator with a bottleneck tap "
                    f"(cyclegan_generator); gen_ab is {type(inner_ab).__name__}."
                )
            self.domain_head = DomainAdversarialLoss(
                latent_dim=inner_ab.latent_channels,
                alpha=1.0,
                loss_weight=float(gan_cfg.lambda_domain),
            )

        device = getattr(self, "device", torch.device("cpu"))
        for module in (self.gen_ab, self.gen_ba, self.disc_a, self.disc_b):
            module.to(device)
        if self.domain_head is not None:
            self.domain_head.to(device)

        # I2: ``base.__init__`` AMP-configures ``env.generator`` (gen_ab) and
        # ``env.discriminator`` (disc_b) BEFORE ``_setup_strategy_specific_components``
        # runs (base.py:434-436 vs :492), so the strategy-owned ``gen_ba`` / ``disc_a``
        # built here would otherwise never be AMP-configured and their grads may not
        # scale correctly under a real AMP GradScaler. Guarded for the object.__new__
        # unit-test path that bypasses ``__init__`` (no ``amp_helper``).
        if hasattr(self, "amp_helper") and self.amp_helper is not None:
            self.amp_helper.configure_model_for_amp(self.gen_ba)
            self.amp_helper.configure_model_for_amp(self.disc_a)
            if self.domain_head is not None:
                self.amp_helper.configure_model_for_amp(self.domain_head)

        # The env models are wrapped by the parallel backend; everything built here is owned.
        if self.gen_ab is not primary_gen:
            self.register_owned_module("gen_ab", self.gen_ab)
        self.register_owned_module("gen_ba", self.gen_ba)
        self.register_owned_module("disc_a", self.disc_a)
        if self.disc_b is not primary_disc:
            self.register_owned_module("disc_b", self.disc_b)
        if self.domain_head is not None:
            self.register_owned_module("domain_head", self.domain_head)

        self._register_owned_params()

    def _register_owned_params(self) -> None:
        """Add the strategy-owned ``gen_ba`` / ``disc_a`` to the env optimizers.

        No-op when the optimizers are absent (unit-test / scripting path). Uses
        :meth:`_sync_schedulers_after_param_group_addition` so a scheduler built
        before this addition does not raise on its next ``step()``.
        """
        env = getattr(self, "env", None)
        if env is None:
            return
        opt_cfg = getattr(self.config, "optimization", None)
        lr = float(getattr(getattr(opt_cfg, "optimizer", None), "learning_rate", 2e-4) or 2e-4)

        opt_g = getattr(env, "opt_g", None)
        if opt_g is not None and self.gen_ba is not getattr(env, "generator", None):
            gen_ba_params = list(self.gen_ba.parameters())
            if not self._params_in_optimizer(opt_g, gen_ba_params):
                opt_g.add_param_group({"params": gen_ba_params})
                self._sync_schedulers_after_param_group_addition(lr)

        opt_d = getattr(env, "opt_d", None)
        if opt_d is not None and self.disc_a is not getattr(env, "discriminator", None):
            disc_a_params = list(self.disc_a.parameters())
            if not self._params_in_optimizer(opt_d, disc_a_params):
                opt_d.add_param_group({"params": disc_a_params})
                self._sync_schedulers_after_param_group_addition(lr)

        # The head's loss is in g_total, so its classifier trains in the generator step.
        if opt_g is not None and self.domain_head is not None:
            head_params = list(self.domain_head.parameters())
            if not self._params_in_optimizer(opt_g, head_params):
                opt_g.add_param_group({"params": head_params})
                self._sync_schedulers_after_param_group_addition(lr)

    @staticmethod
    def _params_in_optimizer(optimizer: torch.optim.Optimizer, params: list[nn.Parameter]) -> bool:
        """True if ANY of ``params`` is already registered on ``optimizer``.

        Mirrors :meth:`CUTTrainingStrategy._params_in_optimizer`; used by
        :meth:`_register_owned_params` to guard ``add_param_group`` against
        double-registering parameters on a second ``setup_models`` call (pitfall #9:
        no silent crash via PyTorch's ``ValueError: some parameters appear in more
        than one parameter group``).
        """
        existing = {id(p) for group in optimizer.param_groups for p in group["params"]}
        return any(id(p) in existing for p in params)

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
                "CycleGANTrainingStrategy requires BOTH an A-domain 'input' and a "
                "B-domain 'target' sample (unpaired). Got "
                f"input={type(real_a)!r}, target={type(real_b)!r}."
            )
        device = getattr(self, "device", real_a.device)
        return real_a.to(device), real_b.to(device)

    def _weights(self) -> tuple[float, float]:
        """Read ``(lambda_cycle, lambda_identity)`` from the config SSOT."""
        gan_cfg = self.config.losses.gan if self.config.losses is not None else None
        if gan_cfg is None:
            raise ValueError(
                "CycleGANTrainingStrategy requires a `losses.gan:` block "
                "(the home of lambda_cycle / lambda_identity)."
            )
        return float(gan_cfg.lambda_cycle), float(gan_cfg.lambda_identity)

    # ------------------------------------------------------------------ #
    # Loss computation (the required extension point)
    # ------------------------------------------------------------------ #
    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute the CycleGAN losses for one unpaired batch."""
        real_a, real_b = self._resolve_domains(input_batch, target_batch, kwargs)
        lambda_cycle, lambda_identity = self._weights()
        field_ab, field_ba = self._fields_from(kwargs.get("batch"))
        if field_ab is None and field_ba is None:
            field_ab, field_ba = self._step_fields
        return compute_cyclegan_losses(
            self.forward_of("gen_ab"),
            self.forward_of("gen_ba"),
            self.forward_of("disc_a"),
            self.forward_of("disc_b"),
            real_a,
            real_b,
            lambda_cycle=lambda_cycle,
            lambda_identity=lambda_identity,
            source_fold=lambda pred, src: unpaired_source_terms(self, pred, src),
            field_ab=field_ab,
            field_ba=field_ba,
            domain_fold=self._domain_term if self.domain_head is not None else None,
        )

    def _domain_term(
        self, z_a: torch.Tensor, z_b: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Weighted GRL domain loss on Z_anat, labelled A = batch input (0), B = target (1)."""
        assert self.domain_head is not None
        head = self.forward_of("domain_head")
        z = torch.cat([z_a, z_b], dim=0)
        labels = torch.cat([z_a.new_zeros(z_a.shape[0], 1), z_a.new_ones(z_b.shape[0], 1)], dim=0)
        out = head(z, labels)
        return out["domain_loss"], {
            "loss_domain": out["domain_loss"].detach(),
            "domain_acc": out["domain_acc"],
        }

    def _fields_from(self, batch: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return ``(field_strength_t, field_strength_source_t)`` stamped on a batch, or Nones."""
        if batch is None or torch.is_tensor(batch) or not hasattr(batch, "get"):
            return None, None
        device = getattr(self, "device", None)

        def _to(v: Any) -> torch.Tensor | None:
            if v is None:
                return None
            t = torch.as_tensor(v, dtype=torch.float32).reshape(-1)
            return t.to(device) if device is not None else t

        return _to(batch.get("field_strength_t")), _to(batch.get("field_strength_source_t"))

    # ------------------------------------------------------------------ #
    # Two-optimizer alternating step (D on disc_a+disc_b, G on gen_ab+gen_ba)
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
        # ``train_step`` overrides the base one (base.py:921-925) that would
        # otherwise flip the model back to ``train()``; ``_validation_forward``
        # leaves ``gen_ab`` in ``eval()`` and only ``gen_ab`` is toggled there, so
        # without this every net could run BatchNorm/Dropout in inference mode
        # during training after the first validation pass.
        for _m in (self.gen_ab, self.gen_ba, self.disc_a, self.disc_b):
            _m.train()
        if self.domain_head is not None:
            self.domain_head.train()
        first = input_batch if input_batch is not None else batch
        real_a, real_b = self._resolve_domains(first, target_batch, kwargs)
        self._step_fields = self._fields_from(batch if batch is not None else first)
        field_ab, field_ba = self._step_fields
        self._last_step_metrics: dict[str, torch.Tensor] = {}
        step_configs: list[dict[str, Any]] = []

        env = self.env

        def d_closure() -> torch.Tensor:
            for disc in (self.disc_a, self.disc_b):
                disc.requires_grad_(True)
            gen_ab, gen_ba = self.forward_of("gen_ab"), self.forward_of("gen_ba")
            disc_a, disc_b = self.forward_of("disc_a"), self.forward_of("disc_b")
            with torch.no_grad():
                fake_b = _call_gen(gen_ab, real_a, field_ab)
                fake_a = _call_gen(gen_ba, real_b, field_ba)
            d_loss = _lsgan_d_loss(disc_a(real_a), disc_a(fake_a)) + _lsgan_d_loss(
                disc_b(real_b), disc_b(fake_b)
            )
            with torch.no_grad():
                self._last_step_metrics["d_total_loss"] = d_loss.detach()
            return d_loss

        opt_d = getattr(env, "opt_d", None)
        if opt_d is not None:
            step_configs.append(
                {
                    "optimizer": opt_d,
                    "closure": d_closure,
                    "model": self.disc_a,
                    "name": "discriminator",
                }
            )

        def g_closure() -> torch.Tensor:
            # Freeze discriminators for the generator step (gradients still flow
            # to the generators THROUGH the frozen discriminator activations).
            for disc in (self.disc_a, self.disc_b):
                disc.requires_grad_(False)
            # #1190: direct ``_compute_losses_impl`` call, so the wrapper that
            # emits the ``model_output`` snapshot is bypassed -- arm it here.
            #
            # ``self.gen_ab`` is the A->B generator the step config already
            # declares as this closure's ``model``; it reuses ``env.generator``
            # only when one was built, so the module must be named explicitly.
            # ``gen_ba`` is deliberately NOT hooked: one snapshot per step, and
            # A->B is the direction the arm is evaluated on.
            #
            # Armed TIGHTLY: ``d_closure`` above runs ``self.gen_ab(real_a)``
            # under ``no_grad`` to make its fake, which would otherwise overwrite
            # the stash.
            with self._capture_model_output(
                module=self.gen_ab,
                input_batch=real_a,
                target_batch=real_b,
                step=iteration,
            ):
                losses = self._compute_losses_impl(
                    input_batch=real_a, target_batch=real_b, epoch=epoch
                )
            g_total = losses["g_total_loss"]
            # I3 + M2: route the detached G-losses through the base NaN/Inf
            # anomaly guard (mirrors ``BaseTrainingStrategy.train_step``'s
            # g_closure at base.py:1101-1104) instead of bypassing ``_handle_anomalies``,
            # and expose the metrics under their PLAIN names
            # (cycle / identity / adv_g / adv_d / g_total_loss) that the reporter and
            # early-stopping consume -- NOT the old ``g_``-prefixed keys. ``update``
            # preserves ``d_total_loss`` written by the D-closure. No extra host sync
            # is introduced: ``_handle_anomalies`` keeps the tensors on the device
            # for the training loop's log-gated conversion.
            detached = {k: v.detach() for k, v in losses.items()}
            self._last_step_metrics.update(self._handle_anomalies(detached, {}, epoch))
            return g_total

        step_configs.append(
            {
                "optimizer": getattr(env, "opt_g", None),
                "closure": g_closure,
                "model": self.gen_ab,
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
        """Validation prediction is the A->B translation ``gen_ab(input)``."""
        self.gen_ab.eval()
        ctx = args[0] if args and isinstance(args[0], dict) else kwargs.get("batch_context") or {}
        field_ab = self._fields_from(ctx)[0]
        return _call_gen(self.gen_ab, input_batch, field_ab)

    @torch.no_grad()
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor | None = None,
        field_strength_t: torch.Tensor | None = None,
        field_strength_source_t: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Translate ``input`` A->B and score against ``target`` when a metrics
        computer is available (evaluation only -- NOT a training pixel loss)."""
        ctx = {"field_strength_t": field_strength_t, "field_strength_source_t": field_strength_source_t}
        pred = self._validation_forward(input_batch, ctx)
        computer = getattr(self, "validation_metrics_computer", None)
        if computer is not None and target_batch is not None:
            try:
                return dict(computer.compute(pred, target_batch))
            except Exception as exc:  # pragma: no cover - defensive; never break val
                logger.warning("validation metric computation failed: %s", exc)
                return {}
        return {}
