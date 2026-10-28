"""In-process training entry point for the scripting mode — ``fit`` / ``Trainer``.

The imperative peer of the config-driven ``spectramr train``: a user brings their
own ``model`` / ``optimizer`` / dataloaders in Python and runs training directly,
**reusing the same loop** the config path uses. ``fit`` assembles a
:class:`TrainingEnvironment` from those objects (via
:meth:`TrainingEnvironment.from_components`) and drives
:func:`spectramr.pipelines.train.run_training_pipeline` with that environment, so
there is exactly one training loop — no duplicate engine.

Paradigms: ``fit(paradigm=...)`` selects the training paradigm (the strategy the
factory resolves) and the sensible default config/loss block for it:

* ``reconstruction`` (default) — supervised image L1.
* ``diffusion`` — adds the ``training.diffusion`` block (timesteps/schedule).
* ``vae`` / ``vqvae`` — adds the ``losses.latent`` (KL / commitment) block.
* ``gan`` — wires a second model (``discriminator``) + optimizer (``opt_d``) and
  the mandatory ``losses.gan`` block; you MUST pass ``discriminator=`` (or set
  ``model.discriminator_component`` in ``config``) or it raises (a GAN without a
  discriminator silently collapses to plain L1 — pitfall #9/#16).
* ``ssl`` / ``mae`` / ``physics_equivariant_ssl`` — self-supervised paradigms.

``paradigm`` is validated against the registered strategy short-names; an unknown
value raises. For full control, pass a ready ``strategy=`` (used verbatim) or a
complete ``config`` (its ``training.strategy_class`` is authoritative).

.. note::

   Your dataloaders must yield batches as a mapping with the canonical keys
   ``"input"`` and ``"target"`` (``BatchAdapter`` raises on non-canonical keys);
   for non-recon paradigms the ``model`` you pass must itself be
   paradigm-appropriate (e.g. a VAE model that emits ``(recon, mu, logvar)`` — a
   plain UNet would degrade to a zero-KL autoencoder; a diffusion model must
   accept a timestep).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.builders.leaf.optimizer_builders import OptimizerBuilder
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.builders.loss_builder import LossBuilder
from spectramr.pipelines.train import run_training_pipeline

logger = logging.getLogger(__name__)


class TrainingFailedError(RuntimeError):
    """A scripted ``fit`` run did not succeed.

    ``run_training_pipeline`` reports failure in its RETURN VALUE
    (``{"success": False, "error": ...}``) because the CLI translates that into
    an exit code. A script has no exit code to read, so the same convention
    means a caller who forgets ``if result["success"]`` proceeds with an
    untrained model -- and the model object they hold looks entirely normal.
    """


#: The default supervised image-L1 loss block (v6.x list-based, domain-aware).
_RECON_LOSSES: dict[str, Any] = {
    "output_domain": "image",
    "image_losses": [{"name": "l1", "weight": 1.0}],
}

#: Per-paradigm default ``losses`` + ``training`` sub-blocks. The key is the
#: strategy short-name (validated against the factory's registry). A paradigm not
#: listed here still works via the generic reconstruction-style defaults — this
#: table only tailors the well-known paradigms' loss/strategy blocks.
_PARADIGM_DEFAULTS: dict[str, dict[str, Any]] = {
    "reconstruction": {"losses": _RECON_LOSSES, "training": {}},
    "diffusion": {
        "losses": _RECON_LOSSES,
        "training": {
            "diffusion": {
                "timesteps": 1000,
                "noise_schedule": "cosine",
                "sampling_steps": 50,
            }
        },
    },
    "vae": {
        "losses": {
            **_RECON_LOSSES,
            "latent": {"lambda_kl": 1.0, "latent_loss_type": "kl_divergence"},
        },
        "training": {"vae": {"enable_kl_annealing": True, "kl_anneal_steps": 10000}},
    },
    "vqvae": {
        "losses": {
            **_RECON_LOSSES,
            "latent": {"lambda_kl": 1.0, "latent_loss_type": "kl_divergence"},
        },
        "training": {"vae": {"enable_vq_commitment": True}},
    },
    "gan": {
        # losses.gan is MANDATORY: AdversarialMixin._resolve_disc_updates raises
        # if it is None (invoked from GANTrainingStrategy.__init__).
        # enable_adversarial:True is LOAD-BEARING — the adversarial term is gated
        # on it (config/schemas/loss.py), so without it get_enabled_losses()
        # returns only {"l1"} and the GAN silently collapses to an L1 denoiser
        # (the discriminator is built but never updated) — a pitfall-#16 facade.
        "losses": {
            **_RECON_LOSSES,
            "gan": {"enable_adversarial": True, "lambda_adv": 1.0, "gan_loss_type": "lsgan"},
        },
        "training": {"gan": {}},
    },
    "ssl": {"losses": {**_RECON_LOSSES, "ssl": {}}, "training": {}},
    "mae": {"losses": {**_RECON_LOSSES, "ssl": {}}, "training": {}},
    "physics_equivariant_ssl": {
        "losses": {**_RECON_LOSSES, "ssl": {}},
        "training": {},
    },
    # EDM (Karras) diffusion — subclasses DiffusionTrainingStrategy, reads
    # training.diffusion and WARNS unless parameterization=edm (pitfall #10), so
    # set it explicitly. Standard (input,target) data; the model is a diffusion
    # model (accepts a timestep), supplied by the caller.
    "edm": {
        "losses": _RECON_LOSSES,
        "training": {
            "diffusion": {
                "timesteps": 1000,
                "noise_schedule": "cosine",
                "sampling_steps": 50,
                "parameterization": "edm",
            }
        },
    },
    # MAE masked pretraining — subclasses ReconstructionTrainingStrategy and masks
    # internally, so it runs on standard (input,target) data.
    "masked": {
        "losses": _RECON_LOSSES,
        "training": {
            "masked": {
                "mask_ratio": 0.5,
                "patch_size": 16,
                "masking_strategy": "random",
            }
        },
    },
    # Cartoon/texture-safe reconstruction — subclass of recon; its distinctive
    # CartoonTextureSafeLoss is built FROM this training block (so set it
    # explicitly rather than relying on all-defaults — pitfall #15).
    "cartoon_texture_safe": {
        "losses": _RECON_LOSSES,
        "training": {
            "cartoon_texture_safe": {
                # No ``lambda_l1`` here: the L1 weight moved onto
                # ``losses.image_losses`` (one owner, corpus review) and the
                # legacy spelling now RAISES at load, so writing it made
                # ``fit --paradigm cartoon_texture_safe`` unloadable. The
                # ``_RECON_LOSSES`` block above carries ``l1`` at 1.0, the
                # value this line used to set.
                "lambda_ct": 0.5,
                "tv_weight": 0.15,
                "n_iter": 40,
                "cartoon_weight": 1.0,
                "texture_weight": 0.3,
                "texture_pool": 4,
            }
        },
    },
    # Recoverability VIB — recon subclass; the information-bottleneck Lagrangian
    # (recon + beta*rate) reads this block. Standard (input,target) data.
    "recoverability_vib": {
        "losses": _RECON_LOSSES,
        # ``lambda_recon`` is retired for the same reason as ``lambda_l1`` above;
        # ``_RECON_LOSSES`` carries the reconstruction weight at 1.0.
        "training": {"recoverability_vib": {"beta": 0.001}},
    },
    # Generic generative strategy — reads nothing beyond the base; the model's
    # log_prob/training_loss is consulted with a documented fallback order.
    "generative": {"losses": _RECON_LOSSES, "training": {}},
}

#: Paradigm aliases — variants that share another paradigm's default block because
#: their config CONTRACT is identical to the family's default (they read exactly
#: the keys that default sets; the distinctive loss is the strategy's own, so fit
#: need not add a loss key). Verified per strategy. Bespoke variants that ALSO
#: need physics / field / k-space / multi-rep data (cold / field / Riemannian /
#: Bloch diffusion, ssdu, n2n, …) are intentionally NOT listed — they stay
#: config-only (run them with a full ``config=`` or a campaign).
_PARADIGM_ALIASES: dict[str, str] = {
    # Exact ``DiffusionTrainingStrategy`` class.
    "flow_matching": "diffusion",
    "rectified_flow": "diffusion",
    "resetting_diffusion": "diffusion",
    "tissue_diffusion_pretrain": "diffusion",
    # Distinct subclasses with a byte-identical ``training.diffusion`` contract
    # (synthesize source when absent / cross-modal block optional).
    "flow_matching_pfode": "diffusion",
    "stochastic_interpolants": "diffusion",
    "x_diffusion": "diffusion",
    "cross_modal_diffusion": "diffusion",
    # Reconstruction subclass with the identical recon contract (physics-prompt
    # conditioning is optional — graceful skip when the batch lacks it).
    "universal_reconstruction": "reconstruction",
}


def _valid_paradigms() -> set[str]:
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    return set(TrainingStrategyFactory.STRATEGY_CLASS_PATHS)


def _recon_routed_paradigms() -> set[str]:
    """Paradigms whose strategy class IS ``ReconstructionTrainingStrategy``.

    These are reconstruction variants — the generic reconstruction defaults are
    correct for them, so ``fit(paradigm=...)`` works with no extra config. (A
    paradigm whose strategy is a *different* class needs its own config, which a
    synthetic default cannot fake — see ``_resolve_fit_config``.)
    """
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    return {
        k
        for k, v in TrainingStrategyFactory.STRATEGY_CLASS_PATHS.items()
        if v.endswith(".ReconstructionTrainingStrategy")
    }


def _resolve_fit_config(
    config: TrainingSettings | dict[str, Any] | None,
    *,
    paradigm: str,
    epochs: int | None,
    max_iterations: int | None,
) -> TrainingSettings:
    """Resolve the ``fit`` config to a frozen :class:`TrainingSettings`.

    A fully-built ``TrainingSettings`` is used as-is (its ``strategy_class`` is
    authoritative). A dict (or ``None``) is completed with the paradigm's default
    model / loss / training blocks and validated via ``settings_from_dict``.
    ``epochs`` / ``max_iterations`` override the training length.
    """
    if isinstance(config, TrainingSettings):
        return config

    valid = _valid_paradigms()
    if paradigm not in valid:
        raise ValueError(
            f"Unknown paradigm {paradigm!r}. Choose one of {sorted(valid)}, "
            f"or pass a ready strategy= / a full TrainingSettings config."
        )

    # Categorize the paradigm so we never silently mis-default it (pitfall #9):
    #   * tailored  → use its default block;
    #   * recon-routed (strategy IS ReconstructionTrainingStrategy) → generic recon
    #     defaults are correct;
    #   * otherwise → fit() cannot synthesize a faithful default. Accept it only
    #     when the caller supplied a config dict that configures it (trust the
    #     user); with no config, RAISE with guidance rather than run a facade.
    family = _PARADIGM_ALIASES.get(paradigm, paradigm)
    if family in _PARADIGM_DEFAULTS:
        defaults = _PARADIGM_DEFAULTS[family]
    elif paradigm in _recon_routed_paradigms():
        defaults = _PARADIGM_DEFAULTS["reconstruction"]
    elif isinstance(config, dict) and (config.get("training") or {}):
        # The caller is explicitly configuring a non-default paradigm — trust it.
        defaults = {"losses": _RECON_LOSSES, "training": {}}
    else:
        tailored = sorted(set(_PARADIGM_DEFAULTS) | set(_PARADIGM_ALIASES))
        raise ValueError(
            f"fit(paradigm={paradigm!r}) has no tailored default and is not a "
            f"reconstruction variant — a synthetic default would be a facade. "
            f"Pass a full config= (a TrainingSettings, or a dict with this "
            f"paradigm's training/losses blocks) or a ready strategy=. "
            f"Auto-defaulted paradigms: {tailored} (+ reconstruction variants)."
        )

    base: dict[str, Any] = dict(config or {})
    base.setdefault("model", {"model_type": "unet"})
    # ``synthetic`` avoids a real-data availability check at container build;
    # the user's own dataloaders are injected via the environment regardless.
    base.setdefault("data", {"dataset_type": "synthetic"})
    base.setdefault("optimization", {})
    base.setdefault("logging", {})
    base.setdefault("checkpoint", {})  # required by the container builder
    base.setdefault("losses", defaults["losses"])

    training = dict(base.get("training") or {})
    training.setdefault("strategy_class", paradigm)
    for key, value in defaults["training"].items():
        training.setdefault(key, value)
    if epochs is not None:
        training["epochs"] = epochs
    if max_iterations is not None:
        training["max_iterations"] = max_iterations
    # The config validator requires a defined run length (epochs >= 1 OR
    # max_iterations). Default to a single epoch so a bare ``fit(model, loader)``
    # is valid; pass ``epochs=`` / ``max_iterations=`` for a real run.
    if training.get("epochs") is None and training.get("max_iterations") is None:
        training["epochs"] = 1
    base["training"] = training

    return TrainingSettings.settings_from_dict(base)


def _build_discriminator_from_config(
    settings: TrainingSettings, device: torch.device
) -> torch.nn.Module | None:
    """Build a discriminator from ``model.discriminator_component`` config.

    Returns ``None`` when no discriminator config is present (``build_discriminator``
    is a no-op then), so the caller can raise a clear error.
    """
    from spectramr.infrastructure.training.builders.model_builder import ModelBuilder

    disc_cfg = getattr(settings.model, "discriminator_component", None)
    try:
        built = ModelBuilder(settings, device).build_discriminator().build()
    except Exception as exc:
        # L3: don't swallow a REAL build failure as "no discriminator" — that
        # mis-routes the caller to the "pass discriminator=" fix path. If a
        # discriminator IS configured, a failure is a genuine error → re-raise
        # with context. Only the genuinely-absent case returns None (logged).
        if disc_cfg:
            raise RuntimeError(
                f"discriminator_component={disc_cfg!r} is configured but failed to build: {exc}"
            ) from exc
        logger.debug("no discriminator built (none configured): %s", exc)
        return None
    return built.get("discriminator")


def fit(
    model: torch.nn.Module,
    train_loader: Any,
    *,
    paradigm: str = "reconstruction",
    optimizer: torch.optim.Optimizer | None = None,
    discriminator: torch.nn.Module | None = None,
    opt_d: torch.optim.Optimizer | None = None,
    val_loader: Any | None = None,
    losses: dict[str, torch.nn.Module] | None = None,
    config: TrainingSettings | dict[str, Any] | None = None,
    strategy: Any | None = None,
    device: str | torch.device = "cpu",
    epochs: int | None = None,
    max_iterations: int | None = None,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Train ``model`` on ``train_loader`` in-process, reusing the standard loop.

    Args:
        model: The network to train (placed under the canonical ``"generator"``
            key the loop + strategies read).
        train_loader: Training dataloader yielding loop-compatible batches.
        paradigm: Training paradigm / strategy short-name (``reconstruction`` by
            default; ``gan``/``diffusion``/``vae``/``vqvae``/``ssl``/``mae``/
            ``physics_equivariant_ssl``). Validated against the registry.
        optimizer: Generator optimizer; if ``None``, built from the config.
        discriminator: GAN only — the discriminator model (or set
            ``model.discriminator_component`` in ``config``).
        opt_d: GAN only — the discriminator optimizer; if ``None``, built.
        val_loader: Optional validation dataloader (defaults to ``train_loader``).
        losses: Optional pre-built loss-module dict to use as ``env.losses``; if
            ``None``, built by :class:`LossBuilder` from ``config.losses`` (the
            canonical, strategy-consumed path).
        config: A full ``TrainingSettings`` **or** a partial dict completed with
            the paradigm defaults, or ``None``.
        strategy: Optional pre-built strategy instance; when given, the strategy
            factory is skipped (paradigm still drives the default config).
        device: Device for the run.
        epochs / max_iterations: Convenience overrides for the training length.
        raise_on_failure: Raise :class:`TrainingFailedError` when the pipeline
            reports ``success is not True``. Defaults to ``True``: the
            config-driven CLI surfaces a failure through its exit code, but a
            script has no such channel, and the pipeline returns
            ``{"success": False, "error": ...}`` rather than raising -- so a
            caller who does not think to check the flag carries on with an
            UNTRAINED model and no indication anything went wrong. Pass ``False``
            to inspect the result dict yourself (the smoke harness does, to keep
            its own diagnostics).

    Returns:
        The training-pipeline result dict (``success``, ``best_metrics``, …).

    Raises:
        TrainingFailedError: The run failed and ``raise_on_failure`` is set.
    """
    settings = _resolve_fit_config(
        config, paradigm=paradigm, epochs=epochs, max_iterations=max_iterations
    )
    torch_device = torch.device(device) if isinstance(device, str) else device

    if optimizer is None:
        optimizer = OptimizerBuilder(settings, params=model.parameters()).validate().build()

    if losses is None:
        # Build the RECON/image losses the canonical way. NOTE: this builds only
        # the reconstruction losses — the paradigm's DISTINCTIVE term (adversarial
        # / KL / score) is NOT built here; it is computed by the resolved
        # strategy's own loss computer, gated on its enable flag in
        # ``config.losses`` (e.g. ``enable_adversarial``). That separation is the
        # blind spot the B2 gan facade exploited, so the distinctive term lives in
        # the paradigm DEFAULT config, not in this builder call.
        losses = LossBuilder(settings, torch_device).build_reconstruction_losses().build()

    models: dict[str, torch.nn.Module] = {"generator": model}
    optimizers: dict[str, torch.optim.Optimizer] = {"opt_g": optimizer}

    # GAN needs a second model + optimizer; without them the adversarial path
    # never fires and the run silently collapses to plain L1 (pitfall #9/#16).
    # A discriminator is wired whenever ONE IS SUPPLIED, not only for
    # ``paradigm == "gan"``. The gate used to be the paradigm name, so
    # ``fit(paradigm="diffusion", discriminator=d)`` accepted the argument and
    # silently dropped it -- the model never entered the env, no ``opt_d`` was
    # built, and the run trained exactly as if nothing had been passed. That is
    # the advertised-but-unwired shape (non-negotiable 16) one level above the
    # strategy: `DiffusionTrainingStrategy` can now train a critic, and this is
    # what lets a caller give it one.
    #
    # GAN keeps its extra REQUIREMENT (a GAN without a discriminator is not a
    # GAN); every other paradigm treats it as the opt-in additive it is.
    disc = discriminator
    if disc is None and paradigm == "gan":
        disc = _build_discriminator_from_config(settings, torch_device)
    if paradigm == "gan" and disc is None:
        raise ValueError(
            "fit(paradigm='gan') requires a discriminator: pass "
            "discriminator=<nn.Module>, or set model.discriminator_component "
            "in config so it can be built."
        )
    if disc is not None:
        models["discriminator"] = disc
        # role="discriminator" is what picks up optimization.discriminator_
        # learning_rate (TTUR). Without it D trained at G's LR here while the
        # same YAML gave D its own LR under `spectramr train`.
        optimizers["opt_d"] = (
            opt_d
            or OptimizerBuilder(settings, params=disc.parameters())
            .with_role("discriminator")
            .validate()
            .build()
        )

    env = TrainingEnvironment.from_components(
        config=settings,
        models=models,
        optimizers=optimizers,
        losses=losses,
        data_loaders={
            "train": train_loader,
            # `is not None` (not truthiness): a real-but-empty val loader is
            # falsy, so `val_loader or train_loader` would silently swap in the
            # train loader for validation. Only an unset (None) val loader
            # should fall back to the train loader.
            "val": val_loader if val_loader is not None else train_loader,
        },
        device=torch_device,
    )

    result = run_training_pipeline(settings, device=str(torch_device), env=env, strategy=strategy)
    if raise_on_failure and result.get("success") is not True:
        raise TrainingFailedError(
            f"fit(paradigm={paradigm!r}) did not succeed: "
            f"{result.get('error', '<no error reported>')}"
        )
    return result


def _build_evaluation_loop(
    settings: TrainingSettings,
    env: TrainingEnvironment,
    *,
    device: torch.device,
    strategy: Any | None = None,
) -> Any:
    """Assemble a :class:`TrainingLoop` for an evaluation-only pass.

    Reuses the SAME SSOT build path as ``run_training_pipeline`` — the DI
    container resolves the services and ``TrainingStrategyFactory`` builds the
    strategy from ``env.config`` — but stops at the loop object so the caller
    invokes ``.evaluate()`` instead of ``.run()`` (no training, no optimizer
    steps). Lazy imports keep fit.py's import graph shallow and mirror the lazy
    ``TrainingLoop`` import ``run_training_pipeline`` itself uses.
    """
    from spectramr import bootstrap
    from spectramr.domain.interfaces.service_interfaces import (
        ICheckpointService,
        ILoggingService,
        IMetricsService,
    )
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
    from spectramr.pipelines.training_loop import TrainingLoop

    container = bootstrap.build_container(settings, device=str(device))
    logging_service = container.resolve(ILoggingService)
    metrics_service = container.resolve(IMetricsService)
    checkpoint_service = container.resolve(ICheckpointService)

    if strategy is None:
        strategy = TrainingStrategyFactory().create_strategy(
            env=env,
            logging_service=logging_service,
            metrics_service=metrics_service,
            checkpoint_service=checkpoint_service,
        )

    return TrainingLoop(
        strategy,
        env,
        settings,
        settings.model.model_type,
        logging_service=logging_service,
        metrics_service=metrics_service,
        checkpoint_service=checkpoint_service,
    )


class Trainer:
    """Fluent, reusable wrapper over :func:`fit`.

    Holds shared run options (paradigm, config, device, length, a custom strategy)
    so the same configuration can be applied to multiple ``fit`` calls:

    >>> trainer = Trainer(paradigm="gan", device="cuda", epochs=50)
    >>> trainer.fit(model, train_loader, discriminator=disc, val_loader=val_loader)

    This is the public scripting ``Trainer``. It is distinct from the internal
    optimizer-step executor
    ``spectramr.infrastructure.training.step_executor.StepExecutor``
    (which is never part of the public surface).
    """

    def __init__(
        self,
        *,
        paradigm: str = "reconstruction",
        config: TrainingSettings | dict[str, Any] | None = None,
        strategy: Any | None = None,
        device: str | torch.device = "cpu",
        epochs: int | None = None,
        max_iterations: int | None = None,
        raise_on_failure: bool = True,
    ) -> None:
        self.paradigm = paradigm
        self.config = config
        self.strategy = strategy
        self.device = device
        self.epochs = epochs
        self.max_iterations = max_iterations
        self.raise_on_failure = raise_on_failure

    def fit(
        self,
        model: torch.nn.Module,
        train_loader: Any,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        discriminator: torch.nn.Module | None = None,
        opt_d: torch.optim.Optimizer | None = None,
        val_loader: Any | None = None,
        losses: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """Run :func:`fit` with this trainer's stored options."""
        return fit(
            model,
            train_loader,
            paradigm=self.paradigm,
            optimizer=optimizer,
            discriminator=discriminator,
            opt_d=opt_d,
            val_loader=val_loader,
            losses=losses,
            config=self.config,
            strategy=self.strategy,
            device=self.device,
            epochs=self.epochs,
            max_iterations=self.max_iterations,
            raise_on_failure=self.raise_on_failure,
        )

    def evaluate(
        self,
        model: torch.nn.Module,
        val_loader: Any,
        *,
        losses: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, float]:
        """Validate ``model`` on ``val_loader`` in-process — no training.

        Builds the same env + strategy :meth:`fit` would (under this trainer's
        paradigm/config/device), then runs ONE validation pass through the shared
        ``TrainingLoop.evaluate`` (eval mode + EMA-swap, no optimizer steps) and
        returns the aggregated metric dict. The numbers match in-training
        validation because the SAME ``_run_validation`` is driven.

        Note: builds a generator-only env (the validation path is
        generator-centric); pass a pre-built ``strategy`` to the ``Trainer`` for
        paradigms whose ``validation_step`` needs more than the generator.
        """
        settings = _resolve_fit_config(
            self.config,
            paradigm=self.paradigm,
            epochs=self.epochs,
            max_iterations=self.max_iterations,
        )
        torch_device = torch.device(self.device) if isinstance(self.device, str) else self.device

        # An optimizer is required to build the env but is NEVER stepped here.
        optimizer = OptimizerBuilder(settings, params=model.parameters()).validate().build()
        if losses is None:
            losses = LossBuilder(settings, torch_device).build_reconstruction_losses().build()

        env = TrainingEnvironment.from_components(
            config=settings,
            models={"generator": model},
            optimizers={"opt_g": optimizer},
            losses=losses,
            data_loaders={"train": val_loader, "val": val_loader},
            device=torch_device,
        )
        loop = _build_evaluation_loop(settings, env, device=torch_device, strategy=self.strategy)
        return loop.evaluate()

    def predict(
        self,
        *,
        config_path: str | Path,
        checkpoint_path: str | Path,
        input_path: str | Path,
        output_path: str | Path,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Run the SSOT inference pipeline on on-disk inputs.

        Delegates to :func:`spectramr.pipelines.infer.run_inference_pipeline`,
        which is config + checkpoint driven by design — data loading and result
        writes live in the data layer (via ``OutputWriter``), never here
        (pitfall #7/#11). It is therefore **path-based**, distinct from the
        in-memory :meth:`fit` / :meth:`evaluate`: inference reconstructs the run
        from its training YAML + a saved checkpoint, not a live model object.
        """
        from spectramr.pipelines.infer import run_inference_pipeline

        torch_device = torch.device(self.device) if isinstance(self.device, str) else self.device
        return run_inference_pipeline(
            Path(config_path),
            Path(checkpoint_path),
            Path(input_path),
            Path(output_path),
            device=str(torch_device),
            batch_size=batch_size,
        )
