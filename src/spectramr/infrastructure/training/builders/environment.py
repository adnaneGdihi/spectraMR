"""Training Environment Dataclass

Immutable container for all training components built by TrainingEnvironmentDirector.
All components are pre-validated and ready for use in training strategies.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from spectramr.config.settings import TrainingSettings
from spectramr.core.compute_device import resolve_torch_device


def _resolve_device(device: "torch.device | str | None", config: TrainingSettings) -> torch.device:
    """Resolve a device spec to a concrete :class:`torch.device`.

    Accepts a ``torch.device`` (taken as-is — the caller already resolved it), a
    string, ``"auto"``, or ``None`` (falls back to ``config.device``).

    Resolution is delegated to the SSOT policy in
    :mod:`spectramr.core.compute_device`, so ``"auto"`` on a GPU-less host RAISES
    for this (heavy, training) path instead of quietly yielding CPU. The old
    ``"cuda" if torch.cuda.is_available() else "cpu"`` line and the ``None →
    "cpu"`` default were two more of the eight silent downgrades.
    """
    if isinstance(device, torch.device):
        return device
    # `run.device` since phase 4b; the flat read returned None for every config,
    # so this handed `spec=None` to the resolver and threw away the declaration.
    if device is not None:
        spec = device
    else:
        spec = getattr(getattr(config, "run", None), "device", None) or getattr(
            config, "device", None
        )
    decision = resolve_torch_device(spec, pipeline="train", source="config.device")
    return torch.device(decision.device)


@dataclass(frozen=True)
class TrainingEnvironment:
    """Immutable container for all training components.

    Built by TrainingEnvironmentDirector using specialized builders.
    All components are pre-validated and ready for use.

    Attributes:
        models: Dictionary of neural network modules
            (generator, discriminator, encoder, decoder)
        optimizers: Dictionary of optimizers (one per model or combined)
        schedulers: Dictionary of learning rate schedulers
        losses: Dictionary of loss functions
            (reconstruction, adversarial, physics, etc.)
        physics: Dictionary of physics operators (FFT, mask generator, data consistency)
        data_loaders: Dictionary of data loaders (train, val, test)
        metrics: Dictionary of metric calculators (PSNR, SSIM, etc.)
        scaler: Optional gradient scaler for mixed precision training
        device: Device where models are placed (cuda/cpu)
        config: Immutable training configuration

    Example:
        >>> director = TrainingEnvironmentDirector(config)
        >>> env = director.build_environment()
        >>> # Access components
        >>> generator = env.models["generator"]
        >>> optimizer = env.optimizers["opt_g"]
        >>> l1_loss = env.losses["l1"]
    """

    models: dict[str, nn.Module]
    optimizers: dict[str, Optimizer]
    schedulers: dict[str, _LRScheduler]
    losses: dict[str, nn.Module | None]
    physics: dict[str, Any]
    data_loaders: dict[str, DataLoader]
    metrics: dict[str, Any]
    scaler: GradScaler | None
    device: torch.device
    config: TrainingSettings
    ema: Any | None = None
    #: Resolved parallelism (``ParallelRuntime``), or ``None`` for single-process.
    #: Carries the strategy name, the step policy that strategy requires (FSDP
    #: needs a sharding-correct gradient clip), and what to stamp into provenance.
    parallel: Any | None = None

    #: What compilation actually did, per model: the stage it ran at and the
    #: resolved mode/backend. Empty when the arm asked for eager. Declared vs
    #: applied is the distinction that matters -- the config says what was
    #: asked for, this says what happened, and `DDP(compile(m))` and
    #: `compile(DDP(m))` are indistinguishable without it.
    compile: Any | None = None

    @classmethod
    def from_components(
        cls,
        *,
        config: TrainingSettings,
        models: dict[str, nn.Module],
        optimizers: dict[str, Optimizer],
        losses: dict[str, nn.Module | None] | None = None,
        schedulers: dict[str, _LRScheduler] | None = None,
        physics: dict[str, Any] | None = None,
        data_loaders: dict[str, DataLoader] | None = None,
        metrics: dict[str, Any] | None = None,
        scaler: GradScaler | None = None,
        device: torch.device | str | None = None,
        ema: Any | None = None,
    ) -> "TrainingEnvironment":
        """Build a ``TrainingEnvironment`` from hand-supplied components.

        The scripting-API peer of ``TrainingEnvironmentDirector.build_environment``:
        it constructs the SAME immutable container from loose objects so the same
        loop + strategies run unchanged. The caller must place the model under
        ``models["generator"]`` and its optimizer under ``optimizers["opt_g"]``
        (the canonical keys every strategy reads via :attr:`generator` /
        :attr:`opt_g`); adversarial paradigms add ``"discriminator"`` / ``"opt_d"``.

        Absent collections become empty dicts (never ``None`` — the loop indexes
        them). ``device`` accepts a :class:`torch.device`, a string, ``"auto"``,
        or ``None`` (resolved from ``config.device``); the frozen instance is
        constructed, never mutated.
        """
        resolved_device = _resolve_device(device, config)
        return cls(
            models=models,
            optimizers=optimizers,
            schedulers=schedulers or {},
            losses=losses or {},
            physics=physics or {},
            data_loaders=data_loaders or {},
            metrics=metrics or {},
            scaler=scaler,
            device=resolved_device,
            config=config,
            ema=ema,
        )

    @property
    def generator(self) -> nn.Module:
        """Access generator model from models dict."""
        return self.models.get("generator")

    @property
    def discriminator(self) -> nn.Module | None:
        """Access discriminator model from models dict."""
        return self.models.get("discriminator")

    @property
    def opt_g(self) -> torch.optim.Optimizer | None:
        """Access generator optimizer from optimizers dict (canonical key 'opt_g')."""
        return self.optimizers.get("opt_g")

    @property
    def opt_d(self) -> torch.optim.Optimizer | None:
        """Access discriminator optimizer from optimizers dict."""
        return self.optimizers.get("opt_d")

    @property
    def optimizer_g(self) -> torch.optim.Optimizer | None:
        """Alias for opt_g — required by CheckpointDirector interface."""
        return self.opt_g

    @property
    def optimizer_d(self) -> torch.optim.Optimizer | None:
        """Alias for opt_d — required by CheckpointDirector interface."""
        return self.opt_d

    @property
    def scheduler_g(self) -> Any | None:
        """Access generator scheduler — required by CheckpointDirector interface."""
        return self.schedulers.get("opt_g") or self.schedulers.get("main")

    @property
    def scheduler_d(self) -> Any | None:
        """Access discriminator scheduler — required by CheckpointDirector interface."""
        return self.schedulers.get("opt_d")

    @property
    def model_type(self) -> str:
        """Get model type from config."""
        return self.config.model.model_type

    @property
    def run_name(self) -> str:
        """Get run name from config (SSOT: logging.run_name or folder name)."""
        if self.config.logging.identity.run:
            return self.config.logging.identity.run
        return self.config.metadata.name if self.config.metadata else "unknown_run"

    @property
    def loss_function(self) -> dict[str, nn.Module | None]:
        """Alias for losses dict (SSOT: backward compatibility for loss computers).

        Loss computers expect self.state.loss_function, but TrainingEnvironment
        stores losses in self.losses. This property provides the expected interface.
        """
        return self.losses
