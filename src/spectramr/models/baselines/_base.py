"""Base contract for prior-method baseline adapters.

See ``TODO/backlog_baseline_replication_experiment_11.md`` for the
replication design. The base class is **deliberately thin**: it
captures only what every adapter must declare for provenance,
canonical-layout interop, and metric comparability — it does NOT
inherit from any specific paradigm class. Each adapter picks its
own forward / sampling implementation while routing loss + mask
generation through this repo's existing Strategy and physics SSOT.
"""

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from enum import StrEnum
from typing import ClassVar

import torch
import torch.nn as nn


class FFTNorm(StrEnum):
    """FFT normalisation conventions.

    This repo's canonical convention is ``"ortho"`` (orthonormal,
    centered). Adapters whose upstream uses a different convention
    must declare it here so a shim can be wrapped at the adapter
    boundary; never reimplement the normalisation inside the model.
    """

    ORTHO = "ortho"
    BACKWARD = "backward"  # numpy / scipy default
    FORWARD = "forward"


class UpstreamLossFamily(StrEnum):
    """The objective an upstream's own training step computes.

    Declared so a strategy can refuse an arm whose ``losses:`` block names a
    different family from the one the paper's code will actually minimise
    (non-negotiable 8). CDiffMR and Shen both use L1; FDB uses MSE on x_0.
    """

    L1 = "l1"
    L2 = "l2"


class CoilHandling(StrEnum):
    """Coil-combination strategies expected by the adapter."""

    RSS = "rss"
    MULTI_COIL_KSPACE = "multi_coil_kspace"
    SENSE_COMBINED = "sense_combined"


class _BaselineAdapterMeta(ABCMeta):
    """Check a subclass's declarations once its abstractness is actually known.

    ``__init_subclass__`` runs inside ``type.__new__``, which ``ABCMeta.__new__``
    calls *before* it computes ``__abstractmethods__`` -- and that name is a type
    slot rather than an inherited class attribute, so reading it there raises
    ``AttributeError`` and every "is this class abstract?" test answered no. The
    escape hatch for abstract intermediates was therefore unreachable, and the
    check fired on the classes it was written to skip (#2157).
    """

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, object],
        **kwargs: object,
    ) -> _BaselineAdapterMeta:
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        if cls.__abstractmethods__:
            return cls
        validate = namespace.get("_validate_concrete_declarations") or getattr(
            cls, "_validate_concrete_declarations", None
        )
        if validate is not None:
            cls._validate_concrete_declarations()
        return cls


class BaselineAdapter(nn.Module, ABC, metaclass=_BaselineAdapterMeta):
    """Base contract for adapters that wrap an upstream baseline.

    Subclass requirements:

    1. Declare the class attributes ``REPO_NAME``, ``PAPER_REF``,
       ``PREFERRED_MASK_TYPE``, ``PREFERRED_FFT_NORM``, ``COIL_HANDLING``
       and ``UPSTREAM_LOSS_FAMILY``. A class-creation check refuses any
       left at the sentinel ``"<MUST_OVERRIDE>"`` — the CLAUDE.md
       pitfall #9 guard against silent fallbacks.
    2. Implement :meth:`forward` returning a tensor in this repo's
       canonical layout: ``[B, C, H, W]`` with complex dtype,
       ``fft2c``-centred when in k-space, and :meth:`training_loss`
       returning the objective the upstream's own step minimises.
    3. Optionally override :meth:`provenance` to return a dict that
       extends the default with adapter-specific keys (e.g., upstream
       commit SHA — collected by ``baseline_provenance`` in
       ``src/infrastructure/reporting/metadata.py``).

    The check in (1) applies to CONCRETE subclasses only: an abstract
    intermediate that still carries an ``@abstractmethod`` may leave the
    sentinels, which is why it hangs off :class:`_BaselineAdapterMeta`
    rather than ``__init_subclass__`` (#2157).
    """

    REPO_NAME: ClassVar[str] = "<MUST_OVERRIDE>"
    PAPER_REF: ClassVar[str] = "<MUST_OVERRIDE>"
    PREFERRED_MASK_TYPE: ClassVar[str] = "<MUST_OVERRIDE>"
    PREFERRED_FFT_NORM: ClassVar[FFTNorm] = FFTNorm.ORTHO
    COIL_HANDLING: ClassVar[CoilHandling] = CoilHandling.RSS

    _REQUIRED_OVERRIDES: ClassVar[tuple[str, ...]] = (
        "REPO_NAME",
        "PAPER_REF",
        "PREFERRED_MASK_TYPE",
    )

    @classmethod
    def _validate_concrete_declarations(cls) -> None:
        """Every declaration a CONCRETE adapter owes, checked at class creation.

        Driven by :class:`_BaselineAdapterMeta` rather than
        ``__init_subclass__``, which cannot yet tell an abstract intermediate
        from a finished adapter.
        """
        missing = [
            name for name in cls._REQUIRED_OVERRIDES if getattr(cls, name) == "<MUST_OVERRIDE>"
        ]
        if missing:
            raise TypeError(
                f"BaselineAdapter subclass {cls.__name__} must override class "
                f"attribute(s) {missing}. Silent fallbacks are forbidden."
            )
        cls._reject_unresolvable_mask_type()
        cls._reject_unknown_loss_family()

    @classmethod
    def _reject_unknown_loss_family(cls) -> None:
        """The declared objective must be one the strategy can check an arm against."""
        family = getattr(cls, "UPSTREAM_LOSS_FAMILY", None)
        if family is None or str(family) not in {m.value for m in UpstreamLossFamily}:
            raise TypeError(
                f"BaselineAdapter subclass {cls.__name__} declares "
                f"UPSTREAM_LOSS_FAMILY={family!r}. Declare the objective the "
                f"upstream's own training step minimises, one of: "
                f"{sorted(m.value for m in UpstreamLossFamily)}"
            )

    @classmethod
    def _reject_unresolvable_mask_type(cls) -> None:
        """A convention the framework cannot resolve is worse than an absent one.

        ``gaussian_density`` was declared by two adapters and names neither an
        accelerator nor a ``MaskType`` -- so the attribute read as a checked
        convention while resolving to nothing (#2087). Both vocabularies are
        legal: the accelerator registry drives the timestep ladder, ``MaskType``
        covers the static path, and FDB's peripheral-to-central mask exists only
        on the latter.

        Imported inside the method rather than at module scope to keep
        class-definition time free of a models -> infrastructure import edge;
        the dependency itself is allowed (the ``infrastructure.physics``
        carve-out, non-negotiable 5).
        """
        from spectramr.infrastructure.physics.sampling import MaskType
        from spectramr.infrastructure.physics.sampling_registry import (
            SamplingPatternRegistry,
        )

        known = set(SamplingPatternRegistry.list_accepted()) | {m.value for m in MaskType}
        if cls.PREFERRED_MASK_TYPE not in known:
            raise TypeError(
                f"BaselineAdapter subclass {cls.__name__} declares "
                f"PREFERRED_MASK_TYPE={cls.PREFERRED_MASK_TYPE!r}, which names no "
                f"registered accelerator and no MaskType. Declare one of: "
                f"{sorted(known)}"
            )

    @abstractmethod
    def training_loss(
        self,
        x_0: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run the AUTHORS' training step and return their loss.

        This is the seam that separates a reproduction from an architecture
        comparison. ``forward`` wraps the upstream *network*; this wraps the
        upstream *method* -- its degradation schedule, its timestep sampling and
        its objective -- by calling the upstream diffusion object rather than
        reimplementing it (#2080).

        Args:
            x_0: The clean target the upstream expects, ``[B, 2, H, W]`` real /
                imaginary unless the adapter documents otherwise.
            batch: The rest of the batch, for upstreams that need more than
                ``x_0`` (Shen needs the acquisition mask's patch geometry).

        Returns:
            A scalar loss, on the upstream's own scale and with no weight from
            this repository applied.
        """

    @abstractmethod
    def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor in canonical layout ``[B, C, H, W]``,
                complex dtype if k-space.

        Returns:
            Output tensor in canonical layout.
        """
        raise NotImplementedError

    def validation_sample(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run the AUTHORS' own multi-step reverse process, for validation.

        ``forward`` is one denoise step; grading it at whatever timestep an absent
        argument defaults to is not the paper's reconstruction procedure. This is the
        validation-side sibling of :meth:`training_loss`: an adapter that can run its
        reverse process overrides this to return an actual reconstruction, and
        :class:`~spectramr.infrastructure.training.strategies.upstream_process_strategy.UpstreamProcessStrategy`
        is the only caller (non-negotiable 6 -- resolved through the adapter, not a
        branch on baseline name).

        Args:
            input_batch: The loader's measurement -- the starting point for a reverse
                process that conditions on the acquisition (e.g. FDB's bridge).
            target_batch: The clean target -- the starting point for a reverse process
                that degrades the target itself, the same convention :meth:`training_loss`
                uses (e.g. CDiffMR, Shen).
            batch: The rest of the batch, for a reverse process that needs more than
                the two tensors (e.g. the acquisition mask).

        Returns:
            A reconstruction in this adapter's native real layout, same shape as
            ``target_batch``.

        Raises:
            NotImplementedError: the base implementation always raises. An adapter
                that cannot run its reverse process (no wiring, or plumbing this
                adapter cannot supply on its own) must say so here rather than let a
                caller fall back to a single ``forward()`` and report the result as
                this baseline's reconstruction metric (non-negotiable 3).
        """
        del input_batch, target_batch, batch
        raise NotImplementedError(
            f"{type(self).__name__} has no wired reverse-sampling validation path. "
            "A single forward() at whatever timestep is left to default must not be "
            "reported as this baseline's reconstruction metric."
        )

    def _record_unconsumed(self, kwargs: dict[str, object]) -> None:
        """Remember the ``**kwargs`` this adapter did not consume.

        Every adapter ends its signature with ``**kwargs``, so a YAML
        ``model_kwargs`` key that names nothing is accepted and dropped without
        a word. That is how ``baseline_fdb`` came to declare eleven knobs of
        which one was read, and how ``timesteps`` sat next to a parameter
        actually spelled ``bridge_steps`` — matching only because the default
        happened to agree.

        Rejecting them outright is the stricter fix and is NOT what this does:
        the factory injects framework-side kwargs (``kspace_log_scaled`` and
        friends) that no adapter signature names, so a bare ``if kwargs: raise``
        would refuse every arm. Recording them instead makes the drop **visible
        in the run summary** without changing what constructs, which is the half
        that can be fixed without a live training path to test against.
        """
        self._unconsumed_kwargs = sorted(str(k) for k in kwargs)

    def provenance(self) -> dict[str, str]:
        """Return adapter provenance metadata for the run summary.

        Override to add upstream-commit SHA or other adapter-specific
        keys. The base implementation returns the declared class-level
        attributes, plus whatever :meth:`_record_unconsumed` was handed.
        """
        return {
            "repo_name": self.REPO_NAME,
            "paper_ref": self.PAPER_REF,
            "preferred_mask_type": self.PREFERRED_MASK_TYPE,
            "preferred_fft_norm": self.PREFERRED_FFT_NORM.value,
            "coil_handling": self.COIL_HANDLING.value,
            # Empty string, not an absent key: "this adapter consumed everything
            # it was given" and "nobody asked the question" must not render alike.
            "unconsumed_model_kwargs": ", ".join(getattr(self, "_unconsumed_kwargs", [])),
        }
