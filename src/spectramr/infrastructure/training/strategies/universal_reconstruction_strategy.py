"""Universal / foundation reconstruction strategy (PR-7).

One model, many regimes. This thin subclass of
:class:`ReconstructionTrainingStrategy` adds the framework's foundation-recon
differentiator: the generator is conditioned on a **physics-derived prompt
embedding** (from acquisition parameters TR / TE / flip-angle / field-strength)
rather than on empirically-learned prompt tokens, so a single model
generalises across mixed contrast / field-strength / acceleration cohorts.

Data side (not reimplemented here)
----------------------------------
The actual multi-manifold / multi-domain loading routes through
:meth:`DataPipelineDirector.build_multi_domain_dataloaders` (see
``.claude/rules/data.md``). This strategy does **not** touch data loading; it
only *accepts and stamps* a ``task_mixture`` config (a per-cohort sampling
weight map) so the resolved mixture is recorded in the run's provenance, per
CLAUDE.md pitfall #15 (every exposed knob is read + stamped).

Override pattern follows
``stochastic_interpolants_strategy.StochasticInterpolantsStrategy``: a thin
``_compute_losses_impl`` that reads the generator via
``getattr(self.env, "generator", None)``, the batch via
``self._resolve_legacy_batch``, and ``self.device``.

Registration (returned to the caller — do NOT edit strategy_factory.py here)::

    "universal_reconstruction": "spectramr.infrastructure.training.strategies.universal_reconstruction_strategy.UniversalReconstructionStrategy",
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch.nn.functional import l1_loss

from spectramr.data.batch_types import read_batch_field
from spectramr.models.blocks.physics_prompt import PhysicsPromptEmbedding

from .mixins.utils import _callable_accepts_kwarg
from .reconstruction import ReconstructionTrainingStrategy


class UniversalReconstructionStrategy(ReconstructionTrainingStrategy):
    """Foundation reconstruction conditioned on a physics-derived prompt.

    The prompt embedding is lazily constructed on first use so the strategy
    can be instantiated through the standard ``ReconstructionTrainingStrategy``
    path without any new schema keys (``training_mode`` alone selects it).
    """

    #: Loss ownership (issue #1918). Computes l1_loss(pred, target) inline -- image space, against the target --
    #: and returns it as the whole objective. Declared so the fold skips it.
    #: CAVEAT: the weight is hardcoded 1.0, not the declared lambda_l1.
    inline_losses: ClassVar[frozenset[str]] = frozenset({"l1"})
    folds_image_losses: ClassVar[bool] = False

    #: Acquisition-parameter layout the physics prompt expects.
    PROMPT_PARAM_ORDER = ("TR", "TE", "flip_angle", "field_strength_T")
    #: Conditioning embedding dimension (constructor-default knob).
    PROMPT_EMBED_DIM = 128

    def _get_prompt_embedder(self, in_features: int) -> PhysicsPromptEmbedding:
        embedder = getattr(self, "_prompt_embedder", None)
        if embedder is None or embedder.in_features != in_features:
            embedder = PhysicsPromptEmbedding(
                in_features=in_features,
                embed_dim=self.PROMPT_EMBED_DIM,
            ).to(self.device)
            self._prompt_embedder = embedder
            # The prompt embedder is learnable (two nn.Linear heads); register
            # it on opt_g or the physics-derived conditioning stays at random
            # init. The in_features rebuild guard above prevents a double-add.
            opt = getattr(self.env, "opt_g", None)
            if opt is not None:
                existing = {p for g in opt.param_groups for p in g["params"]}
                new_params = [p for p in embedder.parameters() if p not in existing]
                if new_params:
                    opt.add_param_group({"params": new_params})
        return embedder

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # Stamp the task-mixture knob (read from kwarg, else keep any preset
        # attr) into run provenance — pitfall #15: an advertised knob must be
        # read and recorded, even though the actual multi-domain sampling is
        # done upstream by DataPipelineDirector.build_multi_domain_dataloaders.
        if "task_mixture" in kwargs:
            self.task_mixture = kwargs["task_mixture"]
        elif not hasattr(self, "task_mixture"):
            self.task_mixture = None

        batch = self._resolve_legacy_batch(input_batch, kwargs)
        gen = getattr(self.env, "generator", None)

        target = batch.get("target") if isinstance(batch, dict) else target_batch
        source = batch.get("input") if isinstance(batch, dict) else input_batch
        if source is None and isinstance(batch, dict):
            source = batch.get("source")

        if gen is None or target is None or source is None:
            return {"loss_total": torch.tensor(0.0, device=self.device)}

        # Physics-derived conditioning — only when the batch carries the
        # acquisition parameters. Absent params => unconditioned forward
        # (graceful skip, no silent default tokens).
        prompt = None
        acq = read_batch_field(batch, "acquisition_params")
        if acq is not None:
            acq = acq.to(self.device)
            embedder = self._get_prompt_embedder(acq.shape[-1])
            prompt = embedder(acq)

        pred = self._forward_generator(gen, source, prompt)
        loss = l1_loss(pred, target)
        return {"loss_total": loss, "loss_recon_l1": loss.detach()}

    @staticmethod
    def _forward_generator(
        gen: Any,
        source: torch.Tensor,
        prompt: torch.Tensor | None,
    ) -> torch.Tensor:
        """Forward the generator, passing ``prompt`` only when supported.

        A plain reconstruction backbone whose ``forward`` does not accept a
        ``prompt`` kwarg is run unconditioned — the conditioning is skipped
        gracefully rather than crashing.

        Introspection is the whole decision (SAQ-001). A ``try:
        gen(source, prompt=prompt) except TypeError: gen(source)`` tail used to
        follow this check and contradicted it twice: it re-attempted the kwarg
        call that ``_callable_accepts_kwarg`` had just ruled out, and when
        ``prompt`` was ``None`` it retried ``gen(source)`` with the identical
        ``gen(source)`` — a call that can only raise the same error again, so
        the sole effect either way was to swallow a genuine in-forward
        ``TypeError`` and blame it on the signature.

        One behavioural consequence, deliberate: a generator whose signature
        cannot be introspected makes ``_callable_accepts_kwarg`` return
        ``False``, so it is now run unconditioned instead of being probed with
        the kwarg first. That matches how every other conditioning gate in this
        package resolves an unreadable signature.
        """
        if prompt is not None and _callable_accepts_kwarg(gen.forward, "prompt"):
            return gen(source, prompt=prompt)
        return gen(source)
