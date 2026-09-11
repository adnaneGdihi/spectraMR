"""Capability metadata for the registry-dispatcher pattern.

Each registered component (model, loss, strategy, adapter) can declare
what shapes / domains / channel layouts it expects and produces. The
audit then mechanically cross-checks data → model → loss → strategy at
config-load time, instead of waiting for the first conv to crash.

All capability fields are :data:`Optional` and default to ``None`` —
which the audit interprets as "unannotated, skip the check". This keeps
the rollout non-breaking: existing registrations are unaffected, new
capability fields apply only to components that opt in. See
``docs/superpowers/specs/2026-05-05-experiment-spec-card-and-adapters-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from spectramr.config.schemas.enums import Regime, Task

# Component domains. Keep the literal closed so a typo in a decorator
# fails the import, not at audit time.
Domain = Literal[
    "image",  # real-valued magnitude image, [B, C, H, W] or [B, C, H, W, D]
    "kspace",  # k-space, real/imag interleaved as 2C channels
    "complex_image",  # complex image (torch.complex tensor or 2C interleaved)
    "latent",  # learned latent (post-VAE encoder)
    "pde_grid",  # PDE benchmark grid (Burgers, Darcy, etc.)
    "mesh",  # graph / mesh data (graph_mri)
    "spectrum",  # MRS free induction decay / spectrum, real/imag as 2T channels
]

ChannelSpec = Literal["any", "matches_input", "even", "complex_pair"]


@dataclass(frozen=True)
class ModelCapabilities:
    """What a registered model declares it can consume and produce.

    All fields default to ``None`` ("unannotated"). The audit's
    ``check_data_model_compatibility`` skips any unannotated field.

    Domain fields accept either a single ``Domain`` literal or a tuple
    of literals when the model genuinely handles multiple domains
    (e.g. cs_mno_operator works on both ``pde_grid`` and ``image``
    real-valued grid tensors).
    """

    spatial_dims: tuple[int, ...] | None = None  # e.g. (2,), (3,), (2, 3)
    input_domain: Domain | tuple[Domain, ...] | None = None
    output_domain: Domain | tuple[Domain, ...] | None = None
    accepts_complex: bool | None = None  # torch.complex tensor input
    expects_real_imag_interleaved: bool | None = None  # 2C real channels = C complex coils
    requires_paired_data: bool | None = None  # cycle GANs etc set False explicitly
    # Conditioning capabilities. These say what EXTRA SIGNAL the model's
    # ``forward`` consumes, not what shape it consumes -- so they are
    # deliberately NOT part of ``CONTRACT_FIELDS`` below.
    #
    # ``supports_contrast_conditioning`` is True when ``forward`` accepts a
    # ``contrast_idx`` / ``contrast_id`` tensor and actually uses it for
    # FiLM-style conditioning. ``supports_vendor_conditioning`` is the same
    # statement for a vendor id.
    #
    # ``None`` means unannotated, exactly as for every other field; ``False``
    # is a positive claim that the model ignores the id. The distinction is
    # load-bearing here: the audit fails an arm that enables multi-contrast
    # against a model that ignores the id, and "nobody said" must not read as
    # "declared unsupported".
    supports_contrast_conditioning: bool | None = None
    supports_vendor_conditioning: bool | None = None
    # Physical-quantity tags for the field/trajectory metrics. A model that
    # emits a B0 field declares output_field_units="Hz"; a spiral-trajectory
    # estimator declares trajectory_parametrization="spiral". The field-domain
    # metrics + the Tier-1 audit read these to enforce the parametrization
    # guard (pitfall #16): a metric that grades Hz-vs-Hz refuses to grade a
    # model whose declared output is an image, and a trajectory metric refuses
    # a Cartesian-per-line estimate against a spiral reference.
    output_field_units: str | None = None  # "Hz" for a B0/off-resonance field
    trajectory_parametrization: str | None = None  # "spiral" | "cartesian" | "radial"
    # Workflow tagging (imaging-regime × task contract). ``None`` = unannotated
    # = "agnostic, skip the check" — see module docstring. Only tag components
    # that are genuinely regime/task-specific.
    workflows: frozenset[Regime] | None = None
    tasks: frozenset[Task] | None = None


#: The fields that constitute a model's *dimension contract* -- the shape/domain
#: agreement the audit cross-checks data → model → loss against.
#:
#: This is deliberately narrower than "every field of :class:`ModelCapabilities`".
#: Two audit gates need to ask "did this model declare a contract?", and both
#: used to answer it with "is ``get_model_capabilities`` non-None?" -- a proxy
#: that was only ever correct while the dataclass held nothing BUT contract
#: fields. Adding the conditioning flags broke that proxy: a model declaring
#: only ``supports_contrast_conditioning`` would have started reading as
#: contract-annotated and silently dropped out of the default-deny bucket.
#:
#: One constant, imported by both gates, so the two cannot drift apart
#: (non-negotiable 17).
CONTRACT_FIELDS: tuple[str, ...] = ("spatial_dims", "input_domain", "output_domain")


@dataclass(frozen=True)
class LossCapabilities:
    """What a registered loss declares it operates on."""

    domain: Domain | None = None
    #: The loss imposes NO domain constraint -- ``l1``, ``l2``, ``bce`` and the
    #: other elementwise distances are defined on any tensor, so they are legal
    #: under every loss block.
    #:
    #: This is deliberately a flag rather than a ``Domain`` member. A signal
    #: domain says what the tensor *is*; "agnostic" says the loss does not care,
    #: which is a different kind of statement -- admitting it into the literal
    #: would put a non-domain in the vocabulary ``SignalDomain`` exists to keep
    #: closed.
    #:
    #: It is also distinct from ``domain=None``. ``None`` means *unannotated*:
    #: nobody has said what the loss operates on, so the block check has to skip
    #: it. ``domain_agnostic=True`` is a positive claim that skipping is correct
    #: and permanent. Collapsing the two makes an un-audited loss and a
    #: deliberately-generic one indistinguishable, which is what let 104 of 214
    #: registrations sit uncovered while reading as intentional.
    domain_agnostic: bool = False
    spatial_dims: tuple[int, ...] | None = None
    channels: ChannelSpec | None = None
    requires_paired_target: bool | None = None
    # Workflow tagging (imaging-regime × task contract). ``None`` = unannotated
    # = "agnostic, skip the check" — see module docstring. Only tag components
    # that are genuinely regime/task-specific.
    workflows: frozenset[Regime] | None = None
    tasks: frozenset[Task] | None = None


@dataclass(frozen=True)
class StrategyCapabilities:
    """Declarative metadata for a training strategy.

    ``required_config_fields`` is the field that catches the stage2
    ``Missing required config: training.diffusion.num_timesteps``-class
    bug — listed dotted paths must resolve to non-None at audit time.
    """

    required_config_fields: tuple[str, ...] = ()
    supported_paradigms: frozenset[str] = field(default_factory=frozenset)
    #: Validation keys the strategy's own ``validation_step`` writes directly
    #: (``val_field_mse``, ...), bypassing the MetricsRegistry computer. Declared
    #: so ``validation.scoring.compute`` / ``best_metric_name`` can be validated
    #: against registry-or-emitted instead of against nothing (cohort review
    #: 2026-09-02, T0.5). Empty means "undeclared", which the gate reports
    #: rather than punishes -- the ratchet promotes it once the census is clean.
    emitted_metrics: frozenset[str] = field(default_factory=frozenset)
    # Workflow tagging (imaging-regime × task contract). ``None`` = unannotated
    # = "agnostic, skip the check" — see module docstring. Only tag components
    # that are genuinely regime/task-specific.
    workflows: frozenset[Regime] | None = None
    tasks: frozenset[Task] | None = None


@dataclass(frozen=True)
class AdapterCapabilities:
    """A bridge between two component capability shapes.

    The audit's adapter-chain composer verifies, for each declared
    chain, that successive ``bridges_to`` matches the next adapter's
    ``bridges_from``. Side-effects are surfaced verbatim on the Spec
    Card so reviewers see the cost of every adapter.
    """

    bridges_from: dict[str, object] = field(default_factory=dict)
    bridges_to: dict[str, object] = field(default_factory=dict)
    invertible: bool = False
    side_effects: tuple[str, ...] = ()
    insertion_points: tuple[
        str, ...
    ] = ()  # pre_model | post_model | pre_loss_pred | pre_loss_target | pre_metric
    # Workflow tagging (imaging-regime × task contract). ``None`` = unannotated
    # = "agnostic, skip the check" — see module docstring. Only tag components
    # that are genuinely regime/task-specific.
    workflows: frozenset[Regime] | None = None
    tasks: frozenset[Task] | None = None


VALID_INSERTION_POINTS: frozenset[str] = frozenset(
    {"pre_model", "post_model", "pre_loss_pred", "pre_loss_target", "pre_metric"}
)


__all__ = [
    "VALID_INSERTION_POINTS",
    "AdapterCapabilities",
    "ChannelSpec",
    "Domain",
    "LossCapabilities",
    "ModelCapabilities",
    "StrategyCapabilities",
]
