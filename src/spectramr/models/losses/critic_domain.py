"""Put a critic's input in the domain that critic declared (#1920).

A diffusion arm may predict in k-space while its critic consumes images -- or
the exact reverse. Both are legitimate configurations, and until now the seam
between them was undeclared: ``_align_for_critic`` changed the *representation*
(complex -> two real channels) but never the *domain*, so a k-space generator
attached to an image critic handed it raw k-space and the critic scored the
wrong space silently.

The fix is a declaration on BOTH sides plus a conversion driven by it:

* the generator's domain comes from ``losses.policy.output_domain`` -- the same
  key ``LossBuilder._build_list_based_losses`` already trusts to decide which
  loss needs a Fourier bridge, so the critic seam and the loss bridge cannot
  disagree about what the generator emits;
* the critic's domain comes from its own ``@register_model(input_domain=...)``,
  read through ``get_model_capabilities``.

With both sides declared, the transform is **explicit**, not a guess. That is
the distinction non-negotiable 3 draws: an ``ifft2c`` inserted because the code
suspected a mismatch would be a silent substitution; one inserted because two
registrations state different domains is the declared conversion
``mixins/kspace.py:45-105`` already performs for model inputs.

**The class name is not the declaration.** Every critic in
``models/discriminators/kspace_discriminator.py`` is named for k-space and every
one of them consumes an **image** -- they call ``fft2c`` themselves
(``:168``, ``:377``). ``sense_bridge_patchgan`` is the mirror image: it consumes
**k-space** and bridges inward. Declaring a domain from the name would give the
first group ``F{F{x}}``, which is the spatially-reversed image: finite,
brain-shaped, and wrong. Read ``forward`` before declaring.

**Two feed points, one owner.** The critic is called from the D step
(``DiffusionTrainingStrategy._align_for_critic``) and, independently, from the G
step (``UnifiedDiffusionLossComputer.compute`` at
``unified_diffusion_reconstruction.py:511``, ``fake_pred = discriminator(pred)``).
Converting on one and not the other is worse than converting on neither: the
critic would then score images while training and k-space while scoring the
generator. :func:`to_critic_input` is the single owner both paths call.

**Consolidation target.** ``mixins/kspace.py:82-101`` performs the same
declared conversion for *model* inputs with its own copy of the reshape. The two
should elect one owner (non-negotiable 17); this module does not refactor it,
because that mixin serves 106 image-domain arms and its own reshape convention
is load-bearing there.
"""

from __future__ import annotations

from typing import Any

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.models.registry import get_model_capabilities

#: Domains that name the same physical space. ``image`` and ``complex_image``
#: differ in REPRESENTATION (one real channel vs. a real/imag pair), not in
#: domain, so crossing between them is the ``torch.cat`` realification below --
#: never a Fourier transform. Putting them in one group is what stops a
#: ``complex_image`` generator feeding an ``image`` critic through a spurious
#: FFT.
IMAGE_SIDE: frozenset[str] = frozenset({"image", "complex_image"})

#: The k-space side. Kept as a set rather than a bare string so a future
#: hybrid-space domain joins it without touching :func:`resolve_conversion`.
KSPACE_SIDE: frozenset[str] = frozenset({"kspace"})

#: Domains this seam can transform between. Mirrors the bridge matrix in
#: ``LossBuilder._build_list_based_losses`` (loss_builder.py:473-485), which
#: also transforms exactly ``kspace <-> image`` and raises on the rest. The
#: remaining ``Domain`` values -- ``latent``, ``pde_grid``, ``mesh``,
#: ``spectrum`` -- have no Fourier relationship to either side, so a mismatch
#: involving one of them is a configuration error, not a conversion.
TRANSFORMABLE: frozenset[str] = IMAGE_SIDE | KSPACE_SIDE


def _same_side(a: str, b: str) -> bool:
    """Whether two domain names refer to the same physical space."""
    return a == b or (a in IMAGE_SIDE and b in IMAGE_SIDE)


def resolve_conversion(from_domain: str | None, declared: Any) -> str | None:
    """The domain to convert INTO, or ``None`` when no transform is needed.

    ``declared`` is the critic's ``input_domain`` capability, which
    ``ModelCapabilities`` types as ``Domain | tuple[Domain, ...] | None`` -- a
    critic may accept several. A tuple that already contains the generator's
    domain means the critic is happy as it is, so nothing is converted; that is
    why membership is tested before a target is chosen.

    Returns ``None`` -- meaning "hand the tensor over untouched" -- whenever
    either side is undeclared. Before #1920 that was 15 of 21 registered
    discriminators; it is now 3, each undeclared with a stated reason in source
    because no word in the ``Domain`` vocabulary names its input. Those three
    still take this path and their behaviour is unchanged.

    Args:
        from_domain: the generator's declared output domain, or None.
        declared: the critic's declared ``input_domain``, or None.

    Returns:
        The single domain name to convert into, or None for no conversion.

    Raises:
        ValueError: the two sides are declared, differ, and at least one names
            a domain with no Fourier relationship to the other.
    """
    if from_domain is None or declared is None:
        return None

    accepted: tuple[str, ...] = (declared,) if isinstance(declared, str) else tuple(declared)
    if not accepted:
        return None

    if any(_same_side(from_domain, d) for d in accepted):
        return None

    target = accepted[0]
    if from_domain not in TRANSFORMABLE or target not in TRANSFORMABLE:
        raise ValueError(
            f"critic domain mismatch that no transform can bridge: the generator "
            f"declares output_domain={from_domain!r} (losses.policy.output_domain) "
            f"and the critic declares input_domain={accepted!r}. This seam converts "
            f"only between {sorted(TRANSFORMABLE)}; the pair above has no Fourier "
            f"relationship, so converting would invent data. Attach a critic whose "
            f"input_domain matches, or correct losses.policy.output_domain."
        )
    return target


def to_critic_input(
    x: torch.Tensor,
    *,
    from_domain: str | None = None,
    to_domain: Any = None,
    takes_complex: bool = False,
) -> torch.Tensor:
    """Render one tensor as the configured critic declared it wants it.

    Domain first, representation second -- and the order is not cosmetic.
    ``torch.cat([x.real, x.imag], dim=1)`` destroys the complex structure an
    FFT needs, so realifying before converting would transform a tensor whose
    channels no longer mean real and imaginary parts.

    ``fft2c``/``ifft2c`` route through ``fft_ops._to_complex``, which reads a
    real ``[..., C, H, W]`` with even ``C`` as INTERLEAVED ``[R1,I1,R2,I2,...]``
    -- exactly the layout the generator emits -- so an already-real fake needs
    no hand reassembly.

    One consequence worth stating, because the ``_align_for_critic`` docstring
    documents its absence: realification writes a BLOCK layout
    (``[R1..Rn,I1..In]``) while the generator emits interleaved. On the
    UNCONVERTED path that asymmetry survives, which is why a critic that
    declares ``accepts_complex`` is handed both sides untouched. On the
    CONVERTED path it disappears -- both sides pass through complex and come out
    block -- so conversion incidentally makes the two sides comparable.

    Args:
        x: the tensor destined for the critic.
        from_domain: the generator's declared output domain, or None.
        to_domain: the critic's declared ``input_domain``, or None.
        takes_complex: the critic's ``accepts_complex`` capability. When True
            the critic owns its own realification and receives complex.

    Returns:
        The tensor in the critic's domain and representation.
    """
    target = resolve_conversion(from_domain, to_domain)
    if target is not None:
        x = fft2c(x) if target in KSPACE_SIDE else ifft2c(x)

    if takes_complex:
        return x
    if torch.is_complex(x):
        return torch.cat([x.real, x.imag], dim=1)
    return x


class DomainAdaptedCritic:
    """Wrap a critic so the G step converts exactly as the D step does.

    **Deliberately not an ``nn.Module``.** Registering the critic as a
    submodule of a per-step wrapper would give it a second parent, and
    ``base.py:514`` (AMP configuration) and ``base.py:1602`` (``.train()``
    bookkeeping) both reach for ``discriminator_model`` expecting the bare
    module. The loss computer uses its ``discriminator`` argument as a callable
    and nothing else (``unified_diffusion_reconstruction.py:479-511``: fetched,
    tested against None, called), so a plain callable is the whole contract --
    and it leaves parameter identity, ``opt_d``, AMP and DDP untouched.

    **The one place ``nn.Module``-ness is load-bearing is R1**, which is why
    this wrapper must never be the thing a regularizer receives.
    ``R1RegularizationLoss.forward`` opens with
    ``if not isinstance(discriminator, nn.Module): return 0.0`` -- a guard for
    generic ``loss_fn(pred, target)`` validation calls that cannot tell itself
    apart from a legitimately-wrapped critic. Handing it a
    ``DomainAdaptedCritic`` would turn a declared regularizer into a silent
    zero for a whole run. The D step therefore passes the BARE module and
    converts its *tensors* instead (``diffusion.py`` ``_align_for_critic``);
    only the G step, which runs no R1, wraps.

    Construct per step rather than caching: the wrapper holds no tensors, and a
    cached one goes stale the moment an EMA or DDP swap replaces
    ``discriminator_model``.

    Note the auxiliary arguments the computer passes beside the critic output --
    ``real_images=target, fake_images=pred`` -- stay in the GENERATOR's domain.
    Every kspace_filling arm declares ``gan_loss_type: hinge``, which scores
    ``fake_outputs_d`` alone and reads neither, so nothing is currently wrong;
    a relativistic or feature-matching loss would need them converted too.
    """

    def __init__(
        self,
        critic: Any,
        *,
        from_domain: str | None,
        to_domain: Any,
        takes_complex: bool,
    ) -> None:
        self.critic = critic
        self.from_domain = from_domain
        self.to_domain = to_domain
        self.takes_complex = takes_complex

    def __call__(self, x: torch.Tensor, **cond: Any) -> torch.Tensor:
        """Score ``x`` after putting it in the critic's declared domain.

        ``**cond`` is forwarded UNINSPECTED to the wrapped critic (#1931). The
        conditioning payload -- ``timesteps`` and ``contrast_idx`` for a critic
        declaring ``supports_contrast_conditioning`` -- describes the *sample*,
        not its domain, so a domain conversion neither consumes nor rewrites it.

        Forwarding blind is deliberate. Filtering ``cond`` against the wrapped
        critic's signature here is exactly the introspection-guarded forwarding
        that silently dropped ``contrast_idx`` at the generator seam and made
        #1931 necessary: a critic that declares the flag but cannot take the
        kwargs must ``TypeError`` on step 1, loudly, not score unconditioned
        for a whole run. The strategy decides whether to send a payload at all
        (``_critic_conditioning``); this wrapper only refuses to lose it.
        """
        return self.critic(
            to_critic_input(
                x,
                from_domain=self.from_domain,
                to_domain=self.to_domain,
                takes_complex=self.takes_complex,
            ),
            **cond,
        )


def critic_component_name(config: Any) -> str | None:
    """The registered name of this arm's critic, or ``None``.

    **The one owner of this resolution** (non-negotiable 17). Four callers need
    the same fact -- ``critic_input_domain`` (what space does it score in),
    ``critic_accepts_complex`` (does it take complex input),
    ``UnifiedGANLossComputer._resolve_critic_seam`` (is a conversion owed) and
    ``DiffusionTrainingStrategy._critic_conditioning`` (may it be sent
    t/contrast) -- and they had drifted into two spellings.

    **The direct attribute reads are the contract, not untidiness.** A
    ``getattr(model, "discriminator_component", None)`` answers ``None`` for a
    field that was RENAMED exactly as for one legitimately absent: the arm would
    report "no critic", every caller would silently degrade -- no conversion, no
    complex input, no conditioning -- and the run would train to completion with
    a critic scored in the wrong space. That is non-negotiable 3's silent
    fallback and the shape #368 bans. ``2e765cc4b`` removed the getattr chain
    from the capability reader for this reason, and
    ``test_a_renamed_discriminator_component_raises_rather_than_reporting_false``
    plus ``test_the_conditioning_reader_is_as_loud_as_the_capability_reader``
    (both in ``tests/unit/infrastructure/training/strategies/``) pin it. Electing
    this function as the one owner (non-negotiable 17) inherits that bar rather
    than lowering it (non-negotiable 20).

    **So an object without the field is a defective test double, not a config.**
    ``discriminator_component`` is a declared field on ``ModelConfigSchema``
    (``config/schemas/model.py``), and Pydantic materializes every declared field
    on every instance -- it is present, defaulted to ``None``, on every real
    config that has ever existed. Nothing reaching this function from production
    can lack it. A double that does is asserting a schema shape the schema does
    not have, and the ``AttributeError`` naming the field is the correct report.

    Only the states the schema really allows are absorbed: no model block, no
    critic configured, or a blank name.
    """
    model = config.model
    component = model.discriminator_component if model is not None else None
    if component is None or not component.name:
        return None
    return component.name


def critic_accepts_complex(config: Any) -> bool:
    """Whether the configured critic declares ``accepts_complex``.

    Read through ``get_model_capabilities``, NOT ``model_supports``: the latter
    looks the flag up with ``entry.get(capability)`` at the TOP level of the
    registry entry, while the real capability flags live nested in
    ``entry["capabilities"]``. It therefore answers False for every model that
    declares ``accepts_complex``. Do not "simplify" this call back to it.
    Tracked as #1916.
    """
    name = critic_component_name(config)
    if name is None:
        return False
    caps = get_model_capabilities(name)
    return bool(caps is not None and caps.accepts_complex)


def critic_input_domain(config: Any) -> Any:
    """The configured critic's declared ``input_domain``, or None.

    Read through ``get_model_capabilities`` rather than ``model_supports`` for
    the reason ``critic_accepts_complex`` documents: the capability flags live
    nested under ``entry["capabilities"]``, and the top-level lookup answers
    None for every model that declares one (#1916).
    """
    name = critic_component_name(config)
    if name is None:
        return None
    caps = get_model_capabilities(name)
    return None if caps is None else caps.input_domain


def generator_output_domain(config: Any) -> str | None:
    """The generator's declared output domain, or None.

    ``losses.policy.output_domain`` is the SSOT rather than
    ``model.model_domain`` or the generator's registered ``output_domain``:
    across all 60 kspace_filling arms the three agree, and this is the one the
    loss bridge already keys on, so electing it keeps the critic seam and the
    loss bridge from ever disagreeing (non-negotiable 17).
    """
    policy = getattr(getattr(config, "losses", None), "policy", None)
    domain = getattr(policy, "output_domain", None)
    return str(domain) if domain is not None else None
