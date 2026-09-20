"""Single source of truth for ``config -> generator __init__ kwargs``.

Every path that constructs a generator resolves its kwargs **here**: the
training builder (``ModelBuilder.build_generator``), the leaf builder
(``GeneratorBuilder.build``) and the audit probe
(``infrastructure.validation.forward_probe``). Before this module the
resolution existed as two independent copies in two files -- the top-level
SSOT injections in ``model_builder.py`` and the opportunistic
``ModelConfigSchema`` sweep in ``leaf/model_builders.py`` -- and the probe had
a hand-rolled third. That is why ``audit --probe`` could validate a
differently-constructed model than training built, and an arm could pass the
probe and still diverge in training.

The guarantee is structural, not conventional: ``build_generator`` and
``GeneratorBuilder.build`` retain **no** resolution logic of their own, so a
future edit cannot reintroduce a second answer without deleting a call site.

Resolution is **two halves**, kept apart because their callers differ:

``resolve_generator_kwargs`` (owned by ``ModelBuilder``)
    1. ``config.model.model_kwargs`` -- what the arm wrote.
    2. Diffusion wrapper fields (``denoising_model``, ``base_diffusion_config``).
    3. Contract-gated SSOT injections (acceleration, log-scaling, DC, device).

``apply_model_field_sweep`` (owned by ``GeneratorBuilder``)
    4. ``in_channels``/``out_channels`` stripped -- the factory passes those.
    5. ``declared_keys`` snapshot for the landed-nowhere check (#560, #878).
    6. Opportunistic top-level ``model.*`` sweep.

They are not fused into one function because ``GeneratorBuilder`` has two
further callers -- ``pipeline_strategy._create_stage_model`` and
``diffusion``'s prior-model load -- that build a **different** ``model_type``
(``stage_cfg.model_type`` / ``prior_config.source``) and deliberately do not
inherit ``config.model.model_kwargs``. Fusing would silently change what those
two build. Callers wanting the whole training pipeline in one call --
the audit probe -- use :func:`resolve_full_generator_kwargs`, which composes
the halves in the same order ``ModelBuilder`` -> ``GeneratorBuilder`` does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

# Constructor-contract resolution and the post-construction checkpoint wrapper
# live in sibling modules (300-LOC ceiling, NN20). Re-exported here under their
# historical names so each keeps exactly one owner (NN17) while the external
# importers -- forward_probe.py, model_builder.py and three test modules --
# resolve them through this module unchanged.
from spectramr.infrastructure.builders.generator_contract import (
    SKIP_MODEL_FIELDS as _SKIP_MODEL_FIELDS,
)
from spectramr.infrastructure.builders.generator_contract import (
    accepts as _accepts,
)
from spectramr.infrastructure.builders.generator_contract import (
    resolve_contract as resolve_contract,
)
from spectramr.infrastructure.builders.gradient_checkpointing import (
    apply_gradient_checkpointing as apply_gradient_checkpointing,
)
from spectramr.infrastructure.physics.dc_settings import ssot_pairs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedGeneratorKwargs:
    """The kwargs a generator is constructed with, plus what the arm declared.

    Attributes:
        kwargs: Constructor keyword arguments, excluding ``in_channels`` and
            ``out_channels`` (the factory passes those explicitly).
        declared_keys: Keys present before the opportunistic sweep. The factory
            needs the two apart: a top-level field forwarded on spec and
            swallowed by a ``**kwargs`` constructor is expected, while a
            ``model_kwargs`` entry that lands nowhere is the ``sobolev_order``
            failure (#560, #878).
    """

    kwargs: dict[str, Any]
    declared_keys: frozenset[str]


def resolve_generator_kwargs(
    config: Any,
    *,
    model_cls: type | None = None,
    model_type: str | None = None,
    device: Any = None,
) -> dict[str, Any]:
    """Resolve the full generator constructor kwargs from ``config``.

    Args:
        config: A ``TrainingSettings``, or any object exposing ``model``. Only
            ``config.model.model_kwargs`` is required; every other block is
            probed with ``hasattr``/``getattr``, so a partial stand-in resolves
            the subset it actually carries.
        model_cls: The generator class when already resolved. Preferred: a
            caller holding a class the registry does not know still gets a real
            contract.
        model_type: The type actually being built. Defaults to
            ``config.model.model_type``, but ``pipeline_strategy`` and
            ``diffusion`` build ``stage_cfg.model_type`` /
            ``prior_config.source``, and the contract must describe the class
            being constructed, not the arm's headline model.
        device: The run's already-resolved compute device
            (``ModelBuilder._device``, or the probe's own). Injected by step 3d
            into constructors that name ``device`` explicitly. ``None`` skips
            the injection entirely -- this function never resolves a device of
            its own (non-negotiable 9b).

    Returns:
        The constructor kwargs from steps 1-3.

    Raises:
        ValueError: If ``model_kwargs`` contradicts an SSOT block -- a declared
            ``kspace_log_scaled`` disagreeing with
            ``data.processing.enable_log_scaling``, or a DC key disagreeing
            with ``physics.data_consistency``. Both describe the same fact
            twice; silent divergence here previously disabled experiment-32a's
            adversarial training (pitfall #9).
    """
    model_cfg = config.model
    kwargs: dict[str, Any] = dict(model_cfg.model_kwargs or {})
    if model_type is None:
        model_type = getattr(model_cfg, "model_type", None)

    # 2. Diffusion wrapper fields. Separate fields on ModelConfigSchema, not
    #    part of model_kwargs, but required by the wrapper handlers.
    for field in ("denoising_model", "base_diffusion_config"):
        value = getattr(model_cfg, field, None)
        if value is not None and field not in kwargs:
            kwargs[field] = dict(value)

    contract = resolve_contract(model_cls=model_cls, model_type=model_type)

    # 3a. Global acceleration config, only if the constructor accepts it.
    if hasattr(config, "undersampling") and "acceleration_config" in contract.accepted:
        kwargs["acceleration_config"] = config.undersampling

    # 3a-bis. training.diffusion.sampler. The generator resolved this only from
    #     model_kwargs, which nothing wrote, so the declared value never left
    #     the config and every arm fell through to the `cold_mri` literal. The
    #     value is injected here and validated by the generator that consumes
    #     it -- a model that never reads `sampler` must not be refused for a
    #     name its own reverse loop would never look up.
    _diffusion = getattr(getattr(config, "training", None), "diffusion", None)
    _declared_sampler = getattr(_diffusion, "sampler", None) if _diffusion else None
    if _declared_sampler and "sampler" not in kwargs and (
        "sampler" in contract.accepted or contract.accepts_var_kwargs
    ):
        kwargs["sampler"] = str(_declared_sampler)

    # 3b. data.processing.enable_log_scaling. The magnitude ceilings in the
    #     cold-diffusion path enforce a PHYSICAL ratio, so they must know
    #     whether the k-space they bound is log1p-compressed; without this a
    #     declared reverse_clip_ratio of 1.3 realised 29.8x on real M4Raw data
    #     (#1281). Reading the existing data-block knob keeps ONE source of
    #     truth and adds no new YAML key.
    if "kspace_log_scaled" in contract.accepted:
        try:
            ssot = bool(config.data.processing.enable_log_scaling)
        except AttributeError:
            # No data.processing block: leave the kwarg unset. The generator
            # then raises at the point of use if (and only if) a magnitude
            # bound is actually enabled -- never a silent default.
            logger.debug(
                "No data.processing.enable_log_scaling for %s; kspace_log_scaled left unset.",
                model_type,
            )
        else:
            declared = kwargs.get("kspace_log_scaled")
            if declared is not None and bool(declared) != ssot:
                raise ValueError(
                    "kspace_log_scaled is declared in model_kwargs as "
                    f"{bool(declared)} but data.processing.enable_log_scaling "
                    f"is {ssot}. These describe the same fact; remove the "
                    "model_kwargs copy and let the data block be the single "
                    "source of truth."
                )
            kwargs["kspace_log_scaled"] = ssot

    # 3c. physics.data_consistency is the SSOT for DC behaviour
    #     (TODO/backlog_unify_dc_config.md).
    #
    #     WHICH keys this block owns lives in ``dc_settings.DC_SSOT_KEYS``, not
    #     inline here: the audit needs the same mapping to decide whether a
    #     declared knob can matter, and two copies of it diverge as a wrong
    #     number rather than an error (non-negotiable 17). The noise-simulation
    #     keys joined that tuple in #1525 -- they were validated, documented
    #     schema fields that this loop never forwarded, so every DC layer used
    #     its own hard-coded 0.01/0.005 and no config could reach them.
    physics = getattr(config, "physics", None)
    dc = getattr(physics, "data_consistency", None) if physics else None
    if dc:
        for name, ssot_value in ssot_pairs(dc):
            if not _accepts(contract, name):
                continue
            existing = kwargs.get(name)
            if existing is not None and existing != ssot_value:
                raise ValueError(
                    f"Data-consistency configuration conflict for '{name}': "
                    f"model.model_kwargs has {existing!r} but "
                    f"physics.data_consistency has {ssot_value!r}. The SSOT is "
                    "physics.data_consistency — remove the redundant "
                    "model_kwargs entry."
                )
            kwargs[name] = ssot_value

    # 3d. The run's compute device. A generator whose *internals* build
    #     device-resident state cannot discover the device from its own
    #     parameters at construction time, and sniffing it from a tensor at call
    #     time is what non-negotiable 9b forbids -- so it is inherited from the
    #     configuration here, alongside the other SSOT injections.
    #
    #     Gated on EXPLICIT acceptance (``in contract.accepted``), not the
    #     tolerant :func:`_accepts` used by 3c. Almost every generator declares
    #     ``**kwargs``, so the tolerant test would inject ``device`` into all 586
    #     registered generators -- and several forward ``**kwargs`` into strict
    #     sub-configs that raise on an unexpected key, which is the exact failure
    #     ``SKIP_MODEL_FIELDS`` above was written to stop. Measured on this tree:
    #     ``device`` is explicitly named by 1 registered generator
    #     (``kspace_cold_diffusion``), so the blast radius is that one class.
    if device is not None and "device" in contract.accepted:
        declared = kwargs.get("device")
        if declared is not None and str(declared) != str(device):
            raise ValueError(
                f"Device configuration conflict: model.model_kwargs has "
                f"device={declared!r} but the run resolved {str(device)!r}. "
                "These describe the same fact; remove the model_kwargs entry "
                "and let the run's device resolution "
                "(spectramr.core.compute_device) be the single source of truth."
            )
        kwargs["device"] = device

    return kwargs


def apply_model_field_sweep(kwargs: dict[str, Any], config: Any) -> ResolvedGeneratorKwargs:
    """Strip the explicit channel args, snapshot, then sweep top-level fields.

    The second half of resolution, owned by ``GeneratorBuilder.build``.

    Args:
        kwargs: Kwargs so far -- from :func:`resolve_generator_kwargs` on the
            training path, or a caller's own ``with_parameter`` values.
        config: Provides ``config.model`` for the sweep. A Pydantic schema is
            swept over its declared ``model_fields`` (so an unset field still
            contributes its default); anything else is swept over its instance
            ``__dict__``. Both honour ``_SKIP_MODEL_FIELDS``, so a duck-typed
            stand-in forwards ``spatial_dims`` and withholds ``input_type`` /
            ``output_type`` exactly as a real config does.

    Returns:
        The swept kwargs and the pre-sweep ``declared_keys`` snapshot.
    """
    kwargs = dict(kwargs)

    # 4. The factory passes these explicitly; leaving them here raises
    #    "multiple values for keyword argument".
    kwargs.pop("in_channels", None)
    kwargs.pop("out_channels", None)

    # 5. Snapshot BEFORE the opportunistic sweep below adds to it.
    declared_keys = frozenset(kwargs)

    # 6. Forward top-level config.model.* fields the constructor might accept
    #    (spatial_dims, base_channels, depth, ...). Without this,
    #    `model.spatial_dims: 3` was dropped before reaching MONAI, defaulting
    #    the model to Conv2d and crashing on 3D data (#560, #878). The factory
    #    filters against the signature, so unknown fields drop harmlessly.
    model_cfg = getattr(config, "model", None)
    if model_cfg is None:
        return ResolvedGeneratorKwargs(kwargs=kwargs, declared_keys=declared_keys)
    # Declared fields for a schema; instance attributes for anything else. The
    # probe passes duck-typed stand-ins, and forwarding must not depend on
    # which kind it got -- that difference is the divergence this module exists
    # to remove.
    field_names = getattr(type(model_cfg), "model_fields", None) or vars(model_cfg)
    if field_names:
        for field_name in field_names:
            if field_name in _SKIP_MODEL_FIELDS or field_name in kwargs:
                continue
            value = getattr(model_cfg, field_name, None)
            if value is None:
                continue
            kwargs.setdefault(field_name, value)

    return ResolvedGeneratorKwargs(kwargs=kwargs, declared_keys=declared_keys)


def resolve_full_generator_kwargs(
    config: Any,
    *,
    model_cls: type | None = None,
    model_type: str | None = None,
    device: Any = None,
) -> ResolvedGeneratorKwargs:
    """Both halves, in the order the training path applies them.

    This is what ``audit --probe`` must call: composing the halves here is what
    makes the probed model the model training builds. Callers that are
    themselves one half of the pipeline (``ModelBuilder``, ``GeneratorBuilder``)
    call their own half directly instead.
    """
    kwargs = resolve_generator_kwargs(
        config, model_cls=model_cls, model_type=model_type, device=device
    )
    return apply_model_field_sweep(kwargs, config)
