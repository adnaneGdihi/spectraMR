"""What a generator constructor accepts, and which config fields never reach it.

Split out of :mod:`generator_kwargs` in the Wave 0 exit-criterion work (#1400).
Two things live here, and both answer the same question -- *may this name be
injected?* -- while ``generator_kwargs`` answers *what value does it get?*:

* :func:`resolve_contract` / :func:`accepts` -- the constructor's own signature.
* :data:`SKIP_MODEL_FIELDS` -- ``config.model`` fields that are not constructor
  parameters at all. Each entry records the arm it crashed when it leaked; do
  not prune one without reproducing that arm.
"""

from __future__ import annotations

import logging

from spectramr.core.component_signature import SignatureContract, signature_contract

logger = logging.getLogger(__name__)


#: Empty contract: "we could not learn anything about this constructor".
#: Distinct from a contract that genuinely accepts nothing, but treated the
#: same by every caller -- injection is skipped rather than guessed, which is
#: the tolerant behaviour the audit probe depends on (it must return a
#: structured result, never raise, on an unresolvable model).
_NO_CONTRACT = SignatureContract(accepted=frozenset(), accepts_var_kwargs=False, owner="")

#: ``config.model`` fields that are NOT constructor parameters. Each entry
#: below records the arm it crashed when it leaked; do not prune without
#: reproducing that arm.
SKIP_MODEL_FIELDS = frozenset(
    {
        "model_type",
        "model_kwargs",
        "in_channels",
        "out_channels",
        "name",
        "model_name",
        "trellis",
        "trellis_vae",
        "generator_component",
        "discriminator_component",
        "discriminator",
        "denoising_model",
        # Domain-inference metadata, consumed by infer_output_domain() /
        # data-model-compatibility, not by model __init__. Strict configs
        # (e.g. UNetConfig) raise on unexpected kwargs, so leaking these
        # crashed eval_c5_exchangeability_test with "Unexpected keyword
        # argument 'target_domain' for UNetConfig" (smoke audit 2026-06-03).
        "target_domain",
        "model_domain",
        "output_type",
        "input_type",
        # Adaptive-conditioning sub-block consumed by the training STRATEGY
        # (virtual_fiducial_strategy reads model.conditioning.sources to build
        # the FiLM context), not by the constructor. Leaking it crashed
        # equivariance_conformal arms (cluster smoke 20260605, job 7095209).
        "conditioning",
        # Sequential-campaign warm-start knob, consumed by
        # ModelBuilder._warm_start AFTER the generator is built.
        # Leaking it crashed exp_p1_b1_equivariance_conformal (2026-06-16).
        "checkpoint_path",
    }
)


def resolve_contract(
    model_cls: type | None = None, model_type: str | None = None
) -> SignatureContract:
    """Return the constructor contract for a generator, tolerantly.

    Args:
        model_cls: The generator class, when the caller already resolved it.
            Preferred -- it needs no registry lookup, so a caller holding a
            class the registry does not know (the probe, under a patched
            registry) still gets a real contract.
        model_type: Registry key, used only when ``model_cls`` is ``None``.

    Returns:
        The contract, or an empty one when the class cannot be resolved or
        inspected. Never raises: an unresolvable constructor means "inject
        nothing", never a guess and never a crash.
    """
    try:
        if model_cls is None:
            if model_type is None:
                return _NO_CONTRACT
            from spectramr.models.factories.model_factory import ModelRegistry

            registry = ModelRegistry()
            if not registry.has_generator(model_type):
                return _NO_CONTRACT
            model_cls = registry.get_generator_class(model_type)
        return signature_contract(model_cls)
    except Exception as exc:  # tolerance is the contract, not an oversight
        logger.debug("Contract inspection failed for %s: %s", model_type, exc)
        return _NO_CONTRACT


def accepts(contract: SignatureContract, name: str) -> bool:
    """Whether ``name`` may be injected.

    True when the constructor names the parameter explicitly **or** declares
    ``**kwargs``. The second branch is not pedantry: generators such as
    ``KSpaceColdDiffusionGenerator`` read DC keys via ``kwargs.get(...)``, and
    without it they silently bypass SSOT reconciliation (pitfall #9).
    """
    return name in contract.accepted or contract.accepts_var_kwargs


__all__ = ["SKIP_MODEL_FIELDS", "accepts", "resolve_contract"]
