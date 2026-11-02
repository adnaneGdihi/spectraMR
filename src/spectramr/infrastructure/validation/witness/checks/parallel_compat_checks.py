"""The strategy x parallel layout refusal, at audit time.

``strategy_compat.check_strategy_for_parallel`` is the one owner of the rule; the
training director enforces it when the environment is built. This witness runs the
same verdict in ``spectramr audit`` so a refused arm (including a pipeline the
strategy has not declared, or a model without ``pipeline_stages``) fails before a
job is queued rather than at its first build step.

Registration is by import (the witness package walk).
"""

from __future__ import annotations

from spectramr.infrastructure.validation.witness.checks.undersampling_checks import (
    _resolve_strategy,
)
from spectramr.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Subject,
    Tier,
    WitnessVerdict,
    register_witness,
)
from spectramr.infrastructure.validation.witness.subject import WitnessSubject

__all__ = ["parallel_layout_defect", "parallel_layout_is_supported"]

_CATEGORY = "parallel"
_NAME = "parallel_layout_is_supported"


def parallel_layout_defect(settings: object, strategy_cls: type | None) -> str | None:
    """The refusal message, or None when the layout is allowed or not judgeable here."""
    from spectramr.infrastructure.distributed.strategy_compat import check_strategy_for_parallel

    parallel = getattr(settings, "parallel", None)
    if parallel is None or getattr(parallel, "strategy", "none") == "none":
        return None
    if strategy_cls is not None:
        verdict = check_strategy_for_parallel(settings, strategy_cls)
        if verdict.refused:
            return verdict.message
    if int(getattr(getattr(parallel, "mesh", None), "pipeline", 1)) > 1:
        from spectramr.infrastructure.distributed.pipeline_parallel.plugin import (
            has_pipeline_hook,
        )
        from spectramr.models.registry import MODEL_REGISTRY

        model_type = str(getattr(getattr(settings, "model", None), "model_type", ""))
        model_cls = MODEL_REGISTRY.get(model_type, {}).get("class")
        if model_cls is not None and not has_pipeline_hook(model_cls):
            return (
                f"parallel.mesh.pipeline > 1 needs a model with a pipeline_stages(n) hook; "
                f"{model_type!r} ({model_cls.__name__}) has none."
            )
    return None


@register_witness(
    _NAME,
    category=_CATEGORY,
    stage=Stage.DECLARE,
    tiers=(Tier.T1,),
    subjects=(Subject.SETTINGS,),
    severity=Severity.ERROR,
    description="The training strategy and model support the declared parallel layout",
    fix_hint=(
        "Use a strategy whose parallel_compat declares the backend (and 'pipelined' for "
        "parallel.mesh.pipeline > 1), a model with pipeline_stages, or drop the layout."
    ),
)
def parallel_layout_is_supported(subject: WitnessSubject) -> WitnessVerdict:
    message = parallel_layout_defect(subject.settings, _resolve_strategy(subject.settings))
    return WitnessVerdict(
        witness_name=_NAME,
        passed=message is None,
        message=message or "the parallel layout is supported by the strategy and model",
        severity=Severity.INFO if message is None else Severity.ERROR,
        category=_CATEGORY,
        stage=Stage.DECLARE,
        tier=Tier.T1,
        yaml_keys=("parallel.strategy", "parallel.mesh.pipeline"),
    )
