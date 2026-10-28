"""The settings an inference run uses: the run's ``resolved_config.json`` or the YAML.

Lives in ``infrastructure`` so that every inference reader (``infer``, ``export``,
campaign evaluation) resolves settings the same way without importing ``pipelines``.
"""

import logging
from pathlib import Path
from typing import Any

from spectramr.config.settings import TrainingSettings

logger = logging.getLogger(__name__)


def resolve_inference_settings(
    config_path: Path | str | None,
    checkpoint_path: Path | str | None,
    *,
    from_yaml: bool = False,
) -> tuple[TrainingSettings, dict[str, Any]]:
    """The settings an inference run uses, and where they came from.

    Precedence (#1379, the corpus review's T0.13): the run's own
    ``resolved_config.json`` beside the checkpoint wins, because it holds what
    the training run declared after every override; the training YAML is read
    only when ``from_yaml`` is set or no artifact exists. When both exist and
    the YAML disagrees with the artifact, the top-level blocks that differ are
    logged at WARNING and recorded in the source, so a predict run cannot
    silently score under a config the checkpoint never trained under.

    Returns ``(settings, source)`` where ``source`` is
    ``{"kind": "resolved_config" | "yaml", "path": ..., ...}``.
    """
    from spectramr.infrastructure.validation.resolved_config_artifact import (
        has_declared_block,
        resolved_config_beside,
        settings_from_resolved_config,
    )

    artifact = resolved_config_beside(checkpoint_path)
    # An artifact written before the ``_declared`` block cannot rebuild the
    # settings; every run directory on the cluster is in that state today, so
    # the documented ``infer --config ...`` command must keep working there:
    # the YAML is used and the artifact's age is reported, never a refusal.
    predates = artifact is not None and not has_declared_block(artifact)
    if from_yaml or artifact is None or predates:
        if config_path is None or not Path(config_path).exists():
            raise FileNotFoundError(
                "inference needs its settings: pass --config <training YAML>, or run against "
                "a checkpoint whose run directory holds a resolved_config.json with a "
                "`_declared` block"
                + (
                    f" ({artifact} predates that block, written before 2026-09-03)"
                    if predates
                    else (
                        f" (none beside {checkpoint_path})" if checkpoint_path is not None else ""
                    )
                )
                + ("; --from-yaml was set" if from_yaml else "")
                + "."
            )
        settings = TrainingSettings.from_yaml(str(config_path))
        source: dict[str, Any] = {
            "kind": "yaml",
            "path": str(config_path),
            "resolved_config": str(artifact) if artifact is not None else None,
        }
        if predates and not from_yaml:
            source["resolved_config_predates_declared"] = True
            logger.warning(
                "[config] %s predates the `_declared` block (written before 2026-09-03) and cannot "
                "rebuild the run's settings; inference uses %s. Re-running training rewrites the "
                "artifact with the block.",
                artifact,
                config_path,
            )
        return settings, source
    settings = settings_from_resolved_config(artifact)
    source = {"kind": "resolved_config", "path": str(artifact), "yaml": None}
    if config_path is not None and Path(config_path).exists():
        source["yaml"] = str(config_path)
        from_artifact = settings.model_dump(mode="json")
        from_file = TrainingSettings.from_yaml(str(config_path)).model_dump(mode="json")
        diverging = sorted(
            key
            for key in set(from_artifact) | set(from_file)
            if from_artifact.get(key) != from_file.get(key)
        )
        if diverging:
            source["diverging_blocks"] = diverging
            logger.warning(
                "[config] %s and %s disagree in %s; inference uses the run's resolved config. "
                "Pass --from-yaml to score under the YAML instead.",
                artifact,
                config_path,
                diverging,
            )
    return settings, source


def _describe_inference_source(
    config_path: Path | str | None, checkpoint_path: Path | str | None, from_yaml: bool
) -> dict[str, Any]:
    """The ``source`` :func:`resolve_inference_settings` would report, without loading."""
    from spectramr.infrastructure.validation.resolved_config_artifact import (
        has_declared_block,
        resolved_config_beside,
    )

    artifact = resolved_config_beside(checkpoint_path)
    predates = artifact is not None and not has_declared_block(artifact)
    if from_yaml or artifact is None or predates:
        source: dict[str, Any] = {
            "kind": "yaml",
            "path": str(config_path),
            "resolved_config": str(artifact) if artifact else None,
        }
        if predates and not from_yaml:
            source["resolved_config_predates_declared"] = True
        return source
    return {
        "kind": "resolved_config",
        "path": str(artifact),
        "yaml": str(config_path) if config_path else None,
    }
