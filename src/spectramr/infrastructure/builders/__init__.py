"""Builder infrastructure: base classes, leaf component builders, one director.

This package used to advertise a parallel pipeline stack — a
``TrainingPipelineDirector``/``InferencePipelineDirector`` pair under an
``ExperimentDirector``. None of it was reachable. ``main.py`` records why the
experiment path went first: *"the old ExperimentDirector path was
dead-on-arrival: its ``validate()`` required generator/loss config the CLI never
supplied, so it raised before any training ran — the real work was always
``run_training_pipeline``."* The other two were never wired at all.

**The live builder stack is ``infrastructure/training/builders/``**, entered from
``pipelines/train.py`` via ``TrainingEnvironmentDirector``. What survives here is
what that stack and the data layer actually consume:

  - Phase 0 (core): ``BuilderBase``, ``FluentBuilder``, ``DirectorBuilder``,
    ``BuilderRegistry``
  - Phase 1 (leaf): model, optimizer, data, physics and adapter builders
  - Phase 2 (directors): ``DataPipelineDirector`` (the data SSOT, ``pipelines/make.py``)

Loss construction is **not** here: the single owner is
``infrastructure/training/builders/loss_builder.py::LossBuilder``, reached
through ``LossRegistry`` → ``create_loss``. A second ``LossBuilder`` used to be
exported from this package as an alias of ``SingleLossBuilder``; nothing outside
the dead directors ever imported it, and the name collision was a trap.

Usage::

    from spectramr.infrastructure.builders import (
        BuilderBase, FluentBuilder, DirectorBuilder, BuilderRegistry,
        GeneratorBuilder, DiscriminatorBuilder, OptimizerBuilder,
    )
"""

# Phase 0: Core Builder Infrastructure
from spectramr.infrastructure.builders.core.base_builder import (
    BuilderBase,
    DirectorBuilder,
    FluentBuilder,
)

# Data builders
from spectramr.infrastructure.builders.leaf.data_builders import (
    DataLoaderBuilder,
    DatasetBuilder,
)

# Phase 1: Leaf Component Builders
# Model builders
from spectramr.infrastructure.builders.leaf.model_builders import (
    DecoderBuilder,
    DiscriminatorBuilder,
    EncoderBuilder,
    GeneratorBuilder,
)

# Optimizer builders
from spectramr.infrastructure.builders.leaf.optimizer_builders import (
    OptimizerBuilder,
)

# Physics builders
from spectramr.infrastructure.builders.leaf.physics_builders import (
    DataConsistencyBuilder,
    FFTBuilder,
    MaskBuilder,
    PhysicsBuilder,
)
from spectramr.infrastructure.builders.registry import BuilderRegistry

__all__ = [
    # Phase 0: Core
    "BuilderBase",
    "FluentBuilder",
    "DirectorBuilder",
    "BuilderRegistry",
    # Phase 1: Model Builders
    "GeneratorBuilder",
    "DiscriminatorBuilder",
    "EncoderBuilder",
    "DecoderBuilder",
    # Phase 1: Optimizer Builders
    "OptimizerBuilder",
    # Phase 1: Data Builders
    "DatasetBuilder",
    "DataLoaderBuilder",
    # Phase 1: Physics Builders
    "FFTBuilder",
    "MaskBuilder",
    "DataConsistencyBuilder",
    "PhysicsBuilder",
]
