"""Directors: the one orchestrator here that is actually reached.

``TrainingPipelineDirector``, ``InferencePipelineDirector`` and
``ExperimentDirector`` were removed — a parallel pipeline stack with no caller
anywhere in ``src/``. The live training stack is
``infrastructure/training/builders/`` (``TrainingEnvironmentDirector``, entered
from ``pipelines/train.py``); inference goes through
``pipelines/infer.py::run_inference_pipeline``.

- ``DataPipelineDirector`` — the data SSOT every higher layer builds datasets
  through (non-negotiable #7). Imported from
  ``spectramr.infrastructure.builders.directors.data_pipeline_director``.

Checkpoint save and resume live in ``spectramr.infrastructure.checkpointing``.
"""

__all__: list[str] = []
