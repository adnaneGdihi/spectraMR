"""Pipeline parallelism (GPipe) for unrolled cascades split by ``pipeline_stages(n)``.

``plugin`` splits the generator at Stage A, ``module`` runs the schedule inside the
generator's forward, ``step_policy`` owns the backward, ``checkpoint`` consolidates
the stages into the unsplit model's keys.
"""
