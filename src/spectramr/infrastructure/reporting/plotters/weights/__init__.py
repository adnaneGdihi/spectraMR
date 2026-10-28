"""Figures drawn from a run's safetensors checkpoints (no model is built, no data read)."""

from . import layer_drift, trajectory

FIGURES = {
    "fig_4_1_weight_trajectory": trajectory.make,
    "fig_4_2_layer_drift": layer_drift.make,
}
