"""Training-dynamics figures, from the step-wise frame of one run or several arms.

``FIGURES`` maps a figure id to its ``make``; :mod:`..__init__` registers every entry.
"""

from . import gain_over_input, spectral_recovery, training_health

FIGURES = {
    "fig_5_1_spectral_recovery": spectral_recovery.make,
    "fig_5_2_gain_over_input": gain_over_input.make,
    "fig_5_3_training_health": training_health.make,
}
