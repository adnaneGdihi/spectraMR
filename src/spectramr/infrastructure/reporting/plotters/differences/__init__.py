"""Where arms differ: per-slice figures that need subject and slice identity.

``FIGURES`` maps a figure id to its ``make``; :mod:`..__init__` registers every entry.
``IDENTITY_FIGURES`` are fed ``slice_df`` keyed by ``subject|contrast|slice|repetition``
(``slice_case_rows``), never positional per-call rows.
"""

from . import band_fidelity, critical_difference, gain_distribution, slice_profile, win_rate

FIGURES = {
    "fig_6_1_critical_difference": critical_difference.make,
    "fig_6_2_win_rate": win_rate.make,
    "fig_6_3_band_fidelity": band_fidelity.make,
    "fig_6_4_slice_profile": slice_profile.make,
    "fig_6_5_gain_distribution": gain_distribution.make,
}
IDENTITY_FIGURES = tuple(FIGURES)
