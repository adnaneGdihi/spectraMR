"""Design tokens shared by every reporting surface: HTML pages, plotly and matplotlib.

Pure data, no plotting imports, so ``core`` code, ``infrastructure/reporting`` and the
Sim2Rank scripts can all read it. The identity follows the 2026-10-27 lab deck (slate ink
on a cool grey ground, Source Serif 4 headings over IBM Plex Sans, a burnt-orange accent);
the dark steps were re-chosen against the dark surface with the palette validator rather
than inverted.

Colour roles, not series colours: dashboards put arms on an axis and spend colour on at
most three jobs (``baseline`` context, ``series`` for the other arms, ``accent`` for the
best one), because campaigns here run more arms than any categorical palette can separate.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Literal

Mode = Literal["light", "dark"]

_LIGHT = {
    "page": "#ECEFF3",  # deck background
    "surface": "#FFFFFF",  # cards and chart panels
    "surface_alt": "#F6F5F1",  # table stripes, code
    "ink": "#18212E",  # primary text
    "ink_2": "#475263",  # secondary text
    "muted": "#6A7382",  # captions, axis titles
    "rule": "#DCDFE4",  # borders, gridlines
    "rule_strong": "#BCC6D3",
    "accent": "#B44A0B",  # best arm, focus ring, active nav
    "accent_soft": "#FBE9DD",
    "series": "#1F5F99",  # the other arms
    "baseline": "#6A7382",  # the reference arm
    "good": "#2C7A3D",
    "bad": "#A93030",
}

_DARK = {
    "page": "#141B26",
    "surface": "#1D2735",
    "surface_alt": "#243042",  # deck's dark panel
    "ink": "#ECEFF3",
    "ink_2": "#BCC6D3",
    "muted": "#8D97A6",
    "rule": "#2E3A4C",
    "rule_strong": "#3D4B60",
    "accent": "#D8743A",
    "accent_soft": "#3A2A22",
    "series": "#4A8BCB",
    "baseline": "#7E8898",
    "good": "#4FA866",
    "bad": "#D9625E",
}

COLOURS: Final = MappingProxyType(
    {"light": MappingProxyType(_LIGHT), "dark": MappingProxyType(_DARK)}
)

# CSS stacks: the named faces render where installed, otherwise a close system face.
FONT_SERIF: Final = "'Source Serif 4', 'Source Serif Pro', Georgia, 'Times New Roman', serif"
FONT_SANS: Final = "'IBM Plex Sans', 'Segoe UI', system-ui, -apple-system, Arial, sans-serif"
FONT_MONO: Final = "'IBM Plex Mono', 'SFMono-Regular', Menlo, Consolas, monospace"

# Okabe-Ito, the colour-blind-safe categorical set behind METHOD_COLOURS (paper figures).
OKABE_ITO: Final = (
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
)

# Eight categorical hues in a fixed order whose adjacent pairs pass the colour-blind and
# normal-vision separation checks on white (dataviz validator). A ninth series is never a
# new hue: it folds to OVERFLOW, and the figure names it by label.
CATEGORICAL: Final = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
# The same eight hues stepped for a dark surface (validated against it, not flipped).
CATEGORICAL_DARK: Final = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767")
OVERFLOW: Final = "#9AA1AC"

# Single-hue ramp (light -> dark) for magnitudes such as rank and ADR heatmaps.
SEQUENTIAL_BLUE: Final = ("#EEF3F9", "#C9DAEC", "#94B6D8", "#5B8DC0", "#1F5F99", "#123B63")


def colours(mode: Mode = "light") -> MappingProxyType[str, str]:
    """Return the token map for ``mode``; an unknown mode raises."""
    if mode not in COLOURS:
        raise ValueError(f"unknown colour mode {mode!r}; expected one of {sorted(COLOURS)}")
    return COLOURS[mode]


def css_variables(mode: Mode = "light") -> str:
    """The token map as CSS custom properties (``--ink: #18212E; ...``)."""
    return " ".join(f"--{k.replace('_', '-')}: {v};" for k, v in colours(mode).items())
