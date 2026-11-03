r"""Shared helpers for the interactive (plotly) reporting layer.

Single source of truth for: probing whether plotly is installed, inlining the
plotly.js library **once** per report (so the HTML is self-contained and works
offline — unlike the old ``include_plotlyjs="cdn"`` path), the report theme
(colours from :mod:`spectramr.core.design_tokens`), the image colour scales, and the
"how to read this" caption block that makes every interactive panel explainable.

All plotly imports are guarded and happen inside functions, so importing this
module never fails when plotly is absent — the builders simply return ``None``
and the HTML assembler falls back to the static PNG.
"""

from __future__ import annotations

import html

from spectramr.core.design_tokens import FONT_SANS, colours

# Greyscale for magnitude images (black background -> white signal, the MRI
# display convention); Magma for error maps (dark = low error).
GRAY_SCALE = [[0.0, "#000000"], [1.0, "#ffffff"]]
ERROR_SCALE = "Magma"


def role(name: str, mode: str = "light") -> str:
    """A design-token colour by role (``ink``, ``series``, ``accent``, ``baseline`` ...)."""
    return colours(mode)[name]  # type: ignore[arg-type]


def has_plotly() -> bool:
    """Return True when plotly is importable (the interactive layer is available)."""
    try:
        import plotly.graph_objects  # noqa: F401
    except ImportError:
        return False
    return True


def plotlyjs_script() -> str | None:
    """Return a ``<script>`` tag with the plotly.js library inlined, or None.

    Emitted **once** at the top of the report; every div is then rendered with
    ``include_plotlyjs=False`` (see :func:`to_div`), so the whole report needs
    no network access to be interactive.
    """
    try:
        from plotly.offline import get_plotlyjs
    except ImportError:
        return None
    return f"<script type='text/javascript'>{get_plotlyjs()}</script>"


# No toolbar: drag-to-zoom, double-click reset and hover all work without it, and its
# floating icons sat over titles and button rows.
_CONFIG = {"displayModeBar": False, "responsive": True}


def to_div(fig, *, div_id: str | None = None) -> str:
    """Serialise a plotly figure to an inline ``<div>`` (no embedded plotly.js)."""
    return fig.to_html(include_plotlyjs=False, full_html=False, div_id=div_id, config=_CONFIG)


def seat_menus(fig) -> None:
    """Seat a figure's button/dropdown menus between its title and the plot.

    Call after the menus are set. A menu placed in paper coordinates above ``y=1``
    shares the title's strip of top margin and overlaps it; this grows the margin by
    one row and anchors each menu's bottom edge on the plot's top edge instead.
    """
    if not fig.layout.updatemenus:
        return
    fig.update_layout(margin={"t": (fig.layout.margin.t or 20) + 40})
    for menu in fig.layout.updatemenus:
        menu.update(y=1.0, yanchor="bottom", pad={"b": 8})


def apply_theme(fig, *, title: str | None = None, height: int | None = None) -> None:
    """Apply the report theme to a plotly figure in place.

    Backgrounds stay transparent so the figure sits on the page surface in either
    mode; the page's theme toggle re-inks text and gridlines via ``Plotly.relayout``.
    """
    tok = colours("light")
    axis = {"gridcolor": tok["rule"], "linecolor": tok["rule_strong"], "zerolinecolor": tok["rule_strong"]}
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": tok["ink_2"], "size": 12, "family": FONT_SANS},
        hoverlabel={"font": {"family": FONT_SANS}},
        margin={"l": 70, "r": 20, "t": 44 if title else 20, "b": 40},
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    if title is not None:
        fig.update_layout(title={"text": title, "font": {"size": 14, "color": tok["ink"]}})
    if height is not None:
        fig.update_layout(height=height)


def image_axes(fig, *, n_cols: int = 1) -> None:
    """Configure axes for image heatmaps: square pixels, row 0 at the top."""
    for i in range(1, n_cols + 1):
        xa = "xaxis" if i == 1 else f"xaxis{i}"
        ya = "yaxis" if i == 1 else f"yaxis{i}"
        fig.update_layout(
            {
                xa: {
                    "visible": False,
                    "scaleanchor": ya.replace("axis", ""),
                    "constrain": "domain",
                },
                ya: {"visible": False, "autorange": "reversed", "constrain": "domain"},
            }
        )


def caption(text: str) -> str:
    """Render a 'How to read this' explainability caption as an HTML block."""
    return f"<div class='howto'><b>How to read this.</b> {html.escape(text)}</div>"


def flag_colour() -> str:
    """Colour that flags outliers / departures (the ``bad`` status token)."""
    return role("bad")
