r"""Interactive 3-D volumetric slice viewer (the MRIQC mosaic analog).

Renders one recorded case that carries a 3-D volume (``prediction_volume`` /
``target_volume``, shape ``[Z, H, W]``). A **slice slider** scrubs the Z axis
(the mosaic navigation) while **layer buttons** flick between Prediction, Target
and ``|Error|`` — exactly the two-control pattern of the 2-D viewer, but over
slices of a single subject instead of across subjects.

Returns ``None`` (soft-skip) when plotly is absent or no case carries a
``*_volume`` array. On single-slice datasets (e.g. M4Raw) this is the normal
outcome — see ``reporting.record_volumes`` for how volumes get recorded.
"""

from __future__ import annotations

import numpy as np

from .plotly_util import (
    ERROR_SCALE,
    GRAY_SCALE,
    apply_theme,
    caption,
    has_plotly,
    image_axes,
    seat_menus,
    to_div,
)

_MAX_SIDE = 192
_MAX_SLICES = 32


def _find_volume_case(cases) -> dict | None:
    """First case carrying a 3-D ``prediction_volume`` (prefer rank='best')."""
    have = [c for c in cases if np.asarray(c.get("prediction_volume", np.empty(0))).ndim == 3]
    if not have:
        return None
    for c in have:
        if c.get("rank") == "best":
            return c
    return have[0]


def _prep_volume(vol) -> np.ndarray | None:
    """Coerce to ``[Z, H, W]`` float, downsampling slices and pixels."""
    v = np.asarray(vol, dtype=np.float32)
    if v.ndim != 3 or v.size == 0:
        return None
    z_step = max(1, v.shape[0] // _MAX_SLICES)
    xy_step = max(1, max(v.shape[1:]) // _MAX_SIDE)
    return v[::z_step, ::xy_step, ::xy_step]


def _norm_volume(v: np.ndarray) -> np.ndarray:
    lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99))
    if hi <= lo:
        return np.zeros_like(v)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def volume_viewer_3d_div(cases) -> str | None:
    if not has_plotly() or not cases:
        return None
    case = _find_volume_case(cases)
    if case is None:
        return None
    pred = _prep_volume(case.get("prediction_volume"))
    if pred is None:
        return None
    tgt = _prep_volume(case.get("target_volume"))
    if tgt is None or tgt.shape != pred.shape:
        tgt = pred
    err = np.abs(pred - tgt)
    err = err / (err.max() + 1e-8)
    pred_n, tgt_n = _norm_volume(pred), _norm_volume(tgt)

    import plotly.graph_objects as go

    z_n = pred.shape[0]
    mid = z_n // 2
    cid = str(case.get("case_id", "volume"))
    title = f"Volume viewer — {cid}  ·  {z_n} slices"
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=pred_n[mid],
                colorscale=GRAY_SCALE,
                zmin=0,
                zmax=1,
                visible=True,
                colorbar={"title": "norm", "thickness": 12},
                hovertemplate="row %{y}, col %{x}: %{z:.3f}<extra>pred</extra>",
            ),
            go.Heatmap(
                z=tgt_n[mid],
                colorscale=GRAY_SCALE,
                zmin=0,
                zmax=1,
                visible=False,
                showscale=False,
                hovertemplate="row %{y}, col %{x}: %{z:.3f}<extra>target</extra>",
            ),
            go.Heatmap(
                z=err[mid],
                colorscale=ERROR_SCALE,
                zmin=0,
                zmax=1,
                visible=False,
                showscale=False,
                hovertemplate="row %{y}, col %{x}: %{z:.3f}<extra>|error|</extra>",
            ),
        ],
        frames=[
            go.Frame(
                name=str(z),
                data=[go.Heatmap(z=pred_n[z]), go.Heatmap(z=tgt_n[z]), go.Heatmap(z=err[z])],
                layout=go.Layout(title={"text": f"{title}  ·  slice {z}"}),
            )
            for z in range(z_n)
        ],
    )

    steps = [
        {
            "label": str(z),
            "method": "animate",
            "args": [
                [str(z)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for z in range(z_n)
    ]

    apply_theme(fig, title=f"{title}  ·  slice {mid}", height=560)
    image_axes(fig, n_cols=1)
    fig.update_layout(
        sliders=[
            {
                "active": mid,
                "y": 0,
                "yanchor": "top",
                "pad": {"t": 30},
                "currentvalue": {"prefix": "slice: "},
                "steps": steps,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
                "buttons": [
                    {
                        "label": "Prediction",
                        "method": "restyle",
                        "args": [{"visible": [True, False, False]}],
                    },
                    {
                        "label": "Target",
                        "method": "restyle",
                        "args": [{"visible": [False, True, False]}],
                    },
                    {
                        "label": "|Error|",
                        "method": "restyle",
                        "args": [{"visible": [False, False, True]}],
                    },
                ],
            }
        ],
    )
    seat_menus(fig)
    body = to_div(fig, div_id="volume-3d")
    return body + caption(
        "Drag the slider to move through the volume's slices (the mosaic "
        "navigation) and use the buttons to flick between prediction, target and "
        "|error|. Intensities are volume-normalised (1-99 percentile) so contrast "
        "is stable across slices; the error map is scaled to the volume's peak "
        "error. This viewer only appears when a run recorded 3-D volumes "
        "(reporting.record_volumes) — single-slice acquisitions use the 2-D "
        "viewer above."
    )
