r"""Interactive per-subject 2-D viewer (the MRIQC reportlet analog).

One plotly figure with two orthogonal controls:

- a **subject slider** that scrubs through the recorded cases (best -> worst);
- **layer buttons** that flick the single panel between Prediction, Target and
  ``|Error|`` — the before/after reportlet, for spotting where the
  reconstruction departs from the reference.

The slider drives plotly *frames* (which swap the heatmap ``z`` data) while the
buttons drive trace *visibility*, so the two controls never fight each other.
Returns ``None`` (soft-skip) when plotly is absent or no case carries a 2-D
prediction.
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

_MAX_CASES = 12
_MAX_SIDE = 192  # downsample cap so the inlined arrays stay small


def _prep(arr) -> np.ndarray | None:
    """Coerce to a 2-D float array, mid-slicing 3-D input and downsampling."""
    if arr is None:
        return None
    a = np.asarray(arr, dtype=np.float32)
    while a.ndim > 2:
        a = a[a.shape[0] // 2]
    if a.ndim != 2 or a.size == 0:
        return None
    step = max(1, max(a.shape) // _MAX_SIDE)
    return a[::step, ::step]


def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(np.percentile(a, 1)), float(np.percentile(a, 99))
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def _title(case: dict) -> str:
    rank = case.get("rank", case.get("case_id", ""))
    metrics = case.get("metrics", {}) or {}
    bits = [f"{k.upper()} {v:.3g}" for k, v in metrics.items() if k in ("psnr", "ssim", "nmse")]
    tail = ("  ·  " + "  ".join(bits)) if bits else ""
    return f"Subject viewer — {rank}{tail}"


def subject_viewer_2d_div(cases) -> str | None:
    if not has_plotly() or not cases:
        return None
    import plotly.graph_objects as go

    prepared = []
    for c in cases:
        pred = _prep(c.get("prediction"))
        if pred is None:
            continue
        tgt = _prep(c.get("target"))
        if tgt is None or tgt.shape != pred.shape:
            tgt = pred
        err = np.abs(pred - tgt)
        err = err / (err.max() + 1e-8)
        prepared.append((c, _norm(pred), _norm(tgt), err))
        if len(prepared) >= _MAX_CASES:
            break
    if not prepared:
        return None

    c0, p0, t0, e0 = prepared[0]
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=p0,
                colorscale=GRAY_SCALE,
                zmin=0,
                zmax=1,
                visible=True,
                colorbar={"title": "norm", "thickness": 12},
                hovertemplate="row %{y}, col %{x}: %{z:.3f}<extra>pred</extra>",
            ),
            go.Heatmap(
                z=t0,
                colorscale=GRAY_SCALE,
                zmin=0,
                zmax=1,
                visible=False,
                showscale=False,
                hovertemplate="row %{y}, col %{x}: %{z:.3f}<extra>target</extra>",
            ),
            go.Heatmap(
                z=e0,
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
                name=str(c.get("case_id", i)),
                data=[go.Heatmap(z=p), go.Heatmap(z=t), go.Heatmap(z=e)],
                layout=go.Layout(title={"text": _title(c)}),
            )
            for i, (c, p, t, e) in enumerate(prepared)
        ],
    )

    names = [str(prepared[i][0].get("case_id", i)) for i in range(len(prepared))]
    steps = [
        {
            "label": name,
            "method": "animate",
            "args": [
                [name],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for name in names
    ]

    apply_theme(fig, title=_title(c0), height=560)
    image_axes(fig, n_cols=1)
    fig.update_layout(
        sliders=[
            {
                "active": 0,
                "y": 0,
                "yanchor": "top",
                "pad": {"t": 30},
                "currentvalue": {"prefix": "subject: "},
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
    body = to_div(fig, div_id="subject-2d")
    return body + caption(
        "Drag the slider to scrub through the recorded subjects (best -> worst) "
        "and use the buttons to flick the panel between the prediction, the "
        "reference target and the |error| map. Intensities are per-image "
        "normalised (1-99 percentile); the error map is scaled to its own "
        "maximum so hot regions mark where the reconstruction departs from the "
        "reference. These are 2-D single-slice magnitude cases scrubbed across "
        "subjects — not orthogonal cuts through a volume (see the 3-D viewer for "
        "that)."
    )
