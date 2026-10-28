"""Row markers of the main-results table in its LaTeX rendering."""

import pandas as pd

from spectramr.infrastructure.reporting.metric_names import annotate_frame
from spectramr.infrastructure.reporting.tables import main_results


def _mixed_sources_frame() -> pd.DataFrame:
    """``tested`` has a test split, ``val_only`` only a validation pass."""
    rows = [
        {
            "run_id": "tested",
            "method": "tested",
            "split": "test",
            "step": -1,
            "metric": "psnr",
            "value": 30.0,
        },
        {
            "run_id": "val_only",
            "method": "val_only",
            "split": "val",
            "step": 100,
            "metric": "val_psnr",
            "value": 28.0,
        },
    ]
    return annotate_frame(pd.DataFrame(rows))


def test_the_dagger_renders_as_latex_not_as_an_escaped_backslash(tmp_path):
    out = main_results.make(_mixed_sources_frame(), tmp_path)
    tex = out["latex"].read_text()
    assert r"val\_only $^\dagger$" in tex
    assert "textbackslash" not in tex


def test_a_noted_run_carries_a_double_dagger_in_both_renderings(tmp_path):
    out = main_results.make(
        _mixed_sources_frame(),
        tmp_path,
        run_notes={"tested": "status unknown: no run_summary.json"},
        excluded_runs={"gone": "crashed: RuntimeError"},
    )
    md, tex = out["markdown"].read_text(), out["latex"].read_text()
    assert "| tested ‡ |" in md
    assert "‡ tested: status unknown: no run_summary.json." in md
    assert "Not tabulated: gone (crashed: RuntimeError)." in md
    assert r"tested $^\ddagger$" in tex
    assert "textbackslash" not in tex
    assert (
        pd.read_csv(out["csv"])
        .set_index("method")
        .loc["tested", "note"]
        .startswith("status unknown")
    )


def test_an_arm_note_carries_a_section_mark_into_the_caption(tmp_path):
    out = main_results.make(
        _mixed_sources_frame(), tmp_path, arm_notes={"tested": "two attention blocks"}
    )
    md, tex = out["markdown"].read_text(), out["latex"].read_text()
    assert "| tested § |" in md
    assert "§ tested: two attention blocks." in md
    assert r"tested $^\S$" in tex
    caption = next(line for line in tex.splitlines() if line.startswith(r"\caption"))
    assert r"$^\S$ tested: two attention blocks." in caption
    assert pd.read_csv(out["csv"]).set_index("method").loc["tested", "arm_note"] == (
        "two attention blocks"
    )


def test_without_arm_notes_no_row_is_marked(tmp_path):
    out = main_results.make(_mixed_sources_frame(), tmp_path)
    assert "§" not in out["markdown"].read_text()
    assert r"\S" not in out["latex"].read_text()


def _two_arms(metric: str, a: float, b: float) -> pd.DataFrame:
    rows = [
        {
            "run_id": m,
            "method": m,
            "split": "val",
            "step": 100,
            "metric": f"val_{metric}",
            "value": v,
        }
        for m, v in (("arm_a", a), ("arm_b", b))
    ]
    return annotate_frame(pd.DataFrame(rows))


def _body(result) -> str:
    """The tabular rows; the caption always explains ``\\textbf``."""
    return result["latex"].read_text().split(r"\begin{tabular}", 1)[1]


def test_a_metric_with_no_declared_direction_is_not_bolded(tmp_path):
    ranked = main_results.make(_two_arms("psnr", 30.0, 28.0), tmp_path / "r", metrics=["psnr"])
    unranked = main_results.make(
        _two_arms("custom_score", 30.0, 28.0), tmp_path / "u", metrics=["custom_score"]
    )
    assert r"\textbf" in _body(ranked)
    body = _body(unranked)
    assert r"\textbf" not in body and "↑" not in body and "↓" not in body
