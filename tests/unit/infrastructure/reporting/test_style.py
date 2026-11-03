from pathlib import Path

import matplotlib as mpl
import pytest

from spectramr.infrastructure.reporting import style as style_mod


def test_nature_style_file_exists_and_is_sans_serif():
    p = Path(style_mod.__file__).parent / "styles" / "nature.mplstyle"
    assert p.exists()
    text = p.read_text()
    assert "font.family: sans-serif" in text
    assert "axes.grid: False" in text
    assert "savefig.dpi: 600" in text


def test_use_default_style_nature_applies_sans_serif():
    style_mod.use_default_style("nature")
    assert mpl.rcParams["font.family"] == ["sans-serif"]
    assert mpl.rcParams["axes.grid"] is False


def test_column_width_returns_inches():
    assert abs(style_mod.column_width("single") - 89 / 25.4) < 1e-6
    assert abs(style_mod.column_width("double") - 183 / 25.4) < 1e-6


def test_column_width_unknown_raises():
    with pytest.raises(ValueError):
        style_mod.column_width("triple")


def test_use_default_style_unknown_raises():
    with pytest.raises(ValueError):
        style_mod.use_default_style("comic_sans")


def test_panel_label_adds_text():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    style_mod.panel_label(ax, "a")
    texts = [t.get_text() for t in ax.texts]
    assert "a" in texts
    plt.close(fig)


def test_save_figure_emits_requested_formats(tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    written = style_mod.save_figure(fig, tmp_path, "demo", formats=("pdf", "png", "tiff"))
    assert set(written) == {"pdf", "png", "tiff"}
    for ext, path in written.items():
        assert path.exists()
        assert path.suffix.lstrip(".") == ext


def test_panel_label_respects_global_disable():
    import matplotlib.pyplot as plt

    style_mod.set_panel_labels(False)
    try:
        fig, ax = plt.subplots()
        style_mod.panel_label(ax, "a")
        assert "a" not in [t.get_text() for t in ax.texts]
        plt.close(fig)
    finally:
        style_mod.set_panel_labels(True)  # restore global for other tests
    # re-enabled: label is drawn again
    fig, ax = plt.subplots()
    style_mod.panel_label(ax, "b")
    assert "b" in [t.get_text() for t in ax.texts]
    plt.close(fig)


# ---------------------------------------------------------------------
# pretty_label (2026-07-01 clarity SSOT)
# ---------------------------------------------------------------------


def test_pretty_label_known_keys():
    from spectramr.infrastructure.reporting.style import pretty_label

    assert pretty_label("psnr") == "PSNR (dB)"
    assert pretty_label("ssim") == "SSIM"
    assert pretty_label("params_m") == "parameters (M)"
    assert pretty_label("g_total_loss") == "generator total loss"
    assert pretty_label("iterations_per_sec") == "throughput (it/s)"


def test_pretty_label_expands_split_prefixes():
    from spectramr.infrastructure.reporting.style import pretty_label

    assert pretty_label("val_ssim") == "validation SSIM"
    assert pretty_label("train_psnr") == "training PSNR (dB)"
    assert pretty_label("test_nmse") == "test NMSE"


def test_pretty_label_unknown_key_degrades_to_words():
    from spectramr.infrastructure.reporting.style import pretty_label

    assert pretty_label("my_custom_metric") == "my custom metric"
    # bare prefix (no remainder) must not recurse forever
    assert pretty_label("val_") == "val"


def test_caption_templates_for_new_figures():
    from spectramr.infrastructure.reporting.style import caption_for

    assert "Spearman" in caption_for("metric_correlation")
    assert "Generalization gap" in caption_for("train_val_gap", metric="SSIM")
    assert "run-x" in caption_for("run_summary_card", run_id="run-x")


class TestUnregisteredMethodsAreNotAllBlack:
    """#720: `colour_for` defaulted `fallback_index=0`, and `OKABE_ITO[0]` is BLACK.

    Nine registered strategies had no `METHOD_COLOURS` entry, so a figure
    comparing two of them drew both in black -- which is not merely a missing
    colour, it is the reserved identity of `baseline`/`zero_filled`. Two unknowns
    became indistinguishable AND both were disguised as the baseline, silently.
    """

    _PREVIOUSLY_MISSING = [
        "bloch_synth",
        "cut",
        "cyclegan",
        "field_cocycle",
        "generative_refiner",
        "mrs_quantification",
        "perfusion_kinetic",
        "phase_contrast_flow",
        "quality_matching",
    ]

    @pytest.mark.parametrize("method", _PREVIOUSLY_MISSING)
    def test_the_nine_are_registered(self, method):
        from spectramr.infrastructure.reporting.style import METHOD_COLOURS

        assert method in METHOD_COLOURS

    def test_black_stays_reserved_for_the_baseline(self):
        from spectramr.infrastructure.reporting.style import colour_for

        assert colour_for("baseline") == "#000000"
        assert colour_for("zero_filled") == "#000000"

    @pytest.mark.parametrize(
        "unknown", ["not_a_registered_method", "another_unknown", "third_unknown"]
    )
    def test_an_unknown_never_gets_the_baseline_colour(self, unknown):
        from spectramr.infrastructure.reporting.style import colour_for

        assert colour_for(unknown) != "#000000"

    def test_two_unknowns_are_not_forced_to_the_same_colour(self):
        """The systematic collision is what made the old fallback misleading."""
        from spectramr.infrastructure.reporting.style import colour_for

        colours = {colour_for(f"unknown_method_{i}") for i in range(6)}
        assert len(colours) > 1, "every unknown still resolves to one colour"

    def test_the_fallback_is_stable_across_calls(self):
        """`hash()` is salted per process, so it would give one method different
        colours in two figures of the same report."""
        from spectramr.infrastructure.reporting.style import colour_for

        assert colour_for("some_new_arm") == colour_for("some_new_arm")

    def test_an_explicit_index_still_wins(self):
        """Callers that colour by enumerate order must keep working."""
        from spectramr.infrastructure.reporting.style import (
            _FALLBACK_PALETTE,
            colour_for,
        )

        assert colour_for("whatever", fallback_index=2) == _FALLBACK_PALETTE[2]

    def test_the_two_cyclegan_spellings_agree(self):
        """One method must not change colour because a caller spelled it
        differently -- `cyclegan` is the registered strategy, `cycle_gan` the
        spelling older figures used."""
        from spectramr.infrastructure.reporting.style import colour_for

        assert colour_for("cyclegan") == colour_for("cycle_gan")


def test_panel_letters_continue_past_z_instead_of_into_punctuation():
    from spectramr.infrastructure.reporting.style import panel_letter

    assert [panel_letter(i) for i in (0, 25, 26, 27, 52, 701, 702)] == ["a", "z", "aa", "ab", "ba", "zz", "aaa"]


def test_select_style_records_the_choice_without_touching_rcparams(monkeypatch):
    monkeypatch.setattr(style_mod, "_ACTIVE_STYLE", "nature")
    before = dict(mpl.rcParams)
    style_mod.select_style("ieee")
    assert dict(mpl.rcParams) == before and style_mod._ACTIVE_STYLE == "ieee"
    with pytest.raises(ValueError, match="unknown report style"):
        style_mod.select_style("sepia")
