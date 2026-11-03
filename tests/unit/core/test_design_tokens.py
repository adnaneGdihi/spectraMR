"""Tests for :mod:`spectramr.core.design_tokens`."""

from __future__ import annotations

import re

import pytest

from spectramr.core.design_tokens import (
    CATEGORICAL,
    CATEGORICAL_DARK,
    COLOURS,
    OKABE_ITO,
    OVERFLOW,
    colours,
    css_variables,
)


def test_both_modes_define_the_same_roles():
    assert set(COLOURS["light"]) == set(COLOURS["dark"])


def test_every_token_is_a_hex_colour():
    for mode in ("light", "dark"):
        for value in colours(mode).values():
            assert re.fullmatch(r"#[0-9A-F]{6}", value), value


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown colour mode"):
        colours("sepia")  # type: ignore[arg-type]


def test_css_variables_kebab_case_the_role_names():
    css = css_variables("light")
    assert "--ink-2: #475263;" in css and "--accent: #B44A0B;" in css


def test_okabe_ito_is_the_eight_hue_set():
    assert len(OKABE_ITO) == 8 and OKABE_ITO[0] == "#000000"


def test_categorical_has_eight_distinct_slots_per_mode_and_overflow_is_not_one():
    for hues in (CATEGORICAL, CATEGORICAL_DARK):
        assert len(set(hues)) == 8 and OVERFLOW not in hues
