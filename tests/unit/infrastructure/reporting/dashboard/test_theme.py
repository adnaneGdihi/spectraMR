"""Tests for :mod:`spectramr.infrastructure.reporting.dashboard.theme`."""

from __future__ import annotations

from spectramr.core.design_tokens import colours
from spectramr.infrastructure.reporting.dashboard.theme import script, stylesheet


def test_stylesheet_takes_its_colours_from_the_tokens():
    css = stylesheet()
    assert colours("light")["accent"] in css and colours("dark")["page"] in css
    assert "Source Serif 4" in css and "IBM Plex Sans" in css


def test_an_explicit_light_choice_beats_a_dark_system_preference():
    assert ":root:not([data-theme=light])" in stylesheet()


def test_switch_survives_blocked_storage_and_restyles_role_tagged_markers():
    js = script()
    assert "try{return localStorage.getItem" in js and "try{localStorage.setItem" in js
    assert "marker_roles" in js and "Plotly.restyle" in js
