"""Every declared format has exactly one codec, chosen by the file's extension."""

from __future__ import annotations

import numpy as np
import pytest

from spectramr.infrastructure.checkpointing import codecs
from spectramr.infrastructure.checkpointing.codecs._registry import (
    codec_for,
    register_codec,
    registered_formats,
)
from spectramr.infrastructure.checkpointing.layout import FORMATS


def test_every_format_has_a_codec() -> None:
    assert registered_formats() == frozenset(FORMATS)


def test_pt_and_pth_share_one_codec() -> None:
    assert type(codec_for("pt")) is type(codec_for("pth"))


def test_unknown_format_raises() -> None:
    with pytest.raises(ValueError, match="not one of"):
        codec_for("ckpt")


def test_a_second_codec_for_a_format_is_refused() -> None:
    with pytest.raises(ValueError, match="already"):

        @register_codec("pt")
        class _Dup:  # pragma: no cover - never instantiated
            pass


def test_write_refuses_a_meta_format_that_contradicts_the_extension(tmp_path, run_state) -> None:
    with pytest.raises(ValueError, match="format"):
        codecs.write(tmp_path / "step_00000150.pt", run_state("safetensors"))


def test_read_of_a_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        codecs.read(tmp_path / "best.pt")


@pytest.mark.parametrize("fmt", ["safetensors", "pt", "pth"])
def test_numpy_scalars_written_are_read_back(tmp_path, run_state, fmt) -> None:
    state = run_state(fmt, role="best")
    state["meta"]["best_metric"]["value"] = np.float64(30.0)
    state["early_stopping"]["best_value"] = np.float32(0.25)
    path = tmp_path / f"best.{fmt}"
    codecs.write(path, state)
    back = codecs.read(path)
    assert float(back["meta"]["best_metric"]["value"]) == 30.0
    assert float(back["early_stopping"]["best_value"]) == 0.25


@pytest.mark.parametrize("fmt", ["safetensors", "pt", "pth"])
@pytest.mark.parametrize(
    "bad",
    [
        object(),
        np.zeros(2, dtype=[("a", "f4"), ("b", "i4")]),
        np.array(["text"]),
        np.array([object()], dtype=object),
    ],
    ids=["object", "structured", "str-array", "object-array"],
)
def test_values_no_reader_can_load_are_refused_at_write(tmp_path, run_state, fmt, bad) -> None:
    state = run_state(fmt)
    state["strategy"] = {"x": bad}
    with pytest.raises(TypeError, match=r"strategy\.x"):
        codecs.write(tmp_path / f"step_00000150.{fmt}", state)
    assert list(tmp_path.iterdir()) == []
