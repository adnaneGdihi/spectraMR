"""The writer lays a trace out as a manifest, one PNG per stage tensor and an overview."""

from __future__ import annotations

import json

import pytest
import torch

from spectramr.core import snapshot_trace as st
from spectramr.core.snapshot_trace import View
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.training.snapshot.writer import (
    MANIFEST,
    OVERVIEW,
    TENSORS,
    TraceWriter,
)

pytestmark = pytest.mark.unit


def _image() -> torch.Tensor:
    y, x = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij")
    return ((x**2 + y**2) < 0.5).float().view(1, 1, 32, 32)


def _trace() -> st.SnapshotTrace:
    img = _image()
    k = fft2c(img.to(torch.complex64))
    trace = st.SnapshotTrace(phase="train", step=7)
    with st.active(trace):
        st.record(
            "batch",
            "input_raw",
            {"input": k},
            views={"input": View(kspace=True, authoritative=True)},
        )
        st.record("model", "output[generator]", {"out": img})
        st.record("batch", "maps", {"coil_maps": torch.rand(1, 4, 32, 32)})
    return trace


def test_manifest_lists_stages_in_recording_order(tmp_path) -> None:
    out = TraceWriter(tmp_path).write(_trace(), {"strategy": "X"})
    assert out == tmp_path / "train" / "step_000007"
    m = json.loads((out / MANIFEST).read_text())
    assert m["strategy"] == "X" and m["step"] == 7
    assert [(s["group"], s["name"]) for s in m["stages"]] == [
        ("batch", "input_raw"),
        ("model", "output[generator]"),
        ("batch", "maps"),
    ]
    entry = m["stages"][0]["tensors"][0]
    assert entry["view"]["kspace"] and entry["is_complex"]
    assert (tmp_path / entry["png"][0]).is_file()
    assert (out / OVERVIEW).is_file()


def test_unchanged_coil_maps_are_written_once(tmp_path) -> None:
    maps = torch.rand(1, 4, 16, 16)
    writer = TraceWriter(tmp_path)
    pngs = []
    for step in (1, 2):
        trace = st.SnapshotTrace(phase="train", step=step)
        with st.active(trace):
            st.record("batch", "input_raw", {"coil_maps": maps})
        out = writer.write(trace, {})
        pngs.append(json.loads((out / MANIFEST).read_text())["stages"][0]["tensors"][0]["png"])
    assert pngs[0] == pngs[1]
    assert len(list(tmp_path.rglob("*coil_maps*.png"))) == 1


def test_knobs_switch_the_outputs(tmp_path) -> None:
    out = TraceWriter(tmp_path, save_images=False, save_tensors=True).write(_trace(), {})
    assert not list(out.glob("*.png"))
    assert set(torch.load(out / TENSORS)) == {
        "batch/input_raw",
        "model/output[generator]",
        "batch/maps",
    }
    out = TraceWriter(tmp_path / "b", save_json=False).write(_trace(), {})
    assert not (out / MANIFEST).exists()


def test_a_line_mask_applied_in_the_image_domain_is_flagged(tmp_path) -> None:
    img = torch.rand(1, 1, 32, 32) + 0.5
    img[..., ::2, :] = 0.0
    trace = st.SnapshotTrace(phase="train", step=0)
    with st.active(trace):
        st.record(
            "prepare",
            "x",
            {"input": fft2c(img.to(torch.complex64))},
            views={"input": View(kspace=True, authoritative=True)},
        )
    out = TraceWriter(tmp_path).write(trace, {})
    note = json.loads((out / MANIFEST).read_text())["stages"][0]["tensors"][0]["note"]
    assert "DOMAIN SUPERPOSITION" in note
