"""Tests for ``safe_torch_load`` and ``atomic_save_json``.

Targets ``spectramr.shared.utils.safe_io``. The safe-load wrapper enforces
CPU map_location + ``weights_only=True`` (when supported), preventing
arbitrary code execution from a malicious checkpoint. The atomic JSON
saver guarantees crash safety via temp-file + rename.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from spectramr.shared.utils.safe_io import atomic_save_json, safe_torch_load


# ---------------------------------------------------------------------------
# safe_torch_load
# ---------------------------------------------------------------------------


def test_safe_torch_load_missing_file_raises(tmp_path: Path) -> None:
    """File-not-found surfaces as ``FileNotFoundError`` (not silent)."""
    with pytest.raises(FileNotFoundError):
        safe_torch_load(tmp_path / "absent.pt")


def test_safe_torch_load_round_trip_tensor(tmp_path: Path) -> None:
    """A tensor saved via ``torch.save`` round-trips correctly."""
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    f = tmp_path / "data.pt"
    torch.save(tensor, f)
    out = safe_torch_load(f)
    assert torch.equal(out, tensor)


def test_safe_torch_load_round_trip_state_dict(tmp_path: Path) -> None:
    """State-dict checkpoint round-trip works under ``weights_only=True``."""
    state = {"w": torch.zeros(4), "b": torch.ones(4)}
    f = tmp_path / "ckpt.pt"
    torch.save(state, f)
    loaded = safe_torch_load(f)
    assert isinstance(loaded, dict)
    assert torch.equal(loaded["w"], torch.zeros(4))
    assert torch.equal(loaded["b"], torch.ones(4))


def test_safe_torch_load_uses_cpu_map_location(tmp_path: Path) -> None:
    """``map_location='cpu'`` default places loaded tensors on CPU."""
    f = tmp_path / "cpu_default.pt"
    torch.save(torch.randn(4), f)
    loaded = safe_torch_load(f)
    assert loaded.device.type == "cpu"


def test_safe_torch_load_traversal_relative_path_rejected() -> None:
    """A relative path normalising to ``..`` is rejected."""
    with pytest.raises(ValueError, match="Unsafe relative path"):
        safe_torch_load("../../../etc/passwd")


def test_safe_torch_load_absolute_path_works(tmp_path: Path) -> None:
    """Absolute paths bypass the ``..`` guard (covered by FS permissions)."""
    f = tmp_path / "abs.pt"
    torch.save(torch.zeros(2), f)
    loaded = safe_torch_load(str(f.resolve()))
    assert torch.equal(loaded, torch.zeros(2))


# ---------------------------------------------------------------------------
# atomic_save_json
# ---------------------------------------------------------------------------


def test_atomic_save_json_writes_data(tmp_path: Path) -> None:
    """Saved JSON file contains the expected payload."""
    data = {"a": 1, "b": [2, 3], "c": "hello"}
    f = tmp_path / "out.json"
    atomic_save_json(data, f)
    assert f.is_file()
    assert json.loads(f.read_text()) == data


def test_atomic_save_json_creates_parent_directory(tmp_path: Path) -> None:
    """Missing parent directory is created automatically."""
    f = tmp_path / "deep" / "nested" / "out.json"
    assert not f.parent.exists()
    atomic_save_json({"k": 1}, f)
    assert f.is_file()


def test_atomic_save_json_overwrites_existing(tmp_path: Path) -> None:
    """An existing file is overwritten atomically."""
    f = tmp_path / "out.json"
    f.write_text('{"old": 1}')
    atomic_save_json({"new": 2}, f)
    assert json.loads(f.read_text()) == {"new": 2}


def test_atomic_save_json_no_temp_files_left_on_success(tmp_path: Path) -> None:
    """Successful save leaves no `.tmp` siblings in the parent directory."""
    f = tmp_path / "out.json"
    atomic_save_json({"k": 1}, f)
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_atomic_save_json_rejects_non_serializable() -> None:
    """Non-JSON-serialisable input raises (TypeError from json.dump)."""
    with pytest.raises((TypeError, ValueError)):
        atomic_save_json({"bad": object()}, "/tmp/test_should_not_exist.json")


# --------------------------------------------------------------------------- #
# #1352 -- checkpoint writes must be atomic
# --------------------------------------------------------------------------- #
class TestAtomicSaveTorch:
    """A run killed mid-``torch.save`` used to leave a truncated checkpoint.

    A truncated file passes every existence check and only fails at load, often
    days later when the run is resumed or graded.
    """

    def test_payload_round_trips(self, tmp_path):
        import torch

        from spectramr.shared.utils.safe_io import atomic_save_torch

        dst = tmp_path / "ckpt.pt"
        atomic_save_torch({"step": 7}, dst)
        assert torch.load(dst, weights_only=False)["step"] == 7

    def test_missing_parent_directory_is_created(self, tmp_path):
        from spectramr.shared.utils.safe_io import atomic_save_torch

        dst = tmp_path / "nested" / "deeper" / "ckpt.pt"
        atomic_save_torch({"a": 1}, dst)
        assert dst.exists()

    def test_a_failed_write_leaves_the_previous_file_intact(self, tmp_path):
        """The property that makes it atomic, planted rather than assumed."""
        import pytest
        import torch

        from spectramr.shared.utils import safe_io

        dst = tmp_path / "ckpt.pt"
        safe_io.atomic_save_torch({"generation": 1}, dst)

        def explode(obj, path):
            raise RuntimeError("disk full")

        original = safe_io.torch.save
        safe_io.torch.save = explode
        try:
            with pytest.raises(RuntimeError, match="disk full"):
                safe_io.atomic_save_torch({"generation": 2}, dst)
        finally:
            safe_io.torch.save = original

        assert torch.load(dst, weights_only=False)["generation"] == 1

    def test_a_failed_write_leaves_no_temp_file_behind(self, tmp_path):
        import pytest

        from spectramr.shared.utils import safe_io

        dst = tmp_path / "ckpt.pt"

        def explode(obj, path):
            raise RuntimeError("nope")

        original = safe_io.torch.save
        safe_io.torch.save = explode
        try:
            with pytest.raises(RuntimeError):
                safe_io.atomic_save_torch({"a": 1}, dst)
        finally:
            safe_io.torch.save = original

        assert list(tmp_path.iterdir()) == []


class TestAtomicPublish:
    """The one temp-file-then-rename primitive every checkpoint writer uses."""

    def test_publishes_what_write_produced(self, tmp_path):
        from spectramr.shared.utils.safe_io import atomic_publish

        target = tmp_path / "sub" / "best.pt"
        atomic_publish(target, lambda tmp: Path(tmp).write_bytes(b"new"))
        assert target.read_bytes() == b"new"

    def test_failed_write_leaves_target_and_no_temp(self, tmp_path):
        from spectramr.shared.utils.safe_io import atomic_publish

        target = tmp_path / "best.pt"
        target.write_bytes(b"old")

        def boom(tmp: str) -> None:
            Path(tmp).write_bytes(b"half")
            raise RuntimeError("disk full")


        with pytest.raises(RuntimeError, match="disk full"):
            atomic_publish(target, boom)
        assert target.read_bytes() == b"old"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["best.pt"]
