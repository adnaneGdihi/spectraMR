"""Planted violations for the logging-owner ratchet (non-negotiable 15).

One test per rule the gate claims to enforce and one per shape that rule can take,
plus sanity tests that a named logger is not flagged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.utils.repo_scripts import load_script_module

gate = load_script_module("scripts/ci/check_logging_owner.py", "logging_owner_gate")


def _kinds(src: str) -> dict[str, int]:
    return dict(gate.count_in_source(src))


@pytest.mark.parametrize(
    "src",
    [
        "import logging\nlogging.basicConfig(level=10)\n",
        "import logging as std_logging\nstd_logging.basicConfig()\n",
        "from logging import basicConfig\nbasicConfig()\n",
        "import logging\ndef f():\n    logging.basicConfig(force=True)\n",
    ],
)
def test_catches_basicconfig_shapes(src):
    assert _kinds(src) == {"basicconfig": 1}


@pytest.mark.parametrize(
    "src",
    [
        "import logging\nlogging.getLogger().setLevel(10)\n",
        "import logging\nlogging.root.setLevel(10)\n",
        "import logging\nlogging.getLogger('').setLevel(10)\n",
        "import logging\nroot = logging.getLogger()\nroot.setLevel(10)\n",
        "import logging\ndef f():\n    r = logging.root\n    r.setLevel(10)\n",
    ],
)
def test_catches_root_level_shapes(src):
    assert _kinds(src) == {"root-level": 1}


@pytest.mark.parametrize(
    "src",
    [
        "import logging\nlogging.getLogger().addHandler(h)\n",
        "import logging\nlogging.root.removeHandler(h)\n",
        "import logging\nroot = logging.getLogger()\nroot.addHandler(h)\n",
    ],
)
def test_catches_root_handler_shapes(src):
    assert _kinds(src) == {"root-handler": 1}


@pytest.mark.parametrize(
    "src",
    [
        "import logging\nlogging.getLogger = lambda *a: None\n",
        "import logging\nsetattr(logging, 'getLogger', f)\n",
    ],
)
def test_catches_getlogger_patch_shapes(src):
    assert _kinds(src) == {"patch": 1}


@pytest.mark.parametrize(
    "src",
    [
        "svc = LoggingService()\n",
        "svc = ExperimentLogger(name='x')\n",
        "import m\nsvc = m.LoggingService()\n",
    ],
)
def test_catches_service_construction_shapes(src):
    assert sum(_kinds(src).values()) == 1 and "service-ctor" in _kinds(src)


@pytest.mark.parametrize(
    "src",
    [
        "import logging\nlog = logging.getLogger(__name__)\nlog.setLevel(10)\nlog.addHandler(h)\n",
        "import logging\nlogging.getLogger('spectramr.x').setLevel(10)\n",
        "import logging\nlogging.getLogger().info('hello')\n",
        "svc = LoggingServiceFactory.create(cfg)\n",
    ],
)
def test_named_loggers_and_factory_are_not_flagged(src):
    assert _kinds(src) == {}


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    for root in gate.SCAN_ROOTS:
        (tmp_path / root).mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_owner_modules_are_exempt_and_others_are_not(tmp_path):
    bad = "import logging\nlogging.getLogger().setLevel(10)\n"
    owner = sorted(gate.OWNERS)[0]
    repo = _repo(tmp_path, {owner: bad, "src/spectramr/other.py": bad})
    found, unparseable = gate.scan(repo)
    assert found == {"src/spectramr/other.py:root-level": 1}
    assert unparseable == []


def test_unparseable_file_is_reported_not_skipped(tmp_path):
    repo = _repo(tmp_path, {"scripts/broken.py": "def (:\n"})
    assert gate.scan(repo)[1] == ["scripts/broken.py"]


def test_compare_flags_new_grown_and_stale():
    regressions, stale = gate.compare(
        {"a.py:basicconfig": 2, "b.py:patch": 1}, {"a.py:basicconfig": 1, "c.py:patch": 1}
    )
    text = "\n".join(regressions)
    assert "GREW  a.py:basicconfig" in text and "NEW   b.py:patch" in text
    assert [s for s in stale if "c.py:patch" in s]


def test_write_refuses_to_raise_a_ceiling(tmp_path, monkeypatch):
    repo = _repo(
        tmp_path, {"scripts/x.py": "import logging\nlogging.basicConfig()\nlogging.basicConfig()\n"}
    )
    base = repo / "scripts/ci/baselines"
    base.mkdir(parents=True)
    (base / gate.BASELINE_FILENAME).write_text("scripts/x.py:basicconfig 1\n")
    monkeypatch.setattr(gate, "__file__", str(repo / "scripts/ci/check_logging_owner.py"))
    monkeypatch.setattr("sys.argv", ["check_logging_owner", "--write"])
    assert gate.main() == 1


def test_repository_src_tree_is_clean_and_matches_baseline():
    repo = Path(__file__).resolve().parents[3]
    current, _ = gate.scan(repo)
    assert not [k for k in current if k.startswith("src/")]
    baseline = gate._read_baseline(repo / "scripts/ci/baselines" / gate.BASELINE_FILENAME)
    regressions, stale = gate.compare(current, baseline)
    assert not regressions and not stale
