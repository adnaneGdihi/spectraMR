"""Parity gate: the adapters must construct the authors' network, not one like it.

The backlog's doctrine is "faithful or not at all" — a baseline that scores badly
because it was built sloppily stacks the deck. Its Phase A.4 shipped the adapter
base class but never the gate, so nothing checked the claim
(``TODO/backlog_baseline_replication_experiment_11.md``).

**The comparison is against each author's PUBLISHED INVOCATION, never their library
defaults.** FDB's ``model_and_diffusion_defaults()`` says ``num_res_blocks=2,
dropout=0.0``; its README trains with ``--num_res_blocks 3 --dropout 0.3``. A gate
pinned to the defaults would have called the correct values a regression.

**What this gate does NOT prove:** the adapters wrap the authors' *networks* and run
this repo's forward process — CDiffMR's mask ladder and FDB's point-removal
``q_sample`` + calibrated ``w.npy`` schedule are not executed (#2080, #2087). Network
parity is necessary, not sufficient.
"""

from __future__ import annotations

import importlib
import json
import re
import shlex
from pathlib import Path

import pytest
import torch

from spectramr.models.baselines.cdiffmr import (
    _UPSTREAM_NETWORK_FILE,
    CDiffMRBaseline,
    _default_opt,
    _load_upstream_model_class,
)
from spectramr.models.baselines.fdb import (
    _FDB_SCRIPT_UTIL,
    FDBBaseline,
    _ensure_upstream_on_sys_path,
)

_FDB_README = _FDB_SCRIPT_UTIL.parents[1] / "README.md"
_CDIFFMR_OPTION = (
    _UPSTREAM_NETWORK_FILE.parents[3]
    / "options/CDiffMR/FastMRI/ksu"
    / "train_CDiffMR_FastMRIKneePD_m.0.4.s2.ksu.cran.LogSR.d.1.0.cplx.2ch_DEBUG.json"
)

needs_cdiffmr = pytest.mark.skipif(
    not _UPSTREAM_NETWORK_FILE.exists() or not _CDIFFMR_OPTION.exists(),
    reason="CDiffMR submodule not initialised (git submodule update --init)",
)
needs_fdb = pytest.mark.skipif(
    not _FDB_SCRIPT_UTIL.exists() or not _FDB_README.exists(),
    reason="FDB submodule not initialised (git submodule update --init)",
)


def _paper_denoise_fn() -> dict:
    """The ``denoise_fn`` block of CDiffMR's own training option file."""
    return json.loads(re.sub(r"//.*", "", _CDIFFMR_OPTION.read_text()))["denoise_fn"]


def _keys_the_upstream_network_reads() -> set[str]:
    """Derive the consumed keys from upstream's source, not from a hand-list.

    A hand-list silently stops covering a key upstream starts reading; this
    re-derives on every run, so a submodule bump that adds one is visible here.
    """
    src = _UPSTREAM_NETWORK_FILE.read_text()
    # Scope to `Model` -- the file defines five `__init__`s and the first belongs
    # to a helper block that reads no options.
    body = src.split("class Model(nn.Module):", 1)[1]
    body = body.split("def __init__", 1)[1].split("\n    def ", 1)[0]
    return set(re.findall(r"opt\[['\"](\w+)['\"]\]", body))


def _fdb_published_train_flags() -> dict[str, str]:
    """The single-coil ``train.py`` invocation from FDB's README."""
    line = next(
        ln for ln in _FDB_README.read_text().splitlines() if "train.py" in ln and "singlecoil" in ln
    )
    toks = shlex.split(line)
    return {
        toks[i].lstrip("-"): toks[i + 1] for i in range(len(toks) - 1) if toks[i].startswith("--")
    }


# ---------------------------------------------------------------------------
# CDiffMR
# ---------------------------------------------------------------------------


@needs_cdiffmr
def test_cdiffmr_construction_matches_the_published_option_file() -> None:
    """Every key the upstream network reads must carry the paper's value.

    ``resolution`` is the one documented deviation: the option file trains on
    fastMRI knee at 320, the arm trains on M4Raw at 256. The gate pins it so the
    deviation stays deliberate rather than becoming drift.
    """
    paper = _paper_denoise_fn()
    ours = _default_opt(in_channels=2, out_channels=2, resolution=paper["resolution"])
    consumed = _keys_the_upstream_network_reads()
    assert consumed, "failed to derive the consumed keys — upstream layout changed"

    mismatched = {
        k: (paper[k], ours.get(k)) for k in consumed if k in paper and ours.get(k) != paper[k]
    }
    assert not mismatched, f"adapter diverges from the published option file: {mismatched}"
    assert not (consumed - set(ours)), f"adapter omits consumed keys: {consumed - set(ours)}"


@needs_cdiffmr
def test_cdiffmr_adapter_output_is_the_upstream_network_output() -> None:
    """The complex<->real wrapper must be transparent, not merely close."""
    torch.manual_seed(0)
    ours = CDiffMRBaseline(in_channels=2, out_channels=2, resolution=64).eval()
    theirs = _load_upstream_model_class()(
        _default_opt(in_channels=2, out_channels=2, resolution=64)
    ).eval()
    theirs.load_state_dict(ours.upstream.state_dict())

    x = torch.randn(1, 2, 64, 64)
    t = torch.full((1,), 250, dtype=torch.long)
    with torch.no_grad():
        mine, upstream = ours(x, timesteps=t), theirs(x, t)
    assert set(ours.upstream.state_dict()) == set(theirs.state_dict())
    assert torch.equal(mine, upstream), (
        f"wrapper altered the result: {(mine - upstream).abs().max()}"
    )


@needs_cdiffmr
def test_a_drifted_construction_key_is_caught() -> None:
    """The violation this gate exists for, planted."""
    paper = _paper_denoise_fn()
    drifted = dict(_default_opt(in_channels=2, out_channels=2, resolution=paper["resolution"]))
    drifted["num_res_blocks"] = paper["num_res_blocks"] + 1
    consumed = _keys_the_upstream_network_reads()
    mismatched = {
        k: (paper[k], drifted.get(k)) for k in consumed if k in paper and drifted.get(k) != paper[k]
    }
    assert "num_res_blocks" in mismatched


# ---------------------------------------------------------------------------
# FDB
# ---------------------------------------------------------------------------


@needs_fdb
def test_fdb_construction_matches_the_published_train_command() -> None:
    """Compare to the README's invocation, NOT ``model_and_diffusion_defaults()``.

    The defaults function disagrees with the published command on
    ``num_res_blocks`` (2 vs 3) and ``dropout`` (0.0 vs 0.3); the command is the
    authority for what the paper's numbers were produced with.
    """
    flags = _fdb_published_train_flags()
    adapter = FDBBaseline(
        image_size=int(flags["image_size"]),
        bridge_steps=int(flags["diffusion_steps"]),
        undersampling_rate=int(flags["undersampling_rate"]),
    )
    built = adapter._model_kwargs

    _ensure_upstream_on_sys_path()
    consumed = set(importlib.import_module("utils.script_util_duo").model_and_diffusion_defaults())

    def _coerce(raw: str, ref: object) -> object:
        if isinstance(ref, bool):
            return raw.strip().strip("'\"").lower() == "true"
        return type(ref)(raw) if isinstance(ref, (int, float)) else raw.strip("'\"")

    mismatched = {
        k: (flags[k], built.get(k))
        for k in consumed & set(flags)
        if built.get(k) != _coerce(flags[k], built.get(k))
    }
    assert not mismatched, f"adapter diverges from the published command: {mismatched}"


@needs_fdb
def test_the_published_command_and_the_library_defaults_really_do_disagree() -> None:
    """Pins why this gate reads the README.

    If upstream ever reconciles its defaults with its README this test goes red,
    and the rule above can be simplified — deliberately, rather than by accident.
    """
    _ensure_upstream_on_sys_path()
    defaults = importlib.import_module("utils.script_util_duo").model_and_diffusion_defaults()
    flags = _fdb_published_train_flags()
    assert defaults["num_res_blocks"] != int(flags["num_res_blocks"])
    assert defaults["dropout"] != float(flags["dropout"])


# ---------------------------------------------------------------------------
# What parity does not buy
# ---------------------------------------------------------------------------


@needs_fdb
def test_the_authors_forward_process_is_not_what_runs() -> None:
    """Network parity is not method parity — keep the gap visible (#2080).

    FDB's degradation removes individual 2D k-space points behind a shrinking
    circular ACS and weights steps by a calibrated ``w.npy``. The arm runs
    spectraMR's Cartesian line mask instead. Asserting the upstream code exists
    while nothing in ``src/`` calls it is what keeps that honest.
    """
    q_sample_src = (_FDB_SCRIPT_UTIL.parents[0] / "fdb.py").read_text()
    assert "def q_sample" in q_sample_src
    assert 'np.save("w.npy"' in q_sample_src

    root = Path(__file__).resolve().parents[3] / "src" / "spectramr"
    callers = [p for p in root.rglob("*.py") if "w.npy" in p.read_text()]
    assert not callers, f"if this fails, wire the parity claim too: {callers}"
