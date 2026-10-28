"""Locate and load the vendored CDiffMR upstream: its network, its process and its options.

Kept apart from :mod:`spectramr_baselines.cdiffmr` so the adapter carries only the
adaptation. The vendored repository is a script-style codebase with no ``__init__.py``,
so each file is loaded by path rather than by a plain ``import``, which would put the
upstream's top-level ``models`` and ``utils`` packages on ``sys.path``.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import re
import sys
from typing import Any

import torch

from spectramr_baselines._upstream_import import load_isolated_module, vendored_dir

_CDIFFMR_DIR = vendored_dir("cdiffmr")
_UPSTREAM_NETWORK_FILE = _CDIFFMR_DIR / "models" / "network" / "cdiffmr" / "network_cdiff_unet2.py"
_UPSTREAM_PROCESS_FILE = (
    _CDIFFMR_DIR / "models" / "model" / "cdiffmr" / "diffusion_model" / "cdm_ksu_m05.py"
)
_UPSTREAM_MASK_DIR = _CDIFFMR_DIR / "mask" / "fastmri" / "random"


@functools.lru_cache(maxsize=1)
def _upstream_option_dict() -> dict[str, Any]:
    """The published option file, parsed.

    The file is JSON with ``//`` comments, which :mod:`json` rejects, so the comments
    are stripped first. Cached: it is read once per process and copied per use, because
    the diffusion object mutates the dict it is handed.
    """
    path = _CDIFFMR_DIR / _UPSTREAM_OPTION_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"CDiffMR option file not found at {path}. Run the Phase A.1 vendoring "
            "command (git submodule add)."
        )
    return json.loads(re.sub(r"//.*", "", path.read_text()))


def _load_upstream_process_class() -> type:
    """Load upstream's ``GaussianDiffusion`` -- the method, not the network.

    Unlike the network file this one imports ``from utils.utils_kspace_undersampling
    import ...``, so it needs the CDiffMR root on ``sys.path``. FDB ships a ``utils``
    package too, which is why the import is isolated rather than a plain insert.
    """
    if "_cdiffmr_upstream_process" in sys.modules:
        return sys.modules["_cdiffmr_upstream_process"].GaussianDiffusion
    if not _UPSTREAM_PROCESS_FILE.exists():
        raise FileNotFoundError(
            f"CDiffMR upstream not found at {_UPSTREAM_PROCESS_FILE}. "
            "Run the Phase A.1 vendoring command (git submodule add)."
        )
    module = load_isolated_module(_CDIFFMR_DIR, _UPSTREAM_PROCESS_FILE, "_cdiffmr_upstream_process")
    sys.modules["_cdiffmr_upstream_process"] = module
    return module.GaussianDiffusion


def _load_upstream_model_class() -> type[torch.nn.Module]:
    """Load the upstream ``Model`` UNet class without polluting ``sys.path``.

    Uses :func:`importlib.util.spec_from_file_location` to import the
    single file by path. The class is cached on the module after the
    first call so the import cost is paid once.
    """
    if "_cdiffmr_upstream_network" in sys.modules:
        return sys.modules["_cdiffmr_upstream_network"].Model  # type: ignore[attr-defined]
    if not _UPSTREAM_NETWORK_FILE.exists():
        raise FileNotFoundError(
            f"CDiffMR upstream not found at {_UPSTREAM_NETWORK_FILE}. "
            "Run the Phase A.1 vendoring command (git submodule add)."
        )
    spec = importlib.util.spec_from_file_location(
        "_cdiffmr_upstream_network", _UPSTREAM_NETWORK_FILE
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build a spec for {_UPSTREAM_NETWORK_FILE}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_cdiffmr_upstream_network"] = module
    spec.loader.exec_module(module)
    return module.Model


#: The one option file the upstream ships, read for the values below:
#: ``options/CDiffMR/FastMRI/ksu/train_CDiffMR_FastMRIKneePD_m.0.4.s2.ksu.cran.LogSR.d.1.0.cplx.2ch_DEBUG.json``.
#: Its ``denoise_fn`` block is the architecture the paper describes, and
#: ``diffusion.time_step`` its step count; its ``train``
#: block is NOT -- the file is a ``_DEBUG`` variant with ``checkpoint_save: 10``
#: and ``"model_DM_optimizer_lr": 2e-5`` carrying the comment ``// learning rate
#: default 2e-4``. So the architecture is taken from it and the schedule is not.
_UPSTREAM_OPTION_FILE = (
    "options/CDiffMR/FastMRI/ksu/"
    "train_CDiffMR_FastMRIKneePD_m.0.4.s2.ksu.cran.LogSR.d.1.0.cplx.2ch_DEBUG.json"
)


def _default_opt(resolution: int, in_channels: int, out_channels: int) -> dict[str, Any]:
    """The upstream's published ``denoise_fn`` block, verbatim except ``resolution``.

    ``attn_resolutions`` is an absolute feature-map size, not a fraction of the input: with
    ``emb_channels_multi: [1, 2, 2, 2]`` the UNet's four levels visit 256/128/64/32 (or, at the
    upstream's own 320, 320/160/80/40), so the published ``[16]`` matches none of them and no
    per-level ``AttnBlock`` is built -- only the unconditional ``mid.attn_1`` (counted by
    construction: exactly 1, at both 256 and 320). This is upstream's own behaviour, not a gap in
    this adapter: the published option file declares the same ``[1, 2, 2, 2]`` + ``[16]`` pair.

    ``resolution`` stays a parameter because the upstream trains at 320 (fastMRI knee) and this
    corpus patches at 256.
    """
    return {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "resolution": resolution,
        "emb_channels": 128,
        "emb_channels_multi": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "attn_resolutions": [16],
        "dropout": 0.1,
        "resamp_with_conv": True,
    }
