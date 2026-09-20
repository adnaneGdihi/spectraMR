"""CDiffMR adapter — Huang et al. 2023 baseline.

Plan: ``TODO/backlog_baseline_replication_experiment_11.md`` Phase B.

Wraps the upstream CDiffMR UNet from
``external/baselines/cdiffmr/models/network/cdiffmr/network_cdiff_unet2.py``
so it consumes this repo's canonical data layout (``[B, 2, H, W]`` complex,
``fft2c``-centred) without re-implementing the inner architecture.

Huang J, Aviles-Rivero AI, Schönlieb C-B, Yang G. "CDiffMR: Can we Replace
the Gaussian Noise with K-space Undersampling for Fast MRI?" Upstream:
https://github.com/ayanglab/CDiffMR (submodule under
``external/baselines/cdiffmr/``).

**This adapter wraps the upstream network, not the upstream method.** CDiffMR's
contribution is the k-space-undersampling degradation and its reverse routine
(``sampling_routine: x0_step_down`` over ``time_step: 100``, with
``ksu_mask_type: cartesian_random``); :meth:`sample` — the loop that would run it
— raises, and the training path calls :meth:`forward`, one denoise step, under
*this repo's* cold-diffusion schedule. The paradigm is the same family, which is
what makes the substitution easy to miss; the schedule is not the paper's.

The vendored upstream is a script-style codebase (no ``__init__.py``),
so the UNet is loaded via :mod:`importlib.util` rather than a plain
``import`` — this avoids polluting ``sys.path`` with the upstream's
top-level modules (``models``, ``utils``).
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

from spectramr.models.baselines._base import (
    BaselineAdapter,
    CoilHandling,
    FFTNorm,
    UpstreamLossFamily,
)
from spectramr.models.baselines._upstream_import import load_isolated_module
from spectramr.models.registry import register_model

# parents[4], not [3]: this file is `src/spectramr/models/baselines/<x>.py`,
# so [3] is `src/` and [4] is the repo root. It was `src/models/baselines/`
# before the 2026-05 `src -> src/spectramr` refactor, when [3] WAS the root.
# The off-by-one made the vendored upstream unreachable and produced an
# error telling the user to run a vendoring command they had already run.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_CDIFFMR_DIR = _REPO_ROOT / "external" / "baselines" / "cdiffmr"
_UPSTREAM_NETWORK_FILE = _CDIFFMR_DIR / "models" / "network" / "cdiffmr" / "network_cdiff_unet2.py"
_UPSTREAM_PROCESS_FILE = (
    _CDIFFMR_DIR / "models" / "model" / "cdiffmr" / "diffusion_model" / "cdm_ksu_m05.py"
)


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
    module = load_isolated_module(
        _CDIFFMR_DIR, _UPSTREAM_PROCESS_FILE, "_cdiffmr_upstream_process"
    )
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


def _complex_to_real(x: torch.Tensor) -> torch.Tensor:
    """Convert complex ``[B, C, H, W]`` to real ``[B, 2*C, H, W]`` (real|imag stacked)."""
    if torch.is_complex(x):
        return torch.cat([x.real, x.imag], dim=1)
    return x


def _real_to_complex(x: torch.Tensor, out_channels: int) -> torch.Tensor:
    """Convert real ``[B, 2*C, H, W]`` back to complex ``[B, C, H, W]``."""
    if torch.is_complex(x):
        return x
    real, imag = torch.chunk(x, 2, dim=1)
    return torch.complex(real, imag)


@register_model(name="cdiffmr_baseline", training_mode="cold_diffusion")
class CDiffMRBaseline(BaselineAdapter):
    """CDiffMR (Huang 2023) — cascaded diffusion MRI baseline."""

    REPO_NAME = "cdiffmr"
    PAPER_REF = "Huang2023:CDiffMR"
    #: published option file selects fMRI_Ran_AF4_CF0.08_PE320 -- the `random` family.
    PREFERRED_MASK_TYPE = "random_cartesian"
    PREFERRED_FFT_NORM = FFTNorm.ORTHO
    COIL_HANDLING = CoilHandling.RSS
    # `lossfn_type: "l1"` with `lossfn_weight: 1.0` and `alpha: 1`, `beta: null`,
    # `gamma: null` -- so `total_loss` (model_cdiffmr_ksu_m04.py:127) reduces to a
    # single unweighted L1 between the reconstruction and the clean image.
    UPSTREAM_LOSS_FAMILY = UpstreamLossFamily.L1

    def __init__(
        self,
        *,
        in_channels: int = 2,
        out_channels: int = 2,
        resolution: int = 256,
        num_steps: int = 100,
        opt_overrides: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.num_steps = num_steps
        self._record_unconsumed(kwargs)

        # The upstream ``Model`` UNet expects an ``opt`` dict with a small
        # set of keys. We coerce our typed kwargs into that shape and let
        # the caller override individual fields via ``opt_overrides``.
        opt = _default_opt(resolution, in_channels, out_channels)
        if opt_overrides:
            opt.update(opt_overrides)
        self._opt = opt
        self._upstream_process: object | None = None
        ModelCls = _load_upstream_model_class()
        self.upstream = ModelCls(opt)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        *args: object,
        t: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Single denoise-step through the upstream UNet.

        Args:
            x: Image tensor. Accepts either complex ``[B, C, H, W]`` (canonical
                MRI layout) or real ``[B, 2*C, H, W]`` (upstream native layout).
            timesteps: Diffusion timesteps as a 1-D long tensor ``[B]``.
                Named for the kwarg the strategies probe for: both
                ``DiffusionTrainingStrategy`` and ``GraphColdDiffusionStrategy``
                ask ``_callable_accepts_kwarg(forward, "timesteps")``, which
                returns True for any ``**kwargs`` signature -- so under the old
                name ``t`` the value landed in ``**kwargs`` and was dropped,
                pinning every step to t=0 (#2086).
            t: Upstream's spelling, accepted keyword-only so callers written
                against the vendored signature still bind.

        Returns:
            Reconstructed tensor in the same dtype family as the input
            (complex in → complex out, real in → real out).

        Note:
            This implements one denoise step, not the full reverse-diffusion
            sampling loop. The full loop requires a trained checkpoint and
            the upstream ``CDiffMR`` trainer-class wrapper; see
            ``external/baselines/cdiffmr/main_test_cdfiimr_ksu_FastMRI_complex.py``
            for that path.
        """
        input_was_complex = torch.is_complex(x)
        x_real = _complex_to_real(x)
        if timesteps is None:
            timesteps = t
        if timesteps is None:
            timesteps = torch.zeros(x_real.shape[0], dtype=torch.long, device=x_real.device)
        if x_real.shape[2] != self.resolution or x_real.shape[3] != self.resolution:
            # Upstream's own guard (`network_cdiff_unet2.py`) is
            # `assert x.shape[2] == x.shape[3] == self.resolution` with an EMPTY
            # message, so a mismatch here would otherwise die as a bare
            # `AssertionError()`. Naming both values and the two config keys that
            # must agree turns that into something a reader can act on.
            raise ValueError(
                f"{type(self).__name__} was built for resolution={self.resolution} "
                f"(model.model_kwargs.resolution) but received input of spatial "
                f"shape {tuple(x_real.shape[2:])}. Check model.model_kwargs.resolution "
                "against data.sampling.patch_size -- the two must agree."
            )
        out_real = self.upstream(x_real, timesteps)
        if input_was_complex:
            return _real_to_complex(out_real, self.out_channels)
        return out_real

    def _process(self) -> object:
        """Upstream's ``GaussianDiffusion``, built once against the paper's option file.

        Built lazily rather than in ``__init__`` so an arm that only wants the network
        (an architecture comparison) does not pay the isolated import or the mask-ladder
        construction, and so the ladder is built on the device the first step runs on.
        """
        if self._upstream_process is None:
            process_cls = _load_upstream_process_class()
            opt = copy.deepcopy(_upstream_option_dict())
            opt["diffusion"]["time_step"] = self.num_steps
            opt["diffusion"]["degradation"]["pe"] = self.resolution
            opt["diffusion"]["degradation"]["fe"] = self.resolution
            process = process_cls(opt, self.upstream, is_train=True)
            # `process` wraps `self.upstream` as `denoise_fn`, so a plain
            # attribute assignment would register it as an `nn.Module` child and
            # duplicate every parameter into `state_dict()`, which the strict
            # reload in `checkpoint_service`/`checkpoint_director` then refuses.
            # `object.__setattr__` keeps it off the module graph.
            object.__setattr__(self, "_upstream_process", process)
        return self._upstream_process

    def training_loss(
        self,
        x_0: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """``GaussianDiffusion.forward`` plus the L1 its trainer applies to the result.

        Upstream splits these across two files: the diffusion object returns
        ``(x_ksu, x_recon)`` (``cdm_ksu_m05.py:405``) and the trainer computes
        ``lossfn_weight * alpha * L1(x_recon, x_start)`` (``model_cdiffmr_ksu_m04.py:229``
        and ``:146``), with the option file setting both multipliers to 1. Calling the
        object gives the paper's degradation, its uniform timestep draw and its
        x_0-prediction target; only the two-line reduction is spelled out here.

        Args:
            x_0: Clean image, ``[B, 2, H, W]`` real / imaginary.
            batch: Unused -- upstream's training step conditions on nothing. Its three
                conditioning branches are commented out in the source and ``false`` in
                the option file.

        Returns:
            The scalar L1 between upstream's reconstruction and ``x_0``.
        """
        del batch
        _x_ksu, x_recon = self._process()(x_0)
        return torch.nn.functional.l1_loss(x_recon, x_0)

    def sample(
        self,
        x_start: torch.Tensor,
        x_obs: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        batch_size: int = 1,
        timesteps: int | None = None,
    ) -> torch.Tensor:
        """Full CDiffMR reverse-diffusion sampling loop.

        Delegates to ``GaussianDiffusion.sample`` -- the object ``_process()``
        already builds for :meth:`training_loss` -- rather than the ``CDM_KSU``
        trainer wrapper (``models/model/cdiffmr/cdm_ksu_m05.py``): the published
        option file sets ``is_dc: false``, so ``sample()`` never touches
        ``x_obs``/``mask``, and the ``CharbonnierLoss``/``SSIMLoss``/wandb machinery
        the wrapper adds belongs to training a checkpoint, not to running the
        (untrained) reverse loop.

        Raises:
            RuntimeError: upstream hard-codes ``.cuda()`` on its per-step timestep
                tensor (``cdm_ksu_m05.py``), so the loop cannot run off an
                accelerator.
        """
        if x_start.device.type != "cuda":
            raise RuntimeError(
                f"{type(self).__name__}.sample calls upstream GaussianDiffusion.sample, "
                "which hard-codes `.cuda()` on its per-step timestep tensor "
                f"(cdm_ksu_m05.py), so it cannot run on {x_start.device}. Run this arm "
                "on an accelerator (non-negotiable 9b)."
            )
        x_real = _complex_to_real(x_start)
        t = self.num_steps if timesteps is None else timesteps
        _x_t, _direct_recon, x_recon = self._process().sample(
            x_real, x_obs=x_obs, mask=mask, batch_size=batch_size, t=t
        )
        if torch.is_complex(x_start):
            return _real_to_complex(x_recon, self.out_channels)
        return x_recon

    def validation_sample(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Upstream degrades the clean target itself -- ``input_batch`` is unused.

        Mirrors :meth:`training_loss`: the reverse process starts from
        ``target_batch`` and reconstructs it through the paper's own mask ladder,
        the same object :meth:`sample` drives.
        """
        del input_batch, batch
        return self.sample(target_batch, batch_size=target_batch.shape[0])


__all__ = ["CDiffMRBaseline"]
