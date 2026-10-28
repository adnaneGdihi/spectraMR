"""FDB (Fourier-constrained Diffusion Bridge) adapter.

Mirza MU, Dalmaz O, Bedel HA, Elmas G, Korkmaz Y, Gungor A, Dar SUH,
Çukur T. "Learning Fourier-Constrained Diffusion Bridges for MRI
Reconstruction." arXiv:2308.01096, 2023.
Upstream: https://github.com/icon-lab/FDB (submodule under
``external/baselines/fdb/``).

Plan: ``TODO/backlog_baseline_replication_experiment_11.md`` Phase D.

Wraps the upstream ``UNetModel`` and ``DiffusionBridge`` from
``external/baselines/fdb/utils/`` so they consume this repo's canonical
data layout (``[B, 2, H, W]`` complex, ``fft2c``-centred).

``PAPER_REF`` carries the eight authors of the upstream README and the paper's own
title, "Fourier-Constrained" (#2087).

Training runs the authors' ``DiffusionBridge.training_losses`` around their own
``UNetModel`` through :class:`~spectramr_baselines.strategy.UpstreamProcessStrategy`.
The one step this adapter adds is the intensity scale of the authors' ``read_data.py``,
``x / abs(x).max()``, which they apply per subject; the loader here delivers slices, so
it is applied per slice.

Validation runs the authors' reverse bridge (``p_sample_loop_condition``) the way their
``sample.py`` does: at the test acceleration, over the paper's longer test schedule, from
the k-space of the input through a 2-D normal-density mask (:mod:`._fdb_mask`, #2255).
"""

from __future__ import annotations

import collections
import contextlib
import importlib
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spectramr.models.baselines._base import (
    BaselineAdapter,
    CoilHandling,
    FFTNorm,
    UpstreamLossFamily,
)
from spectramr.models.registry import register_model
from spectramr_baselines._fdb_mask import (
    bridge_test_steps,
    check_bridge_feasible,
    last_round_radius,
    normal_density_mask,
    seeded_numpy,
)
from spectramr_baselines._normalization import peak_magnitude, require_image_size
from spectramr_baselines._upstream_import import vendored_dir

logger = logging.getLogger(__name__)

_FDB_DIR = vendored_dir("fdb")
# Sentinel FILE, not the directory. `git submodule add` records a gitlink, so a
# checkout that never ran `git submodule update --init` leaves an EMPTY directory
# behind — `_FDB_DIR.exists()` is True there and the guard waves it through, so
# the failure surfaced as a bare `ModuleNotFoundError: utils.script_util_duo`
# with no mention of vendoring. cdiffmr.py already checks its own sentinel file
# (`_UPSTREAM_NETWORK_FILE`); this restores parity.
_FDB_SCRIPT_UTIL = _FDB_DIR / "utils" / "script_util_duo.py"


def _ensure_upstream_on_sys_path() -> None:
    """Add the FDB root to ``sys.path`` so ``from utils.<x>`` resolves.

    The FDB upstream is a script-style codebase: its ``utils/`` package
    is meant to be imported with the FDB root as the working directory.
    We add the root to ``sys.path`` (idempotent) so the regular Python
    import machinery resolves ``from utils.script_util_duo import ...``.
    """
    if not _FDB_SCRIPT_UTIL.exists():
        raise FileNotFoundError(
            f"FDB upstream not found at {_FDB_SCRIPT_UTIL}. "
            f"The directory {_FDB_DIR} "
            f"{'exists but is empty — the submodule was never initialised' if _FDB_DIR.exists() else 'does not exist'}. "
            "Run `git submodule update --init --recursive`, or the Phase A.1 "
            "vendoring command (git submodule add)."
        )
    dir_str = str(_FDB_DIR)
    if dir_str not in sys.path:
        sys.path.insert(0, dir_str)


def _load_upstream_factory() -> Any:
    """Import + return the upstream ``create_model_and_diffusion`` factory."""
    _ensure_upstream_on_sys_path()
    mod = importlib.import_module("utils.script_util_duo")
    return mod.create_model_and_diffusion


def _default_model_kwargs(image_size: int) -> dict[str, Any]:
    """Defaults matching the upstream's ``model_and_diffusion_defaults()``.

    Kept here so the adapter has a stable construction surface even if
    upstream tweaks its defaults — Phase A.4 provenance records these
    so the regulatory bundle can prove what was used.
    """
    return {
        "image_size": image_size,
        "class_cond": False,
        "learn_sigma": False,
        "num_channels": 128,
        # 3 and 0.3, from the upstream README's own single-coil TRAIN command:
        #   --num_channels 128 --num_res_blocks 3 --learn_sigma False --dropout 0.3
        #   --diffusion_steps 1000 --lr 1e-4 --batch_size 1 --image_size 256
        #   --lr_anneal_steps 100000 --undersampling_rate 2
        # They were 2 and 0.0 -- the upstream's `model_and_diffusion_defaults()`,
        # which this docstring claimed to mirror and did, but those defaults are
        # the library's, not the ones the paper was trained with. The gap is
        # 124.05M parameters against 164.30M, measured.
        "num_res_blocks": 3,
        "num_heads": 4,
        "num_heads_upsample": -1,
        "attention_resolutions": "16,8",
        "dropout": 0.3,
        "diffusion_steps": 1000,
        "use_checkpoint": False,
        "use_scale_shift_norm": True,
        # The paper TRAINS at R=2 and INFERS at R=4 (`sample.py --R 4`); the two
        # are different knobs and this is the training one. 4 was neither.
        "undersampling_rate": 2,
        "data_type": "singlecoil",
    }

_IMAGE_SIZE_KEY = "model.model_kwargs.image_size"


def _complex_to_real(x: torch.Tensor) -> torch.Tensor:
    """Convert complex ``[B, C, H, W]`` to real ``[B, 2*C, H, W]`` (real|imag stacked)."""
    if torch.is_complex(x):
        return torch.cat([x.real, x.imag], dim=1)
    return x


def _real_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert real ``[B, 2*C, H, W]`` back to complex ``[B, C, H, W]``."""
    if torch.is_complex(x):
        return x
    real, imag = torch.chunk(x, 2, dim=1)
    return torch.complex(real, imag)


#: The lowest timestep upstream's ``q_sample`` is defined at. See the note in
#: :meth:`FDBBaseline.training_loss`.
_FDB_MIN_TIMESTEP = 1


@register_model(name="fdb_baseline", training_mode="cold_diffusion", output_domain="image")
class FDBBaseline(BaselineAdapter):
    """FDB (Mirza et al. 2023) — Fourier-constrained diffusion-bridge baseline."""

    REPO_NAME = "fdb"
    #: The string a results table would cite: ``baseline_provenance`` collects
    #: it, though no production path calls that yet (#2087). It read
    #: ``Karaoglu2024:FDB``, naming neither an author nor a publication year.
    PAPER_REF = "Mirza2023:FDB"
    #: The nearest name the framework has, and NOT what upstream does. FDB's
    #: `q_sample` removes individual 2D k-space POINTS behind a circular ACS whose
    #: radius shrinks with t; no registered accelerator implements that, and
    #: `cartesian_peripheral` is static-only (no timestep ladder). Kept because it
    #: names the right ordering -- periphery inward -- and refusing to declare
    #: anything would lose that. The gap is #2087.
    PREFERRED_MASK_TYPE = "cartesian_peripheral"
    PREFERRED_FFT_NORM = FFTNorm.ORTHO
    COIL_HANDLING = CoilHandling.MULTI_COIL_KSPACE
    # `terms["loss"] = mean_flat((x_0 - model_output) ** 2)` (`utils/fdb.py:318`) --
    # MSE on x_0, not the L1 the other two use and not an epsilon target.
    UPSTREAM_LOSS_FAMILY = UpstreamLossFamily.L2
    # Upstream's `UNetModel` ends in `zero_module(conv_nd(...))`
    # (external/baselines/fdb/utils/unet.py:436), so a freshly constructed instance
    # emits exactly zero for any input -- standard guided-diffusion zero-init, not a
    # facade. `synthetic_forward_probe` cannot tell that apart from a genuinely
    # measurement-independent output, so this skip is broader than what it names: the
    # forward pass DOES consume its input once trained, and this blanket opt-out will
    # not catch a real post-training collapse into the same input-invariant shape.
    synthetic_forward_probe_skip = frozenset({"input_invariant"})

    def __init__(
        self,
        *,
        in_channels: int = 2,
        out_channels: int = 2,
        image_size: int = 256,
        bridge_steps: int = 1000,
        undersampling_rate: int = 4,
        model_kwarg_overrides: dict[str, Any] | None = None,
        test_acceleration: int = 4,
        test_mask_sigma: float = 0.3,
        test_mask_seed: int = 0,
        **kwargs: object,
    ) -> None:
        """Build the upstream UNet and bridge.

        Args:
            undersampling_rate: The training R' of the bridge's forward process.
            test_acceleration: The R validation reconstructs at; ``sample.py --R 4``.
            test_mask_sigma: Width of the test mask's normal density, as a fraction of
                the half field of view. The authors' masks are not published, so this
                value is an assumption, recorded in :meth:`provenance`.
            test_mask_seed: Seeds the test mask and the reverse loop's ``create_mask``.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.bridge_steps = bridge_steps
        self._record_unconsumed(kwargs)

        model_kwargs = _default_model_kwargs(image_size)
        model_kwargs["diffusion_steps"] = bridge_steps
        model_kwargs["undersampling_rate"] = undersampling_rate
        if model_kwarg_overrides:
            model_kwargs.update(model_kwarg_overrides)
        self._model_kwargs = model_kwargs

        factory = _load_upstream_factory()
        model, diffusion = factory(**model_kwargs)
        # ``model`` is the UNet (nn.Module); ``diffusion`` is the
        # ``DiffusionBridge`` object that implements the sampling loop.
        self.upstream = model
        self._diffusion = diffusion
        # The bridge calibration lives in ``w.npy`` beside the working directory, so a
        # checkpoint alone resumed or evaluated in a fresh process ran with w = 0. A
        # persistent buffer puts it in the state dict.
        self.register_buffer(
            "calibration_w",
            torch.as_tensor(np.asarray(diffusion.w, dtype=np.float64)).clone(),
        )
        self.undersampling_rate = undersampling_rate
        self._calibration_dir: Path | None = None
        self.test_acceleration = test_acceleration
        self.test_mask_sigma = test_mask_sigma
        self.test_mask_seed = test_mask_seed
        self.test_bridge_steps = bridge_test_steps(
            model_kwargs["diffusion_steps"], model_kwargs["undersampling_rate"], test_acceleration
        )
        self._test_mask: torch.Tensor | None = None

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
            x: Input tensor. Accepts complex ``[B, C, H, W]`` (canonical)
                or real ``[B, 2*C, H, W]`` (upstream native).
            timesteps: Diffusion timesteps as a 1-D long tensor ``[B]``.
                Named for the kwarg the strategies probe for; under the old
                name ``t`` the value landed in ``**kwargs`` and was dropped,
                pinning every step to t=0 (#2086).
            t: Upstream's spelling, accepted keyword-only for compatibility.

        Returns:
            Tensor in the same dtype family as the input.

        Note:
            The full FDB bridge sampling loop lives on
            ``self._diffusion.p_sample_loop_condition(...)`` and requires
            a trained checkpoint + a mask + an optional coil-map. Single
            denoise-step here is the smoke-test path; campaign runs
            invoke the diffusion loop directly.
        """
        input_was_complex = torch.is_complex(x)
        x_real = _complex_to_real(x)
        if timesteps is None:
            timesteps = t
        if timesteps is None:
            timesteps = torch.zeros(x_real.shape[0], dtype=torch.long, device=x_real.device)
        require_image_size(x_real, self.image_size, type(self).__name__, _IMAGE_SIZE_KEY)
        out_real = self.upstream(x_real, timesteps)
        if input_was_complex:
            return _real_to_complex(out_real)
        return out_real

    def training_loss(
        self,
        x_0: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """``DiffusionBridge.training_losses`` -- upstream's own objective, unmodified.

        The bridge removes ``t*N/T`` k-space points outside a radius shrinking from
        ``image_size/2`` to 3, then scores ``mean_flat((x_0 - model_output) ** 2)``
        (``utils/fdb.py:69`` and ``:318``). Nothing here reimplements that.

        Two upstream behaviours the caller has to live with, stated rather than
        worked around:

        * ``q_sample`` reads ``t = int(t[0])`` -- ONE timestep per batch, whatever the
          batch size. The README trains at ``--batch_size 1``, where that is exact; at
          any larger batch every sample shares a degradation level, which is upstream's
          simplification and not this adapter's.
        * ``q_sample`` writes ``w.npy`` into the CURRENT WORKING DIRECTORY on every
          call, and ``DiffusionBridge.__init__`` reads it back if present. It is the
          bridge's calibration, so it belongs beside the run rather than wherever the
          process happened to start; :meth:`set_calibration_dir` moves it.

        Args:
            x_0: Clean image, ``[B, 2, H, W]`` real / imaginary, in any units -- it is
                divided by its per-slice peak magnitude first, as ``read_data.py`` does
                per subject.
            batch: Unused -- the bridge conditions on nothing during training.

        Returns:
            The mean of upstream's per-sample loss.
        """
        del batch
        # The bridge's low-pass radius shrinks from `image_size/2`, so a patch of
        # another size would train under a different degradation without an error.
        require_image_size(x_0, self.image_size, type(self).__name__, _IMAGE_SIZE_KEY)
        x_0 = x_0 / peak_magnitude(x_0)
        # `[1, T)`, not `[0, T)`. Upstream binds `img_t_minus_1` only inside
        # `if i == n - int(N/T)`, within a loop of `n = int(t*N/T)` iterations. At t=0
        # that loop does not run and the next line reads the name unbound
        # (`utils/fdb.py:108`); measured, t=1 is the first defined level. Their own
        # `UniformSampler` draws from `[0, T)`, so upstream's trainer can hit it.
        # Excluding the one undefined level is a deviation forced by the defect, not a
        # preference -- the alternative is a run that dies at a rate of 1/T steps.
        t = torch.randint(
            _FDB_MIN_TIMESTEP, self.bridge_steps, (x_0.shape[0],), device=x_0.device
        ).long()
        with self._calibration_cwd():
            # `model_kwargs={}`, not the signature's default of None: upstream does
            # `model(x_t, t, **model_kwargs)` (`utils/fdb.py:314`), and `**None` raises.
            # Its own TrainLoop always passes a dict, so the default is unreachable
            # upstream and every external caller has to supply one.
            terms = self._diffusion.training_losses(self.upstream, x_0, t, model_kwargs={})
        self.calibration_w.copy_(torch.as_tensor(np.asarray(self._diffusion.w, dtype=np.float64)))
        return terms["loss"].mean()

    def calibration(self) -> np.ndarray:
        """The bridge calibration ``w``: the checkpointed buffer, else the live bridge's."""
        w = self.calibration_w.detach().cpu().numpy()
        if not w.any():
            w = np.asarray(self._diffusion.w, dtype=np.float64)
        if not w.any():
            logger.warning(
                "FDB calibration w is all zero: neither the checkpoint nor w.npy carries "
                "one, so the reverse step runs uncalibrated."
            )
        return w

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Accept a checkpoint saved before ``calibration_w`` existed.

        Inference loads strictly, so without this a pre-buffer checkpoint raises on the
        missing key. The buffer is zeroed instead, which :meth:`calibration` reads as
        "not checkpointed" and resolves through ``w.npy`` or its all-zero warning.
        """
        key = f"{prefix}calibration_w"
        absent = key not in state_dict
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if absent:
            self.calibration_w.zero_()
            if key in missing_keys:
                missing_keys.remove(key)
            logger.warning(
                "FDB checkpoint carries no calibration_w (saved before the buffer existed); "
                "the bridge calibration falls back to w.npy."
            )

    def set_calibration_dir(self, directory: Path) -> None:
        """Choose where upstream's ``w.npy`` is written and read.

        A ``w.npy`` already there, from the run being resumed, is loaded the way
        ``DiffusionBridge.__init__`` loads one from its working directory.

        Args:
            directory: An existing directory, normally the run's output directory.
        """
        self._calibration_dir = directory
        if (directory / "w.npy").is_file():
            with self._calibration_cwd():
                self._diffusion = self._new_bridge(self._model_kwargs["diffusion_steps"])

    def _new_bridge(self, steps: int) -> Any:
        """A ``DiffusionBridge`` like the training one, over ``steps`` steps."""
        return type(self._diffusion)(
            steps=steps,
            undersampling_rate=self._model_kwargs["undersampling_rate"],
            image_size=self._model_kwargs["image_size"],
            data_type=self._model_kwargs["data_type"],
        )

    @contextlib.contextmanager
    def _calibration_cwd(self) -> Iterator[None]:
        """Run upstream's ``np.save("w.npy")`` inside the calibration directory."""
        if self._calibration_dir is None:
            yield
            return
        previous = Path.cwd()
        self._calibration_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(self._calibration_dir)
        try:
            yield
        finally:
            os.chdir(previous)

    def sample(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor,
        coil_map: torch.Tensor | None = None,
        *,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Run the FDB bridge sampling loop end-to-end.

        Delegates to ``self._diffusion.p_sample_loop_condition`` (the
        upstream ``DiffusionBridge`` method) so a campaign run can
        invoke FDB just like P-CD — :meth:`forward` is the per-step
        denoise primitive; this method is the full sampler.

        Args:
            kspace: Undersampled k-space, shape
                ``[batch_size, 2, H, W]`` (real/imag stacked).
            mask: Undersampling mask, shape
                ``[batch_size, 2, H, W]`` (paper convention: replicated
                across real/imag).
            coil_map: Coil sensitivity maps (multi-coil only). ``None``
                for single-coil.
            batch_size: Batch size (used to construct the sample shape).

        Returns:
            The final sampled tensor (last element of the progressive
            iterator). Without trained weights this is noise — the
            scientific use case requires
            ``self.upstream.load_state_dict(...)`` first.

        Note:
            The full sampling loop is slow (``bridge_steps`` UNet
            forwards). Test code should pass a small ``image_size`` and
            a low ``bridge_steps`` (e.g. 5) to keep wall-clock manageable.
        """
        require_image_size(kspace, self.image_size, type(self).__name__, _IMAGE_SIZE_KEY)
        return self._run_bridge(self._diffusion, kspace, mask, coil_map, batch_size)

    def _run_bridge(
        self,
        bridge: Any,
        kspace: torch.Tensor,
        mask: torch.Tensor,
        coil_map: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        """``p_sample_loop_condition(...)[-1]`` without holding every step's sample.

        The upstream wrapper appends all ``T`` intermediates to a list before indexing
        the last, about 1.2 GB per slice at 320 x 320 and T = 1500; draining its
        progressive generator computes the same thing and keeps one.
        """
        shape = (batch_size, 2, self.image_size, self.image_size)
        last = collections.deque(
            bridge.p_sample_loop_condition_progressive(
                self.upstream, shape, kspace, mask, coil_map
            ),
            maxlen=1,
        )
        if not last:
            raise RuntimeError("The FDB bridge has zero reverse steps.")
        return last[0]

    def validation_sample(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Reconstruct the input with the authors' reverse bridge, in the input's units.

        ``sample.py`` feeds ``p_sample_loop_condition`` the unnormalised centred FFT of
        the undersampled image and the ``[B, 2, H, W]`` mask, over a bridge rebuilt at
        the test schedule. The same runs here on ``input_batch`` / its peak magnitude,
        undersampled by :meth:`acquisition_mask_tensor`, through the bridge's own
        ``fft2c`` -- an orthonormal transform would rescale the zero-filled start by
        ``1/sqrt(HW)`` against the loop's inverse. ``target_batch`` is only scored.

        ``create_mask`` draws from the global ``np.random`` unseeded upstream; seeding it
        here makes the metric repeatable, and the training RNG is restored afterwards.
        Each call costs ``test_bridge_steps`` UNet passes.
        """
        del target_batch, batch
        data_type = self._model_kwargs["data_type"]
        if data_type != "singlecoil":
            raise NotImplementedError(
                f"{type(self).__name__}.validation_sample runs data_type='singlecoil' only; "
                f"the multicoil bridge takes coil maps and a complex k-space layout that "
                f"this adapter has not been checked against (got {data_type!r})."
            )
        x = _complex_to_real(input_batch)
        require_image_size(x, self.image_size, type(self).__name__, _IMAGE_SIZE_KEY)
        scale = peak_magnitude(x)
        x = x / scale
        bridge = self._test_bridge()
        mask = self.acquisition_mask_tensor(x.device)
        kspace = bridge.fft2c(torch.complex(x[:, 0:1], x[:, 1:2])) * mask
        kspace = torch.cat([kspace.real, kspace.imag], dim=1)
        mask = mask.expand(x.shape[0], 2, -1, -1)
        with seeded_numpy(self.test_mask_seed):
            recon = self._run_bridge(bridge, kspace, mask, None, x.shape[0]) * scale
        return _real_to_complex(recon) if torch.is_complex(input_batch) else recon

    def _test_bridge(self) -> Any:
        """The bridge at the test schedule, carrying the training bridge's calibration.

        ``sample.py`` gets ``w`` by interpolating ``w.npy`` onto the new length in
        ``DiffusionBridge.__init__``; the live array is what that file holds, so it is
        interpolated directly.
        """
        bridge = self._new_bridge(self.test_bridge_steps)
        w = self.calibration()
        bridge.w = np.interp(
            np.linspace(0, 1, self.test_bridge_steps), np.linspace(0, 1, w.shape[0]), w
        )
        bridge.n = np.ones(self.test_bridge_steps)
        return bridge

    def acquisition_mask_tensor(self, device: torch.device) -> torch.Tensor:
        """The test mask, float 0/1 ``[H, W]``, checked against the reverse schedule."""
        if self._test_mask is None:
            size, steps = self.image_size, self.test_bridge_steps
            data_type = self._model_kwargs["data_type"]
            mask = normal_density_mask(
                size,
                size,
                self.test_acceleration,
                sigma=self.test_mask_sigma,
                centre_radius=last_round_radius(size, size, steps, data_type),
                seed=self.test_mask_seed,
            )
            check_bridge_feasible(mask, steps, data_type)
            self._test_mask = mask
        return self._test_mask.to(device)

    def provenance(self) -> dict[str, str]:
        """The base record plus the test-time knobs validation reads."""
        record = super().provenance()
        record["test_acceleration"] = str(self.test_acceleration)
        record["test_bridge_steps"] = str(self.test_bridge_steps)
        record["test_mask"] = (
            f"normal_density sigma={self.test_mask_sigma} seed={self.test_mask_seed}"
        )
        return record


__all__ = ["FDBBaseline"]
