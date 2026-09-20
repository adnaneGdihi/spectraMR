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

**This module and the class below previously attributed the method to
"Karaoglu 2024" and expanded FDB as "Frequency-Decomposed Bridge".**
Neither is in the upstream: its README names the eight authors above and
the paper's own title is "Fourier-Constrained". The wrong names reached
``PAPER_REF``, the string ``baseline_provenance`` collects for a run
summary -- so they would travel into any results table built from one the
moment a production path calls it. None does today (#2087).

**What this adapter trains is the UNet, not the bridge.** The arm's
strategy calls :meth:`forward` — one denoise step — under *this repo's*
cold-diffusion forward process. The upstream ``DiffusionBridge`` is
constructed and reachable through :meth:`sample`, but nothing on the
training path invokes it. An arm using this adapter is therefore a
capacity-matched architecture comparison, not a reproduction of the
published method.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from spectramr.models.baselines._base import (
    BaselineAdapter,
    CoilHandling,
    FFTNorm,
    UpstreamLossFamily,
)
from spectramr.models.registry import register_model

# parents[4], not [3]: this file is `src/spectramr/models/baselines/<x>.py`,
# so [3] is `src/` and [4] is the repo root. It was `src/models/baselines/`
# before the 2026-05 `src -> src/spectramr` refactor, when [3] WAS the root.
# The off-by-one made the vendored upstream unreachable and produced an
# error telling the user to run a vendoring command they had already run.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_FDB_DIR = _REPO_ROOT / "external" / "baselines" / "fdb"
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


@register_model(name="fdb_baseline", training_mode="cold_diffusion")
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
        **kwargs: object,
    ) -> None:
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
        self.undersampling_rate = undersampling_rate
        self._calibration_dir: Path | None = None

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
            x_0: Clean image, ``[B, 2, H, W]`` real / imaginary.
            batch: Unused -- the bridge conditions on nothing during training.

        Returns:
            The mean of upstream's per-sample loss.
        """
        del batch
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
        return terms["loss"].mean()

    def set_calibration_dir(self, directory: Path) -> None:
        """Choose where upstream's ``w.npy`` is written and read.

        Args:
            directory: An existing directory, normally the run's output directory.
        """
        self._calibration_dir = directory

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
        shape = (batch_size, 2, self.image_size, self.image_size)
        return self._diffusion.p_sample_loop_condition(
            self.upstream,
            shape,
            kspace,
            mask,
            coil_map,
        )[-1]

    def validation_sample(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Not wired: :meth:`sample` needs the loader's real acquisition mask.

        Unlike CDiffMR/Shen, whose upstream degrades the CLEAN target with its own
        generated mask, FDB's ``create_mask`` (``utils/fdb.py``) reads the actual
        undersampling pattern to build its reverse-timestep mask ladder -- and, for
        ``data_type="multicoil"``, ``p_sample_loop_condition_progressive`` expects the
        k-space in a different tensor convention (complex-typed, not the real/imag
        channel stack this repo's ``input_batch`` carries for singlecoil). Neither is
        available from ``(input_batch, target_batch)`` alone, and guessing either would
        report a plausible but unverified number under this baseline's name -- worse
        than the gap this raise states plainly.
        """
        del target_batch, batch
        data_type = self._model_kwargs.get("data_type", "singlecoil")
        raise NotImplementedError(
            f"{type(self).__name__}.validation_sample is not wired: "
            "DiffusionBridge.p_sample_loop_condition needs the loader's real "
            "undersampling mask (create_mask in utils/fdb.py reads its zero pattern), "
            f"which UpstreamProcessStrategy's validation call does not thread through "
            f"yet, and this arm's data_type={data_type!r} tensor convention has not "
            "been verified against the loader's input_batch. Wire the mask through "
            "`batch_data` and confirm the convention before enabling this."
        )


__all__ = ["FDBBaseline"]
