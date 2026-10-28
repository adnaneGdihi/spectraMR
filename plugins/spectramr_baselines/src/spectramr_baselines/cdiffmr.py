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

Training and validation both run the authors' ``GaussianDiffusion``
(``cdm_ksu_m05.py``) through :class:`~spectramr_baselines.strategy.UpstreamProcessStrategy`:
its log-sampling-rate mask ladder, its uniform timestep draw and its ``x0_step_down``
reverse routine. What this adapter adds is only what the upstream's data pipeline did
around that object -- the peak-magnitude normalisation of ``preprocess_normalisation``
and the retrospective acquisition mask its test set applies -- so the network sees the
inputs it was designed for and the output is returned in the target's units.

The vendored upstream is a script-style codebase (no ``__init__.py``),
so the UNet is loaded via :mod:`importlib.util` rather than a plain
``import`` — this avoids polluting ``sys.path`` with the upstream's
top-level modules (``models``, ``utils``).
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from spectramr.models.baselines._base import (
    BaselineAdapter,
    CoilHandling,
    FFTNorm,
    UpstreamLossFamily,
)
from spectramr.models.registry import register_model
from spectramr_baselines._cdiffmr_upstream import (
    _UPSTREAM_MASK_DIR,
    _default_opt,
    _load_upstream_model_class,
    _load_upstream_process_class,
    _upstream_option_dict,
)
from spectramr_baselines._normalization import (
    peak_magnitude,
    phase_encode_mask,
    require_image_size,
    zero_filled,
)

_RESOLUTION_KEY = "model.model_kwargs.resolution"


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


@register_model(name="cdiffmr_baseline", training_mode="cold_diffusion", output_domain="image")
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
        acquisition_mask: str = "random_af4_cf0.08",
        sampling_step: int | None = None,
        **kwargs: object,
    ) -> None:
        """Build the upstream UNet.

        Args:
            acquisition_mask: Stem of an upstream mask file under
                ``mask/fastmri/random/``; ``_pe<resolution>.npy`` is appended. The
                default is the option file's ``fMRI_Ran_AF4_CF0.08``.
            sampling_step: The reverse process's starting rung, upstream's
                ``diffusion.samplng_step`` (sic). ``None`` starts at ``num_steps``, which
                is what the shipped option file does by omitting the key.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.num_steps = num_steps
        if sampling_step is not None and not 1 <= sampling_step <= num_steps:
            raise ValueError(
                f"sampling_step={sampling_step} must lie in [1, num_steps={num_steps}], "
                "or be null to start from num_steps as the upstream option file does."
            )
        self.sampling_step = sampling_step
        self.acquisition_mask = acquisition_mask
        self._mask_path = _UPSTREAM_MASK_DIR / f"{acquisition_mask}_pe{resolution}.npy"
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
        # Upstream's own guard (`network_cdiff_unet2.py`) is a bare `assert` with an
        # empty message; this one names the two config keys that must agree.
        require_image_size(x_real, self.resolution, type(self).__name__, _RESOLUTION_KEY)
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
            x_0: Clean image, ``[B, 2, H, W]`` real / imaginary, in any units: it is
                divided by its peak magnitude first, as ``preprocess_normalisation``
                does to every upstream training image.
            batch: Unused -- upstream's training step conditions on nothing. Its three
                conditioning branches are commented out in the source and ``false`` in
                the option file.

        Returns:
            The scalar L1 between upstream's reconstruction and ``x_0``.
        """
        del batch
        require_image_size(x_0, self.resolution, type(self).__name__, _RESOLUTION_KEY)
        x_0 = x_0 / peak_magnitude(x_0)
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
        require_image_size(x_real, self.resolution, type(self).__name__, _RESOLUTION_KEY)
        t = self.num_steps if timesteps is None else timesteps
        _x_t, _direct_recon, x_recon = self._process().sample(
            x_real, x_obs=x_obs, mask=mask, batch_size=batch_size, t=t
        )
        if torch.is_complex(x_start):
            return _real_to_complex(x_recon, self.out_channels)
        return x_recon

    def acquisition_mask_tensor(self, device: torch.device) -> torch.Tensor:
        """The upstream's own acquisition mask, tiled to ``[1, 1, H, W]``.

        Resolved here rather than in ``__init__`` for the reason :meth:`_process` is
        lazy: a network-only arm never needs a mask, and the upstream ships none below
        pe48.
        """
        import numpy as np

        if not self._mask_path.exists():
            available = sorted(
                p.stem for p in _UPSTREAM_MASK_DIR.glob(f"*_pe{self.resolution}.npy")
            )
            raise FileNotFoundError(
                f"CDiffMR acquisition mask {self._mask_path.name} not found in "
                f"{_UPSTREAM_MASK_DIR}. Available at pe{self.resolution}: {available}."
            )
        mask_1d = torch.from_numpy(np.load(self._mask_path)).to(device=device)
        return phase_encode_mask(mask_1d, self.resolution)

    def provenance(self) -> dict[str, str]:
        """The base record plus the two sampling knobs this adapter reads."""
        record = super().provenance()
        record["acquisition_mask"] = self._mask_path.name
        record["sampling_step"] = str(self.sampling_step or self.num_steps)
        return record

    def validation_sample(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Reconstruct the input as the upstream test script does, in the input's units.

        ``main_test_cdfiimr_ksu_FastMRI_complex.py`` normalises the fully sampled image
        by its peak magnitude, zero-fills it through the acquisition mask and calls
        ``sample(x_start=x_obs, x_obs=x_obs, mask=mask, t=samplng_step)``. The same
        three steps run here on ``input_batch``; ``target_batch`` is only scored, so no
        ground truth reaches the reverse process.
        """
        del target_batch, batch
        scale = peak_magnitude(input_batch)
        mask = self.acquisition_mask_tensor(input_batch.device)
        x_obs = zero_filled(input_batch / scale, mask)
        recon = self.sample(
            x_obs,
            x_obs=x_obs,
            mask=mask.expand(x_obs.shape[0], 1, -1, -1),
            batch_size=x_obs.shape[0],
            timesteps=self.sampling_step,
        )
        return recon * scale


__all__ = ["CDiffMRBaseline"]
