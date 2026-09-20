"""Shen 2024 adapter — K-space cold diffusion (Sci. Reports).

Shen G, Hao X, Sheng M, Zhang Y, Du S. "Learning to reconstruct accelerated
MRI through K-space cold diffusion without noise." *Scientific Reports* 14
(2024). https://doi.org/10.1038/s41598-024-72820-2
Upstream: https://github.com/GuoyaoShen/K-SapceColdDIffusion

**Status: NOT IMPLEMENTED.** :meth:`forward` raises, the class holds zero
parameters, and an arm naming ``shen2024_baseline`` therefore builds a model
that cannot produce an output. That is deliberate (pitfall #9 — never silently
return a wrong answer), but it must be read as *the baseline does not exist
yet*, not as *the baseline is available*.

Two things this docstring said until 2026-09-14, both wrong, and both the
reason the gap survived:

* It described the method as "conditional-diffusion MRI reconstruction with a
  learned k-space prior". The paper is **k-space cold diffusion**: the forward
  process degrades by undersampling k-space rather than by adding Gaussian
  noise, and the network learns the reverse. That is the same family as this
  repo's own ``kspace_cold_diffusion``, which makes it the *most* comparable of
  the three baselines and the one worth wiring first.
* It said the work was "blocked on upstream URL … pending Sci. Reports DAS
  amendment", with three options of which the last was reimplementation. The
  code has been public the whole time; the repository name is misspelled
  upstream (``K-SapceColdDIffusion``), which is why a search for
  "KSpaceColdDiffusion" returned nothing.

**What is actually blocking it**, as of 2026-09-14:

1. ``external/baselines/shen2024/`` exists on this machine as an untracked
   clone at ``35e0dc6`` — it is *not* a submodule, so ``.gitmodules`` does not
   record it and a fresh clone of this repo does not get it. The other two
   baselines are submodules; this one has to become one.
2. The upstream ships **no licence file**. The other two ship ``LICENSE``.
   Vendoring a pointer to an unlicensed repository is a decision for the
   project owner, not a mechanical step, and it is the reason this adapter is
   left raising rather than wired in the same change that corrected this text.

Once (1) and (2) are settled, the upstream has everything the adapter needs:
``net/u_net_diffusion.py`` (the denoiser), ``diffusion/kspace_diffusion.py``
(the forward/reverse process), ``utils/sample_mask.py`` (the degradation
masks), and ``net/{unet,wnet,varnet}/`` (the three reported backbones). Mirror
the shim pattern in :mod:`~spectramr.models.baselines.cdiffmr` — a single-step
``forward`` plus a ``sample()`` that calls the upstream reverse loop — and note
that unlike the other two, this method's forward process is the one this repo
already implements, so a faithful wiring must drive the *upstream's* schedule
rather than reuse ours.
"""

from __future__ import annotations

import functools
import importlib
from pathlib import Path
from typing import Any

import torch

from spectramr.models.baselines._base import (
    BaselineAdapter,
    CoilHandling,
    FFTNorm,
    UpstreamLossFamily,
)
from spectramr.models.baselines._upstream_import import upstream_root
from spectramr.models.registry import register_model

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SHEN_DIR = _REPO_ROOT / "external" / "baselines" / "shen2024"


@functools.cache
def _load_upstream(module_name: str, attribute: str) -> Any:
    """Import one name from the vendored Shen repository.

    The repository is written to run from its own root -- ``kspace_diffusion.py`` does
    ``from help_func import print_var_detail`` -- so the root goes on ``sys.path`` for
    the duration of the import. It also ships a ``utils`` package, which is why this
    goes through :func:`upstream_root` rather than a bare insert: CDiffMR and FDB ship
    one too, and whichever is imported first would answer for all three.
    """
    if not _SHEN_DIR.exists():
        raise FileNotFoundError(
            f"Shen 2024 upstream not found at {_SHEN_DIR}. Clone "
            "https://github.com/GuoyaoShen/K-SapceColdDIffusion there (note the "
            "misspelling in the upstream repository name)."
        )
    with upstream_root(_SHEN_DIR):
        module = importlib.import_module(module_name)
        return getattr(module, attribute)


def _device_of(module: torch.nn.Module) -> torch.device:
    """The device a module's parameters are on, defaulting to CPU when it has none."""
    return next(module.parameters(), torch.empty(0)).device


@register_model(name="shen2024_baseline", training_mode="cold_diffusion")
class Shen2024Baseline(BaselineAdapter):
    """Shen et al. 2024 — conditional diffusion baseline.

    Class attributes:
        REPO_NAME: Vendored directory name.
        PAPER_REF: ``Shen2024:KSpaceColdDiffusion``. It read
            ``Shen2024:CondDiffMRI``, naming a conditional-diffusion method the
            paper is not; ``baseline_provenance`` collects this string for a run
            summary, though nothing calls it in production yet (#2087).
        PREFERRED_MASK_TYPE: ``gaussian_density`` -- the family the PAPER uses.
            It is a declaration, not a selector: no mask code reads it, and the
            arm on this adapter declares a different family (#2087).
        PREFERRED_FFT_NORM: ``ortho``.
        COIL_HANDLING: ``rss`` — paper evaluates on fastMRI single-coil.
    """

    REPO_NAME = "shen2024"
    PAPER_REF = "Shen2024:KSpaceColdDiffusion"
    #: utils/sample_mask.py::RandomMaskGaussian -- a 2D Gaussian density mask.
    PREFERRED_MASK_TYPE = "variable_density_2d_gaussian"
    PREFERRED_FFT_NORM = FFTNorm.ORTHO
    COIL_HANDLING = CoilHandling.RSS
    # `KspaceDiffusion(..., loss_type='l1')` in the notebook; `p_losses` reduces to
    # `(x_start - x_recon).abs().mean()` (`kspace_diffusion.py:218`).
    UPSTREAM_LOSS_FAMILY = UpstreamLossFamily.L1

    def __init__(
        self,
        *,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 64,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        image_size: int = 320,
        timesteps: int = 1000,
        acceleration: int = 4,
        center_fraction: float = 0.08,
        patch_size: int = 4,
        **kwargs: object,
    ) -> None:
        """Build the authors' own ``Unet``.

        Defaults are the notebook's: ``Unet(dim=64, dim_mults=(1,2,4,8), channels=2)``
        at ``img_size = 320``, ``time_steps = 1000``, and
        ``RandomMaskGaussianDiffusion(acceleration=4, center_fraction=0.08)`` with
        ``patch_size=4``.
        """
        super().__init__()
        if in_channels != 2 or out_channels != 2:
            raise ValueError(
                f"Shen 2024 builds `Unet(channels=2)` on single-coil complex data; "
                f"got in_channels={in_channels}, out_channels={out_channels}. The "
                "method handles multiple coils by looping this 2-channel network over "
                "them (kspace_diffusion.py:215), not by widening it."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.timesteps = timesteps
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.patch_size = patch_size
        self._record_unconsumed(kwargs)

        unet_cls = _load_upstream("net.u_net_diffusion", "Unet")
        self.upstream = unet_cls(dim=base_channels, dim_mults=tuple(dim_mults), channels=2)
        self._diffusion: object | None = None
        self._mask_func: object | None = None

    def _process(self) -> object:
        """Upstream's ``KspaceDiffusion``, built once around the authors' ``Unet``."""
        if self._diffusion is None:
            diffusion_cls = _load_upstream("diffusion.kspace_diffusion", "KspaceDiffusion")
            diffusion = diffusion_cls(
                self.upstream,
                image_size=self.image_size,
                device_of_kernel=str(_device_of(self.upstream)),
                channels=2,
                timesteps=self.timesteps,
                loss_type="l1",
                blur_routine="Constant",
                train_routine="Final",
                sampling_routine="x0_step_down",
                discrete=False,
            )
            # `diffusion` wraps `self.upstream` as `denoise_fn`, so a plain
            # attribute assignment would register it as an `nn.Module` child and
            # duplicate every parameter into `state_dict()`, which the strict
            # reload in `checkpoint_service`/`checkpoint_director` then refuses.
            # `object.__setattr__` keeps it off the module graph.
            object.__setattr__(self, "_diffusion", diffusion)
        return self._diffusion

    def _masks(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """The authors' own mask function, called the way their data transform calls it.

        ``RandomMaskGaussianDiffusion.__call__`` takes no arguments and returns
        ``(mask, mask_fold)`` -- the unfolded acquisition mask and its patch-folded
        counterpart, which is the ordering the degradation walks. Building it here
        rather than in the data pipeline keeps the patch geometry with the method that
        defines it; a new draw per batch matches ``data_transform.py:75``, which calls
        it per sample.
        """
        if self._mask_func is None:
            mask_cls = _load_upstream("utils.sample_mask", "RandomMaskGaussianDiffusion")
            self._mask_func = mask_cls(
                acceleration=self.acceleration,
                center_fraction=self.center_fraction,
                size=(1, self.image_size, self.image_size),
                patch_size=self.patch_size,
            )
        masks, folds = [], []
        for _ in range(batch_size):
            mask, mask_fold = self._mask_func()
            masks.append(torch.as_tensor(mask, dtype=torch.float32))
            folds.append(torch.as_tensor(mask_fold, dtype=torch.float32))
        return torch.stack(masks).to(device), torch.stack(folds).to(device)

    def training_loss(
        self,
        x_0: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """``KspaceDiffusion.forward`` -- the authors' degradation, timestep and L1.

        Their ``forward`` takes FULLY SAMPLED k-space ``[B, Nc, H, W, 2]`` and IFFTs it
        internally, so the image is converted here and never pre-degraded. The
        degradation walks the acquisition mask's MISSING patches in random order, so
        t=0 is fully sampled and t=T is exactly the accelerated measurement
        (``kspace_diffusion.py:112``).

        Args:
            x_0: Clean image, ``[B, 2, H, W]`` real / imaginary.
            batch: Unused -- the mask geometry comes from the authors' own mask
                function rather than from this repository's sampler.

        Returns:
            Upstream's scalar L1.
        """
        del batch
        kspace = self._to_upstream_kspace(x_0)
        mask, mask_fold = self._masks(x_0.shape[0], x_0.device)
        return self._process()(kspace, mask, mask_fold)

    def _to_upstream_kspace(self, x_0: torch.Tensor) -> torch.Tensor:
        """``[B, 2, H, W]`` real/imag image -> ``[B, 1, H, W, 2]`` k-space.

        Uses this repository's FFT SSOT (non-negotiable 2) rather than the upstream's,
        which is why ``test_shen_kspace_layout_round_trips`` exists: upstream calls
        ``fastmri.ifft2c`` on whatever it is handed, and a centering or normalisation
        mismatch between the two would train without raising.
        """
        from spectramr.infrastructure.physics.fft_ops import fft2c

        complex_image = torch.complex(x_0[:, 0:1], x_0[:, 1:2])
        k = fft2c(complex_image)
        return torch.stack([k.real, k.imag], dim=-1)

    def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        """One denoiser pass -- the authors' ``Unet(x, time)``.

        Args:
            x: ``[B, 2, H, W]`` real / imaginary.
            *args: An optional positional timestep.
            **kwargs: ``timesteps=`` or ``t=``.

        Returns:
            ``[B, 2, H, W]``.
        """
        t = kwargs.get("timesteps", kwargs.get("t"))
        if t is None and args:
            t = args[0]
        if t is None:
            t = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return self.upstream(x, t)

    def _from_upstream_image(self, img: torch.Tensor) -> torch.Tensor:
        """``[B, 1, H, W, 2]`` image -> ``[B, 2, H, W]`` real/imag.

        The inverse reshape of :meth:`_to_upstream_kspace`'s output layout -- upstream's
        own reverse loop stays in image domain throughout, so no FFT is needed here.
        """
        return torch.cat([img[..., 0], img[..., 1]], dim=1)

    def sample(self, x_0: torch.Tensor, *, timesteps: int | None = None) -> torch.Tensor:
        """Full Shen reverse-diffusion sampling loop, self-masked like :meth:`training_loss`.

        Builds k-space and the authors' own mask/mask_fold pair from ``x_0`` -- the same
        helpers :meth:`training_loss` uses -- then drives ``KspaceDiffusion.sample`` (the
        object :meth:`_process` already builds) through its full reverse loop.

        Raises:
            RuntimeError: upstream hard-codes ``.cuda()`` on its per-step timestep
                tensor (``kspace_diffusion.py``), so the loop cannot run off an
                accelerator.
        """
        if x_0.device.type != "cuda":
            raise RuntimeError(
                f"{type(self).__name__}.sample calls upstream KspaceDiffusion.sample, "
                "which hard-codes `.cuda()` on its per-step timestep tensor "
                f"(kspace_diffusion.py), so it cannot run on {x_0.device}. Run this arm "
                "on an accelerator (non-negotiable 9b)."
            )
        kspace = self._to_upstream_kspace(x_0)
        mask, mask_fold = self._masks(x_0.shape[0], x_0.device)
        t = self.timesteps if timesteps is None else timesteps
        _xt, _direct_recons, img = self._process().sample(kspace, mask, mask_fold, t)
        return self._from_upstream_image(img)

    def validation_sample(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Upstream degrades the clean target itself -- ``input_batch`` is unused.

        Mirrors :meth:`training_loss`: the reverse process starts from
        ``target_batch``, the same convention :meth:`sample` follows.
        """
        del input_batch, batch
        return self.sample(target_batch)


__all__ = ["Shen2024Baseline"]
