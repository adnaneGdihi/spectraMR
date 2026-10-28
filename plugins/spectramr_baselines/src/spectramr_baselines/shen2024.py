"""Shen 2024 adapter — K-space cold diffusion (Sci. Reports).

Shen G, Hao X, Sheng M, Zhang Y, Du S. "Learning to reconstruct accelerated
MRI through K-space cold diffusion without noise." *Scientific Reports* 14
(2024). https://doi.org/10.1038/s41598-024-72820-2
Upstream: https://github.com/GuoyaoShen/K-SapceColdDIffusion

The forward process degrades by undersampling k-space rather than by adding
Gaussian noise: patches of the acquisition mask's missing region are removed in a
random order, so t=0 is fully sampled and t=T is the accelerated measurement. Training
and validation both run the authors' ``KspaceDiffusion`` (``diffusion/kspace_diffusion.py``)
around their own ``Unet`` and their own ``RandomMaskGaussianDiffusion`` masks, through
:class:`~spectramr_baselines.strategy.UpstreamProcessStrategy`.

This adapter adds what the authors' ``DataTransform_Diffusion``
(``utils/data_transform.py``) did before that object ran: the coil combination
``fastmri.rss`` -- which on one coil takes ``(|Re|, |Im|)`` elementwise, a quirk kept
because the network was trained on it -- and the scale ``1 / max|zero-filled image|``,
which is undone on the way out. The transform's 320 centre crop and resize are not
reproduced: M4Raw slices are 256 x 256 and the arm trains at that size.

The upstream ships no licence file, and ``external/baselines/shen2024`` is pinned as a
submodule; redistributing its code is a decision for the project owner (#2095).
"""

from __future__ import annotations

import functools
import importlib
from typing import Any

import torch

from spectramr.models.baselines._base import (
    BaselineAdapter,
    CoilHandling,
    FFTNorm,
    UpstreamLossFamily,
)
from spectramr.models.registry import register_model
from spectramr_baselines._normalization import (
    peak_magnitude,
    require_image_size,
    to_complex,
    to_real_stack,
)
from spectramr_baselines._upstream_import import upstream_root, vendored_dir

_SHEN_DIR = vendored_dir("shen2024")


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


@register_model(name="shen2024_baseline", training_mode="cold_diffusion", output_domain="image")
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
        kspace, mask, mask_fold, _scale = self._prepare(x_0)
        return self._process()(kspace, mask, mask_fold)

    def _prepare(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``DataTransform_Diffusion`` on a ``[B, 2, H, W]`` image batch.

        Returns the scaled k-space ``[B, 1, H, W, 2]``, the drawn ``mask`` and
        ``mask_fold``, and the per-sample scale ``[B, 1, 1, 1]`` that was applied. Uses
        this repository's FFT SSOT (non-negotiable 2) rather than ``fastmri``'s, which
        is why ``test_shen_kspace_layout_round_trips`` exists: a centering or
        normalisation mismatch between the two would train without raising.
        """
        from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

        require_image_size(x, self.image_size, type(self).__name__, "model.model_kwargs.image_size")
        # The k-space of the magnitude image. ``x.abs()`` on the real/imag stack gave
        # |Re| + i|Im|, whose spectrum is neither the acquisition's nor the magnitude's.
        magnitude = to_complex(x).abs()
        kspace = fft2c(torch.complex(magnitude, torch.zeros_like(magnitude)))
        mask, mask_fold = self._masks(x.shape[0], x.device)
        scale = 1.0 / peak_magnitude(to_real_stack(ifft2c(kspace * mask)))
        kspace = kspace * scale
        return torch.stack([kspace.real, kspace.imag], dim=-1), mask, mask_fold, scale

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

        The inverse of :meth:`_prepare`'s trailing real/imag layout -- upstream's own
        reverse loop returns images, so no FFT is needed here.
        """
        return torch.cat([img[..., 0], img[..., 1]], dim=1)

    def sample(self, x_0: torch.Tensor, *, timesteps: int | None = None) -> torch.Tensor:
        """Full Shen reverse-diffusion sampling loop, in the units of ``x_0``.

        ``KspaceDiffusion.sample`` takes FULLY SAMPLED k-space and degrades it to rung
        ``t`` itself, so ``x_0`` is the fully sampled image and the acquisition mask is
        the authors' own draw -- the same :meth:`_prepare` :meth:`training_loss` uses.

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
        kspace, mask, mask_fold, scale = self._prepare(x_0)
        t = self.timesteps if timesteps is None else timesteps
        _xt, _direct_recons, img = self._process().sample(kspace, mask, mask_fold, t)
        return self._from_upstream_image(img) / scale

    def validation_sample(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Reconstruct ``input_batch``; ``target_batch`` is only scored.

        Upstream's ``sample`` undersamples the k-space it is handed, so the fully sampled
        input goes in and the measurement the network sees is the authors' own mask
        applied to it. Training and sampling both run on the magnitude image, so the
        output's magnitude -- what the metrics score -- is the reconstruction.
        """
        del target_batch, batch
        return self.sample(input_batch)


__all__ = ["Shen2024Baseline"]
