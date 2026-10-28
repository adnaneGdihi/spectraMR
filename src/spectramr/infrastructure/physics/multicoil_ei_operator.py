r"""The multi-coil forward/adjoint pair Equivariant Imaging inverts.

EI recovers what a **single** operator :math:`A` leaves unidentifiable: it
assumes the signal set is invariant under a group :math:`G` and uses the orbit
:math:`\{AT_g\}` to pin down what :math:`A` alone cannot. Whether that premise
holds at all is decided by the operator, so the choice is not a detail.

Built explicitly (:math:`A=MF` per coil, physical signal set
:math:`\{x:x_n\in\operatorname{span}(s_n)\}`, 16 pixels, a random unit coil
vector per pixel):

===== === =========== ================ ============== =====
coils  R  signal dims rank(A|signal)   unidentifiable cond
===== === =========== ================ ============== =====
1      4  16          4                **12**         1.0
4      4  16          16               **0**          34.7
===== === =========== ================ ============== =====

At :math:`R=4`, four physical coils leave **nothing** unidentifiable -- the
problem is exactly determined and SENSE solves it. One virtual coil leaves 12 of
16 directions open, which is the under-determination an assumed spatial symmetry
is then asked to repair.

**The network is unchanged.** The reconstructed image is the coil-COMBINED one
either way; the coils enter only through :math:`A=MFS` and :math:`A^H=S^HF^HM`.
So a multi-coil arm keeps ``in_channels: 2`` and keeps its group action, and the
only thing that moves is the operator's conditioning.

Both directions route through the physics SSOT (``sense_forward`` /
``sense_adjoint``), which are an adjoint pair to 8.2e-07 --
:math:`\langle Ax,y\rangle=\langle x,A^Hy\rangle` measured on random complex
inputs.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.coil_sensitivity import coil_support
from spectramr.infrastructure.physics.fft_ops import sense_adjoint, sense_forward

__all__ = ["MulticoilEIOperator", "interleaved_to_complex", "require_complex_maps"]


def interleaved_to_complex(x: torch.Tensor) -> torch.Tensor:
    r"""``[B, 2C, H, W]`` real/imag-interleaved -> ``[B, C, H, W]`` complex.

    The network reconstructs a COMPLEX image and emits it interleaved -- that is
    what ``model.in_channels: 2`` means on an EI arm ("single virtual coil,
    real/imag interleaved"). The strategy used to coerce it with
    ``torch.complex(x, zeros_like(x))``, which reads those two channels as two
    SEPARATE images with zero imaginary part. That was wrong twice over:

    * **Physically**, :math:`A = MF` then took the FFT of the real part and the
      FFT of the imaginary part as independent images, so neither the
      measurement-consistency anchor nor the equivariance branch transformed the
      quantity the operator is defined on.
    * **In shape**, it produced ``[B, 2, H, W]`` complex, and the k-space bridge
      in ``_prepare_generator_inputs`` reads dim 1 of a complex 4-D tensor as
      SLICES and folds it into the batch -- so the second EI branch returned
      twice the batch of the first and the loss refused them:
      ``transformed_recon (4, 2, 256, 256) != prediction (8, 2, 256, 256)``.

    Unreachable until the auxiliary-tensor fix let an EI arm take a step, so it
    surfaced on iteration 1 of the cohort's first real run.
    """
    if torch.is_complex(x):
        return x
    channels = x.shape[1]
    if channels % 2 != 0:
        raise ValueError(
            f"Equivariant Imaging needs a real/imag-interleaved reconstruction, "
            f"got {channels} channel(s). An odd count cannot be paired, and "
            "reading it as a real image with zero imaginary part is what made "
            "A = M F transform the real and imaginary parts separately."
        )
    # An autocast generator hands back fp16/bf16, which `torch.complex` turns into the
    # experimental ComplexHalf the FFT operator should not run in.
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    paired = x.view(x.shape[0], channels // 2, 2, *x.shape[-2:])
    return torch.complex(paired[:, :, 0], paired[:, :, 1])


def require_complex_maps(maps: torch.Tensor | None, *, where: str) -> torch.Tensor:
    """Return complex coil maps, or raise saying what is missing and why.

    Never falls back to the coil-combined operator. The knob's whole content is
    *which operator the arm inverts*, so a silent revert would report an EI
    result for a problem the arm's own coils had already determined (pitfall 9).
    """
    if maps is None:
        raise ValueError(
            f"equivariant_imaging.multicoil_operator=true requires complex coil "
            f"sensitivities {where}, and none arrived. Declare the "
            "`espirit_sensitivity` transform under data.processing.transforms, and "
            "serve the coils uncompressed (data.coils.processing_mode: none) so the "
            "maps describe the array the operator uses."
        )
    if not torch.is_complex(maps):
        raise ValueError(
            f"equivariant_imaging.multicoil_operator=true needs COMPLEX coil "
            f"sensitivities {where}, got {maps.dtype}. A magnitude map drops the "
            "coil phase, which is exactly the information that makes the multi-coil "
            "operator better conditioned than the combined one."
        )
    return maps


def _as_complex(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    return mask if mask.is_complex() else mask.to(torch.complex64)


class MulticoilEIOperator:
    r"""The pair :math:`A=MFS` / :math:`A^H=S^HF^HM` an EI arm may invert."""

    def __init__(self, maps: torch.Tensor | None, mask: torch.Tensor | None, *, where: str) -> None:
        self.maps = require_complex_maps(maps, where=where)
        self.mask = _as_complex(mask)

    @classmethod
    def resolve(
        cls, enabled: bool, batch_context: dict, mask: torch.Tensor | None
    ) -> MulticoilEIOperator | None:
        """The operator this arm inverts, or ``None`` for the coil-combined one."""
        if not enabled:
            return None
        return cls(batch_context.get("coil_sensitivities"), mask, where="in the batch")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r""":math:`Ax`: a coil-combined image to per-coil k-space.

        ``x`` is the network's real/imag-interleaved image (or complex). Reading
        the two channels as two real images made ``sense_forward`` see a 2-"coil"
        image against 4 coil maps and raise (#2326), the bug #2227 fixed on the
        combined operator.
        """
        return sense_forward(interleaved_to_complex(x), self.maps, self.mask)

    def adjoint(self, kspace: torch.Tensor) -> torch.Tensor:
        r""":math:`A^Hk=S^HF^HMk` as a complex ``[B, 1, H, W]`` image (the solver form)."""
        return sense_adjoint(kspace, self.maps, self.mask)

    def lipschitz_bound(self) -> float:
        r"""Upper bound on :math:`\lVert A\rVert^2`: :math:`\max_r\sum_c|S_c(r)|^2`.

        :math:`F` is unitary and :math:`0\le M\le 1`, so
        :math:`\lVert MFSx\rVert^2\le\max_r\sum_c|S_c(r)|^2\,\lVert x\rVert^2`. A few
        power iterations approach :math:`L` from below; this never gives a step
        above :math:`1/L`. Assumes a mask with entries in :math:`[0, 1]`.
        """
        return float(self.maps.abs().pow(2).sum(dim=1).amax())

    def restrict(self, image: torch.Tensor) -> torch.Tensor:
        r"""``image`` zeroed where no coil measures it, in the layout it came in.

        :math:`A = MFS` is blind to a pixel with :math:`S = 0`, so an image network
        is free there: the bound's RSS PSNR fell to 0 dB on a growing background,
        and the EI term asked the network to reproduce that background rotated.
        """
        return image * coil_support(self.maps).to(image.real.dtype)

    def combine(self, kspace: torch.Tensor) -> torch.Tensor:
        r""":math:`S^HF^Hk` with no mask: a k-space network's output as the image EI acts on.

        A k-space-native network (``complex_unet``) predicts every line of every
        coil. The group acts on the anatomy, so the prediction is SENSE-combined
        to the one complex image the maps describe before :math:`T_g` touches it.
        """
        combined = sense_adjoint(interleaved_to_complex(kspace), self.maps, None)
        return torch.cat([combined.real, combined.imag], dim=1)

    def model_input(self, kspace: torch.Tensor) -> torch.Tensor:
        r""":math:`A^Hy` as the real 2-channel tensor the network consumes.

        ``sense_adjoint`` SENSE-combines to one complex channel; the network
        reads real/imaginary interleaved, which is what ``in_channels: 2`` means.
        """
        combined = self.adjoint(kspace)
        return torch.cat([combined.real, combined.imag], dim=1)

    def prepare(self, batch_context: dict, input_batch: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """``(model input, context)`` for the caller's generic input builder.

        Both suppression keys are set because the generic builder now has two
        independent reasons to ``ifft2c``: ``use_dc`` (a data-consistency layer
        exists) and the declared k-space -> image bridge. ``S^H F^H M`` has
        already run here, so either one would transform an image.
        """
        kspace = batch_context.get("measured_kspace")
        context = dict(batch_context)
        context["use_dc"] = False
        context["kspace_adjoint_applied"] = True
        return self.model_input(input_batch if kspace is None else kspace), context
