r"""Tensor plumbing Equivariant Imaging needs around its objective.

:class:`~spectramr.infrastructure.training.strategies.equivariant_imaging_strategy.EquivariantImagingStrategy`
owns the objective; this module owns the three representation questions the
objective raises, so the strategy reads as the method rather than as layout code.

* **Which image the group acts on.** An image-domain network emits it directly. A
  k-space network (``complex_unet``, the Rep-EI arm) emits every line of every
  coil, and :math:`T_g` must act on the anatomy, not on coil images whose
  sensitivities it would rotate with them, so its output is SENSE-combined first.
* **Which k-space the anchor scores.** A k-space network's output as it stands;
  an image network's image through :math:`FS` (or :math:`F` for one coil).
* **Which layout validation tensors arrive in.** The loop flattens depth into the
  batch for ``input``/``target`` only; the context entries riding the batch are
  brought to the same layout here, with a hard shape check.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from spectramr.core.metrics.coil_manifold import as_interleaved
from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.sense_image_metrics import SENSE_METRIC_NAMES, sense_scoring_pair
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c, sense_forward
from spectramr.infrastructure.physics.group_actions import GroupAction
from spectramr.infrastructure.physics.multicoil_ei_operator import (
    MulticoilEIOperator,
    interleaved_to_complex,
)
from spectramr.infrastructure.training.strategies.mixins.validation_context import (
    flatten_like,
    scoring_maps,
)
from spectramr.infrastructure.training.utils.metric_transform import (
    image_magnitude,
    kspace_rss_magnitude,
    resolve_metric_transform,
)
from spectramr.models.losses.masked_kspace_l2_loss import MaskedKSpaceL2Loss

__all__ = [
    "VALIDATION_ALIGNED_KEYS",
    "aligned_validation_context",
    "alternate_repetition_anchor",
    "anchor_scores_kspace_losses",
    "context_metric_kspace",
    "equivariant_imaging_branches",
    "measured_anchor",
    "model_domains",
    "network_image",
    "network_kspace",
    "scoring_transform",
    "ZF_CONTEXT_METRICS",
    "zero_filled_context_metrics",
    "zero_filled_image",
    "zero_filled_metrics",
]

#: The coil-map columns a multi-coil arm is read on, graded for ``A^H y`` too.
ZF_CONTEXT_METRICS: tuple[str, ...] = ("sense_psnr", "sense_ssim", "reference_noise_corrected_mse")


def equivariant_imaging_branches(
    reconstruct: Callable[[torch.Tensor], torch.Tensor],
    forward_op: Callable[[torch.Tensor], torch.Tensor],
    group: GroupAction,
    measured_kspace: torch.Tensor,
    g_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute the two EI branches for one group element.

    Args:
        reconstruct: maps measured k-space :math:`y` to a recon image
            :math:`f(A^* y)` (the network, including the adjoint/zero-fill).
        forward_op: the complex masked-FFT forward operator :math:`A = M F`.
        group: the symmetry group supplying :math:`T_g`.
        measured_kspace: the undersampled measurement :math:`y` (complex).
        g_index: the group element index (``0`` is the identity).

    Returns:
        ``(x_hat, prediction, transformed_recon)`` where
        ``x_hat`` :math:`= f(A^* y)`,
        ``transformed_recon`` :math:`= T_g \hat{x}` (the reference branch), and
        ``prediction`` :math:`= f(A^* A T_g \hat{x})` (the re-measured branch).
        The EI penalty is :math:`\lVert prediction - transformed\_recon\rVert`.
    """
    x_hat = reconstruct(measured_kspace)  # f(A^* y)
    transformed_recon = group.apply(x_hat, g_index)  # T_g x_hat  (reference)
    re_measured = forward_op(transformed_recon)  # A T_g x_hat  (complex k-space)
    prediction = reconstruct(re_measured)  # f(A^* A T_g x_hat)
    return x_hat, prediction, transformed_recon


#: Context entries read off the validation batch that must share ``input``'s layout.
VALIDATION_ALIGNED_KEYS: tuple[str, ...] = (
    "measured_kspace",
    "mask",
    "coil_sensitivities",
    "loss_mask",
)


def network_image(
    output: torch.Tensor, operator: MulticoilEIOperator | None, emits_kspace: bool
) -> torch.Tensor:
    """The real/imag-interleaved image :math:`T_g` acts on, from the network's output."""
    if not emits_kspace:
        return output if operator is None else operator.restrict(output)
    if operator is not None:
        return operator.combine(output)
    coils = interleaved_to_complex(output)
    if coils.shape[1] != 1:
        raise ValueError(
            f"a k-space network emitting {coils.shape[1]} coils needs "
            "equivariant_imaging.multicoil_operator: true. Without coil maps the group "
            "would act on each coil image, rotating the sensitivities with the "
            "anatomy, and the transformed signal is not in the signal set EI assumes."
        )
    return as_interleaved(ifft2c(coils))


def network_kspace(
    output: torch.Tensor, operator: MulticoilEIOperator | None, emits_kspace: bool
) -> torch.Tensor:
    """What the network predicts as unmasked, real/imag-interleaved k-space."""
    if emits_kspace:
        return output
    image = interleaved_to_complex(output)
    kspace = sense_forward(image, operator.maps, None) if operator is not None else fft2c(image)
    return as_interleaved(kspace)


def context_metric_kspace(
    output: torch.Tensor, context: dict[str, Any], multicoil: bool, emits_kspace: bool
) -> tuple[torch.Tensor, str]:
    r"""``(F S x_hat, "kspace")``: the network's output as the coil k-space Rep-BV grades.

    The repetition metrics compare coil images against a coil-k-space reference,
    and an image network emits one SENSE-combined image; scoring that image
    against the reference's RSS is the comparison no :math:`S\hat x` can win
    (#2509). The batch's maps expand it (:math:`F` alone for one coil); a k-space
    network's output already is coil k-space.
    """
    if emits_kspace:
        return output, "kspace"
    operator = MulticoilEIOperator.resolve(multicoil, context, context.get("mask"))
    return network_kspace(output, operator, emits_kspace), "kspace"


def aligned_validation_context(context: dict[str, Any], like: torch.Tensor) -> dict[str, Any]:
    """``context`` with every batch-carried tensor in ``like``'s flattened layout."""
    aligned = dict(context)
    for key in VALIDATION_ALIGNED_KEYS:
        value = aligned.get(key)
        if isinstance(value, torch.Tensor):
            aligned[key] = flatten_like(value, like, name=key)
    return aligned


def zero_filled_image(measured: torch.Tensor, operator: MulticoilEIOperator | None) -> torch.Tensor:
    r"""``[B, 1, H, W]`` magnitude of :math:`A^Hy`, the image the network is handed.

    The zero-filled baseline answers "did the model beat its own input?", so it is
    the adjoint the strategy applies, graded as an image: :math:`S^HF^HMy` under the
    multi-coil operator, :math:`F^Hy` for one coil (``y`` is already masked).
    """
    if operator is not None:
        return image_magnitude(operator.model_input(measured))
    return image_magnitude(ifft2c(interleaved_to_complex(measured)))


def scoring_transform(config: Any) -> str | None:
    """The validation transform the arm's prediction is scored under, by the mixin's resolver."""
    return resolve_metric_transform(config.validation, config.metrics).name


def zero_filled_metrics(
    computer: Any,
    context: dict[str, Any],
    target: torch.Tensor,
    operator: MulticoilEIOperator | None,
    names: tuple[str, ...],
    *,
    transform: str | None = None,
    batch_data: Any = None,
) -> dict[str, float]:
    """``val_zf_<name>``: :func:`zero_filled_image` against the target, combined as the
    prediction is: under ``sense_magnitude`` both through the scoring maps, else the RSS image.

    Raises:
        ValueError: ``sense_magnitude`` without the multi-coil operator, whose
            :math:`A^Hy` is the one image the SENSE combine can grade.
    """
    measured = context.get("measured_kspace")
    if measured is None or computer is None:
        return {}
    with torch.no_grad():
        if transform == "sense_magnitude":
            if operator is None:
                raise ValueError("sense_magnitude grades A^H y and this arm has no multi-coil operator.")
            image = operator.model_input(measured)
            maps = scoring_maps(batch_data, image)[0] if batch_data is not None else None
            zero_filled, reference = (
                t.abs() for t in sense_scoring_pair(image, target.to(image.device), maps)
            )
        else:
            zero_filled = zero_filled_image(measured, operator)
            reference = kspace_rss_magnitude(target.to(zero_filled.device))
    scored = computer.compute(zero_filled, reference, only=names)
    return {f"val_zf_{name}": value for name, value in scored.items()}


def zero_filled_context_metrics(
    computer: Any,
    context: dict[str, Any],
    target: torch.Tensor,
    operator: MulticoilEIOperator | None,
    batch_data: Any,
) -> dict[str, float]:
    r"""``val_zf_<name>`` for :data:`ZF_CONTEXT_METRICS`, on the model's own footing.

    The RSS ``val_zf_*`` cannot be read against ``sense_psnr`` or the corrected MSE.
    Here :math:`A^Hy` is graded as the network's image is: the SENSE columns on the
    combined image with the scoring maps, the corrected MSE on :math:`FS\,A^Hy`.
    Only the names the arm configures are computed.
    """
    measured = context.get("measured_kspace")
    if measured is None or computer is None or operator is None or batch_data is None:
        return {}
    scored: dict[str, float] = {}
    with torch.no_grad():
        image = operator.model_input(measured)
        target = target.to(image.device)
        sense = tuple(n for n in ZF_CONTEXT_METRICS if n in SENSE_METRIC_NAMES)
        maps = scoring_maps(batch_data, image)[0]
        ctx = MetricContext(coil_maps=maps, signal_domain="image")
        scored.update(computer.compute(image, target, only=sense, context=ctx))
        kspace = network_kspace(image, operator, emits_kspace=False)
        ctx = MetricContext(coil_maps=scoring_maps(batch_data, kspace)[0], signal_domain="kspace")
        rnc = ("reference_noise_corrected_mse",)
        scored.update(computer.compute(kspace, target, only=rnc, context=ctx))
    return {f"val_zf_{name}": value for name, value in scored.items()}


def alternate_repetition_anchor(
    output: torch.Tensor,
    target: torch.Tensor,
    context: dict[str, Any],
    operator: MulticoilEIOperator | None,
    emits_kspace: bool,
    env: Any,
) -> torch.Tensor:
    r"""Rep-EI's measurement term: :math:`\lVert M(\hat k - y^{(2)})\rVert^2`.

    Scored by the arm's declared ``masked_kspace_l2``, the loss the CoNEX
    ``same``-mask arm trains on, so ``alpha_equivariance: 0`` reproduces that arm.
    Resolved from the built losses rather than constructed: an anchor the YAML
    does not declare would be a second, unaudited objective.
    """
    loss_mask = context.get("loss_mask")
    if loss_mask is None:
        raise ValueError(
            "equivariant_imaging.measurement_target: alternate_repetition scores the "
            "lines `loss_mask` selects, and the batch carries none. Declare the "
            "`complementary_rep_masking` transform."
        )
    losses = (getattr(env, "losses", None) or {}) if env is not None else {}
    masked_l2 = next((fn for fn in losses.values() if isinstance(fn, MaskedKSpaceL2Loss)), None)
    if masked_l2 is None:
        raise ValueError(
            "equivariant_imaging.measurement_target: alternate_repetition needs "
            "`masked_kspace_l2` in losses.kspace_losses; none was built."
        )
    predicted = network_kspace(output, operator, emits_kspace)
    return masked_l2(predicted, target, loss_mask=loss_mask)


def anchor_scores_kspace_losses(config: Any, loss_names: list[str]) -> bool:
    r"""Is the declared ``kspace_losses`` list exactly what the EI anchor calls on :math:`FS\hat x`?

    Under ``measurement_target: alternate_repetition`` :func:`alternate_repetition_anchor`
    expands an image network's output through the forward operator and hands it
    to the one declared ``masked_kspace_l2``; under any other target, or for any
    other k-space loss, the strategy calls nothing from the list.
    """
    from spectramr.models.losses.registry import LossRegistry

    ei = getattr(getattr(config, "training", None), "equivariant_imaging", None)
    if getattr(ei, "measurement_target", None) != "alternate_repetition" or len(loss_names) != 1:
        return False
    try:
        loss_cls = LossRegistry.get_loss_class(loss_names[0])
    except Exception:  # an unknown loss is the loss-registry check's finding
        return False
    return isinstance(loss_cls, type) and issubclass(loss_cls, MaskedKSpaceL2Loss)


def measured_anchor(
    predicted: torch.Tensor, measured: torch.Tensor, gsure: Any = None
) -> torch.Tensor:
    r""":math:`\lVert A\hat x - y\rVert^2` averaged, or the GSURE risk under Robust EI."""
    if gsure is not None:
        return gsure(predicted, measured)
    resid = predicted - measured
    return (resid.conj() * resid).real.mean()


def model_domains(config: Any) -> tuple[bool, bool]:
    """``(consumes k-space, emits k-space)`` from the model's registered capabilities."""
    from spectramr.models.registry import get_model_capabilities

    caps = get_model_capabilities(str(config.model.model_type))
    if caps is None:
        return False, False

    def _is_kspace(domain: object) -> bool:
        return "kspace" in (domain if isinstance(domain, tuple) else (domain,))

    return _is_kspace(caps.input_domain), _is_kspace(caps.output_domain)
