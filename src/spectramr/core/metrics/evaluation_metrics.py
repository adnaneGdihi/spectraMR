"""Core evaluation metrics implementation.

This module serves as the Single Source of Truth (SSOT) for evaluation metrics
used across the codebase, including PSNR, SSIM, FID, and others.

This module deliberately writes NO environment variables. It used to run, at
import time::

    os.environ.setdefault("TMPDIR", "/tmp/<username>")
    if torch.cuda.is_available():
        os.environ["TORCH_CUDA_EAGER_CACHE_MANAGER"] = "1"

All three parts of that were wrong (#1250):

* the path hardcoded one developer's username, and ``/tmp`` is sticky, so on a
  shared cluster node the directory belongs to whoever created it first;
* ``TMPDIR`` is tier 2 of ``env_resolver.resolve_cache_root``, so a metrics
  import silently redirected the framework-wide cache root -- and whether it
  won depended on import order, which no caller controls;
* the assignment sat BELOW this module's own ``import torch``, where PyTorch
  has already read those variables, so it was inert even on its own terms.

``main.py`` owns this bootstrap: it calls ``configure_cache_environment()`` and
sets ``TORCH_CUDA_EAGER_CACHE_MANAGER`` unconditionally ABOVE ``import torch``,
and every entry point reaches it (``cli/app.py`` imports ``spectramr.main``).
Declare ``TMPDIR`` or ``SPECTRAMR_CACHE_ROOT`` in ``.env`` to steer the cache
root; do not re-add a writer here.
"""

import logging
import math
from collections.abc import Iterator
from typing import Any, NoReturn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.config.schemas.enums import Regime
from spectramr.core.metrics.complex_input import as_phase_bearing
from spectramr.core.metrics.data_range_contract import (
    DataRangeContractError,
    QuantileScaleContract,
)
from spectramr.core.metrics.outcome import (
    MetricNotApplicableError,
    NotApplicableReason,
)
from spectramr.core.metrics.registry import register_metric
from spectramr.core.metrics.sample_aggregation import (
    per_sample_flat,
    per_sample_peak,
)
from spectramr.core.types import Image, Mask, Tensor
from spectramr.infrastructure.physics.fft_ops import fft2c

logger = logging.getLogger(__name__)

try:
    import torchvision
    from torchvision.models import Inception_V3_Weights, inception_v3
except ImportError:
    torchvision = None

# Resolved on first use: importing torchmetrics.image drags in matplotlib.pyplot
# and transformers (~3 s), which the model-registry walk must not pay for.
TORCHMETRICS_AVAILABLE: bool | None = None
_TORCHMETRICS_IMPORT_ERROR: ImportError | None = None


def torchmetrics_available() -> bool:
    """Whether torchmetrics imports; probes once, honours an explicit override."""
    global TORCHMETRICS_AVAILABLE, _TORCHMETRICS_IMPORT_ERROR
    if TORCHMETRICS_AVAILABLE is None:
        try:
            import torchmetrics.image  # noqa: F401

            TORCHMETRICS_AVAILABLE = True
        except ImportError as exc:
            TORCHMETRICS_AVAILABLE = False
            _TORCHMETRICS_IMPORT_ERROR = exc
            logger.warning("torchmetrics unavailable: %r", exc)
    return TORCHMETRICS_AVAILABLE


def _require_torchmetrics(metric_name: str) -> NoReturn:
    """Refuse to fabricate a score when torchmetrics cannot be imported.

    The old fallback returned ``torch.tensor(0.0)`` — a value indistinguishable
    from a real perfect LPIPS / MS-SSIM / UQI / KID score. Several arms set
    ``primary_metric: lpips``; a constant 0.0 makes the plateau monitor register
    one spurious "improvement" and then never again, so early stopping fires at
    ``patience`` and checkpoint selection is arbitrary (pitfalls #9/#10/#18).

    Raise instead. In the ``ValidationMetricsComputer`` loop this is caught and
    recorded as ``NaN`` ("not computed") with a warning naming the root cause —
    honest, not fabricated; a direct call (test/script) gets a hard failure.

    NOTE: ``torchmetrics`` is genuinely undeclared in ``pyproject.toml`` and its
    import can also fail on a version conflict (it requires ``huggingface-hub
    <1.0``); either way this fires. See ``docs/review_fixes_2026_07_01.rst``.
    """
    torchmetrics_available()
    raise RuntimeError(
        f"Metric {metric_name!r} requires torchmetrics, which is unavailable "
        f"(TORCHMETRICS_AVAILABLE=False; root cause: {_TORCHMETRICS_IMPORT_ERROR!r}). "
        f"Refusing to report a fabricated 0.0. Fix the environment so torchmetrics "
        f"imports cleanly (it needs huggingface-hub<1.0) via `pip install -e '.[dev]'`, "
        f"or remove {metric_name!r} from validation.scoring.compute."
    )


#: Images per network forward in the feature/perceptual metrics: a folded validation
#: volume is hundreds of slices, and one pass over all of them holds tens of GiB.
METRIC_CHUNK_IMAGES = 16


def _image_chunks(*tensors: Tensor) -> Iterator[tuple[Tensor, ...]]:
    """Yield aligned slices of at most ``METRIC_CHUNK_IMAGES`` images from each tensor."""
    size = METRIC_CHUNK_IMAGES
    for lo in range(0, tensors[0].shape[0], size):
        yield tuple(t[lo : lo + size] for t in tensors)


class BaseMetric(nn.Module):  # type: ignore[misc]
    """Base class for all metrics.

    Class attributes
    ----------------
    INPUT_SIGNATURE : str
        Declares the expected input contract so the dynamic metric loop
        in :class:`EvaluationMetrics` can skip metrics whose input
        contract does not match a standard ``(pred, target)`` image
        pair. Valid values:

        * ``"image_pair"`` (default) — both inputs are 4-D image tensors
          ``(B, C, H, W)``.
        * ``"landmark_pair"`` — both inputs are arrays of landmark
          coordinates ``(K, ndim)``; e.g. TRE.
        * ``"displacement_field"`` — ``preds`` is a dense displacement
          field ``(B, ndim, *spatial)``; ``target`` is ignored or also a
          field; e.g. folding-fraction, DVF-MAE.
        * ``"volume_3d"`` — ``preds`` is at least 3-D spatial; e.g.
          through-plane FWHM.
        * ``"distribution"`` — requires accumulation across many samples;
          e.g. FID, KID, FRD, MedFID, MMD, sliced-Wasserstein.
        * ``"sequence_temporal"`` — requires a time/repetition axis;
          e.g. ICC(3,1), Bland-Altman LoA across replicates.
        * ``"perfusion_curves"``, ``"diffusion_dwi"``, ``"flow_3d"``,
          ``"spectrum"`` — domain-specific signatures.

        The dynamic metric loop in :class:`EvaluationMetrics` only calls
        metrics with ``INPUT_SIGNATURE == "image_pair"`` to avoid
        propagating spurious NaN values into report tables (regression
        of the 2026-05-05 NaN-tables bug).

    REQUIRES_MATCHING_SHAPES : bool
        When ``True`` (default) :meth:`forward` raises on a pred/target
        shape mismatch instead of letting the subclass broadcast. Set it
        ``False`` only on a metric that *deliberately* consumes a wider
        prediction than its target — e.g. a distribution head scored by
        ``gaussian_nll``, which slices ``[mean, logvar]`` itself.
    ALLOWS_UNEQUAL_SAMPLE_COUNT : bool
        Set ``True`` on a distribution-vs-distribution metric to free the
        leading sample axis while keeping every trailing dim checked. This
        is the correct opt-out for unequal cardinality; clearing
        ``REQUIRES_MATCHING_SHAPES`` for that purpose disables the guard
        completely and was how a ``[mean, logvar]`` head silently scored
        against a 1-channel target.
    """

    INPUT_SIGNATURE: str = "image_pair"

    #: Guard against the silent-broadcast class (pitfall #18). ``PSNR`` and ``SSIM``
    #: assert on shape and surface as NaN, but ``NRMSE``/``NMSE``/``MAE``/``MSE``
    #: reduce with plain broadcasting: a ``[B, 2, H, W]`` prediction against a
    #: ``[B, 1, H, W]`` target folds the second channel into the error and returns a
    #: finite, plausible, meaningless number. A NaN gets noticed; a wrong number gets
    #: published. This makes the mismatch fail loudly for every image-pair metric.
    REQUIRES_MATCHING_SHAPES: bool = True

    #: Narrow relaxation for distribution-vs-distribution metrics (Wasserstein, MMD,
    #: sliced-Wasserstein): they compare two point sets whose cardinality legitimately
    #: differs (N degraded vs M clean), so the leading sample axis is free while every
    #: trailing dim still has to match. Use this instead of clearing
    #: ``REQUIRES_MATCHING_SHAPES``, which switches off shape checking *entirely* and
    #: lets a ``[B, 2, H, W]`` head ravel against a ``[B, 1, H, W]`` target.
    ALLOWS_UNEQUAL_SAMPLE_COUNT: bool = False

    def __init__(self, device: str | torch.device = "cpu"):
        """__init__.

        Args:
            device (Union[str, torch.device]): Description.
        """
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.summarize = False  # Default: compute per batch

    def to(self, *args: Any, **kwargs: Any) -> Any:
        """Move the module *and* keep the ``self.device`` label truthful.

        ``self.device`` is not decoration: ``compute_metric`` implementations do
        ``preds = preds.to(self.device)``. Plain ``nn.Module.to()`` moves the
        buffers but leaves the label, so a metric placed on CUDA after
        construction drags its CUDA inputs back to the CPU and computes there --
        the right answer on the wrong device, silently.
        """
        module = super().to(*args, **kwargs)
        device, *_ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            module.device = device
        return module

    def update(self, preds: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> None:
        """Update metric state.

        Subclasses should override this for summary metrics (is_summary=True).
        """
        pass

    def compute(self) -> Any:
        """Compute metric."""
        pass

    def forward(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """Default forward pass validation.

        forward method for BaseMetric.

        Executes PyTorch tensor operations.

        Args:
            preds (Image): Expected input tensor.
            target (Image): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.

        Raises:
            ValueError: when ``REQUIRES_MATCHING_SHAPES`` and pred/target shapes
                disagree — a broadcast here yields a finite, wrong score.
        """
        self._check_shapes(preds, target)
        return self.compute_metric(preds, target, **kwargs)

    def _check_shapes(self, preds: Image, target: Image) -> None:
        """Fail loudly on a pred/target shape mismatch (no silent broadcast)."""
        if not self.REQUIRES_MATCHING_SHAPES:
            return
        if not (isinstance(preds, torch.Tensor) and isinstance(target, torch.Tensor)):
            return
        if preds.shape == target.shape:
            return
        if self.ALLOWS_UNEQUAL_SAMPLE_COUNT:
            # Only the sample axis may differ. Every trailing dim still has to
            # agree, so a [B, 2, H, W] head against a [B, 1, H, W] target stays
            # an error here instead of being ravelled into a plausible number.
            if (
                preds.ndim == target.ndim
                and preds.ndim >= 1
                and preds.shape[1:] == target.shape[1:]
            ):
                return
            raise ValueError(
                f"{type(self).__name__}: prediction shape {tuple(preds.shape)} != "
                f"target shape {tuple(target.shape)}. This metric compares two point "
                "sets, so the leading sample axis may differ — but every trailing "
                f"dim must match, and {tuple(preds.shape[1:])} != "
                f"{tuple(target.shape[1:])}. Flattening these would return a finite "
                "but meaningless distance."
            )
        raise ValueError(
            f"{type(self).__name__}: prediction shape {tuple(preds.shape)} != target "
            f"shape {tuple(target.shape)}. Broadcasting these would return a finite "
            "but meaningless score (e.g. a 2-channel [mean, logvar] head reduced "
            "against a 1-channel target). Slice the prediction to the channel the "
            "metric grades, or set REQUIRES_MATCHING_SHAPES = False on a metric that "
            "deliberately consumes a distribution head."
        )

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """Compute metric for a batch."""
        raise NotImplementedError

    def _to_tensor(self, x: Any) -> Tensor:
        """_to_tensor.

        Args:
            x (Any): Description.
        Returns:
            Tensor: Description.
        """
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.tensor(x, device=self.device)


#: How far past a normalization contract's peak the observed data may sit before
#: the contract is declared unusable. 2.0 == 6 dB: below that, the overshoot an
#: interpolation or an augmentation leaves behind biases PSNR by a few dB; beyond
#: it, the tensor is on a different scale entirely and the reported number stops
#: being the quantity it is labelled as. See :func:`resolve_image_data_range`.
_DATA_RANGE_CONTRACT_TOLERANCE = 2.0


def resolve_image_data_range(
    target: Tensor,
    declared: float | None,
    *,
    metric_name: str,
    contract: QuantileScaleContract | None = None,
) -> float:
    """Resolve ``data_range`` for an image-domain, range-sensitive metric.

    A declared range always wins -- ``metrics.data_range`` is the config surface
    for exactly this, and a user who states the scale is not overridden by a
    guess. Next, a declared percentile scale (``contract``) sets the range to the
    level it pins, see :func:`_range_under_quantile_contract`. With neither, the
    sign of the target selects the normalization contract (``[-1, 1]`` vs
    ``[0, 1]``), which is what this code has always done and is *correct whenever
    the data honours a contract*.

    What is new is that the choice is then **verified against the tensor** instead
    of assumed. The sign test is a proxy for a contract, and on data that honours
    no contract the proxy does not fail -- it silently returns 1.0 and PSNR pegs at
    its clamp floor while SSIM leaves its own codomain (issue #180: the ``*_mno``
    cohort recorded ``train_mse = 458_341`` and SSIM values of -653.8 and -958.3,
    which is not a bad score but an impossible one).

    Deliberately NOT re-derived from ``target.max() - target.min()``. PSNR's
    ``MAX_I`` is the peak of the *representation*, not of one image, so a per-image
    extent makes the metric incomparable across images -- a dark slice spanning
    [0, 0.3] would read ~10 dB better than an identical reconstruction of a bright
    one. It would also silently restate every number the corpus has ever recorded.
    An unresolvable range is therefore reported as *not applicable*, not repaired.

    Args:
        target: Reference tensor the metric grades against.
        declared: Explicitly configured range, or ``None`` to derive one.
        metric_name: Name used in the not-applicable report.
        contract: The percentile scale the data path declares, or ``None``.

    Returns:
        The data range to use.

    Raises:
        MetricNotApplicableError: The target honours no known normalization
            contract, so no range can be derived. Declare ``metrics.data_range``
            or normalize the data.
        DataRangeContractError: ``contract`` is declared and the target
            contradicts it.
    """
    if declared is not None:
        return declared

    if contract is not None and not torch.is_complex(target):
        resolved = _range_under_quantile_contract(target, contract, metric_name=metric_name)
        if resolved is not None:
            return resolved

    # One host transfer, not two: the sign test this replaces already paid one
    # sync for `target.min() < 0`, and the extent check needs the max as well.
    # Stacking keeps that at parity instead of doubling it (non-negotiable #9 --
    # this runs per range-sensitive metric per validation batch).
    lo, hi = torch.stack([target.min(), target.max()]).float().cpu().tolist()
    sign_range = 2.0 if lo < 0.0 else 1.0

    # Both contracts peak at magnitude 1.0; only their floor differs.
    peak = max(abs(lo), abs(hi))
    if peak > _DATA_RANGE_CONTRACT_TOLERANCE:
        raise MetricNotApplicableError(
            metric_name,
            NotApplicableReason.DATA_RANGE_UNRESOLVED,
            f"target spans [{lo:.4g}, {hi:.4g}], which honours neither the [0, 1] "
            f"nor the [-1, 1] normalization contract (peak magnitude {peak:.4g} "
            f"exceeds the tolerated {_DATA_RANGE_CONTRACT_TOLERANCE:g}). A range "
            "cannot be inferred from the sign of unnormalized data. Declare "
            "`metrics.data_range` for this arm, or normalize the pipeline output.",
        )

    return sign_range


def _range_under_quantile_contract(
    target: Tensor,
    contract: QuantileScaleContract,
    *,
    metric_name: str,
) -> float | None:
    """The range a declared percentile scale establishes, verified against ``target``.

    The scale pins a quantile of the magnitude image, not its peak, so the peak
    test above is the wrong check here: it rejected every correctly scaled M4Raw
    batch whose brightest voxels reach ~2.05 (#2241). The peak is still checked,
    against the bound the declared quantile allows
    (:attr:`QuantileScaleContract.max_peak`, where the derivation is).

    A signed target is not the magnitude image the contract describes (a metric
    reading raw interleaved k-space, say), so the contract says nothing about it
    and ``None`` hands it back to the sign test, unchanged from before.
    """
    # One host transfer for both statistics (non-negotiable 9).
    lo, hi = torch.stack([target.min(), target.max()]).float().cpu().tolist()
    if lo < 0.0:
        return None
    if hi > contract.max_peak:
        raise DataRangeContractError(
            f"{metric_name}: the target peaks at {hi:.4g}, but the run declares "
            f"{contract.describe()}, which tolerates a peak of at most {contract.max_peak:.4g}. "
            "The normalization the configuration declares was not applied to this tensor, "
            "or was applied to a different one. Fix the data path; declaring "
            "`data.processing.data_range` would only hide it."
        )
    return contract.anchor


def _guard_ssim_codomain(value: Tensor, *, metric_name: str) -> Tensor:
    """Raise if an SSIM lands outside its own codomain ``[-1, 1]``.

    SSIM is bounded by construction, so a value outside the interval is not a poor
    score -- it is proof that the inputs violated an assumption the formula makes
    (issue #180 recorded -653.8 and -958.3). The number must never reach a CSV as
    if it were data: a reader has no way to tell it apart from a real score, and a
    checkpoint selector will happily rank on it.

    ``resolve_image_data_range`` closes the known cause. This stays as the backstop
    that keeps the *class* of defect from returning through some other route, so it
    raises rather than reporting not-applicable -- an out-of-codomain result means
    something upstream is wrong, which is a crash, not an N/A.
    """
    scalar = float(value)  # single sync; the caller is about to convert anyway
    if not math.isfinite(scalar) or scalar < -1.0 or scalar > 1.0:
        raise ValueError(
            f"{metric_name} returned {scalar:.6g}, outside its codomain "
            "[-1, 1]. SSIM is bounded by construction, so this is an upstream "
            "defect (typically a data_range that does not match the inputs), not "
            "a low score. Refusing to report it as a measurement."
        )
    return value


@register_metric("psnr", aliases=["PSNR", "peak_snr"])
class PSNR(BaseMetric):
    """Peak Signal-to-Noise Ratio (PSNR) Metric — **the canonical PSNR**.

    Graded over the whole image. When a result says "PSNR" with no qualifier, it
    means this. :class:`RobustMRI_PSNR` is a *different quantity*, not a competing
    implementation of this one — see its docstring before trying to reconcile the
    two, because they are supposed to disagree.

    Supports both image-space and k-space domains with domain-aware data range.
    For k-space, uses physics-aware normalization to avoid negative PSNR values.

    .. math::

        PSNR = 20 \\cdot \\log_{10}\\left(\frac{MAX_I}{\\sqrt{MSE}}\right)

    Where:
    - :math:`MAX_I` is the maximum possible pixel value of the image.
    - :math:`MSE` is the Mean Squared Error.
    """

    def __init__(
        self,
        data_range: float | None = None,
        domain: str = "image",
        device: str | torch.device = "cpu",
        use_target_max: bool = False,
    ):
        """Initialize PSNR metric.

        Args:
            data_range: Fixed data range for PSNR. If None, auto-detects from target.
            domain: "image" (default) or "kspace". Affects default data_range handling.
            device: Device for computation.
            use_target_max: If True, uses target.max() as data_range instead of
                max-min. This prevents data_range hallucination when targets have
                varying dynamic ranges. Default False for backward compatibility.
        """
        super().__init__(device)
        self.data_range = data_range
        self.domain = domain
        self.use_target_max = use_target_max
        self.to(self.device)

    def compute_metric(
        self,
        preds: Image,
        target: Image,
        data_range: float | None = None,
        domain: str | None = None,
        scale_contract: QuantileScaleContract | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute PSNR.

        Args:
            preds: Predicted image/k-space
            target: Target image/k-space
            data_range: Override data range for this computation
            domain: Override domain for this computation
            scale_contract: Declared percentile scale, see
                :func:`resolve_image_data_range`. Image domain only.
            **kwargs: Additional arguments (ignored)

        Returns:
            PSNR value in dB (unbounded; see the note below the computation)
        """
        assert preds.shape == target.shape, "Predictions and target must have same shape"

        preds = preds.to(self.device)
        target = target.to(self.device)

        # Determine data range with domain awareness
        declared = data_range if data_range is not None else self.data_range
        current_domain = domain or self.domain

        # PSNR is graded PER SAMPLE and then averaged (issue #1347). ``log`` is
        # concave, so a batch-level MSE reduced once and logged once is not the
        # mean of the per-sample scores -- it makes the published number a
        # function of how the loader happened to group the images (14.3 dB across
        # batch_size 1 -> 24 on a heterogeneous set). See
        # :mod:`spectramr.core.metrics.sample_aggregation` for the axis rule; a
        # tensor with no sample axis is one sample, which is the old formula.
        preds_flat = per_sample_flat(preds)
        target_flat = per_sample_flat(target)

        if declared is not None:
            dr = declared
        elif current_domain.lower() == "kspace":
            # k-space has no [0,1]/[-1,1] contract to verify against -- its scale
            # IS the acquisition's. Use the target's peak magnitude, floored so a
            # near-empty spectrum cannot divide PSNR into nonsense.
            #
            # Per sample, not per batch: the peak of the loudest spectrum in the
            # batch would otherwise set the reference for every other one, which
            # is the same batch-composition dependence the reduction above fixes.
            dr = per_sample_peak(target_flat, floor=0.01, empty_fallback=1.0)
        elif self.use_target_max:
            # Opt-in per-image range. Kept for callers that grade one image at a
            # time and accept the cross-image incomparability it buys -- so it is
            # resolved per image here too, which is what "per-image range" says.
            dr = per_sample_peak(target_flat, floor=0.0, empty_fallback=1.0)
        else:
            # Deliberately NOT per sample. This path resolves a *contract*
            # ([0, 1] vs [-1, 1]) from the sign of the data, not an extent, and a
            # per-sample sign test would read an all-positive sample of a [-1, 1]
            # dataset as [0, 1] and halve its range. Residual, stated rather than
            # hidden: on mixed-sign data the contract can still differ between two
            # batch compositions. ``metrics.data_range`` is the declared escape.
            dr = resolve_image_data_range(
                target, None, metric_name="psnr", contract=scale_contract
            )

        if torch.is_complex(preds) or torch.is_complex(target):
            # ``F.mse_loss`` has no complex kernel ("mse_cpu"/"mse_cuda" not
            # implemented for 'ComplexFloat'), so ``domain="kspace"`` -- the one
            # domain this metric resolves a data_range for from
            # ``torch.abs(target).max()`` -- crashed on the complex tensors it
            # exists to grade.
            #
            # E[|a-b|^2], NOT ``F.mse_loss(view_as_real(a), view_as_real(b))``:
            # the view_as_real form averages over 2x the elements and so returns
            # HALF the squared error, inflating PSNR by 10*log10(2) = 3.01 dB.
            # The magnitude-squared form is the physical energy of the residual
            # and matches what MSE/NMSE/NRMSE in this module already compute.
            mse = (preds_flat - target_flat).abs().pow(2).mean(dim=1)
        else:
            mse = (preds_flat - target_flat).pow(2).mean(dim=1)

        # No `if mse == 0: return 100.0` sentinel. That constant was calibrated to
        # the `clamp(max=100.0)` removed above and did not survive it: an exact
        # match returned 100.0 while a STRICTLY WORSE prediction scored 139.7 dB,
        # so `higher_is_better` selection preferred the imperfect model by ~40 dB.
        # Under the clamp the two merely tied; without it the order inverted.
        #
        # Nothing replaces it, because nothing needs to. The `+ 1e-10` below
        # already regularizes mse == 0, to 20*log10(dr/1e-10) == 200 dB at dr=1 --
        # finite, above every attainable imperfect score, and monotone in mse,
        # which is the only property selection actually requires. The value is set
        # by that epsilon rather than by physics, so read it as "indistinguishable
        # from the reference at this precision", not as a measurement. Dropping the
        # branch also drops a GPU sync (`mse == 0` on a device scalar).

        # Use torch.as_tensor to avoid copy warning
        if isinstance(dr, torch.Tensor):
            dr_tensor = dr.detach().clone().to(self.device)
        else:
            dr_tensor = torch.tensor(dr, device=self.device, dtype=mse.dtype)

        # Compute PSNR with numerical stability. ``mse`` is ``(N,)`` and ``dr`` is
        # either a scalar or ``(N,)``, so this is the per-sample score vector.
        psnr = 20 * torch.log10(dr_tensor / (torch.sqrt(mse) + 1e-10))

        # NOT clamped. A `torch.clamp(psnr, -30.0, 100.0)` used to sit here to
        # "prevent extreme values (e.g. negative PSNR for k-space)" -- but a
        # saturated metric is not a tamed one, it is a destroyed one. Nine
        # `experiment_vf_*` arms monitored a PSNR and recorded
        # `early_stopping_best_value` of exactly -30.0: once every checkpoint
        # scores the bound, "best checkpoint" is decided by tie-break order, so
        # those arms had no model selection at all (issue #179). -47 dB and
        # -112 dB are both terrible and are NOT the same terrible.
        #
        # The k-space case the clamp was defending against is handled at the
        # source instead: `domain="kspace"` resolves its own data_range from the
        # target's peak magnitude, so the ratio never goes wild to begin with.
        #
        # Averaged over the sample axis LAST. With one sample -- every direct
        # per-image caller, and every tensor with no sample axis -- this is a
        # no-op and the returned scalar is the pre-#1347 one.
        return psnr.mean()


@register_metric("robust_mri_psnr", aliases=["RobustPSNR", "ROI_PSNR"])
class RobustMRI_PSNR(BaseMetric):
    """Foreground-restricted PSNR for MRI — a **different quantity** from ``psnr``.

    An MRI slice is mostly black air. Whole-image PSNR therefore spends most of its
    average on background the model gets right for free, which flatters a
    reconstruction whose anatomy is poor. This metric restricts the error to the
    tissue support so the object dominates the score.

    **It is expected to disagree with** :class:`PSNR`, and the disagreement is the
    point — do not "reconcile" them. Measured on a phantom that is 76% background:
    noise injected *only* into the air region drops whole-image ``psnr`` to 18.2 dB
    while this metric stays at its perfect-match sentinel, because the anatomy was
    untouched. Under uniform noise it reads ~2 dB below ``psnr``, tissue-only error
    being the harder average. ``psnr`` is canonical; this is the one to monitor when
    the claim is about anatomical fidelity rather than whole-frame reproduction.

    Addresses three critical issues in MRI PSNR computation:
    1. Background Zero-Inflation: Only computes PSNR on tissue regions
    2. Data Range Hallucination: Uses robust target range (percentile) as data_range
    3. Magnitude Support: Handles complex-valued MRI data correctly

    This metric prevents artificially inflated PSNR scores caused by:
    - Thousands of "correct" zero-predictions in background air regions
    - Single-pixel outliers (DC spikes) that inflate the data range

    References:
        - Knoll et al. "Assessment of the generalization of learned image
          reconstruction" MRM 2019
    """

    def __init__(
        self,
        background_threshold: float = 1e-5,
        data_range_percentile: float = 0.999,
        device: str | torch.device = "cpu",
    ):
        """Initialize Robust MRI PSNR.

        Args:
            background_threshold: Threshold for tissue vs. background.
                Voxels with magnitude > threshold are considered tissue.
                Default 1e-5 works for normalized data [0, 1].
            data_range_percentile: Percentile for dynamic range estimation (0.0-1.0).
                Default 0.999 ignores top 0.1% outliers (like DC spikes).
            device: Device for computation.
        """
        super().__init__(device)
        self.background_threshold = background_threshold
        self.data_range_percentile = data_range_percentile
        self.to(self.device)

    def compute_metric(
        self,
        preds: Image,
        target: Image,
        background_threshold: float | None = None,
        data_range_percentile: float | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute ROI-masked PSNR only within tissue support region.

        Args:
            preds: Predicted image (complex or real)
            target: Target image (complex or real)
            background_threshold: Override default threshold
            data_range_percentile: Override default percentile
            **kwargs: Additional arguments (ignored)

        Returns:
            PSNR value in dB computed only on tissue voxels
        """
        preds = preds.to(self.device)
        target = target.to(self.device)

        # 1. Compute magnitude for both pred and target
        if torch.is_complex(preds):
            pred_mag = torch.abs(preds)
        elif preds.dim() == 4 and preds.shape[1] == 2:
            # [B, 2, H, W] real/imag encoding
            pred_mag = torch.sqrt(preds[:, 0] ** 2 + preds[:, 1] ** 2)
        else:
            pred_mag = torch.abs(preds)

        if torch.is_complex(target):
            target_mag = torch.abs(target)
        elif target.dim() == 4 and target.shape[1] == 2:
            target_mag = torch.sqrt(target[:, 0] ** 2 + target[:, 1] ** 2)
        else:
            target_mag = torch.abs(target)

        # 2. Create tissue mask from target (Otsu-inspired thresholding)
        thresh = (
            background_threshold if background_threshold is not None else self.background_threshold
        )

        # Use target magnitude to determine tissue regions
        # Adaptive threshold: use percentage of max as threshold (e.g., 5% of max signal)
        # Use robust max for thresholding too
        percentile = (
            data_range_percentile
            if data_range_percentile is not None
            else self.data_range_percentile
        )
        if percentile < 1.0:
            robust_max = torch.quantile(target_mag.flatten(), percentile)
        else:
            robust_max = target_mag.max()

        # Every branch below is a device-side `torch.where`, and the tissue mask is
        # applied by masked sums rather than boolean indexing (a nonzero() sync):
        # the result stays on the device so the metrics computer's fused host
        # transfer is the only sync (#2163).
        cutoff = torch.where(robust_max > thresh, robust_max * 0.05, thresh)
        tissue_mask = target_mag > cutoff
        n_tissue = tissue_mask.sum()
        no_tissue = n_tissue == 0

        # 3. MSE on the tissue support; `where`, not `* mask`, so a non-finite
        #    background voxel cannot leak NaN into the tissue average. With no
        #    tissue the score falls back to whole-image PSNR.
        sq_err = (pred_mag - target_mag) ** 2
        mse_tissue = torch.where(tissue_mask, sq_err, 0).sum() / n_tissue.clamp_min(1)
        mse = torch.where(no_tissue, F.mse_loss(pred_mag, target_mag, reduction="mean"), mse_tissue)

        # 4. Data range: robust max of the target magnitude, which ignores DC
        #    spikes. The two branches guard a degenerate maximum differently.
        data_range = torch.where(
            no_tissue,
            torch.where(robust_max > 0, robust_max, 1.0),
            torch.where(robust_max < 1e-10, 1.0, robust_max),
        )

        # 5. Compute PSNR. No mse == 0 sentinel -- see PSNR.compute_metric: the
        #    constant 100.0 belonged to the clamp that was removed, and outliving
        #    it inverted the order between an exact match and a worse prediction.
        #    The epsilon regularizes the exact-match case.
        psnr = 20 * torch.log10(data_range.to(mse.dtype) / (torch.sqrt(mse) + 1e-10))

        # 6. NOT clamped -- see the note in PSNR.compute_metric. This is the
        #    metric the nine saturated `experiment_vf_*` arms actually monitored
        #    (issue #179), so the bound mattered most exactly here.
        return psnr


@register_metric("mse", aliases=["MSE", "mean_squared_error", "l2"])
class MSE(BaseMetric):
    """Mean Squared Error.

    .. math::

        MSE = \\frac{1}{N} \\sum_{i=1}^{N} |x_i - y_i|^2
    """

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)
        return torch.mean(torch.abs(preds - target) ** 2)


@register_metric("mae", aliases=["MAE", "mean_absolute_error", "l1"])
class MAE(BaseMetric):
    """Mean Absolute Error.

    .. math::

        MAE = \\frac{1}{N} \\sum_{i=1}^{N} |x_i - y_i|
    """

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)
        return torch.mean(torch.abs(preds - target))


@register_metric("rmse", aliases=["RMSE"])
class RMSE(BaseMetric):
    """Root Mean Squared Error.

    .. math::

        RMSE = \\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} |x_i - y_i|^2}
    """

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)
        mse = torch.mean(torch.abs(preds - target) ** 2)
        return torch.sqrt(mse)


def gaussian_kernel(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    g_2d = g[:, None] * g[None, :]
    return g_2d.expand(1, 1, window_size, window_size).contiguous()


def compute_ssim_map(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Compute SSIM map.

    AMP-safe. Under an active ``fp16`` autocast the squared intermediates
    (``img1 * img1`` and the windowed sums) overflow the ``float16`` range
    (max ~6.5e4) for any input magnitude above ~256 -> ``Inf - Inf = NaN``.
    Because ``fft_ops`` already isolates its round-trips this way, we mirror it
    here: disable autocast and upcast to ``float32`` for the whole computation,
    so the result matches the (correct) full-precision path and a genuine NaN in
    the model output still propagates (we upcast the *artifact* away, never a
    real signal -- pitfall #9). ``auto_range`` conditioning alone cannot help
    because the overflow precedes the SSIM ratio.

    ``C1`` and ``C2`` are the only stabilisers: both are strictly positive for a
    positive ``data_range``, so the denominator cannot vanish and SSIM(x, x) is
    exactly 1 on a flat region, which covers most of an MRI slice's background.

    Raises:
        ValueError: A scalar ``data_range`` is not positive, which leaves 0/0 on
            every flat window.
    """
    if not isinstance(data_range, torch.Tensor) and data_range <= 0:
        raise ValueError(f"SSIM data_range must be > 0; got {data_range!r}")
    device_type = "cuda" if img1.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        img1 = img1.float()
        img2 = img2.float()
        window = window.float()
        if isinstance(data_range, torch.Tensor):
            data_range = data_range.float()

        padding = window_size // 2
        channels = img1.shape[1]

        mu1 = F.conv2d(img1, window, padding=padding, groups=channels)
        mu2 = F.conv2d(img2, window, padding=padding, groups=channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channels) - mu1_mu2

        sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

        C1 = (k1 * data_range) ** 2
        C2 = (k2 * data_range) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
    return ssim_map


@register_metric("ssim", aliases=["SSIM"])
class SSIMMetric(BaseMetric):
    """Structural Similarity Index (SSIM) Metric.

    .. math::

        SSIM(x, y) = \\frac{(2\\mu_x\\mu_y + C_1)(2\\sigma_{xy} + C_2)}{(\\mu_x^2 + \\mu_y^2 + C_1)(\\sigma_x^2 + \\sigma_y^2 + C_2)}

    Where:
    - :math:`\\mu_x, \\mu_y`: Local means
    - :math:`\\sigma_x^2, \\sigma_y^2`: Local variances
    - :math:`\\sigma_{xy}`: Local covariance
    - :math:`C_1, C_2`: Stabilization constants
    """

    def __init__(
        self,
        window_size: int = 11,
        data_range: float | None = None,  # None means auto-detect
        sigma: float = 1.5,
        device: str | torch.device = "cpu",
    ):
        """__init__.

        Args:
            window_size (int): Description.
            data_range (float): Description.
            sigma (float): Description.
            device (Union[str, torch.device]): Description.
        """
        super().__init__(device)
        self.window_size = window_size
        self.data_range = data_range  # None = auto-detect from data
        self.sigma = sigma
        self.register_buffer("window", gaussian_kernel(window_size, sigma, self.device))
        self.to(self.device)

    def compute_metric(
        self,
        preds: Image,
        target: Image,
        window_size: int | None = None,
        data_range: float | None = None,
        scale_contract: QuantileScaleContract | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute SSIM; ``scale_contract`` as in :func:`resolve_image_data_range`."""
        assert preds.shape == target.shape
        # SSIM's luminance/contrast/structure terms are defined on a real
        # intensity field; there is no accepted complex form. Rather than
        # silently substitute ``.abs()`` -- which would grade a DIFFERENT
        # quantity than the caller passed, and hide a phase-carrying prediction
        # behind a magnitude score -- say so. (Before this the same call died
        # inside ``resolve_image_data_range`` with `"min_all" not implemented
        # for 'ComplexFloat'`, which names torch's kernel table, not the
        # caller's mistake.)
        if torch.is_complex(preds) or torch.is_complex(target):
            raise TypeError(
                "SSIM is undefined for complex tensors; it grades a real "
                "intensity field. Pass magnitudes explicitly "
                "(`preds.abs()`, `target.abs()`) if that is the comparison you "
                "want, or use a complex-aware metric (psnr with "
                "`domain='kspace'`, nmse, kspace_error)."
            )
        preds = preds.to(self.device)
        target = target.to(self.device)

        # Handle 3D conversion if needed or use 2D slice approximation
        if preds.dim() == 5:
            b, c, d, h, w = preds.shape
            preds = preds.view(-1, c, h, w)
            target = target.view(-1, c, h, w)

        ws = window_size if window_size is not None else self.window_size

        dr = resolve_image_data_range(
            target,
            data_range if data_range is not None else self.data_range,
            metric_name="ssim",
            contract=scale_contract,
        )

        # Recreate window if size changed
        if ws != self.window_size:
            window = gaussian_kernel(ws, self.sigma, self.device).to(preds.device)
        else:
            window = self.window.to(preds.device)

        channels = preds.size(1)
        if window.shape[0] != channels:
            window = window.expand(channels, 1, ws, ws)

        ssim_map = compute_ssim_map(preds, target, window, ws, dr)
        return _guard_ssim_codomain(ssim_map.mean(), metric_name="ssim")


@register_metric("clinical_ssim", aliases=["ClinicalSSIM"])
class ClinicalSSIM(SSIMMetric):
    """Clinical SSIM with dynamic range masking."""

    def compute_metric(  # type: ignore[override]
        self,
        preds: Image,
        target: Image,
        mask: Image | None = None,
        window_size: int | None = None,
        data_range: float | None = None,
        scale_contract: QuantileScaleContract | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
            mask (Union[Image, None]): Description.
            window_size (Optional[int]): Description.
            data_range (Optional[float]): Description.
            scale_contract: Declared percentile scale, see :func:`resolve_image_data_range`.
        Returns:
            Tensor: Description.
        """
        if mask is None:
            # Auto-masking: threshold > 0.05 * max
            mask = (target.abs() > 0.05 * target.abs().max()).float()

        # Ensure mask is same shape
        if mask.shape != preds.shape:
            # Basic handling, assuming N, 1, H, W mask for N, C, H, W img
            if mask.shape[1] == 1 and preds.shape[1] > 1:
                mask = mask.repeat(1, preds.shape[1], 1, 1)

        mask = mask.to(self.device)

        preds = preds.to(self.device)
        target = target.to(self.device)

        ws = window_size if window_size is not None else self.window_size
        # Mirror the parent SSIMMetric data_range auto-detect. A default-
        # constructed ClinicalSSIM (and the registry-built clinical_ssim) has
        # self.data_range=None; the previous `data_range or self.data_range`
        # resolution left dr=None, which flowed into compute_ssim_map as
        # ``(k1 * None) ** 2`` and raised TypeError (dead-by-default metric).
        dr = resolve_image_data_range(
            target,
            data_range if data_range is not None else self.data_range,
            metric_name="clinical_ssim",
            contract=scale_contract,
        )

        if ws != self.window_size:
            window = gaussian_kernel(ws, self.sigma, self.device).to(preds.device)
        else:
            window = self.window.to(preds.device)

        channels = preds.size(1)
        if window.shape[0] != channels:
            window = window.expand(channels, 1, ws, ws)

        ssim_map = compute_ssim_map(
            preds,
            target,
            window,
            ws,
            dr,
        )

        # Masked Mean
        masked_ssim = (ssim_map * mask).sum() / (mask.sum() + 1e-8)
        return _guard_ssim_codomain(masked_ssim, metric_name="clinical_ssim")


def apply_log_filter(
    pred: torch.Tensor, target: torch.Tensor, sigma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Layer of Gaussian filter to both inputs."""

    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    ax = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0, device=pred.device)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    sigma_sq = sigma**2
    r_sq = xx**2 + yy**2
    kernel = (
        -1.0
        / (np.pi * sigma_sq**2)
        * (1.0 - r_sq / (2.0 * sigma_sq))
        * torch.exp(-r_sq / (2.0 * sigma_sq))
    )
    kernel = kernel - kernel.mean()
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(pred.dtype)

    padding = kernel_size // 2
    h, w = pred.shape[-2], pred.shape[-1]
    pred_flat = pred.reshape(-1, 1, h, w)
    target_flat = target.reshape(-1, 1, h, w)

    log_pred = F.conv2d(pred_flat, kernel, padding=padding)
    log_target = F.conv2d(target_flat, kernel, padding=padding)

    return log_pred.reshape(*pred.shape), log_target.reshape(*target.shape)


# NOTE: Not @register_metric-decorated. The canonical "hfen" entry lives
# at ``src/core/metrics/hfen.py`` (LoG kernel_size=15, sigma=1.5, L1
# norm). This older 5D-aware L2 variant is kept here for callers that
# want to operate on volumes; complex input goes through the separately
# registered ``complex_hfen`` below. See TODO/audit/13_metrics.md F1.
class HFEN(BaseMetric):
    """High-Frequency Error Norm (HFEN) — legacy L2 implementation.

    Not registered. Use the canonical ``hfen`` registry name (which
    resolves to ``spectramr.core.metrics.hfen.HFENMetric``) for any new code.
    """

    def compute_metric(
        self, preds: Image, target: Image, sigma: float = 1.5, **kwargs: Any
    ) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
            sigma (float): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)

        # Ensure 4D
        if preds.dim() == 5:
            b, c, d, h, w = preds.shape
            preds = preds.reshape(-1, c, h, w)
            target = target.reshape(-1, c, h, w)
        if preds.dim() == 3:
            preds = preds.unsqueeze(0)
            target = target.unsqueeze(0)
        if preds.dim() == 2:
            preds = preds.unsqueeze(0).unsqueeze(0)
            target = target.unsqueeze(0).unsqueeze(0)

        log_pred, log_target = apply_log_filter(preds, target, sigma)

        numerator = torch.norm(log_pred - log_target)
        denominator = torch.norm(log_target) + 1e-10

        return numerator / denominator


@register_metric("complex_hfen", aliases=["ComplexHFEN"], requires_complex=True)
class ComplexHFEN(BaseMetric):
    """Calculates HFEN on complex data to capture phase edge artifacts."""

    def compute_metric(
        self, preds: torch.Tensor, target: torch.Tensor, sigma: float = 1.5, **kwargs: Any
    ) -> torch.Tensor:
        """compute_metric.

        Args:
            preds (torch.Tensor): Description.
            target (torch.Tensor): Description.
            sigma (float): Description.
        Returns:
            torch.Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)

        # Ensure 4D
        if preds.dim() == 3:
            preds = preds.unsqueeze(1)  # [B, 1, H, W]
            target = target.unsqueeze(1)
        preds = as_phase_bearing("complex_hfen", preds, "prediction")
        target = as_phase_bearing("complex_hfen", target, "target")

        # Apply LoG to Real and Imag
        log_pred_real, log_target_real = apply_log_filter(preds.real, target.real, sigma)
        log_pred_imag, log_target_imag = apply_log_filter(preds.imag, target.imag, sigma)

        diff_real = log_pred_real - log_target_real
        diff_imag = log_pred_imag - log_target_imag

        numerator = torch.sqrt(torch.sum(diff_real**2 + diff_imag**2))
        denominator = torch.sqrt(torch.sum(log_target_real**2 + log_target_imag**2)) + 1e-10

        return numerator / denominator


@register_metric("ipen", aliases=["IPEN"], requires_complex=True)
class IPEN(BaseMetric):
    """Image Phase Error Norm (IPEN)."""

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)

        # No Fourier fallback: the phase of a magnitude image is not a phase error.
        preds = as_phase_bearing("ipen", preds, "prediction")
        target = as_phase_bearing("ipen", target, "target")

        phi_pred = torch.angle(preds)
        phi_gt = torch.angle(target)

        # Robust angular difference
        diff = phi_pred - phi_gt
        diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi

        numerator = torch.norm(diff)
        denominator = torch.norm(phi_gt) + 1e-10

        return numerator / denominator


@register_metric("inception_score", aliases=["IS"])
class InceptionScoreMetric(BaseMetric):
    """Inception Score (IS) Metric."""

    def __init__(self, device: str | torch.device = "cpu"):
        """__init__.

        Args:
            device (Union[str, torch.device]): Description.
        """
        super().__init__(device)
        self.generated_features: list[torch.Tensor] = []
        self.summarize = True  # Summary metric
        if torchvision is None:
            logger.warning("torchvision not found, IS will return 0.0")

        if torchvision:
            self.inception_model = inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False
            )
            self.inception_model.eval()
            self.inception_model.to(self.device)
        else:
            self.inception_model = None

    def forward(self, generated: Image, target: Image, **kwargs: Any) -> Any:  # type: ignore[override]
        """Standard IS forward."""
        return self.update(generated, target, **kwargs)

    def update(self, generated: Image, target: Image, **kwargs: Any) -> Any:  # type: ignore[override]
        """Accumulate features during loop."""
        if self.inception_model is None:
            return

        generated = generated.to(self.device)

        # Resize to 299x299
        if generated.shape[2] != 299 or generated.shape[3] != 299:
            generated = F.interpolate(
                generated, size=(299, 299), mode="bilinear", align_corners=False
            )

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        generated = (generated - mean) / std

        with torch.no_grad():
            for (chunk,) in _image_chunks(generated):
                self.generated_features.append(F.softmax(self.inception_model(chunk), dim=1))

    def compute(self) -> float:
        """compute.

        Returns:
            float: Description.
        """
        # Every rank must reach the gather below, so the empty case cannot return early:
        # a rank that skipped it would leave its peers waiting in the collective.
        local = (
            torch.cat(self.generated_features, dim=0).detach().cpu()
            if self.generated_features
            else None
        )
        pieces = self._gather_probs_across_ranks(local)
        if not pieces:
            return 0.0

        all_probs = torch.cat(pieces, dim=0).to(self.device)
        p_y = torch.mean(all_probs, dim=0)
        kl_divs = []
        for probs in all_probs:
            kl_div = torch.sum(probs * (torch.log(probs + 1e-16) - torch.log(p_y + 1e-16)))
            kl_divs.append(kl_div)

        return float(torch.exp(torch.mean(torch.stack(kl_divs))).item())

    @staticmethod
    def _gather_probs_across_ranks(local: torch.Tensor | None) -> list[torch.Tensor]:
        """Every rank's class probabilities, so IS grades the whole validation set.

        torchmetrics' FID and KID synchronise their own state inside ``compute()``; this
        metric keeps a plain list, so each rank would otherwise report the IS of its own
        shard and ranks would disagree on the early-stopping and best-checkpoint decision.
        """
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return [] if local is None else [local]
        gathered: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        return [piece for piece in gathered if piece is not None]

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        # One-shot compute
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        self.generated_features = []
        self.update(preds, target)
        return torch.tensor(self.compute(), device=self.device)


@register_metric("fid", aliases=["FID"])
class FID(BaseMetric):
    """Frechet Inception Distance (FID) Metric.

    Multi-channel inputs are routed through
    :func:`spectramr.core.metrics.channel_adapter.adapt_to_rgb` before being
    fed to InceptionV3, which expects 3-channel RGB. See ``LPIPS`` for
    the same pattern.
    """

    def __init__(
        self,
        feature_layer: int = 2048,
        device: str | torch.device = "cpu",
        channel_mode: str = "auto",
    ):
        """__init__.

        Args:
            feature_layer (int): Description.
            device (Union[str, torch.device]): Description.
            channel_mode: See :class:`spectramr.core.metrics.channel_adapter.ChannelMode`.
        """
        super().__init__(device)
        from spectramr.core.metrics.channel_adapter import ChannelMode

        self.channel_mode = ChannelMode(channel_mode)
        self.feature_layer = feature_layer
        self.summarize = True  # Compute once at the end

        self.use_torchmetrics = torchmetrics_available()
        if self.use_torchmetrics:
            from torchmetrics.image.fid import FrechetInceptionDistance

            self.fid = FrechetInceptionDistance(feature=feature_layer, normalize=True).to(
                self.device
            )
        elif torchvision:
            self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
            self.inception.fc = nn.Identity()
            self.inception.eval()
            self.to(self.device)
        else:
            logger.warning("FID unavailable (no torchvision/torchmetrics)")
            self.fid = None

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """Compute FID."""
        if self.use_torchmetrics and self.fid:
            # Note: We do NOT reset here to allow accumulation if the object persists.
            # However, ValidationMetricsComputer currently recreates the object.
            # We will fix the recreation issue in strategy or computer.

            preds = preds.to(self.device)
            target = target.to(self.device)

            # Finiteness guard (canonical perceptual-metric input contract):
            # FID accumulates features across batches, so a single NaN-bearing
            # batch would silently corrupt the whole score. Raise here, naming
            # the model, before any feature update — pitfall #9. See
            # ``metric_input_prep`` (shared with LPIPS).
            from spectramr.core.metrics.metric_input_prep import (
                assert_finite_metric_input,
            )

            assert_finite_metric_input("fid", preds=preds, target=target)

            # Ensure internal model is on correct device
            if self.fid.device != self.device:
                self.fid.to(self.device)
                logger.debug(f"[FID] Moved internal state to {self.device}")

            # Adapt to (B, 3, H, W) — handles 1, 3, even-C MRI complex,
            # raises on ambiguous odd C != 1, 3.
            from spectramr.core.metrics.channel_adapter import adapt_to_rgb

            preds = adapt_to_rgb(preds, mode=self.channel_mode)
            target = adapt_to_rgb(target, mode=self.channel_mode)

            # Log device once per session
            if not hasattr(self, "_device_logged"):
                logger.info(f"[FID] Computing on device: {self.fid.device}")
                self._device_logged = True

            return self.update(preds, target, **kwargs)

        return torch.tensor(0.0, device=self.device)

    def update(self, preds: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> Any:
        """Accumulate features for FID."""
        if not self.use_torchmetrics or not self.fid:
            return torch.tensor(0.0, device=self.device)

        preds = preds.to(self.device)
        target = target.to(self.device)

        # Channel-adapter handles 1, 3, complex, and even-C multi-coil
        # — replaces silent grayscale-only fallback.
        from spectramr.core.metrics.channel_adapter import adapt_to_rgb

        preds = adapt_to_rgb(preds, mode=self.channel_mode)
        target = adapt_to_rgb(target, mode=self.channel_mode)

        # Update both real and fake feature stacks
        for t_chunk, p_chunk in _image_chunks(target, preds):
            self.fid.update(t_chunk.clamp(0, 1), real=True)
            self.fid.update(p_chunk.clamp(0, 1), real=False)

        return torch.tensor(0.0, device=self.device)

    def compute(self) -> torch.Tensor:
        """Finalize and compute FID."""
        if self.use_torchmetrics and self.fid:
            return self.fid.compute()
        return torch.tensor(0.0, device=self.device)


@register_metric("nmse", aliases=["NMSE"])
class NMSE(BaseMetric):
    """Normalized Mean Squared Error."""

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)
        mse = torch.mean(torch.abs(preds - target) ** 2)
        norm = torch.mean(torch.abs(target) ** 2)
        return mse / (norm + 1e-10)


@register_metric("nrmse", aliases=["NRMSE"])
class NRMSE(BaseMetric):
    """Normalized Root Mean Squared Error."""

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)
        mse = torch.mean(torch.abs(preds - target) ** 2)
        rmse = torch.sqrt(mse)

        # Normalise by the target's actual dynamic range (max - min), like
        # scikit-image's NRMSE, so scores are comparable across datasets with
        # different intensity scales. (The previous assumed-range heuristic of
        # 1.0 / 2.0 silently mis-scaled any data not in [0, 1] or [-1, 1]; the
        # energy-normalised variant is already provided by ``NMSE`` above.)
        mag = target.abs() if torch.is_complex(target) else target
        data_range = mag.max() - mag.min()

        if data_range < 1e-10:
            return rmse
        return rmse / data_range


@register_metric("gradient_error", aliases=["GradientError"])
class GradientError(BaseMetric):
    """Gradient-based Error."""

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)

        if preds.dim() == 5:
            b, c, d, h, w = preds.shape
            preds = preds.view(-1, c, h, w)
            target = target.view(-1, c, h, w)

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=preds.dtype, device=preds.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=preds.dtype, device=preds.device
        ).view(1, 1, 3, 3)

        if preds.size(1) > 1:
            sobel_x = sobel_x.repeat(preds.size(1), 1, 1, 1)
            sobel_y = sobel_y.repeat(preds.size(1), 1, 1, 1)

        g_pred_x = F.conv2d(preds, sobel_x, padding=1, groups=preds.size(1))
        g_pred_y = F.conv2d(preds, sobel_y, padding=1, groups=preds.size(1))
        g_t_x = F.conv2d(target, sobel_x, padding=1, groups=preds.size(1))
        g_t_y = F.conv2d(target, sobel_y, padding=1, groups=preds.size(1))

        return F.l1_loss(
            torch.abs(g_pred_x) + torch.abs(g_pred_y),
            torch.abs(g_t_x) + torch.abs(g_t_y),
        )


@register_metric("kid", aliases=["KID"])
class KID(BaseMetric):
    """Wrapper for Kernel Inception Distance (stateful metric for batch accumulation).

    KID accumulates predictions and targets across multiple validation batches
    and only computes the final distance when finalize() is called.
    """

    # Mark as stateful/summary metric so ValidationMetricsComputer knows
    # to accumulate across batches rather than computing per-batch
    summarize = True

    def __init__(
        self,
        subset_size: int = 8,
        device: str | torch.device = "cpu",
        channel_mode: str = "auto",
    ):
        """__init__.

        Args:
            subset_size (int): Description.
            device (Union[str, torch.device]): Description.
            channel_mode: See :class:`spectramr.core.metrics.channel_adapter.ChannelMode`.
        """
        super().__init__(device)
        from spectramr.core.metrics.channel_adapter import ChannelMode

        # Re-assert summarize=True: BaseMetric.__init__ sets self.summarize=False as an
        # instance attribute, which would shadow this class's class-level True.
        self.summarize = True
        self.subset_size = subset_size
        self.channel_mode = ChannelMode(channel_mode)
        if torchmetrics_available():
            from torchmetrics.image.kid import KernelInceptionDistance

            self.kid = KernelInceptionDistance(subset_size=subset_size).to(self.device)
        else:
            # Construct cleanly (plumbing: subset_size / summarize) but refuse to
            # produce a score — the raise lives at the fabrication point in
            # compute(), not here, so per-batch accumulation stays a no-op.
            self.kid = None

    def update(
        self,
        preds: Image,
        target: Image,
        **kwargs: Any,
    ) -> None:
        """Accumulate predictions and targets for KID computation.

        Called once per validation batch. Data is accumulated internally
        until finalize() is called.
        """
        if self.kid:
            preds = preds.to(self.device)
            target = target.to(self.device)

            # Channel-adapter handles 1, 3, complex, and even-C multi-coil.
            from spectramr.core.metrics.channel_adapter import adapt_to_rgb

            preds = adapt_to_rgb(preds, mode=self.channel_mode)
            target = adapt_to_rgb(target, mode=self.channel_mode)

            if preds.dtype != torch.uint8:
                preds = (preds * 255).clamp(0, 255).byte()
            if target.dtype != torch.uint8:
                target = (target * 255).clamp(0, 255).byte()

            # Accumulate into internal state (DO NOT RESET)
            for t_chunk, p_chunk in _image_chunks(target, preds):
                self.kid.update(t_chunk, real=True)
                self.kid.update(p_chunk, real=False)

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """Accumulate per-batch data. Actual computation happens in compute().

        This is called for each validation batch and just delegates to update().
        The final KID value is computed only once in finalize().

        Returns 0.0 during accumulation (real value comes from finalize()).
        """
        self.update(preds, target, **kwargs)
        return torch.tensor(0.0, device=self.device)

    def compute(self) -> Tensor:
        """Compute the final KID value after accumulating all batches.

        Should only be called once at the end of validation (by finalize()).
        """
        if self.kid:
            try:
                kid_value = self.kid.compute()
                # KernelInceptionDistance.compute() returns a tensor or tuple
                if isinstance(kid_value, (tuple, list)):
                    return kid_value[0]
                return kid_value
            except (ValueError, RuntimeError) as e:
                # Not enough samples or other error - return 0
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to compute KID (likely insufficient samples): {e}"
                )
                return torch.tensor(0.0, device=self.device)
        # self.kid is None → torchmetrics unavailable. Refuse to fabricate a 0.0
        # KID at finalize (review 2026-07-01); see _require_torchmetrics.
        _require_torchmetrics("kid")


@register_metric("ms_ssim", aliases=["MSSSIM"])
class MSSSIM(BaseMetric):
    """Wrapper for MS-SSIM."""

    def __init__(self, device: str | torch.device = "cpu"):
        """Construct the torchmetrics MS-SSIM backbone ONCE and cache it.

        The old code re-instantiated ``MultiScaleStructuralSimilarityIndex-
        Measure`` on every ``compute_metric`` call — a per-validation-step
        allocation of the whole metric module. Build it once here (guarded by
        ``TORCHMETRICS_AVAILABLE``) and reuse ``self._impl`` across calls.
        """
        super().__init__(device)
        self._impl = None
        if torchmetrics_available():
            from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

            self._impl = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)
        if self._impl is not None:
            # 5-D input is scored with a 3-D window and pooled on every spatial axis, so the
            # floor binds all of them; torchmetrics' own check reads only the last two.
            from spectramr.core.metrics.regions.eligibility import min_side_for

            floor = min_side_for("ms_ssim")
            spatial = tuple(preds.shape[-3:] if preds.ndim == 5 else preds.shape[-2:])
            if min(spatial) < floor:
                raise MetricNotApplicableError(
                    "ms_ssim",
                    NotApplicableReason.IMAGE_TOO_SMALL_FOR_METRIC,
                    f"spatial extent {spatial} is below the {floor} px/side floor of five "
                    "dyadic scales with an 11-tap window",
                )
            # ``.to`` on an nn.Module is in-place and returns self, so the
            # cached backbone object identity is preserved across calls.
            impl = self._impl.to(preds.device)
            return impl(preds, target)
        _require_torchmetrics("ms_ssim")


@register_metric("uqi", aliases=["UQI"])
class UQI(BaseMetric):
    """Wrapper for UQI."""

    def __init__(self, device: str | torch.device = "cpu"):
        """Construct the torchmetrics UQI backbone ONCE and cache it.

        Mirrors :class:`MSSSIM`: build the ``UniversalImageQualityIndex``
        module a single time instead of re-allocating it per call.
        """
        super().__init__(device)
        self._impl = None
        if torchmetrics_available():
            from torchmetrics.image import UniversalImageQualityIndex

            self._impl = UniversalImageQualityIndex().to(self.device)

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)
        if self._impl is not None:
            impl = self._impl.to(preds.device)
            return impl(preds, target)
        _require_torchmetrics("uqi")


@register_metric("snr", aliases=["SNR"])
class SNR(BaseMetric):
    """Signal-to-Noise Ratio."""

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)
        noise = preds - target
        # ``x**2`` on a complex tensor is the complex square, not |x|^2, so the
        # powers came out complex and the ``< 1e-10`` guard below raised
        # ("lt_cpu" not implemented for 'ComplexFloat'). Power is |x|^2 in both
        # domains; for real input ``.abs().pow(2)`` is identical to ``x**2``.
        if torch.is_complex(preds) or torch.is_complex(target):
            signal_power = target.abs().pow(2).mean()
            noise_power = noise.abs().pow(2).mean()
        else:
            signal_power = torch.mean(target**2)
            noise_power = torch.mean(noise**2)
        if noise_power < 1e-10:
            return torch.tensor(100.0, device=preds.device)
        return 10 * torch.log10(signal_power / noise_power)


@register_metric("gmsd", aliases=["GMSD"])
class GMSD(BaseMetric):
    """Gradient Magnitude Similarity Deviation (GMSD)."""

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)

        if preds.dim() == 5:
            b, c, d, h, w = preds.shape
            preds = preds.view(-1, c, h, w)
            target = target.view(-1, c, h, w)
        if preds.dim() == 3:
            preds = preds.unsqueeze(0)
            target = target.unsqueeze(0)
        if preds.dim() == 2:
            preds = preds.unsqueeze(0).unsqueeze(0)
            target = target.unsqueeze(0).unsqueeze(0)

        hx = (
            torch.tensor([[1 / 3, 0, -1 / 3]] * 3, device=preds.device, dtype=preds.dtype)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        hy = hx.transpose(2, 3)

        pred_gx = F.conv2d(preds, hx, padding=1)
        pred_gy = F.conv2d(preds, hy, padding=1)
        target_gx = F.conv2d(target, hx, padding=1)
        target_gy = F.conv2d(target, hy, padding=1)

        pred_gm = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-6)
        target_gm = torch.sqrt(target_gx**2 + target_gy**2 + 1e-6)

        c = 0.0026
        gms = (2 * pred_gm * target_gm + c) / (pred_gm**2 + target_gm**2 + c)
        return torch.std(gms)


@register_metric("fsim", aliases=["FSIM"])
class FSIM(BaseMetric):
    """Feature Similarity Index (FSIM)."""

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)

        if preds.dim() == 5:
            b, c, d, h, w = preds.shape
            preds = preds.view(-1, c, h, w)
            target = target.view(-1, c, h, w)
        if preds.dim() == 3:
            preds = preds.unsqueeze(0)
            target = target.unsqueeze(0)
        if preds.dim() == 2:
            preds = preds.unsqueeze(0).unsqueeze(0)
            target = target.unsqueeze(0).unsqueeze(0)

        sobel_x = (
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=preds.dtype,
                device=preds.device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        sobel_y = (
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                dtype=preds.dtype,
                device=preds.device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        grad_pred_x = F.conv2d(preds, sobel_x, padding=1)
        grad_pred_y = F.conv2d(preds, sobel_y, padding=1)
        grad_target_x = F.conv2d(target, sobel_x, padding=1)
        grad_target_y = F.conv2d(target, sobel_y, padding=1)

        gm_pred = torch.sqrt(grad_pred_x**2 + grad_pred_y**2 + 1e-6)
        gm_target = torch.sqrt(grad_target_x**2 + grad_target_y**2 + 1e-6)

        T1 = 0.85
        T2 = 160

        sim_g = (2 * gm_pred * gm_target + T1) / (gm_pred**2 + gm_target**2 + T1)

        pc_pred = torch.abs(preds)
        pc_target = torch.abs(target)
        sim_pc = (2 * pc_pred * pc_target + T2) / (pc_pred**2 + pc_target**2 + T2)

        weights = torch.maximum(pc_pred, pc_target)
        fsim = (sim_g * sim_pc * weights).sum() / (weights.sum() + 1e-10)

        return torch.clamp(fsim, 0, 1)


@register_metric("vif", aliases=["VIF"])
class VIF(BaseMetric):
    """Visual Information Fidelity (VIF) - Multi-scale."""

    def __init__(
        self, num_scales: int = 4, sigma_nsq: float = 2.0, device: str | torch.device = "cpu"
    ) -> None:
        """__init__.

        Args:
            num_scales (int): Description.
            sigma_nsq (float): Description.
            device (Any): Description.
        """
        super().__init__(device)
        self.num_scales = num_scales
        self.sigma_nsq = sigma_nsq

        # Create Gaussian kernel for filtering
        kernel_size = 11
        sigma = 1.5
        ax = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel.unsqueeze(0).unsqueeze(0))
        self.padding = kernel_size // 2

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)

        if preds.dim() == 5:
            b, c, d, h, w = preds.shape
            preds = preds.view(-1, c, h, w)
            target = target.view(-1, c, h, w)
        if preds.dim() == 3:
            preds = preds.unsqueeze(0)
            target = target.unsqueeze(0)

        kernel = self.kernel.to(preds.dtype).to(preds.device)
        num = 0.0
        den = 0.0

        for scale in range(self.num_scales):
            if scale > 0:
                preds = F.avg_pool2d(preds, 2)
                target = F.avg_pool2d(target, 2)

            if preds.shape[-1] < 11 or preds.shape[-2] < 11:
                break

            mu_p = F.conv2d(preds, kernel, padding=self.padding)
            mu_t = F.conv2d(target, kernel, padding=self.padding)

            sigma_p_sq = F.conv2d(preds**2, kernel, padding=self.padding) - mu_p**2
            sigma_t_sq = F.conv2d(target**2, kernel, padding=self.padding) - mu_t**2
            sigma_pt = F.conv2d(preds * target, kernel, padding=self.padding) - mu_p * mu_t

            sigma_p_sq = torch.clamp(sigma_p_sq, min=1e-10)
            sigma_t_sq = torch.clamp(sigma_t_sq, min=1e-10)

            g = sigma_pt / (sigma_t_sq + 1e-10)
            sv_sq = sigma_p_sq - g * sigma_pt
            sv_sq = torch.clamp(sv_sq, min=1e-10)

            num += torch.sum(torch.log10(1 + g**2 * sigma_t_sq / (sv_sq + self.sigma_nsq)))
            den += torch.sum(torch.log10(1 + sigma_t_sq / self.sigma_nsq))

        vif = num / (den + 1e-10)
        return torch.clamp(vif, 0, 1)


@register_metric("lpips", aliases=["LPIPS"])
class LPIPS(BaseMetric):
    """Learned Perceptual Image Patch Similarity.

    Multi-channel inputs are handled via
    :func:`spectramr.core.metrics.channel_adapter.adapt_to_rgb`. The default
    mode is ``ChannelMode.AUTO`` — which collapses even-``C`` (or
    complex) inputs via root-sum-of-squares magnitude before replicating
    to 3 channels. Override via the ``channel_mode`` constructor arg
    when the calling pipeline knows better (e.g. ``GRAYSCALE_MEAN`` for
    multi-contrast stacks).
    """

    # One-time warning guard for the lpips-package fallback (below), so a stale env
    # logs the backend swap once per run instead of once per validation batch.
    _fallback_warned: bool = False

    def __init__(
        self,
        device: str | torch.device = "cpu",
        channel_mode: str = "auto",
    ):
        """__init__.

        Args:
            device (Union[str, torch.device]): Description.
            channel_mode: One of the values in
                :class:`spectramr.core.metrics.channel_adapter.ChannelMode`.
                Default ``"auto"`` is MRI-aware (RSS for even C).
        """
        super().__init__(device)
        from spectramr.core.metrics.channel_adapter import ChannelMode

        self.channel_mode = ChannelMode(channel_mode)
        if torchmetrics_available():
            from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(self.device)
            # Pin to fp32 + eval + no-grad — matches DISTSMetric. The
            # VGG19 inside LPIPS must stay fp32 so stray outer
            # autocast contexts don't silently change leaderboard
            # numerics.
            self.lpips.float()
            self.lpips.eval()
            for p in self.lpips.parameters():
                p.requires_grad_(False)
            self._backend: str | None = "torchmetrics"
        else:
            # torchmetrics unavailable — fall back to the `lpips` package (the reference
            # AlexNet impl already used by the lpips_alex challenge metric) so val_lpips is
            # a REAL number on a stale cluster env, not a run-long NaN. This computes a
            # genuine perceptual distance (not a fabricated 0.0), so it honours the
            # fail-loud intent while degrading gracefully. Only if the `lpips` package is
            # ALSO absent do we defer to the raise in compute_metric. See _require_torchmetrics.
            try:
                import lpips as _lpips_pkg

                net = _lpips_pkg.LPIPS(net="alex")
                net.eval()
                for p in net.parameters():
                    p.requires_grad_(False)
                self.lpips = net.to(self.device)
                self._backend = "lpips_pkg"
                if not LPIPS._fallback_warned:
                    logger.warning(
                        "torchmetrics unavailable (%r); LPIPS metric fell back to the `lpips` "
                        "package (AlexNet). Values are NOT comparable to the torchmetrics VGG "
                        "backbone — re-sync the env (`pip install -e '.[dev]'`) for parity.",
                        _TORCHMETRICS_IMPORT_ERROR,
                    )
                    LPIPS._fallback_warned = True
            except ImportError:
                # Construct cleanly; the raise lives at the fabrication point in
                # compute_metric so callers that inject a backbone (e.g. the
                # finite-guard tests) can still exercise the input contract.
                self.lpips = None
                self._backend = None

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        if not self.lpips:
            # torchmetrics unavailable — refuse to fabricate a 0.0 LPIPS score
            # (review 2026-07-01); see _require_torchmetrics.
            _require_torchmetrics("lpips")

        preds = preds.to(self.device)
        target = target.to(self.device)

        # Finiteness guard (canonical perceptual-metric input contract): a NaN
        # from a diverged model would otherwise pass silently through the
        # static [-1, 1] normaliser's ``clamp`` and surface as a misleading
        # torchmetrics range error. Raise here, naming the model, before any
        # shape handling — pitfall #9. See ``metric_input_prep``.
        from spectramr.core.metrics.metric_input_prep import assert_finite_metric_input

        assert_finite_metric_input("lpips", preds=preds, target=target)

        # Handle various input shapes to ensure 4D [B, C, H, W]
        if preds.dim() == 3:
            # [B, 1, D] or [B, C, D] - needs to be 4D for LPIPS
            # Assume this is temporal or 1D data - expand to 2D
            preds = preds.unsqueeze(-1)  # [B, C, D, 1]
            target = target.unsqueeze(-1)
        elif preds.dim() == 2:
            # [B, D] - add channel and spatial dims
            preds = preds.unsqueeze(1).unsqueeze(-1)  # [B, 1, D, 1]
            target = target.unsqueeze(1).unsqueeze(-1)

        # Now ensure we have 4D tensor [B, C, H, W]
        if preds.dim() != 4:
            # Fallback: if still not 4D, return neutral score
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # LPIPS requires minimum spatial dimensions for downsampling through network
        # VGG-based LPIPS downsamples by 2 five times (32x reduction)
        # Both spatial dimensions should be >= 8 to survive downsampling to at least 1x1
        if preds.size(-2) < 8 or preds.size(-1) < 8:
            # Return neutral score for images too small for perceptual metric
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # Convert to (B, 3, H, W) via the explicit channel adapter — no
        # silent truncation. AUTO uses RSS for even-C MRI complex data.
        from spectramr.core.metrics.channel_adapter import adapt_to_rgb

        preds = adapt_to_rgb(preds, mode=self.channel_mode)
        target = adapt_to_rgb(target, mode=self.channel_mode)

        # Normalize to [-1, 1] statically
        # LPIPS expects [-1, 1]. We assume data is statically normalized.

        def static_normalize(x: torch.Tensor) -> torch.Tensor:
            """static_normalize.

            Args:
                x (Any): Description.
            Returns:
                Any: Description.
            """
            if x.min() >= 0:
                # Assume [0, 1], map to [-1, 1]; clamp so an un-normalised model
                # output (e.g. a field model whose prediction sits in [0, 3]) cannot
                # exceed [-1, 1] and trip the backbone's range check into a NaN
                # (field_velocity_unet / field_guided_score_unet, 2026-07 cluster pull).
                return torch.clamp(x * 2.0 - 1.0, -1.0, 1.0)
            # Assume already [-1, 1]
            return torch.clamp(x, -1.0, 1.0)

        preds = static_normalize(preds)
        target = static_normalize(target)

        # Ensure internal model is on correct device. The torchmetrics Metric needs an
        # explicit move (it is not always carried by the parent nn.Module .to); the
        # lpips-package net is a plain submodule tracked by the parent, and has no
        # ``.device`` attribute, so only sync the torchmetrics backend here.
        if self._backend == "torchmetrics" and self.lpips.device != self.device:
            self.lpips.to(self.device)
            logger.debug(f"[LPIPS] Moved internal state to {self.device}")

        # Log device once per session. Use the metric's own device (the lpips-package
        # net is a plain nn.Module with no ``.device`` attribute).
        if not hasattr(self, "_device_logged"):
            logger.info(f"[LPIPS] Computing on device: {self.device}")
            self._device_logged = True

        try:
            # Defeat any outer autocast — VGG19 inside LPIPS is a
            # fp32-only path. Inputs are cast to fp32 explicitly so a
            # bf16 leak from upstream cannot reach the conv layers.
            with (
                torch.amp.autocast("cuda", enabled=False),
                torch.amp.autocast("cpu", enabled=False),
            ):
                # torchmetrics returns a scalar; the lpips-package net returns per-image
                # [B,1,1,1]. .mean() is identity on a scalar and reduces the per-image case.
                # A folded volume is hundreds of slices; one VGG pass over all of
                # them holds tens of GiB of activations, so score bounded chunks.
                n = preds.shape[0]
                total = preds.new_zeros((), dtype=torch.float32)
                for p_chunk, t_chunk in _image_chunks(preds, target):
                    part = self.lpips(p_chunk.float(), t_chunk.float())
                    total = total + part.mean() * p_chunk.shape[0]
                return total / n
        except RuntimeError as e:
            # Handle cases where output becomes too small after downsampling
            if "Output size is too small" in str(e) or "output_size" in str(e).lower():
                return torch.tensor(0.0, device=self.device, dtype=torch.float32)
            raise


@register_metric("pearson", aliases=["PearsonCorrelation", "correlation"])
class PearsonCorrelation(BaseMetric):
    """Pearson Correlation Coefficient."""

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)

        pred_flat = preds.flatten().float()
        target_flat = target.flatten().float()

        pred_mean = pred_flat.mean()
        target_mean = target_flat.mean()

        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean

        numerator = (pred_centered * target_centered).sum()
        denominator = torch.sqrt((pred_centered**2).sum() * (target_centered**2).sum())

        if denominator < 1e-10:
            return torch.tensor(0.0, device=self.device)

        return numerator / denominator


@register_metric("cosine_similarity", aliases=["CosineSimilarity"])
class CosineSimilarity(BaseMetric):
    """Cosine Similarity."""

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)

        pred_flat = preds.flatten().float()
        target_flat = target.flatten().float()

        numerator = (pred_flat * target_flat).sum()
        denominator = torch.norm(pred_flat) * torch.norm(target_flat)

        if denominator < 1e-10:
            return torch.tensor(0.0, device=self.device)

        return numerator / denominator


@register_metric("kspace_error", aliases=["KSpaceError"])
class KSpaceError(BaseMetric):
    """K-Space Error (Frequency Domain MSE)."""

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)

        if not torch.is_complex(preds):
            pred_k = fft2c(preds)
            target_k = fft2c(target)
        else:
            pred_k = preds
            target_k = target

        return torch.abs(pred_k - target_k).mean()


@register_metric("phase_mse", aliases=["PhaseMSE"], requires_complex=True)
class PhaseMSE(BaseMetric):
    """Phase MSE masked by magnitude."""

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)

        # No Fourier fallback: the phase of a magnitude image is not a phase error.
        pred_c = as_phase_bearing("phase_mse", preds, "prediction")
        target_c = as_phase_bearing("phase_mse", target, "target")

        pred_phase = torch.angle(pred_c)
        target_phase = torch.angle(target_c)

        magnitude_mask = torch.abs(target_c) > 0.05
        diff = torch.angle(torch.exp(1j * (pred_phase - target_phase)))

        # Masked mean by `where` + count rather than boolean indexing and a host
        # branch, so the result stays on the device for the computer's single
        # fused transfer (#2163). An empty mask scores 0.
        n_masked = magnitude_mask.sum()
        masked_sq = torch.where(magnitude_mask, diff**2, 0).sum()
        return torch.where(n_masked == 0, 0.0, masked_sq / n_masked.clamp_min(1))


@register_metric("gradient_entropy", aliases=["GradientEntropy"], requires_reference=False)
class GradientEntropy(BaseMetric):
    """Gradient Entropy."""

    def compute_metric(self, preds: Image, target: Image, bins: int = 256, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
            bins (int): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=preds.dtype, device=preds.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=preds.dtype, device=preds.device
        ).view(1, 1, 3, 3)

        # Robust channel handling: if too many channels, average to avoid blowup
        if preds.size(1) > 8:
            preds = preds.mean(dim=1, keepdim=True)
            target = target.mean(dim=1, keepdim=True)

        channels = preds.size(1)
        if channels > 1:
            sobel_x = sobel_x.repeat(channels, 1, 1, 1)
            sobel_y = sobel_y.repeat(channels, 1, 1, 1)

        grad_x = F.conv2d(preds, sobel_x, padding=1, groups=channels)
        grad_y = F.conv2d(preds, sobel_y, padding=1, groups=channels)

        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        magnitude_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-6)

        hist = torch.histc(magnitude_norm.view(-1), bins=bins, min=0, max=1)
        p = hist / (hist.sum() + 1e-6)
        p = p[p > 0]

        return -torch.sum(p * torch.log2(p + 1e-6))


@register_metric(
    "normalized_gradient_squared",
    aliases=["NGS", "ngs", "norm_grad_sq"],
    requires_reference=False,
)
class NormalizedGradientSquared(BaseMetric):
    """Normalized gradient-squared focus measure (no-reference).

    The lower-cost autofocus counterpart to :class:`GradientEntropy` (McGee et
    al., *JMRI* 2000): the gradient energy normalized by the image energy,

    .. math::

        \\mathrm{NGS} = \\frac{\\sum_i (\\partial_x I)^2 + (\\partial_y I)^2}
                              {\\sum_i I^2 + \\epsilon}.

    The normalization makes it invariant to intensity scaling, so it compares
    sharpness across images of differing brightness. Higher = sharper; motion
    blur / smoothing lowers it. ``target`` is ignored (no-reference).
    """

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        preds = preds.to(self.device)
        if preds.is_complex():
            preds = preds.abs()
        preds = preds.float()
        if preds.size(1) > 1:
            preds = preds.mean(dim=1, keepdim=True)

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=preds.dtype, device=preds.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=preds.dtype, device=preds.device
        ).view(1, 1, 3, 3)

        grad_x = F.conv2d(preds, sobel_x, padding=1)
        grad_y = F.conv2d(preds, sobel_y, padding=1)
        gradient_energy = (grad_x**2 + grad_y**2).sum()
        image_energy = (preds**2).sum() + 1e-8
        return gradient_energy / image_energy


@register_metric("efc", aliases=["EFC"], requires_reference=False)
class EFC(BaseMetric):
    """Entropy Focus Criterion."""

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)

        if preds.is_complex():
            energy = preds.real**2 + preds.imag**2
        else:
            energy = preds**2

        total_energy = energy.sum() + 1e-10
        p = energy / total_energy
        p_masked = p[p > 0]
        entropy = -torch.sum(p_masked * torch.log(p_masked))

        max_entropy = torch.log(torch.tensor(p.numel(), dtype=p.dtype, device=p.device))
        if max_entropy == 0:
            return torch.zeros_like(entropy)

        return entropy / max_entropy


@register_metric("fber", aliases=["FBER"], requires_reference=False)
class FBER(BaseMetric):
    """Foreground-Background Energy Ratio."""

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        mag = torch.abs(preds)

        # Otsu threshold approximation
        x = mag.flatten()
        min_val, max_val = x.min(), x.max()
        if min_val == max_val:
            return torch.tensor(1.0, device=self.device)

        hist = torch.histc(x, bins=256, min=min_val, max=max_val)
        p = hist / x.numel()
        omega = torch.cumsum(p, dim=0)
        mu = torch.cumsum(p * torch.arange(256, device=x.device), dim=0)
        mu_t = mu[-1]
        sigma_b_squared = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-6)
        threshold = min_val + (torch.argmax(sigma_b_squared).float() / 256) * (max_val - min_val)

        bg_mask = mag < threshold
        fg_mask = mag >= threshold

        if bg_mask.sum() == 0 or fg_mask.sum() == 0:
            return torch.tensor(1.0, device=self.device)

        bg_energy = (mag[bg_mask] ** 2).mean()
        fg_energy = (mag[fg_mask] ** 2).mean()

        if bg_energy == 0:
            return torch.tensor(100.0, device=self.device)

        return fg_energy / bg_energy


@register_metric("cnr", aliases=["CNR"])
class CNR(BaseMetric):
    """Contrast-to-Noise Ratio."""

    def compute_metric(
        self,
        preds: Image,
        target: Image,
        mask1: Mask | None = None,
        mask2: Mask | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
            mask1 (Optional[Mask]): Description.
            mask2 (Optional[Mask]): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)

        # Heuristic segmentation if masks not provided
        if mask1 is None:
            # Very simple background/signal split
            thresh = 0.1 * preds.max()
            mask1 = preds < thresh  # BG
            mask2 = preds >= thresh  # Signal

        if mask2 is None:
            raise ValueError("mask2 is required when mask1 is given")
        if mask1.sum() == 0 or mask2.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        mu1 = preds[mask1].mean()
        mu2 = preds[mask2].mean()
        sigma1 = preds[mask1].std()

        if sigma1 < 1e-6:
            sigma1 = 1.0

        return torch.abs(mu1 - mu2) / sigma1


@register_metric("cjv", aliases=["CJV"], workflows=frozenset({Regime.STRUCTURAL}))
class CJV(BaseMetric):
    """Coefficient of Joint Variation between white and grey matter.

    ``(sigma_wm + sigma_gm) / |mu_wm - mu_gm|``. Lower is better: it drops when
    the two tissue classes stay tight and well separated, and rises with noise or
    INU. Tissue classes are approximated by intensity percentile rather than a
    real segmentation.

    Tagged ``mri_structural`` — and only that. CJV presupposes anatomical
    grey/white contrast, so it is meaningless on a parameter map, a velocity
    field or an ADC map.

    Returns ``NaN`` (skip) when the image admits no two-class separation, rather
    than a number. See :meth:`compute_metric`.
    """

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """Coefficient of joint variation, or ``NaN`` if it is not measurable.

        The degenerate branches return ``NaN`` rather than a sentinel, matching
        ``b0_field_rmse``/``adc_mae``. They previously returned ``1.0`` (no
        separable tissue classes) and ``10.0`` (coincident class means), which
        are both perfectly ordinary CJV readings — so the one image that most
        needs flagging, a collapsed or uniform prediction with no tissue
        contrast at all, scored a healthy ``1.0`` and looked fine (pitfall #9).
        ``NaN`` propagates to the reporting layer as "not computed", which is the
        truth.
        """
        preds = preds.to(self.device)
        nan = torch.tensor(float("nan"), device=self.device)
        # Heuristic segmentation for WM/GM
        flat = preds.view(-1)
        thresh = torch.quantile(flat, 0.99)
        wm_mask = preds >= 0.6 * thresh
        gm_mask = (preds >= 0.2 * thresh) & (preds < 0.6 * thresh)

        # .std() of a single element is NaN, so both classes need >= 2 voxels.
        if wm_mask.sum() < 2 or gm_mask.sum() < 2:
            return nan

        mu_wm = preds[wm_mask].mean()
        mu_gm = preds[gm_mask].mean()
        sigma_wm = preds[wm_mask].std()
        sigma_gm = preds[gm_mask].std()

        denom = torch.abs(mu_wm - mu_gm)
        if denom < 1e-6:
            return nan

        return (sigma_wm + sigma_gm) / denom


# Wrappers for simple logic or placeholders
@register_metric("dice", aliases=["Dice"])
class Dice(BaseMetric):
    """Computes the Sørensen-Dice coefficient.

    A standalone implementation of the Dice score, measuring the overlap
    between two thresholded (binary) masks. It is a common metric in
    segmentation tasks.

    Attributes:
        No specific instance attributes.
    """

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        # Simple dice implementation to avoid monai dependency if not needed
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        smooth = 1e-5
        p = (preds > 0.5).float()
        t = (target > 0.5).float()
        intersection = (p * t).sum()
        return (2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)


@register_metric("iou", aliases=["IoU", "Jaccard"])
class IoU(BaseMetric):
    """Computes the Intersection over Union (IoU) or Jaccard Index.

    A standalone metric calculating the area of intersection divided by
    the area of union between binary predictions and targets.

    Attributes:
        No specific instance attributes.
    """

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        smooth = 1e-5
        p = (preds > 0.5).float()
        t = (target > 0.5).float()
        intersection = (p * t).sum()
        union = p.sum() + t.sum() - intersection
        return (intersection + smooth) / (union + smooth)


@register_metric("ghosting_ratio", aliases=["GSR", "GhostingRatio"], requires_reference=False)
class GhostingRatio(BaseMetric):
    """Ghost-to-Signal Ratio."""

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        # Based on _compute_gsr logic
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        if preds.dim() != 4:
            return torch.tensor(0.0, device=self.device)

        thresh = (
            0.1
            * preds.max(dim=1, keepdim=True)[0]
            .max(dim=2, keepdim=True)[0]
            .max(dim=3, keepdim=True)[0]
        )
        signal_mask = preds > thresh

        def get_ratio(axis: int) -> torch.Tensor:
            """get_ratio.

            Args:
                axis (Any): Description.
            Returns:
                Any: Description.
            """
            roll_shift = preds.shape[axis] // 2
            ghost_mask = torch.roll(signal_mask, shifts=roll_shift, dims=axis)
            bg_ghost_mask = ghost_mask & (~signal_mask)

            if signal_mask.sum() == 0 or bg_ghost_mask.sum() == 0:
                return torch.tensor(0.0, device=self.device)

            mean_signal = preds[signal_mask].mean()
            mean_ghost = preds[bg_ghost_mask].mean()
            return mean_ghost / mean_signal

        return torch.max(get_ratio(2), get_ratio(3))


@register_metric("spike_detection", aliases=["SpikeDetection"])
class SpikeDetection(BaseMetric):
    """Detect K-Space Spikes (Global Z-score > 5)."""

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        # Expects K-Space input logic, but here we likely have image.
        # If image, we convert to K-space
        if not torch.is_complex(preds) and preds.shape[1] != 2:
            kspace = fft2c(preds)
            mag = torch.abs(kspace)
        elif torch.is_complex(preds):
            mag = torch.abs(preds)
        else:
            mag = preds  # Fallback

        mean = mag.mean()
        std = mag.std()

        if std < 1e-9:
            return torch.tensor(0.0, device=self.device)

        z_score = (mag - mean) / std
        max_z = z_score.max()

        if max_z > 5.0:
            return max_z
        return torch.tensor(0.0, device=self.device)


@register_metric(
    "zipper_detection",
    aliases=["ZipperDetection", "ZipperArtifactDetector"],
    # ``compute_metric`` reads ``preds`` only — ``target`` is accepted for
    # interface symmetry and never touched. Declaring requires_reference=True
    # was the same spec-vs-registry drift the g_factor fix chased (2026-07-24
    # NR audit): it is caught by
    # ``test_no_reference_specs_agree_with_registry`` the moment the metric is
    # typed NO_REFERENCE in ``metrics_list``, which it now is.
    requires_reference=False,
)
class ZipperDetection(BaseMetric):
    """Detect zipper artifacts via directional high-frequency asymmetry.

    Reference-free. Zipper corruption runs along one axis, so it inflates the
    gradient variance in that direction relative to the other; the score is the
    ratio ``max(var_x, var_y) / min(var_x, var_y)``, which is ``>= 1`` and rises
    with the artifact. Lower is better.
    """

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """Directional gradient-variance ratio of ``preds``.

        Args:
            preds: Image under test.
            target: Unused; accepted so the full-reference call signature works.

        Returns:
            Scalar tensor ``>= 1.0``; ``1.0`` means perfectly isotropic gradients.
        """
        preds = preds.to(self.device)
        # Zipper artifacts appear as high frequency noise in one direction (PE)
        # We look for lines in image domain with high variance compared to neighbors?
        # Or spots in K-Space.
        # Simple placeholder logic: High frequency energy in one direction disproportionately.

        dy = preds[:, :, 1:, :] - preds[:, :, :-1, :]
        dx = preds[:, :, :, 1:] - preds[:, :, :, :-1]

        var_y = dy.var()
        var_x = dx.var()

        # Zipper artifacts cause asymmetric high-frequency energy.
        # A ratio significantly > 1 indicates PE-direction corruption.
        min_var = torch.min(var_x, var_y)
        max_var = torch.max(var_x, var_y)

        if min_var < 1e-10:
            return torch.tensor(0.0, device=self.device)

        # Return directional variance ratio (≥ 1.0; higher = more artifact)
        return max_var / min_var


@register_metric("nr_iqa", aliases=["NRIQA"], requires_reference=False)
class NRIQA(BaseMetric):
    """No-Reference Image Quality Assessment."""

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        # Simple sharpness/contrast measure
        gray = preds.mean(dim=1, keepdim=True)
        grad_x = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        grad_y = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        sharpness = grad_x.abs().mean() + grad_y.abs().mean()
        return sharpness


@register_metric("power_spectrum_consistency", aliases=["PSC", "SpectralConsistency"])
class PowerSpectrumConsistency(BaseMetric):
    """Power Spectrum Consistency Metric.

    Compares radially averaged power spectra between prediction and target.
    This ensures the model generates correct high-frequency texture
    rather than just smooth reconstructions.

    Lower is better (more consistent with target spectrum).
    """

    def __init__(self, num_bins: int = 64, device: str | torch.device = "cpu"):
        """__init__.

        Args:
            num_bins (int): Description.
            device (Union[str, torch.device]): Description.
        """
        super().__init__(device)
        self.num_bins = num_bins

    def radial_profile(self, img: torch.Tensor) -> torch.Tensor:
        """Compute radially averaged power spectrum.

        Args:
            img: [B, C, H, W] image tensor

        Returns:
            [B, num_bins] radial power profile
        """
        B = img.shape[0]
        C = img.shape[1]
        H, W = img.shape[-2], img.shape[-1]

        # FFT and power spectrum - use physics module for consistency
        if not torch.is_complex(img):
            kspace = fft2c(img)  # Already imported from spectramr.infrastructure.physics.fft_ops
        else:
            kspace = img

        power = torch.abs(kspace) ** 2
        power = power.mean(dim=1)  # Average over channels [B, H, W]

        # Create radial coordinate grid
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=img.device) - cy
        x = torch.arange(W, device=img.device) - cx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        r = torch.sqrt(xx**2 + yy**2).float()

        # ``fft2c`` already returns centered k-space (DC at H/2, W/2),
        # which matches the ``cy/cx`` of the radial-bin grid above.
        # An extra ``fftshift`` here would un-center ``power`` and
        # misalign the bins — CLAUDE.md pitfall #2. See
        # TODO/backlog_ssot_and_layering_cleanup.md Phase 3.

        # Bin radial distances
        max_r = min(cy, cx)
        bin_edges = torch.linspace(0, max_r, self.num_bins + 1, device=img.device)

        # Bin i holds edge[i] <= r < edge[i+1]; pixels at or past max_r land in a spill bin
        # that is dropped. One scatter replaces the per-bin, per-sample mask loop.
        nb = self.num_bins
        idx = (torch.bucketize(r, bin_edges, right=True) - 1).clamp(max=nb).flatten()
        sums = torch.zeros(B, nb + 1, device=img.device, dtype=power.dtype)
        sums.scatter_add_(1, idx.expand(B, -1), power.reshape(B, -1))
        counts = torch.bincount(idx, minlength=nb + 1).clamp_min(1).to(power.dtype)
        profile = (sums / counts)[:, :nb].float()

        return profile

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """compute_metric.

        Args:
            preds (Image): Description.
            target (Image): Description.
        Returns:
            Tensor: Description.
        """
        preds = preds.to(self.device)
        target = target.to(self.device)

        # Compute radial profiles
        pred_profile = self.radial_profile(preds)
        target_profile = self.radial_profile(target)

        # Log-scale comparison (PSC often uses log power)
        pred_log = torch.log(pred_profile + 1e-10)
        target_log = torch.log(target_profile + 1e-10)

        # L1 distance in log-space
        psc = (pred_log - target_log).abs().mean()

        return psc


@register_metric("med_fid", aliases=["MedFID", "RadImageNetFID"])
class MedFID(BaseMetric):
    """Medical FID using RadImageNet features (placeholder).

    Standard FID uses ImageNet-pretrained InceptionV3, which is
    suboptimal for medical images. MedFID uses RadImageNet weights.

    Note: Requires RadImageNet weights to be loaded. Falls back to
    standard FID if weights not available.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        use_radimagenet: bool = True,
        device: str | torch.device = "cpu",
    ):
        """__init__.

        Args:
            feature_dim (int): Description.
            use_radimagenet (bool): Description.
            device (Union[str, torch.device]): Description.
        """
        super().__init__(device)
        self.feature_dim = feature_dim
        self.use_radimagenet = use_radimagenet
        self.summarize = True  # Summary-based

        # Accumulate features for batch computation
        self.pred_features: list[torch.Tensor] = []
        self.target_features: list[torch.Tensor] = []

        # Try to load feature extractor
        self._setup_extractor()

    def _setup_extractor(self) -> None:
        """Setup feature extractor (RadImageNet or fallback)."""
        # Placeholder: Use standard InceptionV3 with note about RadImageNet
        if torchvision is not None:
            try:
                self.model = inception_v3(
                    weights=Inception_V3_Weights.DEFAULT,
                    transform_input=False,
                )
                self.model.fc = nn.Identity()  # Remove classification head
                self.model.eval()
                self.model.to(self.device)
                self._model_available = True
            except Exception:
                self._model_available = False
        else:
            self._model_available = False

    def extract_features(self, img: torch.Tensor) -> torch.Tensor:
        """Extract features from image."""
        if not self._model_available:
            # Fallback: use simple CNN features
            img = F.adaptive_avg_pool2d(img, (7, 7))
            return img.view(img.shape[0], -1)

        features = []
        with torch.no_grad():
            for (chunk,) in _image_chunks(img):
                chunk = F.interpolate(chunk, size=(299, 299), mode="bilinear", align_corners=False)
                if chunk.shape[1] == 1:
                    chunk = chunk.repeat(1, 3, 1, 1)
                features.append(self.model(chunk))

        return torch.cat(features, dim=0)

    def update(self, preds: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> None:
        """Accumulate features for FID computation."""
        """Accumulate features for FID computation."""
        pred_feat = self.extract_features(preds.to(self.device))
        target_feat = self.extract_features(target.to(self.device))

        self.pred_features.append(pred_feat)
        self.target_features.append(target_feat)

    def compute(self) -> torch.Tensor:
        """Compute FID from accumulated features."""
        if len(self.pred_features) == 0:
            return torch.tensor(float("inf"), device=self.device)

        pred_feat = torch.cat(self.pred_features, dim=0)
        target_feat = torch.cat(self.target_features, dim=0)

        # Compute statistics
        mu1, sigma1 = pred_feat.mean(dim=0), torch.cov(pred_feat.T)
        mu2, sigma2 = target_feat.mean(dim=0), torch.cov(target_feat.T)

        # Convert to numpy for stable sqrtm computation
        mu1_np = mu1.cpu().numpy()
        mu2_np = mu2.cpu().numpy()
        sigma1_np = sigma1.cpu().numpy()
        sigma2_np = sigma2.cpu().numpy()

        diff_np = mu1_np - mu2_np
        fid_np = diff_np.dot(diff_np)

        # Full FID needs sqrtm of product
        import scipy.linalg as linalg

        eps = 1e-6
        covmean, _ = linalg.sqrtm(sigma1_np.dot(sigma2_np), disp=False)

        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1_np.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1_np + offset).dot(sigma2_np + offset))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid_np = fid_np + np.trace(sigma1_np) + np.trace(sigma2_np) - 2 * np.trace(covmean)

        return torch.tensor(fid_np, device=self.device, dtype=torch.float32)

    def reset(self) -> None:
        """Reset accumulated features."""
        self.pred_features = []
        self.target_features = []

    def compute_metric(self, preds: Image, target: Image, **kwargs: Any) -> Tensor:
        """Single-batch FID approximation."""
        self.reset()
        self.update(preds, target)
        return self.compute()


# Aliases and exports
SSIM = SSIMMetric
IS = InceptionScoreMetric
PSC = PowerSpectrumConsistency
