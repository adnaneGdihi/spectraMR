"""Normalization SSOT — Single Source of Truth for all normalization strategies.

This module is the canonical location for every normalization function used
across the codebase.  All consumers (datasets, subject builders, TorchIO
transforms, training strategies) **must** delegate to these pure functions
instead of reimplementing the math inline.

Provides:

1. ``NormalizationStrategy`` enum — percentile / zscore / minmax / none.
2. ``NormalizationConfig`` frozen dataclass — carries strategy + params.
3. Core pure functions (stateless, no side-effects):
   * ``compute_magnitude`` — complex / stacked magnitude computation.
   * ``normalize_percentile`` — robust magnitude-percentile scaling.
   * ``normalize_zscore`` — zero-mean, unit-std.
   * ``normalize_minmax`` — min-max rescaling.
   * ``normalize_tensor`` — strategy dispatcher.
   * ``denormalize_percentile`` — inverse of percentile scaling.
4. ``KSpaceNormalizationTransform`` — TorchIO wrapper delegating to core.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Any

import torch
import torchio as tio

from spectramr.core.quantile import robust_quantile

logger = logging.getLogger(__name__)

#: The ``data.processing.normalization_type`` spellings this resolver can build.
#: Config vocabulary, NOT enum vocabulary — the two disagree on two members
#: (config ``standard`` is enum ``zscore``), so an error quoting the enum would
#: tell the author to write a value the schema Literal rejects.
#: ``robust_percentile`` is absent by design: it is an alias, folded to
#: ``percentile`` by the table in :meth:`NormalizationStrategy.from_string` --
#: the one licensed gap between the schema Literal and this set, and the one
#: place the fold is spelled (the builder and the predict path used to carry
#: copies; :func:`resolve_image_normalization` is what they call now).
IMPLEMENTED_NORMALIZATION_TYPES: tuple[str, ...] = (
    "none",
    "standard",
    "minmax",
    "percentile",
)

# ---------------------------------------------------------------------------
# 1. Strategy Enum
# ---------------------------------------------------------------------------


class NormalizationStrategy(str, enum.Enum):
    """Available normalization strategies.

    ``PERCENTILE``
        Divide by a robust percentile of the magnitude.  Phase-preserving.
    ``ZSCORE``
        Zero-mean, unit-std (Gaussian whitening).
    ``MINMAX``
        Rescale to an arbitrary output range ``[out_min, out_max]``.
    ``NONE``
        Identity — return the data unchanged.
    """

    PERCENTILE = "percentile"
    ZSCORE = "zscore"
    MINMAX = "minmax"
    NONE = "none"

    @classmethod
    def from_string(cls, value: str) -> NormalizationStrategy:
        """Parse a strategy name (case-insensitive).

        Args:
            value: Strategy name string.

        Returns:
            Matching ``NormalizationStrategy`` enum member.

        Raises:
            ValueError: If *value* is not a recognised strategy name.
        """
        canonical = value.strip().lower()
        # Config-vocabulary aliases. The schema's `normalization_type` Literal
        # and this enum were named independently and disagree on two members:
        # the config says "robust_percentile" and "standard" where the enum says
        # PERCENTILE and ZSCORE. Both are resolved here rather than at each call
        # site, so the adapter is one table instead of a convention every reader
        # has to re-know — the failure mode that made a fifth SignalDomain
        # vocabulary look necessary.
        canonical = {
            "robust_percentile": "percentile",
            "standard": "zscore",
        }.get(canonical, canonical)
        try:
            return cls(canonical)
        except ValueError:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(
                f"Unknown normalization strategy '{value}'. Valid options: {valid}"
            ) from None


# ---------------------------------------------------------------------------
# 2. Config Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizationConfig:
    """Immutable configuration for a normalization pass.

    Attributes:
        strategy: Which normalisation algorithm to apply.
        percentile: Upper quantile for ``PERCENTILE`` strategy (0–1 range).
        out_range: Target output range for ``PERCENTILE`` / ``MINMAX``.
        eps: Small constant for numerical stability.
        clamp: Whether to hard-clamp output to *out_range* (``PERCENTILE`` only).
        min_scale: Minimum allowed scale factor to avoid noise amplification.
    """

    strategy: NormalizationStrategy = NormalizationStrategy.NONE
    percentile: float = 0.99
    out_range: tuple[float, float] = (0.0, 1.0)
    eps: float = 1e-8
    clamp: bool = True
    min_scale: float = 0.0


# ---------------------------------------------------------------------------
# 3. Core Pure Functions
# ---------------------------------------------------------------------------


def compute_magnitude(
    data: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute magnitude from complex or stacked real/imag tensor.

    Handles multiple layouts:
    * Native complex (``data.is_complex()``) — ``torch.abs(data)``.
    * Last dim is 2 (e.g., [..., 2]) — ``sqrt(re² + im²)``.
    * Stacked real/imag with first dim == 2 (e.g., [2, H, W]).
    * Stacked real/imag with second dim == 2 (e.g., [B, 2, H, W]).
    * Channel-stacked (C*2, H, W) — reshaped to (C, 2, H, W).

    Args:
        data: Input tensor.
        eps: Stability constant added inside ``sqrt``.

    Returns:
        Magnitude tensor.
    """
    if torch.is_complex(data):
        return torch.abs(data)

    # Check channel-position layouts BEFORE the trailing-dim heuristic —
    # a tensor like [B=1, 2, H=2, W=2] matches both shape[-1]==2 and
    # shape[1]==2, and picking the trailing dim would split on the
    # spatial axis. Channel position is the canonical real/imag layout
    # for 3D+ tensors in this codebase.

    # Handle [B, 2, H, W]
    if data.ndim >= 3 and data.shape[1] == 2:
        return torch.sqrt(data[:, 0:1] ** 2 + data[:, 1:2] ** 2 + eps)

    # Handle [2, ...]
    if data.ndim >= 2 and data.shape[0] == 2:
        return torch.sqrt(data[0:1] ** 2 + data[1:2] ** 2 + eps)

    # Handle [..., 2] — only safe for low-rank tensors where channel
    # ambiguity is impossible.
    if data.ndim <= 2 and data.shape[-1] == 2:
        return torch.sqrt(data[..., 0:1] ** 2 + data[..., 1:2] ** 2 + eps)

    # Handle channel-stacked [C*2, H, W]
    if data.ndim >= 3 and data.shape[0] % 2 == 0:
        c = data.shape[0] // 2
        view = data.view(c, 2, *data.shape[1:])
        return torch.sqrt(view[:, 0:1] ** 2 + view[:, 1:2] ** 2 + eps)

    # Fallback: treat as real-valued
    return torch.abs(data)


def normalize_percentile(
    data: torch.Tensor,
    percentile: float = 0.99,
    eps: float = 1e-8,
    out_range: tuple[float, float] = (0.0, 1.0),
    clamp: bool = True,
    min_scale: float = 0.0,
    *,
    magnitude: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Robust percentile normalization preserving phase.

    Divides *data* by the *percentile*-th quantile of its magnitude,
    then optionally rescales + clamps to *out_range*.

    Args:
        data: Input tensor (complex, stacked, or real).
        percentile: Quantile in ``(0, 1]`` for robust max.
        eps: Numerical stability constant.
        out_range: ``(min, max)`` output range after scaling.
        clamp: Hard-clamp output to *out_range*.
        min_scale: Minimum allowed scale to prevent noise amplification.
        magnitude: Pre-computed magnitude (saves recomputation if caller
            already has it).

    Returns:
        ``(normalized_data, scale)`` where *scale* is the quantile value
        used.  Call ``denormalize_percentile(normalized, scale)`` to invert.
    """
    if magnitude is None:
        magnitude = compute_magnitude(data, eps=eps)

    scale = robust_quantile(magnitude.flatten().float(), percentile)
    scale = torch.clamp(scale, min=max(eps, min_scale))

    normalized = data / scale

    out_min, out_max = out_range
    if out_min != 0.0 or out_max != 1.0:
        # Shift into target range: first clamp to [0, 1], then rescale
        if clamp:
            normalized = torch.clamp(normalized, 0.0, 1.0)
        normalized = normalized * (out_max - out_min) + out_min
    elif clamp:
        normalized = torch.clamp(normalized, out_min, out_max)

    # NaN safety
    if torch.isnan(normalized).any():
        normalized = torch.nan_to_num(normalized, nan=out_range[0])

    return normalized, scale


def denormalize_percentile(
    data: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Inverse of ``normalize_percentile``.

    Args:
        data: Normalized tensor.
        scale: Scale factor returned by ``normalize_percentile``.

    Returns:
        Data in original physical scale.
    """
    return data * scale


def _log_magnitude_rescale(
    data: torch.Tensor,
    func: callable,
    channel_dim: int,
    eps: float,
) -> torch.Tensor:
    """Apply a positive scalar magnitude map ``func`` while preserving phase.

    Multiplies the complex value by ``func(|k|) / |k|`` so the magnitude
    becomes ``func(|k|)`` and the phase is untouched (the factor is a
    non-negative real scalar). Handles native-complex tensors and
    real-stacked **interleaved** ``[..., R0, I0, R1, I1, ...]`` layouts along
    ``channel_dim`` (the convention used by ``compute_magnitude`` and the
    k-space generators).
    """
    if torch.is_complex(data):
        mag = data.abs()
        factor = func(mag) / mag.clamp_min(eps)
        return data * factor

    cd = channel_dim % data.ndim
    n = data.shape[cd]
    if n % 2 != 0:
        raise ValueError(
            f"_log_magnitude_rescale expects an even channel count for "
            f"real-stacked complex data, got {n} along dim {cd}."
        )
    idx = torch.arange(n, device=data.device)
    real = data.index_select(cd, idx[0::2])
    imag = data.index_select(cd, idx[1::2])
    mag = torch.sqrt(real * real + imag * imag + eps)
    factor = func(mag) / mag.clamp_min(eps)
    real_s, imag_s = real * factor, imag * factor
    # Re-interleave: stack a new pair-axis right after channel_dim, then fold
    # it back in so the layout returns to [..., R0, I0, R1, I1, ...].
    stacked = torch.stack([real_s, imag_s], dim=cd + 1)
    return stacked.reshape(data.shape)


def compress_kspace_log(
    data: torch.Tensor,
    channel_dim: int = 0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Phase-preserving log1p magnitude compression for k-space.

    k-space has a ~200x dynamic range (the DC term dwarfs the periphery),
    which a CNN cannot represent — left raw, the network squashes the range
    and produces a centre "DC blob" after IFFT. This maps the magnitude
    ``m -> log1p(m)`` while preserving phase, compressing the range to a
    CNN-friendly scale. Invert with :func:`decompress_kspace_log`.

    Intended to run **after** percentile normalization (``k / scale``) so the
    log operates on a unit-ish magnitude. See ``KSpaceNormalizationTransform``.

    Args:
        data: Complex or real-stacked interleaved k-space.
        channel_dim: Channel axis for real-stacked input (real/imag
            interleaved). Ignored for native-complex tensors.
        eps: Numerical-stability floor.
    """
    return _log_magnitude_rescale(data, torch.log1p, channel_dim, eps)


# ``expm1`` overflows float32 at an argument of ~88.7 (``ln(FLT_MAX)``). A
# diverged or under-trained model can emit a *compressed* |k| far above that
# (the kernelized-attention arm reached ~1750 at iter 1000), so an un-clamped
# ``expm1`` returns ``inf`` -> ``inf * scale`` -> ``NaN`` that poisons EVERY
# validation metric and makes EarlyStopping pick "best" on ``NaN``. Physical
# compressed magnitudes are <= ~6 (``log1p`` of a few hundred), so a 30.0
# ceiling (``expm1(30) ~ 1e13``) never touches legitimate data and keeps the
# compress->decompress round-trip exact, while turning a pathological blow-up
# into an honestly-bad *finite* score. Applied symmetrically to pred + target.
DECOMPRESS_MAGNITUDE_CEILING = 30.0


def decompress_kspace_log(
    data: torch.Tensor,
    channel_dim: int = 0,
    eps: float = 1e-8,
    max_compressed_magnitude: float = DECOMPRESS_MAGNITUDE_CEILING,
) -> torch.Tensor:
    """Inverse of :func:`compress_kspace_log` (maps magnitude ``m -> expm1(m)``).

    Apply this before any IFFT / metric / visualization that needs physical
    k-space, then multiply by the stored ``kspace_scale`` to undo the
    percentile normalization.

    Args:
        data: Complex or real-stacked interleaved compressed k-space.
        channel_dim: Channel axis for real-stacked input. Ignored for complex.
        eps: Numerical-stability floor.
        max_compressed_magnitude: Ceiling on the compressed magnitude fed to
            ``expm1`` (see :data:`DECOMPRESS_MAGNITUDE_CEILING`). Guards against
            ``inf``/``NaN`` when a model emits a pathologically large |k|; set
            above any physical value so legitimate data is untouched.
    """

    def _clamped_expm1(mag: torch.Tensor) -> torch.Tensor:
        return torch.expm1(mag.clamp(max=max_compressed_magnitude))

    return _log_magnitude_rescale(data, _clamped_expm1, channel_dim, eps)


def kspace_image_domain_scale(
    data: torch.Tensor,
    *,
    percentile: float = 0.95,
    channel_dim: int = 0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Parseval-compliant robust scale: a percentile of the IMAGE-domain RSS.

    The alternative to scaling by a k-space magnitude percentile. k-space is
    heavy-tailed at DC, so a k-space quantile says little about the brightness
    of the reconstructed image; taking the quantile *after* ``ifft2c`` + coil
    RSS instead puts the reconstructed **image** at ~unit scale, which is what
    image-domain losses and metrics are graded in.

    This is the scale ``M4RawRepetitionDataset`` used to compute inline. It
    lives here so the morphing stays in the transform layer (canonical home)
    and the dataset can go back to matching and serving.

    The spatial plane is the two axes **immediately following** *channel_dim*,
    which is what every layout in this codebase uses: TorchIO ``(C, H, W, D)``
    and bare ``(C, H, W)`` with ``channel_dim=0``, batched ``(B, C, H, W)`` with
    ``channel_dim=1``. Taking the trailing two axes instead would transform over
    ``(W, D)`` for a multi-slice TorchIO subject — a ~2.6x wrong scale on
    structured anatomy (and, note, an error invisible to white-noise fixtures,
    which are invariant under any unitary transform).

    Args:
        data: Complex or real-stacked interleaved k-space, with real/imag
            interleaved along *channel_dim* when real.
        percentile: Quantile in ``(0, 1]`` of the image-domain RSS magnitude.
        channel_dim: Channel/coil axis. Spatial axes are the next two.
        eps: Numerical-stability floor.

    Returns:
        Scalar scale tensor, clamped to ``>= eps``.
    """
    from spectramr.infrastructure.physics.fft_ops import ifft2c

    cd = channel_dim % data.ndim
    if torch.is_complex(data):
        complex_k = data
    else:
        n = data.shape[cd]
        if n % 2 != 0:
            # Not a real/imag interleaved stack — no complex image to form.
            # Fall back to the k-space magnitude quantile rather than pairing
            # unrelated channels (which would silently invent a phase).
            mag = compute_magnitude(data, eps=eps)
            return robust_quantile(mag.flatten().float(), percentile).clamp(min=eps)
        idx = torch.arange(n, device=data.device)
        real = data.index_select(cd, idx[0::2])
        imag = data.index_select(cd, idx[1::2])
        complex_k = torch.complex(real.float(), imag.float())

    if complex_k.ndim < cd + 3:
        raise ValueError(
            f"[kspace_image_domain_scale] channel_dim={channel_dim} needs two "
            f"spatial axes after it, but the tensor is {tuple(complex_k.shape)}."
        )

    # Coil -> axis 0, spatial (H, W) -> the last two, so ifft2c sees the
    # [N, H, W] layout it centres/orthonormalises per slice.
    moved = torch.movedim(complex_k, (cd, cd + 1, cd + 2), (0, -2, -1))
    h, w = moved.shape[-2], moved.shape[-1]
    images = ifft2c(moved.reshape(-1, h, w)).reshape(moved.shape)

    # Coil RSS along the (now leading) coil axis.
    if moved.shape[0] > 1:
        rss = torch.sqrt((images.abs() ** 2).sum(dim=0) + eps)
    else:
        rss = images.abs()

    return robust_quantile(rss.flatten().float(), percentile).clamp(min=eps)


def normalize_kspace_robust(
    data: torch.Tensor,
    *,
    percentile: float = 0.99,
    log_scaling: bool = False,
    scale: torch.Tensor | float | None = None,
    channel_dim: int = 0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SSOT robust k-space normalization (percentile divide + optional log).

    This is the single entry point every k-space normalizer must call —
    ``KSpaceNormalizationTransform``, ``FastMRISubjectBuilder._normalize_kspace``,
    and ``M4RawRepetitionDataset`` previously each rolled their own copy
    (an SSOT violation that let ``log_scaling`` silently no-op in some paths).

    Steps:

    1. Divide by a robust scale (``percentile``-th quantile of the magnitude),
       which brings the bulk of k-space to ~unit scale but leaves the DC term
       far above 1 (a ~200x dynamic range).
    2. If ``log_scaling``, apply phase-preserving ``log1p`` magnitude
       compression (:func:`compress_kspace_log`) to tame that range to ~20x —
       the step that prevents the network from squashing its output into a
       centre "DC blob".

    Args:
        data: Complex or real-stacked interleaved k-space.
        percentile: Quantile in ``(0, 1]`` for the robust scale. Ignored if
            ``scale`` is supplied.
        log_scaling: Apply log1p magnitude compression after the divide.
        scale: Pre-computed scale (e.g. a centre-patch quantile). When ``None``
            the whole-tensor quantile is used.
        channel_dim: Channel axis for real-stacked input (real/imag
            interleaved); ignored for native-complex tensors.
        eps: Numerical-stability floor.

    Returns:
        ``(normalized, scale)``. Invert exactly with
        :func:`denormalize_kspace_robust` (pass the same ``log_scaling``).
    """
    if scale is None:
        magnitude = compute_magnitude(data, eps=eps)
        scale = robust_quantile(magnitude.flatten().float(), percentile)
        scale = torch.clamp(scale, min=eps)
    elif not torch.is_tensor(scale):
        scale = torch.as_tensor(scale, dtype=torch.float32, device=data.device)

    out = data / scale
    if log_scaling:
        out = compress_kspace_log(out, channel_dim=channel_dim, eps=eps)
    return out, scale


def denormalize_kspace_robust(
    data: torch.Tensor,
    scale: torch.Tensor | float,
    *,
    log_scaling: bool = False,
    channel_dim: int = 0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Exact inverse of :func:`normalize_kspace_robust`.

    Decompress (when ``log_scaling``) then multiply by ``scale``. Apply at the
    k-space -> image boundary (validation metrics, visualization, inference
    writing) for predictions *and* targets, so the whole pipeline — input,
    prediction, and output — lives in the same (compressed) domain and the
    scales stay consistent.
    """
    if log_scaling:
        data = decompress_kspace_log(data, channel_dim=channel_dim, eps=eps)
    return data * scale


#: The advertised set of scale domains. An unknown value raises rather than
#: degrading to a default (pitfall #9).
KSPACE_SCALE_DOMAINS = ("kspace", "image")

#: The five knobs :meth:`KSpaceNormalizationSpec.from_data_config` resolves, as
#: they are spelled on ``data.processing`` today. Two were also renamed by the
#: phase-9 block decomposition (``normalize_kspace`` ->
#: ``enable_kspace_normalization``, ``log_scaling`` -> ``enable_log_scaling``).
_PROCESSING_KNOBS: tuple[str, ...] = (
    "enable_kspace_normalization",
    "kspace_percentile",
    "enable_log_scaling",
    "kspace_scale_domain",
    "log_scaling_center_fraction",
)

#: Their pre-decomposition spellings, flat on ``data`` itself. Reachable only
#: from a stored/serialized config (a checkpoint's ``data`` dict, an inference
#: payload) — absent from every current schema object, which is precisely why a
#: flat-ONLY reader resolved silently to the defaults.
_LEGACY_FLAT_KNOBS: tuple[str, ...] = (
    "normalize_kspace",
    "kspace_percentile",
    "log_scaling",
    "kspace_scale_domain",
    "log_scaling_center_fraction",
)


def _read_declared_knobs(
    processing: Any,
    names: tuple[str, ...],
    *,
    nullable: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Read every *name* off *processing*, raising when one is not declared.

    A plain ``getattr(..., default)`` over a *declared* field is how a rename
    disables a mechanism in silence — the reader keeps the old spelling, every
    read misses, and nothing goes red because "absent" and "off" are the same
    boolean. Absent-*block* and absent-*field* are different facts and get
    different answers, the same discipline as
    ``infrastructure.training.utils.kspace_view.log_scaling_enabled``.

    Args:
        processing: The ``data.processing`` block, as a schema object or dict.
        names: Field names that MUST be declared.
        nullable: Subset of *names* whose declared value may be ``None``.

    Raises:
        AttributeError: A name in *names* is not declared on *processing*.
        ValueError: A non-*nullable* name is declared as ``None``.
    """
    out: dict[str, Any] = {}
    for name in names:
        if isinstance(processing, dict):
            declared = name in processing
            value = processing.get(name)
        else:
            declared = hasattr(processing, name)
            value = getattr(processing, name, None)
        if not declared:
            raise AttributeError(
                # Reachable only from a RAW mapping. Training passes
                # ``config.data`` and inference ``config.model_dump()``, and
                # pydantic has filled the default in both, so nothing in
                # production reaches this branch. The question is asked where it
                # can still be answered, by
                # ``ConfigHealthChecker.check_kspace_scale_domain_is_declared``.
                f"data.processing exists but declares no {name!r}. This is a "
                "k-space normalization knob every training and inference path "
                "resolves through KSpaceNormalizationSpec (issue #572); if it "
                "was renamed, update `_PROCESSING_KNOBS` with it. Falling back "
                "to the default here is what made `enable_log_scaling: true` "
                "resolve to False for a whole schema migration (CLAUDE.md #3b)."
            )
        if value is None and name not in nullable:
            raise ValueError(
                f"data.processing.{name} is declared as None, but it has no "
                "null meaning. Remove the key to accept the schema default, or "
                "give it a value — substituting the default for an explicit "
                "None is the silent substitution this resolver exists to stop."
            )
        out[name] = value
    return out


@dataclass(frozen=True)
class KSpaceNormalizationSpec:
    """The ONE resolver for k-space normalization parameters (issue #572).

    Training normalizes through ``KSpaceNormalizationTransform``, driven by
    ``data.processing.kspace_percentile`` / ``.enable_log_scaling`` /
    ``.kspace_scale_domain`` (this docstring taught the flat pre-decomposition
    spelling for the whole campaign in which :meth:`from_data_config` was silently
    reading it). The inference strategies used to key off an unrelated block
    (``data.normalization_kwargs`` — which belongs to ``normalization_type``, the
    *image* normalization strategy) and never applied ``log_scaling`` at all, so
    a model trained on percentile-divided, log-compressed k-space was fed raw
    k-space at inference and its output decoded with the wrong inverse.

    One resolver, read by every consumer — the same discipline as
    ``build_loss_weight_table`` for loss weights (pitfall #13b).

    The spatial plane is always the two axes after *channel_dim*, so this works
    for TorchIO ``(C, H, W, D)`` (``channel_dim=0``) and batched inference
    ``(B, C, H, W)`` (``channel_dim=1``) alike.
    """

    enabled: bool = False
    percentile: float = 0.99
    log_scaling: bool = False
    scale_domain: str = "kspace"
    center_fraction: float | None = None
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.scale_domain not in KSPACE_SCALE_DOMAINS:
            raise ValueError(
                f"[KSpaceNormalizationSpec] Unknown scale_domain="
                f"{self.scale_domain!r}. Valid: {list(KSPACE_SCALE_DOMAINS)}."
            )

    @classmethod
    def from_data_config(cls, data: Any) -> KSpaceNormalizationSpec:
        """Resolve from a ``DataConfigSchema`` or the equivalent plain dict.

        Inference receives the run config as a nested dict, training as the
        frozen pydantic model, so both are accepted.

        The five knobs live on ``data.processing``. This used to read only their
        flat pre-decomposition spellings off ``data`` itself, with
        ``getattr(data, name, default)`` — and post-decomposition those names
        exist on no schema object at all, with no forwarding shim, so **every
        read missed and every declared value was replaced by its default**:

        ==============================  ========  =========
        declared (exp_11_attention_none)  resolved  default?
        ==============================  ========  =========
        ``enable_kspace_normalization: true``  ``False``   yes
        ``enable_log_scaling: true``          ``False``   yes
        ``kspace_percentile: 0.95``           ``0.99``    yes
        ``kspace_scale_domain: image``        ``kspace``  yes
        ==============================  ========  =========

        Because :meth:`normalize` is a *silent* no-op when disabled, the
        ``apply_kspace_normalization`` fallback that exists to normalize a batch
        the dataloader served raw returned the raw batch and a unit scale while
        reporting success. ``experiment_11_attention_none`` trained on raw
        k-space at ``|k|max ~ 2400``, and the render path — which reads the
        *declaration* — then ``expm1``-ed it anyway, clamping every
        contrast-carrying bin to ``DECOMPRESS_MAGNITUDE_CEILING`` and rendering
        a phase-only edge map instead of anatomy.

        Absent-*block* and absent-*field* get different answers (see
        :func:`_read_declared_knobs`): no ``processing`` **and** no flat knobs
        resolves to the defaults, because nothing was declared at all; a
        ``processing`` block that exists and omits a knob **raises**.

        Reading ``log_scaling_center_fraction`` from the block is a deliberate
        behaviour change: the schema defaults it to ``0.25``, and
        ``TorchIOTransformBuilder`` already passes that to
        ``KSpaceNormalizationTransform``. The fallback's stated job is to match
        what the transform would have produced, so it must use the same centre
        fraction. It stays inert for ``scale_domain='image'``, which ignores it.
        """
        processing = (
            data.get("processing") if isinstance(data, dict) else getattr(data, "processing", None)
        )

        if processing is not None:
            declared = _read_declared_knobs(
                processing,
                _PROCESSING_KNOBS,
                nullable=("log_scaling_center_fraction",),
            )
            return cls(
                enabled=bool(declared["enable_kspace_normalization"]),
                percentile=float(declared["kspace_percentile"]),
                log_scaling=bool(declared["enable_log_scaling"]),
                scale_domain=str(declared["kspace_scale_domain"]),
                center_fraction=declared["log_scaling_center_fraction"],
            )

        present = [
            name
            for name in _LEGACY_FLAT_KNOBS
            if (name in data if isinstance(data, dict) else hasattr(data, name))
        ]
        if not present:
            # Nothing declared anywhere: a strategy built standalone in a unit
            # test, or a paradigm with no data section. Defaults are the honest
            # answer, and normalizing on a guess would be worse than not.
            return cls()

        logger.warning(
            "[kspace-norm] resolved k-space normalization from the "
            "pre-decomposition FLAT names %s — this config predates the "
            "`data.processing` block. The values were honoured, but re-save the "
            "config so the nested spelling is the one on record.",
            present,
        )

        def get(name: str, default: Any) -> Any:
            if isinstance(data, dict):
                value = data.get(name, default)
            else:
                value = getattr(data, name, default)
            return default if value is None else value

        return cls(
            enabled=bool(get("normalize_kspace", False)),
            percentile=float(get("kspace_percentile", 0.99)),
            log_scaling=bool(get("log_scaling", False)),
            scale_domain=str(get("kspace_scale_domain", "kspace")),
            center_fraction=get("log_scaling_center_fraction", None),
        )

    # -- scale ------------------------------------------------------------

    def compute_scale(self, kspace: torch.Tensor, *, channel_dim: int = 0) -> torch.Tensor:
        """Robust scale for *kspace*, measured in :pyattr:`scale_domain`."""
        if self.scale_domain == "image":
            return kspace_image_domain_scale(
                kspace,
                percentile=self.percentile,
                channel_dim=channel_dim,
                eps=self.eps,
            )

        magnitude = compute_magnitude(kspace, eps=self.eps)
        sample = magnitude
        # Centre patch: the spatial plane sits right after the channel axis,
        # matching kspace_image_domain_scale rather than assuming trailing dims.
        if self.log_scaling and self.center_fraction is not None:
            cd = channel_dim % kspace.ndim
            if magnitude.ndim >= cd + 3:
                sh, sw = magnitude.shape[cd + 1], magnitude.shape[cd + 2]
                ch = max(1, int(sh * self.center_fraction))
                cw = max(1, int(sw * self.center_fraction))
                sample = magnitude.narrow(cd + 1, (sh - ch) // 2, ch).narrow(
                    cd + 2, (sw - cw) // 2, cw
                )

        scale = robust_quantile(sample.flatten().float(), self.percentile)
        return torch.clamp(scale, min=self.eps)

    # -- apply ------------------------------------------------------------

    def normalize(
        self,
        kspace: torch.Tensor,
        *,
        scale: torch.Tensor | float | None = None,
        channel_dim: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Percentile divide + optional log1p, exactly as training applied it.

        Returns ``(normalized, scale)``. A disabled spec is a no-op returning a
        unit scale, so callers never branch on ``enabled`` themselves.
        """
        if not self.enabled:
            return kspace, torch.ones((), device=kspace.device)
        if scale is None:
            scale = self.compute_scale(kspace, channel_dim=channel_dim)
        return normalize_kspace_robust(
            kspace,
            scale=scale,
            log_scaling=self.log_scaling,
            channel_dim=channel_dim,
            eps=self.eps,
        )

    def denormalize(
        self,
        kspace: torch.Tensor,
        scale: torch.Tensor | float,
        *,
        channel_dim: int = 0,
    ) -> torch.Tensor:
        """Exact inverse of :meth:`normalize` (expm1 then rescale)."""
        if not self.enabled:
            return kspace
        return denormalize_kspace_robust(
            kspace,
            scale,
            log_scaling=self.log_scaling,
            channel_dim=channel_dim,
            eps=self.eps,
        )


def normalize_zscore(
    data: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Z-score normalization (zero-mean, unit-std).

    Args:
        data: Input tensor.
        eps: Stability constant added to std.

    Returns:
        ``(normalized_data, (mean, std))`` tuple.
    """
    mean = data.mean()
    std = data.std() + eps
    return (data - mean) / std, (mean, std)


def normalize_minmax(
    data: torch.Tensor,
    out_range: tuple[float, float] = (0.0, 1.0),
    eps: float = 1e-8,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Min-max rescaling.

    Maps ``[data.min(), data.max()]`` → ``out_range``.

    Args:
        data: Input tensor.
        out_range: ``(min, max)`` output range.
        eps: Stability constant for near-constant tensors.

    Returns:
        ``(normalized_data, (min_val, max_val))`` tuple.  If the input
        range is near-zero the data is returned unchanged with
        ``(0, 1)`` as dummy stats.
    """
    min_val = data.min()
    max_val = data.max()

    if (max_val - min_val) < eps:
        return data, (min_val, max_val)

    out_min, out_max = out_range
    normalized = (data - min_val) / (max_val - min_val)
    scaled = normalized * (out_max - out_min) + out_min
    return scaled, (min_val, max_val)


# ---------------------------------------------------------------------------
# 4. Strategy Dispatcher
# ---------------------------------------------------------------------------


def normalize_tensor(
    data: torch.Tensor,
    config: NormalizationConfig,
) -> torch.Tensor:
    """Apply normalization according to *config*.

    Convenience dispatcher that routes to the appropriate core function.

    Args:
        data: Input tensor.
        config: ``NormalizationConfig`` specifying strategy + params.

    Returns:
        Normalized tensor.  (Statistics are discarded — use the core
        functions directly when you need scale / stats for denormalization.)

    Raises:
        ValueError: If *config.strategy* is unrecognised.
    """
    if config.strategy is NormalizationStrategy.NONE:
        return data

    if config.strategy is NormalizationStrategy.PERCENTILE:
        normalized, _scale = normalize_percentile(
            data,
            percentile=config.percentile,
            eps=config.eps,
            out_range=config.out_range,
            clamp=config.clamp,
            min_scale=config.min_scale,
        )
        return normalized

    if config.strategy is NormalizationStrategy.ZSCORE:
        normalized, _stats = normalize_zscore(data, eps=config.eps)
        return normalized

    if config.strategy is NormalizationStrategy.MINMAX:
        normalized, _stats = normalize_minmax(data, out_range=config.out_range, eps=config.eps)
        return normalized

    raise ValueError(f"Unknown normalization strategy: {config.strategy}")


# ---------------------------------------------------------------------------
# 5. TorchIO Wrapper (k-space specialisation)
# ---------------------------------------------------------------------------


class KSpaceNormalizationTransform(tio.Transform):
    """Normalize k-space data by magnitude percentile while preserving phase.

    Only applies to images whose key contains ``'kspace'`` or is listed
    in :pyattr:`KSPACE_KEYS`.  Image-domain keys (``image``, ``mri``,
    ``gt``) are skipped.

    Crucially: Stores the normalization 'scale' in the subject's metadata
    so it can be reversed (denormalized) later in the model.

    Delegates to :func:`compute_magnitude` + :func:`normalize_percentile`.

    Controlled by ``config.normalize_kspace = True``.
    """

    # Keys that are k-space domain (will be normalized)
    KSPACE_KEYS = {
        "kspace",
        "kspace_sub",
        "kspace_full",
        "measured_kspace",
        "input",
        "target",
    }

    # Keys that are image domain (will NOT be normalized)
    IMAGE_KEYS = {"image", "mri", "gt"}

    #: Where the robust scale is measured. ``"kspace"`` takes the quantile of
    #: the k-space magnitude; ``"image"`` takes it of the coil-RSS magnitude
    #: after ``ifft2c`` (Parseval-compliant — puts the reconstructed image at
    #: ~unit scale). Advertised set, so an illegal YAML value raises.
    SCALE_DOMAINS = ("kspace", "image")

    def __init__(
        self,
        percentile: float = 0.99,
        eps: float = 1e-8,
        log_scaling: bool = False,
        center_fraction: float | None = None,
        scale_domain: str = "kspace",
    ) -> None:
        """__init__.

        Args:
            percentile (float): Quantile in ``(0, 1]`` for the robust scale.
            eps (float): Numerical-stability floor.
            log_scaling (bool): Apply phase-preserving log1p compression.
            center_fraction (float | None): Centre patch used for the k-space
                scale. Ignored when ``scale_domain='image'``.
            scale_domain (str): One of :pyattr:`SCALE_DOMAINS`.
        """
        super().__init__()
        # The spec validates scale_domain (raises on an unknown value) and owns
        # the scale computation shared with the inference path.
        self.spec = KSpaceNormalizationSpec(
            enabled=True,
            percentile=percentile,
            log_scaling=log_scaling,
            scale_domain=scale_domain,
            center_fraction=center_fraction,
            eps=eps,
        )
        self.percentile = percentile
        self.eps = eps
        self.log_scaling = log_scaling
        self.center_fraction = center_fraction
        self.scale_domain = scale_domain

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Normalize ONLY k-space images by magnitude percentile."""
        # 1. Identify valid k-space images
        target_images = [
            name
            for name in subject.get_images_names()
            if "kspace" in name.lower() or name.lower() in self.KSPACE_KEYS
        ]

        if not target_images:
            return subject

        # 2. Calculate Global Scale
        # derive scale from 'kspace' (fully sampled) if available,
        # otherwise use the first available k-space tensor.
        ref_key = next(
            (k for k in ["kspace", "target", "input"] if k in target_images),
            target_images[0],
        )
        ref_data = subject[ref_key].data

        # Delegate to the SSOT resolver so the training-time scale and the
        # inference-time scale are computed by the SAME code (issue #572).
        scale = self.spec.compute_scale(ref_data, channel_dim=0)

        # 3. Store Scale in Subject (Critical for denormalization)
        subject["kspace_scale"] = scale
        # Flag the log step so downstream denormalization (validation
        # metrics / visualization / inference) knows to apply expm1 before
        # multiplying by ``kspace_scale``. Without this, the inverse is a
        # plain ``* scale`` and the reconstruction is wrong.
        subject["kspace_log_scaled"] = bool(self.log_scaling)
        # POSITIVE evidence that this transform ran, distinct from the scale's
        # VALUE. Consumers used to infer "the dataloader normalized" from the
        # mere presence of ``kspace_scale`` — but a dataset that serves raw
        # k-space still publishes an identity ``kspace_scale = 1.0`` to keep the
        # published scale matching the tensor beside it
        # (``M4RawRepetitionDataset``). Presence therefore proved nothing, and
        # the one consumer that relied on it (``DiffusionTrainingStrategy``)
        # skipped its own normalization fallback and trained on raw k-space with
        # the full ~200x DC-vs-periphery range this transform exists to tame.
        subject["kspace_normalized"] = True

        # 4. Apply to ALL k-space images via the SSOT normalizer: percentile
        #    divide (reusing the centre-patch ``scale`` computed above), then
        #    (when enabled) phase-preserving log1p magnitude compression — the
        #    step that tames the ~200x DC-vs-periphery range and prevents the
        #    centre "DC blob". TorchIO data is [C, H, W(, D)] real/imag
        #    interleaved along channel 0 (ignored for complex tensors).
        for name in target_images:
            image = subject[name]
            normed, _ = normalize_kspace_robust(
                image.data,
                log_scaling=self.log_scaling,
                scale=scale,
                channel_dim=0,
                eps=self.eps,
            )
            image.set_data(normed)

        # 5. Mirror the marker keys into ``Subject.__dict__`` so they survive patch
        #    extraction. ``kspace_scale`` / ``kspace_log_scaled`` /
        #    ``kspace_normalized`` are NEW mapping keys, and ``Subject.__setitem__``
        #    is not defined, so ``dict.__setitem__`` never re-syncs. ``tio.Crop`` —
        #    the engine behind every ``PatchSampler``/``tio.Queue`` — builds its
        #    output solely from ``__dict__``, so without this the markers are DROPPED
        #    at patch extraction and the batch reaches ``train_step`` carrying no
        #    evidence it was normalized. That is exactly the ``kspace_scale is None``
        #    route ``_batch_is_already_normalized`` reads (#1211): the strategy then
        #    re-normalizes with a scale of its own and masks this transform's work.
        #    (#1213)
        subject.update_attributes()

        return subject


# ---------------------------------------------------------------------------
# 6. TorchIO Wrapper (image-domain specialisation)
# ---------------------------------------------------------------------------

#: What ``ContrastAwarePairedDataset`` applied internally before the image
#: normalization SSOT moved into the transform chain. Preserved verbatim because
#: 66 contrast-aware arms declare no ``normalization_type`` at all and therefore
#: relied on exactly these numbers; resolving them to "none" would have trained
#: those arms on unnormalized intensities.
_CONTRAST_AWARE_LEGACY_DEFAULT = NormalizationConfig(
    strategy=NormalizationStrategy.PERCENTILE,
    percentile=0.995,
    out_range=(0.0, 1.0),
    clamp=True,
    min_scale=0.05,  # noise floor: never amplify background
)

#: ``dataset_type`` values served by ``ContrastAwarePairedDataset``.
_CONTRAST_AWARE_TYPES = frozenset({"contrast_aware_paired", "nifti_paired"})


@dataclass(frozen=True)
class ImageNormalizationSpec:
    """Resolved image-domain normalization: WHAT runs, and WHY it was chosen.

    One resolver, so image intensity is normalized in exactly one place. Before
    this existed there were two, and they ran back to back inside a single
    ``__getitem__``: ``ContrastAwarePairedDataset`` normalized ``input`` and
    ``target``, then called ``self.transform(subject)`` on its last line, and
    the chain it called appended a *second* normalizer whose only gate was
    ``normalize_kspace``. Reading either file alone looked correct — the same
    shape as the k-space double-normalization in #571.

    ``source`` is provenance, not decoration: which rule fired is the thing a
    reader needs when the numbers move, and it is stamped into the resolved
    config rather than left to be re-derived.
    """

    config: NormalizationConfig
    source: str

    @property
    def enabled(self) -> bool:
        return self.config.strategy is not NormalizationStrategy.NONE

    @classmethod
    def from_declared(
        cls,
        normalization_type: str,
        dataset_type: str | None,
        normalization_kwargs: dict | None = None,
    ) -> ImageNormalizationSpec:
        """Resolve the spec for an arm.

        Args:
            normalization_type: ``data.processing.normalization_type``.
            dataset_type: ``data.dataset_type`` — needed only to decide whether
                the contrast-aware legacy default applies.
            normalization_kwargs: ``data.processing.normalization_kwargs``.

        Returns:
            The resolved spec.

        Raises:
            ValueError: on an unrecognised ``normalization_type`` (via
                :meth:`NormalizationStrategy.from_string`) — never a silent
                degrade to "none" (#9).
        """
        kwargs = normalization_kwargs or {}
        try:
            strategy = NormalizationStrategy.from_string(normalization_type)
        except ValueError as exc:
            # Re-raise in the CONFIG vocabulary. ``from_string`` reports the enum
            # members (``zscore``), but this classmethod is the config-facing
            # entry point and the author writes ``standard`` in YAML — quoting
            # the enum would name a value the schema Literal rejects.
            raise ValueError(
                f"[NORMALIZATION] Unknown normalization_type: "
                f"{normalization_type!r}. Valid values: "
                f"{', '.join(repr(v) for v in IMPLEMENTED_NORMALIZATION_TYPES)} "
                "('robust_percentile' is an alias of 'percentile', folded by "
                "NormalizationStrategy.from_string). Add the strategy "
                "to NormalizationStrategy and to IMPLEMENTED_NORMALIZATION_TYPES "
                "together if introducing one."
            ) from exc

        if strategy is NormalizationStrategy.NONE:
            if dataset_type in _CONTRAST_AWARE_TYPES:
                # Inherit what the dataset used to do, rather than resolving to
                # "no normalization" for an arm that never opted out of it.
                return cls(
                    config=_CONTRAST_AWARE_LEGACY_DEFAULT,
                    source="contrast_aware_legacy_default",
                )
            return cls(
                config=NormalizationConfig(strategy=NormalizationStrategy.NONE),
                source="disabled",
            )

        percentile = float(kwargs.get("percentile", 0.99))
        if percentile > 1.0:  # accept the 0-100 spelling
            percentile /= 100.0
        out_range = tuple(kwargs.get("out_range", (0.0, 1.0)))
        return cls(
            config=NormalizationConfig(
                strategy=strategy,
                percentile=percentile,
                out_range=(float(out_range[0]), float(out_range[1])),
                clamp=bool(kwargs.get("clamp", True)),
                min_scale=float(kwargs.get("min_scale", 0.0)),
            ),
            source="declared",
        )


def resolve_image_normalization(
    *,
    normalization_type: str,
    dataset_type: str | None,
    normalization_kwargs: dict | None,
    kspace_normalization_enabled: bool,
) -> ImageNormalizationSpec | None:
    """Decide, once, whether and how an arm normalizes image intensity.

    The one owner of a decision that ``TorchIOTransformBuilder`` (train and val
    chains) and ``pipelines/infer.py::_normalize_like_training`` each used to
    make with a copy of their own. A copy that drifts does not raise; it
    surfaces as a differently scaled tensor on one side of the train/predict
    pair (non-negotiable 17). Three parts, in order:

    1. **Mutual exclusion.** ``enable_kspace_normalization`` wins. Image
       normalizers clamp and shift values, which destroys complex k-space
       (negative values, phase), so an arm that normalizes k-space gets no
       image normalization at all. Returns ``None`` and says so at INFO in one
       wording, so a train log and a predict log read side by side.
    2. **Vocabulary.** The schema admits ``robust_percentile``; the enum says
       ``percentile``. The alias table in :meth:`NormalizationStrategy.from_string`
       is the only fold, reached through :meth:`ImageNormalizationSpec.from_declared`.
       No caller folds beforehand.
    3. **Resolution.** :meth:`ImageNormalizationSpec.from_declared` builds the
       spec and raises on an unknown type rather than degrading to ``'none'``
       (non-negotiable 3).

    Args:
        normalization_type: ``data.processing.normalization_type``, as declared.
        dataset_type: ``data.dataset_type``; decides whether the contrast-aware
            legacy default applies when the type is ``'none'``.
        normalization_kwargs: ``data.processing.normalization_kwargs``.
        kspace_normalization_enabled: ``data.processing.enable_kspace_normalization``.

    Returns:
        ``None`` when k-space normalization takes precedence; otherwise the
        resolved spec, whose ``enabled`` is False for ``'none'`` on a dataset
        type that never normalized internally.

    Raises:
        ValueError: On an unrecognised ``normalization_type``.
    """
    if kspace_normalization_enabled:
        if normalization_type != "none":
            logger.info(
                "[NORMALIZATION] Image normalization '%s' SKIPPED because "
                "normalize_kspace=True. Preserving complex k-space values.",
                normalization_type,
            )
        return None

    spec = ImageNormalizationSpec.from_declared(
        normalization_type=normalization_type,
        dataset_type=dataset_type,
        normalization_kwargs=normalization_kwargs,
    )
    if spec.enabled:
        logger.info(
            "[NORMALIZATION] Image %s (percentile=%s, clamp=%s, min_scale=%s) -- source=%s",
            spec.config.strategy.value,
            spec.config.percentile,
            spec.config.clamp,
            spec.config.min_scale,
            spec.source,
        )
    else:
        logger.info(
            "[NORMALIZATION] Image normalization disabled "
            "(normalization_type='none' on a dataset_type that never normalized internally)."
        )
    return spec


class ImageNormalizationTransform(tio.Transform):
    """Normalize IMAGE-domain intensity, once, through the SSOT dispatcher.

    Replaces the hand-rolled ``tio.ZNormalization`` / ``tio.RescaleIntensity``
    branches the transform builder used to append. That matters beyond
    de-duplication: ``RescaleIntensity(percentiles=(0, p))`` **clips** at the
    percentile before rescaling, while the builder's own comment described the
    operation as "divide by the 99th". :func:`normalize_percentile` divides, and
    clamps only when asked — so the declared semantics and the applied ones
    finally agree (plan item B5).

    Skips k-space keys outright. Image normalizers assume magnitude images and
    clamp/shift values, which destroys complex k-space (negative values, phase);
    the caller additionally never constructs this transform when
    ``normalize_kspace`` is set.
    """

    #: Image-domain keys this transform owns.
    IMAGE_KEYS = {"input", "target", "image", "mri", "gt"}

    def __init__(self, spec: ImageNormalizationSpec) -> None:
        """__init__.

        Args:
            spec: Resolved :class:`ImageNormalizationSpec`.
        """
        super().__init__()
        self.spec = spec

    #: Strategies that are NOT phase-safe on a complex tensor. ``PERCENTILE``
    #: divides by a magnitude quantile and so preserves phase by construction;
    #: ``ZSCORE``/``MINMAX`` would take a COMPLEX mean/min/max and produce
    #: complex output where the caller expects an intensity-normalized image.
    #: The wrapper this transform replaced (``_ComplexSafeIntensityTransform``)
    #: converted every complex image to magnitude first; that guarantee is kept
    #: here, narrowed to the strategies that actually need it.
    _MAGNITUDE_ONLY = frozenset({NormalizationStrategy.ZSCORE, NormalizationStrategy.MINMAX})

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Normalize every image-domain key, independently, in place."""
        if not self.spec.enabled:
            return subject

        for name in subject.get_images_names():
            lowered = name.lower()
            if "kspace" in lowered or lowered not in self.IMAGE_KEYS:
                continue
            image = subject[name]
            data = image.data
            if data.is_complex() and self.spec.config.strategy in self._MAGNITUDE_ONLY:
                data = data.abs()
            image.set_data(normalize_tensor(data, self.spec.config))

        # Provenance for the run record: which rule chose these numbers. A
        # reader comparing two arms' intensities needs to know whether the
        # config asked for this or inherited it.
        subject["image_normalization_source"] = self.spec.source
        return subject


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "KSPACE_SCALE_DOMAINS",
    "ImageNormalizationSpec",
    "ImageNormalizationTransform",
    "KSpaceNormalizationSpec",
    "KSpaceNormalizationTransform",
    "NormalizationConfig",
    "NormalizationStrategy",
    "compress_kspace_log",
    "compute_magnitude",
    "decompress_kspace_log",
    "denormalize_kspace_robust",
    "denormalize_percentile",
    "kspace_image_domain_scale",
    "normalize_kspace_robust",
    "normalize_minmax",
    "normalize_percentile",
    "normalize_tensor",
    "normalize_zscore",
]
