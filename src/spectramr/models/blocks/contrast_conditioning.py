r"""Shared contrast-conditioning helper for field-aware FiLM generators.

The mrixfields corpus mixes three contrasts (T1w / T2w / T2-FLAIR) at every
field strength, and the dataset always emits a per-sample ``contrast_id``
(see ``data.datasets.mrixfields_dataset._CONTRAST_INDEX``). A field-FiLM
generator becomes *contrast-aware* by widening its :class:`FieldFiLMBlock`
``sequence_dim`` by ``num_contrasts`` and concatenating a contrast one-hot
into the FiLM ``sequence_features`` vector — the field strength itself stays
the separate leading scalar of :class:`FieldFiLMBlock` (the ``1 +`` in its
input ``Linear``).

This module is the single home for that construction so every generator
enforces the same two always-on invariants (CLAUDE.md):

* **#15 (wire the knob):** ``use_contrast_conditioning=True`` with no
  ``contrast_id`` in the batch **raises** — a wired knob never silently
  degrades to unconditional mode-averaging.
* **#9 (no silent fallback):** an out-of-range contrast id **raises** (via
  :func:`torch.nn.functional.one_hot`), never a silent clamp. The one-hot is
  built without a ``.item()`` host sync so it is training-loop safe
  (``performance.md``).

Two shapes of base sequence are supported:

* **No intrinsic sequence features** (``base_seq_dim == 0``): the field
  scalar is the only conditioning. Off-mode uses a single zero placeholder
  (``FieldFiLMBlock`` requires ``sequence_dim >= 1``); on-mode uses the
  contrast one-hot alone. This reproduces ``FieldVelocityUNet``'s original
  ``seq_dim = num_contrasts if use_contrast_conditioning else 1``.
* **Real base features** (``base_seq_dim >= 1``, e.g. a diffusion time
  embedding or a source/target field pair): the base is PRESERVED and the
  contrast one-hot is APPENDED, so contrast conditioning never clobbers the
  existing conditioning channel.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import one_hot

from spectramr.models.blocks.timestep_embedding import sinusoidal_timestep_embedding


def contrast_sequence_dim(
    base_seq_dim: int,
    num_contrasts: int,
    enabled: bool,
) -> int:
    """FiLM ``sequence_dim`` after optionally appending a contrast one-hot.

    Args:
        base_seq_dim: Width of the model's intrinsic FiLM sequence features
            (``0`` when the model conditions on the field scalar only).
        num_contrasts: Number of contrasts in the one-hot (>= 2 when enabled).
        enabled: Whether contrast conditioning is on.

    Returns:
        The ``sequence_dim`` to pass to :class:`FieldFiLMBlock` (always >= 1).
    """
    if base_seq_dim < 0:
        raise ValueError(f"base_seq_dim must be >= 0; got {base_seq_dim}")
    if enabled:
        if num_contrasts < 2:
            raise ValueError(f"contrast conditioning needs num_contrasts >= 2; got {num_contrasts}")
        return base_seq_dim + num_contrasts
    return base_seq_dim if base_seq_dim > 0 else 1


def build_contrast_sequence(
    base_seq: torch.Tensor | None,
    contrast_id: torch.Tensor | None,
    num_contrasts: int,
    enabled: bool,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the ``sequence_features`` tensor for :class:`FieldFiLMBlock`.

    Args:
        base_seq: The model's intrinsic sequence features ``[B, base_seq_dim]``
            or ``None`` when the model has no base features (field-scalar only).
        contrast_id: Per-sample contrast index ``[B]`` (long). Required when
            ``enabled``; ignored otherwise.
        num_contrasts: One-hot width.
        enabled: Whether contrast conditioning is on.
        batch_size: Batch size ``B`` — used only to size the off-mode zero
            placeholder when ``base_seq is None``.
        device / dtype: Target device/dtype for a constructed tensor.

    Returns:
        ``[B, sequence_dim]`` matching :func:`contrast_sequence_dim`.

    Raises:
        ValueError: ``enabled`` but ``contrast_id is None`` (#15).
        RuntimeError: an out-of-range / negative id (#9, from ``one_hot``).
    """
    if not enabled:
        if base_seq is not None:
            return base_seq
        return torch.zeros(batch_size, 1, device=device, dtype=dtype)

    if contrast_id is None:
        raise ValueError(
            "contrast conditioning is enabled but the batch carries no "
            "'contrast_id'. The mrixfields dataset emits contrast_id — thread "
            "it through the strategy to the model forward (CLAUDE.md #15)."
        )
    # one_hot raises on negative / >= num_contrasts ids (no silent clamp, #9)
    # and avoids a host sync (no .item()) in the forward path.
    cid = contrast_id.reshape(-1).long()
    onehot = one_hot(cid, num_contrasts).to(device=device, dtype=dtype)
    if base_seq is None:
        return onehot
    return torch.cat([base_seq, onehot], dim=-1)


def broadcast_conditioning_map(
    image: torch.Tensor,
    timesteps: torch.Tensor | None,
    contrast_idx: torch.Tensor | None,
    *,
    time_embed_dim: int,
    num_contrasts: int,
    projection: nn.Module,
    owner: str,
) -> torch.Tensor:
    """Broadcast ``(t, contrast)`` to a spatial map shaped like ``image``.

    The channel-concatenation half of contrast conditioning, as
    :func:`build_contrast_sequence` is the vector half. A conditioned CRITIC
    cannot use FiLM the way the generators above do -- it wraps whatever inner
    critic the registry hands back and has no access to that critic's block
    structure -- so it widens the input instead: project ``[t_embedding,
    one_hot(contrast)]`` to ``cond_channels``, broadcast over the spatial dims,
    concatenate. Lives here rather than on the critic so both halves of the
    same construction share one home (non-negotiable 6), and so the raises
    below are written once.

    **It raises rather than defaulting.** A caller reaches this function only
    by declaring ``supports_contrast_conditioning``; scoring unconditioned
    because a payload was absent would make that declaration a lie for a whole
    run while every log line looked correct (non-negotiable 3). The strategy
    that supplies the payload never introspects the caller's signature -- it
    forwards because the registry says to -- so a missing payload is a wiring
    defect, and the only useful response is to say so on step 1.

    Args:
        image: The tensor about to be scored; supplies batch, device, dtype and
            the spatial rank to broadcast over. Any rank ``>= 3`` works, so 2D
            ``(B, C, H, W)`` and 3D ``(B, C, D, H, W)`` need no special case.
        timesteps: Per-sample diffusion timestep. **Unnormalized** -- see the
            conditioned critic's module docstring for why no horizon is taken.
        contrast_idx: Per-sample contrast id.
        time_embed_dim: Width of the sinusoidal ``t`` embedding.
        num_contrasts: One-hot width; an out-of-range id raises (#9).
        projection: Maps ``(B, sequence_dim)`` to ``(B, cond_channels)``. Owned
            by the caller, because ``cond_channels`` is what widened the inner
            critic's first convolution and only the caller knows it.
        owner: Class name for the error messages, so a missing payload names
            the critic that needed it rather than this helper.

    Returns:
        ``(B, cond_channels, *image.shape[2:])``, ready to concatenate on dim 1.

    Raises:
        ValueError: If either payload is absent, or if its batch does not match
            ``image``.
    """
    batch = image.shape[0]
    if timesteps is None:
        raise ValueError(
            f"{owner} was called without `timesteps`. This model declares "
            "supports_contrast_conditioning and scores a t-labelled "
            "distribution; there is no meaningful unconditioned behaviour to "
            "fall back to. The diffusion strategy supplies the payload via "
            "`_critic_conditioning` (#1931)."
        )
    if contrast_idx is None:
        raise ValueError(
            f"{owner} was called without `contrast_idx`. The arm declares "
            "data.multi_contrast, so the batch carries one; thread it through "
            "rather than scoring the marginal over contrasts (CLAUDE.md #15)."
        )

    # A mismatch here is the silent-corruption case: broadcasting would pair
    # each sample with the WRONG label instead of raising. ``contrast_idx`` is
    # repeat_interleaved when 5D volumes are flattened to slices
    # (``diffusion.py`` ``_prepare_diffusion_inputs``), so the two can
    # legitimately disagree with the pre-flatten batch and must be checked
    # against the tensor actually being scored.
    t_flat = timesteps.reshape(-1)
    c_flat = contrast_idx.reshape(-1)
    if t_flat.shape[0] != batch or c_flat.shape[0] != batch:
        raise ValueError(
            f"{owner} conditioning batch mismatch: image has {batch} samples "
            f"but timesteps has {t_flat.shape[0]} and contrast_idx has "
            f"{c_flat.shape[0]}. Broadcasting these would label samples with "
            "another sample's timestep or contrast."
        )

    t_emb = sinusoidal_timestep_embedding(t_flat, time_embed_dim)
    t_emb = t_emb.to(device=image.device, dtype=image.dtype)
    seq = build_contrast_sequence(
        t_emb,
        c_flat,
        num_contrasts,
        enabled=True,
        batch_size=batch,
        device=image.device,
        dtype=image.dtype,
    )
    cond = projection(seq)
    spatial = image.shape[2:]
    channels = cond.shape[1]
    cond = cond.reshape(batch, channels, *([1] * len(spatial)))
    return cond.expand(batch, channels, *spatial)


__all__ = [
    "broadcast_conditioning_map",
    "build_contrast_sequence",
    "contrast_sequence_dim",
]
