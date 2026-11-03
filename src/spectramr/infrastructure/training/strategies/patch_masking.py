"""Image-domain patch mask shared by the masked-image-modelling strategies.

One owner for the random patch mask used by ``MaskedPretrainingStrategy`` and by
``MAEPretrainingStrategy`` when ``training.mae.mask_domain`` is ``image``.
"""

from __future__ import annotations

import torch


def create_patch_mask(
    batch: int,
    height: int,
    width: int,
    patch_size: int,
    mask_ratio: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw an independent random patch mask per sample.

    Args:
        batch: Number of samples.
        height: Image height in pixels.
        width: Image width in pixels.
        patch_size: Side of the square patch in pixels.
        mask_ratio: Fraction of patches masked; ``int(total * ratio)`` are masked.
        device: Device of the returned mask.
        generator: Seeded stream for the draw. The mask is drawn on the generator's
            device and moved to ``device``; ``None`` draws from the global stream
            on ``device``.

    Returns:
        ``[B, 1, H, W]`` boolean mask, ``True`` where the pixel is masked.
    """
    if patch_size < 1 or patch_size > height or patch_size > width:
        raise ValueError(
            f"patch_size={patch_size} must lie in [1, min(H, W)] for a {height}x{width} image"
        )
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    total_patches = num_patches_h * num_patches_w
    num_masked = int(total_patches * mask_ratio)
    draw_device = generator.device if generator is not None else device

    masks = []
    for _ in range(batch):
        patch_indices = torch.randperm(total_patches, device=draw_device, generator=generator)[
            :num_masked
        ]
        # A flat scatter: [idx // W, idx % W] addresses flat[idx] of a contiguous 2-D tensor.
        patch_mask = torch.zeros(total_patches, device=draw_device)
        patch_mask[patch_indices] = 1
        patch_mask = patch_mask.view(num_patches_h, num_patches_w)
        mask = patch_mask.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
        masks.append(mask[:height, :width])

    mask = torch.stack(masks, dim=0).unsqueeze(1) > 0.5
    if mask.shape[-2:] != (height, width):
        # A non-divisible extent leaves an unmasked border strip; pad it as kept pixels.
        padded = torch.zeros(batch, 1, height, width, dtype=torch.bool, device=draw_device)
        padded[..., : mask.shape[-2], : mask.shape[-1]] = mask
        mask = padded
    return mask.to(device, non_blocking=True)
