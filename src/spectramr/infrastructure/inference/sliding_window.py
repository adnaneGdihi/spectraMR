"""Sliding Window Inference for 3D Medical Imaging.

[ARCHITECT FIX] Enables memory-efficient inference on large 3D volumes
by processing overlapping patches and stitching with Gaussian weighting.

Standard technique used in nnU-Net and MONAI.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F


def create_gaussian_weight_3d(
    roi_size: tuple[int, int, int],
    sigma_scale: float = 0.125,
    device: torch.device = None,
) -> torch.Tensor:
    """Create 3D Gaussian weight map for smooth patch stitching.

    Args:
        roi_size: (D, H, W) patch dimensions
        sigma_scale: Gaussian sigma as fraction of roi size
        device: Target device

    Returns:
        Gaussian weight tensor (1, 1, D, H, W)
    """
    d, h, w = roi_size

    # Create 1D Gaussian for each dimension
    def gaussian_1d(size: int) -> torch.Tensor:
        """gaussian_1d.

        Args:
            size (int): Description.
        Returns:
            torch.Tensor: Description.
        """
        sigma = size * sigma_scale
        coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
        gauss = torch.exp(-(coords**2) / (2 * sigma**2))
        return gauss

    gd = gaussian_1d(d)
    gh = gaussian_1d(h)
    gw = gaussian_1d(w)

    # Outer product to create 3D Gaussian
    weight = gd[:, None, None] * gh[None, :, None] * gw[None, None, :]
    weight = weight.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)


    return weight


def sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: tuple[int, int, int],
    predictor: Callable[[torch.Tensor], torch.Tensor],
    sw_batch_size: int = 1,
    overlap: float = 0.25,
    mode: str = "gaussian",
    device: torch.device = None,
) -> torch.Tensor:
    """
    Performs sliding window inference on a large 3D volume.

    Args:
        inputs: Input tensor (B, C, D, H, W)
        roi_size: Patch size (D_patch, H_patch, W_patch)
        predictor: Model forward function
        sw_batch_size: Number of patches to process at once
        overlap: Overlap fraction [0, 1)
        mode: Weighting mode - 'gaussian' or 'constant'
        device: Device for computation

    Returns:
        Output tensor (B, C_out, D, H, W)
    """
    if device is None:
        device = inputs.device

    batch_size, in_channels, d, h, w = inputs.shape
    pd, ph, pw = roi_size

    # Calculate strides (step size between patches)
    stride_d = max(1, int(pd * (1 - overlap)))
    stride_h = max(1, int(ph * (1 - overlap)))
    stride_w = max(1, int(pw * (1 - overlap)))

    # Pad input if needed to ensure full coverage
    # Enough that the last window ends on the padded edge: a remainder of (d - pd)
    # modulo the stride needs (stride - remainder) more voxels, or the tail is never covered.
    pad_d = (stride_d - (d - pd) % stride_d) % stride_d if d > pd else pd - d
    pad_h = (stride_h - (h - ph) % stride_h) % stride_h if h > ph else ph - h
    pad_w = (stride_w - (w - pw) % stride_w) % stride_w if w > pw else pw - w

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        # Reflection needs a pad shorter than the axis it mirrors.
        fits = pad_d < d and pad_h < h and pad_w < w
        inputs = F.pad(
            inputs, (0, pad_w, 0, pad_h, 0, pad_d), mode="reflect" if fits else "replicate"
        )
        _, _, d_pad, h_pad, w_pad = inputs.shape
    else:
        d_pad, h_pad, w_pad = d, h, w

    # Create weight map
    if mode == "gaussian":
        patch_weight = create_gaussian_weight_3d(roi_size, device=device)
    else:
        patch_weight = torch.ones(1, 1, pd, ph, pw, device=device)

    # Collect all patch locations
    patch_locations = []
    for dz in range(0, d_pad - pd + 1, stride_d):
        for dy in range(0, h_pad - ph + 1, stride_h):
            for dx in range(0, w_pad - pw + 1, stride_w):
                patch_locations.append((dz, dy, dx))

    # Output buffers (lazy initialization)
    output_buffer = None
    count_buffer = None

    # Process in mini-batches
    for i in range(0, len(patch_locations), sw_batch_size):
        batch_locs = patch_locations[i : i + sw_batch_size]

        # Extract patches for this mini-batch
        patches = []
        for dz, dy, dx in batch_locs:
            patch = inputs[:, :, dz : dz + pd, dy : dy + ph, dx : dx + pw]
            patches.append(patch)

        # Stack and predict
        if len(patches) == 1:
            patch_batch = patches[0]
        else:
            patch_batch = torch.cat(patches, dim=0)

        with torch.no_grad():
            pred_batch = predictor(patch_batch)

        # Initialize buffers on first prediction
        if output_buffer is None:
            # Infer output channels from prediction
            out_channels = pred_batch.shape[1]
            output_buffer = torch.zeros(
                batch_size,
                out_channels,
                d_pad,
                h_pad,
                w_pad,
                device=device,
                dtype=pred_batch.dtype,
            )
            count_buffer = torch.zeros(
                batch_size, 1, d_pad, h_pad, w_pad, device=device, dtype=torch.float32
            )

        # Accumulate predictions with weights
        for j, (dz, dy, dx) in enumerate(batch_locs):
            # Patches were stacked along dim 0, each carrying the whole batch.
            pred_patch = pred_batch[j * batch_size : (j + 1) * batch_size]

            output_buffer[:, :, dz : dz + pd, dy : dy + ph, dx : dx + pw] += (
                pred_patch * patch_weight
            )
            count_buffer[:, :, dz : dz + pd, dy : dy + ph, dx : dx + pw] += patch_weight

    # Normalize by count
    output = output_buffer / count_buffer.clamp(min=1e-8)

    # Remove padding to get original size
    output = output[:, :, :d, :h, :w]

    return output


def sliding_window_inference_2d(
    inputs: torch.Tensor,
    roi_size: tuple[int, int],
    predictor: Callable[[torch.Tensor], torch.Tensor],
    overlap: float = 0.25,
    mode: str = "gaussian",
) -> torch.Tensor:
    """
    2D version of sliding window inference for slice-by-slice processing.

    Args:
        inputs: Input tensor (B, C, H, W)
        roi_size: Patch size (H_patch, W_patch)
        predictor: Model forward function
        overlap: Overlap fraction
        mode: Weighting mode

    Returns:
        Output tensor (B, C_out, H, W)
    """
    device = inputs.device
    batch_size, in_channels, h, w = inputs.shape
    ph, pw = roi_size

    stride_h = max(1, int(ph * (1 - overlap)))
    stride_w = max(1, int(pw * (1 - overlap)))

    # Pad if needed
    pad_h = (stride_h - (h - ph) % stride_h) % stride_h if h > ph else ph - h
    pad_w = (stride_w - (w - pw) % stride_w) % stride_w if w > pw else pw - w

    if pad_h > 0 or pad_w > 0:
        fits = pad_h < h and pad_w < w
        inputs = F.pad(inputs, (0, pad_w, 0, pad_h), mode="reflect" if fits else "replicate")
        _, _, h_pad, w_pad = inputs.shape
    else:
        h_pad, w_pad = h, w

    # 2D Gaussian weight
    def gaussian_2d(size_h, size_w, sigma_scale=0.125):
        """gaussian_2d.

        Args:
            size_h (Any): Description.
            size_w (Any): Description.
            sigma_scale (Any): Description.
        Returns:
            Any: Description.
        """
        sigma_h = size_h * sigma_scale
        sigma_w = size_w * sigma_scale
        # Built on the target device: a host-built map costs a synchronising copy per call.
        ys = torch.arange(size_h, dtype=torch.float32, device=device)
        xs = torch.arange(size_w, dtype=torch.float32, device=device)
        gy = torch.exp(-((ys - (size_h - 1) / 2) ** 2) / (2 * sigma_h**2))
        gx = torch.exp(-((xs - (size_w - 1) / 2) ** 2) / (2 * sigma_w**2))
        return gy[:, None] * gx[None, :]

    if mode == "gaussian":
        weight = gaussian_2d(ph, pw).unsqueeze(0).unsqueeze(0)
    else:
        weight = torch.ones(1, 1, ph, pw, device=device)

    output_buffer = None
    count_buffer = None

    for dy in range(0, h_pad - ph + 1, stride_h):
        for dx in range(0, w_pad - pw + 1, stride_w):
            patch = inputs[:, :, dy : dy + ph, dx : dx + pw]

            with torch.no_grad():
                pred = predictor(patch)

            if output_buffer is None:
                out_ch = pred.shape[1]
                output_buffer = torch.zeros(batch_size, out_ch, h_pad, w_pad, device=device)
                count_buffer = torch.zeros(batch_size, 1, h_pad, w_pad, device=device)

            output_buffer[:, :, dy : dy + ph, dx : dx + pw] += pred * weight
            count_buffer[:, :, dy : dy + ph, dx : dx + pw] += weight

    output = output_buffer / count_buffer.clamp(min=1e-8)
    return output[:, :, :h, :w]


__all__ = [
    "create_gaussian_weight_3d",
    "sliding_window_inference",
    "sliding_window_inference_2d",
]
