"""Data Adapters for Training Strategies.

This module provides adapters for converting between different data formats
used in the training pipeline, particularly TorchIO Subjects and PyTorch tensors.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class TorchIOAdapter:
    """Handles TorchIO Subject -> PyTorch Tensor conversions.

    TorchIO stores tensors in a specific format:
    - Subject["key"] returns an Image object
    - Image.data returns the actual tensor
    - Tensor format is (C, H, W, D) instead of PyTorch's (B, C, H, W)

    This adapter centralizes all conversion logic to avoid scattered
    reshaping code throughout the training strategies.
    """

    @staticmethod
    def extract_tensor(batch_data: Any, key: str) -> torch.Tensor | None:
        """Extract a tensor from TorchIO Subject or dict.

        Args:
            batch_data: TorchIO Subject, dict, or None
            key: Key to extract (e.g., "kspace", "target", "input")

        Returns:
            Tensor if found, None otherwise
        """
        if batch_data is None:
            return None

        try:
            # Try dict-like access
            obj = batch_data[key]

            # TorchIO Image objects have .data attribute
            if hasattr(obj, "data"):
                return obj.data

            # Direct tensor
            if isinstance(obj, torch.Tensor):
                return obj

            return None
        except (KeyError, TypeError, AttributeError):
            return None

    @staticmethod
    def to_batch_format(tensor: torch.Tensor, expected_channels: int = 2) -> torch.Tensor:
        """Convert TorchIO format (C, H, W, D) to PyTorch format (B, C, H, W).

        Handles various input shapes:
        - (C, H, W, 1): Single slice -> (1, C, H, W)
        - (C, H, W, D): Volume -> (D, C, H, W) where D becomes batch
        - (B, C, H, W): Already correct format -> unchanged
        - (B, C, H, W, D): 5D volume -> (B*D, C, H, W) flattened batch

        Args:
            tensor: Input tensor in TorchIO or PyTorch format
            expected_channels: Expected number of channels (default 2 for complex)

        Returns:
            Tensor in (B, C, H, W) format
        """
        if tensor is None:
            return None

        # Already in correct format: (B, C, H, W) where C == expected
        if tensor.ndim == 4 and tensor.shape[1] == expected_channels:
            return tensor

        if tensor.ndim == 5:
            # Standard 5D case: (B, C, H, W, D) or (B, C, H, W, Slices)
            b, c, h, w, d = tensor.shape

            # Case 1: Standard TorchIO batch with depth (B, C, H, W, D) -> Unpack D to batch
            # We assume for 2D models we want (Batch*Depth, Channels, Height, Width)
            if c == expected_channels:
                # Permute to (B, D, C, H, W) then flatten B*D
                tensor_reshaped = tensor.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
                return tensor_reshaped

            # Case 2: Channels last (B, H, W, D, C) -> rare but possible
            elif d == expected_channels:
                tensor_reshaped = tensor.permute(0, 4, 1, 2, 3).reshape(
                    b * d, expected_channels, h, w
                )
                return tensor_reshaped

            # Case 3: (B, C, H, W, D) where C > expected_channels or we have multi-coil complex arrays.
            # Flatten D into B; channel adjustment happens downstream.
            elif (c >= expected_channels or c % 2 == 0) and d > 1:
                logger.debug(
                    f"[TorchIOAdapter] 5D multi-channel input {tensor.shape} "
                    f"(c={c} > expected={expected_channels}): flattening D={d} into batch."
                )
                return tensor.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)

            # Fallback: unexpected layout, still flatten D into B but warn
            logger.warning(
                f"[TorchIOAdapter] Unhandled 5D shape: {tensor.shape}. Flattening D into B."
            )
            return tensor.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)

        # TorchIO format: (C, H, W, D) - SINGLE SUBJECT (no batch dim)
        if tensor.ndim == 4:
            # Single slice: (C, H, W, 1)
            if tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)  # (C, H, W)
                if tensor.ndim == 3:
                    tensor = tensor.unsqueeze(0)  # (1, C, H, W)
                return tensor

            # Volume: (C, H, W, D) where D > 1 -> Treat D as batch
            if tensor.shape[0] == expected_channels:
                # Permute to (D, C, H, W)
                return tensor.permute(3, 0, 1, 2)

            # Channels last: (B, H, W, C)
            if tensor.shape[-1] == expected_channels:
                return tensor.permute(0, 3, 1, 2)

        # 3D tensor: (C, H, W) or (H, W, C)
        if tensor.ndim == 3:
            if tensor.shape[0] == expected_channels:
                return tensor.unsqueeze(0)  # (1, C, H, W)
            if tensor.shape[-1] == expected_channels:
                return tensor.permute(2, 0, 1).unsqueeze(0)

        logger.warning(f"[TorchIOAdapter] Unexpected tensor shape: {tensor.shape}")
        return tensor

    @staticmethod
    def ensure_channel_match(
        prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ensure prediction and target have compatible channel counts.

        Converts between 2-channel complex and 1-channel magnitude as needed.

        Args:
            prediction: Model output tensor
            target: Ground truth tensor

        Returns:
            Tuple of (prediction, target) with matching channels
        """
        pred_ch = prediction.shape[1]
        tgt_ch = target.shape[1]

        if pred_ch == tgt_ch:
            return prediction, target

        # Prediction is 2ch (complex), Target is 1ch (magnitude)
        # -> Convert prediction to magnitude
        if pred_ch == 2 and tgt_ch == 1:
            real = prediction[:, 0:1]
            imag = prediction[:, 1:2]
            prediction = torch.sqrt(real**2 + imag**2 + 1e-8)
            logger.debug("[TorchIOAdapter] Converted 2ch prediction to magnitude")

        # Target is 2ch (complex), Prediction is 1ch (magnitude)
        # -> Convert target to magnitude
        elif tgt_ch == 2 and pred_ch == 1:
            real = target[:, 0:1]
            imag = target[:, 1:2]
            target = torch.sqrt(real**2 + imag**2 + 1e-8)
            logger.debug("[TorchIOAdapter] Converted 2ch target to magnitude")

        # Target is 4ch (2 coils, split real/imag), Prediction is 2ch (1 coil complex-stacked)
        # -> Convert target to 2ch RSS magnitude/phase or just RSS over coils
        elif tgt_ch == 4 and pred_ch == 2:
            # Reconstruct two complex coils
            coil1_real, coil1_imag = target[:, 0:1], target[:, 1:2]
            coil2_real, coil2_imag = target[:, 2:3], target[:, 3:4]
            coil1 = torch.complex(coil1_real, coil1_imag)
            coil2 = torch.complex(coil2_real, coil2_imag)

            # RSS combination
            target_mag = torch.sqrt(torch.abs(coil1) ** 2 + torch.abs(coil2) ** 2 + 1e-8)

            # Since prediction is 2ch (complex-stacked), we convert target_mag
            # into a 2-channel representation (real=mag, imag=0) to strictly match
            target = torch.cat([target_mag, torch.zeros_like(target_mag)], dim=1)
            logger.debug("[TorchIOAdapter] Converted 4ch target to 2ch (RSS real-stacked)")

        elif pred_ch == 4 and tgt_ch == 2:
            coil1_real, coil1_imag = prediction[:, 0:1], prediction[:, 1:2]
            coil2_real, coil2_imag = prediction[:, 2:3], prediction[:, 3:4]
            coil1 = torch.complex(coil1_real, coil1_imag)
            coil2 = torch.complex(coil2_real, coil2_imag)
            pred_mag = torch.sqrt(torch.abs(coil1) ** 2 + torch.abs(coil2) ** 2 + 1e-8)
            prediction = torch.cat([pred_mag, torch.zeros_like(pred_mag)], dim=1)
            logger.debug("[TorchIOAdapter] Converted 4ch prediction to 2ch (RSS real-stacked)")

        elif pred_ch == 8 and tgt_ch == 4:
            # Reconstruct prediction (4 complex coils) -> RSS
            c1_real, c1_imag = prediction[:, 0:1], prediction[:, 1:2]
            c2_real, c2_imag = prediction[:, 2:3], prediction[:, 3:4]
            c3_real, c3_imag = prediction[:, 4:5], prediction[:, 5:6]
            c4_real, c4_imag = prediction[:, 6:7], prediction[:, 7:8]
            c1 = torch.complex(c1_real, c1_imag)
            c2 = torch.complex(c2_real, c2_imag)
            c3 = torch.complex(c3_real, c3_imag)
            c4 = torch.complex(c4_real, c4_imag)
            pred_mag = torch.sqrt(
                torch.abs(c1) ** 2
                + torch.abs(c2) ** 2
                + torch.abs(c3) ** 2
                + torch.abs(c4) ** 2
                + 1e-8
            )
            prediction = torch.cat([pred_mag, torch.zeros_like(pred_mag)], dim=1)

            # Reconstruct target (2 complex coils) -> RSS
            t1_real, t1_imag = target[:, 0:1], target[:, 1:2]
            t2_real, t2_imag = target[:, 2:3], target[:, 3:4]
            t1 = torch.complex(t1_real, t1_imag)
            t2 = torch.complex(t2_real, t2_imag)
            target_mag = torch.sqrt(torch.abs(t1) ** 2 + torch.abs(t2) ** 2 + 1e-8)
            target = torch.cat([target_mag, torch.zeros_like(target_mag)], dim=1)

            logger.debug("[TorchIOAdapter] Converted 8ch pred / 4ch target to 2ch RSS")

        elif pred_ch < tgt_ch and tgt_ch % pred_ch == 0:
            # Generic case: target has more channels than prediction (cross-contrast).
            # Take LAST channels = target contrast (M4Raw stacks [source|target]).
            target = target[:, -pred_ch:]
            logger.info(
                f"[TorchIOAdapter] Sliced target {tgt_ch}ch → last {pred_ch}ch (target contrast)"
            )

        elif pred_ch > tgt_ch and pred_ch % tgt_ch == 0:
            # Generic case: prediction has more channels than target.
            # Truncate prediction to match target channel count.
            prediction = prediction[:, :tgt_ch]
            logger.info(
                f"[TorchIOAdapter] Truncated prediction {pred_ch}ch → {tgt_ch}ch to match target"
            )

        else:
            # Last resort: truncate to minimum
            min_ch = min(pred_ch, tgt_ch)
            prediction = prediction[:, :min_ch]
            target = target[:, :min_ch]
            logger.warning(
                f"[TorchIOAdapter] Channel mismatch: pred={pred_ch}, target={tgt_ch}. "
                f"Truncated both to {min_ch}ch."
            )

        return prediction, target
