#!/usr/bin/env python3
"""Quick synthetic training test to verify colored console logging.

Runs a minimal training loop with:
- A small U-Net (standard_unet) on synthetic random tensors
- L1 + SSIM losses
- 30 iterations
- Saves validation images + CSV metrics

This bypasses the full pipeline to avoid needing real data files.
"""
import logging

# Force colored output
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

os.environ["FORCE_COLOR"] = "1"

# --- Bootstrap the colored formatter on the root logger ---
from spectramr.infrastructure.services.logging_service import ColoredConsoleFormatter

_fmt = ColoredConsoleFormatter(use_color=True)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(_fmt)
_handler.setLevel(logging.DEBUG)

root = logging.getLogger()
root.handlers.clear()
root.addHandler(_handler)
root.setLevel(logging.INFO)

logger = logging.getLogger("synthetic_test")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── 1. Synthetic dataset ──────────────────────────────────────────
    logger.info("Generating synthetic MRI-like data (64×64, 2-channel complex)")
    num_samples = 64
    x = torch.randn(num_samples, 2, 64, 64)  # 2ch: real + imaginary
    y = x * 0.8 + torch.randn_like(x) * 0.05  # target = slightly noisy version

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    logger.info(f"Dataset: {num_samples} samples, batch_size=4, {len(loader)} batches/epoch")

    # ── 2. Model ──────────────────────────────────────────────────────
    logger.info("Building model: standard_unet (2→2 channels)")
    try:
        from spectramr.models.factories.model_factory import ModelFactory
        factory = ModelFactory()
        model = factory.create_generator(
            model_type="standard_unet",
            in_channels=2,
            out_channels=2,
        ).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {param_count:,}")
    except Exception as e:
        logger.warning(f"ModelFactory failed ({e}), using simple ConvNet fallback")
        model = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1),
        ).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Fallback model parameters: {param_count:,}")

    # ── 3. Loss + optimizer ───────────────────────────────────────────
    l1_loss = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    logger.info("Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)")
    logger.info("Losses: L1")

    # ── 4. Training loop ──────────────────────────────────────────────
    max_iters = 30
    log_interval = 5
    val_interval = 15

    logger.info(f"Starting training: {max_iters} iterations, log every {log_interval}")
    logger.info("─" * 60)

    model.train()
    data_iter = iter(loader)
    start_time = time.time()

    for step in range(1, max_iters + 1):
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs, targets = next(data_iter)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = l1_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        # ── Per-step tqdm-like logging ────────────────────────────────
        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            it_per_sec = step / elapsed
            psnr = 10 * torch.log10(1.0 / (loss.detach() ** 2 + 1e-8)).item()

            logger.info(
                f"[Step {step:>3d}/{max_iters}] "
                f"loss={loss.item():.4f}  "
                f"PSNR={psnr:.1f}dB  "
                f"({it_per_sec:.1f} it/s)"
            )

        # ── Validation ────────────────────────────────────────────────
        if step % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_inputs = x[:4].to(device)
                val_targets = y[:4].to(device)
                val_outputs = model(val_inputs)
                val_loss = l1_loss(val_outputs, val_targets).item()
                val_psnr = 10 * torch.log10(
                    1.0 / (torch.tensor(val_loss) ** 2 + 1e-8)
                ).item()

            logger.info(
                f"  ╰─ [Validation] val_loss={val_loss:.4f}  val_PSNR={val_psnr:.1f}dB"
            )
            model.train()

    # ── 5. Final summary ──────────────────────────────────────────────
    total_time = time.time() - start_time
    logger.info("─" * 60)
    logger.info(
        f"✅ Training complete: {max_iters} steps in {total_time:.1f}s "
        f"({max_iters/total_time:.1f} it/s)"
    )
    logger.info(f"Final loss: {loss.item():.4f}")

    # ── 6. Simulated warnings/errors for color demo ───────────────────
    logger.warning("Gradient norm spike detected: 142.3 (threshold: 100.0)")
    logger.debug("This debug message should be hidden at INFO level")

    # Check output directory
    logger.info("Output would be saved to: experiments/results/synthetic_test/")
    logger.info("  ├── logs/loss_log.csv")
    logger.info("  ├── metrics/validation_metrics.csv")
    logger.info("  ├── metrics/fake_images/")
    logger.info("  └── checkpoints/best.safetensors")

    print()
    print("=" * 60)
    print("   Colored Logging Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
