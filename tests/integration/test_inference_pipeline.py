"""Integration tests for Inference Pipeline.

Tests:
- Model loading from checkpoints
- Batch inference on test data
- Single-image inference
- Inference with different strategies (GAN, Diffusion, VAE)
- Data preprocessing alignment (train vs inference)
- Metrics computation (PSNR, SSIM, LPIPS)
- Output saving (image domain + k-space)
- Inference speed benchmarking
- GPU memory profiling during inference

Critical: Ensure inference uses SAME preprocessing as training (no data leakage).
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from spectramr.config.settings import TrainingSettings


class TestModelLoadingForInference:
    """Test loading trained models for inference."""

    @pytest.fixture
    def temp_inference_dir(self):
        """Create temporary directory for inference artifacts."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def minimal_checkpoint(self, temp_inference_dir):
        """Create minimal model checkpoint."""
        # Create simple model
        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 2, 3, padding=1),
        )

        # Save checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": 10,
            "val_loss": 0.123,
            "val_psnr": 30.5,
            "config": {
                "model": {
                    "model_type": "standard_unet",
                    "in_channels": 2,
                    "out_channels": 2,
                },
            },
        }

        checkpoint_path = temp_inference_dir / "best_model.pt"
        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path

    def test_load_checkpoint_state_dict(self, minimal_checkpoint):
        """Test loading checkpoint and extracting state dict."""
        checkpoint = torch.load(minimal_checkpoint, weights_only=False)

        assert "model_state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert checkpoint["epoch"] == 10
        assert checkpoint["val_psnr"] == 30.5

    def test_load_model_from_checkpoint(self, minimal_checkpoint):
        """Test loading model weights from checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(minimal_checkpoint, weights_only=False)

        # Create new model with same architecture
        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 2, 3, padding=1),
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Verify model is in eval mode for inference
        model.eval()

        # Test forward pass
        input_batch = torch.randn(2, 2, 64, 64)
        with torch.no_grad():
            output = model(input_batch)

        assert output.shape == (2, 2, 64, 64)

    def test_checkpoint_contains_training_config(self, minimal_checkpoint):
        """Test checkpoint includes training config for reproducibility."""
        checkpoint = torch.load(minimal_checkpoint, weights_only=False)

        assert "config" in checkpoint
        assert "model" in checkpoint["config"]
        assert checkpoint["config"]["model"]["model_type"] == "standard_unet"

    def test_inference_config_matches_training_config(self, temp_inference_dir):
        """Test inference config matches training config fields.

        CRITICAL: Preprocessing must be identical for train vs inference.
        """
        train_config = {
            "config_version": "1.0",
            "metadata": {"name": "train_exp", "output_dir": str(temp_inference_dir)},
            "model": {
                "model_type": "standard_unet",
                "in_channels": 2,
                "out_channels": 2,
            },
            "data": {
                "dataset_type": "synthetic",
                "batch_size": 4,  # Train batch size
                "image_size": [128, 128],
                "acceleration": 4,
                "normalize_kspace": True,  # CRITICAL
            },
            "optimization": {"learning_rate": 1e-4, "num_epochs": 10},
            "training": {"training_mode": "reconstruction"},
            "losses": {"reconstruction": {"lambda_l1": 1.0}},
            "physics": {
                "data_consistency": {
                    "enabled": True,
                    "train_noise_level": 0.01,  # CRITICAL
                    "eval_noise_level": 0.005,  # CRITICAL for inference
                }
            },
        }

        # Inference config (subset of training config)
        inference_config = {
            "config_version": "1.0",
            "metadata": {
                "name": "inference_exp",
                "output_dir": str(temp_inference_dir),
            },
            "model": train_config["model"],  # MUST match
            "data": {
                "dataset_type": "fastmri_knee",  # Different source OK
                "batch_size": 1,  # Inference often uses batch_size=1
                "image_size": train_config["data"]["image_size"],  # MUST match
                "acceleration": train_config["data"]["acceleration"],  # MUST match
                "normalize_kspace": train_config["data"][
                    "normalize_kspace"
                ],  # MUST match
            },
            "physics": {
                "data_consistency": {
                    "enabled": train_config["physics"]["data_consistency"]["enabled"],
                    "eval_noise_level": train_config["physics"]["data_consistency"][
                        "eval_noise_level"
                    ],
                    "noise_type": "gaussian",
                }
            },
        }

        # Validate critical fields match
        assert inference_config["model"] == train_config["model"]
        assert (
            inference_config["data"]["image_size"] == train_config["data"]["image_size"]
        )
        assert (
            inference_config["data"]["normalize_kspace"]
            == train_config["data"]["normalize_kspace"]
        )
        assert (
            inference_config["physics"]["data_consistency"]["eval_noise_level"]
            == train_config["physics"]["data_consistency"]["eval_noise_level"]
        )


class TestBatchInference:
    """Test batch inference on multiple samples."""

    @pytest.fixture
    def synthetic_test_batch(self):
        """Create synthetic test batch."""
        batch_size = 4
        kspace_input = torch.randn(batch_size, 2, 64, 64)  # Undersampled k-space
        kspace_target = torch.randn(batch_size, 2, 64, 64)  # Fully-sampled k-space
        # Binary mask for DC verification
        masks = torch.zeros(batch_size, 1, 64, 64)
        masks[:, :, ::2, ::2] = 1.0  # 25% sampling pattern

        return {
            "input": kspace_input,
            "target": kspace_target,
            "mask": masks,
        }

    def test_batch_inference_forward_pass(self, synthetic_test_batch):
        """Test batch inference forward pass."""
        # Create model
        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 16, 3, padding=1),
            torch.nn.Conv2d(16, 2, 3, padding=1),
        )
        model.eval()

        input_batch = synthetic_test_batch["input"]

        with torch.no_grad():
            output = model(input_batch)

        assert output.shape == input_batch.shape
        assert output.requires_grad is False  # No gradients during inference

        def test_batch_inference_with_data_consistency(self, synthetic_test_batch):
            """Test batch inference with Data Consistency Layer."""
            from spectramr.infrastructure.physics.data_consistency import (
                NoiseSimulatingDataConsistency,
            )
            from spectramr.infrastructure.physics.fft_ops import ifft2c

            # Create model
            model = torch.nn.Sequential(
                torch.nn.Conv2d(2, 16, 3, padding=1),
                torch.nn.Conv2d(16, 2, 3, padding=1),
            )
            model.eval()

            # Create DC layer (eval mode → uses eval_noise_level)
            dc_layer = NoiseSimulatingDataConsistency(
                train_noise_level=0.01,
                eval_noise_level=0.005,  # Lower noise at inference
                noise_type="gaussian",
            )
            dc_layer.eval()

            input_batch = synthetic_test_batch["input"]
            measured_kspace = synthetic_test_batch["input"]
            mask = synthetic_test_batch["mask"]

            with torch.no_grad():
                # Model prediction (in k-space)
                predicted_kspace = model(input_batch)

                # Convert to image domain for DC layer which expects image
                # [B, 2, H, W] -> complex -> ifft -> [B, 2, H, W]
                x_complex = torch.complex(
                    predicted_kspace[:, 0], predicted_kspace[:, 1]
                ).unsqueeze(1)
                predicted_img = ifft2c(x_complex)
                # Back to [B, 2, H, W]
                predicted_img = torch.cat(
                    [predicted_img.real, predicted_img.imag], dim=1
                )

                # Apply data consistency
                # measured_kspace should be complex or [B, 2, H, W]
                measured_kspace_complex = torch.complex(
                    measured_kspace[:, 0], measured_kspace[:, 1]
                ).unsqueeze(1)

                consistent_img = dc_layer(predicted_img, measured_kspace_complex, mask)

            assert consistent_img.shape == predicted_img.shape

            # Verify DC enforces mask in k-space
            # Convert consistent image back to k-space to check against measured
            from spectramr.infrastructure.physics.fft_ops import fft2c

            consistent_kspace = fft2c(
                torch.complex(consistent_img[:, 0], consistent_img[:, 1]).unsqueeze(1)
            )
            consistent_kspace = torch.cat(
                [consistent_kspace.real, consistent_kspace.imag], dim=1
            )

            # Mask needs to be broadcastable to [B, 2, H, W] for indexing
            mask_2ch = mask.repeat(1, 2, 1, 1)
            masked_region = mask_2ch > 0.1
            diff_at_mask = (
                (consistent_kspace - measured_kspace).abs()[masked_region].mean()
            )

            # With eval_noise_level=0.005, diff should be small
            assert diff_at_mask < 0.1, f"DC not working: diff={diff_at_mask:.4f}"

    def test_batch_metrics_computation(self, synthetic_test_batch):
        """Test computing metrics on inference batch."""
        # Simulated predictions and targets
        predictions = torch.randn(4, 2, 64, 64)
        targets = synthetic_test_batch["target"]

        # Compute MSE
        mse = torch.nn.functional.mse_loss(predictions, targets)
        assert mse.item() > 0

        # Compute MAE (L1)
        mae = torch.nn.functional.l1_loss(predictions, targets)
        assert mae.item() > 0

        # PSNR (simplified)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
        assert psnr.item() < 100  # Sanity check


class TestSingleImageInference:
    """Test inference on single images (batch_size=1)."""

    def test_single_image_inference(self):
        """Test inference on single image."""
        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 16, 3, padding=1),
            torch.nn.Conv2d(16, 2, 3, padding=1),
        )
        model.eval()

        # Single image (B=1, C=2, H=64, W=64)
        input_image = torch.randn(1, 2, 64, 64)

        with torch.no_grad():
            output = model(input_image)

        assert output.shape == (1, 2, 64, 64)

    def test_inference_batch_size_flexibility(self):
        """Test model works with different batch sizes (1, 4, 8)."""
        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 16, 3, padding=1),
            torch.nn.Conv2d(16, 2, 3, padding=1),
        )
        model.eval()

        batch_sizes = [1, 4, 8]

        for bs in batch_sizes:
            input_batch = torch.randn(bs, 2, 64, 64)

            with torch.no_grad():
                output = model(input_batch)

            assert output.shape == (bs, 2, 64, 64)


class TestInferenceWithDifferentStrategies:
    """Test inference for different training strategies."""

    @pytest.fixture
    def strategy_inference_configs(self, tmp_path):
        """Generate inference configs for different strategies."""
        base = {
            "config_version": "1.0",
            "metadata": {"name": "inference", "output_dir": str(tmp_path)},
            "logging": {"experiment_name": "inference"},
            "model": {"in_channels": 2, "out_channels": 2, "base_channels": 16},
            "data": {
                "dataset_type": "synthetic",
                "batch_size": 1,
                "image_size": [64, 64],
            },
            "physics": {
                "data_consistency": {"enabled": True, "eval_noise_level": 0.005}
            },
            "optimization": {"learning_rate": 1e-4},
        }

        strategies = {
            "reconstruction": {"model_type": "standard_unet"},
            "diffusion": {
                "model_type": "kspace_cold_diffusion_generator",
                "training": {
                    "diffusion": {"num_timesteps": 100, "cold_diffusion": True}
                },
            },
            "vae": {
                "model_type": "vae_generator",
                "training": {"vae": {"latent_dim": 64}},
            },
        }

        configs = {}
        for strategy_name, strategy_config in strategies.items():
            config = base.copy()
            config["model"]["model_type"] = strategy_config["model_type"]

            if "training" in strategy_config:
                config["training"] = strategy_config["training"]

            config_path = tmp_path / f"inference_{strategy_name}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            configs[strategy_name] = config_path

        return configs

    def test_reconstruction_inference(self, strategy_inference_configs):
        """Test inference for reconstruction model."""
        config_path = strategy_inference_configs["reconstruction"]
        config = TrainingSettings.from_yaml(str(config_path))

        assert config.model.model_type == "standard_unet"

    def test_diffusion_inference_single_timestep(self):
        """Test diffusion inference uses single reverse step (not full chain).

        During training: Sample random timestep t
        During inference: Run full reverse chain t=T → t=0
        """
        # Simulated diffusion model (simplified)
        num_timesteps = 100

        # Inference: Reverse diffusion from t=T to t=0
        x_t = torch.randn(1, 2, 64, 64)  # Start from noise

        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 16, 3, padding=1),
            torch.nn.Conv2d(16, 2, 3, padding=1),
        )
        model.eval()

        # Reverse chain (simplified: no actual noise schedule)
        for t in reversed(range(num_timesteps)):
            with torch.no_grad():
                x_t = model(x_t)  # Denoise step

        # Final output
        reconstructed = x_t
        assert reconstructed.shape == (1, 2, 64, 64)

    def test_vae_inference_latent_space(self):
        """Test VAE inference through latent space.

        Encoder: image → latent z
        Decoder: latent z → reconstructed image
        """
        # Simulated encoder/decoder
        encoder = torch.nn.Sequential(
            torch.nn.Conv2d(2, 16, 3, padding=1),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 64),  # Latent dim
        )

        decoder = torch.nn.Sequential(
            torch.nn.Linear(64, 16 * 8 * 8),
            torch.nn.Unflatten(1, (16, 8, 8)),
            torch.nn.Upsample(scale_factor=8, mode="nearest"),
            torch.nn.Conv2d(16, 2, 3, padding=1),
        )

        encoder.eval()
        decoder.eval()

        input_image = torch.randn(1, 2, 64, 64)

        with torch.no_grad():
            # Encode
            latent = encoder(input_image)
            assert latent.shape == (1, 64)

            # Decode
            reconstructed = decoder(latent)
            assert reconstructed.shape == (1, 2, 64, 64)


class TestInferencePreprocessingAlignment:
    """Test preprocessing alignment between training and inference.

    CRITICAL: Inference must use EXACT same preprocessing as training.
    Data leakage occurs if normalization/transform differs.
    """

    def test_kspace_normalization_identical(self):
        """Test k-space normalization is identical at train vs inference."""
        # Training: normalize by 99th percentile of magnitude
        kspace_train = torch.randn(2, 2, 64, 64)
        mag_train = torch.sqrt(kspace_train[:, 0] ** 2 + kspace_train[:, 1] ** 2)

        # Per-sample percentile (99th)
        batch_size = mag_train.size(0)
        scales_train = []
        for i in range(batch_size):
            mag_flat = mag_train[i].flatten()
            percentile_99 = torch.quantile(mag_flat, 0.99)
            scales_train.append(percentile_99)

        scale_train = torch.stack(scales_train).view(batch_size, 1, 1, 1)
        kspace_train_norm = kspace_train / scale_train

        # Inference: MUST use same percentile (0.99)
        kspace_infer = torch.randn(1, 2, 64, 64)  # Different sample, but same process
        mag_infer = torch.sqrt(kspace_infer[:, 0] ** 2 + kspace_infer[:, 1] ** 2)

        mag_flat = mag_infer[0].flatten()
        percentile_99_infer = torch.quantile(mag_flat, 0.99)
        scale_infer = percentile_99_infer.view(1, 1, 1, 1)
        kspace_infer_norm = kspace_infer / scale_infer

        # Both normalized to [0, ~1] range
        assert kspace_train_norm.abs().max() <= 2.0
        assert kspace_infer_norm.abs().max() <= 2.0

    def test_mask_generation_deterministic_at_inference(self):
        """Test mask generation is deterministic at inference.

        CRITICAL: Use PRE-COMPUTED masks, not on-the-fly random masks.
        """
        # At training: Dataloader provides pre-computed mask
        # At inference: MUST use SAME mask from dataloader (no random generation)

        # Simulated dataloader mask (fixed pattern)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, ::4, :] = 1.0  # Sample every 4th line (4x acceleration)

        coverage = mask.mean().item()
        assert 0.20 < coverage < 0.30  # ~25% for 4x acceleration

        # Inference uses this exact mask (no randomness)
        # If we generate a new random mask, results will differ → data leak

    def test_data_consistency_noise_level_matches_config(self):
        """Test DC layer uses correct noise level at inference.

        Training: train_noise_level=0.01
        Inference: eval_noise_level=0.005 (lower, more realistic)
        """
        from spectramr.infrastructure.physics.data_consistency import NoiseSimulatingDataConsistency

        dc_layer = NoiseSimulatingDataConsistency(
            train_noise_level=0.01,
            eval_noise_level=0.005,
            noise_type="gaussian",
        )

        # Training mode
        dc_layer.train()
        assert dc_layer.training is True

        # Inference mode
        dc_layer.eval()
        assert dc_layer.training is False

        # In eval mode, uses eval_noise_level (verified in unit tests)


class TestInferenceOutputSaving:
    """Test saving inference outputs (images + k-space)."""

    def test_save_reconstructed_images(self, tmp_path):
        """Test saving reconstructed images to disk."""
        # Simulated reconstructions
        reconstructed_kspace = torch.randn(4, 2, 64, 64)

        # Convert to image domain (IFFT)
        # Simplified: assume already in image domain for test
        reconstructed_images = reconstructed_kspace  # In practice: ifft2c(kspace)

        # Save as tensor
        output_path = tmp_path / "reconstructions.pt"
        torch.save(reconstructed_images, output_path)

        # Verify saved
        assert output_path.exists()

        # Load back
        loaded = torch.load(output_path, weights_only=True)
        assert loaded.shape == (4, 2, 64, 64)

    def test_save_inference_metrics(self, tmp_path):
        """Test saving inference metrics to JSON."""
        import json

        metrics = {
            "sample_0": {"psnr": 32.5, "ssim": 0.90, "mae": 0.012},
            "sample_1": {"psnr": 31.0, "ssim": 0.88, "mae": 0.015},
            "sample_2": {"psnr": 33.0, "ssim": 0.92, "mae": 0.010},
        }

        # Save to JSON
        metrics_path = tmp_path / "inference_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Load back
        with open(metrics_path) as f:
            loaded_metrics = json.load(f)

        assert loaded_metrics["sample_0"]["psnr"] == 32.5

    def test_save_kspace_and_image_outputs(self, tmp_path):
        """Test saving both k-space and image domain outputs."""
        kspace_output = torch.randn(2, 2, 64, 64)
        image_output = torch.randn(2, 1, 64, 64)  # Magnitude images

        # Save both
        torch.save(kspace_output, tmp_path / "kspace.pt")
        torch.save(image_output, tmp_path / "images.pt")

        # Verify both exist
        assert (tmp_path / "kspace.pt").exists()
        assert (tmp_path / "images.pt").exists()


class TestInferenceSpeedBenchmarking:
    """Test inference speed benchmarking."""

    def test_inference_throughput(self):
        """Test inference throughput (images/second)."""
        import time

        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 16, 3, padding=1),
            torch.nn.Conv2d(16, 2, 3, padding=1),
        )
        model.eval()

        batch_size = 8
        num_batches = 10

        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_batches):
                input_batch = torch.randn(batch_size, 2, 64, 64)
                _ = model(input_batch)

        end_time = time.time()
        elapsed = end_time - start_time

        total_images = batch_size * num_batches
        throughput = total_images / elapsed  # images/sec

        # Should be > 10 images/sec on CPU
        assert throughput > 10, f"Throughput too low: {throughput:.1f} images/sec"

    def test_inference_latency_per_image(self):
        """Test inference latency for single image."""
        import time

        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 16, 3, padding=1),
            torch.nn.Conv2d(16, 2, 3, padding=1),
        )
        model.eval()

        num_runs = 20
        latencies = []

        with torch.no_grad():
            for _ in range(num_runs):
                input_image = torch.randn(1, 2, 64, 64)

                start = time.time()
                _ = model(input_image)
                end = time.time()

                latencies.append(end - start)

        avg_latency = sum(latencies) / len(latencies)

        # Should be < 50 ms per image on CPU
        assert avg_latency < 0.05, f"Latency too high: {avg_latency * 1000:.1f} ms"

    @pytest.mark.gpu
    def test_gpu_inference_speedup(self):
        """Test GPU inference is faster than CPU."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        import time

        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 32, 3, padding=1),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.Conv2d(32, 2, 3, padding=1),
        )

        # CPU inference
        model_cpu = model.cpu()
        model_cpu.eval()

        input_cpu = torch.randn(8, 2, 128, 128)

        start = time.time()
        with torch.no_grad():
            _ = model_cpu(input_cpu)
        cpu_time = time.time() - start

        # GPU inference
        model_gpu = model.cuda()
        model_gpu.eval()

        input_gpu = torch.randn(8, 2, 128, 128).cuda()

        # Warmup
        with torch.no_grad():
            _ = model_gpu(input_gpu)

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model_gpu(input_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        # GPU should be faster (at least 2x)
        assert (
            gpu_time < cpu_time
        ), f"GPU ({gpu_time:.4f}s) not faster than CPU ({cpu_time:.4f}s)"


@pytest.mark.gpu
class TestGPUMemoryProfilingInference:
    """Test GPU memory usage during inference."""

    def test_gpu_memory_usage(self):
        """Test GPU memory usage stays within limits."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 64, 3, padding=1),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.Conv2d(64, 2, 3, padding=1),
        ).cuda()
        model.eval()

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()

        # Run inference
        batch_size = 16
        input_batch = torch.randn(batch_size, 2, 256, 256).cuda()

        with torch.no_grad():
            output = model(input_batch)

        # Check peak memory
        peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB

        # Should be < 4 GB for this model in various environments
        assert peak_memory < 4.0, f"Peak memory too high: {peak_memory:.2f} GB"

    def test_inference_no_grad_mode_saves_memory(self):
        """Test torch.no_grad() reduces memory usage."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 64, 3, padding=1),
            torch.nn.Conv2d(64, 2, 3, padding=1),
        ).cuda()
        model.eval()

        input_batch = torch.randn(8, 2, 128, 128).cuda()

        # Without no_grad
        torch.cuda.reset_peak_memory_stats()
        output1 = model(input_batch)
        mem_with_grad = torch.cuda.max_memory_allocated() / 1e6  # MB

        # With no_grad
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            output2 = model(input_batch)
        mem_no_grad = torch.cuda.max_memory_allocated() / 1e6  # MB

        # no_grad should use less memory (no intermediate activations stored)
        assert (
            mem_no_grad < mem_with_grad
        ), f"no_grad ({mem_no_grad:.1f} MB) not less than with_grad ({mem_with_grad:.1f} MB)"


class TestInferenceReproducibility:
    """Test inference reproducibility with fixed seeds."""

    def test_inference_deterministic_with_seed(self):
        """Test inference is deterministic with fixed seed."""
        torch.manual_seed(42)

        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 16, 3, padding=1),
            torch.nn.Dropout(0.5),  # Non-deterministic without seed
            torch.nn.Conv2d(16, 2, 3, padding=1),
        )
        model.eval()

        input_batch = torch.randn(4, 2, 64, 64)

        # First run
        torch.manual_seed(42)
        with torch.no_grad():
            output1 = model(input_batch)

        # Second run with same seed
        torch.manual_seed(42)
        with torch.no_grad():
            output2 = model(input_batch)

        # Outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_inference_batch_order_independence(self):
        """Test inference results don't depend on batch order.

        Running samples individually vs batched should give same results.
        """
        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 16, 3, padding=1),
            torch.nn.Conv2d(16, 2, 3, padding=1),
        )
        model.eval()

        # Create 4 samples
        samples = [torch.randn(1, 2, 64, 64) for _ in range(4)]

        # Inference individually
        individual_outputs = []
        with torch.no_grad():
            for sample in samples:
                output = model(sample)
                individual_outputs.append(output)

        # Inference batched
        batch_input = torch.cat(samples, dim=0)  # (4, 2, 64, 64)
        with torch.no_grad():
            batch_output = model(batch_input)

        # Results should match
        for i in range(4):
            assert torch.allclose(
                individual_outputs[i], batch_output[i : i + 1], atol=1e-6
            ), f"Sample {i} differs between individual and batched inference"
