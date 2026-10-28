"""Streaming Inference Engine
==========================

Provides memory-efficient inference for large 3D volumes by processing them
in chunks (windows) along the depth dimension. This allows processing volumes
that are larger than GPU memory.

Key Features:
- Constant peak memory usage (O(window_size) instead of O(volume_size))
- Configurable window size and overlap
- Support for 3D volumes (NIfTI, etc.)
"""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from spectramr.data.io_strategies import lazy_load_nifti, save_nifti

logger = logging.getLogger(__name__)


class StreamingInferenceEngine:
    """Inference with constant peak memory regardless of volume size."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        window_size: int = 32,
        overlap: int = 0,
    ):
        """Initialize streaming inference engine.

        Args:
            model: PyTorch model to use for inference
            device: Device to run inference on
            window_size: Number of slices to process at once
            overlap: Number of overlapping slices (for smooth blending, not yet implemented)
        """
        self.model = model
        self.device = device
        self.window_size = window_size
        self.overlap = overlap

    def infer_volume_streaming(
        self,
        volume_path: str | Path,
        output_path: str | Path,
        preprocessing_fn: Callable[[np.ndarray], torch.Tensor] | None = None,
        postprocessing_fn: Callable[[torch.Tensor], np.ndarray] | None = None,
    ) -> None:
        """
        Infer on 3D volume with streaming (constant peak memory).

        Optimized: Uses threaded Producer-Consumer pattern to decouple I/O,
        Preprocessing, and Compute.
        """
        import queue
        import threading

        volume_path = Path(volume_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load header only (metadata)
        try:
            # Lazy-load via the data SSOT (CLAUDE.md pitfall #11). We
            # need the ``dataobj`` proxy for memory-mapped chunked reads;
            # NiftiStrategy.load() would eagerly load the full volume.
            nifti_img = lazy_load_nifti(volume_path)

            # Check if we can use array proxy
            if hasattr(nifti_img, "dataobj"):
                volume_proxy = nifti_img.dataobj
            else:
                volume_proxy = nifti_img.get_fdata()

            volume_shape = nifti_img.shape
            affine = nifti_img.affine
            dtype = np.float32

            logger.info(f"Streaming inference on {volume_path.name}, shape: {volume_shape}")

            if len(volume_shape) == 3 or len(volume_shape) == 4:
                slice_dim = 2
                num_slices = volume_shape[2]
            else:
                raise ValueError(f"Unsupported volume dimensionality: {len(volume_shape)}")

            # Create output file for streaming writes
            temp_npy_path = output_path.with_suffix(".temp.npy")
            output_memmap = np.memmap(temp_npy_path, dtype=dtype, mode="w+", shape=volume_shape)

            # Queues for pipeline
            read_queue = queue.Queue(maxsize=2)
            write_queue = queue.Queue(maxsize=2)

            # Reader Thread
            def reader():
                """reader.

                Returns:
                    Any: Description.
                """
                try:
                    for start in range(0, num_slices, self.window_size):
                        end = min(start + self.window_size, num_slices)

                        # Read chunk (I/O bound)
                        if slice_dim == 2:
                            chunk_data = np.array(volume_proxy[:, :, start:end])
                        else:
                            chunk_data = np.array(volume_proxy[:, :, start:end])

                        # Preprocess (CPU bound)
                        if preprocessing_fn:
                            chunk_tensor = preprocessing_fn(chunk_data)
                        else:
                            chunk_tensor = torch.from_numpy(chunk_data.astype(np.float32))
                            chunk_tensor = chunk_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

                        read_queue.put((start, end, chunk_tensor))
                    read_queue.put(None)
                except Exception as e:
                    logger.error(f"Reader thread failed: {e}")
                    read_queue.put(None)

            # Writer Thread
            def writer(memmap_array):
                """writer.

                Args:
                    memmap_array (Any): Description.
                Returns:
                    Any: Description.
                """
                try:
                    while True:
                        item = write_queue.get()
                        if item is None:
                            break
                        start, end, data_np = item

                        # Write chunk (I/O bound)
                        if slice_dim == 2:
                            memmap_array[:, :, start:end] = data_np
                        else:
                            # Handle other dims if needed, currently code implies slice_dim=2 logic
                            memmap_array[:, :, start:end] = data_np

                        write_queue.task_done()
                        logger.info(f"Processed slices {start}-{end}/{num_slices}")
                except Exception as e:
                    logger.error(f"Writer thread failed: {e}")

            # Start workers
            t_read = threading.Thread(target=reader, daemon=True)
            t_write = threading.Thread(target=writer, args=(output_memmap,), daemon=True)
            t_read.start()
            t_write.start()

            # GPU Loop (Main Thread)
            try:
                while True:
                    item = read_queue.get()
                    if item is None:
                        break
                    start, end, tensor = item

                    # Async Transfer + Compute
                    tensor = tensor.to(self.device, non_blocking=True)

                    with torch.no_grad():
                        output = self.model(tensor)

                    # Postprocess
                    if postprocessing_fn:
                        output_np = postprocessing_fn(output)
                    else:
                        output_np = output.squeeze().cpu().numpy()
                        if output_np.ndim == 3:
                            output_np = output_np.transpose(1, 2, 0)

                    # Send to writer
                    write_queue.put((start, end, output_np))
                    read_queue.task_done()

            except Exception as e:
                logger.error(f"GPU inference loop failed: {e}")
                raise
            finally:
                # Signal writer to stop
                write_queue.put(None)
                t_read.join()
                t_write.join()

            # Flush memmap
            output_memmap.flush()

            # Convert memmap to NIfTI and save
            save_nifti(output_path, output_memmap, affine)

            # Cleanup
            del output_memmap
            if temp_npy_path.exists():
                temp_npy_path.unlink()

            logger.info(f"Saved streaming output to {output_path}")

        except Exception as e:
            logger.error(f"Streaming inference failed: {e}")
            raise
