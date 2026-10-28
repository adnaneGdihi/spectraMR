"""Meta-learning dataset wrapper for MRI reconstruction tasks.

This module provides a dataset wrapper that samples "tasks" for meta-learning.
In the context of MRI reconstruction, a task can be defined by:
- A specific acceleration factor
- A specific anatomy (if dataset supports it)
- A specific contrast (if dataset supports it)
"""

import random
from typing import Any

import torch
from torch.utils.data import Dataset

from spectramr.infrastructure.physics.fft_ops import FFTTransformer
from spectramr.infrastructure.training.utils.kspace_masks import create_kspace_mask_generator

#: Rungs of the mask ladder a requested acceleration is quantised onto. It is a mask
#: resolution, not a model's diffusion horizon, and the generator and the mapping must share it.
MASK_LADDER_STEPS = 1000


class MetaLearningDataset(Dataset):
    """Dataset wrapper for meta-learning that samples tasks.

    Each item returned by this dataset is a "task batch" containing:
    - Support set: (support_x, support_y)
    - Query set: (query_x, query_y)

    The task is defined by sampling specific parameters (e.g., acceleration factor)
    and applying them to a batch of data.

    .. mermaid::

        flowchart TD
            Start[Sample Task] --> Params[Task Params]
            Params -->|Acceleration| Support{Support Set}
            Params -->|Acceleration| Query{Query Set}

            Support -->|Fetch| Indices1[Indices]
            Query -->|Fetch| Indices2[Indices]

            Indices1 --> Batch1[Base Data]
            Indices2 --> Batch2[Base Data]

            Batch1 & Batch2 --> Degrade[Apply Task Physics]
            Degrade --> Output[Task Batch]
    """

    def __init__(
        self,
        base_dataset: Dataset,
        task_config: dict[str, Any] | None = None,
        support_size: int = 4,
        query_size: int = 4,
        tasks_per_epoch: int = 100,
        config: Any = None,
    ):
        """Initialize MetaLearningDataset.

        Args:
            base_dataset: The underlying dataset providing MRI data.
            task_config: Configuration for task sampling.
                Example: {'acceleration_factors': [2, 4, 8, 16]}.
                Optional when ``config`` is provided (Phase 4f).
            support_size: Number of examples in support set (legacy kwarg).
            query_size: Number of examples in query set (legacy kwarg).
            tasks_per_epoch: Number of tasks to generate per epoch (legacy kwarg).
            config: Optional :class:`MetaLearningConfigSchema`. When
                supplied, support_size / query_size / tasks_per_epoch /
                task_params are read from the schema; explicit kwargs
                still override (the schema is the default source).
        """
        # Phase 4f, Amendment J: when a schema is provided, source values
        # from it (preserves backward compatibility for legacy callers).
        if config is not None:
            task_config = task_config or dict(config.task_params)
            support_size = config.support_size
            query_size = config.query_size
            tasks_per_epoch = config.tasks_per_epoch

        self.base_dataset = base_dataset
        self.task_config = task_config or {}
        self.support_size = support_size
        self.query_size = query_size
        self.tasks_per_epoch = tasks_per_epoch

        # Cache dataset length
        self.base_len = len(base_dataset)

        # Initialize physics tools
        self.mask_generator = create_kspace_mask_generator(
            num_timesteps=MASK_LADDER_STEPS,
            device=torch.device("cpu"),  # Dataset usually runs on CPU
            default_pattern="linear",
        )
        self.fft_transformer = FFTTransformer()

    def __len__(self) -> int:
        """__len__.

        Returns:
            int: Description.
        """
        return self.tasks_per_epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Sample a task and return support/query sets.

        Args:
            index: Index (ignored, tasks are sampled stochastically).

        Returns:
            Dictionary containing 'support' and 'query' sets.
        """
        # 1. Sample task parameters
        task_params = self._sample_task_params()

        # 2. Sample data indices for support and query sets
        # We need support_size + query_size distinct examples
        total_samples = self.support_size + self.query_size
        indices = random.sample(range(self.base_len), total_samples)

        support_indices = indices[: self.support_size]
        query_indices = indices[self.support_size :]

        # 3. Fetch and process data for support set
        support_x, support_y = self._fetch_and_process_batch(support_indices, task_params)

        # 4. Fetch and process data for query set
        query_x, query_y = self._fetch_and_process_batch(query_indices, task_params)

        return {
            "support": (support_x, support_y),
            "query": (query_x, query_y),
            "task_params": task_params,
        }

    def _sample_task_params(self) -> dict[str, Any]:
        """Sample parameters defining a task."""
        params = {}

        # Sample acceleration factor if configured
        if "acceleration_factors" in self.task_config:
            acc_factors = self.task_config["acceleration_factors"]
            params["acceleration"] = random.choice(acc_factors)

        return params

    def _get_timestep_for_acceleration(self, acceleration: float) -> int:
        """Map acceleration factor to diffusion timestep for mask generation.

        Assumes LinearKSpaceAccelerator with max_acceleration=64.0.
        """
        max_acc = 64.0
        # Linear mapping: acc = 1.0 + ratio * (max - 1.0)
        ratio = (acceleration - 1.0) / (max_acc - 1.0)
        ratio = max(0.0, min(1.0, ratio))
        return int(ratio * (MASK_LADDER_STEPS - 1))

    def _fetch_and_process_batch(
        self, indices: list[int], task_params: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch data and apply task-specific processing (e.g., undersampling)."""
        batch_x = []
        batch_y = []

        for idx in indices:
            # Fetch raw item from base dataset
            # Assuming base dataset returns {'kspace': ..., 'target': ...} or similar
            item = self.base_dataset[idx]

            # Extract ground truth (fully sampled k-space or image)
            # This depends on base dataset structure. Let's assume standard keys.
            if isinstance(item, dict):
                # Handle different dataset formats
                if "hr" in item:
                    target = item["hr"]
                elif "target" in item:
                    target = item["target"]
                elif "kspace" in item:
                    target = item["kspace"]
                else:
                    # Fallback: assume item itself is the target if not dict
                    target = item
            else:
                target = item

            # Ensure target is a tensor
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)

            # Apply task-specific degradation (undersampling)
            # For now, we simulate this if 'acceleration' is in task_params
            if "acceleration" in task_params:
                acc = task_params["acceleration"]

                # Calculate timestep for mask generator
                timestep = self._get_timestep_for_acceleration(acc)

                # Convert to k-space (assuming target is image)
                # If target is already complex, we assume it's image domain complex data
                # unless we have metadata saying otherwise.
                kspace = self.fft_transformer.image_to_kspace(target)

                # Generate mask
                if kspace.dim() >= 2:
                    spatial_shape = (kspace.shape[-2], kspace.shape[-1])
                else:
                    spatial_shape = (256, 256)

                mask = self.mask_generator.generate_acceleration_mask(
                    timestep=timestep, image_shape=spatial_shape, pattern="linear"
                )

                # Apply mask
                # Ensure mask is on correct device
                mask = mask.to(kspace.device)
                masked_kspace = kspace * mask

                # Convert back to image (Zero-Filled Recon)
                input_data = self.fft_transformer.kspace_to_image(masked_kspace)

                # If target was real, take magnitude of input
                if not torch.is_complex(target):
                    input_data = torch.abs(input_data)

            else:
                input_data = target.clone()

            batch_x.append(input_data)
            batch_y.append(target)

        # Stack into batch tensors
        return torch.stack(batch_x), torch.stack(batch_y)
