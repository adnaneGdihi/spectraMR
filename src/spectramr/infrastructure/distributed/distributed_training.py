"""
Phase 8.1: Distributed Training Support
Multi-GPU loss validation and metrics synchronization.
"""

import logging
import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class DistributedAuditResult:
    """Result of distributed loss audit."""

    rank: int
    world_size: int
    loss_valid: bool
    loss_value: float | None = None
    error_message: str | None = None
    gradient_norm: float | None = None
    backward_successful: bool = False
    sync_time_ms: float = 0.0
    all_losses_synchronized: bool = False


@dataclass
class DistributedMetricsSnapshot:
    """Snapshot of metrics from all ranks."""

    rank: int
    world_size: int
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0
    device: str = "cpu"
    synchronized: bool = False
    sync_time_ms: float = 0.0


class RankUtility:
    """Utilities for GPU rank coordination."""

    @staticmethod
    def get_rank() -> int:
        """Get current process rank.

        Returns:
            Rank if distributed, 0 otherwise
        """
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @staticmethod
    def get_local_rank() -> int:
        """Get the node-LOCAL rank (the CUDA device index for this process).

        DDP's ``device_ids`` must be a node-local device index, NOT the global
        rank: on a multi-node run the global rank exceeds the per-node device
        count, so ``device_ids=[global_rank]`` selects a non-existent or wrong
        GPU (or raises). Prefer ``LOCAL_RANK`` (set by torchrun / our
        ``launcher``); fall back to ``global_rank % device_count()`` for
        contiguous per-node rank assignment.
        """
        local = os.environ.get("LOCAL_RANK")
        if local is not None:
            return int(local)
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            if n > 0:
                return RankUtility.get_rank() % n
        return 0

    @staticmethod
    def get_world_size() -> int:
        """Get total number of processes.

        Returns:
            World size if distributed, 1 otherwise
        """
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    @staticmethod
    def is_main_rank() -> bool:
        """Check if current process is rank 0.

        Returns:
            True if rank 0, False otherwise
        """
        return RankUtility.get_rank() == 0

    @staticmethod
    def broadcast_tensor(tensor: torch.Tensor, src_rank: int = 0) -> torch.Tensor:
        """Broadcast tensor from source rank to all ranks.

        Args:
            tensor: Tensor to broadcast
            src_rank: Source rank (default 0)

        Returns:
            Broadcast tensor
        """
        if not dist.is_available() or not dist.is_initialized():
            return tensor

        dist.broadcast(tensor, src=src_rank)
        return tensor

    @staticmethod
    def all_gather_tensor(tensor: torch.Tensor) -> list[torch.Tensor]:
        """Gather tensors from all ranks to rank 0.

        Args:
            tensor: Tensor to gather

        Returns:
            List of tensors from all ranks (only rank 0), empty list on others
        """
        if not dist.is_available() or not dist.is_initialized():
            return [tensor]

        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)

        return gathered if RankUtility.is_main_rank() else []

    @staticmethod
    def broadcast_object(obj: object, src_rank: int = 0) -> object:
        """Broadcast a picklable Python object from ``src_rank`` to all ranks.

        Used for rank-consistent control decisions (e.g. the loss-schedule
        override map + checkpoint-request flag): rank 0 decides, every rank
        applies the identical result, so a decision derived from a per-rank
        quantity (such as the per-rank training loss feeding a ``loss_plateau``
        trigger) cannot diverge across ranks under DDP.

        No-op (returns ``obj`` unchanged) when ``torch.distributed`` is not
        initialised, so the single-process path is unaffected.

        Args:
            obj: A picklable object (only meaningful on ``src_rank``; ignored
                on others, which receive the broadcast value).
            src_rank: Source rank (default 0).

        Returns:
            The object from ``src_rank`` on every rank.
        """
        if not dist.is_available() or not dist.is_initialized():
            return obj
        holder = [obj]
        dist.broadcast_object_list(holder, src=src_rank)
        return holder[0]

    @staticmethod
    def barrier():
        """Synchronize all ranks."""
        if dist.is_available() and dist.is_initialized():
            dist.barrier()


def _all_reduce_mean(tensor: torch.Tensor) -> None:
    """In-place mean across ranks that also works on ``gloo``.

    ``ReduceOp.AVG`` is NCCL-only, so on the CPU test lane it raised and the
    callers' ``except`` turned that into a logged warning instead of a result.
    SUM then divide is identical on every backend.
    """
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()


class DistributedLossAudit:
    """Audits loss computation across multiple GPUs."""

    def __init__(self, loss_fn, logger: logging.Logger | None = None):
        """Initialize distributed loss audit.

        Args:
            loss_fn: Loss function to audit
            logger: Optional logger instance
        """
        self.loss_fn = loss_fn
        self.logger = logger or logging.getLogger(__name__)
        self.rank = RankUtility.get_rank()
        self.world_size = RankUtility.get_world_size()
        #: Whether this rank's last audit produced a finite loss (and, when asked,
        #: a successful backward). ``None`` until an audit has run.
        self._last_valid: bool | None = None

    def audit_loss_computation(
        self,
        batch: dict[str, torch.Tensor],
        model_output: torch.Tensor,
        target: torch.Tensor,
        backward: bool = True,
    ) -> DistributedAuditResult:
        """Audit loss computation on current rank and synchronize.

        Args:
            batch: Input batch dict
            model_output: Model output
            target: Target output
            backward: Whether to perform backward pass

        Returns:
            DistributedAuditResult with audit information
        """
        import time

        start_time = time.time()
        result = DistributedAuditResult(
            rank=self.rank,
            world_size=self.world_size,
            loss_valid=False,
        )

        try:
            # Compute loss on current rank
            loss = self.loss_fn(model_output, target)

            # Validate loss value
            if not torch.isfinite(loss):
                raise ValueError(f"Loss is not finite: {loss}")

            result.loss_value = loss.item()
            result.loss_valid = True

            # Optional: compute gradient norm
            if backward:
                loss.backward()
                grad_norm = self._compute_grad_norm(self.loss_fn)
                result.gradient_norm = grad_norm
                result.backward_successful = True

        except Exception as e:
            result.error_message = str(e)
            self.logger.error(f"[Rank {self.rank}] Loss audit failed: {e}")

        # Outside the try: a rank whose loss was non-finite raised above and, with
        # the sync inside it, skipped the collective its peers were already waiting
        # in -- a hang, not an error.
        if self.world_size > 1:
            self._synchronize_losses(result)

        self._last_valid = result.loss_valid and (not backward or result.backward_successful)
        result.sync_time_ms = (time.time() - start_time) * 1000
        return result

    def _compute_grad_norm(self, loss_fn) -> float:
        """Compute gradient norm.

        Args:
            loss_fn: Loss function with gradients

        Returns:
            Gradient norm value
        """
        total_norm = 0.0
        for p in loss_fn.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm**0.5

    def _synchronize_losses(self, result: DistributedAuditResult):
        """Synchronize loss values across ranks.

        Args:
            result: Audit result to update
        """
        if not dist.is_available() or not dist.is_initialized():
            return

        try:
            # [loss or 0, 1 if valid else 0], summed: the mean is over the ranks that
            # produced a loss, and every rank joins the collective whatever its outcome.
            valid = result.loss_valid
            packed = torch.tensor(
                [result.loss_value if valid else 0.0, 1.0 if valid else 0.0],
                device=(torch.cuda.current_device() if torch.cuda.is_available() else "cpu"),
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)

            result.all_losses_synchronized = True

            if RankUtility.is_main_rank() and packed[1].item() > 0:
                mean_loss = (packed[0] / packed[1]).item()
                self.logger.info(f"[Rank {self.rank}] Synchronized loss: {mean_loss:.6f}")

        except Exception as e:
            self.logger.warning(f"[Rank {self.rank}] Failed to synchronize losses: {e}")

    def validate_all_ranks(self) -> bool:
        """Validate loss computation succeeded on all ranks.

        Returns:
            True if valid on all ranks
        """
        if not dist.is_available() or not dist.is_initialized():
            return True

        # Broadcast validity from each rank
        result_tensor = torch.tensor(
            [1 if self._last_valid else 0],
            device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
        )

        dist.all_reduce(result_tensor, op=dist.ReduceOp.MIN)

        return result_tensor.item() > 0


class DistributedMetricsCollector:
    """Collects and synchronizes metrics across multiple GPUs."""

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize metrics collector.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.rank = RankUtility.get_rank()
        self.world_size = RankUtility.get_world_size()
        self.local_metrics: dict[str, list[float]] = {}

    def update_metrics(self, metrics: dict[str, float]):
        """Update local metrics.

        Args:
            metrics: Dict of metric name -> value
        """
        for name, value in metrics.items():
            if name not in self.local_metrics:
                self.local_metrics[name] = []
            self.local_metrics[name].append(value)

    def synchronize_metrics(self) -> DistributedMetricsSnapshot:
        """Synchronize metrics across all ranks.

        Returns:
            DistributedMetricsSnapshot with synchronized metrics
        """
        import time

        start_time = time.time()

        snapshot = DistributedMetricsSnapshot(
            rank=self.rank,
            world_size=self.world_size,
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )

        if not dist.is_available() or not dist.is_initialized():
            # Single-process: just compute local averages
            snapshot.metrics = self._compute_local_averages()
            snapshot.synchronized = True
            snapshot.sync_time_ms = (time.time() - start_time) * 1000
            return snapshot

        try:
            # Compute local averages
            local_avgs = self._compute_local_averages()

            # Synchronize each metric
            for name, value in local_avgs.items():
                metric_tensor = torch.tensor(
                    [value],
                    device=(torch.cuda.current_device() if torch.cuda.is_available() else "cpu"),
                )

                # Average across all ranks
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.AVG)

                snapshot.metrics[name] = metric_tensor.item()

            snapshot.synchronized = True

            if RankUtility.is_main_rank():
                self.logger.info(f"[Rank {self.rank}] Synchronized metrics: {snapshot.metrics}")

        except Exception as e:
            self.logger.warning(f"[Rank {self.rank}] Failed to synchronize metrics: {e}")

        snapshot.sync_time_ms = (time.time() - start_time) * 1000
        return snapshot

    def _compute_local_averages(self) -> dict[str, float]:
        """Compute averages of local metrics.

        Returns:
            Dict of metric name -> average value
        """
        averages = {}
        for name, values in self.local_metrics.items():
            if values:
                averages[name] = sum(values) / len(values)
        return averages

    def reset_metrics(self):
        """Reset local metrics."""
        self.local_metrics = {}


class DistributedTrainingContext:
    """Context manager for distributed training."""

    def __init__(self, backend: str = "nccl"):
        """Initialize distributed context.

        Args:
            backend: Distributed backend (nccl, gloo, etc.)
        """
        self.backend = backend
        self.rank = RankUtility.get_rank()
        self.world_size = RankUtility.get_world_size()
        self.is_initialized = False

    def __enter__(self):
        """Enter context and initialize distributed training.

        Raises when the process group cannot be created. Warning and carrying on
        left every rank running as an independent single-process job that still
        reported success.
        """
        if not dist.is_available():
            return self

        if not dist.is_initialized():
            dist.init_process_group(backend=self.backend)
            self.is_initialized = True
            # Re-read: rank and world size are only authoritative once the group exists.
            self.rank = RankUtility.get_rank()
            self.world_size = RankUtility.get_world_size()
            logging.info(
                f"Initialized distributed training: "
                f"rank={self.rank}, world_size={self.world_size}"
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and cleanup."""
        if self.is_initialized and dist.is_available():
            try:
                dist.destroy_process_group()
            except Exception as e:
                logging.warning(f"Failed to destroy process group: {e}")


def wrap_model_distributed(model: torch.nn.Module) -> torch.nn.Module:
    """Wrap model with DistributedDataParallel if distributed.

    Args:
        model: Model to wrap

    Returns:
        Wrapped model if distributed, original otherwise
    """
    if dist.is_available() and dist.is_initialized():
        # device_ids must be the node-LOCAL device index, not the global rank
        # (multi-node: global rank > per-node GPU count → invalid index).
        return DDP(model, device_ids=[RankUtility.get_local_rank()])
    return model


def synchronize_across_ranks(value: float, op: str = "mean") -> float:
    """Synchronize a scalar value across all ranks.

    Args:
        value: Scalar value to synchronize
        op: Operation (mean, sum, min, max)

    Returns:
        Synchronized value
    """
    if not dist.is_available() or not dist.is_initialized():
        return value

    tensor = torch.tensor(
        [value],
        device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
    )

    if op == "mean":
        _all_reduce_mean(tensor)
    elif op == "sum":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif op == "min":
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    elif op == "max":
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    else:
        raise ValueError(f"Unknown reduction operation: {op}")

    return tensor.item()
