"""Multi-Stage Training Strategy — Recursive Graph Interpreter.

The engine reads YAML-defined routines as an instruction list and
recursively executes them, maintaining a ``state`` memory bank and
a ``loop_context`` for iterator variables.

Supported actions:
- ``call``:   invoke a stage model (or a specific method)
- ``loop``:   iterate and recursively execute body actions
- ``assign``: copy a value into the state memory bank
- ``detach``: sever the grad graph on a state tensor

If no ``routines`` are configured, the strategy falls back to ``flow``
or sequential stage execution (backward compatible).

Reference:
    Recursive interpreter pattern inspired by CompVis LDM and TRELLIS.
    Sliding window inference reuses ``spectramr.infrastructure.inference.sliding_window``.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.config.schemas.training.pipeline import (
    ActionSchema,
    MultiStageSchema,
    MultiTrainingConfigSchema,
    TrainingStateSchema,
)
from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.checkpointing.restore import CheckpointPartMissingError, weights
from spectramr.infrastructure.services.metric_keys import resolve_metric_key
from spectramr.infrastructure.training.builders.loss_builder import LossBuilder
from spectramr.infrastructure.training.optimizer_registry import create_optimizer
from spectramr.infrastructure.training.strategies import pipeline_stage_state
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.models.lora import inject_lora_adapters

logger = logging.getLogger(__name__)


class MultiTrainingStrategy(BaseTrainingStrategy):
    """Multi-stage training strategy with recursive graph interpreter.

    Stitches multiple registered generators into a configurable graph,
    with named routines, loop support, per-stage freeze/unfreeze/LoRA,
    gradient checkpointing, hybrid 2D/3D spatial dims, and sliding
    window inference.

    Attributes:
        multi_stages: Dict mapping stage name → nn.Module.
        stage_configs: Original per-stage configuration schemas.
        routines: Dict mapping routine name → list of ActionSchema.
        _finetune_epoch: Epoch at which all stages are unfrozen.
    """

    # parallel_compat: Holds the stage models outside env.models, freezes them mid-run and builds
    # its own per-stage optimizers.
    parallel_compat = frozenset()

    def __init__(
        self,
        env: Any,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, device=device, **kwargs)
        # No second ``self._setup_strategy_specific_components()``: the base __init__
        # already runs it, and it is not idempotent once it adds optimizer param groups.

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_strategy_specific_components(self) -> None:
        """Instantiate, configure, and register multi-stage models."""
        multi_cfg = self._get_multi_config()

        self.multi_stages: dict[str, nn.Module] = {}
        self.stage_configs: dict[str, MultiStageSchema] = {}

        for stage_cfg in multi_cfg.stages:
            model = self._create_stage_model(stage_cfg)

            if stage_cfg.checkpoint_path:
                self._load_stage_weights(model, stage_cfg.checkpoint_path, stage_cfg.name)

            self._apply_training_state(model, stage_cfg.training_state, stage_cfg.name)

            if stage_cfg.gradient_checkpointing:
                self._enable_gradient_checkpointing(model, stage_cfg.name)

            # Log model architecture for block discoverability
            self._log_stage_architecture(model, stage_cfg.name)

            model.to(self.device, non_blocking=True)
            self.multi_stages[stage_cfg.name] = model
            self.stage_configs[stage_cfg.name] = stage_cfg

        # Build routines dict (priority: routines > flow > sequential)
        self.routines: dict[str, list[ActionSchema]] = dict(multi_cfg.routines)
        if not self.routines and multi_cfg.flow:
            # Auto-promote flow to the 'train' routine
            self.routines["train"] = list(multi_cfg.flow)
        self.stage_order = [s.name for s in multi_cfg.stages]
        self.routing_graph = multi_cfg.routing
        self._use_routines = len(self.routines) > 0

        # End-to-end fine-tuning schedule
        self._finetune_epoch = multi_cfg.end_to_end_finetune_epoch
        self._evaluate_intermediates = multi_cfg.evaluate_intermediates
        self._eval_config = multi_cfg.evaluation

        # Build per-stage losses and optimizers (framework-compliant)
        self._stage_losses: dict[str, dict[str, nn.Module]] = {}
        self._stage_loss_weights: dict[str, dict[str, float]] = {}
        self._stage_optimizers: dict[str, torch.optim.Optimizer] = {}
        self._stage_early_stopping: dict[str, Any] = {}
        # Stages whose early-stopping monitor could not be resolved against
        # the validation metrics -- warned about once each, not once an epoch.
        self._es_unresolved_monitors: set[str] = set()
        self._build_stage_losses()
        self._build_stage_optimizers()

        # Build evaluation metrics from MultiEvalSchema
        self._eval_metrics: dict[str, Any] = {}
        self._eval_metric_schemas: dict[str, Any] = {}
        self._build_eval_metrics()

        # Collect trainable parameters and hand the global-optimizer ones to
        # opt_g. Without this, stages driven by the global optimizer (those with
        # no per-stage optimizer) never receive a param_group → their weights
        # stay frozen-in-practice even though requires_grad=True.
        self._trainable_params = self._collect_trainable_params()
        self._register_global_params()
        self._log_multi_summary()

    def _get_multi_config(self) -> MultiTrainingConfigSchema:
        """Extract multi config from training config. Fail-fast."""
        if self.config.training is None or self.config.training.multi is None:
            raise ValueError("Multi-stage strategy requires training.multi configuration.")
        return self.config.training.multi

    def _create_stage_model(self, stage_cfg: MultiStageSchema) -> nn.Module:
        """Create a model via GeneratorBuilder with per-stage overrides."""
        # Only the training ModelBuilder loads a stage-1 VAE; a stage would silently
        # train in a random latent space (NN3/NN8).
        if "vae_checkpoint_path" in (stage_cfg.model_kwargs or {}):
            raise ValueError(
                f"stage '{stage_cfg.name}' sets model_kwargs.vae_checkpoint_path, but "
                "multi-stage stages do not load a stage-1 VAE. Use a single-stage LDM "
                "arm, or load the trained stage through the stage's checkpoint_path."
            )
        from spectramr.infrastructure.builders.leaf.model_builders import GeneratorBuilder

        in_ch = stage_cfg.in_channels or self.config.model.in_channels
        out_ch = stage_cfg.out_channels or self.config.model.out_channels

        builder = (
            GeneratorBuilder(self.config, self.device)
            .with_architecture(stage_cfg.model_type)
            .with_input_channels(in_ch)
            .with_output_channels(out_ch)
        )
        if stage_cfg.spatial_dims is not None:
            builder.with_parameter("spatial_dims", stage_cfg.spatial_dims)
        for k, v in stage_cfg.model_kwargs.items():
            builder.with_parameter(k, v)

        return builder.validate().build()

    def _load_stage_weights(self, model: nn.Module, checkpoint_path: str, stage_name: str) -> None:
        """A stage's own ``stage:<name>`` part when the file has one, else its generator."""
        try:
            try:
                weights(checkpoint_path, f"stage:{stage_name}", model, policy="overlap")
            except CheckpointPartMissingError:
                weights(checkpoint_path, "generator", model, policy="overlap")
            logger.info("[Multi] Loaded weights from '%s'", checkpoint_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint '{checkpoint_path}': {e}") from e

    def _apply_training_state(
        self,
        model: nn.Module,
        state: TrainingStateSchema,
        stage_name: str,
    ) -> None:
        """Apply freeze/unfreeze/LoRA (order: freeze → unfreeze → LoRA)."""
        total_params = sum(p.numel() for p in model.parameters())

        if state.freeze_all:
            for param in model.parameters():
                param.requires_grad = False
            logger.info("[Multi] Stage '%s': all %d params frozen", stage_name, total_params)

        if state.unfreeze_modules:
            unfrozen_count = 0
            for name, param in model.named_parameters():
                if any(pattern in name for pattern in state.unfreeze_modules):
                    param.requires_grad = True
                    unfrozen_count += int(param.numel())
            logger.info(
                "[Multi] Stage '%s': unfroze %d params matching %s",
                stage_name,
                unfrozen_count,
                state.unfreeze_modules,
            )

        if state.inject_lora:
            inject_lora_adapters(
                model,
                target_modules=state.lora_target_modules,
                rank=state.lora_rank,
                alpha=state.lora_alpha,
            )
            lora_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
            logger.info(
                "[Multi] Stage '%s': LoRA (r=%d, α=%d) → %d trainable",
                stage_name,
                state.lora_rank,
                state.lora_alpha,
                lora_params,
            )

        trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
        logger.info(
            "[Multi] Stage '%s': %d/%d trainable (%.1f%%)",
            stage_name,
            trainable,
            total_params,
            100.0 * trainable / max(total_params, 1),
        )

    def _enable_gradient_checkpointing(self, model: nn.Module, stage_name: str) -> None:
        """Enable gradient checkpointing if supported."""
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("[Multi] Stage '%s': gradient checkpointing enabled", stage_name)
        else:
            logger.warning("[Multi] Stage '%s': no gradient_checkpointing_enable()", stage_name)

    def _collect_trainable_params(self) -> list[nn.Parameter]:
        """Collect all trainable parameters across all stages."""
        params: list[nn.Parameter] = []
        for model in self.multi_stages.values():
            params.extend(p for p in model.parameters() if p.requires_grad)
        return params

    def _register_global_params(self) -> list[nn.Parameter]:
        """Hand global-optimizer trainable params to ``self.env.opt_g``.

        Stages with their own optimizer (``_stage_optimizers``) are stepped
        independently in ``_step_stage_optimizers`` and must NOT be added here,
        or they would be updated twice per iteration. Every other trainable
        param belongs to the global optimizer; without this registration those
        stages never appear in any ``param_group`` and silently never train.

        Dedups against params already present in any ``opt_g`` group so this is
        safe to call repeatedly (e.g. again after the E2E unfreeze).

        Returns:
            The list of params actually appended (empty if none / no opt_g).
        """
        opt = getattr(self.env, "opt_g", None)
        if opt is None:
            return []

        # Params already owned by a per-stage optimizer are stepped separately.
        stage_owned: set[int] = set()
        for optimizer in self._stage_optimizers.values():
            for group in optimizer.param_groups:
                stage_owned.update(id(p) for p in group["params"])

        existing = {id(p) for g in opt.param_groups for p in g["params"]}
        new_params = [
            p for p in self._trainable_params if id(p) not in existing and id(p) not in stage_owned
        ]
        if new_params:
            opt.add_param_group({"params": new_params})
        return new_params

    def _log_stage_architecture(self, model: nn.Module, stage_name: str) -> None:
        """Log all named modules with param counts and trainability.

        Enables users to discover which module names to use in
        ``unfreeze_modules``. Each module is logged with its type,
        parameter count, and whether it is currently trainable.

        Args:
            model: The stage model.
            stage_name: Name of the stage for log prefixing.
        """
        lines: list[str] = []
        for name, module in model.named_modules():
            if not name:  # Skip root module
                continue
            # Only log leaf modules (no children = leaf)
            children = list(module.children())
            if children:
                continue

            num_params = sum(p.numel() for p in module.parameters(recurse=False))
            if num_params == 0:
                continue

            trainable = any(p.requires_grad for p in module.parameters(recurse=False))
            status = "trainable" if trainable else "frozen"
            type_name = type(module).__name__
            lines.append(f"    {name:<45s} ({type_name}, {num_params:>8,d} params, {status})")

        if lines:
            header = f"[Multi] Stage '{stage_name}' architecture ({len(lines)} leaf modules):"
            logger.info("%s\n%s", header, "\n".join(lines))

    def _log_multi_summary(self) -> None:
        """Log enhanced summary with per-stage strategy, losses, and state."""
        total = sum(sum(p.numel() for p in m.parameters()) for m in self.multi_stages.values())
        trainable = sum(p.numel() for p in self._trainable_params)

        # Global summary
        if self._use_routines:
            routine_names = list(self.routines.keys())
            logger.info(
                "[Multi] %d stages, routines=%s, %d total, %d trainable (%.1f%%)",
                len(self.multi_stages),
                routine_names,
                total,
                trainable,
                100.0 * trainable / max(total, 1),
            )
        else:
            logger.info(
                "[Multi] %d stages (sequential), %d total, %d trainable (%.1f%%)",
                len(self.multi_stages),
                total,
                trainable,
                100.0 * trainable / max(total, 1),
            )

        # Per-stage detail lines
        for name, stage_cfg in self.stage_configs.items():
            parts: list[str] = []

            # Strategy class
            if (
                stage_cfg.stage_config is not None
                and stage_cfg.stage_config.training is not None
                and stage_cfg.stage_config.training.strategy_class
            ):
                parts.append(f"strategy={stage_cfg.stage_config.training.strategy_class}")

            # Losses
            if name in self._stage_losses:
                loss_names = list(self._stage_losses[name].keys())
                parts.append(f"losses={loss_names}")

            # Training state
            ts = stage_cfg.training_state
            if ts.freeze_all and not ts.inject_lora and not ts.unfreeze_modules:
                parts.append("FROZEN")
            elif ts.inject_lora:
                parts.append(f"LoRA(r={ts.lora_rank})")
            elif ts.unfreeze_modules:
                parts.append(f"unfreeze={ts.unfreeze_modules}")

            # Optimizer override
            if name in self._stage_optimizers:
                parts.append("own_optimizer")

            # Early stopping
            if name in self._stage_early_stopping:
                es = self._stage_early_stopping[name]
                parts.append(f"early_stop(patience={es.patience})")

            detail = ", ".join(parts) if parts else "default"
            logger.info("[Multi]   └─ '%s': %s", name, detail)

        if self._finetune_epoch:
            logger.info("[Multi] E2E fine-tuning at epoch %d", self._finetune_epoch)

    # ------------------------------------------------------------------
    # Recursive graph interpreter
    # ------------------------------------------------------------------

    def _resolve(
        self,
        val_path: str,
        batch_data: dict[str, Any],
        state: dict[str, Any],
        loop_context: dict[str, Any],
    ) -> Any:
        """Resolve a namespace reference to an actual value.

        Supports:
        - ``batch.<key>``:        data loader tensor
        - ``state.<key>``:        interpreter state memory
        - ``loop_context.<var>``: current loop variable
        - Literal values (numbers, strings without '.')
        """
        if not isinstance(val_path, str) or "." not in val_path:
            return val_path

        namespace, key = val_path.split(".", 1)
        if namespace == "batch":
            if key not in batch_data:
                raise ValueError(
                    f"Resolve: batch key '{key}' not found. Available: {list(batch_data.keys())}"
                )
            return batch_data[key]
        if namespace == "state":
            if key not in state:
                raise ValueError(
                    f"Resolve: state key '{key}' not found. Available: {list(state.keys())}"
                )
            return state[key]
        if namespace == "loop_context":
            if key not in loop_context:
                raise ValueError(
                    f"Resolve: loop_context key '{key}' not found. "
                    f"Available: {list(loop_context.keys())}"
                )
            return loop_context[key]
        # Fallback: return the full path as a literal
        return val_path

    def _execute_flow(
        self,
        actions: list[ActionSchema],
        batch_data: dict[str, Any],
        state: dict[str, Any],
        loop_context: dict[str, Any] | None = None,
        training: bool = True,
    ) -> None:
        """Recursively execute a list of actions.

        This is the core interpreter loop. It dispatches each action
        to its handler and mutates the shared ``state`` dict.
        """
        if loop_context is None:
            loop_context = {}

        for action in actions:
            if action.action == "call":
                self._execute_call(action, batch_data, state, loop_context, training)
            elif action.action == "loop":
                self._execute_loop(action, batch_data, state, loop_context, training)
            elif action.action == "assign":
                self._execute_assign(action, batch_data, state, loop_context)
            elif action.action == "detach":
                self._execute_detach(action, state)
            else:
                raise ValueError(f"Unknown action type: '{action.action}'")

    def _execute_call(
        self,
        action: ActionSchema,
        batch_data: dict[str, Any],
        state: dict[str, Any],
        loop_context: dict[str, Any],
        training: bool,
    ) -> None:
        """Execute a 'call' action: invoke model forward() or method."""
        node_name = action.node
        assert node_name is not None
        model = self.multi_stages[node_name]
        stage_cfg = self.stage_configs[node_name]

        is_frozen = (
            stage_cfg.training_state.freeze_all
            and not self._all_unfrozen
            and not stage_cfg.training_state.inject_lora
        )

        # Resolve input kwargs
        kwargs: dict[str, Any] = {}
        for param_name, source_ref in action.inputs.items():
            kwargs[param_name] = self._resolve(source_ref, batch_data, state, loop_context)

        # Determine callable: forward() or specific method
        if action.method:
            func = getattr(model, action.method)
        else:
            func = model

        # Execute with appropriate grad context.
        # Use ``torch.no_grad()`` (NOT ``inference_mode()``): the frozen output
        # is re-armed below with ``requires_grad_(True)`` so the downstream
        # trainable stage receives gradients, and that call raises a
        # RuntimeError on an inference-mode tensor ("Setting requires_grad=True
        # on inference tensor outside InferenceMode is not allowed"). no_grad
        # tensors are ordinary tensors and can be re-armed.
        if is_frozen and training:
            with torch.no_grad():
                if kwargs:
                    output = (
                        func(**kwargs) if len(kwargs) > 1 else func(next(iter(kwargs.values())))
                    )
                else:
                    output = func()
            # Detach frozen output but allow gradient flow from here
            if isinstance(output, torch.Tensor):
                output = output.detach().requires_grad_(True)
        else:
            if kwargs:
                output = func(**kwargs) if len(kwargs) > 1 else func(next(iter(kwargs.values())))
            else:
                output = func()

        # Write outputs to state
        if action.outputs:
            for state_key, output_attr in action.outputs.items():
                if isinstance(output, dict) and output_attr in output:
                    state[state_key] = output[output_attr]
                elif hasattr(output, output_attr):
                    state[state_key] = getattr(output, output_attr)
                else:
                    # Default: treat as main tensor output
                    state[state_key] = output
        else:
            state[node_name] = output

    def _execute_loop(
        self,
        action: ActionSchema,
        batch_data: dict[str, Any],
        state: dict[str, Any],
        loop_context: dict[str, Any],
        training: bool,
    ) -> None:
        """Execute a 'loop' action: iterate and recurse on body."""
        assert action.iterator is not None
        assert action.loop_var is not None
        iterator = self._resolve(action.iterator, batch_data, state, loop_context)
        loop_var = action.loop_var

        # Create a child loop context (don't mutate parent)
        child_ctx = dict(loop_context)

        for item in iterator:
            child_ctx[loop_var] = item
            # Recursively execute the loop body
            self._execute_flow(action.body, batch_data, state, child_ctx, training)

    def _execute_assign(
        self,
        action: ActionSchema,
        batch_data: dict[str, Any],
        state: dict[str, Any],
        loop_context: dict[str, Any],
    ) -> None:
        """Execute an 'assign' action: write to state memory bank."""
        assert action.target is not None
        assert action.source is not None

        # Extract state key (strip 'state.' prefix if present)
        target_key = action.target
        if target_key.startswith("state."):
            target_key = target_key[len("state.") :]

        value = self._resolve(action.source, batch_data, state, loop_context)
        state[target_key] = value

    def _execute_detach(
        self,
        action: ActionSchema,
        state: dict[str, Any],
    ) -> None:
        """Execute a 'detach' action: sever grad graph on state tensor."""
        assert action.target is not None

        target_key = action.target
        if target_key.startswith("state."):
            target_key = target_key[len("state.") :]

        if target_key not in state:
            raise ValueError(
                f"Detach: state key '{target_key}' not found. Available: {list(state.keys())}"
            )

        tensor = state[target_key]
        if isinstance(tensor, torch.Tensor):
            state[target_key] = tensor.detach()
        else:
            logger.warning(
                "[Multi] detach on non-tensor state '%s' (type=%s), skipping",
                target_key,
                type(tensor).__name__,
            )

    # ------------------------------------------------------------------
    # Sequential fallback (backward compat)
    # ------------------------------------------------------------------

    def _execute_sequential(
        self,
        input_tensor: torch.Tensor,
        training: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fallback: execute stages in order (no routines configured)."""
        intermediates: dict[str, torch.Tensor] = {}
        x = input_tensor

        for stage_name in self.stage_order:
            model = self.multi_stages[stage_name]
            stage_cfg = self.stage_configs[stage_name]
            is_frozen = (
                stage_cfg.training_state.freeze_all
                and not self._all_unfrozen
                and not stage_cfg.training_state.inject_lora
            )

            if is_frozen and training:
                # no_grad (not inference_mode) so the re-arm below is legal —
                # see _execute_action for the full rationale.
                with torch.no_grad():
                    x = model(x)
                x = x.detach().requires_grad_(True)
            else:
                x = model(x)

            intermediates[stage_name] = x

        return x, intermediates

    # ------------------------------------------------------------------
    # Run a routine
    # ------------------------------------------------------------------

    def run_routine(
        self,
        routine_name: str,
        batch_data: dict[str, Any],
        training: bool = True,
    ) -> dict[str, Any]:
        """Execute a named routine and return the state dict.

        Args:
            routine_name: Name of the routine (e.g. 'train', 'generate_3d').
            batch_data: Dict of tensors from the data loader.
            training: Whether gradients should flow for hot stages.

        Returns:
            State memory bank after execution.

        Raises:
            ValueError: If routine not found.
        """
        if routine_name not in self.routines:
            available = list(self.routines.keys())
            raise ValueError(f"Routine '{routine_name}' not found. Available: {available}")

        state: dict[str, Any] = {}
        self._execute_flow(self.routines[routine_name], batch_data, state, training=training)
        return state

    # ------------------------------------------------------------------
    # Epoch hooks
    # ------------------------------------------------------------------

    def on_epoch_start(self, epoch: int) -> None:
        """Check for end-to-end fine-tuning threshold."""
        super().on_epoch_start(epoch)

        if (
            self._finetune_epoch is not None
            and epoch >= self._finetune_epoch
            and not self._all_unfrozen
        ):
            logger.info(
                "[Multi] Epoch %d ≥ finetune_epoch %d → unfreezing ALL",
                epoch,
                self._finetune_epoch,
            )
            for stage_name, model in self.multi_stages.items():
                for param in model.parameters():
                    param.requires_grad = True
                logger.info("[Multi] Stage '%s': fully unfrozen", stage_name)
            self._trainable_params = self._collect_trainable_params()
            # Newly-unfrozen params must be handed to opt_g, else they keep
            # requires_grad=True but receive no updates. _register_global_params
            # dedups against existing groups, so this only adds the new ones.
            self._register_global_params()
            self._all_unfrozen = True
            # The unfreeze released every stage, early-stopped ones included.
            self._es_frozen: set[str] = set()

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Check per-stage early stopping conditions.

        Each stage with ``stage_config.early_stopping`` is evaluated
        against the corresponding validation metric. If patience is
        exceeded, freezes the stage and logs a warning.

        Args:
            epoch: Current epoch index.
            metrics: Validation metrics dict from ``validation_step``.
        """
        super().on_epoch_end(epoch, metrics)

        if not metrics or not self._stage_early_stopping:
            return

        for stage_name, es_config in self._stage_early_stopping.items():
            if not es_config.enabled:
                continue

            # Determine the monitor metric. The default is the key
            # ``validation_step`` actually emits for a stage:
            # ``metrics[f"val_{stage}_l1"]`` (line ~1319), which it writes only
            # when ``training.pipeline.evaluate_intermediates`` is on.
            #
            # This used to default to ``val_{stage}/l1`` -- a slash form that
            # NOTHING in the repository emits -- with the underscore form as a
            # hand-written fallback. So the primary lookup could never hit, and
            # the pair was a second resolver of an invariant the framework has
            # already elected an owner for (non-negotiable 17):
            # ``metric_keys.resolve_metric_key``, which the training loop's own
            # early stopping uses. Route through it instead of re-spelling it.
            monitor = getattr(es_config, "metric", None) or f"val_{stage_name}_l1"
            resolved = resolve_metric_key(monitor, metrics.keys())
            if resolved is None:
                # Absent is a state to REPORT, never one to infer (#18). A bare
                # ``continue`` here is how a declared early-stopping block spends
                # a whole run doing nothing while the startup log calls it
                # active. Once per stage per process, not once per epoch.
                if stage_name not in self._es_unresolved_monitors:
                    self._es_unresolved_monitors.add(stage_name)
                    logger.warning(
                        "[Multi] Stage '%s': early stopping is configured on "
                        "monitor '%s', but no alias of it is present in the "
                        "validation metrics (%s). The stage will never be "
                        "frozen. Per-stage L1 keys are emitted only when "
                        "training.pipeline.evaluate_intermediates is true.",
                        stage_name,
                        monitor,
                        sorted(metrics)[:12],
                    )
                continue
            current_val = float(metrics[resolved])
            monitor = resolved

            # Initialize tracking state
            if not hasattr(self, "_es_best"):
                self._es_best: dict[str, float] = {}
                self._es_wait: dict[str, int] = {}

            best = self._es_best.get(stage_name, float("inf"))
            if current_val < best:
                self._es_best[stage_name] = current_val
                self._es_wait[stage_name] = 0
            else:
                self._es_wait[stage_name] = self._es_wait.get(stage_name, 0) + 1

            wait = self._es_wait[stage_name]
            if wait >= es_config.patience:
                logger.warning(
                    "[Multi] Stage '%s': early stopping triggered "
                    "(patience=%d, monitor='%s', best=%.6f, current=%.6f). "
                    "Freezing stage.",
                    stage_name,
                    es_config.patience,
                    monitor,
                    self._es_best[stage_name],
                    current_val,
                )
                # Freeze the stage
                model = self.multi_stages[stage_name]
                for param in model.parameters():
                    param.requires_grad = False
                self._es_frozen = {*getattr(self, "_es_frozen", ()), stage_name}
                self._trainable_params = self._collect_trainable_params()
            elif wait > 0:
                logger.info(
                    "[Multi] Stage '%s': no improvement (%d/%d), "
                    "monitor='%s' best=%.6f current=%.6f",
                    stage_name,
                    wait,
                    es_config.patience,
                    monitor,
                    best,
                    current_val,
                )

    @property
    def _all_unfrozen(self) -> bool:
        """Whether E2E unfreezing has been triggered."""
        return getattr(self, "_has_unfrozen_all", False)

    @_all_unfrozen.setter
    def _all_unfrozen(self, value: bool) -> None:
        self._has_unfrozen_all = value

    # ------------------------------------------------------------------
    # Checkpoint state (#2962): the stages are models, the rest a strategy entry
    # ------------------------------------------------------------------

    def checkpoint_models(self) -> dict[str, nn.Module]:
        return pipeline_stage_state.stage_models(self)

    def strategy_state_dict(self) -> dict[str, Any]:
        state = super().strategy_state_dict()
        state[pipeline_stage_state.PIPELINE_KEY] = pipeline_stage_state.capture_pipeline_state(self)
        return state

    def check_strategy_state(self, state: dict[str, Any] | None) -> None:
        entry = (state or {}).get(pipeline_stage_state.PIPELINE_KEY)
        pipeline_stage_state.check_pipeline_state(self, entry)

    def load_strategy_state_dict(
        self, state: dict[str, Any] | None, *, load_optimizer: bool = True
    ) -> None:
        self.check_strategy_state(state)
        super().load_strategy_state_dict(state, load_optimizer=load_optimizer)
        # check_strategy_state refused a None state above.
        entry = cast("dict[str, Any]", state)[pipeline_stage_state.PIPELINE_KEY]
        pipeline_stage_state.restore_pipeline_state(self, entry, load_optimizer=load_optimizer)

    # ------------------------------------------------------------------
    # Per-stage framework integration (LossBuilder, Optimizers)
    # ------------------------------------------------------------------

    def _merge_stage_config(self, stage_cfg: MultiStageSchema) -> TrainingSettings:
        """Build a config proxy overlaying stage_config onto global config.

        Creates a new TrainingSettings instance where the ``losses``,
        ``optimization``, ``early_stopping``, and ``metrics`` sections
        are replaced by the stage-specific overrides from
        ``StageEnvironmentSchema``. This allows ``LossBuilder`` to work
        without modification (SSOT compliance).

        Args:
            stage_cfg: The stage schema containing a ``stage_config``.

        Returns:
            A new ``TrainingSettings`` with merged overrides.
        """
        base_data = self.config.model_dump(mode="python")
        stage_env = stage_cfg.stage_config

        if stage_env is None:
            return self.config

        if stage_env.loss is not None:
            base_data["losses"] = stage_env.loss.model_dump(mode="python")
        if stage_env.optimization is not None:
            opt_override = stage_env.optimization.model_dump(mode="python", exclude_none=True)
            existing_opt = base_data.get("optimization", {})
            if isinstance(existing_opt, dict):
                existing_opt.update(opt_override)
                base_data["optimization"] = existing_opt
            else:
                base_data["optimization"] = opt_override
        if stage_env.early_stopping is not None:
            base_data["early_stopping"] = stage_env.early_stopping.model_dump(mode="python")
        if stage_env.metrics is not None:
            base_data["metrics"] = stage_env.metrics.model_dump(mode="python")

        return cast(TrainingSettings, TrainingSettings.model_validate(base_data))

    def _build_stage_losses(self) -> None:
        """Build loss functions for each trainable stage via LossBuilder (SSOT).

        Stages without ``stage_config`` or ``stage_config.loss`` use the
        global loss configuration. Frozen stages with no loss config are
        skipped entirely (no loss computation needed).
        """
        for name, stage_cfg in self.stage_configs.items():
            # Skip fully frozen stages that have no loss config
            is_fully_frozen = (
                stage_cfg.training_state.freeze_all
                and not stage_cfg.training_state.inject_lora
                and not stage_cfg.training_state.unfreeze_modules
            )
            if is_fully_frozen and (
                stage_cfg.stage_config is None or stage_cfg.stage_config.loss is None
            ):
                logger.info(
                    "[Multi] Stage '%s': frozen, no loss config → skipping loss build",
                    name,
                )
                continue

            # Build merged config proxy. A merge/validation failure must raise
            # rather than silently degrading to the global config — falling back
            # would discard the stage's declared losses/optimization/early-
            # stopping/metrics overrides without any indication (pitfall #9/#10).
            try:
                merged_config = self._merge_stage_config(stage_cfg)
            except Exception as e:
                raise RuntimeError(f"Stage '{name}' config merge failed: {e}") from e

            # Use LossBuilder (same as BaseTrainingStrategy pipeline)
            builder = LossBuilder(config=merged_config, device=self.device)
            builder._build_all_dynamic()
            built_losses = builder.build()
            if built_losses:
                self._stage_losses[name] = built_losses
                self._stage_loss_weights[name] = builder.get_loss_weights()
                logger.info(
                    "[Multi] Stage '%s': built %d losses → %s",
                    name,
                    len(built_losses),
                    list(built_losses.keys()),
                )
            else:
                logger.info("[Multi] Stage '%s': no losses configured", name)

            # Store early stopping config if present
            if (
                stage_cfg.stage_config is not None
                and stage_cfg.stage_config.early_stopping is not None
            ):
                self._stage_early_stopping[name] = stage_cfg.stage_config.early_stopping

    def _build_stage_optimizers(self) -> None:
        """Build per-stage optimizers for stages with optimization overrides.

        Stages without ``stage_config.optimization`` share the global
        optimizer (managed by BaseTrainingStrategy). Stages with overrides
        get independent optimizer instances.
        """
        for name, stage_cfg in self.stage_configs.items():
            if stage_cfg.stage_config is None or stage_cfg.stage_config.optimization is None:
                continue

            model = self.multi_stages[name]
            trainable = [p for p in model.parameters() if p.requires_grad]
            if not trainable:
                continue

            opt_config = stage_cfg.stage_config.optimization
            lr = opt_config.optimizer.learning_rate
            opt_type = opt_config.optimizer.type
            weight_decay = opt_config.optimizer.weight_decay

            # Route through the OptimizerRegistry — raises ValueError on an
            # unknown optimizer_type instead of silently selecting AdamW (NN#3).
            optimizer = create_optimizer(
                opt_type,
                trainable,
                lr=lr,
                weight_decay=weight_decay,
            )

            self._stage_optimizers[name] = optimizer
            logger.info(
                "[Multi] Stage '%s': built %s optimizer (lr=%.2e)",
                name,
                opt_type,
                lr,
            )

    def _build_eval_metrics(self) -> None:
        """Dynamically instantiate evaluation metrics from MultiEvalSchema.

        Reads ``self._eval_config.metrics`` and uses ``importlib`` to import
        each metric class from its ``target`` path, then instantiates it
        with the provided ``params``.
        """
        if not self._eval_config or not self._eval_config.metrics:
            return

        for metric_name, metric_schema in self._eval_config.metrics.items():
            target_path = metric_schema.target
            params = dict(metric_schema.params)

            try:
                module_path, class_name = target_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                metric_class = getattr(module, class_name)
                metric_instance = metric_class(**params)
                self._eval_metrics[metric_name] = metric_instance
                self._eval_metric_schemas[metric_name] = metric_schema
                logger.info(
                    "[Multi] Eval metric '%s': %s(%s)",
                    metric_name,
                    class_name,
                    params,
                )
            except (ImportError, AttributeError, TypeError) as e:
                logger.warning(
                    "[Multi] Failed to instantiate eval metric '%s' from '%s': %s",
                    metric_name,
                    target_path,
                    e,
                )

    def _step_stage_optimizers(
        self,
        state: dict[str, Any],
        target_batch: torch.Tensor,
        epoch: int,
    ) -> dict[str, torch.Tensor]:
        """Run per-stage backward/step for stages with their own optimizers.

        For each stage that has a dedicated optimizer:
        1. Compute per-stage losses
        2. zero_grad on the stage optimizer
        3. backward on the stage's total loss
        4. step on the stage optimizer
        5. Return losses (detached) for logging

        Args:
            state: Interpreter state memory bank after routine execution.
            target_batch: Ground truth tensor.
            epoch: Current epoch index.

        Returns:
            Dict of ``{stage/loss_name: detached_loss}`` for logging only.
        """
        stepped_losses: dict[str, torch.Tensor] = {}
        stepped_stages: set[str] = set()

        for stage_name, optimizer in self._stage_optimizers.items():
            # Find the stage output in state
            output = state.get(stage_name)
            if not isinstance(output, torch.Tensor):
                continue
            out_tensor: torch.Tensor = output
            if out_tensor.shape != target_batch.shape:
                continue

            stage_losses = self._compute_stage_losses(
                stage_name, cast(torch.Tensor, output), target_batch
            )
            if not stage_losses:
                continue

            # Sum per-stage losses carefully for type checkers
            loss_vals = list(stage_losses.values())
            stage_total: torch.Tensor = loss_vals[0]
            for v in loss_vals[1:]:
                stage_total = stage_total + v

            # Per-stage backward/step cycle
            optimizer.zero_grad(set_to_none=True)
            stage_total.backward(retain_graph=True)  # type: ignore
            optimizer.step()

            # Store detached losses for logging (these are already stepped)
            for key, val in stage_losses.items():
                stepped_losses[key] = val.detach()
            stepped_stages.add(stage_name)

            # No .item() here: Python evaluates logger args eagerly regardless of
            # level, so a .item() in this per-step stage-optimizer path forces a
            # GPU sync every step (performance.md: no .item() in the loop). The
            # detached per-stage losses are already aggregated for epoch logging.
            logger.debug(
                "[Multi] Stage '%s': optimizer stepped",
                stage_name,
            )

        return stepped_losses

    def _compute_stage_losses(
        self,
        stage_name: str,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute losses for a single stage using its LossBuilder-built modules.

        Args:
            stage_name: Name of the stage.
            prediction: Model output tensor.
            target: Ground truth tensor.

        Returns:
            Dict of ``{stage_name/loss_name: loss_value}`` tensors.
        """
        stage_losses: dict[str, torch.Tensor] = {}
        if stage_name not in self._stage_losses:
            return stage_losses

        losses = self._stage_losses[stage_name]
        weights = self._stage_loss_weights.get(stage_name, {})

        for loss_name, loss_fn in losses.items():
            # No try/except here: a configured loss that errors must fail loudly
            # (pitfall #9/#10). Silently dropping the term would train on a
            # subset of the declared objective with a green-looking run.
            val = loss_fn(prediction, target)
            weight = weights.get(loss_name, 1.0)
            stage_losses[f"{stage_name}/{loss_name}"] = weight * val
        return stage_losses

    def _aggregate_stage_losses(
        self,
        state: dict[str, Any],
        target_batch: torch.Tensor,
        epoch: int,
    ) -> dict[str, torch.Tensor]:
        """Aggregate per-stage losses into a single loss dict with g_total_loss.

        Stages with their own optimizers are stepped independently via
        ``_step_stage_optimizers()`` — their losses appear in the return dict
        (detached) for logging but do NOT contribute to ``g_total_loss``.

        Stages without their own optimizer contribute to ``g_total_loss``,
        which is then handled by the Trainer's global optimizer.

        Args:
            state: The interpreter state memory bank after routine execution.
            target_batch: Ground truth tensor.
            epoch: Current epoch index.

        Returns:
            Aggregated loss dict with ``g_total_loss`` for the global backward pass.
        """
        all_losses: dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Step 1: Handle stages with their own optimizers
        stepped_losses = self._step_stage_optimizers(state, target_batch, epoch)
        all_losses.update(stepped_losses)

        # Track which stages were already stepped
        stepped_stages = set(self._stage_optimizers.keys())

        # Step 2: Aggregate losses for stages using the GLOBAL optimizer
        for key, output in state.items():
            if not isinstance(output, torch.Tensor):
                continue
            if output.shape != target_batch.shape:
                continue
            if key in stepped_stages:
                continue  # Already stepped independently

            stage_losses = self._compute_stage_losses(key, output, target_batch)
            for loss_key, loss_val in stage_losses.items():
                all_losses[loss_key] = loss_val
                total = total + loss_val

        # If no per-stage losses were configured, fall back to global losses
        if not all_losses:
            final_key = list(state.keys())[-1] if state else None
            final_output = state[final_key] if final_key else target_batch
            if isinstance(final_output, torch.Tensor):
                loss_l1 = F.l1_loss(final_output, target_batch)
                loss_mse = F.mse_loss(final_output, target_batch)
                all_losses["loss_l1"] = loss_l1
                all_losses["loss_mse"] = loss_mse
                lambda_l1 = self._get_loss_weight("l1", epoch)
                lambda_l2 = self._get_loss_weight("l2", epoch)
                total = lambda_l1 * loss_l1 + lambda_l2 * loss_mse

        all_losses["g_total_loss"] = total
        return all_losses

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Forward through stages and compute losses via LossBuilder (SSOT).

        Uses per-stage loss functions built by ``_build_stage_losses()``
        instead of hardcoded loss functions. Falls back to global L1/MSE
        if no per-stage losses are configured.
        """
        losses: dict[str, torch.Tensor] = {}

        if self._use_routines:
            # Build batch_data dict
            batch = kwargs.get("batch", {})
            if isinstance(batch, dict):
                batch_data = dict(batch)
                batch_data.setdefault("input", input_batch)
                batch_data.setdefault("target", target_batch)
            else:
                batch_data = {"input": input_batch, "target": target_batch}

            # Execute the 'train' routine (or the only available routine)
            routine_name = "train" if "train" in self.routines else next(iter(self.routines))
            state = self.run_routine(routine_name, batch_data, training=True)

            # Aggregate per-stage losses using LossBuilder-constructed modules
            losses = self._aggregate_stage_losses(state, target_batch, epoch)

            # Per-stage intermediate monitoring losses (always detached)
            if self._evaluate_intermediates:
                for key, output in state.items():
                    if isinstance(output, torch.Tensor) and output.shape == target_batch.shape:
                        with torch.no_grad():
                            losses[f"stage_{key}_l1"] = F.l1_loss(output.detach(), target_batch)
        else:
            # Sequential fallback
            final_output, intermediates = self._execute_sequential(input_batch, training=True)

            # Aggregate losses from intermediates
            losses = self._aggregate_stage_losses(intermediates, target_batch, epoch)

            if self._evaluate_intermediates:
                for stage_name, output in intermediates.items():
                    with torch.no_grad():
                        losses[f"stage_{stage_name}_l1"] = F.l1_loss(output.detach(), target_batch)

        return losses

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validate through the multi-stage graph.

        Uses per-stage loss functions (LossBuilder-constructed) for
        evaluation metrics. Falls back to global L1/MSE if no per-stage
        losses are configured.
        """
        val_batch = (input_batch, target_batch)
        batch_idx = kwargs.get("batch_idx", 0)
        if isinstance(val_batch, (list, tuple)):
            input_data, target = val_batch[0], val_batch[1]
        elif isinstance(val_batch, dict):
            input_data = val_batch.get("input", val_batch.get("lr"))
            target = val_batch.get("target", val_batch.get("hr"))
        else:
            input_data = target = val_batch

        input_data = self._to_device(input_data)
        target = self._to_device(target)

        with torch.no_grad():
            if self._use_routines:
                batch_data = {"input": input_data, "target": target}
                if isinstance(val_batch, dict):
                    batch_data.update(val_batch)

                # Use 'evaluate' routine if available, else 'train'
                routine_name = (
                    "evaluate"
                    if "evaluate" in self.routines
                    else ("train" if "train" in self.routines else next(iter(self.routines)))
                )
                state = self.run_routine(routine_name, batch_data, training=False)
                final_key = list(state.keys())[-1]
                final_output = state[final_key]
                intermediates = state
            else:
                final_output, intermediates = self._execute_sequential(input_data, training=False)

            # Sliding window for 3D
            if self._eval_config and self._eval_config.sliding_window:
                sw_cfg = self._eval_config.sliding_window
                final_output = self._apply_sliding_window(input_data, sw_cfg)

            metrics: dict[str, float] = {}

            # Per-stage validation losses via LossBuilder modules
            for key, output in intermediates.items():
                if not isinstance(output, torch.Tensor):
                    continue
                if output.shape != target.shape:
                    continue
                stage_losses = self._compute_stage_losses(key, output, target)
                for loss_key, loss_val in stage_losses.items():
                    metrics[f"val_{loss_key}"] = loss_val.item()

            # Global validation metrics (always computed)
            metrics["val_l1"] = F.l1_loss(final_output, target).item()
            metrics["val_mse"] = F.mse_loss(final_output, target).item()

            # Custom eval metrics from MultiEvalSchema.metrics
            if self._eval_metrics:
                for metric_name, metric_fn in self._eval_metrics.items():
                    try:
                        # Resolve custom inputs if schema provides them
                        schema = self._eval_metric_schemas.get(metric_name)
                        if schema and schema.inputs:
                            # Build kwargs from namespace references
                            metric_kwargs: dict[str, Any] = {}
                            for param, source_ref in schema.inputs.items():
                                metric_kwargs[param] = self._resolve(
                                    source_ref,
                                    batch_data if self._use_routines else {},
                                    intermediates,
                                    {},
                                )
                            result = metric_fn(**metric_kwargs)
                        else:
                            # Default: pass final output and target
                            result = metric_fn(final_output, target)

                        if isinstance(result, torch.Tensor):
                            result = result.mean().item()
                        metrics[f"val_{metric_name}"] = float(result)
                    except Exception as e:
                        logger.warning(
                            "[Multi] Eval metric '%s' failed: %s",
                            metric_name,
                            e,
                        )

            if self._evaluate_intermediates:
                for key, output in intermediates.items():
                    if isinstance(output, torch.Tensor) and output.shape == target.shape:
                        metrics[f"val_{key}_l1"] = F.l1_loss(output, target).item()

        return metrics

    def _apply_sliding_window(self, inputs: torch.Tensor, sw_cfg: Any) -> torch.Tensor:
        """Apply sliding window inference using the full pipeline."""
        roi_size = tuple(sw_cfg.roi_size)

        def predictor(x: torch.Tensor) -> torch.Tensor:
            if self._use_routines:
                batch_data = {"input": x}
                routine_name = (
                    "evaluate" if "evaluate" in self.routines else next(iter(self.routines))
                )
                state = self.run_routine(routine_name, batch_data, training=False)
                return state[list(state.keys())[-1]]
            else:
                output, _ = self._execute_sequential(x, training=False)
                return output

        if len(roi_size) == 3:
            from spectramr.infrastructure.inference.sliding_window import (
                sliding_window_inference,
            )

            return sliding_window_inference(
                inputs=inputs,
                roi_size=roi_size,
                predictor=predictor,
                sw_batch_size=sw_cfg.sw_batch_size,
                overlap=sw_cfg.overlap,
                mode=sw_cfg.mode,
            )
        else:
            from spectramr.infrastructure.inference.sliding_window import (
                sliding_window_inference_2d,
            )

            return sliding_window_inference_2d(
                inputs=inputs,
                roi_size=roi_size,
                predictor=predictor,
                overlap=sw_cfg.overlap,
                mode=sw_cfg.mode,
            )


# Backward compatibility alias
PipelineTrainingStrategy = MultiTrainingStrategy

__all__ = ["MultiTrainingStrategy", "PipelineTrainingStrategy"]
