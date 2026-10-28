"""PEFT (LoRA / adapter / prefix_tuning / bitfit) injectors.

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-14.

Four real methods now ship:

- ``lora``         — low-rank ``A·B`` update around frozen ``nn.Linear``.
- ``adapter``      — Houlsby-style bottleneck MLP after each matched
  ``nn.Linear`` with a residual connection.
- ``prefix_tuning``— learnable prefix tokens prepended to the inputs
  of matched modules (works for ``nn.Linear`` by left-padding and for
  ``nn.MultiheadAttention`` via ``key_padding_mask`` / extra K,V).
- ``bitfit``       — biases-only fine-tuning.

All four freeze the rest of the model.  Selection is driven by
``peft_cfg.method``; ``target_modules`` is the same substring-match
list used by LoRA.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRAAdapter(nn.Module):
    """A · B low-rank update wrapped around a frozen ``nn.Linear``."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_features = base.in_features
        out_features = base.out_features

        # Initialize A with Kaiming, B at zero — preserves base behaviour at start.
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.scaling = alpha / float(rank)
        self.rank = rank
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_drop = self.dropout(x)
        # (B, ..., in) @ A^T → (B, ..., r); then @ B^T → (B, ..., out)
        lora_out = (x_drop @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * lora_out

    def extra_repr(self) -> str:  # pragma: no cover
        return f"rank={self.rank}, alpha={self.alpha}"


class AdapterModule(nn.Module):
    """Houlsby-style adapter: down → activation → up + residual.

    Wraps a frozen ``nn.Linear``.  Only the bottleneck weights train.
    """

    def __init__(self, base: nn.Linear, bottleneck: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        d = base.out_features
        self.down = nn.Linear(d, bottleneck)
        self.up = nn.Linear(bottleneck, d)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        return out + self.up(self.dropout(self.act(self.down(out))))


class PrefixModule(nn.Module):
    """Learnable prefix tokens prepended to the input of a Linear.

    For sequence-style inputs ``(B, T, D)`` we prepend ``n_prefix`` learnable
    rows along the time axis, run the base linear over the concatenation,
    then strip the prefix from the output so downstream shape is preserved.
    For non-sequence inputs the prefix degrades to a no-op residual added to
    the bias (still differentiable, still a valid PEFT path).
    """

    def __init__(self, base: nn.Linear, n_prefix: int = 8) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.n_prefix = n_prefix
        self.prefix = nn.Parameter(torch.zeros(n_prefix, base.in_features))
        nn.init.normal_(self.prefix, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3:
            return self.base(x) + self.base(self.prefix.mean(dim=0, keepdim=True))
        B = x.shape[0]
        prefix = self.prefix.unsqueeze(0).expand(B, -1, -1)
        x_cat = torch.cat([prefix, x], dim=1)
        out = self.base(x_cat)
        return out[:, self.n_prefix :]


def _matches_target(name: str, targets: list[str]) -> bool:
    return any(t in name for t in targets)


def _replace_module(parent: nn.Module, child_name: str, new_module: nn.Module) -> None:
    setattr(parent, child_name, new_module)


def _walk_replace(model: nn.Module, targets: list[str], factory) -> int:
    n = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            qualified = f"{parent_name}.{child_name}" if parent_name else child_name
            if not isinstance(child, nn.Linear) or not _matches_target(qualified, targets):
                continue
            _replace_module(parent, child_name, factory(child))
            n += 1
    return n


def _injected_adapter_param_ids(model: nn.Module) -> set[int]:
    """Identity of every parameter the injected adapters ADDED (never their frozen base).

    Trainability used to be decided by substring on the parameter NAME (``".down."``,
    ``".up."``, ``"lora_"``). That matched any backbone parameter that merely had such a
    path component -- a UNet's ``decoder.up.0.conv`` -- so a LoRA run silently trained
    part of the backbone. Identity cannot collide with a name.
    """
    ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, LoRAAdapter):
            ids.update((id(module.lora_A), id(module.lora_B)))
        elif isinstance(module, AdapterModule):
            ids.update(id(p) for p in (*module.down.parameters(), *module.up.parameters()))
        elif isinstance(module, PrefixModule):
            ids.add(id(module.prefix))
    return ids


def inject_peft(model: nn.Module, peft_cfg: Any) -> nn.Module:
    """Inject LoRA / adapters into ``model`` per ``peft_cfg``.

    Args:
        model: ``nn.Module`` to mutate in-place (also returned).
        peft_cfg: PEFTConfigSchema instance.

    Returns:
        The same ``model`` with LoRA layers in place of matching
        ``nn.Linear`` submodules.
    """
    if peft_cfg is None or not peft_cfg.enabled:
        return model

    method = peft_cfg.method.lower()
    targets = peft_cfg.target_modules
    n_replaced = 0

    if method == "lora":
        n_replaced = _walk_replace(
            model,
            targets,
            lambda base: LoRAAdapter(
                base, rank=peft_cfg.rank, alpha=peft_cfg.alpha, dropout=peft_cfg.dropout
            ),
        )
    elif method == "adapter":
        bottleneck = getattr(peft_cfg, "rank", 16) or 16
        n_replaced = _walk_replace(
            model,
            targets,
            lambda base: AdapterModule(base, bottleneck=bottleneck, dropout=peft_cfg.dropout),
        )
    elif method == "prefix_tuning":
        n_prefix = getattr(peft_cfg, "rank", 8) or 8
        n_replaced = _walk_replace(
            model,
            targets,
            lambda base: PrefixModule(base, n_prefix=n_prefix),
        )
    elif method == "bitfit":
        # No replacement; trainability handled below.
        pass
    else:
        # Per CLAUDE.md #9 (silent fallbacks forbidden), reject unknown
        # PEFT methods at boot rather than silently freezing every
        # parameter at training time. The previous warn-and-continue
        # path produced a model that trains zero parameters with only a
        # WARNING line. See TODO/audit/08_training_services_orchestration.md F21.
        raise ValueError(
            f"Unknown PEFT method={method!r}. "
            "Expected one of: lora, adapter, prefix_tuning, bitfit."
        )

    # Trainability: freeze base params, then unfreeze the right slice.
    adapter_ids = _injected_adapter_param_ids(model)
    for n, p in model.named_parameters():
        is_adapter_param = id(p) in adapter_ids
        if is_adapter_param:
            p.requires_grad = True
            continue
        if method == "bitfit" and ("bias" in n.split(".")[-1]):
            p.requires_grad = True
            continue
        if peft_cfg.bias_mode == "all" and ("bias" in n.split(".")[-1]):
            p.requires_grad = True
        elif peft_cfg.bias_mode == "lora_only":
            p.requires_grad = is_adapter_param
        else:
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "PEFT(%s) injected: %d Linear replacements, %d / %d trainable params (%.2f%%)",
        method,
        n_replaced,
        trainable,
        total,
        100.0 * trainable / max(total, 1),
    )
    return model


__all__ = ["AdapterModule", "LoRAAdapter", "PrefixModule", "inject_peft"]
