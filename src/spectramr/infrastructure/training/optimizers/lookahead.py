"""Lookahead (Zhang et al., 2019) — a wrapper, not an optimizer name.

Keeps a slow copy of the weights and, every ``k`` steps, pulls the fast weights
toward it: ``slow += alpha * (fast - slow)``, then ``fast = slow``.

It is reached through ``optimization.optimizer.lookahead``, **not** through
``optimizer_type``, because a Lookahead with no inner optimizer is meaningless —
registering it under a name would advertise something unconstructible.

Moved here from ``gradient_stability.py``, where it was implemented, tested, and
completely unreachable from YAML: the only constructor that took ``use_lookahead``
was ``GradientStabilityManager.create_optimizer``, which had no production caller
anywhere in ``src/``. ``gradient_stability`` keeps a re-export so its existing
tests and any external import keep resolving.

One trap worth naming, because it is invisible until an LR schedule misbehaves:
``__init__`` builds a *copy* of the inner optimizer's ``param_groups``, and the
scheduler is attached to the WRAPPER. A scheduler stepping the wrapper therefore
writes ``group["lr"]`` on the copy, and the inner optimizer never sees it — the
run trains at the initial LR forever while the logs draw a perfect decay curve.
``step()`` closes this by syncing the wrapper's group hyperparameters onto the
inner optimizer before delegating (:meth:`Lookahead._sync_hyperparams_to_inner`).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any, overload

import torch
from torch.optim import Optimizer

__all__ = ["Lookahead"]


class Lookahead(Optimizer):
    """PyTorch implementation of the Lookahead optimizer.
    Reference: https://arxiv.org/abs/1907.08610
    """

    def __init__(self, optimizer: Optimizer, la_steps: int = 5, la_alpha: float = 0.5) -> None:
        """__init__.

        Args:
            optimizer (Any): Description.
            la_steps (Any): Description.
            la_alpha (Any): Description.
        """
        if not 0.0 <= la_alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {la_alpha}")
        if not la_steps >= 1:
            raise ValueError(f"Invalid lookahead steps: {la_steps}")

        # Store optimizer reference first
        self.optimizer = optimizer
        self.la_steps = la_steps
        self.la_alpha = la_alpha

        # Create a copy of param_groups to avoid conflicts with the inner
        # optimizer
        param_groups_copy = []
        for group in optimizer.param_groups:
            group_copy = dict(group)  # Shallow copy of the group dict
            group_copy["params"] = list(group["params"])  # Copy the params list
            param_groups_copy.append(group_copy)

        # A copy of the defaults: the update below would otherwise write
        # la_steps/la_alpha into the inner optimizer's dict, and from there into
        # every group it adds.
        super().__init__(param_groups_copy, dict(optimizer.defaults))

        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        # Update defaults to include lookahead parameters
        self.defaults.update({"la_steps": la_steps, "la_alpha": la_alpha})
        for group in self.param_groups:
            group["counter"] = 0
        self._pushed = [self._hyperparams(group) for group in self.param_groups]
        self._constructed = True

    @staticmethod
    def _hyperparams(group: dict[str, Any]) -> dict[str, Any]:
        """The group entries the sync may push inward, as the objects now held."""
        return {k: v for k, v in group.items() if k not in ("params", "counter")}

    def _update_slow(self, group: dict[str, Any]) -> None:
        """_update_slow.

        Args:
            group (Any): Description.
        Returns:
            Any: Description.
        """
        for p in group["params"]:
            if p.grad is None:
                continue
            param_state = self.state[p]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.clone(p.data)
            slow = param_state["slow_param"]
            slow += self.la_alpha * (p.data - slow)
            p.data.copy_(slow)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """step.

        Args:
            closure (Any): Description.
        Returns:
            Any: Description.
        """
        self._sync_hyperparams_to_inner()
        loss: float | None = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self._update_slow(group)
            group["counter"] = (group["counter"] + 1) % self.la_steps
        return loss

    def _sync_hyperparams_to_inner(self) -> None:
        """Push this wrapper's group hyperparameters onto the inner optimizer.

        ``__init__`` copies ``param_groups`` (see the module docstring), and a
        scheduler is attached to the WRAPPER -- ``optimization_builder`` hands it
        ``self._optimizers[name]``, which is this object. So without this sync
        every ``scheduler.step()`` decayed a copy nobody reads: the wrapper's
        reported LR fell while the inner optimizer kept stepping at the initial
        LR forever. Training proceeded and the logs showed a textbook decay
        curve, which is why nothing caught it.

        Copying at step() time rather than aliasing the group dicts keeps
        Lookahead's own bookkeeping (``counter``) out of the inner optimizer,
        and picks up ANY scheduler write -- lr, weight_decay, momentum, betas --
        not just the ones we thought to enumerate.

        Only entries the wrapper has *reassigned* since the last push are
        copied, detected by object identity (a scheduler writes a new value, it
        does not mutate the old one). Copying every shared key overwrote group
        state the inner optimizer advances itself: schedule-free's ``k`` and
        ``weight_sum`` stayed at 1 and 1e-4, and its ``train_mode`` was reset to
        the stale ``False`` before every step.
        """
        for outer, inner, pushed in zip(
            self.param_groups, self.optimizer.param_groups, self._pushed, strict=True
        ):
            for key, value in self._hyperparams(outer).items():
                if key in inner and pushed.get(key) is not value:
                    inner[key] = value
                    pushed[key] = value

    def state_dict(self) -> dict[str, Any]:
        """state_dict.

        Returns:
            Any: Description.
        """
        # The saved groups are the inner optimizer's, which see a scheduler
        # write only at the next step(); unsynced, a resume restored the LR from
        # one scheduler step earlier and a recursive schedule kept that offset.
        self._sync_hyperparams_to_inner()
        fast_state_dict = self.optimizer.state_dict()
        # Save slow parameters by parameter index within each group
        slow_state = {}
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, p in enumerate(group["params"]):
                if p in self.state and "slow_param" in self.state[p]:
                    key = f"group_{group_idx}_param_{param_idx}"
                    slow_state[key] = self.state[p]["slow_param"]

        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        # Persist the per-group sync counters; without them a resumed
        # Lookahead resets to 0 and syncs the slow weights one step too early,
        # corrupting the (la_steps-periodic) update cadence.
        counters = [group.get("counter", 0) for group in self.param_groups]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
            "counters": counters,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Load fast state into inner optimizer
        """load_state_dict.

        Args:
            state_dict (Any): Description.
        Returns:
            Any: Description.
        """
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

        # Load slow state into self.state
        self.state.clear()
        for key, slow_param in state_dict["slow_state"].items():
            # Parse the key to get group and param indices
            parts = key.split("_")
            if len(parts) == 4 and parts[0] == "group" and parts[2] == "param":
                group_idx = int(parts[1])
                param_idx = int(parts[3])
                if group_idx < len(self.param_groups):
                    group = self.param_groups[group_idx]
                    if param_idx < len(group["params"]):
                        p = group["params"][param_idx]
                        self.state[p] = {"slow_param": slow_param}

        # Restore the group HYPERPARAMETERS only -- never the ``params`` lists.
        #
        # ``state_dict["param_groups"]`` comes from the inner optimizer's
        # ``state_dict()``, where ``params`` holds integer INDICES, not tensors.
        # Installing it wholesale and then repairing positionally from a flat
        # ``list(self.state.keys())`` was wrong twice over: the flat list was
        # indexed by the position *within* a group, so group 1 slot 0 resolved
        # to group 0's first tensor (the second module then silently stopped
        # receiving the outer update while the first was slow-synced twice);
        # and a parameter that never entered ``slow_state`` -- ``_update_slow``
        # skips ``p.grad is None``, so any frozen parameter -- had no entry to
        # repair from, leaving the raw ``int`` in ``param_groups`` until the
        # step where ``counter`` wrapped raised
        # ``AttributeError: 'int' object has no attribute 'grad'``.
        #
        # ``self.param_groups`` already holds the correct tensors: this object
        # was constructed around the live model, and resume only needs to carry
        # the hyperparameters forward.
        # strict=True rather than truncating to the shorter list: zipping past a
        # mismatch would restore some groups' hyperparameters and leave the rest
        # at construction-time values, i.e. a partially-resumed optimizer that
        # reports success. In practice the inner optimizer's own
        # load_state_dict above rejects a group-count mismatch first; this keeps
        # the invariant local instead of relying on that ordering.
        for group, saved in zip(self.param_groups, state_dict["param_groups"], strict=True):
            for key, value in saved.items():
                if key == "params":
                    continue
                group[key] = value

        # Restore the per-group sync counters so the slow-weight update cadence
        # resumes where it left off (falling back to 0 for older checkpoints
        # that predate counter persistence).
        saved_counters = state_dict.get("counters", [])
        for idx, group in enumerate(self.param_groups):
            if idx < len(saved_counters):
                group["counter"] = saved_counters[idx]
            elif "counter" not in group:
                group["counter"] = 0
        # The inner optimizer loaded the same group values above.
        self._pushed = [self._hyperparams(group) for group in self.param_groups]

    def __getattr__(self, name: str) -> Any:
        """Forward the schedule-free ``train()``/``eval()`` mode API inward.

        Schedule-free optimizers keep an averaged sequence separate from the
        iterate the gradient is taken at, and ``optimizer.eval()`` /
        ``optimizer.train()`` swap between them. ``_set_optimizer_eval_mode``
        in the training loop finds those methods by duck-typing the object in
        ``pipeline.optimizers`` -- which, once Lookahead is enabled, is THIS
        wrapper.

        ``Optimizer`` defines neither method and this class defined no
        forwarding, so ``optimizer.type: schedulefree_adamw`` together with
        ``optimizer.lookahead.enabled: true`` made the hook find nothing:
        validation silently measured the un-averaged iterate, which reads as
        "schedule-free just underperforms here" rather than as a bug. Same
        family as the ``_sync_hyperparams_to_inner`` trap above -- wrapping an
        optimizer drops whatever part of its API the wrapper forgets to relay.

        Scoped to these two names rather than a blanket forward: a catch-all
        would relay arbitrary attributes, masking real ``AttributeError``s and
        making ``hasattr`` lie about capabilities the wrapper does not have.
        Reads ``__dict__`` directly because ``__getattr__`` fires for
        ``optimizer`` itself before ``__init__`` binds it, and ``self.optimizer``
        here would recurse.
        """
        if name in ("train", "eval"):
            hook = getattr(self.__dict__.get("optimizer"), name, None)
            if callable(hook):
                return hook
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add a group to the inner optimizer and mirror it here.

        Only the inner optimizer steps, so a group added to the wrapper alone
        was never trained: 29 strategies register their own parameters this way
        (``strategies/base.py``). ``Optimizer.__init__`` also routes the initial
        groups through this method, before ``_constructed`` is set; those are
        already in the inner optimizer and are only mirrored.
        """
        if not self.__dict__.get("_constructed", False):
            param_group["counter"] = 0
            super().add_param_group(param_group)
            return
        self.optimizer.add_param_group(
            {k: v for k, v in param_group.items() if k != "counter"}
        )
        added = self.optimizer.param_groups[-1]
        mirror = dict(added)
        mirror["params"] = list(added["params"])
        mirror["counter"] = 0
        super().add_param_group(mirror)
        self._pushed.append(self._hyperparams(self.param_groups[-1]))
