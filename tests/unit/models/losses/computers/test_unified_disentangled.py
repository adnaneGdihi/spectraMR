"""Tests for ``UnifiedDisentangledLossComputer``'s config-field mapping.

``_get_loss_weight`` translates a loss-component name into a
``ReconstructionLossesConfig`` field name and then reads it with ``getattr``.
A field name that no longer exists therefore does not fail -- it returns the
``getattr`` default, so the loss keeps running at a weight nobody chose. That is
the failure mode this file exists to make impossible, and it is a live risk
because #421 renamed one of these fields.

The mapping is read STATICALLY (AST) rather than by constructing the computer:
the dict is a literal inside a method, and the assertion is about field names, so
building a model would add a GPU-shaped dependency for no extra coverage.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from spectramr.config.schemas.loss import ReconstructionLossesConfig

MODULE = (
    Path(__file__).resolve().parents[5]
    / "src/spectramr/models/losses/computers/unified_disentangled.py"
)


def _weight_mapping() -> dict[str, str]:
    """The ``{loss_component: config_field}`` literal inside ``_get_loss_weight``."""
    tree = ast.parse(MODULE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_get_loss_weight":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict) and all(
                    isinstance(k, ast.Constant) and isinstance(v, ast.Constant)
                    for k, v in zip(sub.keys, sub.values, strict=True)
                ):
                    mapping = {
                        k.value: v.value
                        for k, v in zip(sub.keys, sub.values, strict=True)
                    }
                    if any(str(v).startswith("lambda_") for v in mapping.values()):
                        return mapping
    pytest.fail("could not locate the loss-name -> config-field mapping")


#: Components whose mapped field exists on NEITHER section `_get_loss_weight`
#: searches, so the lookup falls through to its `return 1.0`. That is a silent
#: default substitution (non-negotiable 3): the loss trains at a weight nobody
#: chose, and no config declaration can change it because no such field exists.
#: Baselined, not forgiven -- pre-existing debt found by this guard while landing
#: #421, tracked separately. The set may only SHRINK; `lambda_patch_nce` now exists on
#: `LatentLossesConfig`, so it is empty.
KNOWN_UNRESOLVABLE: dict[str, str] = {}


def _searched_fields() -> set[str]:
    """The field names ``_get_loss_weight`` can actually resolve.

    It walks ``[losses.reconstruction, losses.latent]`` and takes the first
    section carrying the attribute, so a target on EITHER resolves. A guard that
    checked only `reconstruction` would report `lambda_kl` (a `latent` field) as
    broken -- the mapping and the lookup have to be read together.
    """
    from spectramr.config.schemas.loss import LatentLossesConfig

    return set(ReconstructionLossesConfig.model_fields) | set(LatentLossesConfig.model_fields)


class TestMappingTargetsAreRealSchemaFields:
    def test_no_new_unresolvable_mapping_targets(self):
        """A stale target is invisible: the lookup falls through to 1.0, silently.

        This is the assertion that would have caught a plain deletion of
        ``lambda_content`` during #421 -- the weight would have vanished with no
        error, and ``content_consistency`` would have trained at 1.0 rather than
        at the author's value.
        """
        unresolvable = {
            component: field
            for component, field in _weight_mapping().items()
            if field not in _searched_fields()
        }
        assert unresolvable == KNOWN_UNRESOLVABLE, (
            f"unresolvable mapping targets changed: {unresolvable}. `_get_loss_weight` "
            "searches losses.reconstruction then losses.latent and returns 1.0 when "
            "neither carries the field, so a new entry here is a loss silently "
            "training at an unchosen weight. If you FIXED one, shrink "
            "KNOWN_UNRESOLVABLE."
        )

    def test_content_consistency_reads_the_renamed_field(self):
        """#421 split an overloaded key; this consumer must follow it.

        ``lambda_content`` served two different losses -- the VGG perceptual
        weight and this disentanglement term -- because
        ``LossRegistry._aliases`` maps ``content -> perceptual`` while
        ``KNOWN_LOSS_COMPONENTS`` lists ``content_consistency`` and
        ``perceptual`` as separate components.
        """
        assert _weight_mapping()["content_consistency"] == "lambda_content_consistency"

    def test_perceptual_still_reads_its_own_field(self):
        """Negative control: the two components must not collapse onto one field."""
        mapping = _weight_mapping()
        assert mapping["perceptual"] == "lambda_perceptual"
        assert mapping["content_consistency"] != mapping["perceptual"]

    def test_both_components_are_declared_distinct(self):
        from spectramr.models.losses.computers.unified_disentangled import (
            KNOWN_LOSS_COMPONENTS,
        )

        assert "content_consistency" in KNOWN_LOSS_COMPONENTS
        assert "perceptual" in KNOWN_LOSS_COMPONENTS


def _weighted_computer(**latent):
    """The computer with only the config sections ``_get_loss_weight`` searches."""
    from types import SimpleNamespace

    import torch.nn as nn

    from spectramr.config.schemas.loss import LatentLossesConfig
    from spectramr.models.losses.computers.unified_disentangled import (
        UnifiedDisentangledLossComputer,
    )

    comp = UnifiedDisentangledLossComputer.__new__(UnifiedDisentangledLossComputer)
    nn.Module.__init__(comp)
    comp.config = SimpleNamespace(
        losses=SimpleNamespace(
            reconstruction=ReconstructionLossesConfig(), latent=LatentLossesConfig(**latent)
        )
    )
    return comp


def test_patch_nce_trains_at_the_declared_weight() -> None:
    """The term was always on at a weight no config could change: the field did not exist."""
    assert _weighted_computer(lambda_patch_nce=0.25)._get_loss_weight("patch_nce") == 0.25
    assert _weighted_computer(lambda_patch_nce=0.0)._get_loss_weight("patch_nce") == 0.0


def test_patch_nce_default_keeps_the_weight_every_earlier_arm_trained_at() -> None:
    assert _weighted_computer()._get_loss_weight("patch_nce") == 1.0


def _recon_computer(loss_type: str):
    """The computer with only what _compute_recon_losses reads (#2016)."""
    from types import SimpleNamespace

    import torch.nn as nn

    from spectramr.models.losses.computers.unified_disentangled import (
        UnifiedDisentangledLossComputer,
    )

    comp = UnifiedDisentangledLossComputer.__new__(UnifiedDisentangledLossComputer)
    nn.Module.__init__(comp)
    comp.losses = {"mse": nn.MSELoss()}
    comp.config = SimpleNamespace(
        losses=SimpleNamespace(reconstruction=SimpleNamespace(loss_type=loss_type))
    )
    return comp


def test_recon_trains_the_declared_loss() -> None:
    import torch

    pred, target = torch.full((1, 1, 2, 2), 2.0), torch.zeros(1, 1, 2, 2)
    out = _recon_computer("mse")._compute_recon_losses(pred, target)
    assert torch.isclose(out["recon_a"], torch.tensor(4.0))  # mse; the old L1 gave 2.0


def test_an_unbuilt_loss_type_is_refused_at_construction(monkeypatch) -> None:
    """Fail where audit and smoke see it, not at the first training step (#2016)."""
    from types import SimpleNamespace

    import torch.nn as nn

    import spectramr.infrastructure.training.builders.loss_builder as lb
    from spectramr.models.losses.computers.unified_disentangled import (
        UnifiedDisentangledLossComputer,
    )

    class _Builder:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return lambda: self if name != "build" else {"mse": nn.MSELoss()}

    monkeypatch.setattr(lb, "LossBuilder", _Builder)
    cfg = SimpleNamespace(losses=SimpleNamespace(reconstruction=SimpleNamespace(loss_type="l1")))
    with pytest.raises(ValueError, match="not among the built losses"):
        UnifiedDisentangledLossComputer(cfg)


def test_recon_refuses_a_loss_type_that_was_not_built() -> None:
    """The old path built a bare nn.L1Loss weighted 10.0 that no YAML declared (#2016)."""
    import torch

    with pytest.raises(ValueError, match="not among the built losses"):
        _recon_computer("l1")._compute_recon_losses(torch.ones(1), torch.zeros(1))
