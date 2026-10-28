"""The physics kwargs the diffusion strategy hands to every declarative loss.

``_call_safe_loss`` filters by the CALLEE's parameter names, so a term reaches
its physics inputs only if the caller supplied them under a name that term
declares. ``sense_adjoint_l1`` names ``smaps``; the coil-manifold barrier names
``coil_sensitivities`` and has no ``**kwargs``, so it bound ``None`` and raised.

The reconciliation lives in ``core.coil_map_names.kwargs_accepted_by`` -- called by
``_call_safe_loss`` and by the Fourier bridge in front of every bridged term --
NOT in the caller: the strategy supplies ONE spelling and the five-way table has
one owner (non-negotiable 17). This module pins both halves -- that the strategy
still supplies the keys only it can know (``mask``, and maps under some alias),
and that the resolver re-files them for a callee spelling them differently.

The AST half asserts the PRODUCER, because ``test_coil_subspace_residual.py``
hands the key in by hand and therefore stayed green while the strategy sent
nothing. Parsed rather than grepped: a substring match anywhere in a 4000-line
module would pass on a comment.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from spectramr.core.coil_map_names import COIL_MAP_ALIASES, coil_map_kwargs_for
from spectramr.infrastructure.training.strategies import diffusion as _diffusion_module


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(Path(inspect.getsourcefile(_diffusion_module)).read_text())


def _training_compute_call(tree: ast.Module) -> ast.Call:
    """The ``...compute(...)`` that carries the generator's loss kwargs."""
    hits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compute"
        and any(kw.arg == "smaps" for kw in node.keywords)
    ]
    assert len(hits) == 1, f"expected one loss-computer compute() call, found {len(hits)}"
    return hits[0]


def _validation_loss_kwargs(tree: ast.Module) -> ast.Dict:
    """The hand-built ``loss_kwargs`` dict on the validation path."""
    hits = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign | ast.Assign)
        and isinstance(node.value, ast.Dict)
        and any(
            isinstance(t, ast.Name) and t.id == "loss_kwargs"
            for t in ([node.target] if isinstance(node, ast.AnnAssign) else node.targets)
        )
    ]
    assert len(hits) == 1, f"expected one loss_kwargs dict, found {len(hits)}"
    return hits[0]


def _val_keys(tree: ast.Module) -> set[str]:
    return {k.value for k in _validation_loss_kwargs(tree).keys if isinstance(k, ast.Constant)}


class TestBothSitesSupplyWhatOnlyTheyKnow:
    """Two bags, built independently -- keyword arguments to ``compute`` on the
    training path, a hand-written dict on the validation path. A key added to one
    and not the other gives a term that trains and then raises in validation."""

    @pytest.mark.parametrize("key", ["mask"])
    def test_both_bags_carry_it(self, tree, key) -> None:
        training = {kw.arg for kw in _training_compute_call(tree).keywords}
        assert key in training, f"the training loss kwargs drop {key!r}"
        assert key in _val_keys(tree), f"the validation loss_kwargs drop {key!r}"

    def test_both_bags_carry_the_coil_maps_under_some_alias(self, tree) -> None:
        """One spelling is enough, but it has to be one the table knows."""
        training = {kw.arg for kw in _training_compute_call(tree).keywords}
        assert training & set(COIL_MAP_ALIASES), (
            f"the training bag names no coil-map alias; the resolver has nothing to "
            f"re-file. present: {sorted(n for n in training if n)}"
        )
        assert _val_keys(tree) & set(COIL_MAP_ALIASES), "the validation bag names none"

    def test_neither_bag_re_spells_the_maps_itself(self, tree) -> None:
        """The reconciliation has ONE owner. A caller passing the same tensor
        under two aliases is the workaround this resolver replaced, and two
        owners of one table is how they start to disagree (non-negotiable 17)."""
        training = {kw.arg for kw in _training_compute_call(tree).keywords} & set(COIL_MAP_ALIASES)
        validation = _val_keys(tree) & set(COIL_MAP_ALIASES)
        assert len(training) == 1, f"training bag spells the maps {len(training)} ways: {training}"
        assert len(validation) == 1, f"validation bag spells them {len(validation)} ways"


class TestTheResolverReconcilesTheSpellings:
    def test_it_refiles_for_a_callee_naming_another_alias(self) -> None:
        maps = object()
        got = coil_map_kwargs_for({"coil_sensitivities", "mask"}, {"smaps": maps})
        assert got == {"coil_sensitivities": maps}

    def test_it_leaves_a_callee_naming_the_same_alias_alone(self) -> None:
        assert coil_map_kwargs_for({"smaps"}, {"smaps": object()}) == {}

    def test_the_callers_own_value_wins(self) -> None:
        """A re-filing must never overwrite something the caller set deliberately."""
        held, other = object(), object()
        assert (
            coil_map_kwargs_for(
                {"smaps", "coil_sensitivities"}, {"smaps": held, "coil_sensitivities": other}
            )
            == {}
        )

    def test_no_maps_held_means_no_entries(self) -> None:
        assert coil_map_kwargs_for({"coil_sensitivities"}, {"mask": None}) == {}
        assert coil_map_kwargs_for({"coil_sensitivities"}, {"smaps": None}) == {}

    def test_a_varkw_loss_is_not_handed_five_copies(self) -> None:
        """Planted: passing ``sig.parameters`` of a ``**kwargs`` callee must not
        expand to the whole alias table. ``hfen`` is the real such loss."""
        from spectramr.models.losses.registry import LossRegistry

        hfen = LossRegistry._custom_losses["hfen"]()
        params = inspect.signature(hfen.forward).parameters
        assert coil_map_kwargs_for(params, {"smaps": object()}) == {}

    def test_the_table_has_one_owner(self) -> None:
        """``data.batch_types`` re-exports rather than redeclaring."""
        from spectramr.data import batch_types

        assert batch_types.COIL_MAP_ALIASES is COIL_MAP_ALIASES


class TestTheBarrierIsReachableFromOneSpelling:
    """The behavioural half: the strategy's real bag, through the real call."""

    @staticmethod
    def _phantom():
        import torch

        torch.manual_seed(0)
        b, c, s = 2, 4, 16
        m = torch.randn(b, 1, s, s, dtype=torch.complex64)
        maps = torch.randn(b, c, s, s, dtype=torch.complex64)
        maps = maps / maps.abs().pow(2).sum(1, keepdim=True).sqrt()
        return maps * m, maps

    @staticmethod
    def _bag(maps):
        import torch

        # Exactly what the strategy holds: ONE alias.
        return {
            "smaps": maps,
            "mask": torch.ones(maps.shape[0], 1, *maps.shape[-2:]),
            "timesteps": torch.zeros(maps.shape[0]),
            "sample_measurement": None,
        }

    def _call(self, pred, target, maps):
        from spectramr.models.losses.complex.coil_subspace_residual import (
            CoilSubspaceResidualLoss,
        )
        from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
            _call_safe_loss,
        )

        loss = CoilSubspaceResidualLoss(input_domain="image")
        return float(_call_safe_loss(loss, pred, target, **self._bag(maps)))

    def test_a_realisable_image_costs_nothing(self) -> None:
        imgs, maps = self._phantom()
        assert self._call(imgs, imgs, maps) < 1e-6

    def test_energy_off_the_manifold_costs_a_lot(self) -> None:
        imgs, maps = self._phantom()
        off = imgs.clone()
        off[:, 0] += 0.5
        assert self._call(off, off, maps) > 1e-2

    def test_a_bag_with_no_maps_at_all_still_raises(self) -> None:
        """Planted violation: the resolver must not paper over absent maps."""
        from spectramr.models.losses.complex.coil_subspace_residual import (
            CoilSubspaceResidualLoss,
        )
        from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
            _call_safe_loss,
        )

        imgs, maps = self._phantom()
        bag = {k: v for k, v in self._bag(maps).items() if k != "smaps"}
        with pytest.raises(ValueError, match="requires coil_sensitivities"):
            _call_safe_loss(CoilSubspaceResidualLoss(input_domain="image"), imgs, imgs, **bag)
