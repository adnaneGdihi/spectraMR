"""The five spellings coil maps travel under, reconciled in one place.

The table moved to ``core/`` because it has two boundaries to serve in different
layers: ``data.batch_types`` when a batch crosses into the strategy, and
``_call_safe_loss`` when a loss is invoked. ``models/`` cannot import ``data/``,
so a table in ``data/`` could serve only one of them -- and the second was being
served by callers passing the same tensor twice, which is two owners of one
table (non-negotiable 17).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.core.coil_map_names import (
    CANONICAL_COIL_MAP_KEY,
    COIL_MAP_ALIASES,
    coil_map_kwargs_for,
    kwargs_accepted_by,
)


class TestTheTableItself:
    def test_it_is_the_five_live_spellings(self) -> None:
        assert set(COIL_MAP_ALIASES) == {
            "coil_maps",
            "sensitivity",
            "sensitivity_maps",
            "coil_sensitivities",
            "smaps",
        }

    def test_the_canonical_key_is_first_so_resolution_is_deterministic(self) -> None:
        assert CANONICAL_COIL_MAP_KEY == COIL_MAP_ALIASES[0] == "coil_maps"

    def test_data_batch_types_re_exports_rather_than_redeclaring(self) -> None:
        """A second tuple with the same contents is how they start to diverge."""
        from spectramr.data import batch_types

        assert batch_types.COIL_MAP_ALIASES is COIL_MAP_ALIASES


class TestItRefilesOnlyWhenItShould:
    def test_a_callee_naming_another_alias_gets_the_maps(self) -> None:
        maps = object()
        assert coil_map_kwargs_for({"coil_sensitivities"}, {"smaps": maps}) == {
            "coil_sensitivities": maps
        }

    def test_a_callee_naming_the_same_alias_is_untouched(self) -> None:
        assert coil_map_kwargs_for({"smaps"}, {"smaps": object()}) == {}

    def test_a_callee_naming_no_alias_gets_nothing(self) -> None:
        assert coil_map_kwargs_for({"mask", "timesteps"}, {"smaps": object()}) == {}

    def test_the_callers_own_value_is_never_overwritten(self) -> None:
        held, deliberate = object(), object()
        out = coil_map_kwargs_for(
            {"smaps", "coil_sensitivities"},
            {"smaps": held, "coil_sensitivities": deliberate},
        )
        assert out == {}, "a re-filing must not clobber a value the caller set"

    @pytest.mark.parametrize("kwargs", [{}, {"mask": object()}, {"smaps": None}])
    def test_no_maps_held_means_no_entries(self, kwargs) -> None:
        """``None`` under an alias is 'no maps', not 'maps that are None'."""
        assert coil_map_kwargs_for(set(COIL_MAP_ALIASES), kwargs) == {}

    def test_it_serves_every_alias_the_callee_declares(self) -> None:
        maps = object()
        out = coil_map_kwargs_for({"coil_maps", "coil_sensitivities"}, {"smaps": maps})
        assert out == {"coil_maps": maps, "coil_sensitivities": maps}

    def test_resolution_is_deterministic_when_several_are_held(self) -> None:
        """Table order decides, so two runs cannot disagree about which won."""
        first, second = object(), object()
        held = {"sensitivity": first, "smaps": second}
        assert coil_map_kwargs_for({"coil_sensitivities"}, held) == {"coil_sensitivities": first}


class TestTheVarkwTrap:
    def test_passing_an_everything_container_would_hand_over_five_copies(self) -> None:
        """Why the caller must pass ``sig.parameters`` and not a ``**kwargs``
        universe. Pinned as the shape to avoid, not as desired behaviour."""

        class Everything:
            def __contains__(self, _item: object) -> bool:
                return True

        out = coil_map_kwargs_for(Everything(), {"smaps": object()})
        assert len(out) == len(COIL_MAP_ALIASES) - 1, (
            "an unbounded `accepted` yields one entry per alias -- this is the "
            "trap `_call_safe_loss` avoids by passing explicit parameter names"
        )


class TestKwargsAcceptedBy:
    """The one filter every loss-invoking hop narrows kwargs through.

    Two hand-written copies of it drifted: ``_call_safe_loss`` reconciled the
    coil maps and ``DifferentiableFourierBridge`` did not, so every bridged
    ``coil_subspace_residual`` bound ``None`` and raised at iteration 1.
    """

    def test_a_named_callee_gets_only_what_it_names(self) -> None:
        def loss(pred, target, mask=None):
            return None

        mask = object()
        assert kwargs_accepted_by(loss, {"mask": mask, "timesteps": object()}) == {"mask": mask}

    def test_it_refiles_the_maps_for_a_callee_spelling_them_differently(self) -> None:
        def loss(pred, target, coil_sensitivities=None):
            return None

        maps = object()
        assert kwargs_accepted_by(loss, {"smaps": maps}) == {"coil_sensitivities": maps}

    def test_a_varkw_callee_gets_everything_once(self) -> None:
        def loss(pred, target, **kwargs):
            return None

        bag = {"smaps": object(), "mask": object()}
        assert kwargs_accepted_by(loss, bag) == bag

    def test_a_module_is_read_through_forward(self) -> None:
        """``Module.__call__`` is ``(*args, **kwargs)``; reading it would admit all."""

        class Loss(torch.nn.Module):
            def forward(self, pred, target, coil_sensitivities=None):
                return pred

        maps = object()
        out = kwargs_accepted_by(Loss(), {"smaps": maps, "timesteps": object()})
        assert out == {"coil_sensitivities": maps}
