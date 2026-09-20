"""Tests for ``core/module_utils.py`` — the wrapper-unwrapping SSOT.

The defect these guard: every checkpoint writer called ``model.state_dict()``
with no unwrapping, so a ``compile_model: true`` or DDP run wrote
``_orig_mod.``/``module.``-prefixed keys that no inference path could read back
(they all build a bare model and load with ``strict=True``; under
``strict=False`` nothing matches, nothing loads, and the load reports success).

Meanwhile 26 sites had grown their own inline unwrap. Only two handled
``_orig_mod``, none handled FSDP, and the one that looked canonical
(``pipelines.parallel.unwrap_model``) handled only DP/DDP and had zero callers.

The prefix assertions below are deliberately written against **real** wrappers
where one can be built on CPU (``torch.compile``, ``DataParallel``) rather than
against hand-written stand-ins, because the thing under test is precisely what
PyTorch names its attributes — a stand-in would agree with my assumption instead
of with torch.
"""

from __future__ import annotations

import re

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from spectramr.core.module_utils import (  # noqa: E402
    WRAPPER_ATTRS,
    is_wrapped,
    strip_wrapper_prefixes,
    unwrap_model,
)


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, padding=1)
        self.bn = nn.BatchNorm2d(2)

    def forward(self, x):  # pragma: no cover - never called
        return self.bn(self.conv(x))


# ---------------------------------------------------------------------------
# unwrap_model
# ---------------------------------------------------------------------------


class TestUnwrapModel:
    def test_bare_module_is_returned_unchanged(self) -> None:
        """No-op on an unwrapped model, so save sites can call it unconditionally."""
        model = Tiny()
        assert unwrap_model(model) is model
        assert is_wrapped(model) is False

    def test_unwraps_torch_compile(self) -> None:
        model = Tiny()
        assert unwrap_model(torch.compile(model)) is model
        assert is_wrapped(torch.compile(model)) is True

    def test_unwraps_data_parallel(self) -> None:
        model = Tiny()
        assert unwrap_model(nn.DataParallel(model)) is model

    def test_unwraps_a_nested_stack(self) -> None:
        """DDP(compile(model)) is the realistic shape; peel both layers."""
        model = Tiny()
        assert unwrap_model(nn.DataParallel(torch.compile(model))) is model

    def test_is_idempotent(self) -> None:
        model = Tiny()
        once = unwrap_model(nn.DataParallel(model))
        assert unwrap_model(once) is once is model

    @pytest.mark.parametrize("attr", WRAPPER_ATTRS)
    def test_unwraps_each_declared_attribute(self, attr: str) -> None:
        """Duck-typed on purpose: FSDP and OptimizedModule cannot be imported on
        a torch-less/CPU-shimmed environment, and this module is imported by the
        config-adjacent layers."""
        model = Tiny()
        wrapper = nn.Module()
        setattr(wrapper, attr, model)
        assert unwrap_model(wrapper) is model

    def test_survives_a_self_referential_attribute(self) -> None:
        """A module whose ``.module`` is itself must not spin forever."""

        class SelfRef(nn.Module):
            @property
            def module(self):
                return self

        obj = SelfRef()
        assert unwrap_model(obj) is obj

    def test_an_unset_wrapper_attribute_does_not_collapse_to_none(self) -> None:
        """A wrapper declaring the attribute but leaving it None must not turn
        the model into None — that would be a far worse failure than a prefix."""
        wrapper = nn.Module()
        wrapper.module = None
        assert unwrap_model(wrapper) is wrapper


# ---------------------------------------------------------------------------
# The prefixes this exists to remove — asserted against real torch wrappers
# ---------------------------------------------------------------------------


class TestWrapperPrefixesAreRealAndHandled:
    def test_compile_prefixes_state_dict_keys(self) -> None:
        model = Tiny()
        compiled_keys = set(torch.compile(model).state_dict())
        assert compiled_keys != set(model.state_dict())
        assert all(k.startswith("_orig_mod.") for k in compiled_keys)

    def test_data_parallel_prefixes_state_dict_keys(self) -> None:
        model = Tiny()
        dp_keys = set(nn.DataParallel(model).state_dict())
        assert all(k.startswith("module.") for k in dp_keys)

    @pytest.mark.parametrize(
        "wrap",
        [
            lambda m: torch.compile(m),
            lambda m: nn.DataParallel(m),
            lambda m: nn.DataParallel(torch.compile(m)),
        ],
        ids=["compile", "data_parallel", "data_parallel_of_compile"],
    )
    def test_unwrapping_restores_the_bare_key_set(self, wrap) -> None:
        """The property every save site depends on: unwrap first, and the keys
        are the ones a freshly-built bare model will ask for."""
        model = Tiny()
        expected = set(model.state_dict())
        assert set(unwrap_model(wrap(model)).state_dict()) == expected


# ---------------------------------------------------------------------------
# strip_wrapper_prefixes — the load-side analogue, for old checkpoints
# ---------------------------------------------------------------------------


class TestStripWrapperPrefixes:
    def test_strips_a_single_prefix(self) -> None:
        out = strip_wrapper_prefixes({"module.conv.weight": 1, "module.bn.bias": 2})
        assert out == {"conv.weight": 1, "bn.bias": 2}

    def test_strips_a_doubled_prefix(self) -> None:
        """A compiled model inside ModelEma yields ``module._orig_mod.<...>`` —
        one pass over a fixed prefix list would leave half of it behind."""
        out = strip_wrapper_prefixes({"module._orig_mod.conv.weight": 1})
        assert out == {"conv.weight": 1}

    def test_strips_fsdp_and_checkpoint_prefixes(self) -> None:
        out = strip_wrapper_prefixes(
            {"_fsdp_wrapped_module._checkpoint_wrapped_module.conv.weight": 1}
        )
        assert out == {"conv.weight": 1}

    def test_leaves_unprefixed_keys_alone(self) -> None:
        original = {"conv.weight": 1, "bn.bias": 2}
        assert strip_wrapper_prefixes(original) == original

    def test_returns_a_new_dict(self) -> None:
        """Never hand back the input, or a caller can mutate a live module's
        state dict by accident."""
        original = {"conv.weight": 1}
        assert strip_wrapper_prefixes(original) is not original

    def test_does_not_strip_a_mid_key_occurrence(self) -> None:
        """Only *leading* prefixes are wrappers. A submodule legitimately named
        ``module`` deeper in the tree must survive."""
        out = strip_wrapper_prefixes({"encoder.module.weight": 1})
        assert out == {"encoder.module.weight": 1}

    def test_a_collision_raises_rather_than_silently_merging(self) -> None:
        """Two distinct modules whose names differ only by a wrapper prefix would
        otherwise load as a hybrid of the two, last-one-wins."""
        with pytest.raises(ValueError, match="collides"):
            strip_wrapper_prefixes({"conv.weight": 1, "module.conv.weight": 2})

    def test_round_trips_a_real_compiled_state_dict(self) -> None:
        model = Tiny()
        compiled = torch.compile(model)
        assert set(strip_wrapper_prefixes(compiled.state_dict())) == set(
            model.state_dict()
        )

    def test_stripped_compiled_state_dict_loads_into_a_bare_model(self) -> None:
        """The end-to-end property: this is what #619 F3 broke."""
        src = Tiny()
        with torch.no_grad():
            src.conv.weight.fill_(0.5)
        compiled_sd = torch.compile(src).state_dict()

        dst = Tiny()
        # Pre-fix behaviour: the prefixed dict is not loadable strictly.
        with pytest.raises(RuntimeError):
            dst.load_state_dict(compiled_sd, strict=True)

        dst.load_state_dict(strip_wrapper_prefixes(compiled_sd), strict=True)
        assert torch.allclose(dst.conv.weight, src.conv.weight)


def test_pipelines_parallel_reexports_the_same_helper() -> None:
    """``pipelines.parallel.unwrap_model`` stays importable (it has tests and is
    the historical name) but must not be a second implementation."""
    from spectramr.pipelines.parallel import unwrap_model as reexported

    assert reexported is unwrap_model


# ---------------------------------------------------------------------------
# resolve_state_dict -- the one checkpoint reader
# ---------------------------------------------------------------------------


class TestResolveStateDict:
    """The single envelope-unwrapping reader every load path now shares.

    Before this existed, *six* call sites each knew a different subset of the
    envelope vocabulary, and the two writers in the repo disagreed with each
    other: ``checkpoint_director`` writes ``{"generator": ...}`` while
    ``checkpoint_service`` writes ``{"model_state_dict": ...}``. A reader that
    knew only one of them either raised ``KeyError: 'generator'`` (#1310) or --
    far worse -- fell through to loading the *whole envelope* under
    ``strict=False``, matching nothing, and reporting success. That is the
    Method-C facade: a run that looks warm-started while training from scratch
    (pitfalls #9/#16).

    Selection is by **key overlap with the model, not by key order**. The
    order-driven draft of this function picked the config dict out of a
    ``{"model": <config>, "generator": <weights>}`` payload -- which is a real
    checkpoint shape here, not a hypothetical one.
    """

    @staticmethod
    def _weights() -> dict:
        return Tiny().state_dict()

    @staticmethod
    def _keys() -> set[str]:
        return set(Tiny().state_dict().keys())

    def test_generator_envelope_resolves(self) -> None:
        """The envelope ``checkpoint_director`` actually writes (#1310)."""
        from spectramr.core.module_utils import resolve_state_dict

        w = self._weights()
        out = resolve_state_dict({"generator": w, "epoch": 7}, self._keys(), source="t")
        assert set(out) == set(w)

    def test_model_state_dict_envelope_resolves(self) -> None:
        """The envelope ``checkpoint_service`` writes -- the other writer."""
        from spectramr.core.module_utils import resolve_state_dict

        w = self._weights()
        out = resolve_state_dict({"model_state_dict": w}, self._keys(), source="t")
        assert set(out) == set(w)

    def test_bare_state_dict_resolves(self) -> None:
        """A raw ``model.state_dict()`` saved with no envelope at all."""
        from spectramr.core.module_utils import resolve_state_dict

        w = self._weights()
        assert set(resolve_state_dict(w, self._keys(), source="t")) == set(w)

    def test_wrapper_prefixes_are_stripped(self) -> None:
        """A DDP/compile-wrapped payload loads into a bare model."""
        from spectramr.core.module_utils import resolve_state_dict

        w = self._weights()
        wrapped = {f"module.{k}": v for k, v in w.items()}
        out = resolve_state_dict({"generator": wrapped}, self._keys(), source="t")
        assert set(out) == set(w)

    def test_overlap_beats_order_when_a_config_shares_the_key(self) -> None:
        """The decoy case the order-driven version got wrong.

        ``"model"`` precedes ``"generator"`` in the key vocabulary, and some
        checkpoints store the *config* under ``"model"``. Selecting the first
        recognised key would hand a config dict to ``load_state_dict``.
        """
        from spectramr.core.module_utils import resolve_state_dict

        w = self._weights()
        payload = {"model": {"model_type": "unet", "base_channels": 32}, "generator": w}
        out = resolve_state_dict(payload, self._keys(), source="t")
        assert set(out) == set(w), "picked the config dict over the weights"

    def test_zero_overlap_raises_instead_of_loading_nothing(self) -> None:
        """No silent no-op load: a payload that matches nothing must raise."""
        from spectramr.core.module_utils import resolve_state_dict

        foreign = {"totally.other.weight": torch.zeros(1)}
        with pytest.raises(ValueError, match="zero parameter"):
            resolve_state_dict(foreign, self._keys(), source="ckpt.pt")

    def test_non_mapping_payload_raises_typeerror(self) -> None:
        """A pickled ``nn.Module`` (the deprecated format) is not a state dict."""
        from spectramr.core.module_utils import resolve_state_dict

        with pytest.raises(TypeError):
            resolve_state_dict(Tiny(), self._keys(), source="ckpt.pt")

    def test_source_is_named_in_the_error(self) -> None:
        """The message must say *which* checkpoint failed, not just that one did."""
        from spectramr.core.module_utils import resolve_state_dict

        with pytest.raises(ValueError, match=re.escape("run42/best.pt")):
            resolve_state_dict(
                {"totally.other.weight": torch.zeros(1)},
                self._keys(),
                source="run42/best.pt",
            )


class TestMidPathWrapperSegments:
    """Regional compilation wraps a ModuleList child, not the root.

    That puts the marker mid-path -- ``blocks.0._orig_mod.conv.weight`` -- which
    a leading-only strip leaves in the checkpoint. Every inference path builds a
    bare model and loads with ``strict=True``, so the key fails there; under
    ``strict=False`` it matches nothing, loads nothing, and reports success.
    """

    def test_a_mid_path_orig_mod_is_stripped(self) -> None:
        assert strip_wrapper_prefixes({"blocks.0._orig_mod.conv.weight": 1}) == {
            "blocks.0.conv.weight": 1
        }

    def test_a_mid_path_fsdp_segment_is_stripped(self) -> None:
        assert strip_wrapper_prefixes({"layers.2._fsdp_wrapped_module.w": 1}) == {
            "layers.2.w": 1
        }

    def test_a_real_submodule_named_module_survives(self) -> None:
        """``module`` is an ordinary attribute name, unlike the synthetic
        underscore-prefixed wrappers, so it is only stripped while it leads."""
        assert strip_wrapper_prefixes({"encoder.module.weight": 1}) == {
            "encoder.module.weight": 1
        }

    def test_a_leading_module_is_still_stripped(self) -> None:
        assert strip_wrapper_prefixes({"module.encoder.weight": 1}) == {
            "encoder.weight": 1
        }

    def test_a_regionally_compiled_state_dict_loads_bare(self) -> None:
        """The property that matters end to end."""

        class Blk(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 2, 3)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([Blk() for _ in range(2)])

        model = Net()
        model.blocks[0] = torch.compile(model.blocks[0])
        Net().load_state_dict(strip_wrapper_prefixes(model.state_dict()), strict=True)

    @pytest.mark.parametrize(
        "order",
        [
            {"module.w": 1, "w": 2},
            {"w": 2, "module.w": 1},
        ],
        ids=["wrapped-first", "bare-first"],
    )
    def test_a_collision_raises_in_either_order(self, order) -> None:
        """The guard used to ask "was THIS key transformed" rather than "did two
        keys land on one name", so it fired in one order and silently kept the
        last value in the other -- and a state dict's order is just
        module-traversal order.
        """
        with pytest.raises(ValueError, match="collides"):
            strip_wrapper_prefixes(order)
